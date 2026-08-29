"""Fail-closed RunPod GPU-cost watchdog.

The watchdog uses RunPod's live Pod metadata rather than a caller-supplied
hourly rate. It runs independently from the experiment and only calls the
non-destructive stop endpoint. Pod deletion remains an explicit post-sync
operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from model_forensics.io import write_json

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
RUNPOD_POD_LOOKUP_DOC = "https://docs.runpod.io/api-reference/pods/GET/pods/podId"
RUNPOD_POD_STOP_DOC = "https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop"
WATCHDOG_VERSION = "runpod-gpu-cost-watchdog-v2"
_POD_ID_RE = re.compile(r"[A-Za-z0-9_-]{3,128}\Z")
_GPU_FAMILY_RE = {
    "H100": re.compile(r"(?:^|[^A-Z0-9])H100(?:$|[^A-Z0-9])", re.IGNORECASE),
    "A100": re.compile(r"(?:^|[^A-Z0-9])A100(?:$|[^A-Z0-9])", re.IGNORECASE),
}
_EXPECTED_FAMILY_ALIASES = {
    "H100": "H100",
    "H100_80GB": "H100",
    "A100": "A100",
    "A100_80GB": "A100",
}
_TERMINAL_STATUSES = frozenset({"EXITED", "TERMINATED"})
_CONTAINER_DIGEST_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_MACHINE_ID_RE = re.compile(r"[A-Za-z0-9_-]{3,191}\Z")


class WatchdogError(RuntimeError):
    """The watchdog cannot establish or enforce a safe stop deadline."""


@dataclass(frozen=True, slots=True)
class WatchdogLimits:
    """User-approved cumulative limits for one provider-backed GPU session.

    ``prior_committed_gpu_usd`` is supplied by the provider-neutral canonical
    ledger gate.  It includes prior incurred GPU cost plus any pre-existing GPU
    commitment, but excludes the current session's reservation.  The watchdog
    can therefore enforce the remaining global balance rather than granting
    every phase a fresh copy of the full cap.
    """

    gpu_hard_stop_usd: float
    maximum_runtime_hours: float
    safety_margin_fraction: float = 0.03
    maximum_approved_hourly_total_usd: float | None = None
    maximum_approved_storage_hourly_usd: float = 0.0
    prior_committed_gpu_usd: float = 0.0

    def __post_init__(self) -> None:
        numeric = (
            self.gpu_hard_stop_usd,
            self.maximum_runtime_hours,
            self.safety_margin_fraction,
            self.maximum_approved_storage_hourly_usd,
            self.prior_committed_gpu_usd,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in numeric):
            raise ValueError("watchdog limits must be finite numeric values")
        if self.gpu_hard_stop_usd <= 0 or self.maximum_runtime_hours <= 0:
            raise ValueError("GPU hard stop and maximum runtime must be positive")
        if not 0 < self.safety_margin_fraction < 0.25:
            raise ValueError("safety margin must be between 0 and 0.25")
        if self.prior_committed_gpu_usd < 0:
            raise ValueError("prior committed GPU cost must be non-negative")
        if self.maximum_approved_storage_hourly_usd < 0:
            raise ValueError("approved storage hourly cost must be non-negative")
        approved = self.maximum_approved_hourly_total_usd
        if approved is not None and (
            isinstance(approved, bool) or not math.isfinite(float(approved)) or approved <= 0
        ):
            raise ValueError("maximum approved hourly total must be positive and finite")
        if approved is not None and self.maximum_approved_storage_hourly_usd >= approved:
            raise ValueError("approved storage hourly cost must be below the all-in hourly total")
        if self.prior_committed_gpu_usd >= self.global_safe_budget_usd:
            raise ValueError("prior committed GPU cost leaves no safety-adjusted GPU budget")

    @property
    def global_safe_budget_usd(self) -> float:
        return self.gpu_hard_stop_usd * (1 - self.safety_margin_fraction)

    @property
    def safe_budget_usd(self) -> float:
        return self.global_safe_budget_usd - self.prior_committed_gpu_usd


@dataclass(frozen=True, slots=True)
class PodMetadata:
    """Sanitized subset of the authoritative RunPod Pod response."""

    pod_id: str
    gpu_count: int
    provider_gpu_id: str
    gpu_display_name: str
    machine_gpu_identity: tuple[str, ...]
    machine_id_hash: str
    data_center_id: str
    secure_cloud: bool
    container_image: str
    desired_status: str
    cost_per_hr: float
    adjusted_cost_per_hr: float
    last_started_at: datetime
    observed_at: datetime
    locked: bool
    network_volume_attached: bool

    @property
    def effective_hourly_usd(self) -> float:
        # RunPod documents adjustedCostPerHr as the effective hourly cost after
        # Savings Plans. It is therefore the authoritative billable rate.
        return self.adjusted_cost_per_hr

    @property
    def runtime_seconds(self) -> float:
        return max(0.0, (self.observed_at - self.last_started_at).total_seconds())

    @property
    def incurred_cost_usd(self) -> float:
        # RunPod bills running Pods per minute. Round the live estimate upward
        # so a partial provider billing minute cannot make the state optimistic.
        billed_minutes = math.ceil(self.runtime_seconds / 60)
        return self.effective_hourly_usd * billed_minutes / 60

    def public_dict(self) -> dict[str, Any]:
        """Return only non-secret fields safe to persist in a public manifest."""

        return {
            "pod_id": self.pod_id,
            "gpu_count": self.gpu_count,
            "provider_gpu_id": self.provider_gpu_id,
            "gpu_display_name": self.gpu_display_name,
            "machine_gpu_identity": list(self.machine_gpu_identity),
            "machine_id_hash": self.machine_id_hash,
            "data_center_id": self.data_center_id,
            "secure_cloud": self.secure_cloud,
            "container_image": self.container_image,
            "desired_status": self.desired_status,
            "cost_per_hr": self.cost_per_hr,
            "adjusted_cost_per_hr": self.adjusted_cost_per_hr,
            "effective_hourly_usd": self.effective_hourly_usd,
            "last_started_at": self.last_started_at.isoformat(),
            "observed_at": self.observed_at.isoformat(),
            "locked": self.locked,
            "network_volume_attached": self.network_volume_attached,
        }


@dataclass(frozen=True, slots=True)
class DerivedDeadline:
    """Absolute deadline derived from live provider metadata and approved limits."""

    budget_deadline: datetime
    runtime_deadline: datetime
    deadline: datetime
    calculation_hourly_usd: float
    incurred_cost_usd: float

    @property
    def reason(self) -> str:
        if self.budget_deadline <= self.runtime_deadline:
            return "safe_budget"
        return "maximum_runtime"


HttpTransport = Callable[[str, str, float], tuple[int, str]]
StopTransport = HttpTransport
MetadataTransport = HttpTransport


def _default_transport(
    url: str,
    api_key: str,
    timeout: float,
    *,
    method: str,
) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")
    except (TimeoutError, urllib.error.URLError) as exc:
        # Never include request headers or the API key in the exception.
        raise WatchdogError(f"RunPod {method} transport failed: {type(exc).__name__}") from exc


def _default_stop_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
    return _default_transport(url, api_key, timeout, method="POST")


def _default_metadata_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
    return _default_transport(url, api_key, timeout, method="GET")


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise WatchdogError(f"RunPod metadata field {field} must be a nonempty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WatchdogError(f"RunPod metadata field {field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise WatchdogError(f"RunPod metadata field {field} must include a timezone")
    return parsed.astimezone(UTC)


def _parse_positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise WatchdogError(f"RunPod metadata field {field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WatchdogError(f"RunPod metadata field {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise WatchdogError(f"RunPod metadata field {field} must be positive and finite")
    return parsed


def _parse_container_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _CONTAINER_DIGEST_RE.fullmatch(value) is None:
        raise WatchdogError(f"RunPod metadata field {field} must be an exact image digest")
    return value


def _hash_machine_id(value: Any) -> str:
    if not isinstance(value, str) or _MACHINE_ID_RE.fullmatch(value) is None:
        raise WatchdogError("RunPod metadata machineId must be a nonempty provider identifier")
    digest = hashlib.sha256(f"runpod-machine-id-v1:{value}".encode()).hexdigest()
    return f"sha256:{digest}"


def parse_pod_metadata(
    payload: Mapping[str, Any],
    *,
    expected_pod_id: str,
    observed_at: datetime,
) -> PodMetadata:
    """Parse and sanitize a live ``GET /v1/pods/{id}`` response."""

    if payload.get("id") != expected_pod_id:
        raise WatchdogError("RunPod metadata returned a different Pod id")
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        raise WatchdogError("RunPod metadata is missing the GPU object")
    count = gpu.get("count")
    if isinstance(count, bool) or not isinstance(count, (int, float)) or int(count) != count:
        raise WatchdogError("RunPod metadata gpu.count must be an integer")
    provider_gpu_id = gpu.get("id")
    if not isinstance(provider_gpu_id, str) or not provider_gpu_id.strip():
        raise WatchdogError("RunPod metadata gpu.id must be nonempty")
    display_name = gpu.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        raise WatchdogError("RunPod metadata gpu.displayName must be nonempty")
    machine = payload.get("machine")
    if not isinstance(machine, Mapping):
        raise WatchdogError("RunPod metadata is missing includeMachine output")
    machine_labels: list[str] = []
    for value in (machine.get("gpuTypeId"), machine.get("gpuDisplayName")):
        if isinstance(value, str) and value.strip():
            machine_labels.append(value.strip())
    machine_gpu_type = machine.get("gpuType")
    if isinstance(machine_gpu_type, Mapping):
        for key in ("id", "displayName"):
            value = machine_gpu_type.get(key)
            if isinstance(value, str) and value.strip():
                machine_labels.append(value.strip())
    machine_labels = list(dict.fromkeys(machine_labels))
    if not machine_labels:
        raise WatchdogError("RunPod includeMachine output has no GPU identity")
    secure_cloud = machine.get("secureCloud")
    if not isinstance(secure_cloud, bool):
        raise WatchdogError("RunPod includeMachine output has no secureCloud flag")
    data_center_id = machine.get("dataCenterId")
    if not isinstance(data_center_id, str) or not data_center_id.strip():
        raise WatchdogError("RunPod includeMachine output has no dataCenterId")
    desired_status = payload.get("desiredStatus")
    if desired_status not in {"RUNNING", "EXITED", "TERMINATED"}:
        raise WatchdogError("RunPod metadata desiredStatus is unsupported")
    locked = payload.get("locked")
    if not isinstance(locked, bool):
        raise WatchdogError("RunPod metadata locked flag must be boolean")
    current = observed_at.astimezone(UTC)
    started = _parse_timestamp(payload.get("lastStartedAt"), field="lastStartedAt")
    if started > current + timedelta(minutes=5):
        raise WatchdogError("RunPod lastStartedAt is implausibly in the future")
    return PodMetadata(
        pod_id=expected_pod_id,
        gpu_count=int(count),
        provider_gpu_id=provider_gpu_id.strip(),
        gpu_display_name=display_name.strip(),
        machine_gpu_identity=tuple(machine_labels),
        machine_id_hash=_hash_machine_id(payload.get("machineId")),
        data_center_id=data_center_id.strip(),
        secure_cloud=secure_cloud,
        container_image=_parse_container_digest(payload.get("image"), field="image"),
        desired_status=str(desired_status),
        cost_per_hr=_parse_positive_number(payload.get("costPerHr"), field="costPerHr"),
        adjusted_cost_per_hr=_parse_positive_number(
            payload.get("adjustedCostPerHr"), field="adjustedCostPerHr"
        ),
        last_started_at=started,
        observed_at=current,
        locked=locked,
        network_volume_attached=isinstance(payload.get("networkVolume"), Mapping),
    )


def normalize_gpu_family(value: str) -> str:
    try:
        return _EXPECTED_FAMILY_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported expected GPU family: {value}") from exc


def validate_live_metadata(
    metadata: PodMetadata,
    *,
    expected_gpu_count: int,
    expected_gpu_family: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: tuple[str, ...],
    expected_container_image: str,
    limits: WatchdogLimits,
) -> None:
    """Validate the exact live Pod against the pre-approved execution profile."""

    family = normalize_gpu_family(expected_gpu_family)
    approved_image = _parse_container_digest(
        expected_container_image,
        field="expected_container_image",
    )
    if expected_gpu_count != 8 or metadata.gpu_count != expected_gpu_count:
        raise WatchdogError(
            f"RunPod live metadata must report exactly 8 GPUs; observed {metadata.gpu_count}"
        )
    if _GPU_FAMILY_RE[family].search(metadata.gpu_display_name) is None:
        raise WatchdogError(
            "RunPod live GPU family does not match the approved profile: "
            f"expected {family}, observed {metadata.gpu_display_name!r}"
        )
    if metadata.provider_gpu_id != expected_provider_gpu_id:
        raise WatchdogError(
            "RunPod live provider GPU id does not match the frozen quote: "
            f"expected {expected_provider_gpu_id!r}, observed {metadata.provider_gpu_id!r}"
        )
    if not allowed_data_center_ids or metadata.data_center_id not in allowed_data_center_ids:
        raise WatchdogError(
            "RunPod live data center is outside the frozen launch set: "
            f"observed {metadata.data_center_id!r}"
        )
    recognized_machine_families = {
        candidate
        for label in metadata.machine_gpu_identity
        for candidate, pattern in _GPU_FAMILY_RE.items()
        if pattern.search(label)
    }
    if recognized_machine_families != {family}:
        raise WatchdogError(
            "RunPod machine GPU identity disagrees with the approved homogeneous family: "
            f"expected {family}, observed {metadata.machine_gpu_identity!r}"
        )
    if expected_provider_gpu_id not in metadata.machine_gpu_identity:
        raise WatchdogError(
            "RunPod machine GPU identity does not contain the exact frozen provider GPU id"
        )
    if not metadata.secure_cloud:
        raise WatchdogError("RunPod live machine is not in Secure Cloud")
    if metadata.container_image != approved_image:
        raise WatchdogError(
            "RunPod live image does not match the approval-bound digest: "
            f"expected {approved_image!r}, observed {metadata.container_image!r}"
        )
    if metadata.desired_status != "RUNNING":
        raise WatchdogError(
            f"RunPod Pod must be RUNNING when watchdog is armed; got {metadata.desired_status}"
        )
    if metadata.locked:
        raise WatchdogError("RunPod Pod is locked; the official API says locked Pods cannot stop")
    approved = limits.maximum_approved_hourly_total_usd
    live_rate_ceiling = max(metadata.cost_per_hr, metadata.effective_hourly_usd)
    if approved is not None and live_rate_ceiling > approved + 0.01:
        raise WatchdogError(
            "RunPod live hourly cost exceeds the approved quote: "
            f"${live_rate_ceiling:.4f}/h > ${approved:.4f}/h"
        )


def derive_deadline(
    metadata: PodMetadata,
    limits: WatchdogLimits,
    *,
    calculation_hourly_usd: float | None = None,
) -> DerivedDeadline:
    """Derive incurred cost and an absolute deadline from ``lastStartedAt``."""

    rate = calculation_hourly_usd or metadata.effective_hourly_usd
    if not math.isfinite(rate) or rate <= 0:
        raise WatchdogError("authoritative hourly rate must be positive and finite")
    budget_runtime = timedelta(hours=limits.safe_budget_usd / rate)
    runtime_limit = timedelta(hours=limits.maximum_runtime_hours)
    budget_deadline = metadata.last_started_at + budget_runtime
    runtime_deadline = metadata.last_started_at + runtime_limit
    return DerivedDeadline(
        budget_deadline=budget_deadline,
        runtime_deadline=runtime_deadline,
        deadline=min(budget_deadline, runtime_deadline),
        calculation_hourly_usd=rate,
        incurred_cost_usd=rate * math.ceil(metadata.runtime_seconds / 60) / 60,
    )


class RunpodStopClient:
    """Minimal RunPod client with injectable GET and POST transports for tests."""

    def __init__(
        self,
        *,
        pod_id: str,
        api_key_env: str = "RUNPOD_API_KEY",
        endpoint_base: str = RUNPOD_REST_BASE,
        transport: StopTransport | None = None,
        metadata_transport: MetadataTransport | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        if _POD_ID_RE.fullmatch(pod_id) is None:
            raise ValueError("pod_id contains unsupported characters")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise WatchdogError(f"required secret environment variable is unset: {api_key_env}")
        if not endpoint_base.startswith("https://"):
            raise ValueError("RunPod endpoint must use HTTPS")
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("RunPod request timeout must be in (0, 60]")
        self.pod_id = pod_id
        pod_endpoint = f"{endpoint_base.rstrip('/')}/pods/{pod_id}"
        query = urllib.parse.urlencode(
            {
                "includeMachine": "true",
                "includeNetworkVolume": "true",
                "includeSavingsPlans": "true",
            }
        )
        self._metadata_endpoint = f"{pod_endpoint}?{query}"
        self._stop_endpoint = f"{pod_endpoint}/stop"
        self._api_key = api_key
        self._stop_transport = transport or _default_stop_transport
        self._metadata_transport = metadata_transport or _default_metadata_transport
        self._timeout_seconds = timeout_seconds

    def _get_payload(self) -> Mapping[str, Any]:
        status, body = self._metadata_transport(
            self._metadata_endpoint,
            self._api_key,
            self._timeout_seconds,
        )
        if status < 200 or status >= 300:
            # Do not propagate an error body: API responses can echo request or account data.
            raise WatchdogError(f"RunPod metadata GET returned HTTP {status}")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WatchdogError("RunPod metadata GET returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise WatchdogError("RunPod metadata GET returned a non-object payload")
        return payload

    def metadata(self, *, observed_at: datetime | None = None) -> PodMetadata:
        return parse_pod_metadata(
            self._get_payload(),
            expected_pod_id=self.pod_id,
            observed_at=observed_at or datetime.now(UTC),
        )

    def desired_status(self) -> str:
        payload = self._get_payload()
        if payload.get("id") != self.pod_id:
            raise WatchdogError("RunPod metadata returned a different Pod id")
        status = payload.get("desiredStatus")
        if status not in {"RUNNING", "EXITED", "TERMINATED"}:
            raise WatchdogError("RunPod metadata desiredStatus is unsupported")
        return str(status)

    def stop(self) -> None:
        status, _body = self._stop_transport(
            self._stop_endpoint,
            self._api_key,
            self._timeout_seconds,
        )
        if status < 200 or status >= 300:
            raise WatchdogError(f"RunPod stop returned HTTP {status}")


def _state(
    *,
    pod_id: str,
    limits: WatchdogLimits,
    status: str,
    armed_at: datetime,
    metadata: PodMetadata | None,
    derived: DerivedDeadline | None,
    now: datetime,
    stop_reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    remaining_seconds = None
    if derived is not None:
        remaining_seconds = max(0.0, (derived.deadline - now).total_seconds())
    return {
        "schema_version": 2,
        "watchdog_version": WATCHDOG_VERSION,
        "pod_id": pod_id,
        "status": status,
        "armed_at": armed_at.isoformat(),
        "updated_at": now.isoformat(),
        "live_metadata": metadata.public_dict() if metadata is not None else None,
        "limits": {
            "gpu_hard_stop_usd": limits.gpu_hard_stop_usd,
            "global_safe_budget_usd": limits.global_safe_budget_usd,
            "safe_budget_usd": limits.safe_budget_usd,
            "safety_margin_fraction": limits.safety_margin_fraction,
            "maximum_runtime_hours": limits.maximum_runtime_hours,
            "maximum_approved_hourly_total_usd": limits.maximum_approved_hourly_total_usd,
            "maximum_approved_storage_hourly_usd": (
                limits.maximum_approved_storage_hourly_usd
            ),
            "prior_committed_gpu_usd": limits.prior_committed_gpu_usd,
        },
        "deadline": (
            {
                "budget_deadline": derived.budget_deadline.isoformat(),
                "runtime_deadline": derived.runtime_deadline.isoformat(),
                "effective_deadline": derived.deadline.isoformat(),
                "limiting_reason": derived.reason,
                "remaining_seconds": round(remaining_seconds or 0.0, 3),
                "calculation_hourly_usd": derived.calculation_hourly_usd,
                "incurred_cost_usd": round(derived.incurred_cost_usd, 6),
            }
            if derived is not None
            else None
        ),
        "stop_reason": stop_reason,
        "action": "stop_only_preserve_volume",
        "deletion": "manual_after_verified_sync",
        "error": error,
    }


def _stop_until_confirmed(
    *,
    client: RunpodStopClient,
    state_path: str | Path,
    limits: WatchdogLimits,
    armed_at: datetime,
    metadata: PodMetadata | None,
    derived: DerivedDeadline | None,
    stop_reason: str,
    stop_attempts: int,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, stop_attempts + 1):
        try:
            status_before = client.desired_status()
        except WatchdogError as exc:
            status_before = "UNKNOWN"
            last_error = str(exc)
        if status_before in _TERMINAL_STATUSES:
            status = "stopped_confirmed" if status_before == "EXITED" else "terminated_externally"
            completed = _state(
                pod_id=client.pod_id,
                limits=limits,
                status=status,
                armed_at=armed_at,
                metadata=metadata,
                derived=derived,
                now=now().astimezone(UTC),
                stop_reason=stop_reason,
                error=last_error,
            )
            write_json(state_path, completed)
            return completed
        try:
            client.stop()
        except WatchdogError as exc:
            last_error = str(exc)
        else:
            last_error = None
        write_json(
            state_path,
            _state(
                pod_id=client.pod_id,
                limits=limits,
                status=f"stop_retry_{attempt}",
                armed_at=armed_at,
                metadata=metadata,
                derived=derived,
                now=now().astimezone(UTC),
                stop_reason=stop_reason,
                error=last_error,
            ),
        )
        if attempt < stop_attempts:
            sleep(5)
    try:
        final_status = client.desired_status()
    except WatchdogError as exc:
        final_status = "UNKNOWN"
        last_error = str(exc)
    if final_status in _TERMINAL_STATUSES:
        status = "stopped_confirmed" if final_status == "EXITED" else "terminated_externally"
        completed = _state(
            pod_id=client.pod_id,
            limits=limits,
            status=status,
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            now=now().astimezone(UTC),
            stop_reason=stop_reason,
            error=last_error,
        )
        write_json(state_path, completed)
        return completed
    failed = _state(
        pod_id=client.pod_id,
        limits=limits,
        status="stop_unconfirmed",
        armed_at=armed_at,
        metadata=metadata,
        derived=derived,
        now=now().astimezone(UTC),
        stop_reason=stop_reason,
        error=last_error or f"desiredStatus remained {final_status}",
    )
    write_json(state_path, failed)
    raise WatchdogError(f"failed to confirm Pod stop after {stop_attempts} attempts")


def run_watchdog(
    *,
    pod_id: str,
    expected_gpu_family: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: tuple[str, ...],
    expected_container_image: str,
    limits: WatchdogLimits,
    state_path: str | Path,
    client: RunpodStopClient,
    expected_gpu_count: int = 8,
    stop_request_path: str | Path | None = None,
    poll_seconds: float = 15,
    stop_attempts: int = 12,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Monitor a single live Pod and stop it at the provider-derived deadline."""

    if client.pod_id != pod_id:
        raise ValueError("stop client targets a different Pod")
    normalize_gpu_family(expected_gpu_family)
    if expected_gpu_count != 8:
        raise ValueError("this experiment requires exactly 8 GPUs")
    if poll_seconds <= 0 or poll_seconds > 60 or stop_attempts <= 0:
        raise ValueError("poll interval must be in (0, 60] and attempts positive")

    armed_at = now().astimezone(UTC)
    armed_monotonic = monotonic()
    metadata: PodMetadata | None = None
    derived: DerivedDeadline | None = None
    try:
        metadata = client.metadata(observed_at=armed_at)
        validate_live_metadata(
            metadata,
            expected_gpu_count=expected_gpu_count,
            expected_gpu_family=expected_gpu_family,
            expected_provider_gpu_id=expected_provider_gpu_id,
            allowed_data_center_ids=allowed_data_center_ids,
            expected_container_image=expected_container_image,
            limits=limits,
        )
        # RunPod's live Pod rate has historically represented compute while
        # volume/container storage is billed separately. Add the frozen storage
        # allowance even if a future response includes it; double-counting the
        # small storage component is deliberately conservative.
        calculation_rate = (
            metadata.effective_hourly_usd + limits.maximum_approved_storage_hourly_usd
        )
        derived = derive_deadline(metadata, limits, calculation_hourly_usd=calculation_rate)
    except WatchdogError as exc:
        write_json(
            state_path,
            _state(
                pod_id=pod_id,
                limits=limits,
                status="live_verification_failed",
                armed_at=armed_at,
                metadata=metadata,
                derived=derived,
                now=now().astimezone(UTC),
                stop_reason="live_verification_failed",
                error=str(exc),
            ),
        )
        _stop_until_confirmed(
            client=client,
            state_path=state_path,
            limits=limits,
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            stop_reason="live_verification_failed",
            stop_attempts=stop_attempts,
            sleep=sleep,
            now=now,
        )
        raise

    if armed_at >= derived.deadline:
        return _stop_until_confirmed(
            client=client,
            state_path=state_path,
            limits=limits,
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            stop_reason=derived.reason,
            stop_attempts=stop_attempts,
            sleep=sleep,
            now=now,
        )

    monotonic_deadline = armed_monotonic + (derived.deadline - armed_at).total_seconds()

    write_json(
        state_path,
        _state(
            pod_id=pod_id,
            limits=limits,
            status="armed",
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            now=armed_at,
        ),
    )
    initial_started_at = metadata.last_started_at
    initial_machine_id_hash = metadata.machine_id_hash
    while True:
        current_time = now().astimezone(UTC)
        current_monotonic = monotonic()
        if stop_request_path is not None and Path(stop_request_path).exists():
            stop_reason = "external_stop_request"
            break
        if current_time >= derived.deadline or current_monotonic >= monotonic_deadline:
            stop_reason = derived.reason
            break
        sleep(
            min(
                poll_seconds,
                max(0.0, (derived.deadline - current_time).total_seconds()),
                max(0.0, monotonic_deadline - current_monotonic),
            )
        )
        current_time = now().astimezone(UTC)
        try:
            current = client.metadata(observed_at=current_time)
            if current.desired_status in _TERMINAL_STATUSES:
                terminal = _state(
                    pod_id=pod_id,
                    limits=limits,
                    status=(
                        "stopped_confirmed"
                        if current.desired_status == "EXITED"
                        else "terminated_externally"
                    ),
                    armed_at=armed_at,
                    metadata=current,
                    derived=derived,
                    now=current_time,
                    stop_reason="stopped_outside_watchdog",
                )
                write_json(state_path, terminal)
                return terminal
            validate_live_metadata(
                current,
                expected_gpu_count=expected_gpu_count,
                expected_gpu_family=expected_gpu_family,
                expected_provider_gpu_id=expected_provider_gpu_id,
                allowed_data_center_ids=allowed_data_center_ids,
                expected_container_image=expected_container_image,
                limits=limits,
            )
            if current.last_started_at != initial_started_at:
                raise WatchdogError("RunPod lastStartedAt changed after watchdog arming")
            if current.machine_id_hash != initial_machine_id_hash:
                raise WatchdogError("RunPod machineId changed after watchdog arming")
            # Never extend the deadline if RunPod later reports a lower rate.
            calculation_rate = max(
                calculation_rate,
                current.effective_hourly_usd + limits.maximum_approved_storage_hourly_usd,
            )
            metadata = current
            derived = derive_deadline(
                current,
                limits,
                calculation_hourly_usd=calculation_rate,
            )
            # A provider rate increase can shorten the absolute deadline. A
            # wall-clock adjustment or rate decrease can never extend the
            # monotonic guard established at arming.
            monotonic_deadline = min(
                monotonic_deadline,
                monotonic() + max(0.0, (derived.deadline - current_time).total_seconds()),
            )
        except WatchdogError as exc:
            stop_reason = "live_metadata_became_unsafe"
            write_json(
                state_path,
                _state(
                    pod_id=pod_id,
                    limits=limits,
                    status="metadata_unsafe_stop_pending",
                    armed_at=armed_at,
                    metadata=metadata,
                    derived=derived,
                    now=current_time,
                    stop_reason=stop_reason,
                    error=str(exc),
                ),
            )
            break
        write_json(
            state_path,
            _state(
                pod_id=pod_id,
                limits=limits,
                status="armed",
                armed_at=armed_at,
                metadata=metadata,
                derived=derived,
                now=current_time,
            ),
        )

    return _stop_until_confirmed(
        client=client,
        state_path=state_path,
        limits=limits,
        armed_at=armed_at,
        metadata=metadata,
        derived=derived,
        stop_reason=stop_reason,
        stop_attempts=stop_attempts,
        sleep=sleep,
        now=now,
    )


__all__ = [
    "RUNPOD_POD_LOOKUP_DOC",
    "RUNPOD_POD_STOP_DOC",
    "RUNPOD_REST_BASE",
    "WATCHDOG_VERSION",
    "DerivedDeadline",
    "PodMetadata",
    "RunpodStopClient",
    "WatchdogError",
    "WatchdogLimits",
    "derive_deadline",
    "normalize_gpu_family",
    "parse_pod_metadata",
    "run_watchdog",
    "validate_live_metadata",
]
