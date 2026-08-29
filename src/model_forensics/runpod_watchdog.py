"""Fail-closed RunPod GPU-cost watchdog.

The watchdog uses RunPod's v2 live Pod metadata rather than a caller-supplied
hourly rate. It runs independently from the experiment and only sends the
non-destructive ``{"action":"stop"}`` Pod action. Pod deletion remains an
explicit post-sync operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from model_forensics.io import write_json

RUNPOD_API_BASE = "https://api.runpod.io/v2"
RUNPOD_REST_BASE = RUNPOD_API_BASE
RUNPOD_POD_LOOKUP_DOC = "https://api.runpod.io/v2/openapi.yaml"
RUNPOD_POD_STOP_DOC = "https://api.runpod.io/v2/openapi.yaml"
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
_EXPECTED_CONTAINER_DISK_GB = 50
_EXPECTED_VOLUME_DISK_GB = 650
_EXPECTED_VOLUME_MOUNT_PATH = "/workspace"
_EXPECTED_PORTS = ("22/tcp",)
_EXPECTED_STATIC_ENV = {
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
}
_EXPECTED_SECRET_ENV_KEYS = frozenset({"HF_TOKEN", "GPU_BUDGET_SESSION_ID"})
_ALLOWED_PROVIDER_ENV_KEYS = frozenset({"PUBLIC_KEY"})


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
    """Secret-free subset of the authoritative RunPod v2 Pod response."""

    pod_id: str
    gpu_count: int
    provider_gpu_id: str
    gpu_display_name: str
    runtime_gpu_count: int
    execution_identity_hash: str
    data_center_id: str
    cuda_version: str
    secure_cloud: bool
    container_image: str
    container_disk_gb: int
    persistent_volume_disk_gb: int
    persistent_volume_mount_path: str
    ports: tuple[str, ...]
    global_networking_enabled: bool
    ssh_ready: bool
    environment_verified: bool
    desired_status: str
    cost_per_hr: float
    adjusted_cost_per_hr: float
    last_started_at: datetime
    observed_at: datetime
    locked: bool
    network_volume_attached: bool

    @property
    def effective_hourly_usd(self) -> float:
        # The v2 Pod resource exposes ``cost`` as the current compute rate.  The
        # frozen storage rate is added separately by ``run_watchdog``.
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
            "runtime_gpu_count": self.runtime_gpu_count,
            "execution_identity_hash": self.execution_identity_hash,
            "data_center_id": self.data_center_id,
            "cuda_version": self.cuda_version,
            "secure_cloud": self.secure_cloud,
            "container_image": self.container_image,
            "container_disk_gb": self.container_disk_gb,
            "persistent_volume_disk_gb": self.persistent_volume_disk_gb,
            "persistent_volume_mount_path": self.persistent_volume_mount_path,
            "ports": list(self.ports),
            "global_networking_enabled": self.global_networking_enabled,
            "ssh_ready": self.ssh_ready,
            "environment_verified": self.environment_verified,
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


MetadataTransport = Callable[[str, str, float], tuple[int, str]]
StopTransport = Callable[[str, str, float, Mapping[str, str]], tuple[int, str]]


def _default_transport(
    url: str,
    api_key: str,
    timeout: float,
    *,
    method: str,
    payload: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    encoded = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if payload is not None:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=headers,
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


def _default_stop_transport(
    url: str,
    api_key: str,
    timeout: float,
    payload: Mapping[str, str],
) -> tuple[int, str]:
    return _default_transport(url, api_key, timeout, method="POST", payload=payload)


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


def _execution_identity_hash(
    *,
    pod_id: str,
    started_at: datetime,
    provider_gpu_id: str,
    data_center_id: str,
    cuda_version: str,
) -> str:
    canonical = json.dumps(
        {
            "cuda_version": cuda_version,
            "data_center_id": data_center_id,
            "pod_id": pod_id,
            "provider_gpu_id": provider_gpu_id,
            "started_at": started_at.isoformat(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"runpod-v2-execution-identity-v1:{canonical}".encode()).hexdigest()
    return f"sha256:{digest}"


def _parse_exact_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise WatchdogError(f"RunPod metadata field {field} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise WatchdogError(f"RunPod metadata field {field} must be positive")
    return parsed


def _validate_secret_environment(value: Any) -> None:
    """Validate approval-bound environment shape without returning secret values."""

    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise WatchdogError("RunPod v2 environment is missing or malformed")
    allowed = set(_EXPECTED_STATIC_ENV) | _EXPECTED_SECRET_ENV_KEYS | _ALLOWED_PROVIDER_ENV_KEYS
    if set(value) - allowed or not (
        set(_EXPECTED_STATIC_ENV) | _EXPECTED_SECRET_ENV_KEYS
    ).issubset(value):
        raise WatchdogError("RunPod v2 environment violates the approval-bound allow-list")
    if any(not value[key] for key in _EXPECTED_SECRET_ENV_KEYS):
        raise WatchdogError("RunPod v2 environment is missing an approval-bound secret")
    if any(value.get(key) != expected for key, expected in _EXPECTED_STATIC_ENV.items()):
        raise WatchdogError("RunPod v2 compatibility/cache environment drifted")
    public_key = value.get("PUBLIC_KEY")
    if public_key is not None and (not public_key.strip() or len(public_key) > 65536):
        raise WatchdogError("RunPod v2 provider-managed SSH environment is malformed")


def _ssh_ready(value: Any) -> bool:
    if not isinstance(value, Mapping):
        raise WatchdogError("RunPod v2 SSH metadata is missing or malformed")
    ready = False
    for kind in ("proxy", "direct"):
        endpoint = value.get(kind)
        if endpoint is None:
            continue
        if not isinstance(endpoint, Mapping):
            raise WatchdogError("RunPod v2 SSH endpoint is malformed")
        host = endpoint.get("host")
        username = endpoint.get("username")
        port = endpoint.get("port")
        if (
            not isinstance(host, str)
            or not host.strip()
            or not isinstance(username, str)
            or not username.strip()
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 < port < 65536
        ):
            raise WatchdogError("RunPod v2 SSH endpoint is malformed")
        ready = True
    if not ready:
        raise WatchdogError("RunPod v2 SSH endpoint is not ready")
    return True


def parse_pod_metadata(
    payload: Mapping[str, Any],
    *,
    expected_pod_id: str,
    observed_at: datetime,
) -> PodMetadata:
    """Parse and sanitize a live ``GET /v2/pods/{id}`` response.

    The v2 response includes ``env`` and SSH routing details.  Both are
    validated in memory and deliberately discarded before state persistence.
    """

    if payload.get("id") != expected_pod_id:
        raise WatchdogError("RunPod metadata returned a different Pod id")
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        raise WatchdogError("RunPod metadata is missing the GPU object")
    count = _parse_exact_positive_integer(gpu.get("count"), field="gpu.count")
    provider_gpu_id = gpu.get("id")
    if not isinstance(provider_gpu_id, str) or not provider_gpu_id.strip():
        raise WatchdogError("RunPod metadata gpu.id must be nonempty")
    display_name = gpu.get("displayName", provider_gpu_id)
    if not isinstance(display_name, str) or not display_name.strip():
        raise WatchdogError("RunPod metadata GPU display identity must be nonempty")

    cloud = payload.get("cloud")
    if not isinstance(cloud, str) or not cloud:
        raise WatchdogError("RunPod v2 metadata cloud must be nonempty")
    data_center_id = payload.get("dataCenterId")
    if not isinstance(data_center_id, str) or not data_center_id.strip():
        raise WatchdogError("RunPod v2 metadata has no dataCenterId")
    cuda_version = payload.get("cudaVersion")
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        raise WatchdogError("RunPod v2 metadata has no cudaVersion")
    desired_status = payload.get("status")
    if desired_status not in {"RUNNING", "EXITED", "TERMINATED"}:
        raise WatchdogError("RunPod v2 metadata status is unsupported")
    locked = payload.get("locked")
    if not isinstance(locked, bool):
        raise WatchdogError("RunPod metadata locked flag must be boolean")

    container_disk_gb = _parse_exact_positive_integer(payload.get("disk"), field="disk")
    mounts = payload.get("mounts")
    if not isinstance(mounts, Mapping) or set(mounts) != {"persistent"}:
        raise WatchdogError("RunPod v2 persistent mount metadata is missing or unexpected")
    if payload.get("networkVolume") is not None:
        raise WatchdogError("RunPod v2 Pod unexpectedly has a network volume")
    persistent = mounts.get("persistent")
    if not isinstance(persistent, Mapping):
        raise WatchdogError("RunPod v2 persistent mount metadata is malformed")
    volume_disk_gb = _parse_exact_positive_integer(
        persistent.get("size"), field="mounts.persistent.size"
    )
    volume_mount_path = persistent.get("path")
    if not isinstance(volume_mount_path, str) or not volume_mount_path:
        raise WatchdogError("RunPod v2 persistent mount path is missing")
    ports = payload.get("ports")
    if not isinstance(ports, list) or not all(isinstance(item, str) for item in ports):
        raise WatchdogError("RunPod v2 port metadata is malformed")
    global_networking = payload.get("globalNetworking")
    if not isinstance(global_networking, Mapping) or not isinstance(
        global_networking.get("enabled"), bool
    ):
        raise WatchdogError("RunPod v2 globalNetworking metadata is malformed")

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise WatchdogError("RunPod v2 runtime metadata is missing")
    runtime_gpus = runtime.get("gpus")
    if not isinstance(runtime_gpus, list):
        raise WatchdogError("RunPod v2 runtime.gpus metadata is missing")
    uptime = runtime.get("uptime")
    if isinstance(uptime, bool) or not isinstance(uptime, int) or uptime < 0:
        raise WatchdogError("RunPod v2 runtime.uptime must be a nonnegative integer")

    _validate_secret_environment(payload.get("env"))
    ssh_ready = _ssh_ready(payload.get("ssh"))
    current = observed_at.astimezone(UTC)
    created = _parse_timestamp(payload.get("createdAt"), field="createdAt")
    started = _parse_timestamp(payload.get("startedAt"), field="startedAt")
    if started < created - timedelta(minutes=5):
        raise WatchdogError("RunPod startedAt predates createdAt")
    if started > current + timedelta(minutes=5):
        raise WatchdogError("RunPod startedAt is implausibly in the future")
    if uptime > max(0.0, (current - started).total_seconds()) + 600:
        raise WatchdogError("RunPod runtime.uptime disagrees with startedAt")

    image = _parse_container_digest(payload.get("image"), field="image")
    cost = _parse_positive_number(payload.get("cost"), field="cost")
    execution_hash = _execution_identity_hash(
        pod_id=expected_pod_id,
        started_at=started,
        provider_gpu_id=provider_gpu_id.strip(),
        data_center_id=data_center_id.strip(),
        cuda_version=cuda_version.strip(),
    )
    return PodMetadata(
        pod_id=expected_pod_id,
        gpu_count=count,
        provider_gpu_id=provider_gpu_id.strip(),
        gpu_display_name=display_name.strip(),
        runtime_gpu_count=len(runtime_gpus),
        execution_identity_hash=execution_hash,
        data_center_id=data_center_id.strip(),
        cuda_version=cuda_version.strip(),
        secure_cloud=cloud == "SECURE",
        container_image=image,
        container_disk_gb=container_disk_gb,
        persistent_volume_disk_gb=volume_disk_gb,
        persistent_volume_mount_path=volume_mount_path,
        ports=tuple(ports),
        global_networking_enabled=bool(global_networking["enabled"]),
        ssh_ready=ssh_ready,
        environment_verified=True,
        desired_status=str(desired_status),
        cost_per_hr=cost,
        adjusted_cost_per_hr=cost,
        last_started_at=started,
        observed_at=current,
        locked=locked,
        network_volume_attached=False,
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
    allowed_cuda_versions: tuple[str, ...],
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
    if _GPU_FAMILY_RE[family].search(metadata.gpu_display_name) is None or _GPU_FAMILY_RE[
        family
    ].search(metadata.provider_gpu_id) is None:
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
    if not allowed_cuda_versions or metadata.cuda_version not in allowed_cuda_versions:
        raise WatchdogError(
            "RunPod live CUDA host version is outside the frozen launch set: "
            f"observed {metadata.cuda_version!r}"
        )
    if metadata.runtime_gpu_count != expected_gpu_count:
        raise WatchdogError("RunPod runtime GPU inventory does not contain exactly 8 GPUs")
    if not metadata.secure_cloud:
        raise WatchdogError("RunPod live machine is not in Secure Cloud")
    if metadata.container_image != approved_image:
        raise WatchdogError(
            "RunPod live image does not match the approval-bound digest: "
            f"expected {approved_image!r}, observed {metadata.container_image!r}"
        )
    if metadata.container_disk_gb != _EXPECTED_CONTAINER_DISK_GB:
        raise WatchdogError("RunPod live container disk differs from the frozen 50 GB spec")
    if (
        metadata.persistent_volume_disk_gb != _EXPECTED_VOLUME_DISK_GB
        or metadata.persistent_volume_mount_path != _EXPECTED_VOLUME_MOUNT_PATH
    ):
        raise WatchdogError("RunPod live persistent volume differs from the frozen 650 GB spec")
    if metadata.ports != _EXPECTED_PORTS:
        raise WatchdogError("RunPod live ports differ from the frozen SSH-only spec")
    if metadata.global_networking_enabled:
        raise WatchdogError("RunPod global networking must remain disabled")
    if metadata.network_volume_attached:
        raise WatchdogError("RunPod must not attach a network volume")
    if not metadata.ssh_ready or not metadata.environment_verified:
        raise WatchdogError("RunPod SSH or environment verification is incomplete")
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
    """Minimal RunPod v2 client with injectable GET and action transports."""

    def __init__(
        self,
        *,
        pod_id: str,
        api_key_env: str = "RUNPOD_API_KEY",
        endpoint_base: str = RUNPOD_API_BASE,
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
        self._metadata_endpoint = pod_endpoint
        self._stop_endpoint = f"{pod_endpoint}/action"
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
        status = payload.get("status")
        if status not in {"RUNNING", "EXITED", "TERMINATED"}:
            raise WatchdogError("RunPod v2 metadata status is unsupported")
        return str(status)

    def stop(self) -> None:
        status, _body = self._stop_transport(
            self._stop_endpoint,
            self._api_key,
            self._timeout_seconds,
            {"action": "stop"},
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
    allowed_cuda_versions: tuple[str, ...],
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
            allowed_cuda_versions=allowed_cuda_versions,
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
    initial_execution_identity_hash = metadata.execution_identity_hash
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
                allowed_cuda_versions=allowed_cuda_versions,
                expected_container_image=expected_container_image,
                limits=limits,
            )
            if current.last_started_at != initial_started_at:
                raise WatchdogError("RunPod lastStartedAt changed after watchdog arming")
            if current.execution_identity_hash != initial_execution_identity_hash:
                raise WatchdogError("RunPod v2 execution identity changed after watchdog arming")
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
    "RUNPOD_API_BASE",
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
