"""Fail-closed RunPod GPU-cost watchdog.

The watchdog uses RunPod's supported REST v1 Pod metadata rather than a
caller-supplied hourly rate.  It runs independently from the experiment and
only invokes the non-destructive Pod stop endpoint.  Pod deletion remains an
explicit post-sync operation.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash, write_json

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
RUNPOD_API_BASE = RUNPOD_REST_BASE
RUNPOD_POD_LOOKUP_DOC = "https://docs.runpod.io/api-reference/pods/GET/pods/podId"
RUNPOD_POD_STOP_DOC = "https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop"
WATCHDOG_VERSION = "runpod-gpu-cost-watchdog-v2"
HOST_REARM_ACK_PROTOCOL = "runpod-host-rearm-watchdog-ack-v2"
HOST_REARM_ACK_FILENAME = "host_rearm_watchdog_ack.json"
HOST_REARM_HEARTBEAT_MAX_AGE_SECONDS = 20.0
PROVIDER_API = "rest-v1"
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
_RUNPOD_GO_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))? (?P<offset>[+-]\d{4}) UTC\Z"
)
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
_NAMESPACED_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HOST_REARM_ACK_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "status",
        "expected_session_hash",
        "expected_phase",
        "lifecycle_before_hash",
        "pod_id_hash",
        "watcher_pid",
        "watcher_process_identity_hash",
        "acknowledged_at",
        "record_hash",
    }
)
_WATCHDOG_STATE_KEYS = frozenset(
    {
        "schema_version",
        "watchdog_version",
        "pod_id",
        "status",
        "armed_at",
        "updated_at",
        "live_metadata",
        "limits",
        "deadline",
        "stop_reason",
        "action",
        "deletion",
        "error",
    }
)


class WatchdogError(RuntimeError):
    """The watchdog cannot establish or enforce a safe stop deadline."""


def _host_ack_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise WatchdogError("host re-arm watchdog acknowledgement timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WatchdogError(
            "host re-arm watchdog acknowledgement timestamp is malformed"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WatchdogError("host re-arm watchdog acknowledgement timestamp is malformed")
    return parsed.astimezone(UTC)


def _host_process_identity_hash(pid: int) -> str:
    """Return a stable boot/process-start identity, not merely a reusable PID."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise WatchdogError("host re-arm watchdog process identity is malformed")
    proc = Path("/proc") / str(pid)
    try:
        if proc.is_dir():
            stat_raw = (proc / "stat").read_text(encoding="utf-8")
            closing = stat_raw.rfind(")")
            fields = stat_raw[closing + 2 :].split() if closing >= 0 else []
            if len(fields) <= 19:
                raise WatchdogError("host re-arm watchdog process identity is unavailable")
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            ).strip()
            if not boot_id:
                raise WatchdogError("host re-arm watchdog boot identity is unavailable")
            return stable_hash(
                {
                    "boot_id": boot_id,
                    "pid": pid,
                    "process_start_ticks": fields[19],
                }
            )
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired) as exc:
        raise WatchdogError("host re-arm watchdog process identity is unavailable") from exc
    started = result.stdout.strip()
    if result.returncode != 0 or not started:
        raise WatchdogError("host re-arm watchdog process is not live")
    return stable_hash({"pid": pid, "process_started_at": started})


def _host_ack_payload(
    *,
    expected_session_hash: str,
    expected_phase: str,
    lifecycle_before_hash: str,
    pod_id: str,
    watcher_pid: int,
    acknowledged_at: datetime,
    watcher_process_identity_hash: str | None = None,
) -> dict[str, Any]:
    process_identity = watcher_process_identity_hash or _host_process_identity_hash(
        watcher_pid
    )
    if _NAMESPACED_HASH_RE.fullmatch(process_identity) is None:
        raise WatchdogError("host re-arm watchdog process identity hash is malformed")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": HOST_REARM_ACK_PROTOCOL,
        "status": "armed_and_provider_exited_verified",
        "expected_session_hash": expected_session_hash,
        "expected_phase": expected_phase,
        "lifecycle_before_hash": lifecycle_before_hash,
        "pod_id_hash": stable_hash({"runpod_pod_id": pod_id}),
        "watcher_pid": watcher_pid,
        "watcher_process_identity_hash": process_identity,
        "acknowledged_at": acknowledged_at.astimezone(UTC).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _write_host_rearm_ack(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    if os.path.lexists(destination):
        raise WatchdogError("host re-arm watchdog acknowledgement is already claimed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise WatchdogError(
            "host re-arm watchdog acknowledgement was concurrently claimed"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def validate_host_rearm_ack(
    path: str | Path,
    *,
    expected_session_hash: str,
    expected_phase: str,
    expected_lifecycle_hash: str,
    expected_pod_id: str,
    observed_at: datetime | None = None,
    maximum_age_seconds: float = 600,
) -> dict[str, Any]:
    """Authenticate a live host watcher before a re-arm can request start."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WatchdogError("host re-arm watchdog acknowledgement is missing or unsafe")
    details = source.lstat()
    if details.st_nlink != 1 or details.st_uid != os.getuid():
        raise WatchdogError("host re-arm watchdog acknowledgement ownership is unsafe")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WatchdogError("host re-arm watchdog acknowledgement is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != _HOST_REARM_ACK_KEYS:
        raise WatchdogError("host re-arm watchdog acknowledgement schema is unsupported")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if payload.get("record_hash") != stable_hash(unsigned):
        raise WatchdogError("host re-arm watchdog acknowledgement hash mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_version") != HOST_REARM_ACK_PROTOCOL
        or payload.get("status") != "armed_and_provider_exited_verified"
        or payload.get("expected_session_hash") != expected_session_hash
        or payload.get("expected_phase") != expected_phase
        or payload.get("lifecycle_before_hash") != expected_lifecycle_hash
        or payload.get("pod_id_hash")
        != stable_hash({"runpod_pod_id": expected_pod_id})
    ):
        raise WatchdogError("host re-arm watchdog acknowledgement binding mismatch")
    if (
        not math.isfinite(maximum_age_seconds)
        or maximum_age_seconds <= 0
        or maximum_age_seconds > 600
    ):
        raise ValueError("host re-arm acknowledgement age limit is invalid")
    current = (observed_at or datetime.now(UTC)).astimezone(UTC)
    acknowledged_at = _host_ack_timestamp(payload.get("acknowledged_at"))
    age = (current - acknowledged_at).total_seconds()
    if age < -60 or age > maximum_age_seconds:
        raise WatchdogError("host re-arm watchdog acknowledgement is stale or future-dated")
    pid = payload.get("watcher_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise WatchdogError("host re-arm watchdog process identity is malformed")
    expected_process_identity = payload.get("watcher_process_identity_hash")
    if (
        not isinstance(expected_process_identity, str)
        or _NAMESPACED_HASH_RE.fullmatch(expected_process_identity) is None
        or not hmac.compare_digest(
            expected_process_identity,
            _host_process_identity_hash(pid),
        )
    ):
        raise WatchdogError("host re-arm watchdog process identity changed")
    return payload


def validate_host_rearm_waiting_state(
    path: str | Path,
    *,
    expected_pod_id: str,
    expected_maximum_runtime_hours: float,
    expected_hourly_total_usd: float,
    observed_at: datetime | None = None,
    maximum_age_seconds: float = HOST_REARM_HEARTBEAT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Validate the live heartbeat written by the pre-start host watcher."""

    if (
        not math.isfinite(maximum_age_seconds)
        or maximum_age_seconds <= 0
        or maximum_age_seconds > HOST_REARM_HEARTBEAT_MAX_AGE_SECONDS
    ):
        raise ValueError("host re-arm heartbeat age limit is invalid")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise WatchdogError("host re-arm watchdog heartbeat is missing or unsafe")
    before = source.lstat()
    if before.st_nlink != 1 or before.st_uid != os.getuid():
        raise WatchdogError("host re-arm watchdog heartbeat ownership is unsafe")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WatchdogError("host re-arm watchdog heartbeat is unreadable") from exc
    after = source.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise WatchdogError("host re-arm watchdog heartbeat changed while read")
    if not isinstance(payload, dict) or set(payload) != _WATCHDOG_STATE_KEYS:
        raise WatchdogError("host re-arm watchdog heartbeat schema is unsupported")
    if (
        payload.get("schema_version") != 2
        or payload.get("watchdog_version") != WATCHDOG_VERSION
        or payload.get("pod_id") != expected_pod_id
        or payload.get("status") != "waiting_for_start"
        or payload.get("live_metadata") is not None
        or payload.get("deadline") is not None
        or payload.get("stop_reason") is not None
        or payload.get("action") != "stop_only_preserve_volume"
        or payload.get("deletion") != "manual_after_verified_sync"
        or payload.get("error") is not None
    ):
        raise WatchdogError("host re-arm watchdog heartbeat binding mismatch")
    limits = payload.get("limits")
    if not isinstance(limits, dict):
        raise WatchdogError("host re-arm watchdog heartbeat limits are missing")
    runtime = limits.get("maximum_runtime_hours")
    hourly_total = limits.get("maximum_approved_hourly_total_usd")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) <= 0
        or float(runtime) > expected_maximum_runtime_hours + 1e-9
        or isinstance(hourly_total, bool)
        or not isinstance(hourly_total, (int, float))
        or not math.isclose(
            float(hourly_total),
            expected_hourly_total_usd,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise WatchdogError("host re-arm watchdog heartbeat limit binding mismatch")
    current = (observed_at or datetime.now(UTC)).astimezone(UTC)
    armed_at = _host_ack_timestamp(payload.get("armed_at"))
    updated_at = _host_ack_timestamp(payload.get("updated_at"))
    age = (current - updated_at).total_seconds()
    if age < -5 or age > maximum_age_seconds or updated_at < armed_at:
        raise WatchdogError("host re-arm watchdog heartbeat is stale or future-dated")
    return payload


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
    maximum_approved_compute_hourly_usd: float | None = None

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
        approved_compute = self.maximum_approved_compute_hourly_usd
        if approved_compute is not None and (
            isinstance(approved_compute, bool)
            or not math.isfinite(float(approved_compute))
            or approved_compute <= 0
        ):
            raise ValueError("maximum approved compute hourly cost must be positive and finite")
        if approved_compute is not None and approved is not None and not math.isclose(
            approved_compute + self.maximum_approved_storage_hourly_usd,
            approved,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("approved compute plus storage must equal the all-in hourly total")
        if (
            approved_compute is None
            and approved is not None
            and approved - self.maximum_approved_storage_hourly_usd <= 0
        ):
            raise ValueError("all-in hourly total leaves no approved compute hourly cost")
        if self.prior_committed_gpu_usd >= self.global_safe_budget_usd:
            raise ValueError("prior committed GPU cost leaves no safety-adjusted GPU budget")

    @property
    def global_safe_budget_usd(self) -> float:
        return self.gpu_hard_stop_usd * (1 - self.safety_margin_fraction)

    @property
    def safe_budget_usd(self) -> float:
        return self.global_safe_budget_usd - self.prior_committed_gpu_usd

    @property
    def approved_compute_hourly_usd(self) -> float | None:
        """Return the compute-only ceiling, excluding separately billed storage."""

        if self.maximum_approved_compute_hourly_usd is not None:
            return self.maximum_approved_compute_hourly_usd
        if self.maximum_approved_hourly_total_usd is None:
            return None
        return self.maximum_approved_hourly_total_usd - self.maximum_approved_storage_hourly_usd


@dataclass(frozen=True, slots=True)
class PodMetadata:
    """Secret-free subset of the authoritative RunPod REST v1 Pod response."""

    pod_id: str
    pod_name: str
    gpu_count: int
    provider_gpu_id: str
    gpu_display_name: str
    runtime_gpu_count: int | None
    machine_id_hash: str
    execution_identity_hash: str
    data_center_id: str
    cuda_version: str | None
    secure_cloud: bool
    container_image: str
    container_disk_gb: int
    persistent_volume_disk_gb: int
    persistent_volume_mount_path: str
    ports: tuple[str, ...]
    global_networking_enabled: bool | None
    ssh_ready: bool
    direct_ssh_ready: bool
    direct_ssh_endpoint_hash: str | None
    environment_verified: bool
    desired_status: str
    cost_per_hr: float
    adjusted_cost_per_hr: float
    last_started_at: datetime
    observed_at: datetime
    locked: bool | None
    interruptible: bool | None
    network_volume_attached: bool

    @property
    def effective_hourly_usd(self) -> float:
        # Conservatively use the larger of the list and adjusted rates.  The
        # frozen storage rate is added separately by ``run_watchdog``.
        return max(self.cost_per_hr, self.adjusted_cost_per_hr)

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
            "provider_api": PROVIDER_API,
            "provider_evidence_unavailable": [
                "cuda_version",
                "global_networking_enabled",
                "interruptible",
                "locked",
                "runtime_gpu_count",
            ],
            "pod_id": self.pod_id,
            "pod_name": self.pod_name,
            "gpu_count": self.gpu_count,
            "provider_gpu_id": self.provider_gpu_id,
            "gpu_display_name": self.gpu_display_name,
            "runtime_gpu_count": self.runtime_gpu_count,
            "machine_id_hash": self.machine_id_hash,
            "execution_identity_hash": self.execution_identity_hash,
            "data_center_id": self.data_center_id,
            "cuda_version": self.cuda_version,
            "secure_cloud": self.secure_cloud,
            "container_image": self.container_image,
            "container_disk_gb": self.container_disk_gb,
            "persistent_volume_disk_gb": self.persistent_volume_disk_gb,
            "persistent_volume_mount_path": self.persistent_volume_mount_path,
            "ports": list(self.ports),
            # These fields are deliberately null because the currently
            # returned REST v1 schema does not attest to them.  If optional
            # fields appear, they may tighten in-memory rejection but are not
            # elevated into portable evidence.
            "global_networking_enabled": None,
            "ssh_ready": self.ssh_ready,
            "direct_ssh_ready": self.direct_ssh_ready,
            "direct_ssh_endpoint_hash": self.direct_ssh_endpoint_hash,
            "environment_verified": self.environment_verified,
            "desired_status": self.desired_status,
            "cost_per_hr": self.cost_per_hr,
            "adjusted_cost_per_hr": self.adjusted_cost_per_hr,
            "effective_hourly_usd": self.effective_hourly_usd,
            "last_started_at": self.last_started_at.isoformat(),
            "observed_at": self.observed_at.isoformat(),
            "locked": None,
            "interruptible": None,
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
StopTransport = Callable[[str, str, float, Mapping[str, str] | None], tuple[int, str]]


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
    payload: Mapping[str, str] | None,
) -> tuple[int, str]:
    return _default_transport(url, api_key, timeout, method="POST", payload=payload)


def _default_metadata_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
    return _default_transport(url, api_key, timeout, method="GET")


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise WatchdogError(f"RunPod metadata field {field} must be a nonempty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # The live REST v1 service currently serializes lastStartedAt using
        # Go's ``String`` form (for example ``... +0000 UTC``), despite the
        # OpenAPI contract documenting ISO-8601.  Accept only that exact second
        # provider form; do not fall back to a permissive date parser.
        match = _RUNPOD_GO_TIMESTAMP_RE.fullmatch(value)
        if match is None:
            raise WatchdogError(
                f"RunPod metadata field {field} is neither ISO-8601 nor RunPod Go time"
            ) from None
        fraction = (match.group("fraction") or "0")[:6].ljust(6, "0")
        normalized = f"{match.group('date')}.{fraction} {match.group('offset')}"
        try:
            parsed = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S.%f %z")
        except ValueError as exc:  # pragma: no cover - regex already constrains shape
            raise WatchdogError(f"RunPod metadata field {field} is malformed") from exc
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
    pod_name: str,
    started_at: datetime,
    machine_id: str,
    provider_gpu_id: str,
    data_center_id: str,
    container_image: str,
) -> str:
    canonical = json.dumps(
        {
            "container_image": container_image,
            "data_center_id": data_center_id,
            "machine_id": machine_id,
            "pod_id": pod_id,
            "pod_name": pod_name,
            "provider_gpu_id": provider_gpu_id,
            "started_at": started_at.isoformat(),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        f"runpod-rest-v1-execution-identity-v1:{canonical}".encode()
    ).hexdigest()
    return f"sha256:{digest}"


def _opaque_hash(namespace: str, value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(f'{namespace}:{canonical}'.encode()).hexdigest()}"


def _parse_exact_positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise WatchdogError(f"RunPod metadata field {field} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise WatchdogError(f"RunPod metadata field {field} must be positive")
    return parsed


def _validate_secret_environment(
    value: Any,
    *,
    expected_session_hash: str,
    expected_hf_token_hash: str,
) -> None:
    """Bind provider secrets in memory without returning or persisting their values."""

    if (
        _NAMESPACED_HASH_RE.fullmatch(expected_session_hash) is None
        or _NAMESPACED_HASH_RE.fullmatch(expected_hf_token_hash) is None
    ):
        raise WatchdogError("watchdog secret-binding hashes are malformed")

    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise WatchdogError("RunPod REST v1 environment is missing or malformed")
    allowed = set(_EXPECTED_STATIC_ENV) | _EXPECTED_SECRET_ENV_KEYS | _ALLOWED_PROVIDER_ENV_KEYS
    if set(value) - allowed or not (set(_EXPECTED_STATIC_ENV) | _EXPECTED_SECRET_ENV_KEYS).issubset(
        value
    ):
        raise WatchdogError("RunPod REST v1 environment violates the approval-bound allow-list")
    if any(not value[key] for key in _EXPECTED_SECRET_ENV_KEYS):
        raise WatchdogError("RunPod REST v1 environment is missing an approval-bound secret")
    if any(value.get(key) != expected for key, expected in _EXPECTED_STATIC_ENV.items()):
        raise WatchdogError("RunPod REST v1 compatibility/cache environment drifted")
    observed_session_hash = stable_hash(
        {"opaque_gpu_session_id": value["GPU_BUDGET_SESSION_ID"]}
    )
    observed_hf_token_hash = stable_hash({"hf_token": value["HF_TOKEN"]})
    if not hmac.compare_digest(observed_session_hash, expected_session_hash):
        raise WatchdogError("RunPod REST v1 GPU session identity disagrees with approval")
    if not hmac.compare_digest(observed_hf_token_hash, expected_hf_token_hash):
        raise WatchdogError("RunPod REST v1 Hugging Face credential identity drifted")
    public_key = value.get("PUBLIC_KEY")
    if public_key is not None and (not public_key.strip() or len(public_key) > 65536):
        raise WatchdogError("RunPod REST v1 provider-managed SSH environment is malformed")


def _direct_ssh_evidence(
    public_ip: Any,
    port_mappings: Any,
    *,
    required: bool,
) -> tuple[bool, str | None]:
    """Validate direct SSH routing without persisting the address or port."""

    absent = public_ip in (None, "") and port_mappings in (None, {})
    if absent and not required:
        return False, None
    if not isinstance(public_ip, str) or not public_ip:
        raise WatchdogError("RunPod REST v1 direct SSH publicIp is unavailable")
    try:
        parsed_ip = ipaddress.ip_address(public_ip)
    except ValueError as exc:
        raise WatchdogError("RunPod REST v1 direct SSH publicIp is malformed") from exc
    if parsed_ip.version != 4:
        raise WatchdogError("RunPod REST v1 direct SSH publicIp must be IPv4")
    if not isinstance(port_mappings, Mapping) or set(port_mappings) != {"22"}:
        raise WatchdogError("RunPod REST v1 direct SSH portMappings are not SSH-only")
    port = port_mappings.get("22")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise WatchdogError("RunPod REST v1 direct SSH port mapping is malformed")
    return True, _opaque_hash(
        "runpod-rest-v1-direct-ssh-v1",
        {"public_ip": str(parsed_ip), "public_port": port},
    )


def _reconcile_aliases(*values: Any, field: str) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        raise WatchdogError(f"RunPod REST v1 metadata field {field} is missing")
    if any(value != present[0] for value in present[1:]):
        raise WatchdogError(f"RunPod REST v1 metadata aliases for {field} disagree")
    return present[0]


def _optional_boolean(payload: Mapping[str, Any], field: str) -> bool | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, bool):
        raise WatchdogError(f"RunPod REST v1 metadata field {field} must be boolean")
    return value


def _optional_global_networking(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("enabled"), bool):
        return bool(value["enabled"])
    raise WatchdogError("RunPod REST v1 globalNetworking metadata is malformed")


def _network_volume_attached(payload: Mapping[str, Any]) -> bool:
    """Reconcile every supported network-volume alias and reject split-brain metadata."""

    network_volume = payload.get("networkVolume")
    nested_identifiers: list[str] = []
    object_present = network_volume is not None
    if isinstance(network_volume, Mapping):
        for key in ("id", "networkVolumeId", "networkVolumeID"):
            if key not in network_volume or network_volume[key] is None:
                continue
            value = network_volume[key]
            if not isinstance(value, str) or not value.strip():
                raise WatchdogError("RunPod REST v1 network volume identity is malformed")
            nested_identifiers.append(value.strip())
    elif isinstance(network_volume, str):
        if not network_volume.strip():
            raise WatchdogError("RunPod REST v1 network volume identity is malformed")
        nested_identifiers.append(network_volume.strip())
    elif network_volume is not None:
        raise WatchdogError("RunPod REST v1 networkVolume metadata is malformed")

    alias_identifiers: list[str] = []
    for key in ("networkVolumeId", "networkVolumeID", "network_volume_id"):
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise WatchdogError("RunPod REST v1 network volume alias is malformed")
        alias_identifiers.append(value.strip())

    identifiers = [*nested_identifiers, *alias_identifiers]
    if identifiers and any(value != identifiers[0] for value in identifiers[1:]):
        raise WatchdogError("RunPod REST v1 network volume aliases disagree")
    return object_present or bool(alias_identifiers)


def parse_pod_metadata(
    payload: Mapping[str, Any],
    *,
    expected_pod_id: str,
    expected_session_hash: str,
    expected_hf_token_hash: str,
    observed_at: datetime,
) -> PodMetadata:
    """Parse and sanitize a live ``GET /v1/pods/{id}`` response.

    The response includes ``env`` and direct SSH routing details.  Secret
    values and endpoint coordinates are validated in memory and deliberately
    discarded before state persistence.
    """

    if payload.get("id") != expected_pod_id:
        raise WatchdogError("RunPod metadata returned a different Pod id")
    pod_name = payload.get("name")
    if not isinstance(pod_name, str) or not pod_name.strip():
        raise WatchdogError("RunPod REST v1 metadata name must be nonempty")

    machine = payload.get("machine")
    if not isinstance(machine, Mapping):
        raise WatchdogError("RunPod REST v1 machine metadata is missing")
    provider_gpu_id = machine.get("gpuTypeId")
    if not isinstance(provider_gpu_id, str) or not provider_gpu_id.strip():
        raise WatchdogError("RunPod REST v1 machine.gpuTypeId must be nonempty")
    provider_gpu_id = provider_gpu_id.strip()
    data_center_id = machine.get("dataCenterId")
    if not isinstance(data_center_id, str) or not data_center_id.strip():
        raise WatchdogError("RunPod REST v1 machine metadata has no dataCenterId")
    secure_cloud = machine.get("secureCloud")
    if not isinstance(secure_cloud, bool):
        raise WatchdogError("RunPod REST v1 machine.secureCloud must be boolean")
    machine_id = payload.get("machineId")
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise WatchdogError("RunPod REST v1 machineId must be nonempty")

    gpu = payload.get("gpu")
    if gpu is not None and not isinstance(gpu, Mapping):
        raise WatchdogError("RunPod REST v1 gpu metadata is malformed")
    nested_count = gpu.get("count") if isinstance(gpu, Mapping) else None
    count = _parse_exact_positive_integer(
        _reconcile_aliases(payload.get("gpuCount"), nested_count, field="gpuCount"),
        field="gpuCount",
    )
    if isinstance(gpu, Mapping) and gpu.get("id") is not None and gpu.get("id") != provider_gpu_id:
        raise WatchdogError("RunPod REST v1 GPU and machine identities disagree")
    display_candidates = (
        machine.get("gpuDisplayName"),
        gpu.get("displayName") if isinstance(gpu, Mapping) else None,
        provider_gpu_id,
    )
    display_name = next(
        (value.strip() for value in display_candidates if isinstance(value, str) and value.strip()),
        provider_gpu_id,
    )

    desired_status = payload.get("desiredStatus")
    if desired_status not in {"RUNNING", "EXITED", "TERMINATED"}:
        raise WatchdogError("RunPod REST v1 metadata desiredStatus is unsupported")
    locked = _optional_boolean(payload, "locked")
    interruptible = _optional_boolean(payload, "interruptible")

    container_disk_gb = _parse_exact_positive_integer(
        payload.get("containerDiskInGb"), field="containerDiskInGb"
    )
    volume_disk_gb = _parse_exact_positive_integer(payload.get("volumeInGb"), field="volumeInGb")
    volume_mount_path = payload.get("volumeMountPath", payload.get("mount"))
    if not isinstance(volume_mount_path, str) or not volume_mount_path:
        raise WatchdogError("RunPod REST v1 persistent volume mount path is missing")
    ports = payload.get("ports")
    if not isinstance(ports, list) or not all(isinstance(item, str) for item in ports):
        raise WatchdogError("RunPod REST v1 port metadata is malformed")
    global_networking = _optional_global_networking(payload.get("globalNetworking"))
    network_volume_attached = _network_volume_attached(payload)

    _validate_secret_environment(
        payload.get("env"),
        expected_session_hash=expected_session_hash,
        expected_hf_token_hash=expected_hf_token_hash,
    )
    direct_ssh_ready, direct_ssh_hash = _direct_ssh_evidence(
        payload.get("publicIp"),
        payload.get("portMappings"),
        required=desired_status == "RUNNING",
    )
    current = observed_at.astimezone(UTC)
    started = _parse_timestamp(payload.get("lastStartedAt"), field="lastStartedAt")
    if started > current + timedelta(minutes=5):
        raise WatchdogError("RunPod lastStartedAt is implausibly in the future")

    image = _parse_container_digest(
        _reconcile_aliases(payload.get("imageName"), payload.get("image"), field="imageName"),
        field="imageName",
    )
    cost = _parse_positive_number(payload.get("costPerHr"), field="costPerHr")
    adjusted_raw = payload.get("adjustedCostPerHr")
    adjusted_cost = (
        cost
        if adjusted_raw is None
        else _parse_positive_number(adjusted_raw, field="adjustedCostPerHr")
    )
    machine_hash = _opaque_hash(
        "runpod-rest-v1-machine-v1",
        {"machine_id": machine_id.strip()},
    )
    execution_hash = _execution_identity_hash(
        pod_id=expected_pod_id,
        pod_name=pod_name.strip(),
        started_at=started,
        machine_id=machine_id.strip(),
        provider_gpu_id=provider_gpu_id,
        data_center_id=data_center_id.strip(),
        container_image=image,
    )
    return PodMetadata(
        pod_id=expected_pod_id,
        pod_name=pod_name.strip(),
        gpu_count=count,
        provider_gpu_id=provider_gpu_id,
        gpu_display_name=display_name,
        runtime_gpu_count=None,
        machine_id_hash=machine_hash,
        execution_identity_hash=execution_hash,
        data_center_id=data_center_id.strip(),
        cuda_version=None,
        secure_cloud=secure_cloud,
        container_image=image,
        container_disk_gb=container_disk_gb,
        persistent_volume_disk_gb=volume_disk_gb,
        persistent_volume_mount_path=volume_mount_path,
        ports=tuple(ports),
        global_networking_enabled=global_networking,
        ssh_ready=direct_ssh_ready,
        direct_ssh_ready=direct_ssh_ready,
        direct_ssh_endpoint_hash=direct_ssh_hash,
        environment_verified=True,
        desired_status=str(desired_status),
        cost_per_hr=cost,
        adjusted_cost_per_hr=adjusted_cost,
        last_started_at=started,
        observed_at=current,
        locked=locked,
        interruptible=interruptible,
        network_volume_attached=network_volume_attached,
    )


def normalize_gpu_family(value: str) -> str:
    try:
        return _EXPECTED_FAMILY_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported expected GPU family: {value}") from exc


@dataclass(frozen=True, slots=True)
class LifecyclePodBinding:
    """Private, authenticated lifecycle identity used before constructing a stop client."""

    pod_id: str
    operation: str
    phase: str
    session_hash: str
    pod_status: str


def _load_lifecycle_binding(lifecycle_state_path: str | Path) -> tuple[LifecyclePodBinding, dict[str, Any]]:
    """Load the lifecycle record through its secure owner/link/hash/schema checks."""

    # Import the stdlib-only reader: this must run on a fresh provider image
    # before Pydantic, PyYAML, or the project virtual environment is installed.
    from model_forensics.runpod_lifecycle_state import (
        RunpodLifecycleStateError,
        authorization_from_state,
        load_lifecycle_state,
    )

    try:
        state = load_lifecycle_state(Path(lifecycle_state_path))
        authorization = authorization_from_state(state)
    except (OSError, TypeError, ValueError, RunpodLifecycleStateError) as exc:
        raise WatchdogError("private RunPod lifecycle binding is invalid") from exc
    pod = state.get("pod")
    if not isinstance(pod, Mapping):
        raise WatchdogError("private RunPod lifecycle has no bound Pod")
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or _POD_ID_RE.fullmatch(pod_id) is None:
        raise WatchdogError("private RunPod lifecycle Pod identity is malformed")
    operation = state.get("operation")
    pod_status = pod.get("status")
    if not isinstance(operation, str) or not isinstance(pod_status, str):
        raise WatchdogError("private RunPod lifecycle operation or status is malformed")
    binding = LifecyclePodBinding(
        pod_id=pod_id,
        operation=operation,
        phase=authorization.phase,
        session_hash=authorization.session_hash,
        pod_status=pod_status,
    )
    return binding, state


def bind_lifecycle_pod(
    *,
    lifecycle_state_path: str | Path,
    expected_session_hash: str,
    expected_phase: str,
    ambient_pod_id: str | None = None,
    waiting_for_rearm: bool = False,
) -> str:
    """Return only the lifecycle-bound Pod id after all local identity checks.

    In normal mode the active authorization must already match the authenticated
    reservation.  ``waiting_for_rearm`` is the one deliberate exception: it
    binds a stopped Pod before mutation while requiring the supplied session
    hash to be fresh relative to the entire lifecycle history.
    """

    if _NAMESPACED_HASH_RE.fullmatch(expected_session_hash) is None:
        raise WatchdogError("expected GPU session hash is malformed")
    if not isinstance(expected_phase, str) or not expected_phase.strip():
        raise WatchdogError("expected GPU phase is malformed")
    if ambient_pod_id is not None and _POD_ID_RE.fullmatch(ambient_pod_id) is None:
        raise WatchdogError("ambient RunPod Pod identity is malformed")
    binding, state = _load_lifecycle_binding(lifecycle_state_path)
    if ambient_pod_id is not None and not hmac.compare_digest(binding.pod_id, ambient_pod_id):
        # Never construct a stop-capable client for an ambient or unrelated Pod.
        raise WatchdogError("ambient Pod is not the private lifecycle-bound research Pod")

    if waiting_for_rearm:
        if binding.operation != "stopped" or binding.pod_status != "EXITED":
            raise WatchdogError("host watcher can wait only on an authenticated stopped Pod")
        history = state.get("authorization_history")
        if not isinstance(history, list):  # already schema-checked; defensive for typing
            raise WatchdogError("private RunPod lifecycle history is malformed")
        used_hashes = {
            str(item.get("session_hash")) for item in history if isinstance(item, Mapping)
        }
        used_hashes.add(binding.session_hash)
        if expected_session_hash in used_hashes:
            raise WatchdogError("host watcher requires a fresh GPU session identity")
        return binding.pod_id

    if binding.operation not in {"created", "rearmed"} or binding.pod_status != "RUNNING":
        raise WatchdogError("private RunPod lifecycle is not an authenticated running Pod")
    if not hmac.compare_digest(binding.session_hash, expected_session_hash):
        raise WatchdogError("private lifecycle session disagrees with authenticated reservation")
    if binding.phase != expected_phase:
        raise WatchdogError("private lifecycle phase disagrees with authenticated reservation")
    return binding.pod_id


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
    if (
        _GPU_FAMILY_RE[family].search(metadata.gpu_display_name) is None
        or _GPU_FAMILY_RE[family].search(metadata.provider_gpu_id) is None
    ):
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
    if not allowed_cuda_versions:
        raise WatchdogError("RunPod approved CUDA host-version set is empty")
    if metadata.cuda_version is not None and metadata.cuda_version not in allowed_cuda_versions:
        raise WatchdogError(
            "RunPod live CUDA host version is outside the frozen launch set: "
            f"observed {metadata.cuda_version!r}"
        )
    if metadata.runtime_gpu_count is not None and metadata.runtime_gpu_count != expected_gpu_count:
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
    if metadata.global_networking_enabled is True:
        raise WatchdogError("RunPod global networking must remain disabled")
    if metadata.network_volume_attached:
        raise WatchdogError("RunPod must not attach a network volume")
    if (
        not metadata.ssh_ready
        or not metadata.direct_ssh_ready
        or metadata.direct_ssh_endpoint_hash is None
        or not metadata.environment_verified
    ):
        raise WatchdogError("RunPod SSH or environment verification is incomplete")
    if metadata.desired_status != "RUNNING":
        raise WatchdogError(
            f"RunPod Pod must be RUNNING when watchdog is armed; got {metadata.desired_status}"
        )
    if metadata.locked is True:
        raise WatchdogError("RunPod Pod is locked; the official API says locked Pods cannot stop")
    if metadata.interruptible is True:
        raise WatchdogError("RunPod Pod is interruptible; the approved rental mode drifted")
    approved = limits.approved_compute_hourly_usd
    live_rate_ceiling = max(metadata.cost_per_hr, metadata.effective_hourly_usd)
    if approved is not None and live_rate_ceiling > approved + 1e-6:
        raise WatchdogError(
            "RunPod live compute hourly cost exceeds the approved compute quote: "
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
    """Minimal RunPod REST v1 client with injectable GET and stop transports."""

    def __init__(
        self,
        *,
        pod_id: str,
        expected_session_hash: str,
        api_key_env: str = "RUNPOD_API_KEY",
        hf_token_env: str = "HF_TOKEN",
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
        if _NAMESPACED_HASH_RE.fullmatch(expected_session_hash) is None:
            raise WatchdogError("expected GPU session hash is malformed")
        hf_token = os.environ.get(hf_token_env)
        if not hf_token:
            raise WatchdogError(
                f"required secret environment variable is unset: {hf_token_env}"
            )
        if not endpoint_base.startswith("https://"):
            raise ValueError("RunPod endpoint must use HTTPS")
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("RunPod request timeout must be in (0, 60]")
        self.pod_id = pod_id
        pod_endpoint = f"{endpoint_base.rstrip('/')}/pods/{pod_id}"
        self._metadata_endpoint = (
            f"{pod_endpoint}?includeMachine=true&includeNetworkVolume=true&includeTemplate=true"
        )
        self._stop_endpoint = f"{pod_endpoint}/stop"
        self._api_key = api_key
        self._stop_transport = transport or _default_stop_transport
        self._metadata_transport = metadata_transport or _default_metadata_transport
        self._timeout_seconds = timeout_seconds
        self._expected_session_hash = expected_session_hash
        self._expected_hf_token_hash = stable_hash({"hf_token": hf_token})

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
            expected_session_hash=self._expected_session_hash,
            expected_hf_token_hash=self._expected_hf_token_hash,
            observed_at=observed_at or datetime.now(UTC),
        )

    def desired_status(self) -> str:
        payload = self._get_payload()
        if payload.get("id") != self.pod_id:
            raise WatchdogError("RunPod metadata returned a different Pod id")
        status = payload.get("desiredStatus")
        if status not in {"RUNNING", "EXITED", "TERMINATED"}:
            raise WatchdogError("RunPod REST v1 metadata desiredStatus is unsupported")
        return str(status)

    def stop(self) -> None:
        status, _body = self._stop_transport(
            self._stop_endpoint,
            self._api_key,
            self._timeout_seconds,
            None,
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
            "maximum_approved_compute_hourly_usd": limits.approved_compute_hourly_usd,
            "maximum_approved_storage_hourly_usd": (limits.maximum_approved_storage_hourly_usd),
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


def _try_write_state(path: str | Path, payload: Mapping[str, Any]) -> OSError | None:
    """Persist stop-path evidence when possible without suppressing a stop attempt."""

    try:
        write_json(path, payload)
    except OSError as exc:
        return exc
    return None


def _stop_until_confirmed(
    *,
    client: RunpodStopClient,
    state_path: str | Path,
    limits: WatchdogLimits,
    armed_at: datetime,
    metadata: PodMetadata | None,
    derived: DerivedDeadline | None,
    stop_reason: str,
    stop_attempts: int | None,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    last_error: str | None = None
    attempt = 0
    while stop_attempts is None or attempt < stop_attempts:
        attempt += 1
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
            state_error = _try_write_state(state_path, completed)
            if state_error is not None:
                raise WatchdogError(
                    "Pod stop was confirmed but watchdog state persistence failed"
                ) from state_error
            return completed
        try:
            client.stop()
        except WatchdogError as exc:
            last_error = str(exc)
        else:
            last_error = None
        _try_write_state(
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
        if stop_attempts is None or attempt < stop_attempts:
            sleep(5)
    assert stop_attempts is not None  # finite mode exists only for deterministic tests
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
        state_error = _try_write_state(state_path, completed)
        if state_error is not None:
            raise WatchdogError(
                "Pod stop was confirmed but watchdog state persistence failed"
            ) from state_error
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
    _try_write_state(state_path, failed)
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
    stop_attempts: int | None = None,
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
    if poll_seconds <= 0 or poll_seconds > 60 or (
        stop_attempts is not None and stop_attempts <= 0
    ):
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
        # RunPod's live Pod rate represents compute while
        # volume/container storage is billed separately. Add the frozen storage
        # allowance even if a future response includes it; double-counting the
        # small storage component is deliberately conservative.
        calculation_rate = (
            metadata.effective_hourly_usd + limits.maximum_approved_storage_hourly_usd
        )
        derived = derive_deadline(metadata, limits, calculation_hourly_usd=calculation_rate)
    except WatchdogError as exc:
        _try_write_state(
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

    armed_state_error = _try_write_state(
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
    if armed_state_error is not None:
        _stop_until_confirmed(
            client=client,
            state_path=state_path,
            limits=limits,
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            stop_reason="state_persistence_failed",
            stop_attempts=stop_attempts,
            sleep=sleep,
            now=now,
        )
        raise WatchdogError("watchdog could not persist its armed state") from armed_state_error
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
                state_error = _try_write_state(state_path, terminal)
                if state_error is not None:
                    raise WatchdogError(
                        "Pod stopped externally but watchdog state persistence failed"
                    ) from state_error
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
                raise WatchdogError(
                    "RunPod REST v1 execution identity changed after watchdog arming"
                )
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
            _try_write_state(
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
        state_error = _try_write_state(
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
        if state_error is not None:
            stop_reason = "state_persistence_failed"
            break

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


def wait_for_rearm_then_run_watchdog(
    *,
    lifecycle_state_path: str | Path,
    expected_session_hash: str,
    expected_phase: str,
    pod_id: str,
    expected_gpu_family: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: tuple[str, ...],
    allowed_cuda_versions: tuple[str, ...],
    expected_container_image: str,
    limits: WatchdogLimits,
    state_path: str | Path,
    acknowledgement_path: str | Path,
    client: RunpodStopClient,
    expected_gpu_count: int = 8,
    stop_request_path: str | Path | None = None,
    poll_seconds: float = 5,
    running_readiness_timeout_seconds: float = 300,
    stop_attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Arm on the host before re-arm, then enforce the new provider start.

    The initial acknowledgement is written while the lifecycle is still
    ``stopped``.  The watcher never trusts an ambient Pod id: it keeps checking
    the same private lifecycle record, requires the fresh authorization before
    accepting RUNNING, and then delegates to the normal lastStartedAt-derived
    deadline monitor.  Thus it survives a re-arm client dying after POST /start.
    """

    bound_id = bind_lifecycle_pod(
        lifecycle_state_path=lifecycle_state_path,
        expected_session_hash=expected_session_hash,
        expected_phase=expected_phase,
        ambient_pod_id=pod_id,
        waiting_for_rearm=True,
    )
    if client.pod_id != bound_id:
        raise ValueError("stop client targets a different lifecycle-bound Pod")
    state_destination = Path(state_path).resolve()
    acknowledgement_destination = Path(acknowledgement_path).resolve()
    if (
        acknowledgement_destination.name != HOST_REARM_ACK_FILENAME
        or acknowledgement_destination.parent != state_destination.parent
    ):
        raise ValueError(
            "host re-arm acknowledgement must use its canonical name beside watchdog state"
        )
    if (
        poll_seconds <= 0
        or poll_seconds > 5
        or not math.isfinite(running_readiness_timeout_seconds)
        or running_readiness_timeout_seconds <= 0
        or running_readiness_timeout_seconds > 600
    ):
        raise ValueError("start polling/readiness limits are invalid")

    waiting_at = now().astimezone(UTC)
    readiness_deadline = monotonic() + running_readiness_timeout_seconds
    waiting_state = _state(
        pod_id=bound_id,
        limits=limits,
        status="waiting_for_start",
        armed_at=waiting_at,
        metadata=None,
        derived=None,
        now=waiting_at,
    )
    write_json(state_path, waiting_state)
    acknowledged = False
    allowed_transition_operations = {
        "rearm_intent",
        "rearm_patched",
        "rearm_start_intent",
        "rearm_start_requested",
        "rearm_start_pending",
        "rearm_timeout",
        "rearmed",
        "rearm_verification_failed",
        "rearm_failed_terminal",
    }

    while True:
        current_time = now().astimezone(UTC)
        if monotonic() >= readiness_deadline:
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="rearm_start_or_readiness_timeout",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise WatchdogError("RunPod re-arm start/readiness observation timed out")
        if stop_request_path is not None and Path(stop_request_path).exists():
            return _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="external_stop_request",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )

        heartbeat_error = _try_write_state(
            state_path,
            _state(
                pod_id=bound_id,
                limits=limits,
                status="waiting_for_start",
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                now=current_time,
            ),
        )
        if heartbeat_error is not None:
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="host_heartbeat_persistence_failed",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise WatchdogError("host re-arm watchdog heartbeat persistence failed")

        try:
            binding, state_record = _load_lifecycle_binding(lifecycle_state_path)
        except WatchdogError as exc:
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="lifecycle_read_failed_during_rearm",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise exc
        if not hmac.compare_digest(binding.pod_id, bound_id):
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="lifecycle_pod_identity_changed_during_rearm",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise WatchdogError("private lifecycle Pod identity changed while awaiting re-arm")
        pre_transition = binding.operation == "stopped" and binding.pod_status == "EXITED"
        transition_bound = (
            binding.operation in allowed_transition_operations
            and binding.session_hash == expected_session_hash
            and binding.phase == expected_phase
        )
        if not pre_transition and not transition_bound:
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="lifecycle_became_unsafe",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise WatchdogError("private lifecycle authorization became unsafe during re-arm")

        try:
            provider_status = client.desired_status()
        except WatchdogError:
            sleep(poll_seconds)
            continue
        if provider_status in _TERMINAL_STATUSES:
            if provider_status == "EXITED":
                if pre_transition and not acknowledged:
                    lifecycle_hash = state_record.get("record_hash")
                    if (
                        not isinstance(lifecycle_hash, str)
                        or _NAMESPACED_HASH_RE.fullmatch(lifecycle_hash) is None
                    ):
                        raise WatchdogError(
                            "private stopped lifecycle hash is unavailable for host acknowledgement"
                        )
                    _write_host_rearm_ack(
                        acknowledgement_destination,
                        _host_ack_payload(
                            expected_session_hash=expected_session_hash,
                            expected_phase=expected_phase,
                            lifecycle_before_hash=lifecycle_hash,
                            pod_id=bound_id,
                            watcher_pid=os.getpid(),
                            acknowledged_at=current_time,
                        ),
                    )
                    acknowledged = True
                sleep(poll_seconds)
                continue
            terminal = _state(
                pod_id=bound_id,
                limits=limits,
                status="terminated_externally",
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                now=current_time,
                stop_reason="stopped_outside_watchdog",
            )
            write_json(state_path, terminal)
            return terminal
        if pre_transition:
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="provider_started_before_fresh_authorization",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise WatchdogError("provider Pod started before fresh lifecycle authorization")

        try:
            candidate = client.metadata(observed_at=current_time)
            validate_live_metadata(
                candidate,
                expected_gpu_count=expected_gpu_count,
                expected_gpu_family=expected_gpu_family,
                expected_provider_gpu_id=expected_provider_gpu_id,
                allowed_data_center_ids=allowed_data_center_ids,
                allowed_cuda_versions=allowed_cuda_versions,
                expected_container_image=expected_container_image,
                limits=limits,
            )
        except WatchdogError as exc:
            if monotonic() < readiness_deadline:
                sleep(poll_seconds)
                continue
            _stop_until_confirmed(
                client=client,
                state_path=state_path,
                limits=limits,
                armed_at=waiting_at,
                metadata=None,
                derived=None,
                stop_reason="start_readiness_verification_failed",
                stop_attempts=stop_attempts,
                sleep=sleep,
                now=now,
            )
            raise exc

        return run_watchdog(
            pod_id=bound_id,
            expected_gpu_family=expected_gpu_family,
            expected_provider_gpu_id=expected_provider_gpu_id,
            allowed_data_center_ids=allowed_data_center_ids,
            allowed_cuda_versions=allowed_cuda_versions,
            expected_container_image=expected_container_image,
            limits=limits,
            state_path=state_path,
            client=client,
            expected_gpu_count=expected_gpu_count,
            stop_request_path=stop_request_path,
            poll_seconds=poll_seconds,
            stop_attempts=stop_attempts,
            sleep=sleep,
            now=now,
            monotonic=monotonic,
        )


__all__ = [
    "PROVIDER_API",
    "RUNPOD_API_BASE",
    "RUNPOD_POD_LOOKUP_DOC",
    "RUNPOD_POD_STOP_DOC",
    "RUNPOD_REST_BASE",
    "WATCHDOG_VERSION",
    "DerivedDeadline",
    "LifecyclePodBinding",
    "PodMetadata",
    "RunpodStopClient",
    "WatchdogError",
    "WatchdogLimits",
    "bind_lifecycle_pod",
    "derive_deadline",
    "normalize_gpu_family",
    "parse_pod_metadata",
    "run_watchdog",
    "validate_live_metadata",
    "wait_for_rearm_then_run_watchdog",
]
