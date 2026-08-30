"""Host-owned stop control for one authenticated RunPod session.

The host re-arm watcher is deliberately independent of the provider checkout.
This module creates its canonical stop request and authenticates the watcher's
local ``stopped_confirmed`` record without reading any remote filesystem.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_forensics.gpu_budget import (
    GpuPhaseBudgetReservation,
    load_gpu_phase_budget_reservation,
)
from model_forensics.io import stable_hash
from model_forensics.runpod_lifecycle_state import (
    LifecycleStateAuthorization,
    authorization_from_state,
    load_lifecycle_state,
)
from model_forensics.runpod_session_path import canonical_host_session_directory
from model_forensics.runpod_watchdog import (
    HOST_REARM_ACK_FILENAME,
    HOST_REARM_ACK_PROTOCOL,
    WATCHDOG_VERSION,
    _host_process_identity_hash,
)

HOST_WATCHDOG_FILENAME = "host_rearm_watchdog.json"
HOST_STOP_REQUEST_FILENAME = "runpod_stop.request"
_MAXIMUM_STATE_BYTES = 2 * 1024 * 1024
_STATE_KEYS = {
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
_LIMIT_KEYS = {
    "gpu_hard_stop_usd",
    "global_safe_budget_usd",
    "safe_budget_usd",
    "safety_margin_fraction",
    "maximum_runtime_hours",
    "maximum_approved_hourly_total_usd",
    "maximum_approved_compute_hourly_usd",
    "maximum_approved_storage_hourly_usd",
    "prior_committed_gpu_usd",
}
_ACK_KEYS = {
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
_ARMED_HEARTBEAT_MAX_AGE_SECONDS = 20.0


class RunpodHostControlError(RuntimeError):
    """The local host watcher cannot authenticate this stop operation."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunpodHostControlError("host watchdog contains a duplicate JSON key")
        result[key] = value
    return result


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _owned_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RunpodHostControlError(f"{label} is missing or unsafe")
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid():
        raise RunpodHostControlError(f"{label} ownership is unsafe")
    return path


def _owned_regular(path: Path, *, label: str, require_empty: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RunpodHostControlError(f"{label} is missing or unsafe")
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.getuid()
        or details.st_nlink != 1
        or (require_empty and details.st_size != 0)
    ):
        raise RunpodHostControlError(f"{label} identity is unsafe")
    return path


def _read_watchdog(path: Path) -> dict[str, Any]:
    source = _owned_regular(path, label="host watchdog state")
    before = source.lstat()
    if before.st_size > _MAXIMUM_STATE_BYTES:
        raise RunpodHostControlError("host watchdog state exceeds the safe size limit")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise RunpodHostControlError("host watchdog changed before read")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAXIMUM_STATE_BYTES:
                raise RunpodHostControlError("host watchdog state exceeds the safe size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _file_identity(after) != _file_identity(opened)
        or _file_identity(source.lstat()) != _file_identity(opened)
        or size != opened.st_size
    ):
        raise RunpodHostControlError("host watchdog changed during read")
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodHostControlError("host watchdog state is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _STATE_KEYS:
        raise RunpodHostControlError("host watchdog state schema is unsupported")
    return payload


def _read_acknowledgement(path: Path) -> dict[str, Any]:
    source = _owned_regular(path, label="host re-arm acknowledgement")
    before = source.lstat()
    if before.st_size > _MAXIMUM_STATE_BYTES:
        raise RunpodHostControlError("host acknowledgement exceeds the safe size limit")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise RunpodHostControlError("host acknowledgement changed before read")
        raw = os.read(descriptor, _MAXIMUM_STATE_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > _MAXIMUM_STATE_BYTES
        or _file_identity(after) != _file_identity(opened)
        or _file_identity(source.lstat()) != _file_identity(opened)
        or len(raw) != opened.st_size
    ):
        raise RunpodHostControlError("host acknowledgement changed during read")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodHostControlError("host acknowledgement is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _ACK_KEYS:
        raise RunpodHostControlError("host acknowledgement schema is unsupported")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if payload.get("record_hash") != stable_hash(unsigned):
        raise RunpodHostControlError("host acknowledgement hash mismatch")
    return payload


def _same_number(observed: Any, expected: float, *, label: str) -> None:
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(observed))
        or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise RunpodHostControlError(f"host watchdog {label} binding mismatch")


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunpodHostControlError(f"host watchdog {label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodHostControlError(f"host watchdog {label} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunpodHostControlError(f"host watchdog {label} timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _authenticated_session(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
) -> tuple[
    Path,
    Path,
    GpuPhaseBudgetReservation,
    dict[str, Any],
    LifecycleStateAuthorization,
]:
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise RunpodHostControlError("project root is not a directory")
    try:
        session = canonical_host_session_directory(
            project_root=root,
            phase=phase,
            reservation_path=reservation_path,
        )
        reservation = load_gpu_phase_budget_reservation(reservation_path)
        lifecycle = load_lifecycle_state(root / ".runpod" / "pod_lifecycle.json")
        authorization = authorization_from_state(lifecycle)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunpodHostControlError(
            "host session reservation or lifecycle is unauthenticated"
        ) from exc
    if (
        reservation.phase != phase
        or authorization.phase != phase
        or authorization.session_hash != reservation.session_hash
        or authorization.reservation_id != reservation.reservation_id
        or authorization.reservation_record_hash != reservation.manifest()["record_hash"]
    ):
        raise RunpodHostControlError("host lifecycle and canonical reservation bindings disagree")
    _owned_directory(root / ".runpod", label="private RunPod root")
    _owned_directory(root / ".runpod" / "sessions", label="private sessions root")
    _owned_directory(session, label="canonical host session")
    return root, session, reservation, lifecycle, authorization


def _validate_state_binding(
    payload: Mapping[str, Any],
    *,
    lifecycle: Mapping[str, Any],
    reservation: GpuPhaseBudgetReservation,
    expected_status: str,
) -> None:
    pod = lifecycle.get("pod")
    if not isinstance(pod, Mapping) or not isinstance(pod.get("id"), str):
        raise RunpodHostControlError("host lifecycle Pod binding is missing")
    if (
        payload.get("schema_version") != 2
        or payload.get("watchdog_version") != WATCHDOG_VERSION
        or payload.get("pod_id") != pod["id"]
        or payload.get("status") != expected_status
        or payload.get("action") != "stop_only_preserve_volume"
        or payload.get("deletion") != "manual_after_verified_sync"
    ):
        raise RunpodHostControlError("host watchdog state binding mismatch")
    limits = payload.get("limits")
    if not isinstance(limits, Mapping) or set(limits) != _LIMIT_KEYS:
        raise RunpodHostControlError("host watchdog limits schema is unsupported")
    expected_limits = {
        "gpu_hard_stop_usd": reservation.global_gpu_hard_stop_usd,
        "global_safe_budget_usd": reservation.safety_adjusted_gpu_ceiling_usd,
        "safe_budget_usd": reservation.remaining_safe_gpu_before_phase_usd,
        "safety_margin_fraction": reservation.safety_margin_fraction,
        "maximum_runtime_hours": reservation.maximum_safe_runtime_hours,
        "maximum_approved_hourly_total_usd": reservation.live_hourly_total_usd,
        "prior_committed_gpu_usd": reservation.prior_committed_gpu_usd,
    }
    for key, expected in expected_limits.items():
        _same_number(limits.get(key), expected, label=key)
    compute_rate = limits.get("maximum_approved_compute_hourly_usd")
    storage_rate = limits.get("maximum_approved_storage_hourly_usd")
    if (
        isinstance(compute_rate, bool)
        or not isinstance(compute_rate, (int, float))
        or not math.isfinite(float(compute_rate))
        or float(compute_rate) <= 0
        or isinstance(storage_rate, bool)
        or not isinstance(storage_rate, (int, float))
        or not math.isfinite(float(storage_rate))
        or float(storage_rate) < 0
        or not math.isclose(
            float(compute_rate) + float(storage_rate),
            reservation.live_hourly_total_usd,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RunpodHostControlError("host watchdog compute/storage rate binding mismatch")
    metadata = payload.get("live_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("provider_api") != "rest-v1"
        or metadata.get("pod_id") != pod["id"]
    ):
        raise RunpodHostControlError("host watchdog provider binding is incomplete")


def request_host_stop(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
) -> Path:
    """Durably request stop from the independently running host watcher."""

    root, session, reservation, lifecycle, _authorization = _authenticated_session(
        project_root=project_root,
        phase=phase,
        reservation_path=reservation_path,
    )
    pod = lifecycle.get("pod")
    if (
        lifecycle.get("operation") != "rearmed"
        or not isinstance(pod, Mapping)
        or pod.get("status") != "RUNNING"
    ):
        raise RunpodHostControlError(
            "host stop request requires the authenticated running re-arm lifecycle"
        )
    state = _read_watchdog(session / HOST_WATCHDOG_FILENAME)
    _validate_state_binding(
        state,
        lifecycle=lifecycle,
        reservation=reservation,
        expected_status="armed",
    )
    if state.get("stop_reason") is not None or state.get("error") is not None:
        raise RunpodHostControlError("host watchdog is not cleanly armed")
    current = datetime.now(UTC)
    updated = _timestamp(state.get("updated_at"), label="update")
    age = (current - updated).total_seconds()
    if age < -5 or age > _ARMED_HEARTBEAT_MAX_AGE_SECONDS:
        raise RunpodHostControlError("host watchdog heartbeat is stale or future-dated")

    acknowledgement = _read_acknowledgement(session / HOST_REARM_ACK_FILENAME)
    pod_id = str(pod["id"])
    if (
        acknowledgement.get("schema_version") != 1
        or acknowledgement.get("protocol_version") != HOST_REARM_ACK_PROTOCOL
        or acknowledgement.get("status") != "armed_and_provider_exited_verified"
        or acknowledgement.get("expected_session_hash") != reservation.session_hash
        or acknowledgement.get("expected_phase") != phase
        or acknowledgement.get("pod_id_hash") != stable_hash({"runpod_pod_id": pod_id})
    ):
        raise RunpodHostControlError("host acknowledgement binding mismatch")
    pid = acknowledgement.get("watcher_pid")
    expected_identity = acknowledgement.get("watcher_process_identity_hash")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or not isinstance(expected_identity, str)
    ):
        raise RunpodHostControlError("host watcher process identity is malformed")
    try:
        live_identity = _host_process_identity_hash(pid)
    except RuntimeError as exc:
        raise RunpodHostControlError("host watcher process is not live") from exc
    if not hmac.compare_digest(expected_identity, live_identity):
        raise RunpodHostControlError("host watcher process identity changed")

    request = session / HOST_STOP_REQUEST_FILENAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(request, flags, 0o600)
    except FileExistsError:
        _owned_regular(request, label="host stop request", require_empty=True)
    else:
        try:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(session, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    if not request.is_relative_to(root / ".runpod" / "sessions"):
        raise RunpodHostControlError("host stop request escaped private session state")
    return request


def validate_host_stop_confirmation(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
    watchdog_path: str | Path,
    stop_request_path: str | Path,
) -> dict[str, Any]:
    """Authenticate the local watcher's provider-confirmed normal stop."""

    _root, session, reservation, lifecycle, _authorization = _authenticated_session(
        project_root=project_root,
        phase=phase,
        reservation_path=reservation_path,
    )
    expected_watchdog = session / HOST_WATCHDOG_FILENAME
    expected_request = session / HOST_STOP_REQUEST_FILENAME
    supplied_watchdog = Path(os.path.abspath(watchdog_path))
    supplied_request = Path(os.path.abspath(stop_request_path))
    if supplied_watchdog != expected_watchdog or supplied_request != expected_request:
        raise RunpodHostControlError("host stop controls are not canonical for this session")
    _owned_regular(expected_request, label="host stop request", require_empty=True)
    state = _read_watchdog(expected_watchdog)
    _validate_state_binding(
        state,
        lifecycle=lifecycle,
        reservation=reservation,
        expected_status="stopped_confirmed",
    )
    if state.get("stop_reason") != "external_stop_request":
        raise RunpodHostControlError(
            "host watchdog did not confirm the canonical local stop request"
        )
    return state


__all__ = [
    "HOST_STOP_REQUEST_FILENAME",
    "HOST_WATCHDOG_FILENAME",
    "RunpodHostControlError",
    "request_host_stop",
    "validate_host_stop_confirmation",
]
