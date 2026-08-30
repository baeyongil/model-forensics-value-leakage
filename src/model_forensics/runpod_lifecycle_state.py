"""Zero-dependency reader for the private RunPod lifecycle binding.

Only the watchdog imports this module on a fresh provider image.  Lifecycle
mutations remain in :mod:`model_forensics.runpod_lifecycle`; this reader mirrors
its owner/link/location/schema/content-hash checks without importing Pydantic or
PyYAML before the watchdog is armed.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash
from model_forensics.runpod_contract import (
    GPU_COMMAND_PHASES,
    LIFECYCLE_PROTOCOL,
    LIFECYCLE_STATE_FILENAME,
)

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXISTING_POD_ID_HASH_RE = re.compile(r"runpod-pod-id-sha256:[0-9a-f]{64}\Z")
_AUTHORIZATION_KEYS = {
    "acknowledged_existing_pod_id_hashes",
    "approval_hash",
    "approved_phase_maximum_usd",
    "approved_runtime_hours",
    "bindings_hash",
    "gpu_lock_hash",
    "immutable_spec_hash",
    "launch_spec_hash",
    "live_hourly_total_usd",
    "phase",
    "quote_hash",
    "reservation_id",
    "reservation_record_hash",
    "session_hash",
}
_AUTHORIZATION_HASH_FIELDS = (
    "reservation_id",
    "reservation_record_hash",
    "session_hash",
    "approval_hash",
    "bindings_hash",
    "gpu_lock_hash",
    "quote_hash",
    "immutable_spec_hash",
    "launch_spec_hash",
)
_MAXIMUM_LIFECYCLE_STATE_BYTES = 2 * 1024 * 1024


class RunpodLifecycleStateError(RuntimeError):
    """The private lifecycle state cannot authenticate one watchdog target."""


@dataclass(frozen=True, slots=True)
class LifecycleStateAuthorization:
    phase: str
    reservation_id: str
    reservation_record_hash: str
    session_hash: str
    approval_hash: str
    immutable_spec_hash: str
    approved_runtime_hours: float
    approved_phase_maximum_usd: float
    live_hourly_total_usd: float


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunpodLifecycleStateError("private lifecycle contains a duplicate JSON key")
        result[key] = value
    return result


def _secure_state_file(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise RunpodLifecycleStateError("lifecycle state path must be absolute")
    if path.name != LIFECYCLE_STATE_FILENAME or path.parent.name != ".runpod":
        raise RunpodLifecycleStateError("lifecycle state must remain directly under .runpod")
    private = path.parent
    if os.path.lexists(private) and private.is_symlink():
        raise RunpodLifecycleStateError("private .runpod directory must not be a symlink")
    try:
        private_details = private.lstat()
    except OSError as exc:
        raise RunpodLifecycleStateError("private .runpod directory is missing") from exc
    if not stat.S_ISDIR(private_details.st_mode) or private_details.st_uid != os.getuid():
        raise RunpodLifecycleStateError("private .runpod directory is unsafe")
    if path.is_symlink() or not path.is_file():
        raise RunpodLifecycleStateError("lifecycle state must be a regular non-symlink file")
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
    ):
        raise RunpodLifecycleStateError("lifecycle state owner or link count is unsafe")
    if details.st_size > _MAXIMUM_LIFECYCLE_STATE_BYTES:
        raise RunpodLifecycleStateError("lifecycle state exceeds the safe size limit")
    return details


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def load_lifecycle_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    before = _secure_state_file(source)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                _file_identity(opened) != _file_identity(before)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.getuid()
                or opened.st_nlink != 1
                or opened.st_size > _MAXIMUM_LIFECYCLE_STATE_BYTES
            ):
                raise RunpodLifecycleStateError(
                    "private lifecycle state changed before authenticated read"
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, _MAXIMUM_LIFECYCLE_STATE_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _MAXIMUM_LIFECYCLE_STATE_BYTES:
                    raise RunpodLifecycleStateError(
                        "lifecycle state exceeds the safe size limit"
                    )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = source.lstat()
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
            or size != opened.st_size
        ):
            raise RunpodLifecycleStateError(
                "private lifecycle state changed during authenticated read"
            )
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except RunpodLifecycleStateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodLifecycleStateError("private lifecycle state is unreadable") from exc
    expected_keys = {
        "schema_version",
        "protocol_version",
        "operation",
        "updated_at",
        "immutable_spec",
        "current_authorization",
        "authorization_history",
        "pod",
        "record_hash",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RunpodLifecycleStateError("private lifecycle state has an unexpected schema")
    if value.get("schema_version") != 1 or value.get("protocol_version") != LIFECYCLE_PROTOCOL:
        raise RunpodLifecycleStateError("private lifecycle state protocol is unsupported")
    unsigned = {key: item for key, item in value.items() if key != "record_hash"}
    if value.get("record_hash") != stable_hash(unsigned):
        raise RunpodLifecycleStateError("private lifecycle state content hash mismatch")
    _validate_authorization_history(value)
    return value


def _authorization_from_manifest(
    value: Any,
    *,
    immutable_spec: Mapping[str, Any],
    label: str,
) -> LifecycleStateAuthorization:
    if not isinstance(value, Mapping) or set(value) != _AUTHORIZATION_KEYS:
        raise RunpodLifecycleStateError(
            f"private lifecycle {label} authorization has an unexpected schema"
        )
    if any(
        not isinstance(value.get(field), str)
        or _HASH_RE.fullmatch(str(value[field])) is None
        for field in _AUTHORIZATION_HASH_FIELDS
    ):
        raise RunpodLifecycleStateError(
            f"private lifecycle {label} authorization hash is malformed"
        )
    if value["immutable_spec_hash"] != stable_hash(dict(immutable_spec)):
        raise RunpodLifecycleStateError(
            f"private lifecycle {label} immutable launch specification drifted"
        )
    phase = value.get("phase")
    if not isinstance(phase, str) or phase not in GPU_COMMAND_PHASES:
        raise RunpodLifecycleStateError(f"private lifecycle {label} phase is malformed")
    acknowledged = value.get("acknowledged_existing_pod_id_hashes")
    if not isinstance(acknowledged, list) or not all(
        isinstance(item, str)
        and _EXISTING_POD_ID_HASH_RE.fullmatch(item) is not None
        for item in acknowledged
    ):
        raise RunpodLifecycleStateError(
            f"private lifecycle {label} Pod acknowledgement list is malformed"
        )
    if acknowledged != sorted(set(acknowledged)):
        raise RunpodLifecycleStateError(
            f"private lifecycle {label} Pod acknowledgement list is not canonical"
        )
    numeric: dict[str, float] = {}
    for field in (
        "approved_runtime_hours",
        "approved_phase_maximum_usd",
        "live_hourly_total_usd",
    ):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RunpodLifecycleStateError(
                f"private lifecycle {label} authorization cost is malformed"
            )
        parsed = float(raw)
        if not math.isfinite(parsed) or parsed <= 0:
            raise RunpodLifecycleStateError(
                f"private lifecycle {label} authorization cost is malformed"
            )
        numeric[field] = parsed
    return LifecycleStateAuthorization(
        phase=phase,
        reservation_id=str(value["reservation_id"]),
        reservation_record_hash=str(value["reservation_record_hash"]),
        session_hash=str(value["session_hash"]),
        approval_hash=str(value["approval_hash"]),
        immutable_spec_hash=str(value["immutable_spec_hash"]),
        approved_runtime_hours=numeric["approved_runtime_hours"],
        approved_phase_maximum_usd=numeric["approved_phase_maximum_usd"],
        live_hourly_total_usd=numeric["live_hourly_total_usd"],
    )


def _validate_authorization_history(state: Mapping[str, Any]) -> None:
    spec = state.get("immutable_spec")
    current = state.get("current_authorization")
    history = state.get("authorization_history")
    if not isinstance(spec, Mapping) or not isinstance(history, list):
        raise RunpodLifecycleStateError("private lifecycle authorization history is malformed")
    authorizations = [
        _authorization_from_manifest(
            item,
            immutable_spec=spec,
            label=f"historical[{index}]",
        )
        for index, item in enumerate(history)
    ]
    authorizations.append(
        _authorization_from_manifest(
            current,
            immutable_spec=spec,
            label="current",
        )
    )
    sessions = [item.session_hash for item in authorizations]
    reservations = [item.reservation_id for item in authorizations]
    approval_phases = [(item.approval_hash, item.phase) for item in authorizations]
    if (
        len(sessions) != len(set(sessions))
        or len(reservations) != len(set(reservations))
        or len(approval_phases) != len(set(approval_phases))
    ):
        raise RunpodLifecycleStateError(
            "private lifecycle authorization history reuses a session, reservation, "
            "or approval-phase pair"
        )


def authorization_from_state(state: Mapping[str, Any]) -> LifecycleStateAuthorization:
    spec = state.get("immutable_spec")
    if not isinstance(spec, Mapping):
        raise RunpodLifecycleStateError("private lifecycle authorization is malformed")
    return _authorization_from_manifest(
        state.get("current_authorization"),
        immutable_spec=spec,
        label="current",
    )


__all__ = [
    "LifecycleStateAuthorization",
    "RunpodLifecycleStateError",
    "authorization_from_state",
    "load_lifecycle_state",
]
