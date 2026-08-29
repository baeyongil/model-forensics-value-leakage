"""Private lifecycle records for sequential RunPod GPU phases."""

from __future__ import annotations

import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_forensics.budget import CostLedger
from model_forensics.gpu_budget import (
    GPU_PHASE_BUDGET_PROTOCOL,
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    GpuPhaseBudgetReservation,
    validate_existing_gpu_phase_reservation,
)
from model_forensics.io import read_json, stable_hash

GPU_BUDGET_BOOTSTRAP_FILENAME = "gpu_budget_bootstrap.json"
WATCHDOG_STATE_FILENAME = "runpod_watchdog.json"
SETTLEMENT_FILENAME = "settlement.json"
GPU_PREFLIGHT_FILENAME = "gpu_preflight.json"
WATCHDOG_PID_FILENAME = "runpod_watchdog.pid"
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACED_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class RunpodSessionError(RuntimeError):
    """A private GPU session lifecycle is incomplete or inconsistent."""


def _require_regular_private_record(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunpodSessionError(f"private session record is missing or unsafe: {path}")


def _finite_number(value: Any, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RunpodSessionError(f"active session {field} must be finite numeric")
    parsed = float(value)
    if parsed < 0 or (not allow_zero and parsed == 0):
        raise RunpodSessionError(f"active session {field} must be nonnegative")
    return parsed


def _require_close(value: Any, expected: float, *, field: str) -> float:
    parsed = _finite_number(value, field=field, allow_zero=True)
    if abs(parsed - expected) > 1e-6:
        raise RunpodSessionError(f"active session {field} mismatch")
    return parsed


def _authenticated_record(path: Path, *, protocol: str) -> dict[str, Any]:
    _require_regular_private_record(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError(f"cannot read authenticated session record: {path}") from exc
    if not isinstance(payload, dict):
        raise RunpodSessionError(f"session record must be a JSON object: {path}")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != protocol:
        raise RunpodSessionError(f"session record protocol mismatch: {path}")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(record_hash) is None
        or record_hash != stable_hash(unsigned)
    ):
        raise RunpodSessionError(f"session record content hash mismatch: {path}")
    return payload


def _ledger_entry(ledger: CostLedger, *, reservation_id: str) -> dict[str, Any]:
    document = ledger.document()
    matching = [entry for entry in document["entries"] if entry.get("entry_id") == reservation_id]
    if len(matching) != 1:
        raise RunpodSessionError(
            "canonical ledger does not contain exactly one session reservation"
        )
    return dict(matching[0])


def _validated_bootstrap(path: Path) -> dict[str, Any]:
    payload = _authenticated_record(path, protocol=GPU_PHASE_BUDGET_PROTOCOL)
    if payload.get("passed") is not True:
        raise RunpodSessionError("GPU bootstrap budget gate did not pass")
    for field in ("session_hash", "reservation_id", "reservation_record_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or _NAMESPACED_HASH_RE.fullmatch(value) is None:
            raise RunpodSessionError(f"GPU bootstrap record has invalid {field}")
    return payload


def _validated_watchdog(path: Path) -> dict[str, Any]:
    _require_regular_private_record(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError(f"cannot read prior watchdog state: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise RunpodSessionError("prior watchdog state is malformed")
    if payload.get("watchdog_version") != "runpod-gpu-cost-watchdog-v2":
        raise RunpodSessionError("prior watchdog state version is unsupported")
    if payload.get("status") != "stopped_confirmed":
        raise RunpodSessionError("prior GPU session is not stopped_confirmed")
    return payload


def _validated_settlement(
    path: Path,
    *,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    payload = _authenticated_record(path, protocol=GPU_PHASE_SETTLEMENT_PROTOCOL)
    if payload.get("status") != "settled":
        raise RunpodSessionError("prior GPU session settlement is incomplete")
    for field in ("session_hash", "reservation_id", "reservation_record_hash"):
        if payload.get(field) != bootstrap.get(field):
            raise RunpodSessionError(f"prior settlement {field} disagrees with bootstrap")
    incurred = payload.get("provider_incurred_usd")
    if isinstance(incurred, bool) or not isinstance(incurred, (int, float)) or float(incurred) < 0:
        raise RunpodSessionError("prior settlement incurred cost is invalid")
    return payload


def validate_completed_runpod_sessions(
    *,
    sessions_root: str | Path,
    ledger: CostLedger,
) -> list[dict[str, Any]]:
    """Require every prior private session to be stopped and exactly settled."""

    root = Path(sessions_root)
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise RunpodSessionError("RunPod sessions root must be a real directory")
    summaries: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.is_symlink():
            raise RunpodSessionError(f"unexpected private session entry: {directory}")
        if _RAW_HASH_RE.fullmatch(directory.name) is None:
            raise RunpodSessionError(f"private session directory name is invalid: {directory}")
        bootstrap = _validated_bootstrap(directory / GPU_BUDGET_BOOTSTRAP_FILENAME)
        if bootstrap["session_hash"] != f"sha256:{directory.name}":
            raise RunpodSessionError("private session directory disagrees with session hash")
        watchdog = _validated_watchdog(directory / WATCHDOG_STATE_FILENAME)
        settlement = _validated_settlement(
            directory / SETTLEMENT_FILENAME,
            bootstrap=bootstrap,
        )
        if settlement.get("watchdog_state_hash") != stable_hash(watchdog):
            raise RunpodSessionError("prior settlement watchdog state hash mismatch")
        entry = _ledger_entry(ledger, reservation_id=str(bootstrap["reservation_id"]))
        if entry.get("kind") != "gpu" or entry.get("status") != "incurred":
            raise RunpodSessionError("prior GPU reservation is not settled in canonical ledger")
        if abs(float(entry.get("amount_usd")) - float(settlement["provider_incurred_usd"])) > 1e-6:
            raise RunpodSessionError("prior GPU settlement disagrees with canonical ledger")
        summaries.append(
            {
                "session_hash": bootstrap["session_hash"],
                "reservation_id": bootstrap["reservation_id"],
                "settlement_record_hash": settlement["record_hash"],
                "status": "stopped_confirmed_and_settled",
            }
        )
    return summaries


def prepare_runpod_session_directory(
    *,
    sessions_root: str | Path,
    pending_bootstrap_path: str | Path,
    ledger: CostLedger,
) -> Path:
    """Atomically claim a new private session after all prior phases completed."""

    root = Path(sessions_root)
    pending = Path(pending_bootstrap_path)
    bootstrap = _validated_bootstrap(pending)
    validate_completed_runpod_sessions(sessions_root=root, ledger=ledger)

    session_hash = str(bootstrap["session_hash"])
    session_digest = session_hash.removeprefix("sha256:")
    if _RAW_HASH_RE.fullmatch(session_digest) is None:
        raise RunpodSessionError("current session hash is invalid")
    entry = _ledger_entry(ledger, reservation_id=str(bootstrap["reservation_id"]))
    if entry.get("kind") != "gpu" or entry.get("status") != "estimated":
        raise RunpodSessionError("current GPU reservation is not active in canonical ledger")
    active_gpu_entries = [
        item
        for item in ledger.document()["entries"]
        if item.get("kind") == "gpu" and item.get("status") == "estimated"
    ]
    if (
        len(active_gpu_entries) != 1
        or active_gpu_entries[0].get("entry_id") != bootstrap["reservation_id"]
    ):
        raise RunpodSessionError("current reservation is not the sole active GPU commitment")

    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:  # pragma: no cover
        pass
    target = root / session_digest
    try:
        target.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise RunpodSessionError("GPU phase/session directory has already been claimed") from exc
    destination = target / GPU_BUDGET_BOOTSTRAP_FILENAME
    try:
        os.replace(pending, destination)
        destination.chmod(0o600)
    except BaseException:
        # Leave the claimed but incomplete directory in place. A future launch
        # then fails closed instead of silently reusing this session identity.
        raise
    return target


def validate_active_runpod_session(
    *,
    session_directory: str | Path,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    phase: str,
    session_id: str,
    now: datetime | None = None,
    maximum_watchdog_age_seconds: float = 90,
) -> dict[str, Any]:
    """Authenticate the live private session immediately before GPU backend use."""

    directory = Path(session_directory)
    if not directory.is_dir() or directory.is_symlink():
        raise RunpodSessionError("active RunPod session directory is missing or unsafe")
    if directory.name != reservation.session_hash.removeprefix("sha256:"):
        raise RunpodSessionError("active session directory disagrees with reservation")
    bootstrap = _validated_bootstrap(directory / GPU_BUDGET_BOOTSTRAP_FILENAME)
    if bootstrap.get("phase") != phase:
        raise RunpodSessionError("active session phase disagrees with bootstrap")
    if bootstrap.get("reservation_id") != reservation.reservation_id:
        raise RunpodSessionError("active session reservation disagrees with bootstrap")
    if bootstrap.get("reservation_record_hash") != reservation.manifest()["record_hash"]:
        raise RunpodSessionError("active session receipt hash disagrees with bootstrap")
    validate_existing_gpu_phase_reservation(
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        require_active=True,
    )

    watchdog_path = directory / WATCHDOG_STATE_FILENAME
    _require_regular_private_record(watchdog_path)
    try:
        watchdog = read_json(watchdog_path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("active session watchdog state is unreadable") from exc
    if not isinstance(watchdog, dict) or watchdog.get("schema_version") != 2:
        raise RunpodSessionError("active session watchdog state is malformed")
    if watchdog.get("watchdog_version") != "runpod-gpu-cost-watchdog-v2":
        raise RunpodSessionError("active session watchdog version is unsupported")
    if watchdog.get("status") != "armed":
        raise RunpodSessionError("active session watchdog is not armed")
    if watchdog.get("action") != "stop_only_preserve_volume":
        raise RunpodSessionError("active session watchdog action is unsafe")
    raw_updated = watchdog.get("updated_at")
    if not isinstance(raw_updated, str):
        raise RunpodSessionError("active session watchdog timestamp is missing")
    try:
        updated = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError("active session watchdog timestamp is malformed") from exc
    if updated.tzinfo is None or updated.utcoffset() is None:
        raise RunpodSessionError("active session watchdog timestamp lacks timezone")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age = (current - updated.astimezone(UTC)).total_seconds()
    if age < -300 or age > maximum_watchdog_age_seconds:
        raise RunpodSessionError("active session watchdog state is stale")

    limits = watchdog.get("limits")
    if not isinstance(limits, dict):
        raise RunpodSessionError("active session watchdog limits are missing")
    _require_close(
        limits.get("gpu_hard_stop_usd"),
        reservation.global_gpu_hard_stop_usd,
        field="watchdog GPU hard stop",
    )
    _require_close(
        limits.get("global_safe_budget_usd"),
        reservation.safety_adjusted_gpu_ceiling_usd,
        field="watchdog global safe budget",
    )
    _require_close(
        limits.get("safe_budget_usd"),
        reservation.remaining_safe_gpu_before_phase_usd,
        field="watchdog remaining safe budget",
    )
    _require_close(
        limits.get("safety_margin_fraction"),
        reservation.safety_margin_fraction,
        field="watchdog safety margin",
    )
    _require_close(
        limits.get("maximum_runtime_hours"),
        reservation.maximum_safe_runtime_hours,
        field="watchdog maximum runtime",
    )
    _require_close(
        limits.get("maximum_approved_hourly_total_usd"),
        reservation.live_hourly_total_usd,
        field="watchdog approved hourly total",
    )
    _require_close(
        limits.get("prior_committed_gpu_usd"),
        reservation.prior_committed_gpu_usd,
        field="watchdog prior committed GPU cost",
    )

    deadline = watchdog.get("deadline")
    if not isinstance(deadline, dict):
        raise RunpodSessionError("active session watchdog deadline is missing")
    raw_deadline = deadline.get("effective_deadline")
    if not isinstance(raw_deadline, str):
        raise RunpodSessionError("active session watchdog deadline timestamp is missing")
    try:
        effective_deadline = datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError("active session watchdog deadline timestamp is malformed") from exc
    if effective_deadline.tzinfo is None or effective_deadline.utcoffset() is None:
        raise RunpodSessionError("active session watchdog deadline lacks timezone")
    if effective_deadline.astimezone(UTC) <= current:
        raise RunpodSessionError("active session watchdog deadline has elapsed")
    _finite_number(deadline.get("calculation_hourly_usd"), field="watchdog calculation rate")
    _finite_number(
        deadline.get("incurred_cost_usd"),
        field="watchdog incurred GPU cost",
        allow_zero=True,
    )

    pid_path = directory / WATCHDOG_PID_FILENAME
    _require_regular_private_record(pid_path)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        if pid <= 1:
            raise ValueError
        os.kill(pid, 0)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("active session watchdog process is not alive") from exc

    preflight_path = directory / GPU_PREFLIGHT_FILENAME
    _require_regular_private_record(preflight_path)
    try:
        preflight = read_json(preflight_path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("active session GPU preflight is unreadable") from exc
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != 3
        or preflight.get("passed") is not True
    ):
        raise RunpodSessionError("active session GPU preflight did not pass")
    gate = preflight.get("gpu_budget_reservation")
    if not isinstance(gate, dict):
        raise RunpodSessionError("active session GPU preflight lacks budget binding")
    for field in ("reservation_id", "reservation_record_hash", "session_hash", "phase"):
        if gate.get(field) != bootstrap.get(field):
            raise RunpodSessionError(f"active session GPU preflight {field} mismatch")
    _require_close(
        preflight.get("planned_hours"),
        reservation.maximum_safe_runtime_hours,
        field="GPU preflight planned runtime",
    )
    _require_close(
        preflight.get("prior_committed_gpu_cost_usd"),
        reservation.prior_committed_gpu_usd,
        field="GPU preflight prior committed GPU cost",
    )
    _require_close(
        preflight.get("gpu_budget_usd"),
        reservation.global_gpu_hard_stop_usd,
        field="GPU preflight hard stop",
    )
    price = preflight.get("price")
    if not isinstance(price, dict):
        raise RunpodSessionError("active session GPU preflight price binding is missing")
    _require_close(
        price.get("approved_hourly_total_usd"),
        reservation.live_hourly_total_usd,
        field="GPU preflight approved hourly total",
    )
    preflight_watchdog = preflight.get("watchdog")
    if not isinstance(preflight_watchdog, dict):
        raise RunpodSessionError("active session GPU preflight watchdog binding is missing")
    if preflight_watchdog.get("pid") != pid:
        raise RunpodSessionError("active session GPU preflight watchdog PID mismatch")
    state_path = preflight_watchdog.get("state_path")
    if not isinstance(state_path, str) or Path(state_path).resolve() != watchdog_path.resolve():
        raise RunpodSessionError("active session GPU preflight watchdog path mismatch")
    bound_updated = preflight_watchdog.get("state_updated_at")
    if not isinstance(bound_updated, str):
        raise RunpodSessionError("active session GPU preflight watchdog timestamp is missing")
    try:
        parsed_bound_updated = datetime.fromisoformat(bound_updated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError(
            "active session GPU preflight watchdog timestamp is malformed"
        ) from exc
    if (
        parsed_bound_updated.tzinfo is None
        or parsed_bound_updated.utcoffset() is None
        or parsed_bound_updated.astimezone(UTC) > updated.astimezone(UTC)
    ):
        raise RunpodSessionError("active session GPU preflight watchdog timestamp mismatch")

    payload = {
        "schema_version": 1,
        "protocol_version": "active-runpod-session-v1",
        "phase": phase,
        "session_hash": reservation.session_hash,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "watchdog_updated_at": updated.astimezone(UTC).isoformat(),
        "gpu_preflight_hash": stable_hash(preflight),
        "passed": True,
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "GPU_BUDGET_BOOTSTRAP_FILENAME",
    "GPU_PREFLIGHT_FILENAME",
    "SETTLEMENT_FILENAME",
    "WATCHDOG_PID_FILENAME",
    "WATCHDOG_STATE_FILENAME",
    "RunpodSessionError",
    "prepare_runpod_session_directory",
    "validate_active_runpod_session",
    "validate_completed_runpod_sessions",
]
