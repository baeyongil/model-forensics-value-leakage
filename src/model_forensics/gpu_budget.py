"""Provider-neutral, cumulative GPU phase budget reservations.

The paid-run approval layer decides *which* phase and maximum are approved.
This module performs the separate accounting duty: it atomically subtracts all
prior incurred and unresolved GPU commitments in the canonical cost ledger,
reserves one approved phase maximum, and derives the maximum safe runtime at a
caller-supplied live hourly rate.  It has no provider SDK or network access.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from model_forensics.budget import CostEntry, CostLedger, ReservationConflict
from model_forensics.io import read_json, stable_hash

GPU_PHASE_BUDGET_PROTOCOL = "cumulative-gpu-phase-budget-v1"
_PHASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}\Z")
_USD_QUANTUM = Decimal("0.000001")
GPU_PHASE_SETTLEMENT_PROTOCOL = "cumulative-gpu-phase-settlement-v1"


class GpuBudgetGateError(RuntimeError):
    """A GPU phase cannot be safely reserved under the canonical ledger."""


_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _positive_finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _ceil_usd(value: float) -> float:
    """Round upward to ledger precision so a reservation never understates cost."""

    return float(Decimal(str(value)).quantize(_USD_QUANTUM, rounding=ROUND_CEILING))


def _floor_usd(value: float) -> float:
    """Round a ceiling downward to ledger precision so it is never optimistic."""

    return float(Decimal(str(value)).quantize(_USD_QUANTUM, rounding=ROUND_FLOOR))


def approved_gpu_phase_maximum_usd(
    *,
    gpu_count: int,
    quote_hourly_per_gpu_usd: float,
    running_storage_hourly_usd: float = 0.0,
    approved_runtime_hours: float,
) -> float:
    """Return the upward-rounded maximum implied by the approved GPU quote."""

    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
        raise ValueError("gpu_count must be a positive integer")
    per_gpu = _positive_finite(
        quote_hourly_per_gpu_usd,
        field="quote_hourly_per_gpu_usd",
    )
    if (
        isinstance(running_storage_hourly_usd, bool)
        or not math.isfinite(float(running_storage_hourly_usd))
        or float(running_storage_hourly_usd) < 0
    ):
        raise ValueError("running_storage_hourly_usd must be finite and non-negative")
    runtime = _positive_finite(
        approved_runtime_hours,
        field="approved_runtime_hours",
    )
    return _ceil_usd(
        (gpu_count * per_gpu + float(running_storage_hourly_usd)) * runtime
    )


def write_json_exclusive(path: str | Path, payload: Any) -> Path:
    """Atomically create a JSON artifact and fail if its path was ever claimed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise GpuBudgetGateError(
                f"refusing to overwrite claimed GPU artifact: {destination}"
            ) from exc
        destination.chmod(0o600)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return destination


def _reservation_description(*, phase: str, session_hash: str) -> str:
    return f"GPU phase {phase} session {session_hash}"


@dataclass(frozen=True, slots=True)
class GpuPhaseBudgetReservation:
    """Secret-safe receipt for one atomic, single-use GPU reservation."""

    reservation_id: str
    phase: str
    session_hash: str
    approved_phase_maximum_usd: float
    approved_maximum_runtime_hours: float
    live_hourly_total_usd: float
    safety_margin_fraction: float
    global_gpu_hard_stop_usd: float
    safety_adjusted_gpu_ceiling_usd: float
    prior_incurred_gpu_usd: float
    prior_reserved_gpu_usd: float
    prior_committed_gpu_usd: float
    prior_committed_total_usd: float
    remaining_safe_gpu_before_phase_usd: float
    remaining_total_before_phase_usd: float
    maximum_safe_runtime_hours: float
    committed_gpu_after_reservation_usd: float
    committed_total_after_reservation_usd: float

    @property
    def description(self) -> str:
        return _reservation_description(phase=self.phase, session_hash=self.session_hash)

    def settlement_entry(self, *, incurred_usd: float) -> CostEntry:
        """Build the exact ledger entry used to reconcile this reservation."""

        if (
            isinstance(incurred_usd, bool)
            or not math.isfinite(float(incurred_usd))
            or float(incurred_usd) < 0
        ):
            raise ValueError("incurred GPU cost must be finite and non-negative")
        return CostEntry(
            kind="gpu",
            amount_usd=float(incurred_usd),
            description=self.description,
            status="incurred",
        )

    def watchdog_budget_kwargs(self) -> dict[str, float]:
        """Return the exact cumulative values required by ``WatchdogLimits``."""

        return {
            "gpu_hard_stop_usd": self.global_gpu_hard_stop_usd,
            "maximum_runtime_hours": self.maximum_safe_runtime_hours,
            "safety_margin_fraction": self.safety_margin_fraction,
            "prior_committed_gpu_usd": self.prior_committed_gpu_usd,
        }

    def manifest(self) -> dict[str, Any]:
        """Return a content-addressed receipt that never reveals the session id."""

        payload: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": GPU_PHASE_BUDGET_PROTOCOL,
            **asdict(self),
        }
        payload["record_hash"] = stable_hash(payload)
        return payload


def _expected_reservation_id(*, phase: str, session_hash: str) -> str:
    return stable_hash(
        {
            "protocol": GPU_PHASE_BUDGET_PROTOCOL,
            "phase": phase,
            "session_hash": session_hash,
        }
    )


def _validate_receipt_invariants(reservation: GpuPhaseBudgetReservation) -> None:
    """Reject a self-consistent hash over internally inconsistent accounting."""

    if _PHASE_RE.fullmatch(reservation.phase) is None:
        raise GpuBudgetGateError("GPU reservation phase is invalid")
    if _HASH_RE.fullmatch(reservation.session_hash) is None:
        raise GpuBudgetGateError("GPU reservation session hash is invalid")
    if reservation.reservation_id != _expected_reservation_id(
        phase=reservation.phase,
        session_hash=reservation.session_hash,
    ):
        raise GpuBudgetGateError("GPU reservation identity is inconsistent")
    numeric = {
        field.name: getattr(reservation, field.name)
        for field in fields(GpuPhaseBudgetReservation)
        if field.name not in {"reservation_id", "phase", "session_hash"}
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in numeric.values()
    ):
        raise GpuBudgetGateError("GPU reservation contains a non-finite numeric field")
    positive_fields = {
        "approved_phase_maximum_usd",
        "approved_maximum_runtime_hours",
        "live_hourly_total_usd",
        "safety_margin_fraction",
        "global_gpu_hard_stop_usd",
        "safety_adjusted_gpu_ceiling_usd",
        "remaining_safe_gpu_before_phase_usd",
        "remaining_total_before_phase_usd",
        "maximum_safe_runtime_hours",
        "committed_gpu_after_reservation_usd",
        "committed_total_after_reservation_usd",
    }
    if any(float(numeric[name]) <= 0 for name in positive_fields):
        raise GpuBudgetGateError("GPU reservation contains a non-positive required field")
    if any(float(value) < 0 for name, value in numeric.items() if name not in positive_fields):
        raise GpuBudgetGateError("GPU reservation contains a negative accounting field")
    margin = reservation.safety_margin_fraction
    if margin >= 0.25:
        raise GpuBudgetGateError("GPU reservation safety margin is invalid")
    expected_safe_ceiling = _floor_usd(reservation.global_gpu_hard_stop_usd * (1 - margin))
    if abs(reservation.safety_adjusted_gpu_ceiling_usd - expected_safe_ceiling) > 1e-6:
        raise GpuBudgetGateError("GPU reservation safety-adjusted ceiling is inconsistent")
    if (
        abs(
            reservation.prior_reserved_gpu_usd
            - (reservation.prior_committed_gpu_usd - reservation.prior_incurred_gpu_usd)
        )
        > 1e-6
    ):
        raise GpuBudgetGateError("GPU reservation prior accounting is inconsistent")
    if (
        abs(
            reservation.remaining_safe_gpu_before_phase_usd
            - (expected_safe_ceiling - reservation.prior_committed_gpu_usd)
        )
        > 1e-6
    ):
        raise GpuBudgetGateError("GPU reservation remaining GPU balance is inconsistent")
    if (
        abs(
            reservation.committed_gpu_after_reservation_usd
            - (reservation.prior_committed_gpu_usd + reservation.approved_phase_maximum_usd)
        )
        > 1e-6
    ):
        raise GpuBudgetGateError("GPU reservation post-commit total is inconsistent")
    if (
        reservation.committed_gpu_after_reservation_usd
        > reservation.safety_adjusted_gpu_ceiling_usd + 1e-6
    ):
        raise GpuBudgetGateError("GPU reservation exceeds its safety-adjusted ceiling")
    if (
        abs(
            reservation.committed_total_after_reservation_usd
            - (reservation.prior_committed_total_usd + reservation.approved_phase_maximum_usd)
        )
        > 1e-6
    ):
        raise GpuBudgetGateError("GPU reservation post-commit all-category total is inconsistent")
    inferred_total_hard_stop = (
        reservation.prior_committed_total_usd + reservation.remaining_total_before_phase_usd
    )
    if reservation.committed_total_after_reservation_usd > inferred_total_hard_stop + 1e-6:
        raise GpuBudgetGateError("GPU reservation exceeds its all-category ceiling")
    safe_spend = min(
        reservation.approved_phase_maximum_usd,
        reservation.remaining_safe_gpu_before_phase_usd,
        reservation.remaining_total_before_phase_usd,
    )
    expected_runtime = min(
        reservation.approved_maximum_runtime_hours,
        safe_spend / reservation.live_hourly_total_usd,
    )
    if abs(reservation.maximum_safe_runtime_hours - expected_runtime) > 1e-9:
        raise GpuBudgetGateError("GPU reservation safe runtime is inconsistent")


def load_gpu_phase_budget_reservation(
    path: str | Path,
) -> GpuPhaseBudgetReservation:
    """Load and authenticate a persisted reservation receipt for safe resume."""

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise GpuBudgetGateError("GPU reservation receipt must be a JSON object")
    field_names = {field.name for field in fields(GpuPhaseBudgetReservation)}
    expected_keys = {
        "schema_version",
        "protocol_version",
        "record_hash",
        *field_names,
    }
    if set(payload) != expected_keys:
        raise GpuBudgetGateError("GPU reservation receipt has an unexpected schema")
    if payload.get("schema_version") != 1:
        raise GpuBudgetGateError("GPU reservation receipt schema version is unsupported")
    if payload.get("protocol_version") != GPU_PHASE_BUDGET_PROTOCOL:
        raise GpuBudgetGateError("GPU reservation receipt protocol version is unsupported")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if not isinstance(record_hash, str) or record_hash != stable_hash(unsigned):
        raise GpuBudgetGateError("GPU reservation receipt content hash mismatch")
    try:
        reservation = GpuPhaseBudgetReservation(**{name: payload[name] for name in field_names})
    except TypeError as exc:  # pragma: no cover - exact key check above
        raise GpuBudgetGateError("GPU reservation receipt is malformed") from exc
    _validate_receipt_invariants(reservation)
    return reservation


def validate_existing_gpu_phase_reservation(
    *,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    phase: str,
    session_id: str,
    require_active: bool = True,
) -> dict[str, Any]:
    """Validate, without reserving again, a receipt used to resume one session.

    This is the only safe resume path.  It proves the opaque session identity,
    authenticates the receipt invariants, and requires an exact matching entry
    in the canonical ledger.  A settled reservation cannot authorize more GPU
    work when ``require_active`` is true.
    """

    _validate_receipt_invariants(reservation)
    if phase != reservation.phase:
        raise GpuBudgetGateError("GPU resume phase disagrees with the reservation")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be non-empty")
    expected_session_hash = stable_hash({"opaque_gpu_session_id": session_id})
    if expected_session_hash != reservation.session_hash:
        raise GpuBudgetGateError("GPU resume session disagrees with the reservation")
    return _validated_ledger_entry(
        ledger=ledger,
        reservation=reservation,
        require_active=require_active,
    )


def _validated_ledger_entry(
    *,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    require_active: bool,
) -> dict[str, Any]:
    if abs(ledger.limits.gpu - reservation.global_gpu_hard_stop_usd) > 1e-6:
        raise GpuBudgetGateError("canonical ledger GPU hard stop disagrees with the reservation")
    inferred_total_hard_stop = (
        reservation.prior_committed_total_usd + reservation.remaining_total_before_phase_usd
    )
    if abs(ledger.limits.total - inferred_total_hard_stop) > 1e-6:
        raise GpuBudgetGateError("canonical ledger total hard stop disagrees with the reservation")
    document = ledger.document()
    matching = [
        entry
        for entry in document["entries"]
        if entry.get("entry_id") == reservation.reservation_id
    ]
    if len(matching) != 1:
        raise GpuBudgetGateError("canonical ledger does not contain exactly one reservation")
    entry = matching[0]
    if entry.get("kind") != "gpu" or entry.get("description") != reservation.description:
        raise GpuBudgetGateError("canonical ledger GPU reservation content mismatch")
    status = entry.get("status", "incurred")
    if status not in {"estimated", "incurred"}:
        raise GpuBudgetGateError("canonical ledger GPU reservation status is invalid")
    if (
        status == "estimated"
        and abs(float(entry.get("amount_usd")) - reservation.approved_phase_maximum_usd) > 1e-6
    ):
        raise GpuBudgetGateError("canonical ledger GPU reservation amount mismatch")
    if status == "estimated":
        committed = ledger.totals(document, include_estimates=True)
        if abs(committed["gpu"] - reservation.committed_gpu_after_reservation_usd) > 1e-6:
            raise GpuBudgetGateError(
                "canonical ledger GPU total drifted after the active reservation"
            )
    if require_active and status != "estimated":
        raise ReservationConflict("settled GPU reservation cannot authorize a resumed session")
    return dict(entry)


def settle_gpu_phase_budget(
    *,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    incurred_usd: float,
) -> dict[str, float]:
    """Reconcile a stopped session; exact repeated settlement is idempotent."""

    _validate_receipt_invariants(reservation)
    _validated_ledger_entry(
        ledger=ledger,
        reservation=reservation,
        require_active=False,
    )
    return ledger.settle_reservation(
        reservation.reservation_id,
        reservation.settlement_entry(incurred_usd=incurred_usd),
    )


def validate_gpu_phase_bootstrap(
    *,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    phase: str,
    session_id: str,
    expected_approved_runtime_hours: float,
    expected_live_hourly_total_usd: float,
) -> dict[str, Any]:
    """Validate an authenticated local receipt before any GPU backend starts.

    The raw session nonce is accepted in memory only.  The returned payload is
    safe to persist or pass to a watchdog and contains only its SHA-256 hash.
    """

    expected_runtime = _positive_finite(
        expected_approved_runtime_hours,
        field="expected_approved_runtime_hours",
    )
    expected_rate = _positive_finite(
        expected_live_hourly_total_usd,
        field="expected_live_hourly_total_usd",
    )
    validate_existing_gpu_phase_reservation(
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        require_active=True,
    )
    if abs(reservation.approved_maximum_runtime_hours - expected_runtime) > 1e-9:
        raise GpuBudgetGateError(
            "GPU reservation approved runtime disagrees with the launch contract"
        )
    if abs(reservation.live_hourly_total_usd - expected_rate) > 1e-9:
        raise GpuBudgetGateError(
            "GPU reservation live hourly rate disagrees with the launch contract"
        )
    manifest = reservation.manifest()
    payload = {
        "schema_version": 1,
        "protocol_version": GPU_PHASE_BUDGET_PROTOCOL,
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": manifest["record_hash"],
        "session_hash": reservation.session_hash,
        "approved_phase_maximum_usd": reservation.approved_phase_maximum_usd,
        "approved_maximum_runtime_hours": reservation.approved_maximum_runtime_hours,
        "live_hourly_total_usd": reservation.live_hourly_total_usd,
        "maximum_safe_runtime_hours": reservation.maximum_safe_runtime_hours,
        "prior_incurred_gpu_usd": reservation.prior_incurred_gpu_usd,
        "prior_reserved_gpu_usd": reservation.prior_reserved_gpu_usd,
        "prior_committed_gpu_usd": reservation.prior_committed_gpu_usd,
        "global_gpu_hard_stop_usd": reservation.global_gpu_hard_stop_usd,
        "safety_margin_fraction": reservation.safety_margin_fraction,
        "committed_gpu_after_reservation_usd": (reservation.committed_gpu_after_reservation_usd),
        "passed": True,
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def reserve_gpu_phase_budget(
    *,
    ledger: CostLedger,
    phase: str,
    session_id: str,
    approved_phase_maximum_usd: float,
    approved_maximum_runtime_hours: float,
    live_hourly_total_usd: float,
    safety_margin_fraction: float = 0.03,
) -> GpuPhaseBudgetReservation:
    """Reserve one GPU phase maximum and derive its cumulative safe runtime.

    ``session_id`` is an opaque, caller-generated launch nonce.  Only its hash
    is persisted.  Repeating the same ``phase``/``session_id`` pair is always an
    error, even after settlement, while a new session can proceed after the
    previous GPU reservation has been reconciled.

    The approved dollar maximum is reserved in full.  Runtime is bounded by the
    approved duration, that dollar maximum, the safety-adjusted cumulative GPU
    balance, and the remaining all-category balance, each evaluated at the live
    total hourly rate supplied by the caller.
    """

    if not isinstance(phase, str) or _PHASE_RE.fullmatch(phase) is None:
        raise ValueError("phase must be a stable non-placeholder identifier")
    if (
        not isinstance(session_id, str)
        or not session_id.strip()
        or session_id != session_id.strip()
        or len(session_id) > 512
        or any(ord(character) < 32 for character in session_id)
    ):
        raise ValueError("session_id must be a non-empty opaque identifier")
    phase_maximum = _ceil_usd(
        _positive_finite(
            approved_phase_maximum_usd,
            field="approved_phase_maximum_usd",
        )
    )
    approved_runtime = _positive_finite(
        approved_maximum_runtime_hours,
        field="approved_maximum_runtime_hours",
    )
    live_rate = _positive_finite(live_hourly_total_usd, field="live_hourly_total_usd")
    if (
        isinstance(safety_margin_fraction, bool)
        or not math.isfinite(float(safety_margin_fraction))
        or not 0 < float(safety_margin_fraction) < 0.25
    ):
        raise ValueError("safety_margin_fraction must be in (0, 0.25)")
    margin = float(safety_margin_fraction)

    safe_gpu_ceiling = _floor_usd(ledger.limits.gpu * (1 - margin))
    if phase_maximum > safe_gpu_ceiling + 1e-9:
        raise GpuBudgetGateError(
            "approved phase maximum exceeds the entire safety-adjusted GPU ceiling"
        )

    session_hash = stable_hash({"opaque_gpu_session_id": session_id})
    reservation_id = stable_hash(
        {
            "protocol": GPU_PHASE_BUDGET_PROTOCOL,
            "phase": phase,
            "session_hash": session_hash,
        }
    )
    description = _reservation_description(phase=phase, session_hash=session_hash)
    snapshot = ledger.reserve_once(
        reservation_id,
        CostEntry(
            kind="gpu",
            amount_usd=phase_maximum,
            description=description,
            status="estimated",
        ),
        maximum_totals={"gpu": safe_gpu_ceiling},
        require_no_outstanding_kind=True,
    )

    prior_incurred_gpu = snapshot.incurred_before["gpu"]
    prior_committed_gpu = snapshot.committed_before["gpu"]
    prior_reserved_gpu = prior_committed_gpu - prior_incurred_gpu
    remaining_safe_gpu = max(0.0, safe_gpu_ceiling - prior_committed_gpu)
    remaining_total = max(0.0, ledger.limits.total - snapshot.committed_before["total"])
    safe_spend_for_phase = min(phase_maximum, remaining_safe_gpu, remaining_total)
    maximum_safe_runtime = min(approved_runtime, safe_spend_for_phase / live_rate)
    if maximum_safe_runtime <= 0:  # pragma: no cover - guarded by atomic ceilings
        raise GpuBudgetGateError("no positive GPU runtime remains after reservation")

    return GpuPhaseBudgetReservation(
        reservation_id=reservation_id,
        phase=phase,
        session_hash=session_hash,
        approved_phase_maximum_usd=phase_maximum,
        approved_maximum_runtime_hours=approved_runtime,
        live_hourly_total_usd=live_rate,
        safety_margin_fraction=margin,
        global_gpu_hard_stop_usd=ledger.limits.gpu,
        safety_adjusted_gpu_ceiling_usd=safe_gpu_ceiling,
        prior_incurred_gpu_usd=round(prior_incurred_gpu, 6),
        prior_reserved_gpu_usd=round(prior_reserved_gpu, 6),
        prior_committed_gpu_usd=round(prior_committed_gpu, 6),
        prior_committed_total_usd=round(snapshot.committed_before["total"], 6),
        remaining_safe_gpu_before_phase_usd=round(remaining_safe_gpu, 6),
        remaining_total_before_phase_usd=round(remaining_total, 6),
        maximum_safe_runtime_hours=maximum_safe_runtime,
        committed_gpu_after_reservation_usd=snapshot.committed_after["gpu"],
        committed_total_after_reservation_usd=snapshot.committed_after["total"],
    )


__all__ = [
    "GPU_PHASE_BUDGET_PROTOCOL",
    "GPU_PHASE_SETTLEMENT_PROTOCOL",
    "GpuBudgetGateError",
    "GpuPhaseBudgetReservation",
    "approved_gpu_phase_maximum_usd",
    "load_gpu_phase_budget_reservation",
    "reserve_gpu_phase_budget",
    "settle_gpu_phase_budget",
    "validate_existing_gpu_phase_reservation",
    "validate_gpu_phase_bootstrap",
    "write_json_exclusive",
]
