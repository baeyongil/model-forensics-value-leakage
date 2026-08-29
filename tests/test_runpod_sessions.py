from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    GpuPhaseBudgetReservation,
    reserve_gpu_phase_budget,
    settle_gpu_phase_budget,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_sessions import (
    GPU_BUDGET_BOOTSTRAP_FILENAME,
    GPU_PREFLIGHT_FILENAME,
    SETTLEMENT_FILENAME,
    WATCHDOG_PID_FILENAME,
    WATCHDOG_STATE_FILENAME,
    RunpodSessionError,
    prepare_runpod_session_directory,
    validate_active_runpod_session,
    validate_completed_runpod_sessions,
)


def _ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(
        tmp_path / "cost_ledger.yaml",
        BudgetLimits(gpu=220, api=100, total=325),
    )


def _reserve_pending(
    *,
    tmp_path: Path,
    ledger: CostLedger,
    phase: str,
    session_id: str,
) -> tuple[GpuPhaseBudgetReservation, Path]:
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase=phase,
        session_id=session_id,
        approved_phase_maximum_usd=48,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=24,
    )
    payload = validate_gpu_phase_bootstrap(
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        expected_approved_runtime_hours=2,
        expected_live_hourly_total_usd=24,
    )
    pending = tmp_path / ".runpod" / f"{phase}.pending.json"
    write_json(pending, payload)
    return reservation, pending


def _complete_session(
    *,
    session_dir: Path,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    incurred_usd: float,
) -> None:
    watchdog = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "stopped_confirmed",
        "pod_id": "private-only-pod-id",
        "live_metadata": {"machine_gpu_identity": ["private-gpu-uuid"]},
    }
    write_json(
        session_dir / WATCHDOG_STATE_FILENAME,
        watchdog,
    )
    settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=incurred_usd,
    )
    bootstrap = json.loads(
        (session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).read_text(encoding="utf-8")
    )
    settlement = {
        "schema_version": 1,
        "protocol_version": GPU_PHASE_SETTLEMENT_PROTOCOL,
        "phase": bootstrap["phase"],
        "reservation_id": bootstrap["reservation_id"],
        "reservation_record_hash": bootstrap["reservation_record_hash"],
        "session_hash": bootstrap["session_hash"],
        "provider_incurred_usd": incurred_usd,
        "watchdog_state_hash": stable_hash(watchdog),
        "status": "settled",
    }
    settlement["record_hash"] = stable_hash(settlement)
    write_json(session_dir / SETTLEMENT_FILENAME, settlement)


def test_two_sequential_gpu_phases_require_completed_private_prior_session(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    sessions_root = tmp_path / ".runpod" / "sessions"
    first, first_pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="phase-session-001",
    )
    first_dir = prepare_runpod_session_directory(
        sessions_root=sessions_root,
        pending_bootstrap_path=first_pending,
        ledger=ledger,
    )
    _complete_session(
        session_dir=first_dir,
        ledger=ledger,
        reservation=first,
        incurred_usd=12,
    )
    summaries = validate_completed_runpod_sessions(
        sessions_root=first_dir.parent,
        ledger=ledger,
    )
    assert [item["status"] for item in summaries] == ["stopped_confirmed_and_settled"]

    _second, second_pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_treatment_gpu",
        session_id="phase-session-002",
    )
    second_dir = prepare_runpod_session_directory(
        sessions_root=sessions_root,
        pending_bootstrap_path=second_pending,
        ledger=ledger,
    )

    assert first_dir != second_dir
    assert (first_dir / WATCHDOG_STATE_FILENAME).is_file()
    assert (second_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).is_file()


def test_stale_or_active_prior_private_session_blocks_next_phase(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    sessions_root = tmp_path / ".runpod" / "sessions"
    first, first_pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="stale-session-001",
    )
    first_dir = prepare_runpod_session_directory(
        sessions_root=sessions_root,
        pending_bootstrap_path=first_pending,
        ledger=ledger,
    )
    write_json(
        first_dir / WATCHDOG_STATE_FILENAME,
        {
            "schema_version": 2,
            "watchdog_version": "runpod-gpu-cost-watchdog-v2",
            "status": "armed",
        },
    )
    # Reconcile only the ledger so a new reservation can be formed; the private
    # lifecycle remains intentionally incomplete and must still block re-arm.
    settle_gpu_phase_budget(ledger=ledger, reservation=first, incurred_usd=5)
    _second, second_pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_treatment_gpu",
        session_id="stale-session-002",
    )

    with pytest.raises(RunpodSessionError, match="not stopped_confirmed"):
        prepare_runpod_session_directory(
            sessions_root=sessions_root,
            pending_bootstrap_path=second_pending,
            ledger=ledger,
        )


def test_completed_session_requires_settlement_bound_to_exact_stopped_watchdog(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    reservation, pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="settlement-watchdog-binding",
    )
    session_dir = prepare_runpod_session_directory(
        sessions_root=tmp_path / ".runpod" / "sessions",
        pending_bootstrap_path=pending,
        ledger=ledger,
    )
    _complete_session(
        session_dir=session_dir,
        ledger=ledger,
        reservation=reservation,
        incurred_usd=8,
    )
    watchdog = json.loads((session_dir / WATCHDOG_STATE_FILENAME).read_text(encoding="utf-8"))
    watchdog["stop_reason"] = "tampered-after-settlement"
    write_json(session_dir / WATCHDOG_STATE_FILENAME, watchdog)

    with pytest.raises(RunpodSessionError, match="watchdog state hash"):
        validate_completed_runpod_sessions(
            sessions_root=session_dir.parent,
            ledger=ledger,
        )


def test_active_session_gate_binds_receipt_watchdog_preflight_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    session_id = "active-session-secret-nonce"
    phase = "behavior_baseline_gpu"
    reservation, pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase=phase,
        session_id=session_id,
    )
    session_dir = prepare_runpod_session_directory(
        sessions_root=tmp_path / ".runpod" / "sessions",
        pending_bootstrap_path=pending,
        ledger=ledger,
    )
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    watchdog = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "armed",
        "updated_at": (now - timedelta(seconds=5)).isoformat(),
        "action": "stop_only_preserve_volume",
        "limits": {
            "gpu_hard_stop_usd": 220.0,
            "global_safe_budget_usd": 213.4,
            "safe_budget_usd": 213.4,
            "safety_margin_fraction": 0.03,
            "maximum_runtime_hours": 2.0,
            "maximum_approved_hourly_total_usd": 24.0,
            "prior_committed_gpu_usd": 0.0,
        },
        "deadline": {
            "effective_deadline": (now + timedelta(hours=1)).isoformat(),
            "calculation_hourly_usd": 24.0,
            "incurred_cost_usd": 0.0,
        },
    }
    write_json(session_dir / WATCHDOG_STATE_FILENAME, watchdog)
    (session_dir / WATCHDOG_PID_FILENAME).write_text(f"{os.getpid()}\n", encoding="utf-8")
    bootstrap = json.loads(
        (session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).read_text(encoding="utf-8")
    )
    preflight = {
        "schema_version": 3,
        "passed": True,
        "planned_hours": 2.0,
        "prior_committed_gpu_cost_usd": 0.0,
        "gpu_budget_usd": 220.0,
        "price": {"approved_hourly_total_usd": 24.0},
        "watchdog": {
            "pid": os.getpid(),
            "state_path": str(session_dir / WATCHDOG_STATE_FILENAME),
            "state_updated_at": watchdog["updated_at"],
        },
        "gpu_budget_reservation": {
            field: bootstrap[field]
            for field in ("reservation_id", "reservation_record_hash", "session_hash", "phase")
        },
    }
    write_json(session_dir / GPU_PREFLIGHT_FILENAME, preflight)

    gate = validate_active_runpod_session(
        session_directory=session_dir,
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        now=now,
    )
    assert gate["passed"] is True
    assert session_id not in str(gate)

    refreshed_watchdog = {
        **watchdog,
        "updated_at": (now + timedelta(seconds=10)).isoformat(),
    }
    write_json(session_dir / WATCHDOG_STATE_FILENAME, refreshed_watchdog)
    refreshed_gate = validate_active_runpod_session(
        session_directory=session_dir,
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        now=now + timedelta(seconds=15),
    )
    assert refreshed_gate["watchdog_updated_at"] == refreshed_watchdog["updated_at"]

    stale_watchdog = {
        **watchdog,
        "updated_at": (now - timedelta(seconds=91)).isoformat(),
    }
    write_json(session_dir / WATCHDOG_STATE_FILENAME, stale_watchdog)
    with pytest.raises(RunpodSessionError, match="stale"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
        )

    tampered_watchdog = {**watchdog, "limits": {**watchdog["limits"]}}
    tampered_watchdog["limits"]["prior_committed_gpu_usd"] = 1.0
    write_json(session_dir / WATCHDOG_STATE_FILENAME, tampered_watchdog)
    with pytest.raises(RunpodSessionError, match="prior committed"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
        )
    write_json(session_dir / WATCHDOG_STATE_FILENAME, watchdog)

    external = tmp_path / "external-preflight.json"
    write_json(external, preflight)
    (session_dir / GPU_PREFLIGHT_FILENAME).unlink()
    (session_dir / GPU_PREFLIGHT_FILENAME).symlink_to(external)
    with pytest.raises(RunpodSessionError, match="unsafe"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
        )
