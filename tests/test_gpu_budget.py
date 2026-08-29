from __future__ import annotations

from pathlib import Path

import pytest

from model_forensics.budget import (
    BudgetExceeded,
    BudgetLimits,
    CostEntry,
    CostLedger,
    ReservationConflict,
)
from model_forensics.gpu_budget import (
    GPU_PHASE_BUDGET_PROTOCOL,
    GpuBudgetGateError,
    load_gpu_phase_budget_reservation,
    reserve_gpu_phase_budget,
    settle_gpu_phase_budget,
    validate_existing_gpu_phase_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import stable_hash, write_json


def _ledger(path: Path) -> CostLedger:
    return CostLedger(path, BudgetLimits(gpu=220, api=100, total=325))


def test_reservation_accounts_prior_incurred_and_derives_live_safe_runtime(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    ledger.append(CostEntry(kind="gpu", amount_usd=50, description="completed pilot"))
    ledger.append(CostEntry(kind="api", amount_usd=20, description="completed judging"))

    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="resample_gpu",
        session_id="launch-nonce-001",
        approved_phase_maximum_usd=72,
        approved_maximum_runtime_hours=4,
        live_hourly_total_usd=24,
    )

    assert reservation.prior_incurred_gpu_usd == 50
    assert reservation.prior_reserved_gpu_usd == 0
    assert reservation.prior_committed_gpu_usd == 50
    assert reservation.safety_adjusted_gpu_ceiling_usd == 213.4
    assert reservation.remaining_safe_gpu_before_phase_usd == 163.4
    assert reservation.maximum_safe_runtime_hours == 3
    assert reservation.committed_gpu_after_reservation_usd == 122
    assert reservation.watchdog_budget_kwargs() == {
        "gpu_hard_stop_usd": 220,
        "maximum_runtime_hours": 3,
        "safety_margin_fraction": 0.03,
        "prior_committed_gpu_usd": 50,
    }
    assert ledger.totals(ledger.document(), include_estimates=True)["gpu"] == 122
    manifest = reservation.manifest()
    assert manifest["protocol_version"] == GPU_PHASE_BUDGET_PROTOCOL
    assert manifest["record_hash"] == stable_hash(
        {key: value for key, value in manifest.items() if key != "record_hash"}
    )
    assert "launch-nonce-001" not in str(manifest)


def test_settled_actual_cost_is_used_for_the_next_phase(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    first = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="launch-001",
        approved_phase_maximum_usd=60,
        approved_maximum_runtime_hours=3,
        live_hourly_total_usd=20,
    )
    ledger.settle_reservation(
        first.reservation_id,
        first.settlement_entry(incurred_usd=17.25),
    )

    second = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_treatment_gpu",
        session_id="launch-002",
        approved_phase_maximum_usd=70,
        approved_maximum_runtime_hours=3.5,
        live_hourly_total_usd=20,
    )

    assert second.prior_incurred_gpu_usd == 17.25
    assert second.prior_reserved_gpu_usd == 0
    assert second.committed_gpu_after_reservation_usd == 87.25


def test_same_phase_session_is_never_reusable_even_after_settlement(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    first = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="lens_gpu",
        session_id="launch-once",
        approved_phase_maximum_usd=40,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=20,
    )
    ledger.settle_reservation(
        first.reservation_id,
        first.settlement_entry(incurred_usd=10),
    )

    with pytest.raises(ReservationConflict, match="already been used"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="lens_gpu",
            session_id="launch-once",
            approved_phase_maximum_usd=40,
            approved_maximum_runtime_hours=2,
            live_hourly_total_usd=20,
        )


def test_active_receipt_is_the_only_safe_same_session_resume_path(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="resample_gpu",
        session_id="same-running-pod-session",
        approved_phase_maximum_usd=48,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=24,
    )
    receipt_path = tmp_path / "gpu_reservation.json"
    write_json(receipt_path, reservation.manifest())
    loaded = load_gpu_phase_budget_reservation(receipt_path)

    entry = validate_existing_gpu_phase_reservation(
        ledger=ledger,
        reservation=loaded,
        phase="resample_gpu",
        session_id="same-running-pod-session",
    )
    assert entry["status"] == "estimated"
    assert entry["entry_id"] == reservation.reservation_id
    bootstrap = validate_gpu_phase_bootstrap(
        ledger=ledger,
        reservation=loaded,
        phase="resample_gpu",
        session_id="same-running-pod-session",
        expected_approved_runtime_hours=2,
        expected_live_hourly_total_usd=24,
    )
    assert bootstrap["maximum_safe_runtime_hours"] == 2
    assert bootstrap["prior_committed_gpu_usd"] == 0
    assert "same-running-pod-session" not in str(bootstrap)
    with pytest.raises(GpuBudgetGateError, match="session disagrees"):
        validate_existing_gpu_phase_reservation(
            ledger=ledger,
            reservation=loaded,
            phase="resample_gpu",
            session_id="different-pod-session",
        )
    with pytest.raises(GpuBudgetGateError, match="live hourly rate"):
        validate_gpu_phase_bootstrap(
            ledger=ledger,
            reservation=loaded,
            phase="resample_gpu",
            session_id="same-running-pod-session",
            expected_approved_runtime_hours=2,
            expected_live_hourly_total_usd=25,
        )
    with pytest.raises(ReservationConflict, match="already been used"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="resample_gpu",
            session_id="same-running-pod-session",
            approved_phase_maximum_usd=48,
            approved_maximum_runtime_hours=2,
            live_hourly_total_usd=24,
        )


def test_settlement_is_exactly_idempotent_and_settled_receipt_cannot_resume(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="lens_gpu",
        session_id="lens-pod-session",
        approved_phase_maximum_usd=50,
        approved_maximum_runtime_hours=2.5,
        live_hourly_total_usd=20,
    )

    first = settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=12.75,
    )
    second = settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=12.75,
    )
    assert first == second
    assert first["gpu"] == 12.75
    with pytest.raises(ValueError, match="different content"):
        settle_gpu_phase_budget(
            ledger=ledger,
            reservation=reservation,
            incurred_usd=13,
        )
    with pytest.raises(ReservationConflict, match="cannot authorize"):
        validate_existing_gpu_phase_reservation(
            ledger=ledger,
            reservation=reservation,
            phase="lens_gpu",
            session_id="lens-pod-session",
        )


def test_tampered_persisted_reservation_receipt_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="lens_gpu",
        session_id="tamper-test-session",
        approved_phase_maximum_usd=40,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=20,
    )
    payload = reservation.manifest()
    payload["prior_incurred_gpu_usd"] = 10
    path = tmp_path / "gpu_reservation.json"
    write_json(path, payload)

    with pytest.raises(GpuBudgetGateError, match="content hash"):
        load_gpu_phase_budget_reservation(path)


def test_unsettled_gpu_reservation_blocks_a_second_launch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="launch-001",
        approved_phase_maximum_usd=30,
        approved_maximum_runtime_hours=1.5,
        live_hourly_total_usd=20,
    )

    with pytest.raises(ReservationConflict, match="already outstanding"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="behavior_treatment_gpu",
            session_id="launch-002",
            approved_phase_maximum_usd=30,
            approved_maximum_runtime_hours=1.5,
            live_hourly_total_usd=20,
        )


def test_cumulative_safe_gpu_ceiling_fails_closed_without_writing(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    ledger.append(CostEntry(kind="gpu", amount_usd=190, description="prior GPU work"))
    before = ledger.document()

    with pytest.raises(BudgetExceeded, match="transaction ceiling"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="resample_gpu",
            session_id="launch-over-budget",
            approved_phase_maximum_usd=24,
            approved_maximum_runtime_hours=1,
            live_hourly_total_usd=24,
        )
    assert ledger.document() == before


def test_entire_phase_maximum_and_inputs_are_validated_before_reservation(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "cost.yaml")
    with pytest.raises(GpuBudgetGateError, match="entire safety-adjusted"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="resample_gpu",
            session_id="too-large",
            approved_phase_maximum_usd=214,
            approved_maximum_runtime_hours=9,
            live_hourly_total_usd=24,
        )
    assert not ledger.path.exists()

    with pytest.raises(ValueError, match="live_hourly_total_usd"):
        reserve_gpu_phase_budget(
            ledger=ledger,
            phase="resample_gpu",
            session_id="invalid-rate",
            approved_phase_maximum_usd=24,
            approved_maximum_runtime_hours=1,
            live_hourly_total_usd=float("nan"),
        )
    assert not ledger.path.exists()
