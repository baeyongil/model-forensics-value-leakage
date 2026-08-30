from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import reserve_gpu_phase_budget, settle_gpu_phase_budget
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_sessions import (
    LEGACY_SETTLEMENT_V1_FILENAME,
    RunpodSessionError,
    validate_completed_runpod_sessions,
)
from model_forensics.settlement_upgrade import (
    SettlementUpgradeError,
    upgrade_legacy_gpu_settlement,
)


@dataclass(frozen=True)
class UpgradeFixture:
    root: Path
    ledger_path: Path
    reservation_path: Path
    session_dir: Path
    watchdog_path: Path
    external_path: Path
    settlement_path: Path
    legacy_raw: bytes
    amount: float


def _limits() -> BudgetLimits:
    return BudgetLimits(gpu=220, api=100, total=325)


def _external_receipt(
    *,
    reservation: dict[str, object],
    amount: float,
    billing_status: str,
) -> dict[str, object]:
    pod_id_hash = stable_hash({"runpod_pod_id": "private-pod"})
    stop_evidence = {
        "desired_status": "EXITED",
        "environment_verified": True,
        "started_at": "2026-08-30T00:10:00Z",
        "exited_at": "2026-08-30T00:13:57.777000Z",
        "runtime_ms": 237_777,
    }
    billing_query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": stop_evidence["started_at"],
        "end_time": stop_evidence["exited_at"],
    }
    if billing_status == "pending":
        evidence_kind = "provider_timestamps_conservative_ceiling"
        billing_evidence = {
            "billing_status": "pending",
            "evidence_kind": evidence_kind,
            "pod_id_hash": pod_id_hash,
            "provider_amount_usd": None,
            "settlement_amount_usd": amount,
            "time_billed_ms": None,
            "billing_bucket_time": None,
            "provider_billing_row_hash": None,
            "conservative_ceiling_usd": amount,
            "runtime_ceiling_minutes": 4,
        }
    else:
        evidence_kind = "provider_billing_row"
        billing_evidence = {
            "billing_status": "final",
            "evidence_kind": evidence_kind,
            "pod_id_hash": pod_id_hash,
            "provider_amount_usd": amount,
            "settlement_amount_usd": amount,
            "time_billed_ms": 237_777,
            "billing_bucket_time": "2026-08-30T00:14:00Z",
            "provider_billing_row_hash": stable_hash({"billing": "row"}),
            "conservative_ceiling_usd": amount,
            "runtime_ceiling_minutes": 4,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-external-stop-v1",
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-30T00:20:00Z",
        "prior_lifecycle_operation": "rearmed",
        "lifecycle_before_hash": stable_hash({"lifecycle": "before"}),
        "lifecycle_stopped_hash": stable_hash({"lifecycle": "stopped"}),
        "session_hash": reservation["session_hash"],
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "pod_id_hash": pod_id_hash,
        "stop_evidence": stop_evidence,
        "stop_evidence_hash": stable_hash(stop_evidence),
        "billing_query": billing_query,
        "billing_query_hash": stable_hash(billing_query),
        "billing_evidence": billing_evidence,
        "billing_evidence_hash": stable_hash(billing_evidence),
        "billing_status": billing_status,
        "evidence_kind": evidence_kind,
        "settlement_amount_usd": amount,
        "source_artifact_hashes": [],
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _fixture(
    tmp_path: Path,
    *,
    billing_status: str = "pending",
    ledger_amount: float = 1.761149,
    settlement_amount: float = 1.761149,
) -> UpgradeFixture:
    root = tmp_path
    ledger_path = root / "data" / "manifests" / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, _limits())
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="opaque-current-incident-nonce",
        approved_phase_maximum_usd=39.625834,
        approved_maximum_runtime_hours=1.5,
        live_hourly_total_usd=26.417222,
    )
    reservation_path = root / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
    reservation_path.parent.mkdir(parents=True)
    write_json(reservation_path, reservation.manifest())
    reservation_path.chmod(0o600)
    settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=ledger_amount,
    )

    session_dir = (
        root
        / ".runpod"
        / "sessions"
        / reservation.session_hash.removeprefix("sha256:")
    )
    session_dir.mkdir(parents=True)
    watchdog = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "stopped_confirmed",
        "stop_reason": "provider_exit_verified",
    }
    watchdog_path = session_dir / "runpod_watchdog.json"
    write_json(watchdog_path, watchdog)
    watchdog_path.chmod(0o600)

    legacy: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "cumulative-gpu-phase-settlement-v1",
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
        "provider_incurred_usd": settlement_amount,
        "watchdog_state_hash": stable_hash(watchdog),
        "status": "settled",
    }
    legacy["record_hash"] = stable_hash(legacy)
    # Noncanonical whitespace makes exact-byte preservation observable.
    legacy_raw = (json.dumps(legacy, sort_keys=False, indent=1) + " \n").encode()
    settlement_path = session_dir / "settlement.json"
    settlement_path.write_bytes(legacy_raw)
    settlement_path.chmod(0o600)

    external_path = session_dir / "external_stop_receipt.json"
    write_json(
        external_path,
        _external_receipt(
            reservation=reservation.manifest(),
            amount=settlement_amount,
            billing_status=billing_status,
        ),
    )
    external_path.chmod(0o600)
    return UpgradeFixture(
        root=root,
        ledger_path=ledger_path,
        reservation_path=reservation_path,
        session_dir=session_dir,
        watchdog_path=watchdog_path,
        external_path=external_path,
        settlement_path=settlement_path,
        legacy_raw=legacy_raw,
        amount=settlement_amount,
    )


def _upgrade(item: UpgradeFixture) -> dict[str, object]:
    return upgrade_legacy_gpu_settlement(
        project_root=item.root,
        reservation_receipt_path=item.reservation_path,
        cost_ledger_path=item.ledger_path,
        watchdog_state_path=item.watchdog_path,
        external_stop_receipt_path=item.external_path,
        settlement_path=item.settlement_path,
        limits=_limits(),
    )


@pytest.mark.parametrize("billing_status", ["pending", "final"])
def test_upgrade_preserves_exact_v1_is_ledger_neutral_and_idempotent(
    tmp_path: Path,
    billing_status: str,
) -> None:
    item = _fixture(tmp_path, billing_status=billing_status)
    ledger_before = item.ledger_path.read_bytes()

    first = _upgrade(item)
    canonical_after_first = item.settlement_path.read_bytes()
    second = _upgrade(item)

    archive = item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME
    assert archive.read_bytes() == item.legacy_raw
    assert first == second
    assert item.settlement_path.read_bytes() == canonical_after_first
    assert item.ledger_path.read_bytes() == ledger_before
    assert first["schema_version"] == 2
    assert first["billing_status"] == billing_status
    assert first["accounted_gpu_usd"] == pytest.approx(1.761149)
    assert first["provider_incurred_usd"] == (
        None if billing_status == "pending" else pytest.approx(1.761149)
    )
    assert first["legacy_settlement_v1_file_hash"] == (
        "sha256:" + hashlib.sha256(item.legacy_raw).hexdigest()
    )
    assert first["record_hash"] == stable_hash(
        {key: value for key, value in first.items() if key != "record_hash"}
    )

    summaries = validate_completed_runpod_sessions(
        sessions_root=item.session_dir.parent,
        ledger=CostLedger(item.ledger_path, _limits()),
    )
    assert summaries == [
        {
            "session_hash": first["session_hash"],
            "reservation_id": first["reservation_id"],
            "settlement_record_hash": first["record_hash"],
            "status": "stopped_confirmed_and_settled",
        }
    ]


@pytest.mark.parametrize(
    ("ledger_amount", "settlement_amount", "message"),
    [
        (1.7, 1.761149, "canonical ledger incurred amount"),
        (1.761149, 1.7, "canonical ledger incurred amount"),
    ],
)
def test_upgrade_rejects_amount_drift_before_any_settlement_write(
    tmp_path: Path,
    ledger_amount: float,
    settlement_amount: float,
    message: str,
) -> None:
    item = _fixture(
        tmp_path,
        ledger_amount=ledger_amount,
        settlement_amount=settlement_amount,
    )
    canonical_before = item.settlement_path.read_bytes()
    ledger_before = item.ledger_path.read_bytes()

    with pytest.raises(SettlementUpgradeError, match=message):
        _upgrade(item)

    assert item.settlement_path.read_bytes() == canonical_before
    assert item.ledger_path.read_bytes() == ledger_before
    assert not (item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME).exists()


def test_upgrade_rejects_watchdog_hash_and_exact_path_drift(tmp_path: Path) -> None:
    item = _fixture(tmp_path)
    watchdog = json.loads(item.watchdog_path.read_text(encoding="utf-8"))
    watchdog["stop_reason"] = "tampered"
    write_json(item.watchdog_path, watchdog)
    item.watchdog_path.chmod(0o600)

    with pytest.raises(SettlementUpgradeError, match="watchdog state hash mismatch"):
        _upgrade(item)
    assert not (item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME).exists()

    # Restore the authenticated watchdog, but claim an in-session wrong path.
    legacy = json.loads(item.legacy_raw)
    restored = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "stopped_confirmed",
        "stop_reason": "provider_exit_verified",
    }
    assert stable_hash(restored) == legacy["watchdog_state_hash"]
    write_json(item.watchdog_path, restored)
    item.watchdog_path.chmod(0o600)
    wrong_external = item.session_dir / "external-copy.json"
    wrong_external.write_bytes(item.external_path.read_bytes())
    wrong_external.chmod(0o600)

    with pytest.raises(SettlementUpgradeError, match="path disagrees with the session"):
        upgrade_legacy_gpu_settlement(
            project_root=item.root,
            reservation_receipt_path=item.reservation_path,
            cost_ledger_path=item.ledger_path,
            watchdog_state_path=item.watchdog_path,
            external_stop_receipt_path=wrong_external,
            settlement_path=item.settlement_path,
            limits=_limits(),
        )
    assert not (item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME).exists()


def test_upgrade_rejects_authenticated_external_amount_drift(tmp_path: Path) -> None:
    item = _fixture(tmp_path)
    external = json.loads(item.external_path.read_text(encoding="utf-8"))
    external["settlement_amount_usd"] = 1.8
    external["billing_evidence"]["settlement_amount_usd"] = 1.8
    external["billing_evidence"]["conservative_ceiling_usd"] = 1.8
    external["billing_evidence_hash"] = stable_hash(external["billing_evidence"])
    external["record_hash"] = stable_hash(
        {key: value for key, value in external.items() if key != "record_hash"}
    )
    write_json(item.external_path, external)
    item.external_path.chmod(0o600)

    with pytest.raises(SettlementUpgradeError, match="external-stop settlement amount"):
        _upgrade(item)
    assert not (item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME).exists()


def test_archive_tamper_breaks_repeat_and_completed_session_validation(
    tmp_path: Path,
) -> None:
    item = _fixture(tmp_path)
    _upgrade(item)
    archive = item.session_dir / LEGACY_SETTLEMENT_V1_FILENAME
    archive.write_bytes(archive.read_bytes() + b" \n")
    archive.chmod(0o600)

    with pytest.raises(SettlementUpgradeError, match="existing settlement v2"):
        _upgrade(item)
    with pytest.raises(RunpodSessionError, match="legacy file hash mismatch"):
        validate_completed_runpod_sessions(
            sessions_root=item.session_dir.parent,
            ledger=CostLedger(item.ledger_path, _limits()),
        )


def test_completed_session_validator_rejects_v2_status_drift(tmp_path: Path) -> None:
    item = _fixture(tmp_path)
    _upgrade(item)
    settlement = json.loads(item.settlement_path.read_text(encoding="utf-8"))
    settlement["billing_status"] = "final"
    settlement["record_hash"] = stable_hash(
        {key: value for key, value in settlement.items() if key != "record_hash"}
    )
    write_json(item.settlement_path, settlement)
    item.settlement_path.chmod(0o600)

    with pytest.raises(RunpodSessionError, match="billing_status disagrees"):
        validate_completed_runpod_sessions(
            sessions_root=item.session_dir.parent,
            ledger=CostLedger(item.ledger_path, _limits()),
        )
