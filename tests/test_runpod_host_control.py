from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import model_forensics.runpod_host_control as host_control
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import reserve_gpu_phase_budget
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_host_control import (
    HOST_STOP_REQUEST_FILENAME,
    HOST_WATCHDOG_FILENAME,
    RunpodHostControlError,
    request_host_stop,
    validate_host_stop_confirmation,
)
from model_forensics.runpod_watchdog import HOST_REARM_ACK_PROTOCOL

PHASE = "behavior_baseline_gpu"
POD_ID = "research_pod_123"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    ledger_path = tmp_path / "data" / "manifests" / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase=PHASE,
        session_id="private-host-control-session",
        approved_phase_maximum_usd=48,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=24,
    )
    reservation_path = tmp_path / ".runpod" / "reservations" / f"{PHASE}.json"
    write_json(reservation_path, reservation.manifest())
    reservation_path.chmod(0o600)
    session = tmp_path / ".runpod" / "sessions" / reservation.session_hash.removeprefix("sha256:")
    session.mkdir(parents=True)
    session.chmod(0o700)

    immutable_spec = {"gpu": {"count": 8, "id": "NVIDIA H100 80GB HBM3"}}
    authorization: dict[str, object] = {
        "phase": PHASE,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
        "approval_hash": stable_hash({"approval": 1}),
        "bindings_hash": stable_hash({"bindings": 1}),
        "gpu_lock_hash": stable_hash({"gpu_lock": 1}),
        "quote_hash": stable_hash({"quote": 1}),
        "immutable_spec_hash": stable_hash(immutable_spec),
        "launch_spec_hash": stable_hash({"launch": 1}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 2.0,
        "approved_phase_maximum_usd": 48.0,
        "live_hourly_total_usd": 24.0,
    }
    lifecycle: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-pod-lifecycle-v1",
        "operation": "rearmed",
        "updated_at": "2026-08-30T12:00:00Z",
        "immutable_spec": immutable_spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": POD_ID, "name": "research", "status": "RUNNING"},
    }
    lifecycle["record_hash"] = stable_hash(lifecycle)
    write_json(tmp_path / ".runpod" / "pod_lifecycle.json", lifecycle)

    now = datetime.now(UTC)
    state: dict[str, object] = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "pod_id": POD_ID,
        "status": "armed",
        "armed_at": (now - timedelta(minutes=1)).isoformat(),
        "updated_at": (now - timedelta(seconds=2)).isoformat(),
        "live_metadata": {"provider_api": "rest-v1", "pod_id": POD_ID},
        "limits": {
            "gpu_hard_stop_usd": reservation.global_gpu_hard_stop_usd,
            "global_safe_budget_usd": reservation.safety_adjusted_gpu_ceiling_usd,
            "safe_budget_usd": reservation.remaining_safe_gpu_before_phase_usd,
            "safety_margin_fraction": reservation.safety_margin_fraction,
            "maximum_runtime_hours": reservation.maximum_safe_runtime_hours,
            "maximum_approved_hourly_total_usd": 24.0,
            "maximum_approved_compute_hourly_usd": 23.95,
            "maximum_approved_storage_hourly_usd": 0.05,
            "prior_committed_gpu_usd": reservation.prior_committed_gpu_usd,
        },
        "deadline": {"effective_deadline": (now + timedelta(hours=1)).isoformat()},
        "stop_reason": None,
        "action": "stop_only_preserve_volume",
        "deletion": "manual_after_verified_sync",
        "error": None,
    }
    write_json(session / HOST_WATCHDOG_FILENAME, state)
    process_identity = stable_hash({"live_host_watcher": os.getpid()})
    acknowledgement: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": HOST_REARM_ACK_PROTOCOL,
        "status": "armed_and_provider_exited_verified",
        "expected_session_hash": reservation.session_hash,
        "expected_phase": PHASE,
        "lifecycle_before_hash": stable_hash({"before": 1}),
        "pod_id_hash": stable_hash({"runpod_pod_id": POD_ID}),
        "watcher_pid": os.getpid(),
        "watcher_process_identity_hash": process_identity,
        "acknowledged_at": now.isoformat(),
    }
    acknowledgement["record_hash"] = stable_hash(acknowledgement)
    write_json(session / "host_rearm_watchdog_ack.json", acknowledgement)
    return reservation_path, session, ledger_path, state


def test_request_then_validate_canonical_host_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation, session, _ledger, state = _fixture(tmp_path)
    acknowledgement = json.loads(
        (session / "host_rearm_watchdog_ack.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        host_control,
        "_host_process_identity_hash",
        lambda _pid: acknowledgement["watcher_process_identity_hash"],
    )

    request = request_host_stop(
        project_root=tmp_path,
        phase=PHASE,
        reservation_path=reservation,
    )
    assert request == session / HOST_STOP_REQUEST_FILENAME
    assert request.stat().st_size == 0
    assert (
        request_host_stop(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
        )
        == request
    )

    stopped = {
        **state,
        "status": "stopped_confirmed",
        "updated_at": datetime.now(UTC).isoformat(),
        "stop_reason": "external_stop_request",
    }
    write_json(session / HOST_WATCHDOG_FILENAME, stopped)
    assert (
        validate_host_stop_confirmation(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
            watchdog_path=session / HOST_WATCHDOG_FILENAME,
            stop_request_path=request,
        )
        == stopped
    )


def test_request_rejects_stale_or_dead_watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation, session, _ledger, state = _fixture(tmp_path)
    stale = {
        **state,
        "updated_at": (datetime.now(UTC) - timedelta(seconds=21)).isoformat(),
    }
    write_json(session / HOST_WATCHDOG_FILENAME, stale)
    with pytest.raises(RunpodHostControlError, match="stale"):
        request_host_stop(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
        )

    write_json(session / HOST_WATCHDOG_FILENAME, state)
    monkeypatch.setattr(
        host_control,
        "_host_process_identity_hash",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("dead")),
    )
    with pytest.raises(RunpodHostControlError, match="not live"):
        request_host_stop(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
        )


def test_confirmation_rejects_remote_or_noncanonical_stop(
    tmp_path: Path,
) -> None:
    reservation, session, _ledger, state = _fixture(tmp_path)
    (session / HOST_STOP_REQUEST_FILENAME).touch(mode=0o600)
    stopped = {
        **state,
        "status": "stopped_confirmed",
        "stop_reason": "stopped_outside_watchdog",
    }
    write_json(session / HOST_WATCHDOG_FILENAME, stopped)
    with pytest.raises(RunpodHostControlError, match="canonical local stop request"):
        validate_host_stop_confirmation(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
            watchdog_path=session / HOST_WATCHDOG_FILENAME,
            stop_request_path=session / HOST_STOP_REQUEST_FILENAME,
        )
    with pytest.raises(RunpodHostControlError, match="not canonical"):
        validate_host_stop_confirmation(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
            watchdog_path=tmp_path / "other.json",
            stop_request_path=session / HOST_STOP_REQUEST_FILENAME,
        )


def test_request_rejects_split_rate_drift_and_symlink_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation, session, _ledger, state = _fixture(tmp_path)
    acknowledgement = json.loads(
        (session / "host_rearm_watchdog_ack.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        host_control,
        "_host_process_identity_hash",
        lambda _pid: acknowledgement["watcher_process_identity_hash"],
    )
    limits = dict(state["limits"])
    limits["maximum_approved_compute_hourly_usd"] = 20.0
    write_json(session / HOST_WATCHDOG_FILENAME, {**state, "limits": limits})
    with pytest.raises(RunpodHostControlError, match="compute/storage"):
        request_host_stop(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
        )

    write_json(session / HOST_WATCHDOG_FILENAME, state)
    target = tmp_path / "outside"
    target.touch()
    (session / HOST_STOP_REQUEST_FILENAME).symlink_to(target)
    with pytest.raises(RunpodHostControlError, match="missing or unsafe"):
        request_host_stop(
            project_root=tmp_path,
            phase=PHASE,
            reservation_path=reservation,
        )
