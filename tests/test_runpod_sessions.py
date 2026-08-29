from __future__ import annotations

import json
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
from model_forensics.runpod_recovery import EXTERNAL_STOP_RECEIPT_FILENAME
from model_forensics.runpod_sessions import (
    GPU_BUDGET_BOOTSTRAP_FILENAME,
    GPU_PREFLIGHT_FILENAME,
    SETTLEMENT_FILENAME,
    WATCHDOG_PID_FILENAME,
    WATCHDOG_STATE_FILENAME,
    RunpodSessionError,
    prepare_runpod_session_directory,
    record_watchdog_process_identity,
    validate_active_runpod_session,
    validate_completed_runpod_sessions,
)


def _ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(
        tmp_path / "cost_ledger.yaml",
        BudgetLimits(gpu=220, api=100, total=325),
    )


def _local_gpu_inventory() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": "NVIDIA H100 80GB HBM3",
            "memory_gib": 79.1,
            "uuid": f"GPU-{index}",
            "driver_version": "575.57.08",
            "mig_mode": "Disabled",
        }
        for index in range(8)
    ]


def _fake_proc(
    tmp_path: Path,
    *,
    pid: int = 4242,
    start_ticks: int = 123456,
    argv: tuple[str, ...] = ("python3", "scripts/runpod_watchdog.py", "--state", "state.json"),
) -> Path:
    proc_root = tmp_path / "proc"
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True, exist_ok=True)
    (process_root / "stat").write_text(
        f"{pid} (python3) S " + " ".join(["0"] * 18) + f" {start_ticks}\n",
        encoding="utf-8",
    )
    (process_root / "cmdline").write_bytes(
        b"\0".join(token.encode("utf-8") for token in argv) + b"\0"
    )
    boot_id = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_id.parent.mkdir(parents=True, exist_ok=True)
    boot_id.write_text("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n", encoding="utf-8")
    return proc_root


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


def _complete_external_stop_session(
    *,
    session_dir: Path,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    accounted_usd: float,
) -> None:
    bootstrap = json.loads(
        (session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).read_text(encoding="utf-8")
    )
    pod_id_hash = stable_hash({"runpod_pod_id": "private-pod"})
    stop = {"desired_status": "EXITED", "environment_verified": True}
    query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
    }
    billing = {
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "pod_id_hash": pod_id_hash,
        "provider_amount_usd": None,
        "settlement_amount_usd": accounted_usd,
        "time_billed_ms": None,
        "conservative_ceiling_usd": accounted_usd,
    }
    external = {
        "schema_version": 1,
        "protocol_version": "runpod-external-stop-v1",
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-29T20:00:00Z",
        "prior_lifecycle_operation": "created",
        "lifecycle_before_hash": stable_hash({"state": "before"}),
        "lifecycle_stopped_hash": stable_hash({"state": "stopped"}),
        "session_hash": bootstrap["session_hash"],
        "reservation_id": bootstrap["reservation_id"],
        "reservation_record_hash": bootstrap["reservation_record_hash"],
        "pod_id_hash": pod_id_hash,
        "stop_evidence": stop,
        "stop_evidence_hash": stable_hash(stop),
        "billing_query": query,
        "billing_query_hash": stable_hash(query),
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "settlement_amount_usd": accounted_usd,
        "source_artifact_hashes": [],
    }
    external["record_hash"] = stable_hash(external)
    write_json(session_dir / EXTERNAL_STOP_RECEIPT_FILENAME, external)
    settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=accounted_usd,
    )
    settlement = {
        "schema_version": 2,
        "protocol_version": "cumulative-gpu-phase-settlement-v2",
        "phase": bootstrap["phase"],
        "reservation_id": bootstrap["reservation_id"],
        "reservation_record_hash": bootstrap["reservation_record_hash"],
        "session_hash": bootstrap["session_hash"],
        "provider_incurred_usd": None,
        "accounted_gpu_usd": accounted_usd,
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "external_stop_receipt_hash": external["record_hash"],
        "stop_evidence_hash": external["stop_evidence_hash"],
        "billing_evidence_hash": external["billing_evidence_hash"],
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


def test_external_stop_receipt_and_settlement_v2_authorize_next_phase(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    reservation, pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="external-stop-session",
    )
    session_dir = prepare_runpod_session_directory(
        sessions_root=tmp_path / ".runpod" / "sessions",
        pending_bootstrap_path=pending,
        ledger=ledger,
    )
    _complete_external_stop_session(
        session_dir=session_dir,
        ledger=ledger,
        reservation=reservation,
        accounted_usd=7.48488,
    )
    # The failed bootstrap lived only on the now-stopped remote volume.  The
    # external receipt still closes the local session without fabricating it.
    (session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).unlink()

    summaries = validate_completed_runpod_sessions(
        sessions_root=session_dir.parent,
        ledger=ledger,
    )
    assert summaries[0]["status"] == "stopped_confirmed_and_settled"

    _next, next_pending = _reserve_pending(
        tmp_path=tmp_path,
        ledger=ledger,
        phase="behavior_retry_gpu",
        session_id="external-stop-retry-session",
    )
    next_dir = prepare_runpod_session_directory(
        sessions_root=session_dir.parent,
        pending_bootstrap_path=next_pending,
        ledger=ledger,
    )
    assert next_dir != session_dir


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
        "live_metadata": {
            "provider_api": "rest-v1",
            "pod_id": "pod_123",
            "provider_evidence_unavailable": [
                "cuda_version",
                "global_networking_enabled",
                "interruptible",
                "locked",
                "runtime_gpu_count",
            ],
            "cuda_version": None,
            "global_networking_enabled": None,
            "interruptible": None,
            "locked": None,
            "runtime_gpu_count": None,
            "execution_identity_hash": "sha256:" + "a" * 64,
            "machine_id_hash": "sha256:" + "b" * 64,
            "direct_ssh_endpoint_hash": "sha256:" + "c" * 64,
            "provider_gpu_id": "NVIDIA H100 80GB HBM3",
            "data_center_id": "CA-MTL-1",
            "container_image": "runpod/pytorch@sha256:" + "d" * 64,
            "ssh_ready": True,
            "direct_ssh_ready": True,
            "environment_verified": True,
            "network_volume_attached": False,
        },
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
    proc_root = _fake_proc(tmp_path)
    process_identity = record_watchdog_process_identity(
        session_dir / WATCHDOG_PID_FILENAME,
        pid=4242,
        required_cmdline_tokens=("scripts/runpod_watchdog.py", "state.json"),
        proc_root=proc_root,
        captured_at=now,
    )
    bootstrap = json.loads(
        (session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME).read_text(encoding="utf-8")
    )
    preflight = {
        "schema_version": 3,
        "passed": True,
        "planned_hours": 2.0,
        "prior_committed_gpu_cost_usd": 0.0,
        "gpu_budget_usd": 220.0,
        "pod_id": "pod_123",
        "gpus": _local_gpu_inventory(),
        "allowed_cuda_versions": ["12.8"],
        "cuda_compatibility": {
            "required_environment": "VLLM_ENABLE_CUDA_COMPATIBILITY=1",
            "compatibility_directory": "/usr/local/cuda-13.0/compat",
            "required_libraries": [
                "libcuda.so.1",
                "libnvidia-ptxjitcompiler.so.1",
            ],
        },
        "evidence_boundary": {
            "provider_api": "rest-v1",
            "provider_evidence_unavailable": [
                "cuda_version",
                "global_networking_enabled",
                "interruptible",
                "locked",
                "runtime_gpu_count",
            ],
            "locally_verified_substitutes": {
                "runtime_gpu_count": 8,
                "runtime_gpu_source": "nvidia-smi",
                "cuda_forward_compatibility": True,
                "cuda_source": "local-driver-and-compatibility-libraries",
            },
            "approval_bound_but_not_live_provider_verified": [
                "global_networking_enabled",
                "interruptible",
                "locked",
            ],
        },
        "execution_identity_hash": "sha256:" + "a" * 64,
        "machine_id_hash": "sha256:" + "b" * 64,
        "direct_ssh_endpoint_hash": "sha256:" + "c" * 64,
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "data_center_id": "CA-MTL-1",
        "container_image_digest": "runpod/pytorch@sha256:" + "d" * 64,
        "price": {"approved_hourly_total_usd": 24.0},
        "watchdog": {
            "pid": 4242,
            "process_identity_hash": process_identity["record_hash"],
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
        proc_root=proc_root,
    )
    assert gate["passed"] is True
    assert session_id not in str(gate)

    process_cmdline = proc_root / "4242" / "cmdline"
    original_cmdline = process_cmdline.read_bytes()
    process_cmdline.write_bytes(b"python3\0unrelated.py\0")
    with pytest.raises(RunpodSessionError, match="identity changed"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
            proc_root=proc_root,
        )
    process_cmdline.write_bytes(original_cmdline)

    process_stat = proc_root / "4242" / "stat"
    original_stat = process_stat.read_text(encoding="utf-8")
    process_stat.write_text(original_stat.replace("123456", "123457"), encoding="utf-8")
    with pytest.raises(RunpodSessionError, match="PID was reused"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
            proc_root=proc_root,
        )
    process_stat.write_text(original_stat, encoding="utf-8")

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
        proc_root=proc_root,
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
            proc_root=proc_root,
        )

    future_watchdog = {
        **watchdog,
        "updated_at": (now + timedelta(microseconds=1)).isoformat(),
    }
    write_json(session_dir / WATCHDOG_STATE_FILENAME, future_watchdog)
    with pytest.raises(RunpodSessionError, match="future"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
            proc_root=proc_root,
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
            proc_root=proc_root,
        )
    write_json(session_dir / WATCHDOG_STATE_FILENAME, watchdog)

    missing_boundary = {key: value for key, value in preflight.items() if key != "evidence_boundary"}
    write_json(session_dir / GPU_PREFLIGHT_FILENAME, missing_boundary)
    with pytest.raises(RunpodSessionError, match="evidence boundary"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
            proc_root=proc_root,
        )
    write_json(session_dir / GPU_PREFLIGHT_FILENAME, preflight)

    duplicated_local_gpu = json.loads(json.dumps(preflight))
    duplicated_local_gpu["gpus"][0]["uuid"] = duplicated_local_gpu["gpus"][1]["uuid"]
    write_json(session_dir / GPU_PREFLIGHT_FILENAME, duplicated_local_gpu)
    with pytest.raises(RunpodSessionError, match="UUID evidence"):
        validate_active_runpod_session(
            session_directory=session_dir,
            ledger=ledger,
            reservation=reservation,
            phase=phase,
            session_id=session_id,
            now=now,
            proc_root=proc_root,
        )
    write_json(session_dir / GPU_PREFLIGHT_FILENAME, preflight)

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
            proc_root=proc_root,
        )
