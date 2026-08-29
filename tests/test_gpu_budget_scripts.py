from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    load_gpu_phase_budget_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_sessions import (
    GPU_BUDGET_BOOTSTRAP_FILENAME,
    GPU_PREFLIGHT_FILENAME,
    WATCHDOG_PID_FILENAME,
    WATCHDOG_STATE_FILENAME,
    record_watchdog_process_identity,
)

ROOT = Path(__file__).resolve().parents[1]
RESERVE = ROOT / "scripts" / "gpu_budget_reserve.py"
SETTLE = ROOT / "scripts" / "gpu_budget_settle.py"
ACTIVE_VERIFY = ROOT / "scripts" / "runpod_active_session_verify.py"


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


def _environment(session_id: str) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "GPU_BUDGET_SESSION_ID": session_id,
    }


def _reserve_command(tmp_path: Path, *, receipt: Path) -> list[str]:
    return [
        sys.executable,
        str(RESERVE),
        "--cost-ledger",
        str(tmp_path / "cost_ledger.yaml"),
        "--phase",
        "behavior_baseline_gpu",
        "--approved-phase-runtime-hours",
        "2",
        "--approved-phase-maximum-usd",
        "48",
        "--gpu-count",
        "8",
        "--quote-hourly-per-gpu-usd",
        "3",
        "--running-storage-hourly-usd",
        "0",
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--receipt",
        str(receipt),
    ]


def test_reserve_and_settle_scripts_are_private_secret_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / ".runpod" / "reservations" / "baseline.json"
    reserve = subprocess.run(
        _reserve_command(tmp_path, receipt=receipt),
        cwd=tmp_path,
        env=_environment("raw-reserve-session-nonce"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert reserve.returncode == 0, reserve.stderr
    summary = json.loads(reserve.stdout)
    assert summary["passed"] is True
    assert "raw-reserve-session-nonce" not in reserve.stdout
    assert "raw-reserve-session-nonce" not in " ".join(reserve.args)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert "raw-reserve-session-nonce" not in str(receipt_payload)
    ledger = yaml.safe_load((tmp_path / "cost_ledger.yaml").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["status"] == "estimated"
    assert ledger["entries"][0]["amount_usd"] == 48

    session_digest = receipt_payload["session_hash"].removeprefix("sha256:")
    session_dir = tmp_path / ".runpod" / "sessions" / session_digest
    watchdog_state = session_dir / "runpod_watchdog.json"
    write_json(
        watchdog_state,
        {
            "schema_version": 2,
            "watchdog_version": "runpod-gpu-cost-watchdog-v2",
            "status": "stopped_confirmed",
        },
    )
    settlement = session_dir / "settlement.json"
    settle_command = [
        sys.executable,
        str(SETTLE),
        "--reservation-receipt",
        str(receipt),
        "--cost-ledger",
        str(tmp_path / "cost_ledger.yaml"),
        "--watchdog-state",
        str(watchdog_state),
        "--provider-incurred-usd",
        "11.25",
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--output",
        str(settlement),
    ]
    first = subprocess.run(
        settle_command,
        cwd=tmp_path,
        env=_environment("raw-reserve-session-nonce"),
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        settle_command,
        cwd=tmp_path,
        env=_environment("raw-reserve-session-nonce"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert "raw-reserve-session-nonce" not in first.stdout
    settled_ledger = yaml.safe_load((tmp_path / "cost_ledger.yaml").read_text(encoding="utf-8"))
    assert settled_ledger["entries"][0]["status"] == "incurred"
    assert settled_ledger["entries"][0]["amount_usd"] == 11.25

    changed = settle_command.copy()
    changed[changed.index("11.25")] = "12"
    mismatch = subprocess.run(
        changed,
        cwd=tmp_path,
        env=_environment("raw-reserve-session-nonce"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert "different content" in mismatch.stderr


def test_external_stop_settlement_v2_uses_authenticated_ceiling_and_is_idempotent(
    tmp_path: Path,
) -> None:
    session_id = "external-stop-settlement-secret-nonce"
    reservation_path = tmp_path / ".runpod" / "reservations" / "baseline.json"
    reserved = subprocess.run(
        _reserve_command(tmp_path, receipt=reservation_path),
        cwd=tmp_path,
        env=_environment(session_id),
        check=False,
        capture_output=True,
        text=True,
    )
    assert reserved.returncode == 0, reserved.stderr
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    session_dir = (
        tmp_path
        / ".runpod"
        / "sessions"
        / reservation["session_hash"].removeprefix("sha256:")
    )
    session_dir.mkdir(parents=True)
    pod_id_hash = stable_hash({"runpod_pod_id": "private-pod-id"})
    stop_evidence = {
        "desired_status": "EXITED",
        "environment_verified": True,
        "started_at": "2026-08-29T19:24:57.637000Z",
        "exited_at": "2026-08-29T19:41:22Z",
    }
    billing_query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": "2026-08-29T19:24:57.637000Z",
        "end_time": "2026-08-29T19:41:22Z",
    }
    billing = {
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "pod_id_hash": pod_id_hash,
        "provider_amount_usd": None,
        "settlement_amount_usd": 7.48488,
        "time_billed_ms": None,
        "billing_bucket_time": None,
        "provider_billing_row_hash": None,
        "conservative_ceiling_usd": 7.48488,
        "runtime_ceiling_minutes": 17,
    }
    external: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-external-stop-v1",
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-29T20:00:00Z",
        "prior_lifecycle_operation": "created",
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
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "settlement_amount_usd": 7.48488,
        "source_artifact_hashes": [],
    }
    external["record_hash"] = stable_hash(external)
    external_path = session_dir / "external_stop_receipt.json"
    write_json(external_path, external)
    settlement_path = session_dir / "settlement.json"
    command = [
        sys.executable,
        str(SETTLE),
        "--reservation-receipt",
        str(reservation_path),
        "--cost-ledger",
        str(tmp_path / "cost_ledger.yaml"),
        "--external-stop-receipt",
        str(external_path),
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--output",
        str(settlement_path),
    ]

    settlement_environment = _environment(session_id)
    settlement_environment.pop("GPU_BUDGET_SESSION_ID")
    first = subprocess.run(
        command,
        cwd=tmp_path,
        env=settlement_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=tmp_path,
        env=settlement_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["schema_version"] == 2
    assert payload["protocol_version"] == "cumulative-gpu-phase-settlement-v2"
    assert payload["provider_incurred_usd"] is None
    assert payload["accounted_gpu_usd"] == pytest.approx(7.48488)
    assert payload["billing_status"] == "pending"
    assert payload["external_stop_receipt_hash"] == external["record_hash"]
    assert session_id not in first.stdout
    ledger = yaml.safe_load((tmp_path / "cost_ledger.yaml").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["status"] == "incurred"
    assert ledger["entries"][0]["amount_usd"] == pytest.approx(7.48488)


def test_reserve_script_rejects_wrong_maximum_and_nonprivate_receipt(tmp_path: Path) -> None:
    outside = tmp_path / "receipt.json"
    outside_result = subprocess.run(
        _reserve_command(tmp_path, receipt=outside),
        cwd=tmp_path,
        env=_environment("outside-path-session"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert outside_result.returncode != 0
    assert "under ignored .runpod" in outside_result.stderr
    assert not (tmp_path / "cost_ledger.yaml").exists()

    private = tmp_path / ".runpod" / "reservations" / "wrong.json"
    command = _reserve_command(tmp_path, receipt=private)
    command[command.index("48")] = "47"
    wrong_maximum = subprocess.run(
        command,
        cwd=tmp_path,
        env=_environment("wrong-maximum-session"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert wrong_maximum.returncode != 0
    assert "must exactly equal" in wrong_maximum.stderr
    assert not (tmp_path / "cost_ledger.yaml").exists()


def test_reserve_script_rejects_broken_symlink_receipt_before_ledger_write(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / ".runpod" / "reservations" / "claimed.json"
    receipt.parent.mkdir(parents=True)
    receipt.symlink_to(receipt.parent / "missing-target.json")

    completed = subprocess.run(
        _reserve_command(tmp_path, receipt=receipt),
        cwd=tmp_path,
        env=_environment("broken-symlink-session"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "claimed GPU receipt" in completed.stderr
    assert not (tmp_path / "cost_ledger.yaml").exists()


def test_active_session_verifier_authenticates_before_backend_without_nonce_leak(
    tmp_path: Path,
) -> None:
    session_id = "active-verifier-secret-nonce"
    receipt = tmp_path / ".runpod" / "reservations" / "baseline.json"
    reserve = subprocess.run(
        _reserve_command(tmp_path, receipt=receipt),
        cwd=tmp_path,
        env=_environment(session_id),
        check=False,
        capture_output=True,
        text=True,
    )
    assert reserve.returncode == 0, reserve.stderr
    ledger_path = tmp_path / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    reservation = load_gpu_phase_budget_reservation(receipt)
    bootstrap = validate_gpu_phase_bootstrap(
        ledger=ledger,
        reservation=reservation,
        phase="behavior_baseline_gpu",
        session_id=session_id,
        expected_approved_runtime_hours=2,
        expected_live_hourly_total_usd=24,
    )
    session_dir = (
        tmp_path / ".runpod" / "sessions" / reservation.session_hash.removeprefix("sha256:")
    )
    session_dir.mkdir(parents=True)
    write_json(session_dir / GPU_BUDGET_BOOTSTRAP_FILENAME, bootstrap)
    now = datetime.now(UTC)
    watchdog_state = session_dir / WATCHDOG_STATE_FILENAME
    watchdog = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "armed",
        "updated_at": now.isoformat(),
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
    write_json(watchdog_state, watchdog)
    proc_root = _fake_proc(tmp_path)
    process_identity = record_watchdog_process_identity(
        session_dir / WATCHDOG_PID_FILENAME,
        pid=4242,
        required_cmdline_tokens=("scripts/runpod_watchdog.py", "state.json"),
        proc_root=proc_root,
        captured_at=now,
    )
    write_json(
        session_dir / GPU_PREFLIGHT_FILENAME,
        {
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
                "state_path": str(watchdog_state),
                "state_updated_at": now.isoformat(),
            },
            "gpu_budget_reservation": {
                field: bootstrap[field]
                for field in (
                    "reservation_id",
                    "reservation_record_hash",
                    "session_hash",
                    "phase",
                )
            },
        },
    )
    command = [
        sys.executable,
        str(ACTIVE_VERIFY),
        "--session-directory",
        str(session_dir),
        "--reservation-receipt",
        str(receipt),
        "--cost-ledger",
        str(ledger_path),
        "--phase",
        "behavior_baseline_gpu",
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--proc-root",
        str(proc_root),
    ]
    verified = subprocess.run(
        command,
        cwd=tmp_path,
        env=_environment(session_id),
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["passed"] is True
    assert session_id not in verified.stdout
    assert session_id not in " ".join(verified.args)

    missing_nonce_environment = _environment(session_id)
    missing_nonce_environment.pop("GPU_BUDGET_SESSION_ID")
    missing_nonce = subprocess.run(
        command,
        cwd=tmp_path,
        env=missing_nonce_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_nonce.returncode != 0
    assert "environment variable is unset" in missing_nonce.stderr
