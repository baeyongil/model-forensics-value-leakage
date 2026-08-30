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
from model_forensics.execution_bindings import gpu_quote_lock_content_hash
from model_forensics.gpu_budget import (
    load_gpu_phase_budget_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import stable_hash, write_json
from model_forensics.paid_bundle_rotation import paid_bundle_lock
from model_forensics.runpod_sessions import (
    GPU_BUDGET_BOOTSTRAP_FILENAME,
    GPU_PREFLIGHT_FILENAME,
    WATCHDOG_PID_FILENAME,
    WATCHDOG_STATE_FILENAME,
    record_watchdog_process_identity,
    validate_completed_runpod_sessions,
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


def _ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "manifests" / "cost_ledger.yaml"


def _reserve_command(tmp_path: Path, *, receipt: Path) -> list[str]:
    quote_path = tmp_path / ".runpod" / "gpu_quote_lock.json"
    quote: dict[str, object] = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "gpu-budget-script-fixture",
        "gpu_family": "H100_80GB",
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["CA-MTL-1"],
        "gpu_count": 8,
        "container_disk_gb": 50,
        "volume_disk_gb": 650,
        "usd_per_gpu_hour": 3.0,
        "running_storage_usd_per_hour": 0.1,
        "quoted_at": "2026-08-29T20:00:00Z",
        "phase_runtime_allocations": [
            {"command_phase": phase, "maximum_runtime_hours": 2.0}
            for phase in (
                "behavior_baseline_gpu",
                "behavior_treatment_gpu",
                "resample_gpu",
                "lens_gpu",
            )
        ],
        "source_url": "https://www.runpod.io/pricing",
    }
    quote["content_hash"] = gpu_quote_lock_content_hash(quote)
    write_json(quote_path, quote)
    return [
        sys.executable,
        str(RESERVE),
        "--cost-ledger",
        str(_ledger_path(tmp_path)),
        "--phase",
        "behavior_baseline_gpu",
        "--gpu-quote-lock",
        str(quote_path),
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--receipt",
        str(receipt),
    ]


def test_reserve_is_secret_safe_and_legacy_watchdog_settlement_is_disabled(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
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
    ledger = yaml.safe_load(_ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert ledger["entries"][0]["status"] == "estimated"
    assert ledger["entries"][0]["amount_usd"] == pytest.approx(48.2)

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
        str(_ledger_path(tmp_path)),
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
    assert first.returncode == second.returncode != 0
    assert "legacy watchdog/provider-amount settlement is disabled" in first.stderr
    assert "raw-reserve-session-nonce" not in first.stdout
    settled_ledger = yaml.safe_load(_ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert settled_ledger["entries"][0]["status"] == "estimated"
    assert settled_ledger["entries"][0]["amount_usd"] == pytest.approx(48.2)

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
    assert "legacy watchdog/provider-amount settlement is disabled" in mismatch.stderr


def test_gpu_reservation_cannot_race_paid_bundle_rotation(tmp_path: Path) -> None:
    receipt = tmp_path / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
    command = _reserve_command(tmp_path, receipt=receipt)

    with paid_bundle_lock(project_root=tmp_path, exclusive=True):
        attempted = subprocess.run(
            command,
            cwd=tmp_path,
            env=_environment("rotation-contention-session-nonce"),
            check=False,
            capture_output=True,
            text=True,
        )

    assert attempted.returncode != 0
    assert "already held" in attempted.stderr
    assert not receipt.exists()
    assert not _ledger_path(tmp_path).exists()


def test_external_stop_settlement_v2_uses_authenticated_ceiling_and_is_idempotent(
    tmp_path: Path,
) -> None:
    session_id = "external-stop-settlement-secret-nonce"
    reservation_path = tmp_path / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
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
        tmp_path / ".runpod" / "sessions" / reservation["session_hash"].removeprefix("sha256:")
    )
    session_dir.mkdir(parents=True)
    immutable_spec = {"gpu": {"count": 8, "id": "NVIDIA H100 80GB HBM3"}}
    authorization = {
        "phase": reservation["phase"],
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "session_hash": reservation["session_hash"],
        "approval_hash": stable_hash({"approval": 1}),
        "bindings_hash": stable_hash({"bindings": 1}),
        "gpu_lock_hash": stable_hash({"gpu_lock": 1}),
        "quote_hash": stable_hash({"quote": 1}),
        "immutable_spec_hash": stable_hash(immutable_spec),
        "launch_spec_hash": stable_hash({"launch": 1}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 2.0,
        "approved_phase_maximum_usd": 48.2,
        "live_hourly_total_usd": 24.1,
    }
    lifecycle: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-pod-lifecycle-v1",
        "operation": "stopped",
        "updated_at": "2026-08-29T20:00:00Z",
        "immutable_spec": immutable_spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": "private-pod-id", "name": "research", "status": "EXITED"},
    }
    lifecycle["record_hash"] = stable_hash(lifecycle)
    lifecycle_path = tmp_path / ".runpod" / "pod_lifecycle.json"
    write_json(lifecycle_path, lifecycle)
    pod_id_hash = stable_hash({"runpod_pod_id": "private-pod-id"})
    stop_evidence = {
        "desired_status": "EXITED",
        "environment_verified": True,
        "started_at": "2026-08-29T19:24:57.637000Z",
        "exited_at": "2026-08-29T19:41:22Z",
        "runtime_ms": 984363,
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
        "lifecycle_stopped_hash": lifecycle["record_hash"],
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
        str(_ledger_path(tmp_path)),
        "--external-stop-receipt",
        str(external_path),
        "--lifecycle-state",
        str(lifecycle_path),
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
    ledger = yaml.safe_load(_ledger_path(tmp_path).read_text(encoding="utf-8"))
    assert ledger["entries"][0]["status"] == "incurred"
    assert ledger["entries"][0]["amount_usd"] == pytest.approx(7.48488)
    completed = validate_completed_runpod_sessions(
        sessions_root=tmp_path / ".runpod" / "sessions",
        ledger=CostLedger(
            _ledger_path(tmp_path),
            BudgetLimits(gpu=220, api=100, total=325),
        ),
    )
    assert len(completed) == 1
    assert completed[0]["session_hash"] == reservation["session_hash"]
    assert completed[0]["reservation_id"] == reservation["reservation_id"]
    assert completed[0]["status"] == "stopped_confirmed_and_settled"
    assert completed[0]["settlement_record_hash"] == payload["record_hash"]

    lifecycle["updated_at"] = "2026-08-29T20:00:01Z"
    lifecycle.pop("record_hash")
    lifecycle["record_hash"] = stable_hash(lifecycle)
    write_json(lifecycle_path, lifecycle)
    drifted = subprocess.run(
        command,
        cwd=tmp_path,
        env=settlement_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert drifted.returncode != 0
    assert "stopped lifecycle bindings disagree" in drifted.stderr


def test_no_start_settlement_reconciles_estimate_to_authenticated_zero(
    tmp_path: Path,
) -> None:
    session_id = "no-start-settlement-secret-nonce"
    reservation_path = tmp_path / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
    reserved = subprocess.run(
        _reserve_command(tmp_path, receipt=reservation_path),
        cwd=tmp_path,
        env=_environment(session_id),
        check=False,
        capture_output=True,
        text=True,
    )
    assert reserved.returncode == 0, reserved.stderr
    reservation = json.loads(reservation_path.read_text())
    session_dir = (
        tmp_path / ".runpod" / "sessions" / reservation["session_hash"].removeprefix("sha256:")
    )
    session_dir.mkdir(parents=True)
    immutable_spec = {"gpu": {"count": 8, "id": "NVIDIA H100 80GB HBM3"}}
    authorization = {
        "phase": reservation["phase"],
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "session_hash": reservation["session_hash"],
        "approval_hash": stable_hash({"approval": "no-start"}),
        "bindings_hash": stable_hash({"bindings": "no-start"}),
        "gpu_lock_hash": stable_hash({"gpu_lock": "no-start"}),
        "quote_hash": stable_hash({"quote": "no-start"}),
        "immutable_spec_hash": stable_hash(immutable_spec),
        "launch_spec_hash": stable_hash({"launch": "no-start"}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 2.0,
        "approved_phase_maximum_usd": 48.2,
        "live_hourly_total_usd": 24.1,
    }
    lifecycle_before: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-pod-lifecycle-v1",
        "operation": "rearm_patched",
        "updated_at": "2026-08-30T00:01:00Z",
        "immutable_spec": immutable_spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": "private-no-start-pod", "name": "research", "status": "EXITED"},
    }
    lifecycle_before["record_hash"] = stable_hash(lifecycle_before)
    stopped_lifecycle = {
        **lifecycle_before,
        "operation": "stopped",
        "updated_at": "2026-08-30T00:10:00Z",
        "pod": {"id": "private-no-start-pod", "name": "research", "status": "EXITED"},
    }
    stopped_lifecycle.pop("record_hash")
    stopped_lifecycle["record_hash"] = stable_hash(stopped_lifecycle)
    lifecycle_path = tmp_path / ".runpod" / "pod_lifecycle.json"
    write_json(lifecycle_path, lifecycle_before)
    pod_id_hash = stable_hash({"runpod_pod_id": "private-no-start-pod"})
    observation = {
        "desired_status": "EXITED",
        "pod_id_hash": pod_id_hash,
        "name_hash": stable_hash({"runpod_pod_name": "private-name"}),
        "image_hash": stable_hash({"runpod_image": "private-image"}),
        "machine_id_hash": stable_hash({"runpod_machine_id": "private-machine"}),
        "provider_binding_hash": stable_hash({"provider": "binding"}),
        "immutable_spec_hash": stable_hash({"immutable": "spec"}),
        "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 8},
        "cloud": "SECURE",
        "data_center_id": "CA-MTL-1",
        "container_disk_gb": 50,
        "persistent_disk_gb": 650,
        "persistent_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "environment_verified": True,
        "environment_session_context": "current",
        "pre_start_last_started_at": "2026-08-29T23:00:00Z",
        "observed_last_started_at": "2026-08-29T23:00:00Z",
        "last_started_at_unchanged": True,
        "provider_hourly_compute_usd": 24.0,
        "approved_hourly_all_in_usd": 24.5,
    }
    provider = {
        **observation,
        "observation_count": 1,
        "quiet_window_seconds": 0.0,
        "first_observation_hash": stable_hash(observation),
        "second_observation_hash": None,
    }
    query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": "2026-08-30T00:01:00Z",
        "end_time": "2026-08-30T00:10:00Z",
    }
    billing = {"row_count": 0, "response_hash": stable_hash([])}
    no_start: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-no-start-v1",
        "status": "no_start_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-30T00:10:00Z",
        "prior_lifecycle_operation": "rearm_patched",
        "lifecycle_before_hash": lifecycle_before["record_hash"],
        "lifecycle_stopped_hash": stopped_lifecycle["record_hash"],
        "session_hash": reservation["session_hash"],
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "pod_id_hash": pod_id_hash,
        "provider_evidence": provider,
        "provider_evidence_hash": stable_hash(provider),
        "billing_query": query,
        "billing_query_hash": stable_hash(query),
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "accounted_gpu_usd": 0.0,
    }
    no_start["record_hash"] = stable_hash(no_start)
    no_start_path = session_dir / "no_start_receipt.json"
    write_json(no_start_path, no_start)
    settlement_path = session_dir / "settlement.json"
    command = [
        sys.executable,
        str(SETTLE),
        "--reservation-receipt",
        str(reservation_path),
        "--cost-ledger",
        str(_ledger_path(tmp_path)),
        "--no-start-receipt",
        str(no_start_path),
        "--lifecycle-state",
        str(lifecycle_path),
        "--gpu-hard-stop-usd",
        "220",
        "--api-hard-stop-usd",
        "100",
        "--total-hard-stop-usd",
        "325",
        "--output",
        str(settlement_path),
    ]
    environment = _environment(session_id)
    environment.pop("GPU_BUDGET_SESSION_ID")

    crash_window = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert crash_window.returncode != 0
    assert "stopped lifecycle bindings disagree" in crash_window.stderr
    unsettled = yaml.safe_load(_ledger_path(tmp_path).read_text())
    assert unsettled["entries"][0]["status"] == "estimated"
    write_json(lifecycle_path, stopped_lifecycle)

    first = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["accounted_gpu_usd"] == 0.0
    assert payload["provider_incurred_usd"] == 0.0
    assert payload["billing_status"] == "not_started"
    assert payload["evidence_kind"] == "provider_no_start"
    assert payload["no_start_receipt_hash"] == no_start["record_hash"]
    ledger = yaml.safe_load(_ledger_path(tmp_path).read_text())
    assert ledger["entries"][0]["status"] == "incurred"
    assert ledger["entries"][0]["amount_usd"] == 0.0


def test_reserve_script_rejects_tampered_quote_and_nonprivate_receipt(tmp_path: Path) -> None:
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
    assert not _ledger_path(tmp_path).exists()

    private = tmp_path / ".runpod" / "reservations" / "wrong.json"
    command = _reserve_command(tmp_path, receipt=private)
    quote_path = Path(command[command.index("--gpu-quote-lock") + 1])
    quote = json.loads(quote_path.read_text(encoding="utf-8"))
    quote["phase_runtime_allocations"][0]["maximum_runtime_hours"] = 1.0
    write_json(quote_path, quote)
    tampered_quote = subprocess.run(
        command,
        cwd=tmp_path,
        env=_environment("wrong-maximum-session"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert tampered_quote.returncode != 0
    assert "content hash" in tampered_quote.stderr
    assert not _ledger_path(tmp_path).exists()


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
    assert not _ledger_path(tmp_path).exists()


def test_active_session_verifier_authenticates_before_backend_without_nonce_leak(
    tmp_path: Path,
) -> None:
    session_id = "active-verifier-secret-nonce"
    receipt = tmp_path / ".runpod" / "reservations" / "behavior_baseline_gpu.json"
    reserve = subprocess.run(
        _reserve_command(tmp_path, receipt=receipt),
        cwd=tmp_path,
        env=_environment(session_id),
        check=False,
        capture_output=True,
        text=True,
    )
    assert reserve.returncode == 0, reserve.stderr
    ledger_path = _ledger_path(tmp_path)
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    reservation = load_gpu_phase_budget_reservation(receipt)
    bootstrap = validate_gpu_phase_bootstrap(
        ledger=ledger,
        reservation=reservation,
        phase="behavior_baseline_gpu",
        session_id=session_id,
        expected_approved_runtime_hours=2,
        expected_live_hourly_total_usd=24.1,
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
                "maximum_approved_hourly_total_usd": 24.1,
            "prior_committed_gpu_usd": 0.0,
        },
        "deadline": {
            "effective_deadline": (now + timedelta(hours=1)).isoformat(),
            "calculation_hourly_usd": 24.1,
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
            "price": {"approved_hourly_total_usd": 24.1},
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
