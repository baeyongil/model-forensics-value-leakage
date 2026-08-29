from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    load_gpu_phase_budget_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import write_json
from model_forensics.runpod_sessions import (
    GPU_BUDGET_BOOTSTRAP_FILENAME,
    GPU_PREFLIGHT_FILENAME,
    WATCHDOG_PID_FILENAME,
    WATCHDOG_STATE_FILENAME,
)

ROOT = Path(__file__).resolve().parents[1]
RESERVE = ROOT / "scripts" / "gpu_budget_reserve.py"
SETTLE = ROOT / "scripts" / "gpu_budget_settle.py"
ACTIVE_VERIFY = ROOT / "scripts" / "runpod_active_session_verify.py"


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
    (session_dir / WATCHDOG_PID_FILENAME).write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_json(
        session_dir / GPU_PREFLIGHT_FILENAME,
        {
            "schema_version": 3,
            "passed": True,
            "planned_hours": 2.0,
            "prior_committed_gpu_cost_usd": 0.0,
            "gpu_budget_usd": 220.0,
            "price": {"approved_hourly_total_usd": 24.0},
            "watchdog": {
                "pid": os.getpid(),
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
