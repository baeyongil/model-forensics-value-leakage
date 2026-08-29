from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import reserve_gpu_phase_budget
from model_forensics.io import write_json

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_budget_preflight.py"


def test_gpu_budget_preflight_uses_env_nonce_and_emits_only_secret_safe_receipt(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="resample_gpu",
        session_id="raw-session-must-not-leak",
        approved_phase_maximum_usd=48,
        approved_maximum_runtime_hours=2,
        live_hourly_total_usd=24,
    )
    receipt_path = tmp_path / "receipt.json"
    output_path = tmp_path / "bootstrap.json"
    write_json(receipt_path, reservation.manifest())
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "GPU_BUDGET_SESSION_ID": "raw-session-must-not-leak",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reservation-receipt",
            str(receipt_path),
            "--cost-ledger",
            str(ledger_path),
            "--phase",
            "resample_gpu",
            "--expected-approved-runtime-hours",
            "2",
            "--expected-live-hourly-total-usd",
            "24",
            "--gpu-hard-stop-usd",
            "220",
            "--api-hard-stop-usd",
            "100",
            "--total-hard-stop-usd",
            "325",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    output = output_path.read_text(encoding="utf-8")
    assert "raw-session-must-not-leak" not in output
    assert "raw-session-must-not-leak" not in " ".join(completed.args)
    assert '"maximum_safe_runtime_hours": 2.0' in output
