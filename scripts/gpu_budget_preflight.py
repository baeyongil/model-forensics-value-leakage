#!/usr/bin/env python3
"""Validate a pre-created cumulative GPU reservation without network access."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    load_gpu_phase_budget_reservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import write_json

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservation-receipt", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--expected-approved-runtime-hours", type=float, required=True)
    parser.add_argument("--expected-live-hourly-total-usd", type=float, required=True)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if _ENV_NAME_RE.fullmatch(args.session_id_env) is None:
        raise SystemExit("GPU budget session environment variable name is invalid")
    session_id = os.environ.get(args.session_id_env)
    if not session_id:
        raise SystemExit(
            f"required opaque GPU budget session environment variable is unset: "
            f"{args.session_id_env}"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite GPU budget preflight: {args.output}")
    try:
        reservation = load_gpu_phase_budget_reservation(args.reservation_receipt)
        payload = validate_gpu_phase_bootstrap(
            ledger=CostLedger(
                args.cost_ledger,
                BudgetLimits(
                    gpu=args.gpu_hard_stop_usd,
                    api=args.api_hard_stop_usd,
                    total=args.total_hard_stop_usd,
                ),
            ),
            reservation=reservation,
            phase=args.phase,
            session_id=session_id,
            expected_approved_runtime_hours=args.expected_approved_runtime_hours,
            expected_live_hourly_total_usd=args.expected_live_hourly_total_usd,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        # Validation errors never include the raw session nonce.
        raise SystemExit(str(exc)) from exc
    write_json(args.output, payload)
    try:
        args.output.chmod(0o600)
    except OSError:  # pragma: no cover - platform permission model
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
