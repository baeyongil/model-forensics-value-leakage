#!/usr/bin/env python3
"""Authenticate the current private GPU session before a paid backend starts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import load_gpu_phase_budget_reservation
from model_forensics.runpod_sessions import validate_active_runpod_session

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-directory", type=Path, required=True)
    parser.add_argument("--reservation-receipt", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    args = parser.parse_args()

    private_root = (Path.cwd() / ".runpod").resolve()
    for path in (args.session_directory.resolve(), args.reservation_receipt.resolve()):
        if not path.is_relative_to(private_root):
            raise SystemExit("active GPU session artifacts must remain under ignored .runpod/")
    if _ENV_NAME_RE.fullmatch(args.session_id_env) is None:
        raise SystemExit("GPU budget session environment variable name is invalid")
    session_id = os.environ.get(args.session_id_env)
    if not session_id:
        raise SystemExit(
            "required opaque GPU budget session environment variable is unset: "
            f"{args.session_id_env}"
        )
    try:
        payload = validate_active_runpod_session(
            session_directory=args.session_directory,
            ledger=CostLedger(
                args.cost_ledger,
                BudgetLimits(
                    gpu=args.gpu_hard_stop_usd,
                    api=args.api_hard_stop_usd,
                    total=args.total_hard_stop_usd,
                ),
            ),
            reservation=load_gpu_phase_budget_reservation(args.reservation_receipt),
            phase=args.phase,
            session_id=session_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
