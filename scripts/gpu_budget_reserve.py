#!/usr/bin/env python3
"""Create one pre-launch GPU reservation and private authenticated receipt."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    approved_gpu_phase_maximum_usd,
    reserve_gpu_phase_budget,
    write_json_exclusive,
)

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def _private_runpod_path(path: Path) -> Path:
    private_root = (Path.cwd() / ".runpod").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(private_root):
        raise ValueError("GPU reservation receipt must be created under ignored .runpod/")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--session-id-env", default="GPU_BUDGET_SESSION_ID")
    parser.add_argument("--approved-phase-runtime-hours", type=float, required=True)
    parser.add_argument("--approved-phase-maximum-usd", type=float, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--quote-hourly-per-gpu-usd", type=float, required=True)
    parser.add_argument("--safety-margin-fraction", type=float, default=0.03)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.gpu_count != 8:
            raise ValueError("production GPU reservation requires exactly 8 GPUs")
        if _ENV_NAME_RE.fullmatch(args.session_id_env) is None:
            raise ValueError("GPU budget session environment variable name is invalid")
        session_id = os.environ.get(args.session_id_env)
        if not session_id:
            raise ValueError(
                "required opaque GPU budget session environment variable is unset: "
                f"{args.session_id_env}"
            )
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
            for value in (
                args.approved_phase_runtime_hours,
                args.approved_phase_maximum_usd,
                args.quote_hourly_per_gpu_usd,
            )
        ):
            raise ValueError("GPU reservation quote, runtime, and maximum must be positive")
        implied_maximum = approved_gpu_phase_maximum_usd(
            gpu_count=args.gpu_count,
            quote_hourly_per_gpu_usd=args.quote_hourly_per_gpu_usd,
            approved_runtime_hours=args.approved_phase_runtime_hours,
        )
        if abs(implied_maximum - args.approved_phase_maximum_usd) > 1e-6:
            raise ValueError(
                "approved phase maximum must exactly equal 8 GPUs x quote rate x runtime"
            )
        if os.path.lexists(args.receipt):
            raise ValueError(f"refusing to overwrite claimed GPU receipt: {args.receipt}")
        receipt_path = _private_runpod_path(args.receipt)
        reservation = reserve_gpu_phase_budget(
            ledger=CostLedger(
                args.cost_ledger,
                BudgetLimits(
                    gpu=args.gpu_hard_stop_usd,
                    api=args.api_hard_stop_usd,
                    total=args.total_hard_stop_usd,
                ),
            ),
            phase=args.phase,
            session_id=session_id,
            approved_phase_maximum_usd=args.approved_phase_maximum_usd,
            approved_maximum_runtime_hours=args.approved_phase_runtime_hours,
            live_hourly_total_usd=args.gpu_count * args.quote_hourly_per_gpu_usd,
            safety_margin_fraction=args.safety_margin_fraction,
        )
        payload = reservation.manifest()
        write_json_exclusive(receipt_path, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        # The opaque nonce is never interpolated into an error or output.
        raise SystemExit(str(exc)) from exc

    summary = {
        "schema_version": 1,
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": payload["record_hash"],
        "session_hash": reservation.session_hash,
        "maximum_safe_runtime_hours": reservation.maximum_safe_runtime_hours,
        "prior_committed_gpu_usd": reservation.prior_committed_gpu_usd,
        "committed_gpu_after_reservation_usd": (reservation.committed_gpu_after_reservation_usd),
        "receipt": str(receipt_path.relative_to(Path.cwd().resolve())),
        "passed": True,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
