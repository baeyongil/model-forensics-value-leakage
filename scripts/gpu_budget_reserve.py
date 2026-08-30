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
from model_forensics.execution_bindings import load_gpu_quote_lock
from model_forensics.gpu_budget import (
    approved_gpu_phase_maximum_usd,
    reserve_gpu_phase_budget,
    write_json_exclusive,
)
from model_forensics.paid_bundle_rotation import paid_bundle_lock

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
    parser.add_argument("--gpu-quote-lock", type=Path, required=True)
    parser.add_argument("--safety-margin-fraction", type=float, default=0.03)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    (project_root / ".runpod").mkdir(mode=0o700, exist_ok=True)
    try:
        with paid_bundle_lock(project_root=project_root, exclusive=False):
            quote = load_gpu_quote_lock(args.gpu_quote_lock)
            allocation = next(
                (
                    item
                    for item in quote.phase_runtime_allocations
                    if item.command_phase == args.phase
                ),
                None,
            )
            if allocation is None:
                raise ValueError(f"phase absent from quote lock: {args.phase}")
            gpu_count = quote.gpu_count
            per_gpu_rate = quote.usd_per_gpu_hour
            storage_rate = quote.running_storage_usd_per_hour
            runtime_hours = allocation.maximum_runtime_hours
            phase_maximum = approved_gpu_phase_maximum_usd(
                gpu_count=gpu_count,
                quote_hourly_per_gpu_usd=per_gpu_rate,
                running_storage_hourly_usd=storage_rate,
                approved_runtime_hours=runtime_hours,
            )

            if gpu_count != 8:
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
                for value in (runtime_hours, phase_maximum, per_gpu_rate)
            ):
                raise ValueError("GPU reservation quote, runtime, and maximum must be positive")
            if not math.isfinite(storage_rate) or storage_rate < 0:
                raise ValueError("running storage hourly rate must be finite and nonnegative")
            implied_maximum = approved_gpu_phase_maximum_usd(
                gpu_count=gpu_count,
                quote_hourly_per_gpu_usd=per_gpu_rate,
                running_storage_hourly_usd=storage_rate,
                approved_runtime_hours=runtime_hours,
            )
            if abs(implied_maximum - phase_maximum) > 1e-6:
                raise ValueError(
                    "approved phase maximum must exactly equal all-in compute plus storage x runtime"
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
                approved_phase_maximum_usd=phase_maximum,
                approved_maximum_runtime_hours=runtime_hours,
                live_hourly_total_usd=(gpu_count * per_gpu_rate + storage_rate),
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
