#!/usr/bin/env python3
"""Reserve and settle the explicit RunPod storage allowance."""

from __future__ import annotations

import argparse
import json
import math

from model_forensics.budget import BudgetLimits, CostEntry, CostLedger

STORAGE_RESERVATION_ID = "runpod-local-storage-reserve-v1"
STORAGE_DESCRIPTION = "RunPod container and volume disk across all GPU phases"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reserve", "settle"))
    parser.add_argument("--cost-ledger", required=True)
    parser.add_argument("--amount-usd", type=float, required=True)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.amount_usd) or args.amount_usd < 0:
            raise ValueError("storage amount must be finite and nonnegative")
        if args.mode == "reserve" and abs(args.amount_usd - 5.0) > 1e-9:
            raise ValueError("the preregistered storage reserve must be exactly USD 5")
        ledger = CostLedger(
            args.cost_ledger,
            BudgetLimits(
                gpu=args.gpu_hard_stop_usd,
                api=args.api_hard_stop_usd,
                total=args.total_hard_stop_usd,
            ),
        )
        entry = CostEntry(
            kind="storage",
            amount_usd=args.amount_usd,
            description=STORAGE_DESCRIPTION,
            status="estimated" if args.mode == "reserve" else "incurred",
        )
        if args.mode == "reserve":
            totals = ledger.reserve(STORAGE_RESERVATION_ID, entry)
        else:
            totals = ledger.settle_reservation(STORAGE_RESERVATION_ID, entry)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "schema_version": 1,
                "mode": args.mode,
                "entry_id": STORAGE_RESERVATION_ID,
                "amount_usd": args.amount_usd,
                "totals": totals,
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
