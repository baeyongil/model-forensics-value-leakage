#!/usr/bin/env python3
"""Upgrade one settled v1 GPU receipt to authenticated external-stop v2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.budget import BudgetLimits
from model_forensics.settlement_upgrade import upgrade_legacy_gpu_settlement


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the exact legacy settlement and atomically replace it with v2 "
            "external-stop evidence without changing the canonical cost ledger."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reservation-receipt", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--watchdog-state", type=Path, required=True)
    parser.add_argument("--external-stop-receipt", type=Path, required=True)
    parser.add_argument("--settlement", type=Path, required=True)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    args = parser.parse_args()

    try:
        payload = upgrade_legacy_gpu_settlement(
            project_root=args.project_root,
            reservation_receipt_path=args.reservation_receipt,
            cost_ledger_path=args.cost_ledger,
            watchdog_state_path=args.watchdog_state,
            external_stop_receipt_path=args.external_stop_receipt,
            settlement_path=args.settlement,
            limits=BudgetLimits(
                gpu=args.gpu_hard_stop_usd,
                api=args.api_hard_stop_usd,
                total=args.total_hard_stop_usd,
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = {
        "schema_version": payload["schema_version"],
        "protocol_version": payload["protocol_version"],
        "status": payload["status"],
        "billing_status": payload["billing_status"],
        "evidence_kind": payload["evidence_kind"],
        "accounted_gpu_usd": payload["accounted_gpu_usd"],
        "legacy_settlement_v1_file_hash": payload[
            "legacy_settlement_v1_file_hash"
        ],
        "record_hash": payload["record_hash"],
        "passed": True,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
