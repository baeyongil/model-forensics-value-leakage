#!/usr/bin/env python3
"""Materialize the exact claim-safe host-to-Pod bootstrap file set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.budget import BudgetLimits
from model_forensics.runpod_sync import (
    RunpodSyncError,
    build_selective_sync_plan,
    materialize_selective_sync_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument(
        "--cost-ledger",
        type=Path,
        default=Path("data/manifests/cost_ledger.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = build_selective_sync_plan(
            project_root=args.project_root,
            phase=args.phase,
            reservation_path=args.reservation,
            cost_ledger_path=args.cost_ledger,
            limits=BudgetLimits(gpu=220, api=100, total=325),
        )
        output = materialize_selective_sync_bundle(
            project_root=args.project_root,
            destination=args.output,
            plan=plan,
        )
    except (OSError, RuntimeError, ValueError, RunpodSyncError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = {
        "schema_version": 1,
        "protocol_version": plan["protocol_version"],
        "phase": plan["phase"],
        "file_count": len(plan["files"]),
        "current_host_session_excluded": plan["current_host_session_excluded"],
        "record_hash": plan["record_hash"],
        "output": str(output.relative_to(Path(args.project_root).resolve())),
        "passed": True,
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
