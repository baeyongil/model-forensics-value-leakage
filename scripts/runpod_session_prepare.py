#!/usr/bin/env python3
"""Claim a private RunPod phase directory after prior phases are complete."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.runpod_sessions import prepare_runpod_session_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--pending-budget-bootstrap", type=Path, required=True)
    parser.add_argument("--cost-ledger", type=Path, required=True)
    parser.add_argument("--gpu-hard-stop-usd", type=float, required=True)
    parser.add_argument("--api-hard-stop-usd", type=float, required=True)
    parser.add_argument("--total-hard-stop-usd", type=float, required=True)
    args = parser.parse_args()

    private_root = (Path.cwd() / ".runpod").resolve()
    for path, label in (
        (args.sessions_root.resolve(), "sessions root"),
        (args.pending_budget_bootstrap.resolve(), "pending budget bootstrap"),
    ):
        if not path.is_relative_to(private_root):
            raise SystemExit(f"RunPod {label} must remain under ignored .runpod/")
    try:
        target = prepare_runpod_session_directory(
            sessions_root=args.sessions_root,
            pending_bootstrap_path=args.pending_budget_bootstrap,
            ledger=CostLedger(
                args.cost_ledger,
                BudgetLimits(
                    gpu=args.gpu_hard_stop_usd,
                    api=args.api_hard_stop_usd,
                    total=args.total_hard_stop_usd,
                ),
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
