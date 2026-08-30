#!/usr/bin/env python3
"""Perform the guarded, one-shot host-to-RunPod selective sync."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_forensics.runpod_sync_transfer import (  # noqa: E402
    REMOTE_DESTINATION,
    RunpodSyncTransferError,
    transfer_runpod_sync_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument(
        "--cost-ledger",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "cost_ledger.yaml",
    )
    parser.add_argument(
        "--remote-host",
        required=True,
        help="Exact RunPod direct SSH target as root@canonical-IPv4; hostnames are rejected.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=22,
        help="Exact numeric direct SSH public port (authenticated against the live watchdog).",
    )
    parser.add_argument(
        "--remote-destination",
        type=Path,
        default=REMOTE_DESTINATION,
        help="Pinned for auditability; any non-canonical value is rejected.",
    )
    args = parser.parse_args()
    try:
        summary = transfer_runpod_sync_bundle(
            project_root=args.project_root,
            phase=args.phase,
            reservation_path=args.reservation,
            cost_ledger_path=args.cost_ledger,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_destination=args.remote_destination,
        )
    except (OSError, ValueError, RunpodSyncTransferError) as exc:
        print(f"one-shot RunPod sync failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
