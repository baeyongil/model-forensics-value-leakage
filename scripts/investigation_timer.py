#!/usr/bin/env python3
"""Start, stop, or inspect the auditable investigation-time ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.timeledger import TimeLedger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--ledger", type=Path, default=Path("data/manifests/time_ledger.yaml"))
    parser.add_argument("--category")
    parser.add_argument("--description")
    parser.add_argument("--status", choices=("counted", "excluded"))
    args = parser.parse_args()
    ledger = TimeLedger(args.ledger)
    if args.action == "start":
        if not args.category or not args.description or not args.status:
            parser.error("start requires --category, --description, and --status")
        result = ledger.start(
            category=args.category,
            description=args.description,
            status=args.status,
        )
    elif args.action == "stop":
        result = ledger.stop()
    else:
        result = ledger.status()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
