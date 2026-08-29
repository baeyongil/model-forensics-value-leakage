#!/usr/bin/env python3
"""Record or verify a Linux watchdog process identity without trusting its PID alone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from model_forensics.runpod_sessions import (
    RunpodSessionError,
    record_watchdog_process_identity,
    validate_watchdog_process_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--pid", type=int, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--required-cmdline-token", action="append", required=True)
    record.add_argument("--wait-seconds", type=float, default=5.0)
    record.add_argument("--proc-root", type=Path, default=Path("/proc"))

    verify = subparsers.add_parser("verify")
    verify.add_argument("--identity", type=Path, required=True)
    verify.add_argument("--proc-root", type=Path, default=Path("/proc"))

    args = parser.parse_args()
    try:
        if args.command == "record":
            if args.wait_seconds < 0 or args.wait_seconds > 30:
                raise RunpodSessionError("process identity wait must be between 0 and 30 seconds")
            deadline = time.monotonic() + args.wait_seconds
            while True:
                try:
                    payload = record_watchdog_process_identity(
                        args.output,
                        pid=args.pid,
                        required_cmdline_tokens=tuple(args.required_cmdline_token),
                        proc_root=args.proc_root,
                    )
                    break
                except RunpodSessionError:
                    if args.output.exists() or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        else:
            payload = validate_watchdog_process_identity(
                args.identity,
                proc_root=args.proc_root,
            )
    except (OSError, RunpodSessionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
