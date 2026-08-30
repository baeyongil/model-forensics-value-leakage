#!/usr/bin/env python3
"""Request or verify one canonical host-watcher stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.io import stable_hash
from model_forensics.runpod_host_control import (
    RunpodHostControlError,
    request_host_stop,
    validate_host_stop_confirmation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("request", "verify"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--watchdog", type=Path)
    parser.add_argument("--stop-request", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "request":
            if args.watchdog is not None or args.stop_request is not None:
                raise ValueError("request derives both host-control paths canonically")
            request = request_host_stop(
                project_root=args.project_root,
                phase=args.phase,
                reservation_path=args.reservation,
            )
            summary = {
                "action": "stop_requested",
                "request_path_hash": stable_hash(
                    {"private_relative_path": str(request.relative_to(args.project_root.resolve()))}
                ),
                "passed": True,
            }
        else:
            if args.watchdog is None or args.stop_request is None:
                raise ValueError("verify requires exact watchdog and stop-request paths")
            state = validate_host_stop_confirmation(
                project_root=args.project_root,
                phase=args.phase,
                reservation_path=args.reservation,
                watchdog_path=args.watchdog,
                stop_request_path=args.stop_request,
            )
            summary = {
                "action": "stop_confirmed",
                "watchdog_state_hash": stable_hash(state),
                "passed": True,
            }
    except (OSError, RuntimeError, ValueError, RunpodHostControlError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
