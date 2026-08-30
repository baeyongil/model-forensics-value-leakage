#!/usr/bin/env python3
"""Print the canonical host session directory after authenticating its receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_forensics.runpod_session_path import canonical_host_session_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    args = parser.parse_args()
    try:
        session = canonical_host_session_directory(
            project_root=args.project_root,
            phase=args.phase,
            reservation_path=args.reservation,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
