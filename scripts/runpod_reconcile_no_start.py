#!/usr/bin/env python3
"""Close a RunPod re-arm that authenticated GET evidence proves never started."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from model_forensics.runpod_no_start import (
    NO_START_RECEIPT_FILENAME,
    NoStartReconciliationError,
    attest_no_start,
    safe_no_start_summary,
)
from model_forensics.runpod_recovery import RunpodRecoveryClient

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly read one EXITED Pod and query billing using official REST-v1 GETs; "
            "this command has no provider mutation capability."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet-window-seconds", type=float, default=30.0)
    args = parser.parse_args()

    try:
        if _ENV_NAME_RE.fullmatch(args.api_key_env) is None:
            raise ValueError("RunPod API-key environment variable name is invalid")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"required RunPod API-key environment variable is unset: {args.api_key_env}"
            )
        root = args.project_root.absolute()
        lifecycle = json.loads((root / ".runpod" / "pod_lifecycle.json").read_text())
        authorization = lifecycle.get("current_authorization")
        if not isinstance(authorization, dict) or not isinstance(
            authorization.get("session_hash"), str
        ):
            raise NoStartReconciliationError("private lifecycle session is unavailable")
        session_digest = authorization["session_hash"].removeprefix("sha256:")
        output = args.output or (
            root
            / ".runpod"
            / "sessions"
            / session_digest
            / NO_START_RECEIPT_FILENAME
        )
        receipt = attest_no_start(
            project_root=root,
            client=RunpodRecoveryClient(api_key=api_key),
            output_path=output,
            quiet_window_seconds=args.quiet_window_seconds,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(safe_no_start_summary(receipt), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
