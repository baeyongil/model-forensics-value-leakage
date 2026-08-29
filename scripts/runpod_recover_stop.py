#!/usr/bin/env python3
"""Attest an already-stopped RunPod Pod using REST-v1 GET evidence only."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from model_forensics.runpod_recovery import (
    EXTERNAL_STOP_RECEIPT_FILENAME,
    RunpodRecoveryClient,
    RunpodRecoveryError,
    attest_external_stop,
    safe_recovery_summary,
)

_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{2,63}\Z")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read one EXITED Pod and its billing row; this command has no provider mutation "
            "capability."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failed-watchdog", type=Path)
    parser.add_argument("--failed-log", type=Path)
    parser.add_argument(
        "--allow-pending-billing-ceiling",
        action="store_true",
        help=(
            "If the exact billing query returns no rows, account for the upward-rounded "
            "provider-timestamp ceiling and label billing pending."
        ),
    )
    args = parser.parse_args()

    try:
        if _ENV_NAME_RE.fullmatch(args.api_key_env) is None:
            raise ValueError("RunPod API-key environment variable name is invalid")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"required RunPod API-key environment variable is unset: {args.api_key_env}"
            )
        root = args.project_root.resolve()
        lifecycle = json.loads((root / ".runpod" / "pod_lifecycle.json").read_text())
        authorization = lifecycle.get("current_authorization")
        if not isinstance(authorization, dict) or not isinstance(
            authorization.get("session_hash"), str
        ):
            raise RunpodRecoveryError("private lifecycle session is unavailable")
        session_digest = authorization["session_hash"].removeprefix("sha256:")
        output = args.output or (
            root
            / ".runpod"
            / "sessions"
            / session_digest
            / EXTERNAL_STOP_RECEIPT_FILENAME
        )
        receipt = attest_external_stop(
            project_root=root,
            client=RunpodRecoveryClient(api_key=api_key),
            output_path=output,
            allow_pending_billing_ceiling=args.allow_pending_billing_ceiling,
            failed_watchdog_path=args.failed_watchdog,
            failed_log_path=args.failed_log,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(safe_recovery_summary(receipt), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
