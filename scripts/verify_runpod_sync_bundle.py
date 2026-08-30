#!/usr/bin/env python3
"""Fail closed unless a remote selective RunPod sync is exact and live."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_forensics.runpod_sync_verify import (  # noqa: E402
    RunpodSyncVerificationError,
    verify_selective_sync,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-checkout", type=Path, required=True)
    args = parser.parse_args()
    try:
        source_checkout = args.source_checkout
        if (
            not source_checkout.is_absolute()
            or source_checkout.resolve() != PROJECT_ROOT.resolve()
        ):
            raise RunpodSyncVerificationError(
                "verifier must execute from the exact --source-checkout"
            )
        summary = verify_selective_sync(
            project_root=args.project_root,
            manifest_path=args.manifest,
            source_checkout=source_checkout,
        )
    except (OSError, ValueError, RunpodSyncVerificationError) as exc:
        print(f"selective sync verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
