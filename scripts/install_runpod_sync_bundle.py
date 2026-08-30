#!/usr/bin/env python3
"""Prepare, install, or clean a verified RunPod selective-sync stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_module(source_checkout: Path) -> None:
    # ``-I -S`` deliberately ignores ambient packages and PYTHONPATH.  Only the
    # explicitly pinned source checkout supplies the stdlib-only installer.
    sys.path.insert(0, str(source_checkout / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "install", "cleanup"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-repository-url")
    parser.add_argument("--expected-manifest-record-hash")
    parser.add_argument("--expected-session-hash")
    args = parser.parse_args()
    _load_module(args.source_checkout)
    try:
        from model_forensics.runpod_sync_transfer import (
            RunpodSyncTransferError,
            cleanup_remote_stage,
            install_remote_stage,
            prepare_remote_stage,
        )

        if args.mode == "prepare":
            if not args.expected_source_commit or not args.expected_source_repository_url:
                raise RunpodSyncTransferError("prepare requires the expected source identity")
            summary = prepare_remote_stage(
                stage=args.stage,
                source_checkout=args.source_checkout,
                expected_source_commit=args.expected_source_commit,
                expected_source_repository_url=args.expected_source_repository_url,
            )
        elif args.mode == "install":
            if not all(
                (
                    args.expected_source_commit,
                    args.expected_source_repository_url,
                    args.expected_manifest_record_hash,
                    args.expected_session_hash,
                )
            ):
                raise RunpodSyncTransferError("install requires the complete expected identity")
            summary = install_remote_stage(
                stage=args.stage,
                source_checkout=args.source_checkout,
                expected_manifest_record_hash=args.expected_manifest_record_hash,
                expected_session_hash=args.expected_session_hash,
                expected_source_commit=args.expected_source_commit,
                expected_source_repository_url=args.expected_source_repository_url,
            )
        else:
            summary = cleanup_remote_stage(
                stage=args.stage,
                source_checkout=args.source_checkout,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"remote selective-sync install failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
