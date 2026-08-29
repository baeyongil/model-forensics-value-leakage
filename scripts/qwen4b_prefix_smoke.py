#!/usr/bin/env python3
"""Opt-in entry point for the bounded real Qwen3.5-4B prefix smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.qwen4b_smoke import run_qwen4b_prefix_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pinned Qwen3.5-4B full rollout and one exact-token raw-prefix "
            "continuation. This loads a local GPU model but calls no paid API."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/qwen4b_prefix_gpu_smoke.json"),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--rollout-max-tokens", type=int, default=1024)
    parser.add_argument("--continuation-max-tokens", type=int, default=256)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = run_qwen4b_prefix_smoke(
        args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        rollout_max_tokens=args.rollout_max_tokens,
        continuation_max_tokens=args.continuation_max_tokens,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output.resolve()),
                "manifest_hash": manifest["manifest_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
