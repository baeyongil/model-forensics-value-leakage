#!/usr/bin/env python3
"""Capture a non-secret, content-addressed GPU software environment manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-wheel", type=Path, required=True)
    args = parser.parse_args()
    if not args.vllm_wheel.is_file():
        raise SystemExit("verified vLLM wheel is absent")

    packages = {
        name: _version(name)
        for name in (
            "accelerate",
            "huggingface-hub",
            "jlens",
            "numpy",
            "safetensors",
            "sentence-transformers",
            "torch",
            "transformers",
            "vllm",
        )
    }
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "pip_freeze": sorted(_output([sys.executable, "-m", "pip", "freeze"]).splitlines()),
        "nvidia_smi": _output(["nvidia-smi"]),
        "git_head": _output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _output(["git", "status", "--porcelain"]),
        "vllm_wheel": {
            "filename": args.vllm_wheel.name,
            "size_bytes": args.vllm_wheel.stat().st_size,
            "sha256": _sha256(args.vllm_wheel),
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
