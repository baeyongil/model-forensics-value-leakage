"""Pinned, non-vendored access to the unlicensed starter repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from model_forensics.io import sha256_file, write_json


class UpstreamError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise UpstreamError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def ensure_pinned_checkout(*, repository: str, commit: str, destination: str | Path) -> Path:
    """Fetch a pinned checkout into an ignored cache without copying it into the project."""

    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("commit must be a full lowercase SHA-1")
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _git("clone", "--filter=blob:none", "--no-checkout", repository, str(path))
    actual_remote = _git("remote", "get-url", "origin", cwd=path)
    if actual_remote.rstrip("/") != repository.rstrip("/"):
        raise UpstreamError(f"unexpected upstream remote: {actual_remote}")
    _git("fetch", "--depth", "1", "origin", commit, cwd=path)
    _git("checkout", "--detach", commit, cwd=path)
    actual_commit = _git("rev-parse", "HEAD", cwd=path)
    if actual_commit != commit:
        raise UpstreamError(f"checkout mismatch: expected {commit}, got {actual_commit}")
    return path


def detect_license_files(repository: str | Path) -> list[str]:
    root = Path(repository)
    names = {"license", "license.md", "license.txt", "copying", "copying.md"}
    return sorted(
        str(path.relative_to(root)) for path in root.iterdir() if path.name.lower() in names
    )


def summarize_qwen_reference(repository: str | Path) -> dict[str, Any]:
    root = Path(repository)
    candidates = sorted((root / "runs").glob("qwen3.5-122b-a10b_*/factor.json"))
    if len(candidates) != 1:
        raise UpstreamError(f"expected one Qwen 122B factor file, found {len(candidates)}")
    run_dir = candidates[0].parent
    required = ["config.json", "threshold.json", "factor.json"]
    loaded = {
        filename.removesuffix(".json"): json.loads((run_dir / filename).read_text(encoding="utf-8"))
        for filename in required
    }
    return {
        "upstream_commit": _git("rev-parse", "HEAD", cwd=root),
        "run_directory": str(run_dir.relative_to(root)),
        "license_files_at_root": detect_license_files(root),
        "config": loaded["config"],
        "threshold": loaded["threshold"],
        "factor": loaded["factor"],
        "source_hashes": {filename: sha256_file(run_dir / filename) for filename in required},
        "redistribution": "not_included; fetch from pinned upstream",
    }


def write_reference_summary(repository: str | Path, destination: str | Path) -> Path:
    return write_json(destination, summarize_qwen_reference(repository))
