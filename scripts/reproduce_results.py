#!/usr/bin/env python3
"""Rebuild public aggregate tables and figures without raw generation or credentials."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_forensics.public_results import PublicResultsError, reproduce_release_bundle


def _safe_output_directory(root: Path, requested: Path, *, label: str) -> Path:
    root = root.resolve(strict=True)
    candidate = Path(os.path.abspath(requested if requested.is_absolute() else root / requested))
    if not candidate.is_relative_to(root):
        raise PublicResultsError(f"{label} must remain inside the project")
    current = root
    for component in candidate.relative_to(root).parts:
        current /= component
        if current.is_symlink():
            raise PublicResultsError(f"{label} traverses a symlink")
        if current.exists() and not current.is_dir():
            raise PublicResultsError(f"{label} collides with a non-directory")
    if candidate.exists():
        for directory, directories, files in os.walk(candidate, followlinks=False):
            base = Path(directory)
            if any((base / name).is_symlink() for name in [*directories, *files]):
                raise PublicResultsError(f"{label} contains a symlink")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("reports/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    try:
        root = Path.cwd().resolve(strict=True)
        result = reproduce_release_bundle(
            project_root=root,
            results_dir=_safe_output_directory(root, args.results_dir, label="results directory"),
            figure_dir=_safe_output_directory(root, args.figure_dir, label="figure directory"),
        )
    except (OSError, ValueError, PublicResultsError) as exc:
        print(f"public result reproduction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
