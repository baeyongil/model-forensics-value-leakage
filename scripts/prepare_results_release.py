#!/usr/bin/env python3
"""Stage an aggregate-only public bundle from an authenticated analysis summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_forensics.cli import (
    _normalize_lens_rows,
    _project_root,
    _validate_analysis_summary_bundle,
)
from model_forensics.config import load_run_config
from model_forensics.io import sha256_file
from model_forensics.public_results import (
    PublicResultsError,
    aggregate_lens_rows,
    build_released_evidence,
    load_jsonl_rows,
    reproduce_release_bundle,
    write_release_bundle,
)


def _linked_path(root: Path, link: object, *, label: str) -> Path:
    if not isinstance(link, dict) or not isinstance(link.get("path"), str):
        raise PublicResultsError(f"authenticated analysis has no {label} path")
    path = Path(link["path"])
    if path.is_absolute() or ".." in path.parts:
        raise PublicResultsError(f"authenticated analysis {label} path is unsafe")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - lexical checks already exclude this
        raise PublicResultsError(f"authenticated analysis {label} escapes the project") from exc
    return resolved


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
    parser.add_argument("--config", type=Path, default=Path("config/run_122b.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("reports/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config)
        root = _project_root(config)
        report_dir = config.paths.report_dir
        if not report_dir.is_absolute():
            report_dir = root / report_dir
        summary_path = report_dir.resolve() / "analysis_summary.json"
        summary = _validate_analysis_summary_bundle(config, summary_path)
        if summary.get("synthetic_smoke") is not False:
            raise PublicResultsError("public primary results refuse synthetic smoke evidence")
        behavior_path = _linked_path(root, summary["tables"]["behavior"], label="behavior table")
        effects_path = _linked_path(root, summary["tables"]["effects"], label="effects table")
        lens_rows: list[dict] = []
        if summary.get("lens_evidence_status") == "available_122b":
            lens_path = _linked_path(root, summary["inputs"]["lens"], label="lens input")
            lens_rows = aggregate_lens_rows(_normalize_lens_rows(load_jsonl_rows(lens_path)))
        evidence = build_released_evidence(
            profile=config.profile,
            analysis_hash=str(summary["analysis_hash"]),
            source_analysis_summary_sha256=sha256_file(summary_path),
            lens_evidence_status=str(summary["lens_evidence_status"]),
            behavior_rows=load_jsonl_rows(behavior_path),
            effect_rows=load_jsonl_rows(effects_path),
            lens_rows=lens_rows,
        )
        results_dir = _safe_output_directory(root, args.results_dir, label="results directory")
        # Keep public figure paths project-relative inside the manifest while
        # writing the regenerated files beneath this exact project root.
        figure_output = _safe_output_directory(root, args.figure_dir, label="figure directory")
        figure_relative = figure_output.relative_to(root)
        manifest = write_release_bundle(
            project_root=root,
            results_dir=results_dir,
            figure_dir=figure_relative,
            evidence=evidence,
        )
        reproduction = reproduce_release_bundle(
            project_root=root,
            results_dir=results_dir,
            figure_dir=figure_output,
        )
    except (OSError, ValueError, PublicResultsError) as exc:
        print(f"result release preparation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "aggregate_release_staged",
                "manifest_record_hash": manifest["record_hash"],
                "reproduction": reproduction,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
