"""Sanitized, content-addressed public aggregate result bundles.

The public bundle is intentionally too coarse to contain a model trajectory,
provider response, infrastructure identifier, or per-trace identifier.  Its
only row-level payloads are analysis aggregates needed to rebuild the three
preregistered figures.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from model_forensics.figures import (
    plot_first_vs_final_bias,
    plot_lens_heatmap,
    plot_sentence_effect_forest,
)
from model_forensics.io import canonical_json, read_json, sha256_file, stable_hash, write_json

PUBLIC_RESULTS_SCHEMA_VERSION = 1
EVIDENCE_FILENAME = "released_evidence.json"
MANIFEST_FILENAME = "results_manifest.json"

TABLE_PATHS = {
    "behavior_stage_summary": "tables/behavior_stage_summary.jsonl",
    "sentence_effects": "tables/sentence_effects.jsonl",
    "lens_direction_heatmap": "tables/lens_direction_heatmap.jsonl",
}
FIGURE_PATHS = {
    "first_vs_final_bias": "first_vs_final_bias.png",
    "sentence_causal_effect_forest": "sentence_causal_effect_forest.png",
    "lens_layer_position_heatmap": "lens_layer_position_heatmap.png",
}

BEHAVIOR_FIELDS = (
    "task",
    "condition",
    "stage",
    "metric",
    "rate",
    "ci_low",
    "ci_high",
    "n",
    "n_total",
    "n_missing",
    "missing_rate",
    "good_side_rate",
    "good_side_ci_low",
    "good_side_ci_high",
    "good_side_n",
    "signed_log_ratio_mean",
    "signed_log_ratio_median",
    "signed_log_ratio_n",
    "signed_log_definition",
)
EFFECT_FIELDS = (
    "sentence_class",
    "direction",
    "contrast_id",
    "estimand",
    "estimate",
    "ci_low",
    "ci_high",
    "conclusion",
    "clusters",
    "total_clusters",
    "complete_case_clusters",
    "p_value",
    "p_value_adjusted",
    "inference_tier",
    "is_confirmatory",
    "analysis_population",
    "worst_case_bound",
    "best_case_bound",
    "divergent_coverage_gate_passed",
)
LENS_FIELDS = (
    "lens_type",
    "layer",
    "position",
    "concept_set",
    "signed_contrast",
    "eligible_trace_count",
    "aggregation",
)

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TASKS = {"giraffe", "chicago_coffee"}
_CONDITIONS = {"baseline", "threshold_only", "above_good", "below_good"}
_SENTENCE_CLASSES = {
    "accuracy_commitment",
    "value_threshold_planning",
    "epistemic_control",
}
_DIRECTIONS = {"above_good", "below_good", "pooled"}
_CONCLUSIONS = {"positive", "negative", "practically_null", "inconclusive"}
_INFERENCE_TIERS = {"confirmatory", "exploratory"}
_POSITIONS = {
    "prompt_end",
    "first_estimate_pre",
    "anchor_pre",
    "anchor_post",
    "final_answer_pre",
}


class PublicResultsError(RuntimeError):
    """A public result bundle violates its aggregate-only integrity contract."""


def _json_scalar(value: Any) -> Any:
    """Normalize pandas/NumPy scalars and missing markers to strict JSON values."""

    if value is None or bool(pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _finite(value: Any, *, label: str, nullable: bool = False) -> float | None:
    if value is None:
        if nullable:
            return None
        raise PublicResultsError(f"{label} must be finite")
    if isinstance(value, bool):
        raise PublicResultsError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicResultsError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise PublicResultsError(f"{label} must be finite")
    return number


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise PublicResultsError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PublicResultsError(f"{label} must be an integer") from exc
    if number != value or number < minimum:
        raise PublicResultsError(f"{label} must be an integer >= {minimum}")
    return number


def _exact_fields(row: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    if set(row) != set(fields):
        missing = sorted(set(fields).difference(row))
        extra = sorted(set(row).difference(fields))
        raise PublicResultsError(f"{label} fields changed; missing={missing}, extra={extra}")


def sanitize_behavior_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select only aggregate behavior fields and validate their interpretation."""

    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        row = {field: _json_scalar(source.get(field)) for field in BEHAVIOR_FIELDS}
        row["metric"] = "estimate_above_threshold_probability"
        if row["task"] not in _TASKS or row["condition"] not in _CONDITIONS:
            raise PublicResultsError(f"behavior row {index} has an unknown task or condition")
        if row["stage"] not in {"first", "final"}:
            raise PublicResultsError(f"behavior row {index} has an unknown stage")
        for field in ("rate", "ci_low", "ci_high", "missing_rate"):
            number = _finite(row[field], label=f"behavior row {index} {field}")
            if number is None or not 0 <= number <= 1:  # pragma: no cover - None excluded above
                raise PublicResultsError(f"behavior row {index} {field} is outside [0, 1]")
            row[field] = number
        if not row["ci_low"] <= row["rate"] <= row["ci_high"]:
            raise PublicResultsError(f"behavior row {index} interval does not contain its rate")
        for field in ("n", "n_total", "n_missing", "good_side_n", "signed_log_ratio_n"):
            row[field] = _integer(row[field], label=f"behavior row {index} {field}")
        for field in (
            "good_side_rate",
            "good_side_ci_low",
            "good_side_ci_high",
            "signed_log_ratio_mean",
            "signed_log_ratio_median",
        ):
            row[field] = _finite(
                row[field], label=f"behavior row {index} {field}", nullable=True
            )
        if row["condition"] in {"baseline", "threshold_only"}:
            if any(
                row[field] is not None
                for field in ("good_side_rate", "good_side_ci_low", "good_side_ci_high")
            ) or row["good_side_n"] != 0:
                raise PublicResultsError(
                    f"behavior row {index} assigns a good side to a neutral condition"
                )
        else:
            if any(
                row[field] is None
                for field in ("good_side_rate", "good_side_ci_low", "good_side_ci_high")
            ):
                raise PublicResultsError(f"behavior row {index} omits its good-side aggregate")
        if row["signed_log_definition"] != "direction * log(estimate / threshold)":
            raise PublicResultsError(f"behavior row {index} changed the signed-log definition")
        output.append(row)
    if not output:
        raise PublicResultsError("public behavior aggregate is empty")
    return output


def sanitize_effect_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select effect aggregates while preserving legitimate non-estimable cells."""

    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        row = {field: _json_scalar(source.get(field)) for field in EFFECT_FIELDS}
        sentence_class = row["sentence_class"]
        direction = row["direction"]
        if sentence_class not in _SENTENCE_CLASSES or direction not in _DIRECTIONS:
            raise PublicResultsError(f"effect row {index} has an unknown contrast")
        if row["contrast_id"] != f"{sentence_class}:{direction}":
            raise PublicResultsError(f"effect row {index} has an inconsistent contrast id")
        if row["estimand"] != (
            "equal-base-trace-weighted P(good side | retain) "
            "- P(good side | divergent resample)"
        ):
            raise PublicResultsError(f"effect row {index} changed the frozen estimand")
        effect_values = [row[field] for field in ("estimate", "ci_low", "ci_high")]
        if all(value is None for value in effect_values):
            if row["conclusion"] != "inconclusive":
                raise PublicResultsError(
                    f"effect row {index} gives a conclusion to a non-estimable cell"
                )
        elif any(value is None for value in effect_values):
            raise PublicResultsError(f"effect row {index} has a partial effect interval")
        else:
            for field in ("estimate", "ci_low", "ci_high"):
                row[field] = _finite(row[field], label=f"effect row {index} {field}")
            if not row["ci_low"] <= row["estimate"] <= row["ci_high"]:
                raise PublicResultsError(f"effect row {index} interval excludes its estimate")
        if row["conclusion"] not in _CONCLUSIONS:
            raise PublicResultsError(f"effect row {index} has an unknown conclusion")
        if row["inference_tier"] not in _INFERENCE_TIERS:
            raise PublicResultsError(f"effect row {index} has an unknown inference tier")
        if not isinstance(row["is_confirmatory"], bool):
            raise PublicResultsError(f"effect row {index} has a non-boolean primary label")
        if row["analysis_population"] != "base_trace_complete_case":
            raise PublicResultsError(f"effect row {index} changed the analysis population")
        for field in ("clusters", "total_clusters", "complete_case_clusters"):
            row[field] = _integer(row[field], label=f"effect row {index} {field}")
        for field in (
            "p_value",
            "p_value_adjusted",
            "worst_case_bound",
            "best_case_bound",
        ):
            row[field] = _finite(row[field], label=f"effect row {index} {field}", nullable=True)
        if row["divergent_coverage_gate_passed"] not in {True, False, None}:
            raise PublicResultsError(f"effect row {index} has an invalid coverage-gate result")
        output.append(row)
    if not output:
        raise PublicResultsError("public sentence-effect aggregate is empty")
    return output


def aggregate_lens_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate eligible direction contrasts over a common trace population."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    required = {
        "trace_id",
        "lens_type",
        "layer",
        "position",
        "concept_set",
        "signed_contrast",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise PublicResultsError(f"lens rows omit aggregation fields: {sorted(missing)}")
    if "probe_eligible" in frame:
        frame = frame[frame["probe_eligible"] == True].copy()  # noqa: E712
    frame = frame[frame["concept_set"] == "direction"].copy()
    if frame.empty:
        return []
    positions = frame.groupby("trace_id", observed=True)["position"].agg(
        lambda values: set(values)
    )
    common_traces = {
        trace_id for trace_id, values in positions.items() if _POSITIONS.issubset(values)
    }
    frame = frame[frame["trace_id"].isin(common_traces)].copy()
    if frame.empty:
        return []
    frame["signed_contrast"] = pd.to_numeric(frame["signed_contrast"], errors="raise")
    grouped = (
        frame.groupby(
            ["lens_type", "layer", "position", "concept_set"],
            sort=True,
            observed=True,
        )
        .agg(
            signed_contrast=("signed_contrast", "mean"),
            eligible_trace_count=("trace_id", "nunique"),
        )
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for source in grouped.to_dict("records"):
        records.append(
            {
                "lens_type": str(source["lens_type"]).lower(),
                "layer": int(source["layer"]),
                "position": str(source["position"]),
                "concept_set": "direction",
                "signed_contrast": float(source["signed_contrast"]),
                "eligible_trace_count": int(source["eligible_trace_count"]),
                "aggregation": "mean_over_common_probe_eligible_traces",
            }
        )
    return records


def validate_lens_aggregate(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        _exact_fields(row, LENS_FIELDS, label=f"lens aggregate row {index}")
        if row["lens_type"] not in {"j", "r"}:
            raise PublicResultsError(f"lens aggregate row {index} has an unknown lens type")
        row["layer"] = _integer(row["layer"], label=f"lens aggregate row {index} layer")
        if not 4 <= row["layer"] <= 46 or row["position"] not in _POSITIONS:
            raise PublicResultsError(f"lens aggregate row {index} has an unknown grid cell")
        if row["concept_set"] != "direction":
            raise PublicResultsError(f"lens aggregate row {index} is not a direction contrast")
        row["signed_contrast"] = _finite(
            row["signed_contrast"], label=f"lens aggregate row {index} contrast"
        )
        row["eligible_trace_count"] = _integer(
            row["eligible_trace_count"],
            label=f"lens aggregate row {index} trace count",
            minimum=1,
        )
        if row["aggregation"] != "mean_over_common_probe_eligible_traces":
            raise PublicResultsError(f"lens aggregate row {index} changed its aggregation rule")
        key = (row["lens_type"], row["layer"], row["position"])
        if key in seen:
            raise PublicResultsError(f"duplicate lens aggregate cell: {key}")
        seen.add(key)
        output.append(row)
    return output


def build_released_evidence(
    *,
    profile: str,
    analysis_hash: str,
    source_analysis_summary_sha256: str,
    lens_evidence_status: str,
    behavior_rows: Iterable[Mapping[str, Any]],
    effect_rows: Iterable[Mapping[str, Any]],
    lens_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if profile != "qwen35_122b_primary":
        raise PublicResultsError("only the canonical primary profile can be publicly released")
    if not _HASH.fullmatch(analysis_hash):
        raise PublicResultsError("analysis_hash is not a namespaced SHA-256")
    if not _HEX_SHA256.fullmatch(source_analysis_summary_sha256):
        raise PublicResultsError("analysis-summary digest is not SHA-256")
    if lens_evidence_status not in {"available_122b", "unavailable_not_zero"}:
        raise PublicResultsError("only primary lens statuses can be released")
    behavior = sanitize_behavior_rows(behavior_rows)
    effects = sanitize_effect_rows(effect_rows)
    lens = validate_lens_aggregate(lens_rows)
    if lens_evidence_status == "available_122b" and not lens:
        raise PublicResultsError("available lens evidence has no public heatmap aggregate")
    if lens_evidence_status == "unavailable_not_zero" and lens:
        raise PublicResultsError("unavailable lens evidence cannot expose pseudo-values")
    payload: dict[str, Any] = {
        "schema_version": PUBLIC_RESULTS_SCHEMA_VERSION,
        "status": "aggregate_only_release",
        "profile": profile,
        "analysis_hash": analysis_hash,
        "source_analysis_summary_sha256": source_analysis_summary_sha256,
        "lens_evidence_status": lens_evidence_status,
        "behavior_stage_summary": behavior,
        "sentence_effects": effects,
        "lens_direction_heatmap": lens,
        "privacy_boundary": (
            "no_raw_reasoning_provider_bodies_infrastructure_ids_or_trace_ids"
        ),
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def validate_released_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "profile",
        "analysis_hash",
        "source_analysis_summary_sha256",
        "lens_evidence_status",
        "behavior_stage_summary",
        "sentence_effects",
        "lens_direction_heatmap",
        "privacy_boundary",
        "record_hash",
    }
    if set(payload) != expected_fields:
        raise PublicResultsError("released evidence top-level fields changed")
    if payload.get("schema_version") != PUBLIC_RESULTS_SCHEMA_VERSION:
        raise PublicResultsError("released evidence schema version changed")
    if payload.get("status") != "aggregate_only_release":
        raise PublicResultsError("released evidence status changed")
    if payload.get("privacy_boundary") != (
        "no_raw_reasoning_provider_bodies_infrastructure_ids_or_trace_ids"
    ):
        raise PublicResultsError("released evidence privacy boundary changed")
    expected_hash = stable_hash({key: value for key, value in payload.items() if key != "record_hash"})
    if payload.get("record_hash") != expected_hash:
        raise PublicResultsError("released evidence record hash mismatch")
    behavior = payload.get("behavior_stage_summary")
    effects = payload.get("sentence_effects")
    lens = payload.get("lens_direction_heatmap")
    if not isinstance(behavior, list) or not all(isinstance(row, Mapping) for row in behavior):
        raise PublicResultsError("released behavior aggregate is not a row list")
    if not isinstance(effects, list) or not all(isinstance(row, Mapping) for row in effects):
        raise PublicResultsError("released effect aggregate is not a row list")
    if not isinstance(lens, list) or not all(isinstance(row, Mapping) for row in lens):
        raise PublicResultsError("released lens aggregate is not a row list")
    rebuilt = build_released_evidence(
        profile=str(payload.get("profile", "")),
        analysis_hash=str(payload.get("analysis_hash", "")),
        source_analysis_summary_sha256=str(payload.get("source_analysis_summary_sha256", "")),
        lens_evidence_status=str(payload.get("lens_evidence_status", "")),
        behavior_rows=behavior,
        effect_rows=effects,
        lens_rows=lens,
    )
    if rebuilt != dict(payload):
        raise PublicResultsError("released evidence is not canonical")
    return rebuilt


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _secure_project_output_path(
    project_root: Path, requested: Path, *, label: str
) -> tuple[Path, Path]:
    """Return a project-contained output path without following a symlink.

    ``project_root`` is the trusted boundary.  Every component beneath that
    boundary is checked before any directory is created or file is written.
    The resolved-containment check is deliberately redundant with the
    component walk so a missing child beneath an existing escaping symlink
    cannot be mistaken for a safe lexical path.
    """

    lexical_root = project_root.absolute()
    if lexical_root.is_symlink():
        raise PublicResultsError(f"{label} project root is a symlink")
    try:
        root = project_root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise PublicResultsError(f"{label} project root is unavailable") from exc
    if not root.is_dir():
        raise PublicResultsError(f"{label} project root is not a directory")

    if requested.is_absolute():
        raw_candidate = Path(os.path.abspath(requested))
        if raw_candidate.is_relative_to(lexical_root):
            candidate = root / raw_candidate.relative_to(lexical_root)
        else:
            candidate = raw_candidate
    else:
        candidate = Path(os.path.abspath(root / requested))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PublicResultsError(f"{label} must remain inside the project") from exc

    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise PublicResultsError(f"{label} traverses a symlink")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except (RuntimeError, ValueError, OSError) as exc:
        raise PublicResultsError(f"{label} resolves outside the project") from exc
    return root, candidate


def _reject_symlink_path(root: Path, path: Path, *, label: str) -> None:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    if not path_absolute.is_relative_to(root_absolute):
        raise PublicResultsError(f"{label} escapes its output directory")
    if root_absolute.is_symlink():
        raise PublicResultsError(f"{label} output directory is a symlink")
    current = root_absolute
    for component in path_absolute.relative_to(root_absolute).parts:
        current /= component
        if current.is_symlink():
            raise PublicResultsError(f"{label} traverses a symlink")


def _prepare_output_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise PublicResultsError(f"{label} must be a real directory")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():  # fail closed if the path changed during creation
        raise PublicResultsError(f"{label} must be a real directory")
    return path


def build_results_manifest(
    *, results_dir: Path, figure_dir: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    evidence_path = results_dir / EVIDENCE_FILENAME
    table_rows = {
        "behavior_stage_summary": evidence["behavior_stage_summary"],
        "sentence_effects": evidence["sentence_effects"],
        "lens_direction_heatmap": evidence["lens_direction_heatmap"],
    }
    tables = {
        name: {
            "path": relative,
            "sha256": _sha256_bytes(_jsonl_bytes(rows)),
            "row_count": len(rows),
            "fields": list(
                BEHAVIOR_FIELDS
                if name == "behavior_stage_summary"
                else EFFECT_FIELDS
                if name == "sentence_effects"
                else LENS_FIELDS
            ),
        }
        for name, relative in TABLE_PATHS.items()
        for rows in (table_rows[name],)
    }
    figure_inventory = {
        name: path
        for name, path in FIGURE_PATHS.items()
        if name != "lens_layer_position_heatmap" or bool(evidence["lens_direction_heatmap"])
    }
    payload: dict[str, Any] = {
        "schema_version": PUBLIC_RESULTS_SCHEMA_VERSION,
        "status": "authenticated_aggregate_release",
        "evidence": {
            "path": EVIDENCE_FILENAME,
            "sha256": sha256_file(evidence_path),
            "record_hash": evidence["record_hash"],
        },
        "aggregate_tables": tables,
        "figure_outputs": {
            name: str((figure_dir / relative).as_posix())
            for name, relative in figure_inventory.items()
        },
        "figure_hash_policy": "regenerated_not_cross_platform_bitwise_pinned",
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _write_table(
    path: Path, rows: Iterable[Mapping[str, Any]], *, results_dir: Path
) -> Path:
    _reject_symlink_path(results_dir, path, label="aggregate table")
    _prepare_output_directory(path.parent, label="aggregate table directory")
    _reject_symlink_path(results_dir, path, label="aggregate table")
    path.write_bytes(_jsonl_bytes(rows))
    return path


def render_release_figures(
    *, project_root: Path, evidence: Mapping[str, Any], figure_dir: Path
) -> dict[str, Path]:
    """Regenerate the exact public figures from canonical aggregate evidence."""

    validated = validate_released_evidence(evidence)
    _root, figure_dir = _secure_project_output_path(
        project_root, figure_dir, label="public figure directory"
    )
    _prepare_output_directory(figure_dir, label="public figure directory")
    for filename in FIGURE_PATHS.values():
        _reject_symlink_path(
            figure_dir, figure_dir / filename, label="public figure output"
        )
    behavior = pd.DataFrame(validated["behavior_stage_summary"])
    effects = pd.DataFrame(validated["sentence_effects"])
    figure_paths = {
        "first_vs_final_bias": plot_first_vs_final_bias(
            behavior, figure_dir / FIGURE_PATHS["first_vs_final_bias"]
        ),
        "sentence_causal_effect_forest": plot_sentence_effect_forest(
            effects, figure_dir / FIGURE_PATHS["sentence_causal_effect_forest"]
        ),
    }
    if validated["lens_direction_heatmap"]:
        lens = pd.DataFrame(validated["lens_direction_heatmap"])
        figure_paths["lens_layer_position_heatmap"] = plot_lens_heatmap(
            lens, figure_dir / FIGURE_PATHS["lens_layer_position_heatmap"]
        )
    return figure_paths


def write_release_bundle(
    *,
    project_root: Path,
    results_dir: Path,
    figure_dir: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_released_evidence(evidence)
    root, results_dir = _secure_project_output_path(
        project_root, results_dir, label="public results directory"
    )
    _figure_root, figure_output = _secure_project_output_path(
        root, figure_dir, label="public figure directory"
    )
    figure_dir = figure_output.relative_to(root)
    _prepare_output_directory(results_dir, label="public results directory")
    for relative in [EVIDENCE_FILENAME, MANIFEST_FILENAME, *TABLE_PATHS.values()]:
        _reject_symlink_path(
            results_dir, results_dir / relative, label="public result output"
        )
    write_json(results_dir / EVIDENCE_FILENAME, validated)
    for name, relative in TABLE_PATHS.items():
        _write_table(results_dir / relative, validated[name], results_dir=results_dir)
    manifest = build_results_manifest(
        results_dir=results_dir,
        figure_dir=figure_dir,
        evidence=validated,
    )
    write_json(results_dir / MANIFEST_FILENAME, manifest)
    return manifest


def _validate_results_manifest(
    manifest: Mapping[str, Any], *, results_dir: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "evidence",
        "aggregate_tables",
        "figure_outputs",
        "figure_hash_policy",
        "record_hash",
    }
    if set(manifest) != expected_top:
        raise PublicResultsError("results manifest top-level fields changed")
    if manifest.get("schema_version") != PUBLIC_RESULTS_SCHEMA_VERSION:
        raise PublicResultsError("results manifest schema version changed")
    if manifest.get("status") != "authenticated_aggregate_release":
        raise PublicResultsError("results manifest status changed")
    if manifest.get("figure_hash_policy") != "regenerated_not_cross_platform_bitwise_pinned":
        raise PublicResultsError("results manifest figure hash policy changed")
    expected_record_hash = stable_hash(
        {key: value for key, value in manifest.items() if key != "record_hash"}
    )
    if manifest.get("record_hash") != expected_record_hash:
        raise PublicResultsError("results manifest record hash mismatch")
    evidence_link = manifest.get("evidence")
    if not isinstance(evidence_link, Mapping) or set(evidence_link) != {
        "path",
        "sha256",
        "record_hash",
    }:
        raise PublicResultsError("results manifest evidence link changed")
    if evidence_link.get("path") != EVIDENCE_FILENAME:
        raise PublicResultsError("results manifest evidence path changed")
    evidence_path = results_dir / EVIDENCE_FILENAME
    if evidence_link.get("sha256") != sha256_file(evidence_path):
        raise PublicResultsError("released evidence file hash mismatch")
    if evidence_link.get("record_hash") != evidence.get("record_hash"):
        raise PublicResultsError("released evidence record hash linkage mismatch")
    tables = manifest.get("aggregate_tables")
    if not isinstance(tables, Mapping) or set(tables) != set(TABLE_PATHS):
        raise PublicResultsError("results manifest table inventory changed")
    field_sets = {
        "behavior_stage_summary": BEHAVIOR_FIELDS,
        "sentence_effects": EFFECT_FIELDS,
        "lens_direction_heatmap": LENS_FIELDS,
    }
    for name, relative in TABLE_PATHS.items():
        metadata = tables[name]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "path",
            "sha256",
            "row_count",
            "fields",
        }:
            raise PublicResultsError(f"results manifest {name} metadata changed")
        rows = evidence[name]
        if metadata.get("path") != relative:
            raise PublicResultsError(f"results manifest {name} path changed")
        if metadata.get("sha256") != _sha256_bytes(_jsonl_bytes(rows)):
            raise PublicResultsError(f"results manifest {name} aggregate hash mismatch")
        if metadata.get("row_count") != len(rows):
            raise PublicResultsError(f"results manifest {name} row count changed")
        if metadata.get("fields") != list(field_sets[name]):
            raise PublicResultsError(f"results manifest {name} fields changed")
    expected_figures = {
        name: value
        for name, value in manifest.get("figure_outputs", {}).items()
        if isinstance(name, str) and isinstance(value, str)
    }
    required_names = {"first_vs_final_bias", "sentence_causal_effect_forest"}
    if evidence["lens_direction_heatmap"]:
        required_names.add("lens_layer_position_heatmap")
    if set(expected_figures) != required_names:
        raise PublicResultsError("results manifest figure inventory changed")
    for name, path in expected_figures.items():
        if not path.endswith(FIGURE_PATHS[name]) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PublicResultsError(f"results manifest {name} path is unsafe")
    return dict(manifest)


def reproduce_release_bundle(
    *, project_root: Path, results_dir: Path, figure_dir: Path
) -> dict[str, Any]:
    """Verify public evidence, rebuild aggregate tables, and regenerate figures."""

    root, results_dir = _secure_project_output_path(
        project_root, results_dir, label="public results directory"
    )
    _figure_root, figure_dir = _secure_project_output_path(
        root, figure_dir, label="public figure directory"
    )
    evidence_path = results_dir / EVIDENCE_FILENAME
    manifest_path = results_dir / MANIFEST_FILENAME
    _reject_symlink_path(results_dir, evidence_path, label="released evidence")
    _reject_symlink_path(results_dir, manifest_path, label="results manifest")
    if not evidence_path.is_file() or not manifest_path.is_file():
        raise PublicResultsError(
            "public result evidence is absent; stage the authenticated aggregate release first"
        )
    evidence_payload = read_json(evidence_path)
    manifest_payload = read_json(manifest_path)
    if not isinstance(evidence_payload, Mapping) or not isinstance(manifest_payload, Mapping):
        raise PublicResultsError("public result artifacts must be JSON objects")
    evidence = validate_released_evidence(evidence_payload)
    manifest = _validate_results_manifest(
        manifest_payload,
        results_dir=results_dir,
        evidence=evidence,
    )
    for name, relative in TABLE_PATHS.items():
        table_path = _write_table(
            results_dir / relative, evidence[name], results_dir=results_dir
        )
        expected = manifest["aggregate_tables"][name]["sha256"]
        if sha256_file(table_path) != expected:
            raise PublicResultsError(f"rebuilt {name} table hash mismatch")

    figure_paths = render_release_figures(
        project_root=root,
        evidence=evidence,
        figure_dir=figure_dir,
    )
    return {
        "schema_version": PUBLIC_RESULTS_SCHEMA_VERSION,
        "status": "reproduced_from_authenticated_aggregate_release",
        "evidence_record_hash": evidence["record_hash"],
        "manifest_record_hash": manifest["record_hash"],
        "aggregate_table_hashes": {
            name: metadata["sha256"]
            for name, metadata in manifest["aggregate_tables"].items()
        },
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "raw_generation_performed": False,
    }


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL strictly for the private release-staging adapter."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PublicResultsError(f"{path}:{line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise PublicResultsError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


__all__ = [
    "BEHAVIOR_FIELDS",
    "EFFECT_FIELDS",
    "EVIDENCE_FILENAME",
    "FIGURE_PATHS",
    "LENS_FIELDS",
    "MANIFEST_FILENAME",
    "PUBLIC_RESULTS_SCHEMA_VERSION",
    "TABLE_PATHS",
    "PublicResultsError",
    "aggregate_lens_rows",
    "build_released_evidence",
    "build_results_manifest",
    "load_jsonl_rows",
    "render_release_figures",
    "reproduce_release_bundle",
    "sanitize_behavior_rows",
    "sanitize_effect_rows",
    "validate_lens_aggregate",
    "validate_released_evidence",
    "write_release_bundle",
]
