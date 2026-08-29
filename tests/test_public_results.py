from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forensics.io import read_json, sha256_file
from model_forensics.public_results import (
    PublicResultsError,
    build_released_evidence,
    render_release_figures,
    reproduce_release_bundle,
    validate_released_evidence,
    write_release_bundle,
)


def _behavior_rows() -> list[dict]:
    rows: list[dict] = []
    for condition in ("baseline", "threshold_only", "above_good", "below_good"):
        for stage, rate in (("first", 0.4), ("final", 0.6)):
            directional = condition in {"above_good", "below_good"}
            rows.append(
                {
                    "task": "giraffe",
                    "condition": condition,
                    "stage": stage,
                    "rate": rate,
                    "ci_low": rate - 0.1,
                    "ci_high": rate + 0.1,
                    "n": 10,
                    "n_total": 10,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "good_side_rate": rate if directional else None,
                    "good_side_ci_low": rate - 0.1 if directional else None,
                    "good_side_ci_high": rate + 0.1 if directional else None,
                    "good_side_n": 10 if directional else 0,
                    "signed_log_ratio_mean": 0.1 if directional else None,
                    "signed_log_ratio_median": 0.1 if directional else None,
                    "signed_log_ratio_n": 10 if directional else 0,
                    "signed_log_definition": "direction * log(estimate / threshold)",
                }
            )
    return rows


def _effect_rows() -> list[dict]:
    rows: list[dict] = []
    for index, sentence_class in enumerate(
        ("accuracy_commitment", "value_threshold_planning", "epistemic_control")
    ):
        estimate = None if index == 0 else 0.1 * index
        rows.append(
            {
                "sentence_class": sentence_class,
                "direction": "pooled",
                "contrast_id": f"{sentence_class}:pooled",
                "estimand": (
                    "equal-base-trace-weighted P(good side | retain) "
                    "- P(good side | divergent resample)"
                ),
                "estimate": estimate,
                "ci_low": None if estimate is None else estimate - 0.05,
                "ci_high": None if estimate is None else estimate + 0.05,
                "conclusion": "inconclusive" if estimate is None else "positive",
                "clusters": 0 if estimate is None else 8,
                "total_clusters": 8,
                "complete_case_clusters": 0 if estimate is None else 8,
                "p_value": None if estimate is None else 0.05,
                "p_value_adjusted": None if estimate is None else 0.1,
                "inference_tier": "confirmatory" if index == 0 else "exploratory",
                "is_confirmatory": index == 0,
                "analysis_population": "base_trace_complete_case",
                "worst_case_bound": -0.25,
                "best_case_bound": 0.25,
                "divergent_coverage_gate_passed": True,
            }
        )
    return rows


def _lens_rows() -> list[dict]:
    return [
        {
            "lens_type": lens_type,
            "layer": 4,
            "position": position,
            "concept_set": "direction",
            "signed_contrast": 0.1 + position_index * 0.01,
            "eligible_trace_count": 24,
            "aggregation": "mean_over_common_probe_eligible_traces",
        }
        for lens_type in ("j", "r")
        for position_index, position in enumerate(
            (
                "prompt_end",
                "first_estimate_pre",
                "anchor_pre",
                "anchor_post",
                "final_answer_pre",
            )
        )
    ]


def _evidence() -> dict:
    return build_released_evidence(
        profile="qwen35_122b_primary",
        analysis_hash="sha256:" + "a" * 64,
        source_analysis_summary_sha256="b" * 64,
        lens_evidence_status="available_122b",
        behavior_rows=_behavior_rows(),
        effect_rows=_effect_rows(),
        lens_rows=_lens_rows(),
    )


def test_reproduce_results_authenticates_aggregates_and_regenerates_figures(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "reports/results"
    figure_dir = tmp_path / "reports/figures"
    manifest = write_release_bundle(
        project_root=tmp_path,
        results_dir=results_dir,
        figure_dir=Path("reports/figures"),
        evidence=_evidence(),
    )

    result = reproduce_release_bundle(
        project_root=tmp_path,
        results_dir=results_dir,
        figure_dir=figure_dir,
    )

    assert result["raw_generation_performed"] is False
    assert result["manifest_record_hash"] == manifest["record_hash"]
    assert set(result["figures"]) == {
        "first_vs_final_bias",
        "sentence_causal_effect_forest",
        "lens_layer_position_heatmap",
    }
    for path in result["figures"].values():
        assert Path(path).stat().st_size > 1_000
    for name, metadata in manifest["aggregate_tables"].items():
        assert sha256_file(results_dir / metadata["path"]) == metadata["sha256"], name


def test_released_evidence_rejects_extra_raw_trace_or_provider_fields() -> None:
    evidence = _evidence()
    evidence["behavior_stage_summary"][0]["raw_reasoning"] = "private chain of thought"
    evidence["record_hash"] = "sha256:" + "0" * 64

    with pytest.raises(PublicResultsError, match=r"record hash mismatch|fields changed"):
        validate_released_evidence(evidence)


def test_reproduce_results_fails_closed_on_table_hash_tampering(tmp_path: Path) -> None:
    results_dir = tmp_path / "reports/results"
    write_release_bundle(
        project_root=tmp_path,
        results_dir=results_dir,
        figure_dir=Path("reports/figures"),
        evidence=_evidence(),
    )
    manifest_path = results_dir / "results_manifest.json"
    manifest = read_json(manifest_path)
    manifest["aggregate_tables"]["sentence_effects"]["sha256"] = "0" * 64
    # Preserve valid JSON but intentionally do not forge the record hash.
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(PublicResultsError, match="record hash mismatch"):
        reproduce_release_bundle(
            project_root=tmp_path,
            results_dir=results_dir,
            figure_dir=tmp_path / "reports/figures",
        )


def test_nonestimable_effect_is_preserved_as_null_in_public_evidence() -> None:
    evidence = _evidence()
    row = evidence["sentence_effects"][0]
    assert row["estimate"] is None
    assert row["ci_low"] is None
    assert row["ci_high"] is None
    assert row["conclusion"] == "inconclusive"


def test_write_release_rejects_symlinked_table_directory(tmp_path: Path) -> None:
    results_dir = tmp_path / "reports/results"
    outside = tmp_path / "outside"
    outside.mkdir()
    results_dir.mkdir(parents=True)
    (results_dir / "tables").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PublicResultsError, match="symlink"):
        write_release_bundle(
            project_root=tmp_path,
            results_dir=results_dir,
            figure_dir=Path("reports/figures"),
            evidence=_evidence(),
        )


def test_reproduce_rejects_symlinked_table_output(tmp_path: Path) -> None:
    results_dir = tmp_path / "reports/results"
    write_release_bundle(
        project_root=tmp_path,
        results_dir=results_dir,
        figure_dir=Path("reports/figures"),
        evidence=_evidence(),
    )
    table = results_dir / "tables/behavior_stage_summary.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_text("untouched\n", encoding="utf-8")
    table.unlink()
    table.symlink_to(outside)

    with pytest.raises(PublicResultsError, match="symlink"):
        reproduce_release_bundle(
            project_root=tmp_path,
            results_dir=results_dir,
            figure_dir=tmp_path / "reports/figures",
        )
    assert outside.read_text(encoding="utf-8") == "untouched\n"


def test_write_release_rejects_symlinked_results_ancestor_before_escape(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PublicResultsError, match="traverses a symlink"):
        write_release_bundle(
            project_root=project,
            results_dir=project / "linked/results",
            figure_dir=Path("reports/figures"),
            evidence=_evidence(),
        )
    assert list(outside.iterdir()) == []


def test_render_release_rejects_in_project_symlinked_figure_ancestor(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    real = project / "real"
    real.mkdir(parents=True)
    (project / "alias").symlink_to(real, target_is_directory=True)

    with pytest.raises(PublicResultsError, match="traverses a symlink"):
        render_release_figures(
            project_root=project,
            evidence=_evidence(),
            figure_dir=project / "alias/figures",
        )
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("output_kind", ["results", "figures"])
def test_public_writers_reject_output_outside_project(
    tmp_path: Path, output_kind: str
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()

    with pytest.raises(PublicResultsError, match="inside the project"):
        if output_kind == "results":
            write_release_bundle(
                project_root=project,
                results_dir=outside / "results",
                figure_dir=Path("reports/figures"),
                evidence=_evidence(),
            )
        else:
            render_release_figures(
                project_root=project,
                evidence=_evidence(),
                figure_dir=outside / "figures",
            )
    assert list(outside.iterdir()) == []
