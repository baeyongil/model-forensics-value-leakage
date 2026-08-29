from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import model_forensics.cli as cli
from model_forensics.anchors import FrozenAnchor
from model_forensics.config import load_preregistration, load_run_config
from model_forensics.io import sha256_file, stable_hash, write_json, write_jsonl
from model_forensics.lens_runner import (
    COMPATIBILITY_ATTEMPT_PREFIX_STREAM,
    SMOKE_MODEL_ID,
    SMOKE_MODEL_REVISION,
)
from model_forensics.token_spans import token_stream_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LENS_VERDICT_CRITERIA = {
    "generic_jr_direction_corroboration",
    "direction_signal_present_before_first_estimate",
    "direction_signal_precedes_accuracy_statement",
    "objective_signal_increases_after_accuracy_sentence",
}


def _available_association(*, j_tau: float = 0.5, r_tau: float = 0.25) -> dict:
    return {
        "status": "available",
        "designed_traces_per_direction": {"above_good": 4, "below_good": 4},
        "common_trace_count": 8,
        "traces_per_direction": {"above_good": 4, "below_good": 4},
        "permutation_count": 576,
        "per_lens": {
            "J": {"tau_a": j_tau},
            "R": {"tau_a": r_tau},
        },
    }


def test_lens_verdict_gate_requires_exact_universe_and_strict_positive_jr_tau() -> None:
    raw_values = {name: True for name in _LENS_VERDICT_CRITERIA}
    raw_values["behavioral_control"] = True
    raw_reasons = {name: "raw diagnostic passed" for name in raw_values}

    gated, reasons, gate = cli._gate_lens_verdict_criteria(
        raw_values,
        raw_reasons,
        association=_available_association(),
    )
    assert gate["passed"] is True
    assert gated == raw_values
    assert reasons == raw_reasons

    for association in (
        {**_available_association(), "status": "unavailable"},
        {**_available_association(), "permutation_count": 575},
        _available_association(j_tau=-0.01),
        _available_association(r_tau=0.0),
    ):
        gated, reasons, gate = cli._gate_lens_verdict_criteria(
            raw_values,
            raw_reasons,
            association=association,
        )
        assert gate["passed"] is False
        assert gated["behavioral_control"] is True
        assert all(gated[name] is None for name in _LENS_VERDICT_CRITERIA)
        assert all("association corroboration gate failed" in reasons[name] for name in _LENS_VERDICT_CRITERIA)


def _anchor(index: int) -> FrozenAnchor:
    direction = "above_good" if index < 12 else "below_good"
    sentence_class = (
        "accuracy_commitment"
        if index % 3 == 0
        else ("value_threshold_planning" if index % 3 == 1 else "epistemic_control")
    )
    return FrozenAnchor(
        anchor_id=f"anchor-{index}",
        trace_id=f"trace-{index}",
        sentence_class=sentence_class,
        direction=direction,
        sentence_index=0,
        sentence_text="Sentence.",
        char_start=0,
        char_end=9,
        initial_side="good",
        final_flip=False,
        provenance={},
    )


def _complete_resampling() -> tuple[list[dict], dict[str, FrozenAnchor]]:
    anchors = {_anchor(index).anchor_id: _anchor(index) for index in range(24)}
    rows: list[dict] = []
    for anchor in anchors.values():
        for arm in ("retain", "resample"):
            for sample_index in range(20):
                row = {
                    "resample_id": f"{anchor.anchor_id}:{arm}:{sample_index}",
                    "anchor_id": anchor.anchor_id,
                    "base_trace_id": anchor.trace_id,
                    "sentence_class": anchor.sentence_class,
                    "condition": anchor.direction,
                    "arm": arm,
                    "sample_index": sample_index,
                    "stage": "initial" if sample_index < 10 else "stage_two",
                    "divergent": arm == "resample",
                    "final_good_side": True,
                }
                row["record_hash"] = stable_hash(row)
                rows.append(row)
    return rows, anchors


def _write_config(tmp_path: Path, *, backend: str) -> Path:
    path = tmp_path / "config" / "run.yaml"
    path.parent.mkdir(parents=True)
    primary = backend == "vllm_offline"
    payload = {
        "schema_version": 1,
        "profile": "gate_primary" if primary else "gate_smoke",
        "preregistration": str(PROJECT_ROOT / "config/preregistration.yaml"),
        "model": {
            "id": "Qwen/Qwen3.5-122B-A10B" if primary else "Qwen/Qwen3.5-4B",
            "revision": "dc4d348443bc740c68e2d77492492c11606384d5",
            "require_pinned_revision": primary,
            "dtype": "bfloat16",
            "tensor_parallel_size": 8 if primary else 1,
            "max_model_len": 65536,
            "language_model_only": True,
        },
        "lenses": {
            "repository": "camilablank/workspace-lenses",
            "revision": "d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
            "require_pinned_revision": primary,
            "j_filename": "j/lens.pt",
            "r_filename": "r/lens.pt",
            "j_sha256": "a" * 64 if primary else None,
            "r_sha256": "b" * 64 if primary else None,
            "j_size_bytes": 1 if primary else None,
            "r_size_bytes": 1 if primary else None,
        },
        "upstream": {
            "repository": "https://example.invalid/value-leakage.git",
            "commit": "1" * 40,
            "cache_dir": "data/upstream/value-leakage",
        },
        "paths": {
            "raw_dir": "data/raw/gate",
            "interim_dir": "data/interim/gate",
            "manifest_dir": "data/manifests/gate",
            "figure_dir": "reports/figures/gate",
            "report_dir": "reports/staging/gate",
        },
        "execution": {
            "backend": backend,
            "secret_env": {},
            "gpu_cost_hard_stop_usd": 220 if primary else 0,
            "api_cost_hard_stop_usd": 100 if primary else 0,
            "total_cost_hard_stop_usd": 325 if primary else 0,
            "terminate_compute_after_sync": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_primary_partition_rejects_mixed_or_smoke_rows() -> None:
    with pytest.raises(cli.CLIError, match="mixed smoke/primary"):
        cli._validate_analysis_data_partition(
            primary=True,
            rollouts=[{}, {"synthetic_smoke": True}],
            resampling_rows=[{}],
            lens_rows=[{}],
        )


def test_smoke_partition_requires_every_row_to_be_explicitly_marked() -> None:
    with pytest.raises(cli.CLIError, match="explicitly labelled smoke"):
        cli._validate_analysis_data_partition(
            primary=False,
            rollouts=[{}],
            resampling_rows=[{"synthetic_smoke": True}],
            lens_rows=[{"synthetic_smoke": True}],
        )
    def smoke_row(label: str) -> dict:
        row = {"label": label, "synthetic_smoke": True}
        row["record_hash"] = stable_hash(row)
        return row

    assert cli._validate_analysis_data_partition(
        primary=False,
        rollouts=[smoke_row("rollout")],
        resampling_rows=[smoke_row("resampling")],
        lens_rows=[smoke_row("lens")],
    )
    tampered = smoke_row("tampered")
    tampered["label"] = "changed"
    with pytest.raises(cli.CLIError, match="record_hash mismatch"):
        cli._validate_analysis_data_partition(
            primary=False,
            rollouts=[smoke_row("rollout")],
            resampling_rows=[smoke_row("resampling")],
            lens_rows=[tampered],
        )
    with pytest.raises(cli.CLIError, match="refuses synthetic smoke"):
        cli._validate_analysis_data_partition(
            primary=True,
            rollouts=[{"synthetic_smoke": True}],
            resampling_rows=[{"synthetic_smoke": True}],
            lens_rows=[{"synthetic_smoke": True}],
        )


def test_primary_resampling_requires_exact_complete_inventory() -> None:
    rows, anchors = _complete_resampling()
    cli._validate_primary_resampling_inventory(rows, anchor_by_id=anchors)
    with pytest.raises(cli.CLIError, match="incomplete"):
        cli._validate_primary_resampling_inventory(rows[:-1], anchor_by_id=anchors)


def test_primary_lens_grid_rejects_tamper_and_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_run_config(PROJECT_ROOT / "config/run_122b.yaml")
    preregistration = load_preregistration(config)
    concepts = cli._expected_probe_concepts(preregistration)
    trace = SimpleNamespace(
        trace_id="trace-0",
        position_indices={"prompt_end": 0},
        sequence_token_ids=(999,),
        good_side_direction=1,
    )
    validated = SimpleNamespace(traces=(trace,))
    monkeypatch.setattr(cli, "POSITION_ORDER", ("prompt_end",))
    monkeypatch.setattr(cli, "_PRIMARY_ANALYSIS_LAYERS", (4,))
    monkeypatch.setattr(cli, "_PRIMARY_ANALYSIS_LENS_RECORD_COUNT", 6)
    design = {"manifest_hash": "sha256:" + "c" * 64, "eligible_cell_count": 3}
    cells = {}
    for concept in concepts:
        key = ("trace-0", "prompt_end", concept)
        cell = {
            "record_hash": stable_hash(key),
            "probe_eligible": True,
            "probe_ineligibility_reason": None,
            "collision_evidence_hash": None,
            "causal_prefix_token_count": 1,
            "causal_prefix_token_ids_hash": token_stream_hash(
                (999,), stream="lens_causal_prefix"
            ),
        }
        cells[key] = cell
    rows = []
    for lens_type, lens_hash in (
        ("J", config.lenses.j_sha256),
        ("R", config.lenses.r_sha256),
    ):
        for concept, concept_spec in concepts.items():
            cell = cells[("trace-0", "prompt_end", concept)]
            row = {
                "schema_version": 2,
                "trace_id": "trace-0",
                "prefix_sha256": hashlib.sha256(
                    json.dumps([999], separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "model_id": config.model.id,
                "lens_type": lens_type,
                "lens_file_sha256": lens_hash,
                "target_layer": 46,
                "layer": 4,
                "layer_band": "early",
                "position_name": "prompt_end",
                "token_index": 0,
                "contrast": concept,
                "raw_mean_logit_contrast": 0.1,
                "signed_mean_logit_contrast": 0.1,
                "good_side_direction": 1,
                "positive_token_ids": concept_spec["positive_token_ids"],
                "negative_token_ids": concept_spec["negative_token_ids"],
                "evidence_scope": "observational_readout",
                "causal_claim": False,
                "probe_design_hash": design["manifest_hash"],
                "probe_eligibility_record_hash": cell["record_hash"],
                "probe_eligible": True,
                "probe_ineligibility_reason": None,
                "collision_evidence_hash": None,
                "causal_prefix_token_ids_hash": cell["causal_prefix_token_ids_hash"],
                "causal_prefix_token_count": 1,
                "forward_input_token_ids_hash": token_stream_hash(
                    (999,), stream="lens_forward_input"
                ),
                "forward_input_token_count": 1,
            }
            row["record_hash"] = stable_hash(row)
            rows.append(row)
    assert len(
        cli._validate_primary_lens_grid(
            config=config,
            preregistration=preregistration,
            raw_rows=rows,
            validated_lens_inputs=validated,
            probe_design=design,
            probe_cells=cells,
        )
    ) == 6
    with pytest.raises(cli.CLIError, match="30,960-row"):
        cli._validate_primary_lens_grid(
            config=config,
            preregistration=preregistration,
            raw_rows=rows[:-1],
            validated_lens_inputs=validated,
            probe_design=design,
            probe_cells=cells,
        )
    rows[0]["signed_mean_logit_contrast"] = 999
    with pytest.raises(cli.CLIError, match="record_hash mismatch"):
        cli._validate_primary_lens_grid(
            config=config,
            preregistration=preregistration,
            raw_rows=rows,
            validated_lens_inputs=validated,
            probe_design=design,
            probe_cells=cells,
        )
    rows[0]["record_hash"] = stable_hash(
        {key: value for key, value in rows[0].items() if key != "record_hash"}
    )
    with pytest.raises(cli.CLIError, match="mis-signed contrast"):
        cli._validate_primary_lens_grid(
            config=config,
            preregistration=preregistration,
            raw_rows=rows,
            validated_lens_inputs=validated,
            probe_design=design,
            probe_cells=cells,
        )


def test_probe_design_rejects_lexical_collision_flag_not_recomputed_by_pinned_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recomputed = {
        "manifest_hash": "sha256:" + "a" * 64,
        "cells": [
            {
                "collisions": [
                    {
                        "word": " upward",
                        "token_id": 38453,
                        "exact_token_id_present": False,
                        "lexical_word_present": True,
                    }
                ]
            }
        ],
    }
    monkeypatch.setattr(
        cli,
        "freeze_production_probe_design",
        lambda *args, **kwargs: SimpleNamespace(
            to_manifest=lambda **ignored: recomputed
        ),
    )
    observed = json.loads(json.dumps(recomputed))
    observed["cells"][0]["collisions"][0]["lexical_word_present"] = False
    with pytest.raises(cli.CLIError, match="pinned-tokenizer recomputation"):
        cli._require_recomputed_probe_design(
            observed=observed,
            validated_lens_inputs=SimpleNamespace(),
            candidate_probe_manifest_hash="sha256:" + "b" * 64,
            candidate_probe_manifest_sha256="c" * 64,
        )


def test_failed_primary_gate_writes_no_statistics_or_figures(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, backend="vllm_offline")
    config = load_run_config(config_path)
    raw = tmp_path / config.paths.raw_dir / "rollouts.jsonl"
    interim = tmp_path / config.paths.interim_dir
    write_jsonl(raw, [{"run_id": "incomplete"}])
    write_jsonl(interim / "resampling.jsonl", [{"arm": "retain"}])
    write_jsonl(interim / "lens.jsonl", [{"trace_id": "incomplete"}])

    with pytest.raises(cli.CLIError):
        cli._analyze_artifacts(config, load_preregistration(config))
    assert not (tmp_path / config.paths.figure_dir).exists()
    assert not (tmp_path / config.paths.report_dir).exists()


def test_report_rechecks_all_hashes_before_writing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, backend="fake_or_transformers")
    config = load_run_config(config_path)
    report_dir = tmp_path / config.paths.report_dir
    report_dir.mkdir(parents=True)

    def artifact(relative: str, content: bytes) -> dict[str, str]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": relative, "sha256": sha256_file(path)}

    inputs = {
        name: artifact(f"evidence/{name}.jsonl", name.encode())
        for name in ("rollouts", "resampling", "lens")
    }
    table_names = (
        "behavioral_estimands",
        "behavior",
        "missingness",
        "timing",
        "process",
        "effects",
        "criteria",
        "verdicts",
    )
    tables = {name: artifact(f"tables/{name}.jsonl", name.encode()) for name in table_names}
    figures = {
        name: artifact(f"figures/{name}.png", name.encode())
        for name in (
            "first_vs_final_bias",
            "sentence_causal_effect_forest",
            "lens_layer_position_heatmap",
        )
    }
    summary = {
        "schema_version": 2,
        "profile": config.profile,
        "synthetic_smoke": True,
        "lens_evidence_status": "synthetic_smoke",
        "lens_eligible_rows": 1,
        "final_measurement_rate": 1.0,
        "cluster_effects": [],
        "hypothesis_criteria": [],
        "hypothesis_verdicts": [],
        "lens_resampling_association": {
            "status": "unavailable",
            "reason": (
                "deterministic smoke evidence is not eligible for the primary "
                "lens-resampling association"
            ),
            "inference_tier": "exploratory_observational",
            "causal_claim": False,
            "mediation_claim": False,
            "primary_lens": "J",
            "sensitivity_lens": "R",
            "trace_effects": [],
            "common_trace_count": 0,
            "traces_per_direction": {"above_good": 0, "below_good": 0},
            "permutation_count": 0,
            "permutation_resolution": None,
            "per_lens": {},
        },
        "inputs": inputs,
        "tables": tables,
        "figures": figures,
    }
    summary["analysis_hash"] = stable_hash(summary)
    write_json(report_dir / "analysis_summary.json", summary)
    (tmp_path / tables["effects"]["path"]).write_text("tampered", encoding="utf-8")

    with pytest.raises(cli.CLIError, match=r"tables\.effects SHA-256 mismatch"):
        cli._command_report(Namespace(config=str(config_path)))
    assert not (report_dir / "result_context.json").exists()
    assert not (report_dir / "result_context.md").exists()
    (tmp_path / tables["effects"]["path"]).write_bytes(b"effects")
    summary["analysis_hash"] = "sha256:" + "0" * 64
    write_json(report_dir / "analysis_summary.json", summary)
    with pytest.raises(cli.CLIError, match="analysis_hash mismatch"):
        cli._command_report(Namespace(config=str(config_path)))
    assert not (report_dir / "result_context.json").exists()
    assert not (report_dir / "result_context.md").exists()


def test_report_recomputes_association_from_linked_raw_inputs(tmp_path: Path) -> None:
    resampling: list[dict] = []
    lens: list[dict] = []
    for direction in ("above_good", "below_good"):
        for trace_index, good_count in enumerate((0, 2, 4, 6)):
            trace_id = f"{direction}-{trace_index}"
            for sample_index in range(8):
                for arm in ("retain", "resample"):
                    resampling.append(
                        {
                            "anchor_id": trace_id,
                            "base_trace_id": trace_id,
                            "sentence_class": "accuracy_commitment",
                            "condition": direction,
                            "sample_index": sample_index,
                            "seed": sample_index,
                            "stage": "initial",
                            "arm": arm,
                            "intervention_eligible": True,
                            "divergent": arm == "resample",
                            "final_good_side": int(
                                arm == "resample" and sample_index < good_count
                            ),
                        }
                    )
            change = good_count / 8
            for lens_type, scale in (("J", 1.0), ("R", 0.8)):
                for layer in range(4, 47):
                    for position, value in (
                        ("anchor_pre", 0.0),
                        ("anchor_post", change),
                    ):
                        lens.append(
                            {
                                "trace_id": trace_id,
                                "lens_type": lens_type,
                                "layer": layer,
                                "position_name": position,
                                "contrast": "epistemic",
                                "signed_mean_logit_contrast": value * scale,
                                "probe_eligible": True,
                            }
                        )
    resampling_path = tmp_path / "resampling.jsonl"
    lens_path = tmp_path / "lens.jsonl"
    write_jsonl(resampling_path, resampling)
    write_jsonl(lens_path, lens)
    expected = cli.accuracy_anchor_lens_resampling_association(resampling, lens)
    cli._require_recomputed_lens_resampling_association(
        association=expected,
        resampling_path=resampling_path,
        lens_path=lens_path,
        minimum_pairs_per_trace=8,
    )
    tampered = json.loads(json.dumps(expected))
    tampered["per_lens"]["J"]["tau_a"] = -1.0
    with pytest.raises(cli.CLIError, match="raw-input recomputation"):
        cli._require_recomputed_lens_resampling_association(
            association=tampered,
            resampling_path=resampling_path,
            lens_path=lens_path,
            minimum_pairs_per_trace=8,
        )

def test_authenticated_lens_failure_is_behavior_only_and_rejects_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, backend="vllm_offline")
    config = load_run_config(config_path)
    manifest_dir = tmp_path / config.paths.manifest_dir
    manifest_dir.mkdir(parents=True)
    candidate_path = manifest_dir / "lens_probe_token_verification.json"
    design_path = manifest_dir / "lens_probe_design_manifest.json"
    write_json(candidate_path, {"candidate": True})
    design_payload = {"design": True}
    design_payload["manifest_hash"] = stable_hash(design_payload)
    write_json(design_path, design_payload)
    trace = SimpleNamespace(
        trace_id="trace-0",
        sequence_token_ids=(10, 11),
        position_indices={"final_answer_pre": 1},
    )
    validated = SimpleNamespace(
        anchor_manifest_hash="sha256:" + "1" * 64,
        anchor_selection_hash="sha256:" + "2" * 64,
        position_manifest_hash="sha256:" + "3" * 64,
        rollout_manifest_hash="sha256:" + "4" * 64,
        traces=(trace,),
    )
    probe_design = {
        "manifest_hash": design_payload["manifest_hash"],
        "candidate_probe_manifest_hash": "sha256:" + "6" * 64,
        "protocol_version": "fixed-common-probes-causal-cell-eligibility-v1",
    }
    monkeypatch.setattr(cli, "encode_frozen_4b_compatibility_prefix", lambda: (7,))
    prefixes = cli.freeze_production_compatibility_prefixes(
        validated,
        four_b_token_ids=(7,),
    )
    prefix_manifest = prefixes.to_manifest().to_dict(include_hash=True)
    prefix_path = manifest_dir / "lens_compatibility_prefix_manifest.json"
    write_json(prefix_path, prefix_manifest)
    attempts = [
        {
            "ordinal": 1,
            "stage": "4b_smoke",
            "strategy": "pinned_text_only_single_forward",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": SMOKE_MODEL_REVISION,
            "prefix_token_count": 1,
            "prefix_token_ids_hash": prefix_manifest["four_b"]["token_ids_hash"],
            "status": "passed",
            "details": {},
            "error_type": None,
            "error_message": None,
        },
        {
            "ordinal": 1,
            "stage": "122b_preflight",
            "strategy": "version_fixed_full_prefix",
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "prefix_token_count": 2,
            "prefix_token_ids_hash": prefix_manifest["primary_full"][
                "token_ids_hash"
            ],
            "status": "failed",
            "details": {},
            "error_type": "RuntimeError",
            "error_message": "failed",
        },
        {
            "ordinal": 2,
            "stage": "122b_preflight",
            "strategy": "version_fixed_shortened_prefix",
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "prefix_token_count": 1,
            "prefix_token_ids_hash": prefix_manifest["primary_short"][
                "token_ids_hash"
            ],
            "status": "failed",
            "details": {},
            "error_type": "RuntimeError",
            "error_message": "failed",
        },
    ]
    compatibility = {
        "attempts": attempts,
        "primary_ready": False,
        "transformers_revision": cli.TRANSFORMERS_REVISION,
        "jlens_revision": cli.JLENS_REVISION,
        "maximum_122b_attempts": 2,
        "fallback_model_used": False,
        "fallback_policy": "27B_methodology_support_only_not_122B_substitute",
        "schema_version": 1,
    }
    compatibility["record_hash"] = stable_hash(compatibility)
    compatibility_path = manifest_dir / "lens_compatibility_manifest.json"
    write_json(compatibility_path, compatibility)
    paid_plan = {
        "schema_version": 1,
        "protocol_version": "lens-gpu-paid-plan-v2",
        "command_phase": "lens_gpu",
        "approval_bindings_hash": "sha256:" + "8" * 64,
    }
    paid_plan["plan_hash"] = stable_hash(paid_plan)
    paid_receipt = {
        "schema_version": 1,
        "protocol_version": cli.PAID_PHASE_RECEIPT_PROTOCOL,
        "command_phase": "lens_gpu",
        "approval_content_hash": "sha256:" + "9" * 64,
        "approval_id_hash": "sha256:" + "a" * 64,
        "bindings_hash": paid_plan["approval_bindings_hash"],
        "plan_hash": paid_plan["plan_hash"],
    }
    paid_receipt["receipt_hash"] = stable_hash(paid_receipt)
    active_gate = {
        "schema_version": 1,
        "protocol_version": "active-runpod-session-v1",
        "phase": "lens_gpu",
        "session_hash": "sha256:" + "b" * 64,
        "passed": True,
    }
    active_gate["record_hash"] = stable_hash(active_gate)
    release_authorization = cli._persist_lens_release_authorization(
        config=config,
        paid_plan=paid_plan,
        paid_receipt=paid_receipt,
        active_gpu_gate=active_gate,
        probe_design_path=design_path,
    )
    release_authorization_path = manifest_dir / "lens_release_authorization.json"
    expected = {
        "schema_version": 2,
        "status": "primary_122b_lens_unavailable",
        "failure_stage": "ordered_122b_compatibility_gate",
        "failure_policy": "two_bounded_version_fixed_attempts_then_behavior_only",
        "primary_model_id": config.model.id,
        "primary_model_revision": config.model.revision,
        "anchor_manifest_hash": validated.anchor_manifest_hash,
        "anchor_selection_hash": validated.anchor_selection_hash,
        "position_manifest_hash": validated.position_manifest_hash,
        "rollout_manifest_hash": validated.rollout_manifest_hash,
        "probe_design_manifest_hash": probe_design["manifest_hash"],
        "probe_design_manifest_sha256": sha256_file(design_path),
        "candidate_probe_manifest_hash": probe_design["candidate_probe_manifest_hash"],
        "candidate_probe_manifest_sha256": sha256_file(candidate_path),
        "probe_protocol_version": probe_design["protocol_version"],
        "compatibility_prefix_manifest_hash": prefix_manifest["record_hash"],
        "compatibility_prefix_manifest_sha256": sha256_file(prefix_path),
        "compatibility_manifest_hash": compatibility["record_hash"],
        "compatibility_manifest_sha256": sha256_file(compatibility_path),
        "release_authorization_manifest_hash": release_authorization["manifest_hash"],
        "release_authorization_manifest_sha256": sha256_file(
            release_authorization_path
        ),
        "attempt_count_122b": 2,
        "attempt_strategies": [
            "version_fixed_full_prefix",
            "version_fixed_shortened_prefix",
        ],
        "all_122b_attempts_failed": True,
        "lens_records_absent": True,
        "execution_manifest_absent": True,
        "analysis_mode": "behavior_only",
        "lens_evidence_status": "unavailable_not_zero",
        "lens_claim_eligibility": False,
        "fallback_27b_policy": "methodology_support_only_not_122b_substitute",
        "fallback_27b_used_as_primary": False,
        "causal_claim": False,
    }
    failure = dict(expected)
    failure["record_hash"] = stable_hash(failure)
    failure_path = manifest_dir / "lens_failure_manifest.json"
    write_json(failure_path, failure)
    lens_path = tmp_path / config.paths.interim_dir / "lens.jsonl"

    evidence = cli._authenticate_lens_failure_release(
        config=config,
        validated_lens_inputs=validated,
        probe_design=probe_design,
        lens_artifact=lens_path,
    )
    assert set(evidence) == {
        "lens_compatibility_prefix_manifest",
        "lens_compatibility_manifest",
        "lens_release_authorization",
        "lens_failure_manifest",
    }

    failure["fallback_27b_used_as_primary"] = True
    failure["record_hash"] = stable_hash(
        {key: value for key, value in failure.items() if key != "record_hash"}
    )
    write_json(failure_path, failure)
    with pytest.raises(cli.CLIError, match="fallback_27b_used_as_primary"):
        cli._authenticate_lens_failure_release(
            config=config,
            validated_lens_inputs=validated,
            probe_design=probe_design,
            lens_artifact=lens_path,
        )


def test_compatibility_loader_requires_exact_recomputed_prefix_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_run_config(_write_config(tmp_path, backend="vllm_offline"))
    trace = SimpleNamespace(
        trace_id="trace-0",
        sequence_token_ids=(10, 11, 12, 13),
        position_indices={"final_answer_pre": 3},
    )
    validated = SimpleNamespace(traces=(trace,))
    monkeypatch.setattr(cli, "encode_frozen_4b_compatibility_prefix", lambda: (7, 8))
    prefixes = cli.freeze_production_compatibility_prefixes(
        validated,
        four_b_token_ids=(7, 8),
    )
    prefix_manifest = prefixes.to_manifest().to_dict(include_hash=True)
    prefix_path = tmp_path / "prefixes.json"
    write_json(prefix_path, prefix_manifest)
    attempts = [
        {
            "ordinal": 1,
            "stage": "4b_smoke",
            "strategy": "pinned_text_only_single_forward",
            "model_id": SMOKE_MODEL_ID,
            "model_revision": SMOKE_MODEL_REVISION,
            "prefix_token_count": 2,
            "prefix_token_ids_hash": token_stream_hash(
                (7, 8), stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
            ),
            "status": "passed",
            "details": {},
            "error_type": None,
            "error_message": None,
        },
        {
            "ordinal": 1,
            "stage": "122b_preflight",
            "strategy": "version_fixed_full_prefix",
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "prefix_token_count": 4,
            "prefix_token_ids_hash": token_stream_hash(
                (10, 11, 12, 13), stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
            ),
            "status": "passed",
            "details": {},
            "error_type": None,
            "error_message": None,
        },
    ]
    compatibility = {
        "attempts": attempts,
        "primary_ready": True,
        "transformers_revision": cli.TRANSFORMERS_REVISION,
        "jlens_revision": cli.JLENS_REVISION,
        "maximum_122b_attempts": 2,
        "fallback_model_used": False,
        "fallback_policy": "27B_methodology_support_only_not_122B_substitute",
        "schema_version": 1,
    }
    compatibility["record_hash"] = stable_hash(compatibility)
    compatibility_path = tmp_path / "compatibility.json"
    write_json(compatibility_path, compatibility)
    cli._load_lens_compatibility_manifest(
        config=config,
        path=compatibility_path,
        prefix_path=prefix_path,
        validated_lens_inputs=validated,
        require_ready=True,
    )
    compatibility["attempts"][1]["prefix_token_ids_hash"] = "sha256:" + "f" * 64
    compatibility["record_hash"] = stable_hash(
        {key: value for key, value in compatibility.items() if key != "record_hash"}
    )
    write_json(compatibility_path, compatibility)
    with pytest.raises(cli.CLIError, match="exact prefix"):
        cli._load_lens_compatibility_manifest(
            config=config,
            path=compatibility_path,
            prefix_path=prefix_path,
            validated_lens_inputs=validated,
            require_ready=True,
        )


def test_lens_release_authorization_persists_all_gates_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config = load_run_config(_write_config(tmp_path, backend="vllm_offline"))
    manifest_dir = tmp_path / config.paths.manifest_dir
    probe_path = manifest_dir / "lens_probe_design_manifest.json"
    probe = {"schema_version": 1, "probe": "frozen"}
    probe["manifest_hash"] = stable_hash(probe)
    write_json(probe_path, probe)
    bindings_hash = "sha256:" + "b" * 64
    plan = {
        "schema_version": 1,
        "protocol_version": "lens-gpu-paid-plan-v2",
        "command_phase": "lens_gpu",
        "approval_bindings_hash": bindings_hash,
    }
    plan["plan_hash"] = stable_hash(plan)
    receipt = {
        "schema_version": 1,
        "protocol_version": cli.PAID_PHASE_RECEIPT_PROTOCOL,
        "command_phase": "lens_gpu",
        "approval_content_hash": "sha256:" + "c" * 64,
        "approval_id_hash": "sha256:" + "d" * 64,
        "bindings_hash": bindings_hash,
        "plan_hash": plan["plan_hash"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    active = {
        "schema_version": 1,
        "protocol_version": "active-runpod-session-v1",
        "phase": "lens_gpu",
        "session_hash": "sha256:" + "e" * 64,
        "passed": True,
    }
    active["record_hash"] = stable_hash(active)

    authorization = cli._persist_lens_release_authorization(
        config=config,
        paid_plan=plan,
        paid_receipt=receipt,
        active_gpu_gate=active,
        probe_design_path=probe_path,
    )
    assert authorization == cli._load_lens_release_authorization(
        config=config,
        probe_design_path=probe_path,
    )
    assert authorization == cli._persist_lens_release_authorization(
        config=config,
        paid_plan=plan,
        paid_receipt=receipt,
        active_gpu_gate=active,
        probe_design_path=probe_path,
    )
    active_path = manifest_dir / "lens_active_gpu_session_gate.json"
    tampered = dict(active)
    tampered["passed"] = False
    tampered["record_hash"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "record_hash"}
    )
    write_json(active_path, tampered)
    with pytest.raises(cli.CLIError, match="active GPU session"):
        cli._load_lens_release_authorization(
            config=config,
            probe_design_path=probe_path,
        )


def test_lens_release_authorization_refuses_pre_gate_writes(tmp_path: Path) -> None:
    config = load_run_config(_write_config(tmp_path, backend="vllm_offline"))
    manifest_dir = tmp_path / config.paths.manifest_dir
    probe_path = manifest_dir / "lens_probe_design_manifest.json"
    write_json(probe_path, {"manifest_hash": "sha256:" + "a" * 64})
    plan = {
        "command_phase": "lens_gpu",
        "approval_bindings_hash": "sha256:" + "b" * 64,
    }
    plan["plan_hash"] = stable_hash(plan)
    receipt = {
        "command_phase": "lens_gpu",
        "bindings_hash": plan["approval_bindings_hash"],
        "plan_hash": plan["plan_hash"],
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    active = {"phase": "lens_gpu", "passed": False}
    active["record_hash"] = stable_hash(active)
    with pytest.raises(cli.CLIError, match="active GPU session"):
        cli._persist_lens_release_authorization(
            config=config,
            paid_plan=plan,
            paid_receipt=receipt,
            active_gpu_gate=active,
            probe_design_path=probe_path,
        )
    assert not (manifest_dir / "lens_paid_plan.json").exists()
    assert not (manifest_dir / "lens_paid_receipt.json").exists()
    assert not (manifest_dir / "lens_active_gpu_session_gate.json").exists()


def test_success_lens_root_rejects_coexisting_failure_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, backend="vllm_offline")
    config = load_run_config(config_path)
    manifest_dir = tmp_path / config.paths.manifest_dir
    write_json(manifest_dir / "lens_failure_manifest.json", {"terminal": True})

    with pytest.raises(cli.CLIError, match="coexisting terminal failure manifest"):
        cli._authenticate_lens_release(
            config=config,
            artifact=tmp_path / config.paths.interim_dir / "lens.jsonl",
            raw_rows=[],
            validated_lens_inputs=SimpleNamespace(),
            probe_design={},
            eligible_row_count=0,
        )
