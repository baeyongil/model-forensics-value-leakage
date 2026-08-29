from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import model_forensics.cli as cli
from model_forensics.adjudication import AdjudicationRequest, JudgeProvenance
from model_forensics.io import read_json, read_jsonl, sha256_file, write_jsonl

PROJECT_PREREGISTRATION = Path(__file__).resolve().parents[1] / "config" / "preregistration.yaml"


class _ExactExternalJudge:
    not_for_primary_inference = False

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="fake-external",
            model_id=self._model_id,
            model_revision="frozen-route",
            caller_version="pytest",
            decoding={"temperature": 0},
            metadata={"input_tokens": 1, "output_tokens": 1},
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        if request.instrument_id == "target-final-v1":
            return json.dumps({"status": "KNOWN", "value": "100"})
        return json.dumps({"status": "KNOWN", "values": ["100"]})


def _write_config(
    tmp_path: Path,
    *,
    frozen: bool = True,
    backend: str = "fake_or_transformers",
) -> Path:
    config_path = tmp_path / "config" / "run.yaml"
    config_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "profile": "pytest_smoke",
        "preregistration": str(PROJECT_PREREGISTRATION),
        "model": {
            "id": "Qwen/Qwen3.5-4B" if backend != "vllm_offline" else "Qwen/Qwen3.5-122B-A10B",
            "revision": "a" * 40 if frozen else None,
            "require_pinned_revision": backend == "vllm_offline",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1 if backend != "vllm_offline" else 8,
            "max_model_len": 8192,
            "language_model_only": True,
        },
        "lenses": {
            "repository": "example/lenses",
            "revision": "b" * 40 if frozen else None,
            "require_pinned_revision": backend == "vllm_offline",
            "j_filename": "j/lens.pt",
            "r_filename": "r/lens.pt",
            **(
                {
                    "j_sha256": "c" * 64,
                    "r_sha256": "d" * 64,
                    "j_size_bytes": 100,
                    "r_size_bytes": 100,
                }
                if frozen and backend == "vllm_offline"
                else {}
            ),
        },
        "upstream": {
            "repository": "https://example.invalid/value-leakage.git",
            "commit": "1" * 40,
            "cache_dir": "data/upstream/value-leakage",
        },
        "paths": {
            "raw_dir": "data/raw/pytest",
            "interim_dir": "data/interim/pytest",
            "manifest_dir": "data/manifests/pytest",
            "figure_dir": "reports/figures/pytest",
            "report_dir": "reports/staging/pytest",
        },
        "execution": {
            "backend": backend,
            "secret_env": {},
            "gpu_cost_hard_stop_usd": 220.0 if backend == "vllm_offline" else 0.0,
            "api_cost_hard_stop_usd": 25.0 if backend == "vllm_offline" else 0.0,
            "total_cost_hard_stop_usd": 250.0 if backend == "vllm_offline" else 0.0,
            "terminate_compute_after_sync": True,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_parser_exposes_every_makefile_subcommand() -> None:
    parser = cli.build_parser()
    commands = {
        "reproduce",
        "sample",
        "anchors",
        "resample",
        "positions",
        "lens",
        "analyze",
        "report",
        "smoke",
        "clean",
    }
    for command in commands:
        parsed = parser.parse_args([command, "--config", "config/test.yaml"])
        assert parsed.command == command
        assert callable(parsed.handler)


def test_smoke_is_deterministic_network_free_and_stages_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)

    def unexpected_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("smoke attempted an upstream or model network path")

    monkeypatch.setattr(cli, "ensure_pinned_checkout", unexpected_network)
    monkeypatch.setattr(cli, "VLLMOfflineBackend", unexpected_network)

    assert cli.main(["smoke", "--config", str(config_path)]) == 0
    completion_path = tmp_path / "data/manifests/pytest/smoke_completion.json"
    completion = read_json(completion_path)
    assert completion["status"] == "complete"
    assert completion["synthetic_smoke"] is True
    assert completion["network_or_model_downloads"] is False
    assert completion["counts"] == {
        "anchor_candidates": 24,
        "anchors": 24,
        "lens": 2880,
        "resampling": 480,
        "rollouts": 44,
    }

    rollouts_path = tmp_path / completion["artifacts"]["rollouts"]
    rollouts = read_jsonl(rollouts_path)
    assert len({row["run_id"] for row in rollouts}) == 44
    assert all("rollout_id" not in row for row in rollouts)
    assert all(row["final_estimate"] is not None for row in rollouts)
    assert all(row["trajectory"]["features"]["final_estimate"] is not None for row in rollouts)
    assert all(row["seed"] and row["prompt_hash"] and row["model_hash"] for row in rollouts)

    for relative_path in completion["figures"].values():
        figure = tmp_path / relative_path
        assert figure.is_file()
        assert figure.stat().st_size > 1_000

    deterministic_hashes = {
        "completion": sha256_file(completion_path),
        "rollouts": sha256_file(rollouts_path),
        "resampling": sha256_file(tmp_path / completion["artifacts"]["resampling"]),
        "lens": sha256_file(tmp_path / completion["artifacts"]["lens"]),
    }
    assert cli.main(["smoke", "--config", str(config_path)]) == 0
    assert deterministic_hashes == {
        "completion": sha256_file(completion_path),
        "rollouts": sha256_file(rollouts_path),
        "resampling": sha256_file(tmp_path / completion["artifacts"]["resampling"]),
        "lens": sha256_file(tmp_path / completion["artifacts"]["lens"]),
    }

    assert cli.main(["report", "--config", str(config_path)]) == 0
    context = read_json(tmp_path / "reports/staging/pytest/result_context.json")
    assert context["synthetic_smoke"] is True
    assert context["executive_summary_status"].startswith("SMOKE DATA ONLY")
    markdown = (tmp_path / "reports/staging/pytest/result_context.md").read_text(encoding="utf-8")
    assert "J/R-lens evidence is observational" in markdown


def test_result_context_marks_nonestimable_causal_effects_as_na() -> None:
    context = {
        "title": "Result",
        "author": "Yongil Bae",
        "executive_summary_status": "PRIMARY RESULTS",
        "final_measurement_rate": 1.0,
        "cluster_effects": [
            {
                "sentence_class": "accuracy_commitment",
                "direction": "pooled",
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "conclusion": "inconclusive",
            }
        ],
        "hypothesis_verdicts": [],
        "figures": {},
        "lens_evidence_status": "unavailable_not_zero",
        "lens_resampling_association": {
            "status": "unavailable",
            "reason": "not measured",
        },
    }

    markdown = cli._context_markdown(context)

    assert "| accuracy_commitment | pooled | NA (not estimable) | NA (not estimable) |" in markdown


def test_sample_validation_alias_requires_completed_manifest_without_backend_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, frozen=False, backend="vllm_offline")
    constructed = False

    def forbidden_backend(*args: object, **kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(cli, "VLLMOfflineBackend", forbidden_backend)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sample", "--config", str(config_path)])
    assert exc_info.value.code == 2
    assert constructed is False
    assert "sampling manifest is absent" in capsys.readouterr().err


def test_sample_validation_alias_rejects_legacy_paid_route_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, backend="vllm_offline")
    constructed = False

    def forbidden_backend(*args: object, **kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(cli, "VLLMOfflineBackend", forbidden_backend)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "sample",
                "--config",
                str(config_path),
                "--judge-model",
                "anthropic/claude-opus-5",
                "--judge-input-price",
                "5",
                "--judge-output-price",
                "25",
            ]
        )
    assert exc_info.value.code == 2
    assert constructed is False
    assert "unrecognized arguments: --judge-model" in capsys.readouterr().err


def test_behavioral_execution_freezes_chicago_threshold_after_all_final_consensus(
    tmp_path: Path,
) -> None:
    config = cli.load_run_config(_write_config(tmp_path))
    preregistration = cli.load_preregistration(config)
    primary = _ExactExternalJudge("anthropic/claude-opus-5")
    independent = _ExactExternalJudge("google/gemini-3.1-pro-preview")
    counts = {
        "giraffe": {
            "baseline": 2,
            "threshold_only": 1,
            "above_good": 1,
            "below_good": 1,
        },
        "chicago_coffee": {"baseline": 3, "above_good": 1, "below_good": 1},
    }

    execution = cli._execute_behavioral_sampling(
        config,
        preregistration,
        cli.FakeBackend(cli._smoke_response),
        primary,
        independent_final_caller=independent,
        counts=counts,
        primary_inference=True,
        checkpoint_dir=tmp_path / "behavioral-checkpoint",
    )

    assert execution.thresholds["chicago_coffee"] == 100
    assert execution.final_consensus_summary["gate_passed"] is True
    assert execution.quality_gate["gate_passed"] is True
    assert len(execution.independent_final_records) == 10
    assert all(row["final_consensus_valid"] is True for row in execution.rows)
    assert all(request.instrument_id == "target-final-v1" for request in independent.requests)
    assert len(primary.requests) == 20  # primary final + trajectory for every rollout
    assert len(independent.requests) == 10  # independent final only
    chicago = execution.threshold_manifests["chicago_coffee"]
    assert chicago["threshold_rule"] == "median_of_known_exact_final_consensus"
    assert (tmp_path / "behavioral-checkpoint/baseline_consensus_summary.json").is_file()
    assert (tmp_path / "behavioral-checkpoint/behavioral_quality_gate.json").is_file()


def test_anchors_freezes_exact_manifest_from_canonical_candidate_jsonl(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    candidates_path = tmp_path / "data/interim/pytest/anchor_candidates.jsonl"
    classes = ("accuracy_commitment", "value_threshold_planning", "epistemic_control")
    directions = ("above_good", "below_good")
    rows = []
    for sentence_class in classes:
        for direction in directions:
            for ordinal, (side, flip) in enumerate(
                (("good", False), ("good", True), ("bad", False), ("bad", True))
            ):
                sentence = f"Candidate {sentence_class} {direction} {ordinal}."
                rows.append(
                    {
                        "run_id": f"{sentence_class}:{direction}:{ordinal}",
                        "sentence_class": sentence_class,
                        "condition": direction,
                        "sentence_index": ordinal,
                        "sentence_text": sentence,
                        "char_start": 100,
                        "char_end": 100 + len(sentence),
                        "initial_side": side,
                        "final_flip": flip,
                        "eligible": True,
                        "synthetic_smoke": True,
                    }
                )
    write_jsonl(candidates_path, rows)

    assert cli.main(["anchors", "--config", str(config_path)]) == 0
    manifest = read_json(tmp_path / "data/manifests/pytest/anchor_manifest.json")
    assert len(manifest["anchors"]) == 24
    assert len({anchor["trace_id"] for anchor in manifest["anchors"]}) == 24
    assert manifest["candidate_count"] == 24
    assert manifest["candidate_file_sha256"] == sha256_file(candidates_path)
    assert len(manifest["selection_hash"]) == 64


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("resample", "validation-only"),
        ("lens", "frozen GPU/software lock"),
    ],
)
def test_gpu_specific_commands_fail_safely_without_prepared_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    expected: str,
) -> None:
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        cli.main([command, "--config", str(config_path)])
    assert exc_info.value.code == 2
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "data/manifests/pytest/resampling_validation.json").exists()
    assert not (tmp_path / "data/manifests/pytest/lens_validation.json").exists()


def test_clean_removes_only_strictly_bounded_ignored_outputs(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    removable = (
        tmp_path / "data/raw/pytest/payload.jsonl",
        tmp_path / "data/interim/pytest/payload.jsonl",
        tmp_path / "reports/staging/pytest/context.json",
    )
    for path in removable:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated", encoding="utf-8")
    figure = tmp_path / "reports/figures/pytest/keep.png"
    manifest = tmp_path / "data/manifests/pytest/keep.json"
    for path in (figure, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserve", encoding="utf-8")

    assert cli.main(["clean", "--config", str(config_path)]) == 0
    assert all(not path.exists() for path in removable)
    assert figure.read_text(encoding="utf-8") == "preserve"
    assert manifest.read_text(encoding="utf-8") == "preserve"


def test_clean_rejects_a_broad_configured_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paths"]["raw_dir"] = "data/raw"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["clean", "--config", str(config_path)])
    assert exc_info.value.code == 2
    assert "refusing to clean unbounded path" in capsys.readouterr().err
