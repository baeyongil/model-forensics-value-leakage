from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from model_forensics import cli
from model_forensics.behavioral_phases import run_behavioral_generation_phase
from model_forensics.io import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from model_forensics.paid_response_store import PaidResponseStore
from model_forensics.prompts import build_prompt
from model_forensics.sampling import FakeBackend, SamplingParameters, build_requests

ROOT = Path(__file__).resolve().parents[1]


def _write_baseline_generation(directory: Path) -> None:
    requests = build_requests(
        task="giraffe",
        condition="baseline",
        count=1,
        threshold=None,
        master_seed=7,
        prompt_builder=build_prompt,
        parameters=SamplingParameters(max_new_tokens=32),
        randomize=False,
    )
    backend = FakeBackend()
    result = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase="baseline",
        checkpoint_dir=directory,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )
    assert result.complete is True


def _write_treatment_generation(directory: Path) -> None:
    requests = build_requests(
        task="giraffe",
        condition="above_good",
        count=1,
        threshold=41_000_000.0,
        master_seed=8,
        prompt_builder=build_prompt,
        parameters=SamplingParameters(max_new_tokens=32),
        randomize=False,
    )
    backend = FakeBackend()
    result = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase="treatment",
        checkpoint_dir=directory,
        batch_size=1,
        expected_backend_provenance=backend.provenance,
    )
    assert result.complete is True


def _hashed_row(**values):  # type: ignore[no-untyped-def]
    row = dict(values)
    row["record_hash"] = stable_hash(row)
    return row


def _write_baseline_adjudication_checkpoint(directory: Path) -> dict:
    primary = _hashed_row(run_id="baseline-row", instrument="primary")
    independent = _hashed_row(unit_id="baseline-row", instrument="independent")
    write_jsonl(directory / "primary_manifest.jsonl", [primary])
    write_jsonl(directory / "independent_final_manifest.jsonl", [independent])
    manifest = {
        "schema_version": 1,
        "protocol_version": "behavioral-cpu-adjudication-v1",
        "phase": "baseline",
        "complete": True,
        "execution_id": "fixture-execution",
        "consensus_summary": {"gate_passed": True},
        "quality_gate": {"gate_passed": True},
        "thresholds": {"giraffe": 41_000_000.0},
        "artifacts": {},
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "adjudication_manifest.json", manifest)
    return manifest


def _gate() -> SimpleNamespace:
    routes = (
        SimpleNamespace(
            role="primary_final_and_trajectory",
            provider="openrouter",
            model="anthropic/claude-opus-5",
            input_usd_per_million_tokens=5.0,
            output_usd_per_million_tokens=25.0,
        ),
        SimpleNamespace(
            role="independent_final",
            provider="openrouter",
            model="google/gemini-3.1-pro-preview",
            input_usd_per_million_tokens=2.0,
            output_usd_per_million_tokens=12.0,
        ),
    )
    return SimpleNamespace(
        bindings=SimpleNamespace(
            routes=routes,
            caps_usd=SimpleNamespace(gpu=220.0, api=100.0, total=325.0),
            config_hash="sha256:" + "a1" * 32,
            preregistration_hash="sha256:" + "b2" * 32,
        )
    )


def test_behavior_adjudicate_parser_has_no_model_price_or_ledger_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "behavior-adjudicate",
            "--config",
            str(ROOT / "config/run_122b.yaml"),
            "--phase",
            "baseline",
        ]
    )

    assert args.handler is cli._command_behavior_adjudicate
    assert args.phase == "baseline"
    for forbidden in (
        "judge_model",
        "judge_input_price",
        "judge_output_price",
        "independent_final_model",
        "independent_final_input_price",
        "independent_final_output_price",
        "cost_ledger",
    ):
        assert not hasattr(args, forbidden)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "behavior-adjudicate",
                "--config",
                str(ROOT / "config/run_122b.yaml"),
                "--phase",
                "baseline",
                "--judge-model",
                "attacker/override",
            ]
        )


def test_baseline_approval_and_receipt_precede_exact_bound_client_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_dir = tmp_path / "baseline-generation"
    checkpoint_dir = tmp_path / "baseline-adjudication"
    _write_baseline_generation(generation_dir)
    events: list[str] = []
    caller_kwargs: list[dict] = []

    def approve(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("approval")
        return _gate()

    def authorize(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("receipt")
        assert kwargs["command_phase"] == "behavior_baseline_api"
        assert kwargs["plan_hash"].startswith("sha256:")
        return {"receipt_hash": "sha256:" + "1" * 64}

    class FakeCaller:
        not_for_primary_inference = False

        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            caller_kwargs.append(dict(kwargs))
            events.append(f"client:{kwargs['model_id']}")

    frozen_result = SimpleNamespace(
        thresholds={"giraffe": 41_000_000.0},
        threshold_manifests={"giraffe": {"manifest_hash": "sha256:" + "2" * 64}},
        manifest={"manifest_hash": "sha256:" + "3" * 64},
        quality_gate={"manifest_hash": "sha256:" + "4" * 64, "gate_passed": True},
        consensus_summary={"manifest_hash": "sha256:" + "5" * 64, "gate_passed": True},
        phase_rows=(),
    )

    def run_phase(**kwargs):  # type: ignore[no-untyped-def]
        events.append("run")
        assert kwargs["primary_caller"] is not kwargs["independent_final_caller"]
        return frozen_result

    monkeypatch.setattr(cli, "_project_root", lambda _config: tmp_path)
    monkeypatch.setattr(cli, "_validate_paid_phase", approve)
    monkeypatch.setattr(cli, "_authorize_paid_plan", authorize)
    monkeypatch.setattr(cli, "OpenRouterAdjudicationCaller", FakeCaller)
    monkeypatch.setattr(cli, "run_baseline_behavioral_adjudication_phase", run_phase)
    args = cli.build_parser().parse_args(
        [
            "behavior-adjudicate",
            "--config",
            str(ROOT / "config/run_122b.yaml"),
            "--phase",
            "baseline",
            "--generation-checkpoint-dir",
            str(generation_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
    )

    result = cli._command_behavior_adjudicate(args)

    assert events == [
        "approval",
        "receipt",
        "client:anthropic/claude-opus-5",
        "client:google/gemini-3.1-pro-preview",
        "run",
    ]
    assert [item["price"].input_per_million for item in caller_kwargs] == [5.0, 2.0]
    assert [item["price"].output_per_million for item in caller_kwargs] == [25.0, 12.0]
    assert caller_kwargs[0]["ledger"] is caller_kwargs[1]["ledger"]
    assert caller_kwargs[0]["ledger"].path == tmp_path / "data/manifests/cost_ledger.yaml"
    assert all(isinstance(item["paid_response_store"], PaidResponseStore) for item in caller_kwargs)
    assert caller_kwargs[0]["paid_response_store"].directory != caller_kwargs[1][
        "paid_response_store"
    ].directory
    assert result["status"] == "complete"
    assert (tmp_path / "data/manifests/behavioral_thresholds.json").is_file()


def test_treatment_publishes_complete_frozen_behavioral_release_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_dir = tmp_path / "treatment-generation"
    baseline_dir = tmp_path / "baseline-adjudication"
    treatment_dir = tmp_path / "treatment-adjudication"
    _write_treatment_generation(generation_dir)
    baseline_manifest = _write_baseline_adjudication_checkpoint(baseline_dir)
    threshold_manifest = {
        "schema_version": 1,
        "protocol_version": "behavioral-thresholds-v1",
        "thresholds": {"giraffe": 41_000_000.0},
        "source_adjudication_manifest_hash": baseline_manifest["manifest_hash"],
    }
    threshold_manifest["manifest_hash"] = stable_hash(threshold_manifest)
    write_json(tmp_path / "data/manifests/behavioral_thresholds.json", threshold_manifest)

    baseline_row = _hashed_row(
        run_id="baseline-row",
        task="giraffe",
        condition="baseline",
        final_estimate=41_000_000,
    )
    treatment_row = _hashed_row(
        run_id="treatment-row",
        task="giraffe",
        condition="above_good",
        final_estimate=42_000_000,
    )
    treatment_primary = _hashed_row(run_id="treatment-row", instrument="primary")
    treatment_independent = _hashed_row(
        unit_id="treatment-row", instrument="independent"
    )
    consensus_rows = (
        _hashed_row(run_id="baseline-row", exact_status_value_agreement=True),
        _hashed_row(run_id="treatment-row", exact_status_value_agreement=True),
    )
    consensus_summary = {
        "gate_passed": True,
        "manifest_hash": "sha256:" + "c3" * 32,
    }
    quality_gate = {
        "gate_passed": True,
        "manifest_hash": "sha256:" + "d4" * 32,
    }
    treatment_manifest = {
        "execution_id": "fixture-execution",
        "baseline_adjudication_manifest_hash": baseline_manifest["manifest_hash"],
        "manifest_hash": "sha256:" + "e5" * 32,
    }
    frozen_result = SimpleNamespace(
        complete=True,
        gate_passed=True,
        phase_rows=(treatment_row,),
        all_rows=(baseline_row, treatment_row),
        primary_manifest_rows=(treatment_primary,),
        independent_final_records=(),
        consensus_audit_rows=consensus_rows,
        consensus_summary=consensus_summary,
        quality_gate=quality_gate,
        thresholds={"giraffe": 41_000_000.0},
        threshold_manifests={
            "giraffe": {
                "threshold": 41_000_000.0,
                "manifest_hash": "sha256:" + "f6" * 32,
            }
        },
        manifest=treatment_manifest,
    )

    class FakeCaller:
        not_for_primary_inference = False

        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    def run_phase(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["baseline_adjudication_checkpoint_dir"] == baseline_dir
        write_jsonl(treatment_dir / "primary_manifest.jsonl", [treatment_primary])
        write_jsonl(
            treatment_dir / "independent_final_manifest.jsonl",
            [treatment_independent],
        )
        return frozen_result

    monkeypatch.setattr(cli, "_project_root", lambda _config: tmp_path)
    monkeypatch.setattr(cli, "_validate_paid_phase", lambda *args, **kwargs: _gate())
    monkeypatch.setattr(
        cli,
        "_authorize_paid_plan",
        lambda *args, **kwargs: {"receipt_hash": "sha256:" + "ab" * 32},
    )
    monkeypatch.setattr(cli, "OpenRouterAdjudicationCaller", FakeCaller)
    monkeypatch.setattr(cli, "run_treatment_behavioral_adjudication_phase", run_phase)
    args = cli.build_parser().parse_args(
        [
            "behavior-adjudicate",
            "--config",
            str(ROOT / "config/run_122b.yaml"),
            "--phase",
            "treatment",
            "--generation-checkpoint-dir",
            str(generation_dir),
            "--baseline-adjudication-checkpoint-dir",
            str(baseline_dir),
            "--checkpoint-dir",
            str(treatment_dir),
        ]
    )

    result = cli._command_behavior_adjudicate(args)

    assert result["status"] == "complete"
    assert len(read_jsonl(tmp_path / "data/raw/qwen35_122b/rollouts.jsonl")) == 2
    assert len(read_jsonl(tmp_path / "data/manifests/adjudication_manifest.jsonl")) == 2
    assert len(read_jsonl(tmp_path / "data/manifests/behavioral_final_consensus.jsonl")) == 2
    release = read_json(tmp_path / "data/manifests/behavioral_adjudication_manifest.json")
    assert release["routes"]["primary_final_and_trajectory"]["model"] == (
        "anthropic/claude-opus-5"
    )
    sampling = read_json(tmp_path / "data/manifests/sampling_manifest.json")
    assert sampling["adjudication"]["manifest_hash"] == release["manifest_hash"]

    altered = [_hashed_row(run_id="tampered", task="giraffe", final_estimate=1)]
    write_jsonl(tmp_path / "data/raw/qwen35_122b/rollouts.jsonl", altered)
    with pytest.raises(cli.CLIError, match="existing behavioral rollouts differs"):
        cli._command_behavior_adjudicate(args)
    assert read_jsonl(tmp_path / "data/raw/qwen35_122b/rollouts.jsonl") == altered
