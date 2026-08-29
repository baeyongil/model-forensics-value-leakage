from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from model_forensics import cli
from model_forensics.anchors import AnchorCandidate, select_frozen_anchors
from model_forensics.io import read_json, sha256_file, stable_hash, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _write_production_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load((ROOT / "config/run_122b.yaml").read_text(encoding="utf-8"))
    payload["preregistration"] = str(ROOT / "config/preregistration.yaml")
    payload["paths"] = {
        "raw_dir": "data/raw/primary",
        "interim_dir": "data/interim/primary",
        "manifest_dir": "data/manifests",
        "figure_dir": "reports/figures",
        "report_dir": "reports/staging",
    }
    path = tmp_path / "config/run.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _synthetic_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sentence_class in (
        "accuracy_commitment",
        "value_threshold_planning",
        "epistemic_control",
    ):
        for direction in ("above_good", "below_good"):
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
    return rows


def _fake_gate() -> SimpleNamespace:
    routes = (
        SimpleNamespace(
            role="primary_final_and_trajectory",
            provider="openrouter",
            model="anthropic/approved-primary",
            input_usd_per_million_tokens=11.0,
            output_usd_per_million_tokens=22.0,
        ),
        SimpleNamespace(
            role="classifier_anthropic",
            provider="openrouter",
            model="anthropic/approved-classifier",
            input_usd_per_million_tokens=3.0,
            output_usd_per_million_tokens=4.0,
        ),
        SimpleNamespace(
            role="classifier_google",
            provider="openrouter",
            model="google/approved-classifier",
            input_usd_per_million_tokens=5.0,
            output_usd_per_million_tokens=6.0,
        ),
    )
    return SimpleNamespace(
        bindings=SimpleNamespace(
            config_hash="sha256:" + "1" * 64,
            preregistration_hash="sha256:" + "2" * 64,
            caps_usd=SimpleNamespace(gpu=220.0, api=100.0, total=325.0),
            routes=routes,
        ),
        approval_content_hash="sha256:" + "3" * 64,
        approval_id_hash="sha256:" + "4" * 64,
        bindings_hash="sha256:" + "5" * 64,
    )


def test_anchor_candidates_freeze_locally_without_paid_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_production_config(tmp_path)
    candidates = tmp_path / "data/interim/primary/anchor_candidates.jsonl"
    write_jsonl(candidates, _synthetic_candidates())
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )

    assert cli.main(["anchors", "--config", str(config)]) == 0
    output = tmp_path / "data/manifests/anchor_manifest.json"
    payload = read_json(output)
    assert payload["manifest_hash"] == stable_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_completed_anchor_manifest_is_validation_only_without_candidates_or_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_production_config(tmp_path)
    candidates = tmp_path / "data/interim/primary/anchor_candidates.jsonl"
    write_jsonl(candidates, _synthetic_candidates())
    assert cli.main(["anchors", "--config", str(config)]) == 0
    candidates.unlink()
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )

    args = cli.build_parser().parse_args(["anchors", "--config", str(config)])
    result = cli._command_anchors(args)
    assert result["validation_only"] is True
    assert result["paid_calls_performed"] == 0
    assert result["anchors"] == 24


def test_corrupt_completed_anchor_manifest_fails_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_production_config(tmp_path)
    candidates = tmp_path / "data/interim/primary/anchor_candidates.jsonl"
    write_jsonl(candidates, _synthetic_candidates())
    assert cli.main(["anchors", "--config", str(config)]) == 0
    output = tmp_path / "data/manifests/anchor_manifest.json"
    payload = read_json(output)
    payload["candidate_count"] = 23
    write_json(output, payload)
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )

    args = cli.build_parser().parse_args(["anchors", "--config", str(config)])
    with pytest.raises(cli.CLIError, match="manifest_hash mismatch"):
        cli._command_anchors(args)


def test_anchor_paid_path_authenticates_then_receipts_before_tokenizer_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_production_config(tmp_path)
    rollout_path = tmp_path / "data/raw/primary/rollouts.jsonl"
    sampling_path = tmp_path / "data/manifests/sampling_manifest.json"
    rollout_path.parent.mkdir(parents=True)
    sampling_path.parent.mkdir(parents=True)
    rollout_path.write_text("{}\n", encoding="utf-8")
    sampling_path.write_text("{}\n", encoding="utf-8")
    events: list[str] = []

    def authenticated(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        events.append("inputs")
        return [], {"manifest_hash": "sha256:" + "6" * 64}

    def approved(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("approval")
        return _fake_gate()

    def receipted(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("receipt")
        return {"receipt_hash": "sha256:" + "7" * 64}

    class StopAtTokenizer(RuntimeError):
        pass

    def tokenizer(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("tokenizer")
        raise StopAtTokenizer

    monkeypatch.setattr(cli, "_load_authenticated_behavioral_rollouts", authenticated)
    monkeypatch.setattr(cli, "_validate_paid_phase", approved)
    monkeypatch.setattr(cli, "_authorize_paid_plan", receipted)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer)
    monkeypatch.setattr(
        cli,
        "OpenRouterClassificationCaller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("client too early")),
    )
    args = cli.build_parser().parse_args(["anchors", "--config", str(config_path)])
    with pytest.raises(StopAtTokenizer):
        cli._command_anchors(args)
    assert events == ["inputs", "approval", "receipt", "tokenizer"]
    plan = read_json(tmp_path / "data/interim/primary/checkpoints/anchors/paid_plan.json")
    assert [route["model"] for route in plan["classification"]["routes"]] == [
        "anthropic/approved-classifier",
        "google/approved-classifier",
    ]
    assert plan["cost_ledger"]["path"] == "data/manifests/cost_ledger.yaml"


def _write_completed_position_bundle(tmp_path: Path, config_path: Path) -> None:
    config = cli.load_run_config(config_path)
    rollout_rows: list[dict[str, object]] = []
    candidates: list[AnchorCandidate] = []
    for sentence_class in (
        "accuracy_commitment",
        "value_threshold_planning",
        "epistemic_control",
    ):
        for direction in ("above_good", "below_good"):
            for ordinal, (side, flip) in enumerate(
                (("good", False), ("good", True), ("bad", False), ("bad", True))
            ):
                trace_id = f"{sentence_class}:{direction}:{ordinal}"
                rollout: dict[str, object] = {
                    "run_id": trace_id,
                    "task": "giraffe",
                    "condition": direction,
                    "backend": {
                        "model_id": config.model.id,
                        "model_revision": config.model.revision,
                    },
                }
                rollout["record_hash"] = stable_hash(rollout)
                rollout_rows.append(rollout)
                sentence = f"Sentence {sentence_class} {direction} {ordinal}."
                candidates.append(
                    AnchorCandidate(
                        trace_id=trace_id,
                        sentence_class=sentence_class,
                        direction=direction,
                        sentence_index=ordinal,
                        sentence_text=sentence,
                        char_start=10,
                        char_end=10 + len(sentence),
                        initial_side=side,
                        final_flip=flip,
                        provenance={"source_rollout_hash": rollout["record_hash"]},
                    )
                )
    rollout_path = tmp_path / "data/raw/primary/rollouts.jsonl"
    write_jsonl(rollout_path, rollout_rows)
    frozen = select_frozen_anchors(candidates)
    anchor_payload = frozen.as_dict()
    anchor_payload["manifest_hash"] = stable_hash(anchor_payload)
    anchor_path = tmp_path / "data/manifests/anchor_manifest.json"
    write_json(anchor_path, anchor_payload)
    position_rows = []
    for anchor in frozen.anchors:
        row: dict[str, object] = {
            "trace_id": anchor.trace_id,
            "anchor_id": anchor.anchor_id,
            "anchor_manifest_hash": anchor_payload["manifest_hash"],
            "position_order": list(cli.POSITION_ORDER),
            "position_indices": {name: index for index, name in enumerate(cli.POSITION_ORDER)},
            "causal_claim": False,
        }
        row["record_hash"] = stable_hash(row)
        position_rows.append(row)
    output = tmp_path / "data/manifests/lens_positions.jsonl"
    write_jsonl(output, position_rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "failures": [],
        "positions": "data/manifests/lens_positions.jsonl",
        "positions_sha256": sha256_file(output),
        "position_count": 24,
        "rollouts_sha256": sha256_file(rollout_path),
        "anchor_manifest_hash": anchor_payload["manifest_hash"],
    }
    summary["manifest_hash"] = stable_hash(summary)
    write_json(tmp_path / "data/manifests/lens_position_manifest.json", summary)


def test_completed_positions_are_validation_only_without_approval_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config)
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )
    monkeypatch.setattr(
        cli,
        "OpenRouterAdjudicationCaller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("client is forbidden")),
    )

    args = cli.build_parser().parse_args(["positions", "--config", str(config)])
    result = cli._command_positions(args)
    assert result["validation_only"] is True
    assert result["paid_calls_performed"] == 0
    assert result["positions"] == 24


def test_corrupt_completed_positions_fail_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config)
    output = tmp_path / "data/manifests/lens_positions.jsonl"
    rows = cli.read_jsonl(output)
    rows[0]["causal_claim"] = True
    write_jsonl(output, rows)
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )

    args = cli.build_parser().parse_args(["positions", "--config", str(config)])
    with pytest.raises(cli.CLIError, match="source/output hashes disagree"):
        cli._command_positions(args)


def test_positions_authenticate_then_receipt_before_tokenizer_or_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_production_config(tmp_path)
    rollout_path = tmp_path / "data/raw/primary/rollouts.jsonl"
    anchor_path = tmp_path / "data/manifests/anchor_manifest.json"
    rollout_path.parent.mkdir(parents=True)
    anchor_path.parent.mkdir(parents=True)
    rollout_path.write_text("{}\n", encoding="utf-8")
    anchor_path.write_text("{}\n", encoding="utf-8")
    frozen = SimpleNamespace(
        anchors=(SimpleNamespace(trace_id="trace-1", anchor_id="anchor-1"),),
        selection_hash="selection-hash",
    )
    events: list[str] = []

    def authenticated(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        events.append("inputs")
        return [], {"manifest_hash": "sha256:" + "6" * 64}, frozen

    def approved(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("approval")
        return _fake_gate()

    def receipted(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("receipt")
        return {"receipt_hash": "sha256:" + "7" * 64}

    class StopAtTokenizer(RuntimeError):
        pass

    def tokenizer(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        events.append("tokenizer")
        raise StopAtTokenizer

    monkeypatch.setattr(cli, "_load_authenticated_position_inputs", authenticated)
    monkeypatch.setattr(cli, "_validate_paid_phase", approved)
    monkeypatch.setattr(cli, "_authorize_paid_plan", receipted)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer)
    monkeypatch.setattr(
        cli,
        "OpenRouterAdjudicationCaller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("client too early")),
    )
    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(StopAtTokenizer):
        cli._command_positions(args)
    assert events == ["inputs", "approval", "receipt", "tokenizer"]
    plan = read_json(tmp_path / "data/interim/primary/checkpoints/positions/paid_plan.json")
    assert plan["route"]["model"] == "anthropic/approved-primary"
    assert plan["cost_ledger"]["path"] == "data/manifests/cost_ledger.yaml"


@pytest.mark.parametrize(
    ("command", "legacy_argument"),
    (
        ("anchors", "--classifier-a-model"),
        ("anchors", "--classifier-a-revision"),
        ("anchors", "--classifier-a-input-price"),
        ("anchors", "--classifier-a-output-price"),
        ("anchors", "--classifier-b-model"),
        ("anchors", "--classifier-b-revision"),
        ("anchors", "--classifier-b-input-price"),
        ("anchors", "--classifier-b-output-price"),
        ("anchors", "--cost-ledger"),
        ("positions", "--judge-model"),
        ("positions", "--judge-model-revision"),
        ("positions", "--judge-input-price"),
        ("positions", "--judge-output-price"),
        ("positions", "--cost-ledger"),
    ),
)
def test_anchor_and_position_parsers_reject_legacy_route_and_cost_overrides(
    command: str,
    legacy_argument: str,
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [command, "--config", "config/run.yaml", legacy_argument, "override"]
        )
