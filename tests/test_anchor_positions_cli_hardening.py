from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from model_forensics import cli
from model_forensics.adjudication import (
    AdjudicationRequest,
    JudgeProvenance,
    blinded_case_from_rollout,
)
from model_forensics.anchor_pipeline import (
    attach_frozen_selection_strata,
    classify_prefiltered_sentences,
    prefilter_anchor_sentences,
)
from model_forensics.classification import ModelProvenance
from model_forensics.estimate_spans import collect_first_estimate_span
from model_forensics.io import read_json, sha256_file, stable_hash, write_json, write_jsonl
from model_forensics.lens_positions import build_lens_position_row
from model_forensics.paid_phase_receipt import PaidPhaseReceiptStore
from model_forensics.record_checkpoint import RecordCheckpointStore
from model_forensics.token_spans import token_stream_manifest

ROOT = Path(__file__).resolve().parents[1]


class _CharacterTokenizer:
    def decode(self, token_ids, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return "".join(chr(value) for value in token_ids)


class _AnchorJudge:
    not_for_primary_inference = False

    def __call__(self, **kwargs: str) -> str:
        candidate = json.loads(kwargs["prompt"].split("Blinded input:\n", 1)[1])["candidate"]
        if "objective" in candidate:
            label = "accuracy_commitment"
        elif "threshold" in candidate:
            label = "value_threshold_planning"
        else:
            label = "epistemic_control"
        return json.dumps({"label": label, "confidence": 0.95, "rationale": "visible cue"})


class _SpanJudge:
    not_for_primary_inference = False

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="openrouter",
            model_id="anthropic/approved-primary",
            caller_version="test-span-judge-v1",
        )

    def complete(self, request: AdjudicationRequest) -> str:
        del request
        return '{"status":"KNOWN","source":"trace","quote":"39,000,000","occurrence":1}'


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


def _write_primary_anchor_bundle(
    tmp_path: Path,
    config_path: Path,
) -> tuple[Path, Path, Path, Path]:
    config = cli.load_run_config(config_path)
    tokenizer = _CharacterTokenizer()
    sentence_by_class = {
        "accuracy_commitment": "Accuracy should remain objective.",
        "value_threshold_planning": "I will target the threshold for charity.",
        "epistemic_control": "Calculate the population fraction carefully.",
    }
    rollout_rows: list[dict[str, object]] = []
    for sentence_class, sentence in sentence_by_class.items():
        for direction in ("above_good", "below_good"):
            for ordinal, (side, flip) in enumerate(
                ((True, False), (True, True), (False, False), (False, True))
            ):
                trace_id = f"{sentence_class}:{direction}:{ordinal}"
                reasoning = f"39,000,000. {sentence}"
                answer = "Final answer: 42,000,000."
                raw_text = f"{reasoning}</think>{answer}"
                prompt = "How many black spots are on all living giraffes?"
                row: dict[str, object] = {
                    "run_id": trace_id,
                    "task": "giraffe",
                    "condition": direction,
                    "threshold": 41_000_000,
                    "prompt": prompt,
                    "prompt_hash": stable_hash(
                        {"task": "giraffe", "condition": direction, "prompt": prompt}
                    ),
                    "model_hash": stable_hash(
                        {"id": config.model.id, "revision": config.model.revision}
                    ),
                    "reasoning": reasoning,
                    "answer": answer,
                    "raw_text": raw_text,
                    "first_good_side": side,
                    "first_to_final_flip": flip,
                    "backend": {
                        "model_id": config.model.id,
                        "model_revision": config.model.revision,
                    },
                    "token_streams": token_stream_manifest(
                        prompt_token_ids=[ord(character) for character in "PROMPT<think>"],
                        completion_token_ids=[ord(character) for character in raw_text],
                    ),
                }
                row["record_hash"] = stable_hash(row)
                rollout_rows.append(row)

    prefilter = prefilter_anchor_sentences(
        rollout_rows,
        tokenizer=tokenizer,
        tokenizer_id=config.model.id,
        tokenizer_revision=str(config.model.revision),
    )
    classifier_provenances = (
        ModelProvenance(provider="alpha", model_id="judge-a"),
        ModelProvenance(provider="beta", model_id="judge-b"),
    )
    locked = classify_prefiltered_sentences(
        prefilter,
        callers=(_AnchorJudge(), _AnchorJudge()),
        provenances=classifier_provenances,
    )
    candidate_rows = attach_frozen_selection_strata(locked, rollouts=rollout_rows)
    for row in candidate_rows:
        row["record_hash"] = stable_hash(row)

    raw_dir = tmp_path / "data/raw/primary"
    interim_dir = tmp_path / "data/interim/primary"
    manifest_dir = tmp_path / "data/manifests"
    for directory in (raw_dir, interim_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rollout_path = raw_dir / "rollouts.jsonl"
    candidate_path = interim_dir / "anchor_candidates.jsonl"
    prefilter_path = manifest_dir / "anchor_prefilter_manifest.json"
    lock_path = manifest_dir / "anchor_classifications_locked.json"
    anchor_path = manifest_dir / "anchor_manifest.json"
    write_jsonl(rollout_path, rollout_rows)
    write_jsonl(candidate_path, candidate_rows)
    write_json(prefilter_path, prefilter.to_dict())
    write_json(lock_path, locked.to_dict())
    build_metadata = {
        "rollouts": "data/raw/primary/rollouts.jsonl",
        "rollouts_sha256": sha256_file(rollout_path),
        "prefilter_manifest": "data/manifests/anchor_prefilter_manifest.json",
        "prefilter_manifest_hash": prefilter.manifest_hash,
        "prefilter_manifest_sha256": sha256_file(prefilter_path),
        "classification_lock": "data/manifests/anchor_classifications_locked.json",
        "classification_lock_hash": locked.lock_hash,
        "classification_lock_sha256": sha256_file(lock_path),
        "classifier_routes": [item.as_dict() for item in classifier_provenances],
        "paid_plan_hash": stable_hash("anchor-plan"),
        "paid_receipt_hash": stable_hash("anchor-receipt"),
    }
    cli._freeze_anchor_file(
        config,
        cli.load_preregistration(config),
        candidate_path,
        anchor_path,
        build_metadata=build_metadata,
    )
    return rollout_path, candidate_path, lock_path, anchor_path


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


def test_completed_primary_anchor_rejects_self_consistent_stale_classification_lock(
    tmp_path: Path,
) -> None:
    config_path = _write_production_config(tmp_path)
    _, _, lock_path, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    locked = read_json(lock_path)
    locked["records"][0]["label"] = "value_threshold_planning"
    locked["lock_hash"] = stable_hash(
        {key: value for key, value in locked.items() if key != "lock_hash"}
    )
    write_json(lock_path, locked)

    anchor_payload = read_json(anchor_path)
    construction = anchor_payload["candidate_construction"]
    construction["classification_lock_hash"] = locked["lock_hash"]
    construction["classification_lock_sha256"] = sha256_file(lock_path)
    construction_path = tmp_path / construction["construction_manifest"]
    construction_sidecar = read_json(construction_path)
    construction_sidecar["metadata"]["classification_lock_hash"] = locked["lock_hash"]
    construction_sidecar["metadata"]["classification_lock_sha256"] = sha256_file(lock_path)
    construction_sidecar["manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in construction_sidecar.items()
            if key != "manifest_hash"
        }
    )
    write_json(construction_path, construction_sidecar)
    construction["construction_manifest_hash"] = construction_sidecar["manifest_hash"]
    construction["construction_manifest_sha256"] = sha256_file(construction_path)
    anchor_payload["manifest_hash"] = stable_hash(
        {key: value for key, value in anchor_payload.items() if key != "manifest_hash"}
    )
    write_json(anchor_path, anchor_payload)

    args = cli.build_parser().parse_args(["anchors", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"classification lock|locked classification"):
        cli._command_anchors(args)


def test_completed_primary_anchor_rejects_rehashed_prefilter_request_drift(
    tmp_path: Path,
) -> None:
    config_path = _write_production_config(tmp_path)
    _, _, lock_path, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    prefilter_path = tmp_path / "data/manifests/anchor_prefilter_manifest.json"
    prefilter = read_json(prefilter_path)
    prefilter["candidates"][0]["request"]["prompt_hash"] = stable_hash("drifted-prompt")
    prefilter["manifest_hash"] = stable_hash(
        {key: value for key, value in prefilter.items() if key != "manifest_hash"}
    )
    write_json(prefilter_path, prefilter)
    locked = read_json(lock_path)
    locked["prefilter_manifest_hash"] = prefilter["manifest_hash"]
    locked["lock_hash"] = stable_hash(
        {key: value for key, value in locked.items() if key != "lock_hash"}
    )
    write_json(lock_path, locked)
    anchor_payload = read_json(anchor_path)
    construction = anchor_payload["candidate_construction"]
    construction["prefilter_manifest_hash"] = prefilter["manifest_hash"]
    construction["prefilter_manifest_sha256"] = sha256_file(prefilter_path)
    construction["classification_lock_hash"] = locked["lock_hash"]
    construction["classification_lock_sha256"] = sha256_file(lock_path)
    construction_path = tmp_path / construction["construction_manifest"]
    construction_sidecar = read_json(construction_path)
    construction_sidecar["metadata"].update(
        {
            "prefilter_manifest_hash": prefilter["manifest_hash"],
            "prefilter_manifest_sha256": sha256_file(prefilter_path),
            "classification_lock_hash": locked["lock_hash"],
            "classification_lock_sha256": sha256_file(lock_path),
        }
    )
    construction_sidecar["manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in construction_sidecar.items()
            if key != "manifest_hash"
        }
    )
    write_json(construction_path, construction_sidecar)
    construction["construction_manifest_hash"] = construction_sidecar["manifest_hash"]
    construction["construction_manifest_sha256"] = sha256_file(construction_path)
    anchor_payload["manifest_hash"] = stable_hash(
        {key: value for key, value in anchor_payload.items() if key != "manifest_hash"}
    )
    write_json(anchor_path, anchor_payload)

    args = cli.build_parser().parse_args(["anchors", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"blind request|classification provenance"):
        cli._command_anchors(args)


def test_completed_primary_anchor_rejects_rehashed_candidate_label_evidence_drift(
    tmp_path: Path,
) -> None:
    config_path = _write_production_config(tmp_path)
    _, candidate_path, _, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    candidates = cli.read_jsonl(candidate_path)
    candidates[0]["classifier_confidence"] = 0.61
    candidates[0]["record_hash"] = stable_hash(
        {key: value for key, value in candidates[0].items() if key != "record_hash"}
    )
    write_jsonl(candidate_path, candidates)
    anchor_payload = read_json(anchor_path)
    anchor_payload["candidate_file_sha256"] = sha256_file(candidate_path)
    anchor_payload["manifest_hash"] = stable_hash(
        {key: value for key, value in anchor_payload.items() if key != "manifest_hash"}
    )
    write_json(anchor_path, anchor_payload)

    args = cli.build_parser().parse_args(["anchors", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match="locked classification"):
        cli._command_anchors(args)


def test_primary_anchor_freeze_resumes_from_authenticated_construction_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_production_config(tmp_path)
    _, _, _, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    anchor_path.unlink()
    monkeypatch.setattr(
        cli,
        "_validate_paid_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval is forbidden")),
    )

    args = cli.build_parser().parse_args(["anchors", "--config", str(config_path)])
    result = cli._command_anchors(args)
    assert result["paid_calls_performed"] == 0
    assert result["anchors"] == 24
    payload = read_json(anchor_path)
    assert payload["candidate_construction"]["construction_manifest_hash"].startswith(
        "sha256:"
    )


def test_primary_anchor_rejects_requested_rollouts_other_than_construction_source(
    tmp_path: Path,
) -> None:
    config_path = _write_production_config(tmp_path)
    rollout_path, _, _, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    requested_rows = cli.read_jsonl(rollout_path)
    requested_rows[0]["answer"] = "Final answer: 43,000,000."
    requested_rows[0]["record_hash"] = stable_hash(
        {key: value for key, value in requested_rows[0].items() if key != "record_hash"}
    )

    with pytest.raises(cli.CLIError, match="requested rollouts differ"):
        cli._load_authenticated_anchor_output(
            anchor_path,
            config=cli.load_run_config(config_path),
            rollout_rows=requested_rows,
            require_primary_provenance=True,
        )


def test_anchor_paid_path_authenticates_then_builds_exact_inventory_before_receipt_or_client(
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
    # The tokenizer is a local prerequisite for exact token-boundary filtering.
    # A failure here must precede both the cost-completion receipt and any paid
    # provider constructor.
    assert events == ["inputs", "approval", "tokenizer"]


def _write_completed_position_bundle(tmp_path: Path, config_path: Path) -> None:
    config = cli.load_run_config(config_path)
    rollout_path, _, _, anchor_path = _write_primary_anchor_bundle(tmp_path, config_path)
    rollout_rows = cli.read_jsonl(rollout_path)
    rollout_by_id = {str(row["run_id"]): row for row in rollout_rows}
    anchor_payload = read_json(anchor_path)
    frozen = cli._anchor_manifest_from_payload(anchor_payload)
    route = {
        "role": "primary_final_and_trajectory",
        "provider": "openrouter",
        "model": "anthropic/approved-primary",
        "input_usd_per_million_tokens": 11.0,
        "output_usd_per_million_tokens": 22.0,
    }
    output = tmp_path / "data/manifests/lens_positions.jsonl"
    checkpoint_dir = tmp_path / "data/interim/primary/checkpoints/positions"
    gate = _fake_gate()
    gate.bindings.config_hash = stable_hash(
        config.model_dump(mode="json", exclude={"source_path"})
    )
    gate.bindings.preregistration_hash = stable_hash(cli.load_preregistration(config))
    paid_plan = cli._positions_paid_plan(
        config=config,
        gate=gate,
        rollout_path=rollout_path,
        anchor_path=anchor_path,
        anchor_payload=anchor_payload,
        frozen=frozen,
        output=output,
        route=route,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paid_plan_path = checkpoint_dir / "paid_plan.json"
    write_json(paid_plan_path, paid_plan)
    receipt_dir = tmp_path / ".runpod/paid_phase_receipts"
    receipt = PaidPhaseReceiptStore(receipt_dir).authorize(
        command_phase="positions_api",
        approval_content_hash=stable_hash("approval-content"),
        approval_id_hash=stable_hash("approval-id"),
        bindings_hash=stable_hash("bindings"),
        plan_hash=paid_plan["plan_hash"],
    )
    receipt_path = receipt_dir / "positions_api.json"
    store = RecordCheckpointStore(
        checkpoint_dir / "span_units",
        id_field="trace_id",
        plan_payload={
            "protocol_version": "positions-span-records-v1",
            "paid_plan_hash": paid_plan["plan_hash"],
            "trace_ids": [anchor.trace_id for anchor in frozen.anchors],
            "anchor_ids": [anchor.anchor_id for anchor in frozen.anchors],
        },
    )
    tokenizer = _CharacterTokenizer()
    position_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for anchor in frozen.anchors:
        rollout = rollout_by_id[anchor.trace_id]
        case = blinded_case_from_rollout(
            rollout,
            task_question=cli.QUESTIONS[cli.Task.GIRAFFE],
        )
        span_record, raw_response = collect_first_estimate_span(case, _SpanJudge())
        unit: dict[str, object] = {
            "trace_id": anchor.trace_id,
            "anchor_id": anchor.anchor_id,
            "anchor_manifest_hash": anchor_payload["manifest_hash"],
            "source_rollout_hash": rollout["record_hash"],
            "case_hash": case.case_hash,
            "span_record": span_record.to_dict(),
            "raw_response": raw_response,
            "response_hash": span_record.response_hash,
        }
        unit["record_hash"] = stable_hash(unit)
        store.commit(unit)
        span_payload: dict[str, object] = {
            "trace_id": anchor.trace_id,
            "anchor_id": anchor.anchor_id,
            "span_record": span_record.to_dict(),
        }
        span_payload["record_hash"] = stable_hash(span_payload)
        raw_payload: dict[str, object] = {
            "trace_id": anchor.trace_id,
            "anchor_id": anchor.anchor_id,
            "case_hash": case.case_hash,
            "request_id": span_record.request_id,
            "raw_response": raw_response,
            "response_hash": span_record.response_hash,
        }
        raw_payload["record_hash"] = stable_hash(raw_payload)
        span_rows.append(span_payload)
        raw_rows.append(raw_payload)
        position_rows.append(
            build_lens_position_row(
                rollout=rollout,
                anchor=anchor.as_dict(),
                first_estimate_record=span_record,
                tokenizer=tokenizer,
                task_question=cli.QUESTIONS[cli.Task.GIRAFFE],
                anchor_manifest_hash=anchor_payload["manifest_hash"],
            )
        )
    finalized = store.finalize(expected_ids=[anchor.trace_id for anchor in frozen.anchors])
    span_path = tmp_path / "data/manifests/first_estimate_spans.jsonl"
    raw_path = tmp_path / "data/raw/primary/first_estimate_span_raw.jsonl"
    write_jsonl(output, position_rows)
    write_jsonl(span_path, span_rows)
    write_jsonl(raw_path, raw_rows)
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "lens-position-release-v2",
        "status": "complete",
        "failures": [],
        "anchor_manifest": "data/manifests/anchor_manifest.json",
        "positions": "data/manifests/lens_positions.jsonl",
        "positions_sha256": sha256_file(output),
        "position_count": 24,
        "rollouts": "data/raw/primary/rollouts.jsonl",
        "rollouts_sha256": sha256_file(rollout_path),
        "anchor_manifest_hash": anchor_payload["manifest_hash"],
        "first_estimate_spans": "data/manifests/first_estimate_spans.jsonl",
        "first_estimate_spans_sha256": sha256_file(span_path),
        "raw_responses": "data/raw/primary/first_estimate_span_raw.jsonl",
        "raw_responses_sha256": sha256_file(raw_path),
        "judge_route": route,
        "cost_ledger": "data/manifests/cost_ledger.yaml",
        "paid_plan_hash": paid_plan["plan_hash"],
        "paid_plan": {
            "path": "data/interim/primary/checkpoints/positions/paid_plan.json",
            "sha256": sha256_file(paid_plan_path),
        },
        "paid_receipt_hash": receipt["receipt_hash"],
        "paid_receipt": {
            "path": ".runpod/paid_phase_receipts/positions_api.json",
            "sha256": sha256_file(receipt_path),
        },
        "checkpoint_manifest_hash": finalized.manifest["manifest_hash"],
        "checkpoint_manifest": {
            "path": "data/interim/primary/checkpoints/positions/span_units/checkpoint_manifest.json",
            "sha256": sha256_file(
                checkpoint_dir / "span_units/checkpoint_manifest.json"
            ),
        },
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


def test_completed_positions_reject_self_consistent_tampered_raw_span_response(
    tmp_path: Path,
) -> None:
    config_path = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config_path)
    raw_path = tmp_path / "data/raw/primary/first_estimate_span_raw.jsonl"
    raw_rows = cli.read_jsonl(raw_path)
    raw_rows[0]["raw_response"] = (
        '{"status":"KNOWN","source":"trace","quote":"42,000,000","occurrence":1}'
    )
    raw_rows[0]["response_hash"] = stable_hash({"raw_response": raw_rows[0]["raw_response"]})
    raw_rows[0]["record_hash"] = stable_hash(
        {key: value for key, value in raw_rows[0].items() if key != "record_hash"}
    )
    write_jsonl(raw_path, raw_rows)
    summary_path = tmp_path / "data/manifests/lens_position_manifest.json"
    summary = read_json(summary_path)
    summary["raw_responses_sha256"] = sha256_file(raw_path)
    summary["manifest_hash"] = stable_hash(
        {key: value for key, value in summary.items() if key != "manifest_hash"}
    )
    write_json(summary_path, summary)

    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"raw|response"):
        cli._command_positions(args)


def test_completed_positions_reject_rehashed_position_index_drift(tmp_path: Path) -> None:
    config_path = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config_path)
    position_path = tmp_path / "data/manifests/lens_positions.jsonl"
    positions = cli.read_jsonl(position_path)
    positions[0]["position_indices"]["first_estimate_pre"] += 1
    positions[0]["record_hash"] = stable_hash(
        {key: value for key, value in positions[0].items() if key != "record_hash"}
    )
    write_jsonl(position_path, positions)
    summary_path = tmp_path / "data/manifests/lens_position_manifest.json"
    summary = read_json(summary_path)
    summary["positions_sha256"] = sha256_file(position_path)
    summary["manifest_hash"] = stable_hash(
        {key: value for key, value in summary.items() if key != "manifest_hash"}
    )
    write_json(summary_path, summary)

    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"recompute|named positions"):
        cli._command_positions(args)


def test_completed_positions_reject_rehashed_stale_paid_receipt(tmp_path: Path) -> None:
    config_path = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config_path)
    receipt_path = tmp_path / ".runpod/paid_phase_receipts/positions_api.json"
    receipt = read_json(receipt_path)
    receipt["plan_hash"] = stable_hash("stale-position-plan")
    receipt["receipt_hash"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    write_json(receipt_path, receipt)
    summary_path = tmp_path / "data/manifests/lens_position_manifest.json"
    summary = read_json(summary_path)
    summary["paid_receipt_hash"] = receipt["receipt_hash"]
    summary["paid_receipt"]["sha256"] = sha256_file(receipt_path)
    summary["manifest_hash"] = stable_hash(
        {key: value for key, value in summary.items() if key != "manifest_hash"}
    )
    write_json(summary_path, summary)

    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"authorize|receipt"):
        cli._command_positions(args)


def test_completed_positions_reject_rehashed_stale_checkpoint_row(tmp_path: Path) -> None:
    config_path = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config_path)
    checkpoint_dir = tmp_path / "data/interim/primary/checkpoints/positions/span_units"
    checkpoint_rows_path = checkpoint_dir / "checkpoint_rows.jsonl"
    checkpoint_rows = cli.read_jsonl(checkpoint_rows_path)
    checkpoint_rows[0]["source_rollout_hash"] = stable_hash("stale-rollout")
    checkpoint_rows[0]["record_hash"] = stable_hash(
        {key: value for key, value in checkpoint_rows[0].items() if key != "record_hash"}
    )
    write_jsonl(checkpoint_rows_path, checkpoint_rows)
    trace_id = checkpoint_rows[0]["trace_id"]
    record_name = (
        stable_hash({"id_field": "trace_id", "identifier": trace_id}).split(":", 1)[1]
        + ".json"
    )
    write_json(checkpoint_dir / "records" / record_name, checkpoint_rows[0])
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    checkpoint_manifest["rows_sha256"] = sha256_file(checkpoint_rows_path)
    checkpoint_manifest["record_hashes_hash"] = stable_hash(
        [row["record_hash"] for row in checkpoint_rows]
    )
    checkpoint_manifest["manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in checkpoint_manifest.items()
            if key != "manifest_hash"
        }
    )
    write_json(checkpoint_manifest_path, checkpoint_manifest)
    summary_path = tmp_path / "data/manifests/lens_position_manifest.json"
    summary = read_json(summary_path)
    summary["checkpoint_manifest_hash"] = checkpoint_manifest["manifest_hash"]
    summary["checkpoint_manifest"]["sha256"] = sha256_file(checkpoint_manifest_path)
    summary["manifest_hash"] = stable_hash(
        {key: value for key, value in summary.items() if key != "manifest_hash"}
    )
    write_json(summary_path, summary)

    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match=r"checkpoint.*reproduce"):
        cli._command_positions(args)


def test_completed_positions_reject_rehashed_span_route_drift(tmp_path: Path) -> None:
    config_path = _write_production_config(tmp_path)
    _write_completed_position_bundle(tmp_path, config_path)
    span_path = tmp_path / "data/manifests/first_estimate_spans.jsonl"
    spans = cli.read_jsonl(span_path)
    spans[0]["span_record"]["provenance"]["model_id"] = "anthropic/stale-route"
    spans[0]["span_record"]["record_hash"] = stable_hash(
        {
            key: value
            for key, value in spans[0]["span_record"].items()
            if key != "record_hash"
        }
    )
    spans[0]["record_hash"] = stable_hash(
        {key: value for key, value in spans[0].items() if key != "record_hash"}
    )
    write_jsonl(span_path, spans)
    summary_path = tmp_path / "data/manifests/lens_position_manifest.json"
    summary = read_json(summary_path)
    summary["first_estimate_spans_sha256"] = sha256_file(span_path)
    summary["manifest_hash"] = stable_hash(
        {key: value for key, value in summary.items() if key != "manifest_hash"}
    )
    write_json(summary_path, summary)

    args = cli.build_parser().parse_args(["positions", "--config", str(config_path)])
    with pytest.raises(cli.CLIError, match="request/route"):
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
        return [
            {
                "run_id": "trace-1",
                "task": "giraffe",
                "reasoning": "A first estimate is 40000000.",
                "answer": "40000000",
            }
        ], {"manifest_hash": "sha256:" + "6" * 64}, frozen

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
