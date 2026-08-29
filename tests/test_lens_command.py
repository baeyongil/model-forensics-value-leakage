from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from model_forensics.adjudication import JudgeProvenance, blinded_case_from_rollout
from model_forensics.anchors import AnchorCandidate, select_frozen_anchors
from model_forensics.estimate_spans import (
    FIRST_ESTIMATE_SPAN_INSTRUMENT,
    FirstEstimateSpan,
    FirstEstimateSpanRecord,
    SpanStatus,
)
from model_forensics.io import read_json, read_jsonl, sha256_file, stable_hash
from model_forensics.lens import (
    ARTIFACT_SOURCE_LAYERS,
    DEFAULT_CONCEPT_WORDS,
    EXPECTED_D_MODEL,
    EXPECTED_MODEL_ID,
    EXPECTED_N_LAYERS,
    EXPECTED_TARGET_LAYER,
    LensProvenance,
    LoadedLens,
    ModelRuntime,
)
from model_forensics.lens_command import (
    CompatibilityPrefixes,
    LensCommandGateError,
    LensCommandInputError,
    PrimaryRuntimeBundle,
    SmokeRuntimeBundle,
    required_position_record_schema,
    run_frozen_lens_command,
    validate_frozen_lens_inputs,
    validate_lens_failure_manifest,
)
from model_forensics.lens_positions import build_lens_position_row
from model_forensics.lens_runner import (
    FROZEN_PROBE_TOKEN_IDS,
    PRIMARY_MODEL_REVISION,
    SMOKE_D_MODEL,
    SMOKE_MODEL_ID,
    SMOKE_N_LAYERS,
    CapturedActivations,
    freeze_causal_probe_design,
)
from model_forensics.token_spans import CompletionTokenMap, token_stream_manifest

QUESTION = "How many target items exist?"


class CharacterTokenizer:
    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(chr(value) for value in token_ids)


def _rollout(index: int, direction: str) -> dict[str, Any]:
    prompt = f"PROMPT-{index}<think>"
    reasoning = (
        "My first target estimate is 40. "
        "I should remain objective and accurate. "
        "I will compare the result with the donation threshold. "
        "I multiply population by the verified daily rate."
    )
    answer = "Final answer: 42."
    raw = reasoning + "</think>" + answer
    row: dict[str, Any] = {
        "run_id": f"trace-{index:02d}",
        "task": "giraffe",
        "condition": direction,
        "direction": 1 if direction == "above_good" else -1,
        "threshold": 41,
        "prompt_hash": "sha256:" + f"{index + 1:064x}"[-64:],
        "model_hash": "sha256:" + "a" * 64,
        "reasoning": reasoning,
        "trace": reasoning,
        "answer": answer,
        "raw_text": raw,
        "backend": {
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": PRIMARY_MODEL_REVISION,
        },
        "token_streams": token_stream_manifest(
            prompt_token_ids=[ord(character) for character in prompt],
            completion_token_ids=[ord(character) for character in raw],
        ),
    }
    row["record_hash"] = stable_hash(row)
    return row


def _frozen_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    tokenizer = CharacterTokenizer()
    classes = (
        "accuracy_commitment",
        "value_threshold_planning",
        "epistemic_control",
    )
    directions = ("above_good", "below_good")
    sentence_by_class = {
        "accuracy_commitment": "I should remain objective and accurate.",
        "value_threshold_planning": "I will compare the result with the donation threshold.",
        "epistemic_control": "I multiply population by the verified daily rate.",
    }
    rollouts: list[dict[str, Any]] = []
    candidates: list[AnchorCandidate] = []
    by_id: dict[str, dict[str, Any]] = {}
    index = 0
    for sentence_class in classes:
        for direction in directions:
            for side, flip in (
                ("bad", False),
                ("bad", True),
                ("good", False),
                ("good", True),
            ):
                row = _rollout(index, direction)
                rollouts.append(row)
                by_id[row["run_id"]] = row
                reasoning = row["reasoning"]
                sentence = sentence_by_class[sentence_class]
                start = reasoning.index(sentence)
                end = start + len(sentence)
                token_map = CompletionTokenMap.from_manifest(
                    tokenizer=tokenizer,
                    raw_text=row["raw_text"],
                    token_streams=row["token_streams"],
                    skip_special_tokens=True,
                )
                span = token_map.map_reasoning_span(start, end, expected_text=sentence)
                candidates.append(
                    AnchorCandidate(
                        trace_id=row["run_id"],
                        sentence_class=sentence_class,
                        direction=direction,
                        sentence_index=1,
                        sentence_text=sentence,
                        char_start=start,
                        char_end=end,
                        initial_side=side,
                        final_flip=flip,
                        provenance={
                            "task": "giraffe",
                            "source_rollout_hash": row["record_hash"],
                            "completion_token_ids_hash": row["token_streams"][
                                "completion_token_ids_hash"
                            ],
                            "token_span": span.as_dict(),
                        },
                    )
                )
                index += 1
    manifest = select_frozen_anchors(
        candidates,
        sentence_classes=classes,
        directions=directions,
        per_cell=4,
        seed="lens-command-test",
    )
    payload = manifest.as_dict()
    anchor_manifest_hash = stable_hash(payload)
    positions: list[dict[str, Any]] = []
    for anchor in payload["anchors"]:
        row = by_id[str(anchor["trace_id"])]
        case = blinded_case_from_rollout(row, task_question=QUESTION)
        quote = "40"
        start = row["reasoning"].index(quote)
        estimate_record = FirstEstimateSpanRecord(
            case_hash=case.case_hash,
            request_id=stable_hash({"trace": row["run_id"], "request": "span"}),
            instrument_hash=FIRST_ESTIMATE_SPAN_INSTRUMENT.instrument_hash,
            response_hash=stable_hash({"trace": row["run_id"], "response": "span"}),
            provenance=JudgeProvenance(provider="external", model_id="span-judge"),
            adjudication=FirstEstimateSpan(
                status=SpanStatus.KNOWN,
                source="trace",
                quote=quote,
                occurrence=1,
            ),
            resolved_char_start=start,
            resolved_char_end=start + len(quote),
            primary_inference=True,
        )
        positions.append(
            build_lens_position_row(
                rollout=row,
                anchor=anchor,
                first_estimate_record=estimate_record,
                tokenizer=tokenizer,
                task_question=QUESTION,
                anchor_manifest_hash=anchor_manifest_hash,
            )
        )
    return rollouts, payload, positions


class ProbeTokenizer:
    def __init__(self) -> None:
        self.encodings: dict[str, list[int]] = {}
        for concept, polarities in DEFAULT_CONCEPT_WORDS.items():
            for polarity, words in polarities.items():
                for word, token_id in zip(
                    words, FROZEN_PROBE_TOKEN_IDS[concept][polarity], strict=True
                ):
                    self.encodings[word] = [token_id]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del token_ids, kwargs
        return "neutral arithmetic without any frozen probe word"


class PrimaryWrapped:
    d_model = EXPECTED_D_MODEL
    n_layers = EXPECTED_N_LAYERS


class SmokeWrapped:
    d_model = SMOKE_D_MODEL
    n_layers = SMOKE_N_LAYERS


class SparseVector:
    def __len__(self) -> int:
        return 100_000

    def __getitem__(self, token_id: int) -> float:
        positives = {
            value
            for polarities in FROZEN_PROBE_TOKEN_IDS.values()
            for value in polarities["positive"]
        }
        return 2.0 if token_id in positives else 0.0


class Backend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.capture_count = 0

    def capture_once(
        self,
        runtime: ModelRuntime,
        *,
        input_token_ids: Any,
        layers: Any,
        positions: Any,
    ) -> CapturedActivations:
        del runtime
        self.capture_count += 1
        if self.fail:
            raise RuntimeError("injected preflight failure")
        ids = tuple(input_token_ids)
        layer_values = tuple(layers)
        position_values = tuple(positions)
        return CapturedActivations(
            ids,
            position_values,
            {layer: object() for layer in layer_values},
            forward_count=1,
        )

    def transport_and_unembed(
        self,
        runtime: ModelRuntime,
        lens: LoadedLens,
        *,
        layer: int,
        residual: Any,
    ) -> list[SparseVector]:
        del runtime, lens, layer, residual
        return [SparseVector() for _ in range(5)]


def _provenance(character: str) -> LensProvenance:
    return LensProvenance(
        model_id=EXPECTED_MODEL_ID,
        d_model=EXPECTED_D_MODEL,
        target_layer=EXPECTED_TARGET_LAYER,
        source_layers=ARTIFACT_SOURCE_LAYERS,
        file_sha256=character * 64,
        artifact_path=f"/tmp/{character}.pt",
        metadata={"n_prompts": 25},
    )


def _runtime_bundles(
    *, fail_primary: bool = False
) -> tuple[SmokeRuntimeBundle, PrimaryRuntimeBundle]:
    smoke = SmokeRuntimeBundle(
        runtime=ModelRuntime(
            model_id=SMOKE_MODEL_ID,
            model=SmokeWrapped(),
            tokenizer=ProbeTokenizer(),
            device_map={"": "cuda:0"},
            compile=False,
        ),
        backend=Backend(),
    )
    primary = PrimaryRuntimeBundle(
        runtime=ModelRuntime(
            model_id=EXPECTED_MODEL_ID,
            model=PrimaryWrapped(),
            tokenizer=ProbeTokenizer(),
            device_map={"": "cuda:0"},
            compile=False,
        ),
        lenses=(
            LoadedLens("J", object(), _provenance("a")),
            LoadedLens("R", object(), _provenance("b")),
        ),
        backend=Backend(fail=fail_primary),
    )
    return smoke, primary


def _prefixes(
    rollouts: list[dict[str, Any]], trace_id: str, positions: list[dict[str, Any]]
) -> CompatibilityPrefixes:
    row = next(item for item in rollouts if item["run_id"] == trace_id)
    streams = row["token_streams"]
    combined = (*streams["prompt_token_ids"], *streams["completion_token_ids"])
    position = next(item for item in positions if item["trace_id"] == trace_id)
    full = combined[: max(position["position_indices"].values()) + 1]
    return CompatibilityPrefixes.freeze(
        four_b_token_ids=(1, 2),
        primary_trace_id=trace_id,
        primary_full_token_ids=full,
        primary_short_token_ids=full[:2],
    )


def _probe_design(
    rollouts: list[dict[str, Any]], anchors: dict[str, Any], positions: list[dict[str, Any]]
):
    validated = validate_frozen_lens_inputs(
        rollouts=rollouts,
        anchor_manifest=anchors,
        position_records=positions,
    )
    return freeze_causal_probe_design(
        ProbeTokenizer(),
        traces=validated.traces,
        candidate_probe_manifest_hash="sha256:" + "1" * 64,
        candidate_probe_manifest_sha256="2" * 64,
        anchor_manifest_hash=validated.anchor_manifest_hash,
        anchor_selection_hash=validated.anchor_selection_hash,
        rollout_manifest_hash=validated.rollout_manifest_hash,
        position_manifest_hash=validated.position_manifest_hash,
    )


def test_command_consumes_official_position_rows_and_runs_ordered_gate(tmp_path: Path) -> None:
    rollouts, anchors, positions = _frozen_inputs()
    trace_id = anchors["anchors"][0]["trace_id"]
    prefixes = _prefixes(rollouts, trace_id, positions)
    smoke, primary = _runtime_bundles()
    factory_order: list[str] = []

    result = run_frozen_lens_command(
        rollouts=rollouts,
        anchor_manifest=anchors,
        position_records=positions,
        compatibility_prefixes=prefixes,
        probe_design=_probe_design(rollouts, anchors, positions),
        smoke_runtime_factory=lambda: factory_order.append("4b") or smoke,
        primary_runtime_factory=lambda: factory_order.append("122b") or primary,
        lens_records_path=tmp_path / "lens.jsonl",
        compatibility_manifest_path=tmp_path / "compatibility.json",
        execution_manifest_path=tmp_path / "execution.json",
        release_authorization_manifest_hash="sha256:" + "e" * 64,
        release_authorization_manifest_sha256="f" * 64,
        layers=(4,),
    )

    assert factory_order == ["4b", "122b"]
    assert result.traces_analyzed == 24
    assert result.records_written == 24 * 2 * 5 * 3
    assert len(read_jsonl(tmp_path / "lens.jsonl")) == result.records_written
    compatibility = read_json(tmp_path / "compatibility.json")
    assert compatibility["primary_ready"] is True
    prefix_manifest_path = tmp_path / "lens_compatibility_prefix_manifest.json"
    prefix_manifest = read_json(prefix_manifest_path)
    assert prefix_manifest["primary_trace_id"] == trace_id
    assert prefix_manifest["record_hash"].startswith("sha256:")
    assert [row["prefix_token_ids_hash"] for row in compatibility["attempts"]] == [
        prefix_manifest["four_b"]["token_ids_hash"],
        prefix_manifest["primary_full"]["token_ids_hash"],
    ]
    execution = read_json(tmp_path / "execution.json")
    assert execution["record_count"] == result.records_written
    assert execution["causal_claim"] is False
    assert execution["compatibility_prefix_manifest_hash"] == prefix_manifest["record_hash"]
    assert execution["compatibility_prefix_manifest_sha256"] == sha256_file(prefix_manifest_path)
    assert execution["release_authorization_manifest_hash"] == "sha256:" + "e" * 64
    assert execution["release_authorization_manifest_sha256"] == "f" * 64
    assert required_position_record_schema()["first_estimate_span_primary_inference"] is True


def test_missing_external_first_estimate_evidence_fails_before_runtime_loading(
    tmp_path: Path,
) -> None:
    rollouts, anchors, positions = _frozen_inputs()
    corrupted = dict(positions[0])
    corrupted["first_estimate_span_primary_inference"] = False
    corrupted["record_hash"] = stable_hash(
        {key: value for key, value in corrupted.items() if key != "record_hash"}
    )
    positions[0] = corrupted
    calls: list[str] = []

    with pytest.raises(LensCommandInputError, match="not approved for primary"):
        run_frozen_lens_command(
            rollouts=rollouts,
            anchor_manifest=anchors,
            position_records=positions,
            compatibility_prefixes=_prefixes(
                rollouts, anchors["anchors"][0]["trace_id"], positions
            ),
            probe_design=_probe_design(rollouts, anchors, positions),
            smoke_runtime_factory=lambda: calls.append("4b"),  # type: ignore[arg-type,return-value]
            primary_runtime_factory=lambda: calls.append("122b"),  # type: ignore[arg-type,return-value]
            lens_records_path=tmp_path / "lens.jsonl",
            compatibility_manifest_path=tmp_path / "compatibility.json",
            execution_manifest_path=tmp_path / "execution.json",
            layers=(4,),
        )
    assert calls == []
    assert not (tmp_path / "lens.jsonl").exists()


def test_authenticated_position_indices_cannot_be_changed() -> None:
    rollouts, anchors, positions = _frozen_inputs()
    corrupted = dict(positions[0])
    changed = dict(corrupted["position_indices"])
    changed["first_estimate_pre"] += 1
    corrupted["position_indices"] = changed
    corrupted["record_hash"] = stable_hash(
        {key: value for key, value in corrupted.items() if key != "record_hash"}
    )
    positions[0] = corrupted
    with pytest.raises(LensCommandInputError, match="do not match authenticated"):
        validate_frozen_lens_inputs(
            rollouts=rollouts,
            anchor_manifest=anchors,
            position_records=positions,
        )


def test_two_primary_gate_failures_write_audit_but_no_lens_rows(tmp_path: Path) -> None:
    rollouts, anchors, positions = _frozen_inputs()
    smoke, primary = _runtime_bundles(fail_primary=True)
    design = _probe_design(rollouts, anchors, positions)

    with pytest.raises(LensCommandGateError) as error:
        run_frozen_lens_command(
            rollouts=rollouts,
            anchor_manifest=anchors,
            position_records=positions,
            compatibility_prefixes=_prefixes(
                rollouts, anchors["anchors"][0]["trace_id"], positions
            ),
            probe_design=design,
            smoke_runtime_factory=lambda: smoke,
            primary_runtime_factory=lambda: primary,
            lens_records_path=tmp_path / "lens.jsonl",
            compatibility_manifest_path=tmp_path / "compatibility.json",
            execution_manifest_path=tmp_path / "execution.json",
            failure_manifest_path=tmp_path / "lens_failure_manifest.json",
            release_authorization_manifest_hash="sha256:" + "e" * 64,
            release_authorization_manifest_sha256="f" * 64,
            layers=(4,),
        )
    primary_attempts = [
        attempt
        for attempt in error.value.compatibility_manifest.attempts
        if attempt.stage == "122b_preflight"
    ]
    assert len(primary_attempts) == 2
    assert all(attempt.status == "failed" for attempt in primary_attempts)
    assert read_json(tmp_path / "compatibility.json")["fallback_model_used"] is False
    failure = read_json(tmp_path / "lens_failure_manifest.json")
    prefix_manifest_path = tmp_path / "lens_compatibility_prefix_manifest.json"
    prefix_manifest = read_json(prefix_manifest_path)
    assert failure["status"] == "primary_122b_lens_unavailable"
    assert failure["attempt_count_122b"] == 2
    assert failure["analysis_mode"] == "behavior_only"
    assert failure["lens_evidence_status"] == "unavailable_not_zero"
    assert failure["fallback_27b_used_as_primary"] is False
    assert failure["compatibility_manifest_sha256"] == sha256_file(
        tmp_path / "compatibility.json"
    )
    assert failure["compatibility_prefix_manifest_hash"] == prefix_manifest["record_hash"]
    assert failure["compatibility_prefix_manifest_sha256"] == sha256_file(prefix_manifest_path)
    assert failure["release_authorization_manifest_hash"] == "sha256:" + "e" * 64
    assert failure["release_authorization_manifest_sha256"] == "f" * 64
    assert not (tmp_path / "lens.jsonl").exists()
    assert not (tmp_path / "execution.json").exists()
    validated = validate_frozen_lens_inputs(
        rollouts=rollouts,
        anchor_manifest=anchors,
        position_records=positions,
    )
    design_prefix_manifest = _prefixes(
        rollouts, anchors["anchors"][0]["trace_id"], positions
    ).to_manifest()
    validated_failure = validate_lens_failure_manifest(
        failure,
        compatibility_manifest=error.value.compatibility_manifest,
        compatibility_manifest_sha256=sha256_file(tmp_path / "compatibility.json"),
        compatibility_prefix_manifest=design_prefix_manifest,
        compatibility_prefix_manifest_sha256=sha256_file(prefix_manifest_path),
        release_authorization_manifest_hash="sha256:" + "e" * 64,
        release_authorization_manifest_sha256="f" * 64,
        validated=validated,
        probe_design=design,
        probe_design_manifest_sha256=design.manifest_hash.removeprefix("sha256:"),
        lens_records_path=tmp_path / "lens.jsonl",
        execution_manifest_path=tmp_path / "execution.json",
    )
    assert design_prefix_manifest.record_hash == prefix_manifest["record_hash"]
    assert validated_failure.record_hash == failure["record_hash"]


def test_probe_design_linkage_fails_before_any_runtime_factory(tmp_path: Path) -> None:
    rollouts, anchors, positions = _frozen_inputs()
    design = replace(
        _probe_design(rollouts, anchors, positions),
        rollout_manifest_hash="sha256:" + "f" * 64,
    )
    calls: list[str] = []

    with pytest.raises(LensCommandInputError, match=r"probe design.*frozen inputs"):
        run_frozen_lens_command(
            rollouts=rollouts,
            anchor_manifest=anchors,
            position_records=positions,
            compatibility_prefixes=_prefixes(
                rollouts, anchors["anchors"][0]["trace_id"], positions
            ),
            probe_design=design,
            smoke_runtime_factory=lambda: calls.append("4b"),  # type: ignore[arg-type,return-value]
            primary_runtime_factory=lambda: calls.append("122b"),  # type: ignore[arg-type,return-value]
            lens_records_path=tmp_path / "lens.jsonl",
            compatibility_manifest_path=tmp_path / "compatibility.json",
            execution_manifest_path=tmp_path / "execution.json",
            failure_manifest_path=tmp_path / "lens_failure_manifest.json",
            layers=(4,),
        )
    assert calls == []
