from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import pytest

from model_forensics.adjudication import AdjudicationRequest, JudgeProvenance
from model_forensics.anchors import FrozenAnchor, sentence_spans
from model_forensics.io import stable_hash
from model_forensics.prompts import QUESTIONS, Task
from model_forensics.resample_runner import (
    BaseTrace,
    NeutralControlSpec,
    OutcomeAdjudicationError,
    PrefixIdentityError,
    RawPrefixGenerationRequest,
    RawPrefixGenerationResult,
    ReplacementClassificationRequest,
    ReplacementClassificationResult,
    ReplacementTokenTolerance,
    ResampleAllocationManifest,
    ResampleExecutionError,
    build_fixed_stage_two_allocation_manifest,
    build_initial_allocation_manifest,
    run_sentence_resampling,
    select_confirmatory_pairs,
)
from model_forensics.token_spans import token_stream_hash

_DEFAULT_CLASSIFIER = object()
_DEFAULT_OUTCOME_CALLER = object()


def _anchor_and_base(
    *, anchor_id: str = "anchor-1", trace_id: str = "trace-1"
) -> tuple[FrozenAnchor, BaseTrace]:
    trace = (
        "Opening thought.  Accuracy should not be swayed by threshold 100. "
        "Old downstream estimate 90."
    )
    span = sentence_spans(trace)[1]
    anchor = FrozenAnchor(
        anchor_id=anchor_id,
        trace_id=trace_id,
        sentence_class="accuracy_commitment",
        direction="above_good",
        sentence_index=span.index,
        sentence_text=span.text,
        char_start=span.start,
        char_end=span.end,
        initial_side="bad",
        final_flip=True,
    )
    base = BaseTrace(
        base_trace_id=trace_id,
        prompt="Estimate the quantity.",
        trace=trace,
        threshold=100.0,
        condition="above_good",
        task="giraffe",
        provenance={"run": trace_id},
    )
    return anchor, base


class LookupEmbedder:
    def __init__(self, replacement_vector: list[float] | None = None) -> None:
        self.replacement_vector = replacement_vector or [0.0, 1.0]

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        assert texts[0] == "Accuracy should not be swayed by threshold 100."
        assert texts[1]
        return ([1.0, 0.0], self.replacement_vector)

    @property
    def provenance(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "runtime_kind": "unit_test_double",
            "model_revision": "fixed-test-revision",
            "primary_eligible": True,
        }
        payload["provenance_hash"] = stable_hash(payload)
        return payload

    def assert_primary_eligible(self) -> None:
        return None


class RecordingReplacementClassifier:
    def __init__(
        self,
        *,
        feature_changed: bool = True,
        neutral_function_matched: bool = True,
        return_result: bool = True,
        synthetic_smoke: bool = False,
    ) -> None:
        self.feature_changed = feature_changed
        self.neutral_function_matched = neutral_function_matched
        self.return_result = return_result
        self.synthetic_smoke = synthetic_smoke
        self.requests: list[ReplacementClassificationRequest] = []

    @property
    def provenance(self) -> Mapping[str, object]:
        return {
            "provider": "external-test-double",
            "model_id": "blind-judge",
            "revision": "local",
            "synthetic_smoke": self.synthetic_smoke,
        }

    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult | None:
        self.requests.append(request)
        if not self.return_result:
            return None
        return ReplacementClassificationResult(
            request_hash=request.request_hash,
            adjudication_valid=True,
            target_feature_absent_or_changed=self.feature_changed,
            neutral_control_function_matched=self.neutral_function_matched,
            raw_judgment_hashes=("a" * 64, "b" * 64),
            classifier_provenance_hash=stable_hash(self.provenance),
            rationale="Two blinded judgments agreed on the replacement function.",
        )


class RecordingOutcomeCaller:
    def __init__(
        self,
        *,
        not_for_primary_inference: bool = False,
        forced_value: int | object | None = _DEFAULT_OUTCOME_CALLER,
    ) -> None:
        self.not_for_primary_inference = not_for_primary_inference
        self.forced_value = forced_value
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="external-test-double",
            model_id="blind-final-judge",
            model_revision="fixture-v1",
            caller_version="test-v1",
            metadata={
                "calls_completed": len(self.requests),
                "charged_cost_usd": len(self.requests) * 0.001,
            },
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        if self.forced_value is not _DEFAULT_OUTCOME_CALLER:
            if self.forced_value is None:
                return '{"status":"UNKNOWN","value":null}'
            return f'{{"status":"KNOWN","value":"{self.forced_value}"}}'
        answer = request.user_payload["answer"]
        if answer == "Final answer: 120":
            return '{"status":"KNOWN","value":"120"}'
        if answer == "Final answer: 80":
            return '{"status":"KNOWN","value":"80"}'
        return '{"status":"UNKNOWN","value":null}'


def _neutral_control() -> NeutralControlSpec:
    return NeutralControlSpec(
        control_id="position-matched-epistemic-control",
        function_definition=(
            "A calculation or uncertainty sentence that estimates the quantity without an "
            "accuracy commitment or value-side targeting."
        ),
        version="neutral-control-v1",
    )


class RecordingRawPrefixBackend:
    def __init__(
        self,
        *,
        malformed: str | None = None,
        corrupt_prompt_tokens: bool = False,
        replacement_token_count: int | None = None,
        synthetic_smoke: bool = False,
    ) -> None:
        self.malformed = malformed
        self.corrupt_prompt_tokens = corrupt_prompt_tokens
        self.replacement_token_count = replacement_token_count
        self.synthetic_smoke = synthetic_smoke
        self.prefix_calls: list[tuple[tuple[Mapping[str, object], ...], str]] = []
        self.continuation_calls: list[str] = []
        self.requests: tuple[RawPrefixGenerationRequest, ...] = ()
        self._decoded_pieces: dict[int, str] = {}
        self._next_piece_id = 1_000_000

    @property
    def tokenizer(self):
        return self

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens
        assert clean_up_tokenization_spaces is False
        return "".join(self._decoded_pieces.get(token_id, chr(token_id)) for token_id in token_ids)

    def _piece_id(self, text: str) -> int:
        token_id = self._next_piece_id
        self._next_piece_id += 1
        self._decoded_pieces[token_id] = text
        return token_id

    def _completion_ids(self, generated: str, *, arm: str) -> tuple[int, ...]:
        if arm != "resample" or self.malformed == "missing_close":
            return (self._piece_id(generated),)
        reasoning = generated.split("</think>", 1)[0]
        replacement = sentence_spans(reasoning)[0]
        token_count = self.replacement_token_count or 3
        boundaries = [round(index * len(replacement.text) / token_count) for index in range(token_count + 1)]
        pieces = [
            replacement.text[boundaries[index] : boundaries[index + 1]]
            for index in range(token_count)
        ]
        assert all(pieces)
        suffix = generated[replacement.end :]
        return (
            *(self._piece_id(piece) for piece in pieces),
            self._piece_id(suffix),
        )

    @property
    def provenance(self) -> Mapping[str, object]:
        return {
            "backend": "test-double",
            "model_id": "no-network",
            "revision": "local",
            "synthetic_smoke": self.synthetic_smoke,
        }

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, object]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]:
        self.prefix_calls.append((tuple(messages), raw_thinking_prefix))
        return (101, len(raw_thinking_prefix), 202)

    def encode_continuation(self, raw_text: str) -> Sequence[int]:
        self.continuation_calls.append(raw_text)
        if (
            raw_text == "A genuinely different replacement."
            and self.replacement_token_count is not None
        ):
            return tuple(range(400, 400 + self.replacement_token_count))
        return (301, len(raw_text), 302)

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]:
        self.requests = tuple(requests)
        results = []
        for request in requests:
            if self.malformed == "missing_close":
                generated = "Unclosed reasoning with 120."
            elif self.malformed == "missing_answer":
                generated = "Reasoning ends here.</think>   "
            elif self.malformed == "nonnumeric_answer":
                generated = "Reasoning ends here.</think>Final answer unavailable."
            elif request.arm == "retain":
                generated = " Regenerated downstream estimate is 80.</think>Final answer: 80"
            else:
                generated = (
                    "A genuinely different replacement. Downstream estimate is 120."
                    "</think>Final answer: 120"
                )
            prompt_token_ids = request.prompt_token_ids
            if self.corrupt_prompt_tokens:
                prompt_token_ids = (*prompt_token_ids, 999)
            completion_token_ids = self._completion_ids(generated, arm=request.arm)
            results.append(
                RawPrefixGenerationResult(
                    request_id=request.request_id,
                    generated_text=generated,
                    prompt_token_ids=prompt_token_ids,
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=len(completion_token_ids),
                    backend_metadata={
                        "seed": request.seed,
                        "completion_token_ids": list(completion_token_ids),
                        "completion_token_ids_hash": token_stream_hash(
                            completion_token_ids,
                            stream="completion",
                        ),
                    },
                )
            )
        return results


def _run(
    *,
    backend: RecordingRawPrefixBackend | None = None,
    embedder: LookupEmbedder | None = None,
    classifier: RecordingReplacementClassifier | object | None = _DEFAULT_CLASSIFIER,
    outcome_caller: RecordingOutcomeCaller | object = _DEFAULT_OUTCOME_CALLER,
    primary_inference: bool = True,
    neutral_control: NeutralControlSpec | None = None,
    token_tolerance: ReplacementTokenTolerance | None = None,
):
    anchor, base = _anchor_and_base()
    manifest = build_initial_allocation_manifest((anchor,), master_seed=20260829)
    backend = backend or RecordingRawPrefixBackend()
    if classifier is _DEFAULT_CLASSIFIER:
        classifier = RecordingReplacementClassifier()
    if outcome_caller is _DEFAULT_OUTCOME_CALLER:
        outcome_caller = RecordingOutcomeCaller()
    assert isinstance(outcome_caller, RecordingOutcomeCaller)
    all_records = run_sentence_resampling(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        embedder=embedder or LookupEmbedder(),
        outcome_caller=outcome_caller,
        primary_inference=primary_inference,
        replacement_classifier=classifier,
        neutral_control=neutral_control or _neutral_control(),
        token_tolerance=token_tolerance or ReplacementTokenTolerance(0, 0.0),
    )
    records = tuple(record for record in all_records if record.sample_index == 0)
    return anchor, base, backend, classifier, records


def test_initial_manifest_pairs_seeds_deterministically_and_uniquely() -> None:
    first_anchor, _ = _anchor_and_base()
    second_anchor, _ = _anchor_and_base(anchor_id="anchor-2", trace_id="trace-2")

    first = build_initial_allocation_manifest((first_anchor, second_anchor), master_seed=20260829)
    second = build_initial_allocation_manifest((second_anchor, first_anchor), master_seed=20260829)

    assert first.as_dict() == second.as_dict()
    assert len(first.allocations) == 2 * 10 * 2
    pairs: dict[tuple[str, int], list] = {}
    for allocation in first.allocations:
        pairs.setdefault((allocation.anchor_id, allocation.sample_index), []).append(allocation)
    assert all({item.arm for item in pair} == {"retain", "resample"} for pair in pairs.values())
    assert all(len({item.seed for item in pair}) == 1 for pair in pairs.values())
    assert len({pair[0].seed for pair in pairs.values()}) == 20


def test_runner_conditions_exactly_and_preserves_fixed_prefix_and_full_artifacts() -> None:
    anchor, base, backend, _, records = _run()
    by_arm = {record.arm: record for record in records}
    requests = {request.arm: request for request in backend.requests}
    common_text = base.trace[: anchor.char_start]

    assert backend.prefix_calls == [(base.messages, common_text)]
    assert backend.continuation_calls == [anchor.sentence_text]
    assert requests["resample"].conditioning_text == common_text
    assert requests["retain"].conditioning_text == common_text + anchor.sentence_text
    assert requests["retain"].prompt_token_ids[:3] == requests["resample"].prompt_token_ids
    assert requests["retain"].seed == requests["resample"].seed
    assert requests["retain"].vllm_prompt == {
        "prompt_token_ids": list(requests["retain"].prompt_token_ids)
    }

    assert by_arm["retain"].full_trace.startswith(common_text + anchor.sentence_text)
    assert by_arm["resample"].full_trace.startswith(
        common_text + "A genuinely different replacement."
    )
    assert by_arm["retain"].common_prefix_hash == by_arm["resample"].common_prefix_hash
    assert by_arm["retain"].conditioning_prefix_hash != by_arm["resample"].conditioning_prefix_hash
    assert all(not record.synthetic_smoke for record in records)
    assert all(record.provenance["backend"]["model_id"] == "no-network" for record in records)
    assert all(
        record.provenance["semantic_embedder"]["model_revision"]
        == "fixed-test-revision"
        for record in records
    )
    assert all(
        record.usage["completion_tokens"]
        == len(record.provenance["backend_result"]["completion_token_ids"])
        for record in records
    )


def test_primary_runner_refuses_unattributed_semantic_embedder() -> None:
    class UnattributedEmbedder:
        def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            del texts
            return ([1.0, 0.0], [0.0, 1.0])

    with pytest.raises(ResampleExecutionError, match="eligibility check"):
        _run(embedder=UnattributedEmbedder())  # type: ignore[arg-type]


def test_runner_records_first_replacement_divergence_final_estimate_and_good_side() -> None:
    anchor, _, _, _, records = _run()
    by_arm = {record.arm: record for record in records}
    resample = by_arm["resample"]
    retain = by_arm["retain"]

    assert resample.replacement_sentence == "A genuinely different replacement."
    assert (
        resample.full_trace[resample.replacement_char_start : resample.replacement_char_end]
        == resample.replacement_sentence
    )
    assert resample.cosine_similarity == 0.0
    assert resample.divergent is True
    assert resample.intervention_eligible is True
    assert resample.primary_eligible is True
    assert resample.final_estimate == 120.0
    assert resample.final_good_side is True
    assert resample.final_measurement_valid is True
    assert resample.answer == "Final answer: 120"
    assert resample.trajectory is None
    assert resample.outcome_adjudication_primary_inference is True
    assert resample.outcome_adjudication["status"] == "KNOWN"
    assert resample.outcome_adjudication["value"] == 120
    assert resample.outcome_adjudication["audit_hash"].startswith("sha256:")

    assert retain.replacement_sentence == anchor.sentence_text
    assert retain.cosine_similarity == 1.0
    assert retain.divergent is False
    assert retain.intervention_eligible is True
    assert retain.primary_eligible is True
    assert retain.final_estimate == 80.0
    assert retain.final_good_side is False
    assert retain.as_dict()["replacement_span"] == {
        "start": anchor.char_start,
        "end": anchor.char_end,
    }
    assert retain.as_dict()["record_hash"].startswith("sha256:")


def test_final_outcome_judge_sees_only_neutral_question_trace_and_answer() -> None:
    caller = RecordingOutcomeCaller()
    _, _, _, _, records = _run(outcome_caller=caller)

    assert len(caller.requests) == 20
    for request in caller.requests:
        assert set(request.user_payload) == {"task_question", "trace", "answer"}
        assert request.user_payload["task_question"] == QUESTIONS[Task.GIRAFFE]
        assert "threshold" not in request.user_payload["task_question"].lower()
        assert "good cause" not in request.user_payload["task_question"].lower()
        serialized = repr(dict(request.user_payload)).lower()
        for forbidden_field in ("anchor_id", "base_trace_id", "arm", "condition", "seed"):
            assert forbidden_field not in serialized

    audit = records[0].outcome_adjudication
    assert audit["judge_provenance"]["provider"] == "external-test-double"
    assert audit["judge_provenance"]["metadata"]["charged_cost_usd"] > 0
    assert audit["not_for_primary_inference"] is False
    assert audit["visible_payload_fields"] == ["task_question", "trace", "answer"]
    assert audit["experimental_metadata_included"] is False


def test_primary_final_uses_judge_value_even_when_answer_parser_would_disagree() -> None:
    caller = RecordingOutcomeCaller(forced_value=777)
    _, _, _, _, records = _run(outcome_caller=caller)

    assert {record.final_estimate for record in records} == {777}
    assert all(record.final_good_side is True for record in records)
    assert all(record.outcome_adjudication["value"] == 777 for record in records)


def test_primary_guards_fail_before_generation_and_nonprimary_is_never_confirmatory() -> None:
    synthetic_backend = RecordingRawPrefixBackend(synthetic_smoke=True)
    with pytest.raises(OutcomeAdjudicationError, match="synthetic smoke backend"):
        _run(backend=synthetic_backend)
    assert synthetic_backend.requests == ()

    rejected_caller = RecordingOutcomeCaller(not_for_primary_inference=True)
    ordinary_backend = RecordingRawPrefixBackend()
    with pytest.raises(OutcomeAdjudicationError, match="not_for_primary_inference"):
        _run(backend=ordinary_backend, outcome_caller=rejected_caller)
    assert ordinary_backend.requests == ()
    assert rejected_caller.requests == []

    smoke_backend = RecordingRawPrefixBackend(synthetic_smoke=True)
    _, _, _, _, smoke_records = _run(
        backend=smoke_backend,
        outcome_caller=RecordingOutcomeCaller(not_for_primary_inference=True),
        primary_inference=False,
    )
    assert all(record.final_measurement_valid for record in smoke_records)
    assert all(not record.intervention_eligible for record in smoke_records)
    assert all(not record.primary_eligible for record in smoke_records)
    assert all(record.analysis_tier == "nonprimary_smoke" for record in smoke_records)
    assert select_confirmatory_pairs(smoke_records) == ()


def test_point_eight_resample_is_nondivergent_and_excluded_from_primary() -> None:
    _, _, _, _, records = _run(embedder=LookupEmbedder([0.8, 0.6]))
    by_arm = {record.arm: record for record in records}

    assert by_arm["resample"].cosine_similarity == pytest.approx(0.8)
    assert by_arm["resample"].divergent is False
    assert by_arm["resample"].intervention_eligible is False
    assert by_arm["resample"].primary_eligible is False
    assert by_arm["retain"].primary_eligible is True


def test_confirmatory_gate_uses_only_blinded_input_and_records_raw_audit_hashes() -> None:
    classifier = RecordingReplacementClassifier()
    _, _, _, _, records = _run(classifier=classifier)
    resample = next(record for record in records if record.arm == "resample")
    request = classifier.requests[0]
    visible = request.visible_payload()

    assert set(visible) == {
        "protocol_version",
        "blinded_original_sentence",
        "blinded_replacement_sentence",
        "target_sentence_class",
        "neutral_control_id",
        "neutral_control_function",
        "neutral_control_version",
    }
    assert "[THRESHOLD_REDACTED]" in visible["blinded_original_sentence"]
    serialized_visible = repr(visible).lower()
    for forbidden in (
        "base_trace_id",
        "anchor_id",
        "condition",
        "direction",
        "final_estimate",
        "good_side",
        "seed",
    ):
        assert forbidden not in serialized_visible

    assert resample.replacement_classification_status == "valid"
    assert resample.target_feature_absent_or_changed is True
    assert resample.neutral_control_function_matched is True
    assert resample.classifier_input_blinded is True
    assert resample.classifier_input_hash == request.input_hash
    assert resample.classifier_request_hash == request.request_hash
    assert resample.classifier_judgment_hashes == ("a" * 64, "b" * 64)
    assert resample.classifier_provenance_hash == stable_hash(classifier.provenance)
    assert resample.neutral_control_hash == _neutral_control().control_hash
    assert resample.anchor_token_count == 3
    assert resample.replacement_token_count == 3
    assert resample.token_count_absolute_difference == 0
    assert resample.token_count_relative_difference == 0.0
    assert resample.token_count_absolute_tolerance == 0
    assert resample.token_count_relative_tolerance == 0.0
    assert resample.token_count_within_absolute_tolerance is True
    assert resample.token_count_within_relative_tolerance is True
    assert resample.confirmatory_eligible is True
    assert resample.intervention_eligible is True
    assert resample.analysis_tier == "confirmatory"


def test_missing_replacement_classification_is_explicitly_exploratory() -> None:
    _, _, _, classifier, records = _run(classifier=None)
    assert classifier is None
    by_arm = {record.arm: record for record in records}

    assert by_arm["resample"].replacement_classification_status == "missing"
    assert by_arm["resample"].confirmatory_eligible is False
    assert by_arm["resample"].intervention_eligible is False
    assert by_arm["resample"].primary_eligible is False
    assert by_arm["resample"].analysis_tier == "exploratory"
    assert by_arm["resample"].classifier_judgment_hashes == ()
    assert by_arm["retain"].replacement_classification_status == "paired_reference"
    assert by_arm["retain"].confirmatory_eligible is True
    assert by_arm["retain"].analysis_tier == "paired_reference"
    assert select_confirmatory_pairs(records) == ()


def test_confirmatory_selection_keeps_valid_resample_and_its_paired_retain() -> None:
    _, _, _, _, records = _run()

    selected = select_confirmatory_pairs(records)

    assert selected == records
    assert {record.arm for record in selected} == {"retain", "resample"}


@pytest.mark.parametrize(
    ("feature_changed", "neutral_matched"),
    [(False, True), (True, False)],
)
def test_each_blinded_function_judgment_is_required_for_confirmation(
    feature_changed: bool, neutral_matched: bool
) -> None:
    classifier = RecordingReplacementClassifier(
        feature_changed=feature_changed,
        neutral_function_matched=neutral_matched,
    )
    _, _, _, _, records = _run(classifier=classifier)
    resample = next(record for record in records if record.arm == "resample")

    assert resample.replacement_classification_status == "valid"
    assert resample.confirmatory_eligible is False
    assert resample.primary_eligible is False
    assert resample.analysis_tier == "exploratory"


def test_absolute_and_relative_token_tolerances_are_both_required() -> None:
    backend = RecordingRawPrefixBackend(replacement_token_count=5)
    _, _, _, _, records = _run(
        backend=backend,
        token_tolerance=ReplacementTokenTolerance(
            max_absolute_difference=2,
            max_relative_difference=0.50,
        ),
    )
    resample = next(record for record in records if record.arm == "resample")

    assert resample.token_count_absolute_difference == 2
    assert resample.token_count_relative_difference == pytest.approx(2 / 3)
    assert resample.token_count_within_absolute_tolerance is True
    assert resample.token_count_within_relative_tolerance is False
    assert resample.confirmatory_eligible is False
    assert resample.analysis_tier == "exploratory"


@pytest.mark.parametrize("failure", ["missing_close", "missing_answer"])
def test_runner_isolates_malformed_generated_continuations(failure: str) -> None:
    _, _, _, _, records = _run(backend=RecordingRawPrefixBackend(malformed=failure))

    assert len(records) == 2
    assert all(record.analysis_tier == "generation_invalid" for record in records)
    assert all(not record.final_measurement_valid for record in records)
    assert all(not record.intervention_eligible for record in records)
    assert all(
        record.replacement_classification_status == "not_called_terminal_invalid"
        for record in records
    )
    assert all(record.raw_generated_text for record in records)


def test_unrecoverable_final_is_judge_unknown_not_a_local_parser_failure() -> None:
    _, _, _, _, records = _run(backend=RecordingRawPrefixBackend(malformed="nonnumeric_answer"))

    assert all(record.final_estimate is None for record in records)
    assert all(record.final_good_side is None for record in records)
    assert all(not record.final_measurement_valid for record in records)
    assert all(not record.primary_eligible for record in records)
    assert all(record.intervention_eligible for record in records)
    assert all(record.analysis_tier == "outcome_unmeasured" for record in records)
    assert select_confirmatory_pairs(records) == records


def test_runner_rejects_backend_prompt_token_identity_break() -> None:
    with pytest.raises(PrefixIdentityError, match="different prompt tokens"):
        _run(backend=RecordingRawPrefixBackend(corrupt_prompt_tokens=True))


def test_fixed_stage_two_unconditionally_adds_ten_pairs_for_every_anchor() -> None:
    first, _ = _anchor_and_base()
    second, _ = _anchor_and_base(anchor_id="anchor-2", trace_id="trace-2")
    anchors = (first, second)
    initial = build_initial_allocation_manifest(anchors, master_seed=20260829)

    stage_two = build_fixed_stage_two_allocation_manifest(
        anchors,
        initial_manifest=initial,
        master_seed=20260829,
    )

    assert len(stage_two.allocations) == 2 * 2 * 10
    assert {allocation.sample_index for allocation in stage_two.allocations} == set(range(10, 20))
    assert not {item.seed for item in initial.allocations}.intersection(
        item.seed for item in stage_two.allocations
    )
    assert stage_two.stage == "stage_two"
    assert stage_two.stage_two_policy_hash is not None
    for anchor in anchors:
        for sample_index in range(10, 20):
            pair = [
                item
                for item in stage_two.allocations
                if item.anchor_id == anchor.anchor_id and item.sample_index == sample_index
            ]
            assert {item.arm for item in pair} == {"retain", "resample"}
            assert len({item.seed for item in pair}) == 1

    parameters = inspect.signature(build_fixed_stage_two_allocation_manifest).parameters
    assert "summaries" not in parameters
    assert "class_ci_half_widths" not in parameters
    assert "outcomes" not in parameters


def test_fixed_stage_two_refuses_partial_initial_or_a_different_sample_count() -> None:
    anchor, _ = _anchor_and_base()
    initial = build_initial_allocation_manifest((anchor,), master_seed=20260829)
    partial_allocations = initial.allocations[:-1]
    partial = ResampleAllocationManifest(
        allocations=partial_allocations,
        master_seed=initial.master_seed,
        stage=initial.stage,
        manifest_hash="sha256:partial",
    )

    with pytest.raises(ValueError, match="exactly samples 0--9"):
        build_fixed_stage_two_allocation_manifest(
            (anchor,), initial_manifest=partial, master_seed=20260829
        )
    with pytest.raises(ValueError, match="exactly 10"):
        build_fixed_stage_two_allocation_manifest(
            (anchor,),
            initial_manifest=initial,
            master_seed=20260829,
            additional_samples_per_arm=9,
        )
