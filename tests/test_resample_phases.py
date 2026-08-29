from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from model_forensics.adjudication import AdjudicationRequest, JudgeProvenance
from model_forensics.anchors import FrozenAnchor, sentence_spans
from model_forensics.io import stable_hash
from model_forensics.resample_phases import (
    GENERATION_STATUS_TERMINAL_INVALID,
    ResamplingGenerationRecord,
    adjudicate_sentence_resampling_intermediates,
    generate_sentence_resampling_intermediates,
)
from model_forensics.resample_runner import (
    BaseTrace,
    NeutralControlSpec,
    PrefixIdentityError,
    RawPrefixGenerationRequest,
    RawPrefixGenerationResult,
    ReplacementClassificationRequest,
    ReplacementClassificationResult,
    ReplacementTokenTolerance,
    ResampleExecutionError,
    _replacement_token_audit,
    build_fixed_stage_two_allocation_manifest,
    build_initial_allocation_manifest,
    run_sentence_resampling,
)
from model_forensics.token_spans import token_stream_hash


def _anchor_and_base(
    index: int = 0,
    *,
    exact_anchor_tokens: tuple[int, ...] | None = None,
) -> tuple[FrozenAnchor, BaseTrace]:
    trace = (
        f"Opening thought {index}.  Accuracy should not be swayed by threshold 100. "
        "Old downstream estimate 90."
    )
    span = sentence_spans(trace)[1]
    provenance: dict[str, Any] = {}
    if exact_anchor_tokens is not None:
        completion_hash = token_stream_hash((70, *exact_anchor_tokens, 71), stream="completion")
        provenance = {
            "completion_token_ids_hash": completion_hash,
            "token_span": {
                "schema_version": "1",
                "section": "reasoning",
                "section_char_start": span.start,
                "section_char_end": span.end,
                "completion_char_start": span.start,
                "completion_char_end": span.end,
                "token_start": 1,
                "token_end": 1 + len(exact_anchor_tokens),
                "token_envelope_char_start": span.start,
                "token_envelope_char_end": span.end,
                "text": span.text,
                "leading_envelope_text": "",
                "trailing_envelope_text": "",
                "token_ids": list(exact_anchor_tokens),
                "token_ids_hash": token_stream_hash(exact_anchor_tokens, stream="completion_span"),
                "completion_token_ids_hash": completion_hash,
                "round_trip_verified": True,
            },
        }
    anchor = FrozenAnchor(
        anchor_id=f"anchor-{index:02d}",
        trace_id=f"trace-{index:02d}",
        sentence_class="accuracy_commitment",
        direction="above_good",
        sentence_index=span.index,
        sentence_text=span.text,
        char_start=span.start,
        char_end=span.end,
        initial_side="bad",
        final_flip=True,
        provenance=provenance,
    )
    return anchor, BaseTrace(
        base_trace_id=anchor.trace_id,
        prompt="Estimate the quantity.",
        trace=trace,
        threshold=100.0,
        condition="above_good",
        task="giraffe",
        provenance={"run": anchor.trace_id},
    )


class DeterministicBackend:
    def __init__(self, *, malformed_request_id: str | None = None) -> None:
        self.malformed_request_id = malformed_request_id
        self.generated_request_ids: list[str] = []
        self.continuation_calls: list[str] = []

    @property
    def tokenizer(self) -> Any:
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
        return "".join(chr(token_id) for token_id in token_ids)

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "backend": "deterministic-test",
            "model_id": "no-network",
            "revision": "fixed",
            "synthetic_smoke": False,
        }

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]:
        del messages
        return (101, len(raw_thinking_prefix), 202)

    def encode_continuation(self, raw_text: str) -> Sequence[int]:
        self.continuation_calls.append(raw_text)
        return tuple(ord(character) for character in raw_text)

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]:
        results: list[RawPrefixGenerationResult] = []
        for request in requests:
            self.generated_request_ids.append(request.request_id)
            if request.request_id == self.malformed_request_id:
                text = "Unclosed but preserved raw output."
            elif request.arm == "retain":
                text = " Continued reasoning.</think>Final answer: 80"
            else:
                text = "Accuracy  may never be swayed by threshold 100. More.</think>Final answer: 120"
            completion_ids = tuple(ord(character) for character in text)
            results.append(
                RawPrefixGenerationResult(
                    request_id=request.request_id,
                    generated_text=text,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_tokens=len(request.prompt_token_ids),
                    completion_tokens=len(completion_ids),
                    backend_metadata={
                        "completion_token_ids": list(completion_ids),
                        "completion_token_ids_hash": token_stream_hash(
                            completion_ids, stream="completion"
                        ),
                        "seed": request.seed,
                    },
                )
            )
        return results


class PinnedQwenLeadingSpaceBackend(DeterministicBackend):
    """Frozen regression for Qwen3.5-4B revision 851bf6e8... token boundaries."""

    model_id = "Qwen/Qwen3.5-4B"
    model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    leading_suppose_token_id = 79762
    standalone_suppose_token_ids = (9751, 2806)
    something_token_id = 24999
    period_token_id = 13
    _character_offset = 300_000

    def encode_continuation(self, raw_text: str) -> Sequence[int]:
        self.continuation_calls.append(raw_text)
        raise AssertionError("active primary resampling must not re-tokenize a replacement")

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens
        assert clean_up_tokenization_spaces is False
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == self.leading_suppose_token_id:
                pieces.append(" Suppose")
            elif token_id == self.something_token_id:
                pieces.append(" something")
            elif token_id == self.period_token_id:
                pieces.append(".")
            elif token_id >= self._character_offset:
                pieces.append(chr(token_id - self._character_offset))
            else:
                pieces.append(chr(token_id))
        return "".join(pieces)

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]:
        results: list[RawPrefixGenerationResult] = []
        for request in requests:
            self.generated_request_ids.append(request.request_id)
            if request.arm == "retain":
                text = " Continued reasoning.</think>Final answer: 80"
                completion_ids = tuple(self._character_offset + ord(character) for character in text)
            else:
                suffix = "</think>Final answer: 120"
                text = f" Suppose something.{suffix}"
                completion_ids = (
                    self.leading_suppose_token_id,
                    self.something_token_id,
                    self.period_token_id,
                    *(self._character_offset + ord(character) for character in suffix),
                )
            results.append(
                RawPrefixGenerationResult(
                    request_id=request.request_id,
                    generated_text=text,
                    prompt_token_ids=request.prompt_token_ids,
                    prompt_tokens=len(request.prompt_token_ids),
                    completion_tokens=len(completion_ids),
                    backend_metadata={
                        "completion_token_ids": list(completion_ids),
                        "completion_token_ids_hash": token_stream_hash(
                            completion_ids, stream="completion"
                        ),
                    },
                )
            )
        return results


class IntegrityFailureBackend(DeterministicBackend):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    @property
    def tokenizer(self) -> Any:
        return None if self.mode == "missing_tokenizer" else self

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]:
        original = super().generate(requests)
        results: list[RawPrefixGenerationResult] = []
        for result in original:
            metadata = dict(result.backend_metadata)
            if self.mode == "missing_completion_ids":
                metadata.pop("completion_token_ids", None)
                metadata.pop("completion_token_ids_hash", None)
            elif self.mode == "non_round_trip":
                completion_ids = tuple(metadata["completion_token_ids"])
                corrupted = (ord("X"), *completion_ids[1:])
                metadata["completion_token_ids"] = list(corrupted)
                metadata["completion_token_ids_hash"] = token_stream_hash(
                    corrupted, stream="completion"
                )
            results.append(
                RawPrefixGenerationResult(
                    request_id=result.request_id,
                    generated_text=result.generated_text,
                    prompt_token_ids=result.prompt_token_ids,
                    finish_reason=result.finish_reason,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cost_usd=result.cost_usd,
                    backend_metadata=metadata,
                )
            )
        return results


class DivergentEmbedder:
    def __init__(self, *, cosine: float = 0.0) -> None:
        self.cosine = cosine
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        assert len(texts) == 2
        return ([1.0, 0.0], [self.cosine, (1.0 - self.cosine**2) ** 0.5])

    @property
    def provenance(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "runtime_kind": "unit_test_double",
            "model_revision": "fixed-test-revision",
            "primary_eligible": True,
        }
        payload["provenance_hash"] = stable_hash(payload)
        return payload

    def assert_primary_eligible(self) -> None:
        return None


class CountingClassifier:
    def __init__(self) -> None:
        self.requests: list[ReplacementClassificationRequest] = []

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"provider": "test", "model_id": "classifier", "revision": "fixed"}

    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult:
        self.requests.append(request)
        return ReplacementClassificationResult(
            request_hash=request.request_hash,
            adjudication_valid=True,
            target_feature_absent_or_changed=True,
            neutral_control_function_matched=True,
            raw_judgment_hashes=("a" * 64, "b" * 64),
            classifier_provenance_hash=stable_hash(self.provenance),
            rationale="Both frozen routes agreed.",
        )


class CountingOutcomeCaller:
    not_for_primary_inference = False

    def __init__(self) -> None:
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="test",
            model_id="outcome",
            model_revision="fixed",
            caller_version="v1",
            metadata={"call_index": len(self.requests)},
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        value = 120 if "120" in request.user_payload["answer"] else 80
        return f'{{"status":"KNOWN","value":"{value}"}}'


def _neutral() -> NeutralControlSpec:
    return NeutralControlSpec(
        control_id="neutral",
        function_definition="A position-matched calculation sentence.",
        version="v1",
    )


def _cpu(
    intermediates: Sequence[ResamplingGenerationRecord],
    *,
    anchors: Sequence[FrozenAnchor],
    bases: Mapping[str, BaseTrace],
    manifest: Any,
    embedder: DivergentEmbedder | None = None,
    classifier: CountingClassifier | None = None,
    caller: CountingOutcomeCaller | None = None,
):
    return adjudicate_sentence_resampling_intermediates(
        intermediates,
        anchors=anchors,
        base_traces=bases,
        allocation_manifest=manifest,
        embedder=embedder or DivergentEmbedder(),
        outcome_caller=caller or CountingOutcomeCaller(),
        primary_inference=True,
        replacement_classifier=classifier or CountingClassifier(),
        neutral_control=_neutral(),
        token_tolerance=ReplacementTokenTolerance(0, 0.0),
    )


def test_gpu_phase_has_no_embedding_classification_or_adjudication_path(monkeypatch) -> None:
    anchor, base = _anchor_and_base()
    manifest = build_initial_allocation_manifest((anchor,), master_seed=44)
    backend = DeterministicBackend()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("CPU/API work was invoked during GPU generation")

    monkeypatch.setattr("model_forensics.resample_phases.assess_semantic_divergence", forbidden)
    monkeypatch.setattr(
        "model_forensics.resample_phases._replacement_classification_audit", forbidden
    )
    monkeypatch.setattr(
        "model_forensics.resample_phases.adjudicate_final_resampling_outcome", forbidden
    )
    rows = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        primary_inference=True,
        microbatch_size=3,
    )

    assert len(rows) == 20
    assert all(row.record_hash.startswith("sha256:") for row in rows)


def test_generation_materializes_deterministic_complete_24_by_2_by_20() -> None:
    pairs = tuple(_anchor_and_base(index) for index in range(24))
    anchors = tuple(pair[0] for pair in pairs)
    bases = {pair[1].base_trace_id: pair[1] for pair in pairs}
    initial = build_initial_allocation_manifest(anchors, master_seed=20260829)
    stage_two = build_fixed_stage_two_allocation_manifest(
        anchors,
        initial_manifest=initial,
        master_seed=20260829,
    )
    backend = DeterministicBackend()

    first = generate_sentence_resampling_intermediates(
        anchors,
        base_traces=bases,
        allocation_manifest=initial,
        backend=backend,
        primary_inference=True,
        microbatch_size=37,
    )
    second = generate_sentence_resampling_intermediates(
        anchors,
        base_traces=bases,
        allocation_manifest=stage_two,
        backend=backend,
        primary_inference=True,
        microbatch_size=37,
    )

    rows = (*first, *second)
    assert len(rows) == 24 * 2 * 20
    assert len({row.resample_id for row in rows}) == len(rows)
    assert [row.resample_id for row in first] == [item.request_id for item in initial.allocations]
    assert [row.resample_id for row in second] == [
        item.request_id for item in stage_two.allocations
    ]


def test_resume_uses_exact_ids_without_duplicate_generation_and_rejects_mismatch() -> None:
    anchor, base = _anchor_and_base()
    manifest = build_initial_allocation_manifest((anchor,), master_seed=55)
    first_backend = DeterministicBackend()
    all_rows = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=first_backend,
        primary_inference=True,
        microbatch_size=4,
    )
    resumed = all_rows[:7]
    second_backend = DeterministicBackend()
    unit_ids: list[str] = []
    batches: list[tuple[int, tuple[str, ...]]] = []
    combined = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=second_backend,
        primary_inference=True,
        microbatch_size=4,
        resume_records=[row.as_dict(include_hash=True) for row in resumed],
        on_intermediate=lambda row: unit_ids.append(row.resample_id),
        on_microbatch=lambda index, rows: batches.append(
            (index, tuple(row.resample_id for row in rows))
        ),
    )

    expected_pending = [item.request_id for item in manifest.allocations][7:]
    assert second_backend.generated_request_ids == expected_pending
    assert unit_ids == expected_pending
    assert [row.resample_id for row in combined] == [
        item.request_id for item in manifest.allocations
    ]
    assert all(ids for _, ids in batches)

    duplicate = [resumed[0], resumed[0]]
    with pytest.raises(ResampleExecutionError, match="duplicate resumed"):
        generate_sentence_resampling_intermediates(
            (anchor,),
            base_traces={base.base_trace_id: base},
            allocation_manifest=manifest,
            backend=DeterministicBackend(),
            primary_inference=True,
            resume_records=duplicate,
        )
    tampered = resumed[0].as_dict(include_hash=False)
    tampered["seed"] += 1
    tampered["record_hash"] = stable_hash(tampered)
    with pytest.raises(ResampleExecutionError, match="mismatches frozen seed"):
        generate_sentence_resampling_intermediates(
            (anchor,),
            base_traces={base.base_trace_id: base},
            allocation_manifest=manifest,
            backend=DeterministicBackend(),
            primary_inference=True,
            resume_records=(tampered,),
        )


def test_retain_consumes_exact_original_frozen_anchor_tokens_without_retokenizing() -> None:
    exact = (9001, 9002, 9003, 9004)
    anchor, base = _anchor_and_base(exact_anchor_tokens=exact)
    manifest = build_initial_allocation_manifest((anchor,), master_seed=66)
    backend = DeterministicBackend()
    rows = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        primary_inference=True,
    )

    retains = [row for row in rows if row.arm == "retain"]
    assert all(row.anchor_token_ids == exact for row in rows)
    assert all(row.conditioning_prefix_token_ids[-len(exact) :] == exact for row in retains)
    assert all(row.anchor_token_source == "frozen_original_completion_span" for row in rows)
    assert all(row.anchor_token_identity["used_without_retokenization"] is True for row in rows)
    assert anchor.sentence_text not in backend.continuation_calls


def test_primary_replacement_uses_pinned_qwen_leading_space_tokens_at_tolerance_boundary() -> None:
    anchor, base = _anchor_and_base(exact_anchor_tokens=(9001, 9002, 9003))
    manifest = build_initial_allocation_manifest((anchor,), master_seed=67)
    backend = PinnedQwenLeadingSpaceBackend()

    intermediates = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        primary_inference=True,
    )
    resamples = [row for row in intermediates if row.arm == "resample"]

    assert backend.continuation_calls == []
    assert {row.replacement_sentence for row in resamples} == {"Suppose something."}
    assert {row.replacement_token_ids for row in resamples} == {
        (
            backend.leading_suppose_token_id,
            backend.something_token_id,
            backend.period_token_id,
        )
    }
    assert {
        row.backend_result["replacement_token_span"]["leading_envelope_text"]
        for row in resamples
    } == {" "}
    assert all(
        row.backend_result["replacement_token_span"]["completion_token_ids_hash"]
        == token_stream_hash(row.generated_completion_token_ids or (), stream="completion")
        for row in resamples
    )

    tolerance = ReplacementTokenTolerance(0, 0.0)
    exact = _replacement_token_audit(
        anchor_token_count=3,
        replacement_token_count=3,
        tolerance=tolerance,
    )
    standalone_retokenized = _replacement_token_audit(
        anchor_token_count=3,
        replacement_token_count=4,
        tolerance=tolerance,
    )
    assert exact["within_absolute_tolerance"] is True
    assert exact["within_relative_tolerance"] is True
    assert standalone_retokenized["within_absolute_tolerance"] is False
    assert standalone_retokenized["within_relative_tolerance"] is False

    tampered = resamples[0].as_dict(include_hash=False)
    span = dict(tampered["backend_result"]["replacement_token_span"])
    span["completion_token_ids_hash"] = "sha256:" + "0" * 64
    tampered["backend_result"]["replacement_token_span"] = span
    tampered["backend_result"]["replacement_token_span_hash"] = stable_hash(span)
    tampered["record_hash"] = stable_hash(tampered)
    with pytest.raises(PrefixIdentityError, match="span audit does not reconstruct"):
        ResamplingGenerationRecord.from_dict(tampered)


@pytest.mark.parametrize(
    ("mode", "expected_valid_retain", "expected_valid_resample"),
    [
        ("missing_completion_ids", 0, 0),
        ("missing_tokenizer", 10, 0),
        ("non_round_trip", 10, 0),
    ],
)
def test_primary_generation_fails_rows_closed_without_exact_completion_mapping(
    mode: str,
    expected_valid_retain: int,
    expected_valid_resample: int,
) -> None:
    anchor, base = _anchor_and_base(exact_anchor_tokens=(1, 2, 3))
    manifest = build_initial_allocation_manifest((anchor,), master_seed=68)
    rows = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=IntegrityFailureBackend(mode),
        primary_inference=True,
    )

    assert sum(row.generation_status == "valid" for row in rows if row.arm == "retain") == (
        expected_valid_retain
    )
    assert sum(row.generation_status == "valid" for row in rows if row.arm == "resample") == (
        expected_valid_resample
    )
    assert all(
        row.invalid_reason and ("completion" in row.invalid_reason or "tokenizer" in row.invalid_reason)
        for row in rows
        if row.generation_status == GENERATION_STATUS_TERMINAL_INVALID
    )


def test_one_malformed_continuation_is_terminal_and_does_not_abort_or_call_paid_routes() -> None:
    anchor, base = _anchor_and_base()
    manifest = build_initial_allocation_manifest((anchor,), master_seed=77)
    malformed_id = manifest.allocations[0].request_id
    backend = DeterministicBackend(malformed_request_id=malformed_id)
    intermediates = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        primary_inference=True,
        microbatch_size=5,
    )
    malformed = next(row for row in intermediates if row.resample_id == malformed_id)
    assert malformed.generation_status == GENERATION_STATUS_TERMINAL_INVALID
    assert malformed.raw_generated_text == "Unclosed but preserved raw output."
    assert sum(row.generation_status == "valid" for row in intermediates) == 19

    classifier = CountingClassifier()
    caller = CountingOutcomeCaller()
    canonical = _cpu(
        intermediates,
        anchors=(anchor,),
        bases={base.base_trace_id: base},
        manifest=manifest,
        classifier=classifier,
        caller=caller,
    )
    invalid = next(row for row in canonical if row.resample_id == malformed_id)
    assert invalid.analysis_tier == "generation_invalid"
    assert invalid.intervention_eligible is False
    assert invalid.outcome_adjudication_primary_inference is False
    assert invalid.replacement_classification_status == "not_called_terminal_invalid"
    assert invalid.outcome_adjudication["judge_provenance"]["call_performed"] is False
    assert invalid.provenance["semantic_embedder"]["model_revision"] == "fixed-test-revision"
    assert invalid.raw_generated_text == malformed.raw_generated_text
    assert len(caller.requests) == 19
    assert len(classifier.requests) == 10


def test_split_and_compatibility_wrapper_are_canonically_equivalent_on_valid_rows() -> None:
    anchor, base = _anchor_and_base()
    manifest = build_initial_allocation_manifest((anchor,), master_seed=88)
    split_intermediates = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=DeterministicBackend(),
        primary_inference=True,
    )
    split = _cpu(
        split_intermediates,
        anchors=(anchor,),
        bases={base.base_trace_id: base},
        manifest=manifest,
    )
    combined = run_sentence_resampling(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=DeterministicBackend(),
        embedder=DivergentEmbedder(),
        outcome_caller=CountingOutcomeCaller(),
        primary_inference=True,
        replacement_classifier=CountingClassifier(),
        neutral_control=_neutral(),
        token_tolerance=ReplacementTokenTolerance(0, 0.0),
    )

    assert [row.as_dict(include_hash=True) for row in split] == [
        row.as_dict(include_hash=True) for row in combined
    ]


@pytest.mark.parametrize(
    ("cosine", "replacement_token_tolerance", "status"),
    [
        (0.9, ReplacementTokenTolerance(0, 0.0), "skipped_local_cosine_ineligible"),
        (0.0, ReplacementTokenTolerance(1, 0.1), "skipped_local_token_ineligible"),
    ],
)
def test_local_ineligibility_skips_paid_replacement_classification(
    cosine: float,
    replacement_token_tolerance: ReplacementTokenTolerance,
    status: str,
) -> None:
    anchor, base = _anchor_and_base(exact_anchor_tokens=(1, 2, 3))
    manifest = build_initial_allocation_manifest((anchor,), master_seed=99)
    backend = DeterministicBackend()
    intermediates = generate_sentence_resampling_intermediates(
        (anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        backend=backend,
        primary_inference=True,
    )
    classifier = CountingClassifier()
    canonical = adjudicate_sentence_resampling_intermediates(
        intermediates,
        anchors=(anchor,),
        base_traces={base.base_trace_id: base},
        allocation_manifest=manifest,
        embedder=DivergentEmbedder(cosine=cosine),
        outcome_caller=CountingOutcomeCaller(),
        primary_inference=True,
        replacement_classifier=classifier,
        neutral_control=_neutral(),
        token_tolerance=replacement_token_tolerance,
    )

    assert len(classifier.requests) == 0
    assert {
        row.replacement_classification_status for row in canonical if row.arm == "resample"
    } == {status}
