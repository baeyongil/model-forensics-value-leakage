"""Two-phase, provider-neutral execution for sentence resampling.

The expensive local generation phase is intentionally unable to see an
embedder or an external judge.  It emits content-addressed intermediate rows
that can be copied off the GPU host and authenticated before the CPU/API phase
performs semantic filtering, replacement classification, and final-outcome
adjudication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_forensics.adjudication import FINAL_ANSWER_INSTRUMENT
from model_forensics.anchors import AnchorManifest, FrozenAnchor
from model_forensics.io import stable_hash
from model_forensics.prompts import is_good_outcome
from model_forensics.resampling import (
    TextEmbedder,
    assess_semantic_divergence,
    build_token_identical_prefixes,
    first_generated_replacement_sentence,
)
from model_forensics.schemas import RolloutRecord
from model_forensics.token_spans import token_stream_hash

from .resample_runner import (
    FINAL_OUTCOME_PROTOCOL,
    BaseTrace,
    FinalOutcomeAdjudicationAudit,
    NeutralControlSpec,
    OutcomeAdjudicationError,
    PrefixIdentityError,
    RawPrefixGenerationBackend,
    RawPrefixGenerationRequest,
    RawPrefixGenerationResult,
    ReplacementClassificationError,
    ReplacementClassifier,
    ReplacementTokenTolerance,
    ResampleAllocation,
    ResampleAllocationManifest,
    ResampleExecutionError,
    ResamplingArtifactRecord,
    _anchors,
    _coerce_base_trace,
    _neutral_task_question,
    _replacement_classification_audit,
    _replacement_token_audit,
    _split_generated_continuation,
    _token_hash,
    _validate_execution_manifest,
    adjudicate_final_resampling_outcome,
)

INTERMEDIATE_SCHEMA_VERSION = 1
GENERATION_STATUS_VALID = "valid"
GENERATION_STATUS_TERMINAL_INVALID = "terminal_invalid"

IntermediateCallback = Callable[["ResamplingGenerationRecord"], None]
MicrobatchCallback = Callable[[int, tuple["ResamplingGenerationRecord", ...]], None]


def _token_ids(value: Sequence[int], *, name: str, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise PrefixIdentityError(f"{name} must be a token-ID sequence")
    result = tuple(value)
    if not result and not allow_empty:
        raise PrefixIdentityError(f"{name} must not be empty")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in result):
        raise PrefixIdentityError(f"{name} must contain nonnegative integer token IDs")
    return result


@dataclass(frozen=True, slots=True)
class ResamplingGenerationRecord:
    """Authenticated output of exactly one frozen GPU allocation.

    No field in this schema is produced by an embedding model or external API.
    ``record_hash`` is derived on serialization rather than stored, preventing a
    stale content address after construction.
    """

    resample_id: str
    anchor_id: str
    base_trace_id: str
    sentence_class: str
    condition: str
    arm: str
    sample_index: int
    seed: int
    stage: str
    threshold: float
    allocation_manifest_hash: str
    allocation_hash: str
    common_prefix_text_hash: str
    common_prefix_token_ids: tuple[int, ...]
    common_prefix_hash: str
    conditioning_prefix_text: str
    conditioning_prefix_token_ids: tuple[int, ...]
    conditioning_prefix_hash: str
    consumed_prompt_token_ids: tuple[int, ...]
    consumed_prompt_token_ids_hash: str
    prompt_identity_verified: bool
    anchor_token_ids: tuple[int, ...]
    anchor_token_ids_hash: str
    anchor_token_source: str
    anchor_token_identity: Mapping[str, Any]
    replacement_sentence: str
    replacement_char_start: int
    replacement_char_end: int
    replacement_token_ids: tuple[int, ...]
    replacement_token_ids_hash: str | None
    full_trace: str
    answer: str
    raw_generated_text: str
    generated_completion_token_ids: tuple[int, ...] | None
    generated_completion_token_ids_hash: str | None
    generation_status: str
    invalid_reason: str | None
    finish_reason: str
    usage: Mapping[str, Any]
    backend_provenance: Mapping[str, Any]
    backend_provenance_hash: str
    backend_result: Mapping[str, Any]
    base_trace_provenance: Mapping[str, Any]
    schema_version: int = INTERMEDIATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTERMEDIATE_SCHEMA_VERSION:
            raise ResampleExecutionError("unsupported resampling-intermediate schema")
        if not self.resample_id or not self.anchor_id or not self.base_trace_id:
            raise ResampleExecutionError("intermediate identifiers cannot be blank")
        if self.arm not in {"retain", "resample"}:
            raise ResampleExecutionError("intermediate has an unknown arm")
        if self.generation_status not in {
            GENERATION_STATUS_VALID,
            GENERATION_STATUS_TERMINAL_INVALID,
        }:
            raise ResampleExecutionError("intermediate has an unknown generation status")
        if (self.generation_status == GENERATION_STATUS_VALID) != (self.invalid_reason is None):
            raise ResampleExecutionError("generation status and invalid reason disagree")
        if self.prompt_identity_verified is not True:
            raise PrefixIdentityError("only prompt-identity-verified rows may be materialized")
        for value, name, allow_empty in (
            (self.common_prefix_token_ids, "common prefix", False),
            (self.conditioning_prefix_token_ids, "conditioning prefix", False),
            (self.consumed_prompt_token_ids, "consumed prompt", False),
            (self.anchor_token_ids, "anchor", False),
            (
                self.replacement_token_ids,
                "replacement",
                self.generation_status == GENERATION_STATUS_TERMINAL_INVALID,
            ),
        ):
            _token_ids(value, name=name, allow_empty=allow_empty)
        if self.consumed_prompt_token_ids != self.conditioning_prefix_token_ids:
            raise PrefixIdentityError("consumed prompt IDs differ from the frozen request")
        if self.common_prefix_hash != _token_hash(self.common_prefix_token_ids):
            raise PrefixIdentityError("common-prefix token hash mismatch")
        if self.conditioning_prefix_hash != _token_hash(self.conditioning_prefix_token_ids):
            raise PrefixIdentityError("conditioning-prefix token hash mismatch")
        if self.consumed_prompt_token_ids_hash != _token_hash(self.consumed_prompt_token_ids):
            raise PrefixIdentityError("consumed-prompt token hash mismatch")
        if self.anchor_token_ids_hash != _token_hash(self.anchor_token_ids):
            raise PrefixIdentityError("anchor token hash mismatch")
        expected_replacement_hash = (
            _token_hash(self.replacement_token_ids) if self.replacement_token_ids else None
        )
        if self.replacement_token_ids_hash != expected_replacement_hash:
            raise PrefixIdentityError("replacement token hash mismatch")
        if self.generated_completion_token_ids is None:
            if self.generated_completion_token_ids_hash is not None:
                raise PrefixIdentityError("completion hash exists without completion token IDs")
        elif self.generated_completion_token_ids_hash != _token_hash(
            _token_ids(
                self.generated_completion_token_ids,
                name="generated completion",
                allow_empty=True,
            )
        ):
            raise PrefixIdentityError("generated-completion token hash mismatch")
        if self.backend_provenance_hash != stable_hash(dict(self.backend_provenance)):
            raise ResampleExecutionError("backend provenance hash mismatch")
        if not self.backend_provenance:
            raise ResampleExecutionError("backend provenance cannot be empty")
        if self.full_trace and not self.full_trace.startswith(self.conditioning_prefix_text):
            raise PrefixIdentityError("full trace lost its conditioning text")
        if self.replacement_sentence:
            if (
                not 0
                <= self.replacement_char_start
                < self.replacement_char_end
                <= len(self.full_trace)
            ):
                raise ResampleExecutionError("replacement character span is invalid")
            if (
                self.full_trace[self.replacement_char_start : self.replacement_char_end]
                != self.replacement_sentence
            ):
                raise ResampleExecutionError("replacement span does not reconstruct its text")
        elif self.replacement_char_start != self.replacement_char_end:
            raise ResampleExecutionError("empty replacement must have an empty span")

        object.__setattr__(self, "common_prefix_token_ids", tuple(self.common_prefix_token_ids))
        object.__setattr__(
            self, "conditioning_prefix_token_ids", tuple(self.conditioning_prefix_token_ids)
        )
        object.__setattr__(self, "consumed_prompt_token_ids", tuple(self.consumed_prompt_token_ids))
        object.__setattr__(self, "anchor_token_ids", tuple(self.anchor_token_ids))
        object.__setattr__(self, "replacement_token_ids", tuple(self.replacement_token_ids))
        if self.generated_completion_token_ids is not None:
            object.__setattr__(
                self,
                "generated_completion_token_ids",
                tuple(self.generated_completion_token_ids),
            )
        for name in (
            "anchor_token_identity",
            "usage",
            "backend_provenance",
            "backend_result",
            "base_trace_provenance",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "resample_id": self.resample_id,
            "anchor_id": self.anchor_id,
            "base_trace_id": self.base_trace_id,
            "sentence_class": self.sentence_class,
            "condition": self.condition,
            "arm": self.arm,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "stage": self.stage,
            "threshold": self.threshold,
            "allocation_manifest_hash": self.allocation_manifest_hash,
            "allocation_hash": self.allocation_hash,
            "common_prefix_text_hash": self.common_prefix_text_hash,
            "common_prefix_token_ids": list(self.common_prefix_token_ids),
            "common_prefix_hash": self.common_prefix_hash,
            "conditioning_prefix_text": self.conditioning_prefix_text,
            "conditioning_prefix_token_ids": list(self.conditioning_prefix_token_ids),
            "conditioning_prefix_hash": self.conditioning_prefix_hash,
            "consumed_prompt_token_ids": list(self.consumed_prompt_token_ids),
            "consumed_prompt_token_ids_hash": self.consumed_prompt_token_ids_hash,
            "prompt_identity_verified": self.prompt_identity_verified,
            "anchor_token_ids": list(self.anchor_token_ids),
            "anchor_token_ids_hash": self.anchor_token_ids_hash,
            "anchor_token_source": self.anchor_token_source,
            "anchor_token_identity": dict(self.anchor_token_identity),
            "replacement_sentence": self.replacement_sentence,
            "replacement_char_start": self.replacement_char_start,
            "replacement_char_end": self.replacement_char_end,
            "replacement_token_ids": list(self.replacement_token_ids),
            "replacement_token_ids_hash": self.replacement_token_ids_hash,
            "full_trace": self.full_trace,
            "answer": self.answer,
            "raw_generated_text": self.raw_generated_text,
            "generated_completion_token_ids": (
                None
                if self.generated_completion_token_ids is None
                else list(self.generated_completion_token_ids)
            ),
            "generated_completion_token_ids_hash": self.generated_completion_token_ids_hash,
            "generation_status": self.generation_status,
            "invalid_reason": self.invalid_reason,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "backend_provenance": dict(self.backend_provenance),
            "backend_provenance_hash": self.backend_provenance_hash,
            "backend_result": dict(self.backend_result),
            "base_trace_provenance": dict(self.base_trace_provenance),
        }
        if include_hash:
            payload["record_hash"] = stable_hash(payload)
        return payload

    @property
    def record_hash(self) -> str:
        return str(self.as_dict(include_hash=True)["record_hash"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResamplingGenerationRecord:
        payload = dict(value)
        recorded_hash = payload.pop("record_hash", None)
        if recorded_hash != stable_hash(payload):
            raise ResampleExecutionError("resampling-intermediate record hash mismatch")
        for name in (
            "common_prefix_token_ids",
            "conditioning_prefix_token_ids",
            "consumed_prompt_token_ids",
            "anchor_token_ids",
            "replacement_token_ids",
        ):
            payload[name] = tuple(payload[name])
        completion = payload.get("generated_completion_token_ids")
        payload["generated_completion_token_ids"] = (
            None if completion is None else tuple(completion)
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class _GenerationContext:
    allocation: ResampleAllocation
    anchor: FrozenAnchor
    base: BaseTrace
    request: RawPrefixGenerationRequest
    common_text: str
    common_tokens: tuple[int, ...]
    common_prefix_hash: str
    anchor_tokens: tuple[int, ...]
    anchor_token_source: str
    anchor_token_identity: Mapping[str, Any]
    needs_contextual_replacement_encoding: bool


def _frozen_anchor_tokens(
    anchor: FrozenAnchor,
) -> tuple[tuple[int, ...], str, dict[str, Any]] | None:
    token_span = anchor.provenance.get("token_span")
    if token_span is None:
        return None
    if not isinstance(token_span, Mapping):
        raise PrefixIdentityError("frozen anchor token_span provenance is not a mapping")
    ids_value = token_span.get("token_ids")
    if not isinstance(ids_value, Sequence) or isinstance(ids_value, (str, bytes)):
        raise PrefixIdentityError("frozen anchor token span omits exact token IDs")
    token_ids = _token_ids(ids_value, name="frozen anchor")
    expected_span_hash = token_stream_hash(token_ids, stream="completion_span")
    if (
        token_span.get("text") != anchor.sentence_text
        or token_span.get("section_char_start") != anchor.char_start
        or token_span.get("section_char_end") != anchor.char_end
        or token_span.get("leading_envelope_text") != ""
        or token_span.get("round_trip_verified") is not True
        or token_span.get("token_ids_hash") != expected_span_hash
    ):
        raise PrefixIdentityError("frozen anchor exact-token evidence does not verify")
    completion_hash = token_span.get("completion_token_ids_hash")
    if anchor.provenance.get("completion_token_ids_hash") != completion_hash:
        raise PrefixIdentityError("anchor and token-span completion hashes disagree")
    trailing = token_span.get("trailing_envelope_text", "")
    if not isinstance(trailing, str):
        raise PrefixIdentityError("frozen anchor trailing token envelope is not text")
    identity = {
        "method": "frozen_original_completion_span",
        "round_trip_verified": True,
        "token_span_hash": stable_hash(dict(token_span)),
        "token_ids_hash": expected_span_hash,
        "completion_token_ids_hash": completion_hash,
        "token_start": token_span.get("token_start"),
        "token_end": token_span.get("token_end"),
        "leading_envelope_text": "",
        "trailing_envelope_text": trailing,
        "used_without_retokenization": True,
    }
    return token_ids, trailing, identity


def _prepare_generation_contexts(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    base_traces: Mapping[str, BaseTrace | RolloutRecord | Mapping[str, Any]],
    allocation_manifest: ResampleAllocationManifest,
    backend: RawPrefixGenerationBackend,
    primary_inference: bool,
) -> tuple[tuple[_GenerationContext, ...], dict[str, Any]]:
    ordered_anchors = _anchors(anchors)
    _validate_execution_manifest(
        ordered_anchors,
        allocation_manifest,
        primary_inference=primary_inference,
    )
    expected_trace_ids = {anchor.trace_id for anchor in ordered_anchors}
    if not expected_trace_ids.issubset(base_traces):
        raise ResampleExecutionError(
            f"missing base traces: {sorted(expected_trace_ids.difference(base_traces))!r}"
        )
    backend_provenance = dict(backend.provenance)
    if not backend_provenance:
        raise ResampleExecutionError("backend provenance cannot be empty")
    if primary_inference and bool(backend_provenance.get("synthetic_smoke", False)):
        raise OutcomeAdjudicationError("primary resampling refuses a synthetic smoke backend")

    allocation_by_anchor: dict[str, list[ResampleAllocation]] = {}
    for allocation in allocation_manifest.allocations:
        allocation_by_anchor.setdefault(allocation.anchor_id, []).append(allocation)
    if set(allocation_by_anchor) != {anchor.anchor_id for anchor in ordered_anchors}:
        raise ResampleExecutionError("allocation anchors differ from the frozen anchor set")

    context_by_id: dict[str, _GenerationContext] = {}
    for anchor in ordered_anchors:
        base = _coerce_base_trace(base_traces[anchor.trace_id])
        if base.base_trace_id != anchor.trace_id:
            raise ResampleExecutionError("base trace ID does not match its frozen anchor")
        if base.condition != anchor.direction:
            raise ResampleExecutionError("base trace condition does not match anchor direction")
        if base.trace[anchor.char_start : anchor.char_end] != anchor.sentence_text:
            raise ResampleExecutionError("frozen anchor span no longer matches the base trace")
        _neutral_task_question(base.task)

        common_text = base.trace[: anchor.char_start]
        arm_prefixes = build_token_identical_prefixes(base.messages, common_text, backend)
        common_tokens = tuple(arm_prefixes.retain_token_ids)
        frozen = _frozen_anchor_tokens(anchor)
        if frozen is None:
            anchor_tokens = _token_ids(
                backend.encode_continuation(anchor.sentence_text),
                name="fallback anchor",
            )
            trailing_envelope = ""
            token_source = "backend_retokenized_fallback"
            token_identity = {
                "method": token_source,
                "round_trip_verified": False,
                "used_without_retokenization": False,
                "token_ids_hash": _token_hash(anchor_tokens),
            }
            contextual_replacement = False
        else:
            anchor_tokens, trailing_envelope, token_identity = frozen
            token_source = "frozen_original_completion_span"
            # The backend's optional contextual-append state was deliberately not
            # consumed by re-tokenizing the anchor.  Before measuring each new
            # replacement we therefore re-arm its exact common prefix.
            contextual_replacement = True

        for allocation in allocation_by_anchor[anchor.anchor_id]:
            if allocation.base_trace_id != anchor.trace_id:
                raise ResampleExecutionError("allocation base trace does not match anchor")
            if allocation.arm == "retain":
                conditioning_text = common_text + anchor.sentence_text + trailing_envelope
                conditioning_tokens = common_tokens + anchor_tokens
            else:
                conditioning_text = common_text
                conditioning_tokens = common_tokens
            if conditioning_tokens[: len(common_tokens)] != common_tokens:
                raise PrefixIdentityError("an arm broke the exact common token prefix")
            request = RawPrefixGenerationRequest(
                request_id=allocation.request_id,
                anchor_id=anchor.anchor_id,
                base_trace_id=anchor.trace_id,
                arm=allocation.arm,
                sample_index=allocation.sample_index,
                seed=allocation.seed,
                messages=base.messages,
                conditioning_text=conditioning_text,
                prompt_token_ids=conditioning_tokens,
                common_prefix_token_count=len(common_tokens),
            )
            if request.request_id in context_by_id:
                raise ResampleExecutionError("allocation contains duplicate request IDs")
            context_by_id[request.request_id] = _GenerationContext(
                allocation=allocation,
                anchor=anchor,
                base=base,
                request=request,
                common_text=common_text,
                common_tokens=common_tokens,
                common_prefix_hash=arm_prefixes.prefix_hash,
                anchor_tokens=anchor_tokens,
                anchor_token_source=token_source,
                anchor_token_identity=token_identity,
                needs_contextual_replacement_encoding=contextual_replacement,
            )

    ordered_contexts: list[_GenerationContext] = []
    for allocation in allocation_manifest.allocations:
        context = context_by_id.get(allocation.request_id)
        if context is None:
            raise ResampleExecutionError(
                f"allocation contains an unknown request: {allocation.request_id}"
            )
        ordered_contexts.append(context)
    # Some bounded smoke backends expose counters that change while prefixes are
    # encoded.  Freeze provenance only after request construction so the row
    # describes the state that actually produced the generation requests.
    final_backend_provenance = dict(backend.provenance)
    if not final_backend_provenance:
        raise ResampleExecutionError("backend provenance became empty during preparation")
    return tuple(ordered_contexts), final_backend_provenance


def _completion_tokens_from_result(result: RawPrefixGenerationResult) -> tuple[int, ...] | None:
    value = result.backend_metadata.get("completion_token_ids")
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PrefixIdentityError("backend completion token IDs are not a sequence")
    ids = _token_ids(value, name="generated completion", allow_empty=True)
    recorded_hash = result.backend_metadata.get("completion_token_ids_hash")
    if recorded_hash is not None and recorded_hash != token_stream_hash(ids, stream="completion"):
        raise PrefixIdentityError("backend completion-token manifest hash mismatch")
    return ids


def _terminal_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _intermediate_from_result(
    context: _GenerationContext,
    result: RawPrefixGenerationResult,
    *,
    allocation_manifest: ResampleAllocationManifest,
    backend: RawPrefixGenerationBackend,
    backend_provenance: Mapping[str, Any],
) -> ResamplingGenerationRecord:
    request = context.request
    observed_prompt = _token_ids(result.prompt_token_ids, name="consumed prompt")
    if result.request_id != request.request_id:
        raise ResampleExecutionError("backend result echoes the wrong request ID")
    if observed_prompt != request.prompt_token_ids:
        raise PrefixIdentityError(
            f"backend used different prompt tokens for request {request.request_id}"
        )
    generated_completion_ids = _completion_tokens_from_result(result)

    status = GENERATION_STATUS_VALID
    invalid_reason: str | None = None
    answer = ""
    reasoning_tail = result.generated_text
    replacement_sentence = ""
    replacement_start = len(request.conditioning_text)
    replacement_end = replacement_start
    replacement_tokens: tuple[int, ...] = ()
    try:
        reasoning_tail, answer = _split_generated_continuation(result.generated_text)
        if request.arm == "retain":
            replacement_sentence = context.anchor.sentence_text
            replacement_start = context.anchor.char_start
            replacement_end = context.anchor.char_end
            replacement_tokens = context.anchor_tokens
        else:
            replacement = first_generated_replacement_sentence(reasoning_tail)
            if replacement is None:
                raise ResampleExecutionError("resample emitted no replacement sentence")
            replacement_sentence = replacement.text
            replacement_start = len(request.conditioning_text) + replacement.start
            replacement_end = len(request.conditioning_text) + replacement.end
            if context.needs_contextual_replacement_encoding:
                checked_prefix = _token_ids(
                    backend.encode_prefix(context.base.messages, context.common_text),
                    name="replacement common prefix",
                )
                if checked_prefix != context.common_tokens:
                    raise PrefixIdentityError(
                        "replacement tokenization re-armed a different common prefix"
                    )
            replacement_tokens = _token_ids(
                backend.encode_continuation(replacement_sentence),
                name="replacement",
            )
    except ResampleExecutionError as exc:
        status = GENERATION_STATUS_TERMINAL_INVALID
        invalid_reason = _terminal_reason(exc)
        answer = ""
        replacement_sentence = ""
        replacement_start = len(request.conditioning_text)
        replacement_end = replacement_start
        replacement_tokens = ()

    full_trace = request.conditioning_text + reasoning_tail
    if replacement_sentence and (
        full_trace[replacement_start:replacement_end] != replacement_sentence
    ):
        raise ResampleExecutionError("replacement span does not reconstruct its sentence")
    allocation_hash = stable_hash(context.allocation.as_dict())
    backend_result = dict(result.backend_metadata)
    return ResamplingGenerationRecord(
        resample_id=request.request_id,
        anchor_id=context.anchor.anchor_id,
        base_trace_id=context.anchor.trace_id,
        sentence_class=context.anchor.sentence_class,
        condition=context.anchor.direction,
        arm=request.arm,
        sample_index=request.sample_index,
        seed=request.seed,
        stage=allocation_manifest.stage,
        threshold=context.base.threshold,
        allocation_manifest_hash=allocation_manifest.manifest_hash,
        allocation_hash=allocation_hash,
        common_prefix_text_hash=stable_hash(context.common_text),
        common_prefix_token_ids=context.common_tokens,
        common_prefix_hash=context.common_prefix_hash,
        conditioning_prefix_text=request.conditioning_text,
        conditioning_prefix_token_ids=request.prompt_token_ids,
        conditioning_prefix_hash=_token_hash(request.prompt_token_ids),
        consumed_prompt_token_ids=observed_prompt,
        consumed_prompt_token_ids_hash=_token_hash(observed_prompt),
        prompt_identity_verified=True,
        anchor_token_ids=context.anchor_tokens,
        anchor_token_ids_hash=_token_hash(context.anchor_tokens),
        anchor_token_source=context.anchor_token_source,
        anchor_token_identity=context.anchor_token_identity,
        replacement_sentence=replacement_sentence,
        replacement_char_start=replacement_start,
        replacement_char_end=replacement_end,
        replacement_token_ids=replacement_tokens,
        replacement_token_ids_hash=(
            _token_hash(replacement_tokens) if replacement_tokens else None
        ),
        full_trace=full_trace,
        answer=answer,
        raw_generated_text=result.generated_text,
        generated_completion_token_ids=generated_completion_ids,
        generated_completion_token_ids_hash=(
            _token_hash(generated_completion_ids) if generated_completion_ids is not None else None
        ),
        generation_status=status,
        invalid_reason=invalid_reason,
        finish_reason=result.finish_reason,
        usage={
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
        },
        backend_provenance=backend_provenance,
        backend_provenance_hash=stable_hash(dict(backend_provenance)),
        backend_result=backend_result,
        base_trace_provenance=context.base.provenance,
    )


def _coerce_intermediate(
    value: ResamplingGenerationRecord | Mapping[str, Any],
) -> ResamplingGenerationRecord:
    if isinstance(value, ResamplingGenerationRecord):
        # Round-trip through its content address even for in-memory resume rows.
        return ResamplingGenerationRecord.from_dict(value.as_dict(include_hash=True))
    if not isinstance(value, Mapping):
        raise TypeError("intermediate rows must be records or mappings")
    return ResamplingGenerationRecord.from_dict(value)


def _validate_resumed_record(
    record: ResamplingGenerationRecord,
    context: _GenerationContext,
    *,
    manifest: ResampleAllocationManifest,
    backend_provenance_hash: str,
) -> None:
    allocation = context.allocation
    expected = {
        "resample_id": allocation.request_id,
        "anchor_id": allocation.anchor_id,
        "base_trace_id": allocation.base_trace_id,
        "arm": allocation.arm,
        "sample_index": allocation.sample_index,
        "seed": allocation.seed,
        "stage": allocation.stage,
        "allocation_manifest_hash": manifest.manifest_hash,
        "allocation_hash": stable_hash(allocation.as_dict()),
        "backend_provenance_hash": backend_provenance_hash,
        "common_prefix_text_hash": stable_hash(context.common_text),
        "common_prefix_hash": context.common_prefix_hash,
        "conditioning_prefix_hash": _token_hash(context.request.prompt_token_ids),
        "anchor_token_ids_hash": _token_hash(context.anchor_tokens),
    }
    for name, expected_value in expected.items():
        if getattr(record, name) != expected_value:
            raise ResampleExecutionError(
                f"resumed intermediate {record.resample_id} mismatches frozen {name}"
            )
    if record.conditioning_prefix_token_ids != context.request.prompt_token_ids:
        raise PrefixIdentityError("resumed intermediate has different conditioning token IDs")
    if record.anchor_token_ids != context.anchor_tokens:
        raise PrefixIdentityError("resumed intermediate has different anchor token IDs")
    if record.conditioning_prefix_text != context.request.conditioning_text:
        raise PrefixIdentityError("resumed intermediate has different conditioning text")


def generate_sentence_resampling_intermediates(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    base_traces: Mapping[str, BaseTrace | RolloutRecord | Mapping[str, Any]],
    allocation_manifest: ResampleAllocationManifest,
    backend: RawPrefixGenerationBackend,
    primary_inference: bool,
    microbatch_size: int | None = None,
    resume_records: Sequence[ResamplingGenerationRecord | Mapping[str, Any]] = (),
    on_intermediate: IntermediateCallback | None = None,
    on_microbatch: MicrobatchCallback | None = None,
) -> tuple[ResamplingGenerationRecord, ...]:
    """Run only local generation in frozen order, with exact-ID resume.

    The signature deliberately contains no embedder, classifier, or outcome
    caller.  A callback is invoked once per newly completed unit and once after
    each non-empty frozen microbatch, allowing the caller to atomically persist
    rows before releasing GPU resources.
    """

    if type(primary_inference) is not bool:
        raise TypeError("primary_inference must be an explicit bool")
    contexts, backend_provenance = _prepare_generation_contexts(
        anchors,
        base_traces=base_traces,
        allocation_manifest=allocation_manifest,
        backend=backend,
        primary_inference=primary_inference,
    )
    if microbatch_size is None:
        microbatch_size = len(contexts)
    if isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int):
        raise TypeError("microbatch_size must be an integer")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    context_by_id = {context.allocation.request_id: context for context in contexts}
    backend_hash = stable_hash(dict(backend_provenance))
    completed: dict[str, ResamplingGenerationRecord] = {}
    for value in resume_records:
        record = _coerce_intermediate(value)
        if record.resample_id in completed:
            raise ResampleExecutionError(f"duplicate resumed intermediate ID: {record.resample_id}")
        context = context_by_id.get(record.resample_id)
        if context is None:
            raise ResampleExecutionError(
                f"resumed intermediate is outside the frozen allocation: {record.resample_id}"
            )
        _validate_resumed_record(
            record,
            context,
            manifest=allocation_manifest,
            backend_provenance_hash=backend_hash,
        )
        completed[record.resample_id] = record

    for batch_index, start in enumerate(range(0, len(contexts), microbatch_size)):
        frozen_batch = contexts[start : start + microbatch_size]
        pending = tuple(
            context for context in frozen_batch if context.allocation.request_id not in completed
        )
        if not pending:
            continue
        generated = tuple(backend.generate(tuple(context.request for context in pending)))
        by_id: dict[str, RawPrefixGenerationResult] = {}
        for result in generated:
            if result.request_id in by_id:
                raise ResampleExecutionError("backend returned duplicate request IDs")
            by_id[result.request_id] = result
        expected_ids = {context.allocation.request_id for context in pending}
        if set(by_id) != expected_ids:
            raise ResampleExecutionError("backend request/result IDs do not match")

        batch_records: list[ResamplingGenerationRecord] = []
        for context in pending:
            record = _intermediate_from_result(
                context,
                by_id[context.allocation.request_id],
                allocation_manifest=allocation_manifest,
                backend=backend,
                backend_provenance=backend_provenance,
            )
            completed[record.resample_id] = record
            batch_records.append(record)
            if on_intermediate is not None:
                on_intermediate(record)
        if on_microbatch is not None:
            on_microbatch(batch_index, tuple(batch_records))

    expected_ids = [context.allocation.request_id for context in contexts]
    if set(completed) != set(expected_ids):  # pragma: no cover - guarded per batch
        raise ResampleExecutionError("generation did not materialize every frozen allocation")
    return tuple(completed[request_id] for request_id in expected_ids)


def _authenticate_intermediate_for_cpu(
    record: ResamplingGenerationRecord,
    *,
    allocation: ResampleAllocation,
    anchor: FrozenAnchor,
    base: BaseTrace,
    manifest: ResampleAllocationManifest,
) -> None:
    expected = {
        "resample_id": allocation.request_id,
        "anchor_id": allocation.anchor_id,
        "base_trace_id": allocation.base_trace_id,
        "sentence_class": anchor.sentence_class,
        "condition": anchor.direction,
        "arm": allocation.arm,
        "sample_index": allocation.sample_index,
        "seed": allocation.seed,
        "stage": allocation.stage,
        "threshold": base.threshold,
        "allocation_manifest_hash": manifest.manifest_hash,
        "allocation_hash": stable_hash(allocation.as_dict()),
        "common_prefix_text_hash": stable_hash(base.trace[: anchor.char_start]),
    }
    for name, expected_value in expected.items():
        if getattr(record, name) != expected_value:
            raise ResampleExecutionError(
                f"intermediate {record.resample_id} mismatches frozen {name}"
            )
    if record.conditioning_prefix_text != (
        base.trace[: anchor.char_start]
        if allocation.arm == "resample"
        else base.trace[: anchor.char_start]
        + anchor.sentence_text
        + str(record.anchor_token_identity.get("trailing_envelope_text", ""))
    ):
        raise PrefixIdentityError("intermediate conditioning text does not match the frozen arm")
    frozen = _frozen_anchor_tokens(anchor)
    if frozen is not None:
        frozen_ids, _, identity = frozen
        if (
            record.anchor_token_ids != frozen_ids
            or record.anchor_token_source != "frozen_original_completion_span"
            or dict(record.anchor_token_identity) != identity
        ):
            raise PrefixIdentityError("intermediate did not consume the frozen anchor token span")


def _unmeasured_outcome_audit(
    *,
    task: str,
    trace: str,
    answer: str,
    reason: str,
) -> FinalOutcomeAdjudicationAudit:
    question = _neutral_task_question(task)
    raw_response = ""
    return FinalOutcomeAdjudicationAudit(
        case_hash=stable_hash({"task_question": question, "trace": trace, "answer": answer}),
        task_question_hash=stable_hash(question),
        trace_hash=stable_hash(trace),
        answer_hash=stable_hash(answer),
        request_id=stable_hash(
            {"protocol": FINAL_OUTCOME_PROTOCOL, "terminal_invalid_reason": reason}
        ),
        instrument_hash=FINAL_ANSWER_INSTRUMENT.instrument_hash,
        response_hash=stable_hash({"raw_response": raw_response}),
        raw_response=raw_response,
        status="UNKNOWN",
        value=None,
        primary_inference=False,
        not_for_primary_inference=True,
        judge_provenance={
            "provider": "not_called",
            "model_id": "terminal-invalid-generation",
            "call_performed": False,
            "reason": reason,
        },
    )


def adjudicate_sentence_resampling_intermediates(
    intermediates: Sequence[ResamplingGenerationRecord | Mapping[str, Any]],
    *,
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    base_traces: Mapping[str, BaseTrace | RolloutRecord | Mapping[str, Any]],
    allocation_manifest: ResampleAllocationManifest,
    embedder: TextEmbedder,
    outcome_caller: Any,
    primary_inference: bool,
    replacement_classifier: ReplacementClassifier | None = None,
    neutral_control: NeutralControlSpec | None = None,
    token_tolerance: ReplacementTokenTolerance | None = None,
    on_record: Callable[[ResamplingArtifactRecord], None] | None = None,
) -> tuple[ResamplingArtifactRecord, ...]:
    """Authenticate GPU rows, then run only CPU/paid confirmatory work."""

    if type(primary_inference) is not bool:
        raise TypeError("primary_inference must be an explicit bool")
    if primary_inference and outcome_caller.not_for_primary_inference:
        raise OutcomeAdjudicationError(
            "primary resampling refuses a caller marked not_for_primary_inference"
        )
    if (
        primary_inference
        and replacement_classifier is not None
        and bool(replacement_classifier.provenance.get("synthetic_smoke", False))
    ):
        raise ReplacementClassificationError(
            "primary resampling refuses a synthetic replacement classifier"
        )
    ordered_anchors = _anchors(anchors)
    _validate_execution_manifest(
        ordered_anchors,
        allocation_manifest,
        primary_inference=primary_inference,
    )
    anchor_by_id = {anchor.anchor_id: anchor for anchor in ordered_anchors}
    base_by_id = {trace_id: _coerce_base_trace(value) for trace_id, value in base_traces.items()}
    allocation_by_id = {
        allocation.request_id: allocation for allocation in allocation_manifest.allocations
    }
    if len(allocation_by_id) != len(allocation_manifest.allocations):
        raise ResampleExecutionError("allocation contains duplicate request IDs")

    intermediate_by_id: dict[str, ResamplingGenerationRecord] = {}
    for value in intermediates:
        record = _coerce_intermediate(value)
        if record.resample_id in intermediate_by_id:
            raise ResampleExecutionError(f"duplicate intermediate ID: {record.resample_id}")
        allocation = allocation_by_id.get(record.resample_id)
        if allocation is None:
            raise ResampleExecutionError(
                f"intermediate is outside the frozen allocation: {record.resample_id}"
            )
        anchor = anchor_by_id[allocation.anchor_id]
        base = base_by_id.get(anchor.trace_id)
        if base is None:
            raise ResampleExecutionError(f"missing base trace: {anchor.trace_id}")
        _authenticate_intermediate_for_cpu(
            record,
            allocation=allocation,
            anchor=anchor,
            base=base,
            manifest=allocation_manifest,
        )
        intermediate_by_id[record.resample_id] = record
    if set(intermediate_by_id) != set(allocation_by_id):
        missing = sorted(set(allocation_by_id).difference(intermediate_by_id))
        raise ResampleExecutionError(f"missing frozen intermediate rows: {missing!r}")

    records: list[ResamplingArtifactRecord] = []
    for allocation in allocation_manifest.allocations:
        intermediate = intermediate_by_id[allocation.request_id]
        anchor = anchor_by_id[allocation.anchor_id]
        base = base_by_id[allocation.base_trace_id]
        synthetic_smoke = bool(intermediate.backend_provenance.get("synthetic_smoke", False))

        if intermediate.generation_status == GENERATION_STATUS_TERMINAL_INVALID:
            cosine_similarity = 1.0 if allocation.arm == "retain" else 0.0
            divergent = False
            token_audit = {
                "anchor_token_count": len(intermediate.anchor_token_ids),
                "replacement_token_count": 0,
                "absolute_difference": len(intermediate.anchor_token_ids),
                "relative_difference": 1.0,
                "absolute_tolerance": (
                    token_tolerance.max_absolute_difference if token_tolerance else None
                ),
                "relative_tolerance": (
                    token_tolerance.max_relative_difference if token_tolerance else None
                ),
                "within_absolute_tolerance": False,
                "within_relative_tolerance": False,
            }
            classification_audit = {
                "status": "not_called_terminal_invalid",
                "adjudication_valid": False,
                "target_feature_absent_or_changed": None,
                "neutral_control_function_matched": None,
                "classifier_input_blinded": None,
                "classifier_input_hash": None,
                "classifier_request_hash": None,
                "raw_judgment_hashes": (),
                "classifier_provenance_hash": None,
                "classifier_provenance": {},
                "neutral_control_hash": neutral_control.control_hash if neutral_control else None,
                "rationale": intermediate.invalid_reason,
            }
            outcome_audit = _unmeasured_outcome_audit(
                task=base.task,
                trace=intermediate.full_trace,
                answer=intermediate.answer,
                reason=intermediate.invalid_reason or "terminal invalid generation",
            )
            replacement_eligible = False
            analysis_tier = "generation_invalid"
        else:
            if allocation.arm == "retain":
                cosine_similarity = 1.0
                divergent = False
            else:
                divergence = assess_semantic_divergence(
                    anchor.sentence_text,
                    intermediate.replacement_sentence,
                    embedder,
                )
                cosine_similarity = divergence.cosine_similarity
                divergent = divergence.divergent
            token_audit = _replacement_token_audit(
                anchor_token_count=len(intermediate.anchor_token_ids),
                replacement_token_count=len(intermediate.replacement_token_ids),
                tolerance=token_tolerance,
            )
            if allocation.arm == "retain":
                classification_audit = {
                    "status": "paired_reference",
                    "adjudication_valid": True,
                    "target_feature_absent_or_changed": None,
                    "neutral_control_function_matched": None,
                    "classifier_input_blinded": None,
                    "classifier_input_hash": None,
                    "classifier_request_hash": None,
                    "raw_judgment_hashes": (),
                    "classifier_provenance_hash": None,
                    "classifier_provenance": {},
                    "neutral_control_hash": (
                        neutral_control.control_hash if neutral_control else None
                    ),
                    "rationale": (
                        "Retain is the frozen paired reference; no replacement was judged."
                    ),
                }
                replacement_eligible = True
            elif not divergent:
                classification_audit = {
                    "status": "skipped_local_cosine_ineligible",
                    "adjudication_valid": False,
                    "target_feature_absent_or_changed": None,
                    "neutral_control_function_matched": None,
                    "classifier_input_blinded": None,
                    "classifier_input_hash": None,
                    "classifier_request_hash": None,
                    "raw_judgment_hashes": (),
                    "classifier_provenance_hash": None,
                    "classifier_provenance": {},
                    "neutral_control_hash": (
                        neutral_control.control_hash if neutral_control else None
                    ),
                    "rationale": "Paid classification skipped: cosine similarity was >= 0.8.",
                }
                replacement_eligible = False
            elif (
                token_audit["within_absolute_tolerance"] is not True
                or token_audit["within_relative_tolerance"] is not True
            ):
                classification_audit = {
                    "status": "skipped_local_token_ineligible",
                    "adjudication_valid": False,
                    "target_feature_absent_or_changed": None,
                    "neutral_control_function_matched": None,
                    "classifier_input_blinded": None,
                    "classifier_input_hash": None,
                    "classifier_request_hash": None,
                    "raw_judgment_hashes": (),
                    "classifier_provenance_hash": None,
                    "classifier_provenance": {},
                    "neutral_control_hash": (
                        neutral_control.control_hash if neutral_control else None
                    ),
                    "rationale": "Paid classification skipped: token-length tolerance failed.",
                }
                replacement_eligible = False
            else:
                classification_audit = _replacement_classification_audit(
                    classifier=replacement_classifier,
                    neutral_control=neutral_control,
                    original_sentence=anchor.sentence_text,
                    replacement_sentence=intermediate.replacement_sentence,
                    target_sentence_class=anchor.sentence_class,
                    threshold=base.threshold,
                )
                replacement_eligible = bool(
                    classification_audit["adjudication_valid"]
                    and classification_audit["target_feature_absent_or_changed"] is True
                    and classification_audit["neutral_control_function_matched"] is True
                    and classification_audit["classifier_input_blinded"] is True
                )
            outcome_audit = adjudicate_final_resampling_outcome(
                task=base.task,
                trace=intermediate.full_trace,
                answer=intermediate.answer,
                caller=outcome_caller,
                primary_inference=primary_inference,
            )
            analysis_tier = ""

        final_value = outcome_audit.value
        final_measurement_valid = final_value is not None
        good_side = (
            is_good_outcome(anchor.direction, final_value, base.threshold)
            if final_value is not None
            else None
        )
        intervention_eligible = bool(
            primary_inference
            and replacement_eligible
            and intermediate.generation_status == GENERATION_STATUS_VALID
        )
        confirmatory_eligible = bool(intervention_eligible and final_measurement_valid)
        if intermediate.generation_status == GENERATION_STATUS_TERMINAL_INVALID:
            pass
        elif not primary_inference:
            analysis_tier = "nonprimary_smoke"
        elif not final_measurement_valid:
            analysis_tier = "outcome_unmeasured"
        elif allocation.arm == "retain":
            analysis_tier = "paired_reference"
        else:
            analysis_tier = "confirmatory" if confirmatory_eligible else "exploratory"

        provenance = {
            "backend": dict(intermediate.backend_provenance),
            "backend_result": dict(intermediate.backend_result),
            "base_trace": dict(intermediate.base_trace_provenance),
            "allocation_manifest_hash": allocation_manifest.manifest_hash,
            "generation_intermediate_hash": intermediate.record_hash,
            "generation_intermediate_schema_version": intermediate.schema_version,
            "anchor_token_identity": dict(intermediate.anchor_token_identity),
        }
        record = ResamplingArtifactRecord(
            resample_id=intermediate.resample_id,
            anchor_id=intermediate.anchor_id,
            base_trace_id=intermediate.base_trace_id,
            sentence_class=intermediate.sentence_class,
            condition=intermediate.condition,
            arm=intermediate.arm,
            sample_index=intermediate.sample_index,
            seed=intermediate.seed,
            stage=intermediate.stage,
            threshold=intermediate.threshold,
            common_prefix_hash=intermediate.common_prefix_hash,
            conditioning_prefix_hash=intermediate.conditioning_prefix_hash,
            common_prefix_token_count=len(intermediate.common_prefix_token_ids),
            conditioning_prefix_token_count=len(intermediate.conditioning_prefix_token_ids),
            conditioning_prefix_text=intermediate.conditioning_prefix_text,
            replacement_sentence=intermediate.replacement_sentence,
            replacement_char_start=intermediate.replacement_char_start,
            replacement_char_end=intermediate.replacement_char_end,
            cosine_similarity=cosine_similarity,
            divergent=divergent,
            intervention_eligible=intervention_eligible,
            primary_eligible=confirmatory_eligible,
            confirmatory_eligible=confirmatory_eligible,
            analysis_tier=analysis_tier,
            anchor_token_count=token_audit["anchor_token_count"],
            replacement_token_count=token_audit["replacement_token_count"],
            token_count_absolute_difference=token_audit["absolute_difference"],
            token_count_relative_difference=token_audit["relative_difference"],
            token_count_absolute_tolerance=token_audit["absolute_tolerance"],
            token_count_relative_tolerance=token_audit["relative_tolerance"],
            token_count_within_absolute_tolerance=token_audit["within_absolute_tolerance"],
            token_count_within_relative_tolerance=token_audit["within_relative_tolerance"],
            replacement_classification_status=classification_audit["status"],
            target_feature_absent_or_changed=classification_audit[
                "target_feature_absent_or_changed"
            ],
            neutral_control_function_matched=classification_audit[
                "neutral_control_function_matched"
            ],
            classifier_input_blinded=classification_audit["classifier_input_blinded"],
            classifier_input_hash=classification_audit["classifier_input_hash"],
            classifier_request_hash=classification_audit["classifier_request_hash"],
            classifier_judgment_hashes=tuple(classification_audit["raw_judgment_hashes"]),
            classifier_provenance_hash=classification_audit["classifier_provenance_hash"],
            classifier_provenance=classification_audit["classifier_provenance"],
            neutral_control_hash=classification_audit["neutral_control_hash"],
            classification_rationale=classification_audit["rationale"],
            full_trace=intermediate.full_trace,
            answer=intermediate.answer,
            raw_generated_text=intermediate.raw_generated_text,
            final_estimate=final_value,
            final_good_side=good_side,
            final_measurement_valid=final_measurement_valid,
            outcome_adjudication_primary_inference=outcome_audit.primary_inference,
            outcome_adjudication=outcome_audit.as_dict(include_hash=True),
            trajectory=None,
            usage=intermediate.usage,
            provenance=provenance,
            finish_reason=intermediate.finish_reason,
            synthetic_smoke=synthetic_smoke,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)
    return tuple(records)


__all__ = [
    "GENERATION_STATUS_TERMINAL_INVALID",
    "GENERATION_STATUS_VALID",
    "INTERMEDIATE_SCHEMA_VERSION",
    "ResamplingGenerationRecord",
    "adjudicate_sentence_resampling_intermediates",
    "generate_sentence_resampling_intermediates",
]
