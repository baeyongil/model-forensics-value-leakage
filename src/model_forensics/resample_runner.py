"""Execute paired sentence-resampling interventions from frozen anchors.

The runner is deliberately independent of vLLM, Transformers, and embedding
libraries.  A GPU process supplies a small backend that can render a raw
thinking prefix, tokenize a forced continuation, and generate from token IDs.
This keeps the causal intervention auditable and makes the execution contract
testable without network access or a model download.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    AdjudicationCaller,
    BlindedAdjudicationCase,
    ExternalAdjudicationOutput,
    KnowledgeStatus,
    build_adjudication_request,
    parse_final_adjudication,
)
from model_forensics.anchors import AnchorManifest, FrozenAnchor
from model_forensics.classification import redact_threshold_value
from model_forensics.io import stable_hash
from model_forensics.prompts import QUESTIONS, Task, is_good_outcome
from model_forensics.resampling import (
    INITIAL_SAMPLES_PER_ARM,
    TextEmbedder,
    assess_semantic_divergence,
    build_token_identical_prefixes,
    first_generated_replacement_sentence,
)
from model_forensics.schemas import RolloutRecord

ARMS = ("retain", "resample")
_SEED_MAX = 2**31 - 1
REPLACEMENT_CLASSIFICATION_PROTOCOL = "blinded-replacement-confirmatory-v1"
FINAL_OUTCOME_PROTOCOL = "blinded-resample-final-v1"
FIXED_STAGE_TWO_POLICY = "unconditional-additional-10-per-anchor-arm-v1"
STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM = 10


class ResampleExecutionError(ValueError):
    """Raised when an intervention can no longer satisfy the frozen design."""


class MalformedContinuationError(ResampleExecutionError):
    """Raised when a generated continuation has no closed thought or answer."""


class PrefixIdentityError(ResampleExecutionError):
    """Raised when either arm does not use the audited conditioning tokens."""


class ReplacementClassificationError(ResampleExecutionError):
    """Raised when a supplied blinded classification cannot be authenticated."""


class OutcomeAdjudicationError(ResampleExecutionError):
    """Raised when a final-outcome judgment breaks the blind primary contract."""


@dataclass(frozen=True, slots=True)
class BaseTrace:
    """The minimum base-rollout state required to execute one intervention."""

    base_trace_id: str
    prompt: str
    trace: str
    threshold: float
    condition: str
    task: str = "unknown"
    messages: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_trace_id or not self.prompt or not self.trace:
            raise ValueError("base_trace_id, prompt, and trace must be non-empty")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("base trace threshold must be finite")
        if self.condition not in {"above_good", "below_good"}:
            raise ValueError("sentence resampling requires an incentivized condition")
        messages = self.messages or ({"role": "user", "content": self.prompt},)
        object.__setattr__(self, "messages", tuple(dict(message) for message in messages))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def from_rollout(cls, rollout: RolloutRecord) -> BaseTrace:
        """Convert a canonical rollout without discarding its invocation identity."""

        if rollout.threshold is None:
            raise ValueError("an anchor base rollout must have a threshold")
        metadata_messages = rollout.metadata.get("messages")
        messages: tuple[Mapping[str, Any], ...] = ()
        if isinstance(metadata_messages, Sequence) and not isinstance(
            metadata_messages, (str, bytes)
        ):
            messages = tuple(
                dict(message) for message in metadata_messages if isinstance(message, Mapping)
            )
        return cls(
            base_trace_id=rollout.run_id,
            prompt=rollout.prompt,
            trace=rollout.trace,
            threshold=float(rollout.threshold),
            condition=rollout.condition,
            task=rollout.task,
            messages=messages,
            provenance=rollout.provenance.to_dict(include_hash=True),
        )


@dataclass(frozen=True, slots=True)
class ResampleAllocation:
    """One deterministic arm invocation in an initial or fixed stage-two manifest."""

    request_id: str
    anchor_id: str
    base_trace_id: str
    arm: str
    sample_index: int
    seed: int
    stage: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.anchor_id or not self.base_trace_id:
            raise ValueError("allocation identifiers must be non-empty")
        if self.arm not in ARMS:
            raise ValueError(f"unknown resampling arm: {self.arm!r}")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if not 0 < self.seed < _SEED_MAX:
            raise ValueError("seed must be a positive signed 31-bit integer")
        if self.stage not in {"initial", "stage_two"}:
            raise ValueError("allocation stage must be 'initial' or 'stage_two'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "anchor_id": self.anchor_id,
            "base_trace_id": self.base_trace_id,
            "arm": self.arm,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class ResampleAllocationManifest:
    """Content-addressed paired seeds, frozen before generation starts."""

    allocations: tuple[ResampleAllocation, ...]
    master_seed: int
    stage: str
    manifest_hash: str
    stage_two_policy_hash: str | None = None
    schema_version: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "master_seed": self.master_seed,
            "stage": self.stage,
            "stage_two_policy_hash": self.stage_two_policy_hash,
            "allocations": [allocation.as_dict() for allocation in self.allocations],
            "manifest_hash": self.manifest_hash,
        }


def _anchors(value: AnchorManifest | Sequence[FrozenAnchor]) -> tuple[FrozenAnchor, ...]:
    anchors = value.anchors if isinstance(value, AnchorManifest) else tuple(value)
    if not anchors:
        raise ValueError("at least one frozen anchor is required")
    if len({anchor.anchor_id for anchor in anchors}) != len(anchors):
        raise ValueError("frozen anchors must have unique anchor IDs")
    if len({anchor.trace_id for anchor in anchors}) != len(anchors):
        raise ValueError("frozen anchors must have unique base trace IDs")
    return tuple(sorted(anchors, key=lambda anchor: anchor.anchor_id))


def _manifest_hash(
    allocations: Sequence[ResampleAllocation],
    *,
    master_seed: int,
    stage: str,
    stage_two_policy_hash: str | None,
) -> str:
    return stable_hash(
        {
            "schema_version": 2,
            "master_seed": master_seed,
            "stage": stage,
            "stage_two_policy_hash": stage_two_policy_hash,
            "allocations": [allocation.as_dict() for allocation in allocations],
        }
    )


def _paired_seed(
    *,
    master_seed: int,
    anchor_id: str,
    sample_index: int,
    stage: str,
    unavailable: set[int],
) -> int:
    nonce = 0
    while True:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "design": "paired-sentence-resampling-v1",
                    "master_seed": master_seed,
                    "anchor_id": anchor_id,
                    "sample_index": sample_index,
                    "stage": stage,
                    "nonce": nonce,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big") % (_SEED_MAX - 1) + 1
        if seed not in unavailable:
            return seed
        nonce += 1


def _request_id(anchor: FrozenAnchor, *, arm: str, sample_index: int, seed: int, stage: str) -> str:
    return stable_hash(
        {
            "anchor_id": anchor.anchor_id,
            "base_trace_id": anchor.trace_id,
            "arm": arm,
            "sample_index": sample_index,
            "seed": seed,
            "stage": stage,
        }
    ).split(":", 1)[1][:24]


def _validate_allocation_pairs(allocations: Sequence[ResampleAllocation]) -> None:
    by_pair: dict[tuple[str, int], list[ResampleAllocation]] = {}
    for allocation in allocations:
        by_pair.setdefault((allocation.anchor_id, allocation.sample_index), []).append(allocation)
    seen_pair_seeds: set[int] = set()
    for pair, values in by_pair.items():
        arms = {value.arm for value in values}
        if arms == set(ARMS):
            if len(values) != 2 or len({value.seed for value in values}) != 1:
                raise ValueError(f"paired arms do not share exactly one seed: {pair!r}")
        elif len(values) != 1:
            raise ValueError(f"duplicate one-arm allocation: {pair!r}")
        seed = values[0].seed
        if seed in seen_pair_seeds:
            raise ValueError("seeds must be unique across anchor/sample pairs")
        seen_pair_seeds.add(seed)


def _validate_execution_manifest(
    anchors: Sequence[FrozenAnchor],
    manifest: ResampleAllocationManifest,
    *,
    primary_inference: bool,
) -> None:
    if manifest.schema_version != 2:
        raise ResampleExecutionError("allocation manifest has an unsupported schema version")
    if not manifest.allocations:
        raise ResampleExecutionError("allocation manifest cannot be empty")
    if any(allocation.stage != manifest.stage for allocation in manifest.allocations):
        raise ResampleExecutionError("allocation stage does not match its manifest")
    _validate_allocation_pairs(manifest.allocations)
    if manifest.stage == "initial":
        if manifest.stage_two_policy_hash is not None:
            raise ResampleExecutionError("initial allocation cannot name a stage-two policy")
    elif manifest.stage == "stage_two":
        if manifest.stage_two_policy_hash is None or not _valid_sha256(
            manifest.stage_two_policy_hash
        ):
            raise ResampleExecutionError("stage-two allocation requires a SHA-256 policy hash")
    else:
        raise ResampleExecutionError("allocation manifest has an unsupported stage")
    expected_hash = _manifest_hash(
        manifest.allocations,
        master_seed=manifest.master_seed,
        stage=manifest.stage,
        stage_two_policy_hash=manifest.stage_two_policy_hash,
    )
    if manifest.manifest_hash != expected_hash:
        raise ResampleExecutionError("allocation manifest content hash does not verify")

    if not primary_inference:
        return
    start = 0 if manifest.stage == "initial" else INITIAL_SAMPLES_PER_ARM
    stop = start + INITIAL_SAMPLES_PER_ARM
    expected = {
        (anchor.anchor_id, anchor.trace_id, arm, sample_index)
        for anchor in anchors
        for sample_index in range(start, stop)
        for arm in ARMS
    }
    observed = {
        (
            allocation.anchor_id,
            allocation.base_trace_id,
            allocation.arm,
            allocation.sample_index,
        )
        for allocation in manifest.allocations
    }
    if observed != expected or len(manifest.allocations) != len(expected):
        raise ResampleExecutionError(
            "primary allocation must contain all ten paired samples for every frozen anchor"
        )


def build_initial_allocation_manifest(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    master_seed: int,
    samples_per_arm: int = INITIAL_SAMPLES_PER_ARM,
) -> ResampleAllocationManifest:
    """Freeze the preregistered ten paired seeds per arm and anchor."""

    if samples_per_arm != INITIAL_SAMPLES_PER_ARM:
        raise ValueError("the preregistered initial allocation is exactly 10 per arm")
    ordered_anchors = _anchors(anchors)
    unavailable: set[int] = set()
    allocations: list[ResampleAllocation] = []
    for anchor in ordered_anchors:
        for sample_index in range(samples_per_arm):
            seed = _paired_seed(
                master_seed=master_seed,
                anchor_id=anchor.anchor_id,
                sample_index=sample_index,
                stage="initial",
                unavailable=unavailable,
            )
            unavailable.add(seed)
            for arm in ARMS:
                allocations.append(
                    ResampleAllocation(
                        request_id=_request_id(
                            anchor,
                            arm=arm,
                            sample_index=sample_index,
                            seed=seed,
                            stage="initial",
                        ),
                        anchor_id=anchor.anchor_id,
                        base_trace_id=anchor.trace_id,
                        arm=arm,
                        sample_index=sample_index,
                        seed=seed,
                        stage="initial",
                    )
                )
    frozen = tuple(allocations)
    _validate_allocation_pairs(frozen)
    return ResampleAllocationManifest(
        allocations=frozen,
        master_seed=master_seed,
        stage="initial",
        manifest_hash=_manifest_hash(
            frozen, master_seed=master_seed, stage="initial", stage_two_policy_hash=None
        ),
    )


def _validate_complete_initial_allocation(
    anchors: Sequence[FrozenAnchor],
    initial_manifest: ResampleAllocationManifest,
) -> None:
    """Reject a partial or outcome-selected stage-one manifest."""

    if initial_manifest.stage != "initial":
        raise ValueError("initial_manifest must contain the initial allocation")
    _validate_allocation_pairs(initial_manifest.allocations)
    expected = {
        (anchor.anchor_id, anchor.trace_id, arm, sample_index)
        for anchor in anchors
        for sample_index in range(INITIAL_SAMPLES_PER_ARM)
        for arm in ARMS
    }
    observed = {
        (
            allocation.anchor_id,
            allocation.base_trace_id,
            allocation.arm,
            allocation.sample_index,
        )
        for allocation in initial_manifest.allocations
    }
    if observed != expected or len(initial_manifest.allocations) != len(expected):
        raise ValueError(
            "initial_manifest must contain exactly samples 0--9 for both arms of every anchor"
        )
    if initial_manifest.stage_two_policy_hash is not None:
        raise ValueError("an initial manifest cannot contain a stage-two policy hash")
    expected_hash = _manifest_hash(
        initial_manifest.allocations,
        master_seed=initial_manifest.master_seed,
        stage="initial",
        stage_two_policy_hash=None,
    )
    if initial_manifest.manifest_hash != expected_hash:
        raise ValueError("initial_manifest content hash does not verify")


def build_fixed_stage_two_allocation_manifest(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    initial_manifest: ResampleAllocationManifest,
    master_seed: int,
    additional_samples_per_arm: int = STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM,
) -> ResampleAllocationManifest:
    """Freeze the unconditional additional ten paired samples per anchor/arm.

    The interface deliberately accepts no outcomes, divergence counts, effect
    estimates, or confidence intervals. Every frozen anchor receives samples
    10--19 in both arms regardless of anything observed in stage one.
    """

    if additional_samples_per_arm != STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM:
        raise ValueError("the preregistered stage-two allocation is exactly 10 per arm")
    ordered_anchors = _anchors(anchors)
    _validate_complete_initial_allocation(ordered_anchors, initial_manifest)
    if master_seed != initial_manifest.master_seed:
        raise ValueError("stage two must reuse the initial manifest master seed")

    policy_hash = stable_hash(
        {
            "policy": FIXED_STAGE_TWO_POLICY,
            "initial_manifest_hash": initial_manifest.manifest_hash,
            "initial_samples_per_arm": INITIAL_SAMPLES_PER_ARM,
            "additional_samples_per_arm": STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM,
            "anchor_ids": [anchor.anchor_id for anchor in ordered_anchors],
        }
    )
    initial_seeds = {allocation.seed for allocation in initial_manifest.allocations}
    unavailable = set(initial_seeds)
    allocations: list[ResampleAllocation] = []
    stop = INITIAL_SAMPLES_PER_ARM + STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM
    for anchor in ordered_anchors:
        for sample_index in range(INITIAL_SAMPLES_PER_ARM, stop):
            seed = _paired_seed(
                master_seed=master_seed,
                anchor_id=anchor.anchor_id,
                sample_index=sample_index,
                stage="stage_two",
                unavailable=unavailable,
            )
            unavailable.add(seed)
            for arm in ARMS:
                allocations.append(
                    ResampleAllocation(
                        request_id=_request_id(
                            anchor,
                            arm=arm,
                            sample_index=sample_index,
                            seed=seed,
                            stage="stage_two",
                        ),
                        anchor_id=anchor.anchor_id,
                        base_trace_id=anchor.trace_id,
                        arm=arm,
                        sample_index=sample_index,
                        seed=seed,
                        stage="stage_two",
                    )
                )
    frozen = tuple(allocations)
    _validate_allocation_pairs(frozen)
    if initial_seeds.intersection(allocation.seed for allocation in frozen):
        raise ValueError("stage-two seeds collide with initial seeds")
    return ResampleAllocationManifest(
        allocations=frozen,
        master_seed=master_seed,
        stage="stage_two",
        stage_two_policy_hash=policy_hash,
        manifest_hash=_manifest_hash(
            frozen,
            master_seed=master_seed,
            stage="stage_two",
            stage_two_policy_hash=policy_hash,
        ),
    )


@dataclass(frozen=True, slots=True)
class RawPrefixGenerationRequest:
    """A token-prefill request directly adaptable to ``vllm.LLM.generate``."""

    request_id: str
    anchor_id: str
    base_trace_id: str
    arm: str
    sample_index: int
    seed: int
    messages: tuple[Mapping[str, Any], ...]
    conditioning_text: str
    prompt_token_ids: tuple[int, ...]
    common_prefix_token_count: int

    @property
    def vllm_prompt(self) -> dict[str, list[int]]:
        """Return vLLM's dependency-free ``TokensPrompt`` shape."""

        return {"prompt_token_ids": list(self.prompt_token_ids)}


@dataclass(frozen=True, slots=True)
class RawPrefixGenerationResult:
    """Provider-neutral continuation plus the prompt tokens actually consumed."""

    request_id: str
    generated_text: str
    prompt_token_ids: tuple[int, ...]
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RawPrefixGenerationBackend(Protocol):
    """The small adapter required from a vLLM or equivalent local backend."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]: ...

    def encode_continuation(self, raw_text: str) -> Sequence[int]: ...

    def generate(
        self, requests: Sequence[RawPrefixGenerationRequest]
    ) -> Sequence[RawPrefixGenerationResult]: ...


@dataclass(frozen=True, slots=True)
class NeutralControlSpec:
    """Predeclared sentence function that a confirmatory replacement must match."""

    control_id: str
    function_definition: str
    version: str

    def __post_init__(self) -> None:
        if not self.control_id.strip() or not self.function_definition.strip() or not self.version:
            raise ValueError("neutral-control fields must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "control_id": self.control_id,
            "function_definition": self.function_definition,
            "version": self.version,
        }

    @property
    def control_hash(self) -> str:
        return stable_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class ReplacementTokenTolerance:
    """Frozen absolute and relative length gates for position matching."""

    max_absolute_difference: int
    max_relative_difference: float

    def __post_init__(self) -> None:
        if isinstance(self.max_absolute_difference, bool) or self.max_absolute_difference < 0:
            raise ValueError("absolute token tolerance must be a non-negative integer")
        if not isinstance(self.max_absolute_difference, int):
            raise TypeError("absolute token tolerance must be an integer")
        if (
            not math.isfinite(float(self.max_relative_difference))
            or self.max_relative_difference < 0
        ):
            raise ValueError("relative token tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ReplacementClassificationRequest:
    """The complete outcome-blind payload visible to a replacement judge.

    The schema intentionally has no anchor/base IDs, seed, condition, direction,
    threshold value, estimates, answer, or outcome. Exact threshold numbers are
    redacted before this object is constructed.
    """

    request_id: str
    blinded_original_sentence: str
    blinded_replacement_sentence: str
    target_sentence_class: str
    neutral_control_id: str
    neutral_control_function: str
    neutral_control_version: str
    protocol_version: str = REPLACEMENT_CLASSIFICATION_PROTOCOL

    def __post_init__(self) -> None:
        values = (
            self.request_id,
            self.blinded_original_sentence,
            self.blinded_replacement_sentence,
            self.target_sentence_class,
            self.neutral_control_id,
            self.neutral_control_function,
            self.neutral_control_version,
        )
        if any(not value.strip() for value in values):
            raise ReplacementClassificationError("blinded classification fields cannot be blank")
        if self.protocol_version != REPLACEMENT_CLASSIFICATION_PROTOCOL:
            raise ReplacementClassificationError("unsupported replacement-classification protocol")

    def visible_payload(self) -> dict[str, str]:
        """Return exactly what an injected judge may inspect."""

        return {
            "protocol_version": self.protocol_version,
            "blinded_original_sentence": self.blinded_original_sentence,
            "blinded_replacement_sentence": self.blinded_replacement_sentence,
            "target_sentence_class": self.target_sentence_class,
            "neutral_control_id": self.neutral_control_id,
            "neutral_control_function": self.neutral_control_function,
            "neutral_control_version": self.neutral_control_version,
        }

    @property
    def input_hash(self) -> str:
        return stable_hash(self.visible_payload())

    @property
    def request_hash(self) -> str:
        return stable_hash({"request_id": self.request_id, "input_hash": self.input_hash})


def _valid_sha256(value: str) -> bool:
    digest = value.split(":", 1)[1] if value.startswith("sha256:") else value
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


@dataclass(frozen=True, slots=True)
class ReplacementClassificationResult:
    """Strict adjudicated result returned by an injected blinded classifier."""

    request_hash: str
    adjudication_valid: bool
    target_feature_absent_or_changed: bool | None
    neutral_control_function_matched: bool | None
    raw_judgment_hashes: tuple[str, ...]
    classifier_provenance_hash: str
    rationale: str

    def __post_init__(self) -> None:
        if not _valid_sha256(self.request_hash):
            raise ReplacementClassificationError("classification request hash is not SHA-256")
        if not _valid_sha256(self.classifier_provenance_hash):
            raise ReplacementClassificationError("classifier provenance hash is not SHA-256")
        if not self.raw_judgment_hashes or not all(
            _valid_sha256(value) for value in self.raw_judgment_hashes
        ):
            raise ReplacementClassificationError("raw judgment hashes must be non-empty SHA-256")
        if len(set(self.raw_judgment_hashes)) != len(self.raw_judgment_hashes):
            raise ReplacementClassificationError("raw judgment hashes must be distinct")
        if not self.rationale.strip():
            raise ReplacementClassificationError("classification rationale cannot be blank")
        if self.adjudication_valid:
            if (
                type(self.target_feature_absent_or_changed) is not bool
                or type(self.neutral_control_function_matched) is not bool
            ):
                raise ReplacementClassificationError(
                    "valid adjudication requires two explicit boolean judgments"
                )


@runtime_checkable
class ReplacementClassifier(Protocol):
    """Injected classifier that receives only :meth:`visible_payload` fields."""

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult | None: ...


def _replacement_classification_request(
    *,
    original_sentence: str,
    replacement_sentence: str,
    target_sentence_class: str,
    threshold: float,
    neutral_control: NeutralControlSpec,
) -> ReplacementClassificationRequest:
    visible = {
        "protocol_version": REPLACEMENT_CLASSIFICATION_PROTOCOL,
        "blinded_original_sentence": redact_threshold_value(original_sentence, threshold),
        "blinded_replacement_sentence": redact_threshold_value(replacement_sentence, threshold),
        "target_sentence_class": target_sentence_class,
        "neutral_control_id": neutral_control.control_id,
        "neutral_control_function": neutral_control.function_definition,
        "neutral_control_version": neutral_control.version,
    }
    # The opaque ID is derived only from the visible, redacted payload. It cannot
    # be joined to an arm, condition, outcome, seed, or base trace by the judge.
    request_id = stable_hash(visible).split(":", 1)[1][:24]
    return ReplacementClassificationRequest(request_id=request_id, **visible)


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _neutral_task_question(task: str) -> str:
    """Return only the quantity question, never the experimental treatment."""

    try:
        return QUESTIONS[Task(task)]
    except ValueError as exc:
        raise OutcomeAdjudicationError(
            f"unsupported resampling outcome-adjudication task: {task!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class FinalOutcomeAdjudicationAudit:
    """Auditable final-only judgment made through the existing provider contract."""

    case_hash: str
    task_question_hash: str
    trace_hash: str
    answer_hash: str
    request_id: str
    instrument_hash: str
    response_hash: str
    raw_response: str
    status: str
    value: int | None
    primary_inference: bool
    not_for_primary_inference: bool
    judge_provenance: Mapping[str, Any]
    protocol_version: str = FINAL_OUTCOME_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol_version != FINAL_OUTCOME_PROTOCOL:
            raise OutcomeAdjudicationError("unsupported final-outcome protocol")
        if self.status not in {KnowledgeStatus.KNOWN.value, KnowledgeStatus.UNKNOWN.value}:
            raise OutcomeAdjudicationError("final-outcome status must be KNOWN or UNKNOWN")
        if (self.status == KnowledgeStatus.KNOWN.value) != (self.value is not None):
            raise OutcomeAdjudicationError("final-outcome status and value are inconsistent")
        if self.primary_inference and self.not_for_primary_inference:
            raise OutcomeAdjudicationError(
                "primary outcome audit cannot contain non-primary judge output"
            )
        hashes = (
            self.case_hash,
            self.task_question_hash,
            self.trace_hash,
            self.answer_hash,
            self.request_id,
            self.instrument_hash,
            self.response_hash,
        )
        if not all(_valid_sha256(value) for value in hashes):
            raise OutcomeAdjudicationError("outcome audit hashes must be SHA-256")
        if self.response_hash != stable_hash({"raw_response": self.raw_response}):
            raise OutcomeAdjudicationError("outcome response hash does not verify")
        if not self.judge_provenance:
            raise OutcomeAdjudicationError("outcome judge provenance cannot be empty")

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "protocol_version": self.protocol_version,
            "primary_inference": self.primary_inference,
            "blinded_case_hash": self.case_hash,
            "visible_payload_fields": ["task_question", "trace", "answer"],
            "experimental_metadata_included": False,
            "input_hashes": {
                "task_question": self.task_question_hash,
                "trace": self.trace_hash,
                "answer": self.answer_hash,
            },
            "instrument_id": FINAL_ANSWER_INSTRUMENT.instrument_id,
            "instrument_hash": self.instrument_hash,
            "request_id": self.request_id,
            "response_hash": self.response_hash,
            "raw_response": self.raw_response,
            "status": self.status,
            "value": self.value,
            "not_for_primary_inference": self.not_for_primary_inference,
            "judge_provenance": dict(self.judge_provenance),
            "judge_provenance_hash": stable_hash(dict(self.judge_provenance)),
        }
        if include_hash:
            payload["audit_hash"] = stable_hash(payload)
        return payload


def adjudicate_final_resampling_outcome(
    *,
    task: str,
    trace: str,
    answer: str,
    caller: AdjudicationCaller,
    primary_inference: bool,
) -> FinalOutcomeAdjudicationAudit:
    """Judge one final answer without exposing arm, condition, or threshold fields."""

    if type(primary_inference) is not bool:
        raise TypeError("primary_inference must be an explicit bool")
    if primary_inference and caller.not_for_primary_inference:
        raise OutcomeAdjudicationError(
            "primary resampling refuses a caller marked not_for_primary_inference"
        )
    case = BlindedAdjudicationCase(
        task_question=_neutral_task_question(task),
        trace=trace,
        answer=answer,
    )
    request = build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)
    if set(request.user_payload) != {"task_question", "trace", "answer"}:  # pragma: no cover
        raise OutcomeAdjudicationError("final judge payload crossed the blinded field boundary")
    raw_response = caller.complete(request)
    output = ExternalAdjudicationOutput(
        request_id=request.request_id,
        instrument_id=request.instrument_id,
        instrument_hash=request.instrument_hash,
        raw_response=raw_response,
        provenance=caller.provenance,
        not_for_primary_inference=caller.not_for_primary_inference,
    )
    if primary_inference and output.not_for_primary_inference:
        raise OutcomeAdjudicationError(
            "primary resampling refuses an output marked not_for_primary_inference"
        )
    adjudication = parse_final_adjudication(output.raw_response)
    return FinalOutcomeAdjudicationAudit(
        case_hash=case.case_hash,
        task_question_hash=stable_hash(case.task_question),
        trace_hash=stable_hash(case.trace),
        answer_hash=stable_hash(case.answer),
        request_id=output.request_id,
        instrument_hash=output.instrument_hash,
        response_hash=output.response_hash,
        raw_response=output.raw_response,
        status=adjudication.status.value,
        value=(adjudication.value if adjudication.status is KnowledgeStatus.KNOWN else None),
        primary_inference=primary_inference,
        not_for_primary_inference=output.not_for_primary_inference,
        judge_provenance=output.provenance.to_dict(),
    )


@dataclass(frozen=True, slots=True)
class ResamplingArtifactRecord:
    """One auditable continuation with separate intervention/outcome eligibility.

    ``intervention_eligible`` is fixed entirely by intervention-delivery facts
    (primary run, valid generation, and the frozen replacement checks).  It must
    never depend on whether the final-outcome judges returned a measurement.
    ``confirmatory_eligible`` remains the complete-case outcome flag for legacy
    consumers.  Pair selection uses ``intervention_eligible`` so missing final
    outcomes remain visible to the missingness analysis.
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
    common_prefix_hash: str
    conditioning_prefix_hash: str
    common_prefix_token_count: int
    conditioning_prefix_token_count: int
    conditioning_prefix_text: str
    replacement_sentence: str
    replacement_char_start: int
    replacement_char_end: int
    cosine_similarity: float
    divergent: bool
    intervention_eligible: bool
    primary_eligible: bool
    confirmatory_eligible: bool
    analysis_tier: str
    anchor_token_count: int
    replacement_token_count: int
    token_count_absolute_difference: int
    token_count_relative_difference: float
    token_count_absolute_tolerance: int | None
    token_count_relative_tolerance: float | None
    token_count_within_absolute_tolerance: bool | None
    token_count_within_relative_tolerance: bool | None
    replacement_classification_status: str
    target_feature_absent_or_changed: bool | None
    neutral_control_function_matched: bool | None
    classifier_input_blinded: bool | None
    classifier_input_hash: str | None
    classifier_request_hash: str | None
    classifier_judgment_hashes: tuple[str, ...]
    classifier_provenance_hash: str | None
    classifier_provenance: Mapping[str, Any]
    neutral_control_hash: str | None
    classification_rationale: str | None
    full_trace: str
    answer: str
    raw_generated_text: str
    final_estimate: int | None
    final_good_side: bool | None
    final_measurement_valid: bool
    outcome_adjudication_primary_inference: bool
    outcome_adjudication: Mapping[str, Any]
    trajectory: Mapping[str, Any] | None
    usage: Mapping[str, Any]
    provenance: Mapping[str, Any]
    finish_reason: str
    synthetic_smoke: bool

    def __post_init__(self) -> None:
        if self.arm == "retain":
            expected_intervention_eligible = bool(
                self.outcome_adjudication_primary_inference
                and self.replacement_classification_status == "paired_reference"
            )
        elif self.arm == "resample":
            expected_intervention_eligible = bool(
                self.outcome_adjudication_primary_inference
                and self.divergent
                and self.replacement_classification_status == "valid"
                and self.target_feature_absent_or_changed is True
                and self.neutral_control_function_matched is True
                and self.classifier_input_blinded is True
                and self.token_count_within_absolute_tolerance is True
                and self.token_count_within_relative_tolerance is True
            )
        else:
            raise OutcomeAdjudicationError(f"invalid resampling arm: {self.arm!r}")
        if self.intervention_eligible is not expected_intervention_eligible:
            raise OutcomeAdjudicationError(
                "intervention_eligible must depend only on the frozen intervention checks"
            )
        if self.final_measurement_valid != (self.final_estimate is not None):
            raise OutcomeAdjudicationError(
                "final_measurement_valid must exactly track adjudicated final availability"
            )
        if self.final_measurement_valid != (self.final_good_side is not None):
            raise OutcomeAdjudicationError(
                "final_good_side must be present exactly when an adjudicated final is present"
            )
        if self.primary_eligible and (
            not self.outcome_adjudication_primary_inference or not self.final_measurement_valid
        ):
            raise OutcomeAdjudicationError(
                "primary eligibility requires a valid primary final-outcome adjudication"
            )
        if self.primary_eligible != self.confirmatory_eligible:
            raise OutcomeAdjudicationError(
                "primary_eligible and confirmatory_eligible must be identical"
            )
        if self.confirmatory_eligible and not self.intervention_eligible:
            raise OutcomeAdjudicationError(
                "confirmatory eligibility requires outcome-independent intervention eligibility"
            )
        if self.synthetic_smoke and self.outcome_adjudication_primary_inference:
            raise OutcomeAdjudicationError("synthetic smoke rows cannot claim primary inference")
        audit = dict(self.outcome_adjudication)
        if audit.get("protocol_version") != FINAL_OUTCOME_PROTOCOL:
            raise OutcomeAdjudicationError("outcome adjudication has the wrong protocol")
        if audit.get("primary_inference") is not self.outcome_adjudication_primary_inference:
            raise OutcomeAdjudicationError("outcome audit has the wrong inference tier")
        if audit.get("value") != self.final_estimate:
            raise OutcomeAdjudicationError("record final does not match its outcome audit")
        audit_hash = audit.pop("audit_hash", None)
        if audit_hash != stable_hash(audit):
            raise OutcomeAdjudicationError("outcome audit hash does not verify")
        if self.final_estimate is not None:
            expected_good_side = is_good_outcome(
                self.condition, self.final_estimate, self.threshold
            )
            if expected_good_side is not self.final_good_side:
                raise OutcomeAdjudicationError("final good-side label does not match judged value")

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
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
            "common_prefix_hash": self.common_prefix_hash,
            "conditioning_prefix_hash": self.conditioning_prefix_hash,
            "common_prefix_token_count": self.common_prefix_token_count,
            "conditioning_prefix_token_count": self.conditioning_prefix_token_count,
            "conditioning_prefix_text": self.conditioning_prefix_text,
            "replacement_sentence": self.replacement_sentence,
            "replacement_span": {
                "start": self.replacement_char_start,
                "end": self.replacement_char_end,
            },
            "replacement_char_start": self.replacement_char_start,
            "replacement_char_end": self.replacement_char_end,
            "cosine_similarity": self.cosine_similarity,
            "divergent": self.divergent,
            "intervention_eligible": self.intervention_eligible,
            "primary_eligible": self.primary_eligible,
            "confirmatory_eligible": self.confirmatory_eligible,
            "analysis_tier": self.analysis_tier,
            "anchor_token_count": self.anchor_token_count,
            "replacement_token_count": self.replacement_token_count,
            "token_count_absolute_difference": self.token_count_absolute_difference,
            "token_count_relative_difference": self.token_count_relative_difference,
            "token_count_absolute_tolerance": self.token_count_absolute_tolerance,
            "token_count_relative_tolerance": self.token_count_relative_tolerance,
            "token_count_within_absolute_tolerance": (self.token_count_within_absolute_tolerance),
            "token_count_within_relative_tolerance": (self.token_count_within_relative_tolerance),
            "replacement_classification_status": self.replacement_classification_status,
            "target_feature_absent_or_changed": self.target_feature_absent_or_changed,
            "neutral_control_function_matched": self.neutral_control_function_matched,
            "classifier_input_blinded": self.classifier_input_blinded,
            "classifier_input_hash": self.classifier_input_hash,
            "classifier_request_hash": self.classifier_request_hash,
            "classifier_judgment_hashes": list(self.classifier_judgment_hashes),
            "classifier_provenance_hash": self.classifier_provenance_hash,
            "classifier_provenance": dict(self.classifier_provenance),
            "neutral_control_hash": self.neutral_control_hash,
            "classification_rationale": self.classification_rationale,
            "full_trace": self.full_trace,
            "answer": self.answer,
            "raw_generated_text": self.raw_generated_text,
            "final_estimate": self.final_estimate,
            "final_good_side": self.final_good_side,
            "final_measurement_valid": self.final_measurement_valid,
            "outcome_adjudication_primary_inference": (self.outcome_adjudication_primary_inference),
            "outcome_adjudication": dict(self.outcome_adjudication),
            "trajectory": None if self.trajectory is None else dict(self.trajectory),
            "usage": dict(self.usage),
            "provenance": dict(self.provenance),
            "finish_reason": self.finish_reason,
            "synthetic_smoke": self.synthetic_smoke,
        }
        if include_hash:
            payload["record_hash"] = stable_hash(payload)
        return payload


def _coerce_base_trace(value: BaseTrace | RolloutRecord | Mapping[str, Any]) -> BaseTrace:
    if isinstance(value, BaseTrace):
        return value
    if isinstance(value, RolloutRecord):
        return BaseTrace.from_rollout(value)
    if not isinstance(value, Mapping):
        raise TypeError("base traces must be BaseTrace, RolloutRecord, or mappings")
    trace = value.get("trace", value.get("reasoning"))
    identifier = value.get("base_trace_id", value.get("run_id"))
    threshold = value.get("threshold")
    if not isinstance(trace, str) or not isinstance(identifier, str) or threshold is None:
        raise ValueError("base trace mappings require run_id, trace/reasoning, and threshold")
    messages_value = value.get("messages", ())
    messages = (
        tuple(dict(item) for item in messages_value if isinstance(item, Mapping))
        if isinstance(messages_value, Sequence) and not isinstance(messages_value, (str, bytes))
        else ()
    )
    provenance = value.get("provenance", value.get("backend", {}))
    return BaseTrace(
        base_trace_id=identifier,
        prompt=str(value.get("prompt", "")),
        trace=trace,
        threshold=float(threshold),
        condition=str(value.get("condition", "")),
        task=str(value.get("task", "unknown")),
        messages=messages,
        provenance=provenance if isinstance(provenance, Mapping) else {},
    )


def _split_generated_continuation(text: str) -> tuple[str, str]:
    if "</think>" not in text:
        raise MalformedContinuationError("generated continuation omits </think>")
    reasoning_tail, answer = text.split("</think>", 1)
    if not answer.strip():
        raise MalformedContinuationError("generated continuation omits the final answer")
    return reasoning_tail, answer.strip()


def _replacement_token_audit(
    *,
    anchor_token_count: int,
    replacement_token_count: int,
    tolerance: ReplacementTokenTolerance | None,
) -> dict[str, Any]:
    if anchor_token_count <= 0 or replacement_token_count <= 0:
        raise PrefixIdentityError("anchor and replacement token counts must be positive")
    absolute_difference = abs(replacement_token_count - anchor_token_count)
    relative_difference = absolute_difference / anchor_token_count
    return {
        "anchor_token_count": anchor_token_count,
        "replacement_token_count": replacement_token_count,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "absolute_tolerance": tolerance.max_absolute_difference if tolerance else None,
        "relative_tolerance": tolerance.max_relative_difference if tolerance else None,
        "within_absolute_tolerance": (
            absolute_difference <= tolerance.max_absolute_difference if tolerance else None
        ),
        "within_relative_tolerance": (
            relative_difference <= tolerance.max_relative_difference if tolerance else None
        ),
    }


def _replacement_classification_audit(
    *,
    classifier: ReplacementClassifier | None,
    neutral_control: NeutralControlSpec | None,
    original_sentence: str,
    replacement_sentence: str,
    target_sentence_class: str,
    threshold: float,
) -> dict[str, Any]:
    empty = {
        "status": "missing",
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
        "rationale": None,
    }
    if classifier is None:
        return empty
    if neutral_control is None:
        return {**empty, "status": "missing_preregistered_neutral_control"}

    request = _replacement_classification_request(
        original_sentence=original_sentence,
        replacement_sentence=replacement_sentence,
        target_sentence_class=target_sentence_class,
        threshold=threshold,
        neutral_control=neutral_control,
    )
    provenance = dict(classifier.provenance)
    if not provenance:
        raise ReplacementClassificationError("classifier provenance cannot be empty")
    provenance_hash = stable_hash(provenance)
    result = classifier.classify(request)
    request_audit = {
        "classifier_input_blinded": True,
        "classifier_input_hash": request.input_hash,
        "classifier_request_hash": request.request_hash,
        "classifier_provenance_hash": provenance_hash,
        "classifier_provenance": provenance,
        "neutral_control_hash": neutral_control.control_hash,
    }
    if result is None:
        return {**empty, **request_audit, "status": "missing_result"}
    if result.request_hash != request.request_hash:
        raise ReplacementClassificationError("classifier result echoes the wrong request hash")
    if result.classifier_provenance_hash != provenance_hash:
        raise ReplacementClassificationError("classifier result has the wrong provenance hash")
    return {
        **request_audit,
        "status": "valid" if result.adjudication_valid else "invalid_adjudication",
        "adjudication_valid": result.adjudication_valid,
        "target_feature_absent_or_changed": result.target_feature_absent_or_changed,
        "neutral_control_function_matched": result.neutral_control_function_matched,
        "raw_judgment_hashes": result.raw_judgment_hashes,
        "rationale": result.rationale,
    }


def _run_sentence_resampling_legacy(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    base_traces: Mapping[str, BaseTrace | RolloutRecord | Mapping[str, Any]],
    allocation_manifest: ResampleAllocationManifest,
    backend: RawPrefixGenerationBackend,
    embedder: TextEmbedder,
    outcome_caller: AdjudicationCaller,
    primary_inference: bool,
    replacement_classifier: ReplacementClassifier | None = None,
    neutral_control: NeutralControlSpec | None = None,
    token_tolerance: ReplacementTokenTolerance | None = None,
    on_record: Callable[[ResamplingArtifactRecord], None] | None = None,
) -> tuple[ResamplingArtifactRecord, ...]:
    """Execute frozen interventions and externally judge every final outcome.

    ``primary_inference`` is mandatory. This prevents smoke fixtures from being
    promoted merely because a caller forgot to opt into the scientific path.
    """

    if type(primary_inference) is not bool:
        raise TypeError("primary_inference must be an explicit bool")
    backend_provenance = dict(backend.provenance)
    synthetic_smoke = bool(backend_provenance.get("synthetic_smoke", False))
    if primary_inference and synthetic_smoke:
        raise OutcomeAdjudicationError("primary resampling refuses a synthetic smoke backend")
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
    expected_ids = {anchor.trace_id for anchor in ordered_anchors}
    if not expected_ids.issubset(base_traces):
        raise ResampleExecutionError(
            f"missing base traces: {sorted(expected_ids.difference(base_traces))!r}"
        )

    requests: list[RawPrefixGenerationRequest] = []
    contexts: dict[
        str,
        tuple[FrozenAnchor, BaseTrace, str, tuple[int, ...], tuple[int, ...]],
    ] = {}
    for anchor in ordered_anchors:
        base = _coerce_base_trace(base_traces[anchor.trace_id])
        if base.base_trace_id != anchor.trace_id:
            raise ResampleExecutionError("base trace ID does not match its frozen anchor")
        if base.condition != anchor.direction:
            raise ResampleExecutionError("base trace condition does not match anchor direction")
        if base.trace[anchor.char_start : anchor.char_end] != anchor.sentence_text:
            raise ResampleExecutionError("frozen anchor span no longer matches the base trace")
        _neutral_task_question(base.task)  # validate before the GPU generation call

        common_text = base.trace[: anchor.char_start]
        # The chat template is rendered exactly once for the shared pre-anchor
        # state. Both arms then reuse this exact immutable token tuple.
        arm_prefixes = build_token_identical_prefixes(base.messages, common_text, backend)
        common_tokens = arm_prefixes.retain_token_ids
        forced_anchor_tokens = tuple(backend.encode_continuation(anchor.sentence_text))
        if not forced_anchor_tokens or any(
            type(token) is not int for token in forced_anchor_tokens
        ):
            raise PrefixIdentityError("forced anchor tokenization must be non-empty integers")

        for allocation in allocation_manifest.allocations:
            if allocation.anchor_id != anchor.anchor_id:
                continue
            if allocation.base_trace_id != anchor.trace_id:
                raise ResampleExecutionError("allocation base trace does not match anchor")
            if allocation.stage != allocation_manifest.stage:
                raise ResampleExecutionError("allocation stage does not match its manifest")
            if allocation.arm == "retain":
                conditioning_text = common_text + anchor.sentence_text
                conditioning_tokens = common_tokens + forced_anchor_tokens
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
            requests.append(request)
            contexts[request.request_id] = (
                anchor,
                base,
                arm_prefixes.prefix_hash,
                common_tokens,
                forced_anchor_tokens,
            )

    if len(requests) != len(allocation_manifest.allocations):
        unknown = {
            allocation.anchor_id
            for allocation in allocation_manifest.allocations
            if allocation.anchor_id not in anchor_by_id
        }
        raise ResampleExecutionError(f"allocation contains unknown anchors: {sorted(unknown)!r}")
    if len({request.request_id for request in requests}) != len(requests):
        raise ResampleExecutionError("allocation contains duplicate request IDs")

    generated = tuple(backend.generate(tuple(requests)))
    by_id = {result.request_id: result for result in generated}
    if len(by_id) != len(generated):
        raise ResampleExecutionError("backend returned duplicate request IDs")
    if set(by_id) != {request.request_id for request in requests}:
        raise ResampleExecutionError("backend request/result IDs do not match")

    records: list[ResamplingArtifactRecord] = []
    for request in requests:
        result = by_id[request.request_id]
        if tuple(result.prompt_token_ids) != request.prompt_token_ids:
            raise PrefixIdentityError(
                f"backend used different prompt tokens for request {request.request_id}"
            )
        reasoning_tail, answer = _split_generated_continuation(result.generated_text)
        anchor, base, common_prefix_hash, common_tokens, anchor_tokens = contexts[
            request.request_id
        ]
        full_trace = request.conditioning_text + reasoning_tail
        if not full_trace.startswith(request.conditioning_text):  # pragma: no cover
            raise PrefixIdentityError("constructed trace lost its fixed conditioning prefix")

        if request.arm == "retain":
            replacement_sentence = anchor.sentence_text
            replacement_start = anchor.char_start
            replacement_end = anchor.char_end
            cosine_similarity = 1.0
            divergent = False
            replacement_tokens = anchor_tokens
        else:
            replacement = first_generated_replacement_sentence(reasoning_tail)
            if replacement is None:
                raise MalformedContinuationError("resample emitted no replacement sentence")
            replacement_sentence = replacement.text
            replacement_start = len(request.conditioning_text) + replacement.start
            replacement_end = len(request.conditioning_text) + replacement.end
            divergence = assess_semantic_divergence(
                anchor.sentence_text, replacement_sentence, embedder
            )
            cosine_similarity = divergence.cosine_similarity
            divergent = divergence.divergent
            replacement_tokens = tuple(backend.encode_continuation(replacement_sentence))
            if not replacement_tokens or any(
                type(token) is not int for token in replacement_tokens
            ):
                raise PrefixIdentityError(
                    "replacement tokenization must return non-empty integer tokens"
                )
        if full_trace[replacement_start:replacement_end] != replacement_sentence:
            raise ResampleExecutionError("replacement span does not reconstruct its sentence")

        token_audit = _replacement_token_audit(
            anchor_token_count=len(anchor_tokens),
            replacement_token_count=len(replacement_tokens),
            tolerance=token_tolerance,
        )
        if request.arm == "retain":
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
                "neutral_control_hash": neutral_control.control_hash if neutral_control else None,
                "rationale": "Retain is the frozen paired reference; no replacement was judged.",
            }
            replacement_eligible = True
        else:
            classification_audit = _replacement_classification_audit(
                classifier=replacement_classifier,
                neutral_control=neutral_control,
                original_sentence=anchor.sentence_text,
                replacement_sentence=replacement_sentence,
                target_sentence_class=anchor.sentence_class,
                threshold=base.threshold,
            )
            replacement_eligible = bool(
                classification_audit["adjudication_valid"]
                and classification_audit["target_feature_absent_or_changed"] is True
                and classification_audit["neutral_control_function_matched"] is True
                and classification_audit["classifier_input_blinded"] is True
                and token_audit["within_absolute_tolerance"] is True
                and token_audit["within_relative_tolerance"] is True
                and divergent
            )

        outcome_audit = adjudicate_final_resampling_outcome(
            task=base.task,
            trace=full_trace,
            answer=answer,
            caller=outcome_caller,
            primary_inference=primary_inference,
        )
        final_value = outcome_audit.value
        final_measurement_valid = final_value is not None
        good_side = (
            is_good_outcome(anchor.direction, final_value, base.threshold)
            if final_value is not None
            else None
        )
        if final_value is not None and good_side is None:  # pragma: no cover - BaseTrace guard
            raise ResampleExecutionError("anchor condition has no preregistered good side")
        intervention_eligible = bool(primary_inference and replacement_eligible)
        confirmatory_eligible = bool(intervention_eligible and final_measurement_valid)
        if not primary_inference:
            analysis_tier = "nonprimary_smoke"
        elif not final_measurement_valid:
            analysis_tier = "outcome_unmeasured"
        elif request.arm == "retain":
            analysis_tier = "paired_reference"
        else:
            analysis_tier = "confirmatory" if confirmatory_eligible else "exploratory"
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
        }
        provenance = {
            "backend": backend_provenance,
            "backend_result": dict(result.backend_metadata),
            "base_trace": dict(base.provenance),
            "allocation_manifest_hash": allocation_manifest.manifest_hash,
        }
        record = ResamplingArtifactRecord(
            resample_id=request.request_id,
            anchor_id=anchor.anchor_id,
            base_trace_id=anchor.trace_id,
            sentence_class=anchor.sentence_class,
            condition=anchor.direction,
            arm=request.arm,
            sample_index=request.sample_index,
            seed=request.seed,
            stage=allocation_manifest.stage,
            threshold=base.threshold,
            common_prefix_hash=common_prefix_hash,
            conditioning_prefix_hash=_token_hash(request.prompt_token_ids),
            common_prefix_token_count=len(common_tokens),
            conditioning_prefix_token_count=len(request.prompt_token_ids),
            conditioning_prefix_text=request.conditioning_text,
            replacement_sentence=replacement_sentence,
            replacement_char_start=replacement_start,
            replacement_char_end=replacement_end,
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
            full_trace=full_trace,
            answer=answer,
            raw_generated_text=result.generated_text,
            final_estimate=final_value,
            final_good_side=good_side,
            final_measurement_valid=final_measurement_valid,
            outcome_adjudication_primary_inference=primary_inference,
            outcome_adjudication=outcome_audit.as_dict(include_hash=True),
            trajectory=None,
            usage=usage,
            provenance=provenance,
            finish_reason=result.finish_reason,
            synthetic_smoke=synthetic_smoke,
        )
        records.append(record)
        if on_record is not None:
            on_record(record)
    return tuple(records)


def run_sentence_resampling(
    anchors: AnchorManifest | Sequence[FrozenAnchor],
    *,
    base_traces: Mapping[str, BaseTrace | RolloutRecord | Mapping[str, Any]],
    allocation_manifest: ResampleAllocationManifest,
    backend: RawPrefixGenerationBackend,
    embedder: TextEmbedder,
    outcome_caller: AdjudicationCaller,
    primary_inference: bool,
    replacement_classifier: ReplacementClassifier | None = None,
    neutral_control: NeutralControlSpec | None = None,
    token_tolerance: ReplacementTokenTolerance | None = None,
    on_record: Callable[[ResamplingArtifactRecord], None] | None = None,
) -> tuple[ResamplingArtifactRecord, ...]:
    """Compatibility wrapper over the isolated GPU and CPU/API phases.

    Production callers can persist and transfer the authenticated intermediate
    rows by invoking the two functions in :mod:`model_forensics.resample_phases`
    directly.  This combined entry point remains convenient for bounded smoke
    tests and existing integrations.
    """

    # Preserve the fail-before-GPU contract for clearly invalid paid callers.
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

    from model_forensics.resample_phases import (
        adjudicate_sentence_resampling_intermediates,
        generate_sentence_resampling_intermediates,
    )

    intermediates = generate_sentence_resampling_intermediates(
        anchors,
        base_traces=base_traces,
        allocation_manifest=allocation_manifest,
        backend=backend,
        primary_inference=primary_inference,
    )
    return adjudicate_sentence_resampling_intermediates(
        intermediates,
        anchors=anchors,
        base_traces=base_traces,
        allocation_manifest=allocation_manifest,
        embedder=embedder,
        outcome_caller=outcome_caller,
        primary_inference=primary_inference,
        replacement_classifier=replacement_classifier,
        neutral_control=neutral_control,
        token_tolerance=token_tolerance,
        on_record=on_record,
    )


def select_confirmatory_pairs(
    records: Sequence[ResamplingArtifactRecord],
) -> tuple[ResamplingArtifactRecord, ...]:
    """Return intervention-eligible pairs, including unmeasured final outcomes.

    Filtering on ``confirmatory_eligible`` would condition pair inclusion on the
    final measurement and erase precisely the resample-side missingness that the
    sensitivity analysis must report.  Intervention eligibility is fixed before
    final-outcome adjudication; once a resample intervention qualifies, this
    selector keeps both it and its exact retain reference even when either final
    outcome is missing.
    """

    by_pair: dict[
        tuple[str, int, int, str],
        dict[str, ResamplingArtifactRecord],
    ] = {}
    for record in records:
        key = (record.anchor_id, record.sample_index, record.seed, record.stage)
        arms = by_pair.setdefault(key, {})
        if record.arm in arms:
            raise ResampleExecutionError(f"duplicate {record.arm!r} row in pair {key!r}")
        arms[record.arm] = record

    accepted: set[tuple[str, int, int, str]] = set()
    for key, arms in by_pair.items():
        resample = arms.get("resample")
        if resample is None or not resample.intervention_eligible:
            continue
        retain = arms.get("retain")
        if retain is None:
            raise ResampleExecutionError(
                f"confirmatory resample lacks its paired retain reference: {key!r}"
            )
        if not retain.intervention_eligible:
            continue
        accepted.add(key)
    return tuple(
        record
        for record in records
        if (record.anchor_id, record.sample_index, record.seed, record.stage) in accepted
    )


__all__ = [
    "FINAL_OUTCOME_PROTOCOL",
    "FIXED_STAGE_TWO_POLICY",
    "REPLACEMENT_CLASSIFICATION_PROTOCOL",
    "STAGE_TWO_ADDITIONAL_SAMPLES_PER_ARM",
    "BaseTrace",
    "FinalOutcomeAdjudicationAudit",
    "MalformedContinuationError",
    "NeutralControlSpec",
    "OutcomeAdjudicationError",
    "PrefixIdentityError",
    "RawPrefixGenerationBackend",
    "RawPrefixGenerationRequest",
    "RawPrefixGenerationResult",
    "ReplacementClassificationError",
    "ReplacementClassificationRequest",
    "ReplacementClassificationResult",
    "ReplacementClassifier",
    "ReplacementTokenTolerance",
    "ResampleAllocation",
    "ResampleAllocationManifest",
    "ResampleExecutionError",
    "ResamplingArtifactRecord",
    "adjudicate_final_resampling_outcome",
    "build_fixed_stage_two_allocation_manifest",
    "build_initial_allocation_manifest",
    "run_sentence_resampling",
    "select_confirmatory_pairs",
]
