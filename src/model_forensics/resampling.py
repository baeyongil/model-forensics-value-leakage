"""Model-agnostic primitives for causal sentence resampling.

All model and embedding dependencies enter through small protocols, keeping unit
tests and manifest analysis runnable without loading a tokenizer or neural model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .anchors import SentenceSpan, sentence_spans


@runtime_checkable
class RawThinkingPrefixEncoder(Protocol):
    """Chat-template adapter that ends exactly at a raw thinking prefix."""

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class ArmPrefixes:
    """The shared intervention prefix exposed under both arm names."""

    retain_token_ids: tuple[int, ...]
    resample_token_ids: tuple[int, ...]
    prefix_hash: str

    def __post_init__(self) -> None:
        if self.retain_token_ids != self.resample_token_ids:
            raise ValueError("retain and resample prefixes must be token-identical")
        if not self.retain_token_ids:
            raise ValueError("intervention prefix cannot be empty")


def raw_thinking_prefix_before(trace_text: str, anchor: SentenceSpan) -> str:
    """Return the exact, unnormalized trace slice preceding ``anchor``."""

    if not isinstance(trace_text, str):
        raise TypeError("trace_text must be a string")
    anchor.validate_against(trace_text)
    return trace_text[: anchor.start]


def _token_hash(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_token_identical_prefixes(
    messages: Sequence[Mapping[str, Any]],
    raw_thinking_prefix: str,
    encoder: RawThinkingPrefixEncoder,
) -> ArmPrefixes:
    """Render once and reuse the exact tokens for retain and resample arms.

    Rendering once is intentional: even a stateful or accidentally stochastic
    chat-template adapter cannot create a hidden arm difference.
    """

    if not isinstance(raw_thinking_prefix, str):
        raise TypeError("raw_thinking_prefix must be a string")
    encoded = encoder.encode_prefix(messages, raw_thinking_prefix)
    token_ids = tuple(encoded)
    if any(type(token_id) is not int for token_id in token_ids):
        raise TypeError("prefix encoder must return integer token IDs")
    return ArmPrefixes(
        retain_token_ids=token_ids,
        resample_token_ids=token_ids,
        prefix_hash=_token_hash(token_ids),
    )


DIVERGENCE_COSINE_THRESHOLD = 0.8


@runtime_checkable
class TextEmbedder(Protocol):
    """Minimal sentence-embedding dependency used by the divergence filter."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class SemanticDivergence:
    reference_sentence: str
    replacement_sentence: str
    cosine_similarity: float
    threshold: float
    divergent: bool


def first_generated_replacement_sentence(generated_text: str) -> SentenceSpan | None:
    """Extract the first sentence-like unit emitted after the shared prefix."""

    spans = sentence_spans(generated_text)
    return spans[0] if spans else None


def _coerce_vector(vector: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} embedding must be a numeric vector") from error
    if not values:
        raise ValueError(f"{name} embedding cannot be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} embedding must contain only finite values")
    return values


def assess_semantic_divergence(
    reference_sentence: str,
    replacement_sentence: str,
    embedder: TextEmbedder,
    *,
    threshold: float = DIVERGENCE_COSINE_THRESHOLD,
) -> SemanticDivergence:
    """Classify a replacement as divergent when cosine similarity is below 0.8."""

    if not reference_sentence.strip() or not replacement_sentence.strip():
        raise ValueError("reference and replacement sentences must be non-empty")
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("cosine threshold must be finite and between -1 and 1")

    embeddings = embedder.encode((reference_sentence, replacement_sentence))
    if len(embeddings) != 2:
        raise ValueError("embedder must return exactly one vector per input sentence")
    reference = _coerce_vector(embeddings[0], name="reference")
    replacement = _coerce_vector(embeddings[1], name="replacement")
    if len(reference) != len(replacement):
        raise ValueError("reference and replacement embeddings must have equal dimensions")

    reference_norm = math.sqrt(sum(value * value for value in reference))
    replacement_norm = math.sqrt(sum(value * value for value in replacement))
    if reference_norm == 0.0 or replacement_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero-norm embedding")
    similarity = sum(
        reference_value * replacement_value
        for reference_value, replacement_value in zip(reference, replacement, strict=True)
    ) / (reference_norm * replacement_norm)
    similarity = min(1.0, max(-1.0, similarity))

    return SemanticDivergence(
        reference_sentence=reference_sentence,
        replacement_sentence=replacement_sentence,
        cosine_similarity=similarity,
        threshold=threshold,
        divergent=similarity < threshold,
    )


INITIAL_SAMPLES_PER_ARM = 10
TOP_UP_SAMPLES_PER_ARM = 20
MIN_DIVERGENT_RESAMPLES = 8
MAX_CLASS_CI_HALF_WIDTH = 0.12


class TopUpRuleError(ValueError):
    """Raised when a precision-only top-up decision cannot be audited."""


@dataclass(frozen=True, slots=True)
class AnchorSamplingSummary:
    """Effect-free inputs available after an anchor's initial sampling stage."""

    anchor_id: str
    sentence_class: str
    retain_samples: int
    resample_samples: int
    divergent_resamples: int

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.sentence_class:
            raise ValueError("anchor_id and sentence_class must be non-empty")
        counts = (
            self.retain_samples,
            self.resample_samples,
            self.divergent_resamples,
        )
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("sampling counts must be non-negative integers")
        if self.divergent_resamples > self.resample_samples:
            raise ValueError("divergent resamples cannot exceed resample count")


@dataclass(frozen=True, slots=True)
class AnchorTopUpDecision:
    anchor_id: str
    sentence_class: str
    target_samples_per_arm: int
    additional_retain_samples: int
    additional_resample_samples: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopUpPlan:
    """Stable set of per-anchor allocations produced without point effects."""

    decisions: tuple[AnchorTopUpDecision, ...]
    plan_hash: str

    def decision_for(self, anchor_id: str) -> AnchorTopUpDecision:
        matches = [decision for decision in self.decisions if decision.anchor_id == anchor_id]
        if len(matches) != 1:
            raise KeyError(anchor_id)
        return matches[0]


def _top_up_plan_hash(decisions: Sequence[AnchorTopUpDecision]) -> str:
    payload = [
        {
            "anchor_id": decision.anchor_id,
            "sentence_class": decision.sentence_class,
            "target_samples_per_arm": decision.target_samples_per_arm,
            "additional_retain_samples": decision.additional_retain_samples,
            "additional_resample_samples": decision.additional_resample_samples,
            "reasons": list(decision.reasons),
        }
        for decision in decisions
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def plan_effect_independent_top_up(
    summaries: Iterable[AnchorSamplingSummary],
    *,
    class_ci_half_widths: Mapping[str, float],
    initial_samples_per_arm: int = INITIAL_SAMPLES_PER_ARM,
    top_up_samples_per_arm: int = TOP_UP_SAMPLES_PER_ARM,
    min_divergent_resamples: int = MIN_DIVERGENT_RESAMPLES,
    max_class_ci_half_width: float = MAX_CLASS_CI_HALF_WIDTH,
) -> TopUpPlan:
    """Apply the preregistered 10-to-20 precision/divergence top-up rule.

    The inputs intentionally contain no point estimate or outcome count.  An
    individual anchor tops up when fewer than eight initial resamples diverge;
    every anchor in a sentence class tops up when that class CI's half-width is
    greater than 0.12.  Equality does not trigger either strict threshold.
    """

    if initial_samples_per_arm != INITIAL_SAMPLES_PER_ARM:
        raise TopUpRuleError("the preregistered initial allocation is 10 per arm")
    if top_up_samples_per_arm != TOP_UP_SAMPLES_PER_ARM:
        raise TopUpRuleError("the preregistered top-up allocation is 20 per arm")
    if min_divergent_resamples != MIN_DIVERGENT_RESAMPLES:
        raise TopUpRuleError("the preregistered divergence minimum is 8")
    if max_class_ci_half_width != MAX_CLASS_CI_HALF_WIDTH:
        raise TopUpRuleError("the preregistered CI half-width threshold is 0.12")

    ordered = tuple(sorted(summaries, key=lambda summary: summary.anchor_id))
    if not ordered:
        raise TopUpRuleError("at least one anchor summary is required")
    if len({summary.anchor_id for summary in ordered}) != len(ordered):
        raise TopUpRuleError("anchor summaries must have unique anchor IDs")

    observed_classes = {summary.sentence_class for summary in ordered}
    missing_widths = observed_classes.difference(class_ci_half_widths)
    if missing_widths:
        raise TopUpRuleError(f"missing CI half-widths for classes: {sorted(missing_widths)!r}")
    validated_widths: dict[str, float] = {}
    for sentence_class in observed_classes:
        try:
            width = float(class_ci_half_widths[sentence_class])
        except (TypeError, ValueError) as error:
            raise TopUpRuleError(f"CI half-width for {sentence_class!r} must be numeric") from error
        if not math.isfinite(width) or width < 0.0:
            raise TopUpRuleError(
                f"CI half-width for {sentence_class!r} must be finite and non-negative"
            )
        validated_widths[sentence_class] = width

    decisions: list[AnchorTopUpDecision] = []
    for summary in ordered:
        if (
            summary.retain_samples < initial_samples_per_arm
            or summary.resample_samples < initial_samples_per_arm
        ):
            raise TopUpRuleError(
                f"anchor {summary.anchor_id!r} has not completed 10 samples per arm"
            )

        reasons: list[str] = []
        if summary.divergent_resamples < min_divergent_resamples:
            reasons.append("divergent_resamples_below_8")
        if validated_widths[summary.sentence_class] > max_class_ci_half_width:
            reasons.append("class_ci_half_width_above_0.12")

        already_topped_up = (
            summary.retain_samples > initial_samples_per_arm
            or summary.resample_samples > initial_samples_per_arm
        )
        target = top_up_samples_per_arm if reasons or already_topped_up else initial_samples_per_arm
        decisions.append(
            AnchorTopUpDecision(
                anchor_id=summary.anchor_id,
                sentence_class=summary.sentence_class,
                target_samples_per_arm=target,
                additional_retain_samples=max(0, target - summary.retain_samples),
                additional_resample_samples=max(0, target - summary.resample_samples),
                reasons=tuple(reasons),
            )
        )

    frozen_decisions = tuple(decisions)
    return TopUpPlan(
        decisions=frozen_decisions,
        plan_hash=_top_up_plan_hash(frozen_decisions),
    )
