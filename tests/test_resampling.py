from __future__ import annotations

import inspect

from model_forensics.anchors import sentence_spans
from model_forensics.resampling import (
    DIVERGENCE_COSINE_THRESHOLD,
    INITIAL_SAMPLES_PER_ARM,
    MAX_CLASS_CI_HALF_WIDTH,
    MIN_DIVERGENT_RESAMPLES,
    TOP_UP_SAMPLES_PER_ARM,
    AnchorSamplingSummary,
    assess_semantic_divergence,
    build_token_identical_prefixes,
    first_generated_replacement_sentence,
    plan_effect_independent_top_up,
    raw_thinking_prefix_before,
)


class RecordingPrefixEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[dict[str, str], ...], str]] = []

    def encode_prefix(
        self, messages: tuple[dict[str, str], ...], raw_thinking_prefix: str
    ) -> list[int]:
        self.calls.append((messages, raw_thinking_prefix))
        return [101, len(raw_thinking_prefix), 202]


def test_retain_and_resample_start_from_one_token_identical_prefix() -> None:
    trace = "First thought.  Anchor sentence. Later thought."
    anchor = sentence_spans(trace)[1]
    raw_prefix = raw_thinking_prefix_before(trace, anchor)
    encoder = RecordingPrefixEncoder()
    messages = ({"role": "user", "content": "Estimate carefully."},)

    prefixes = build_token_identical_prefixes(messages, raw_prefix, encoder)

    assert raw_prefix == "First thought.  "
    assert prefixes.retain_token_ids == (101, len(raw_prefix), 202)
    assert prefixes.retain_token_ids is prefixes.resample_token_ids
    assert len(prefixes.prefix_hash) == 64
    assert encoder.calls == [(messages, raw_prefix)]


class LookupEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: tuple[str, ...]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def test_extracts_only_the_first_generated_replacement_sentence() -> None:
    generated = "  A different estimate follows. Keep reasoning after this."

    replacement = first_generated_replacement_sentence(generated)

    assert replacement is not None
    assert replacement.text == "A different estimate follows."
    assert generated[replacement.start : replacement.end] == replacement.text


def test_semantic_divergence_uses_injected_embedder_and_strict_point_eight_cutoff() -> None:
    vectors = {
        "Original accuracy claim.": [1.0, 0.0],
        "Clearly different.": [0.6, 0.8],
        "Exactly boundary.": [0.8, 0.6],
    }
    embedder = LookupEmbedder(vectors)

    divergent = assess_semantic_divergence(
        "Original accuracy claim.", "Clearly different.", embedder
    )
    boundary = assess_semantic_divergence("Original accuracy claim.", "Exactly boundary.", embedder)

    assert DIVERGENCE_COSINE_THRESHOLD == 0.8
    assert divergent.cosine_similarity == 0.6
    assert divergent.divergent is True
    assert boundary.cosine_similarity == 0.8
    assert boundary.divergent is False
    assert embedder.calls == [
        ("Original accuracy claim.", "Clearly different."),
        ("Original accuracy claim.", "Exactly boundary."),
    ]


def test_top_up_targets_only_low_divergence_anchors_or_imprecise_classes() -> None:
    summaries = (
        AnchorSamplingSummary("accuracy-low", "accuracy", 10, 10, 7),
        AnchorSamplingSummary("accuracy-ok", "accuracy", 10, 10, 8),
        AnchorSamplingSummary("anti-bias-a", "anti_bias", 10, 10, 9),
        AnchorSamplingSummary("anti-bias-b", "anti_bias", 10, 10, 8),
        AnchorSamplingSummary("boundary", "search_stopping", 10, 10, 8),
    )

    plan = plan_effect_independent_top_up(
        summaries,
        class_ci_half_widths={
            "accuracy": 0.10,
            "anti_bias": 0.13,
            "search_stopping": 0.12,
        },
    )
    decisions = {decision.anchor_id: decision for decision in plan.decisions}

    assert INITIAL_SAMPLES_PER_ARM == 10
    assert TOP_UP_SAMPLES_PER_ARM == 20
    assert MIN_DIVERGENT_RESAMPLES == 8
    assert MAX_CLASS_CI_HALF_WIDTH == 0.12
    assert decisions["accuracy-low"].target_samples_per_arm == 20
    assert decisions["accuracy-low"].reasons == ("divergent_resamples_below_8",)
    assert decisions["accuracy-ok"].target_samples_per_arm == 10
    assert decisions["accuracy-ok"].reasons == ()
    assert decisions["anti-bias-a"].target_samples_per_arm == 20
    assert decisions["anti-bias-a"].reasons == ("class_ci_half_width_above_0.12",)
    assert decisions["anti-bias-b"].target_samples_per_arm == 20
    assert decisions["boundary"].target_samples_per_arm == 10
    assert decisions["accuracy-low"].additional_retain_samples == 10
    assert decisions["accuracy-low"].additional_resample_samples == 10

    parameter_names = inspect.signature(plan_effect_independent_top_up).parameters
    assert not any("effect" in name or "estimate" in name for name in parameter_names)
