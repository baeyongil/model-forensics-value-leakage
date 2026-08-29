from __future__ import annotations

from dataclasses import replace

from model_forensics.anchors import (
    AnchorCandidate,
    AnchorSelectionError,
    select_frozen_anchors,
    sentence_spans,
    validate_anchor_manifest,
)


def test_sentence_spans_preserve_exact_offsets_and_handle_common_boundaries() -> None:
    text = "  Dr. Rao estimates 3.5 million.  \nBut I should avoid bias! Final fragment  "

    spans = sentence_spans(text)

    assert [span.text for span in spans] == [
        "Dr. Rao estimates 3.5 million.",
        "But I should avoid bias!",
        "Final fragment",
    ]
    assert [span.boundary for span in spans] == ["terminal", "terminal", "end"]
    assert [span.index for span in spans] == [0, 1, 2]
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_sentence_spans_drop_markdown_markers_headings_and_short_fragments() -> None:
    text = (
        "## Analysis\n"
        "1. **Estimate**\n"
        "- I multiply population by the daily rate.\n"
        "2.\n"
        "**Check**\n"
        "The resulting total is 42 million."
    )
    spans = sentence_spans(text)
    assert [span.text for span in spans] == [
        "- I multiply population by the daily rate.",
        "The resulting total is 42 million.",
    ]
    assert all(text[span.start : span.end] == span.text for span in spans)


def _balanced_candidates() -> list[AnchorCandidate]:
    classes = ("accuracy", "anti_bias", "search_stopping")
    directions = ("above_good", "below_good")
    strata = (("good", False), ("good", True), ("bad", False), ("bad", True))
    candidates: list[AnchorCandidate] = []
    for sentence_class in classes:
        for direction in directions:
            for ordinal, (initial_side, final_flip) in enumerate(strata):
                trace_id = f"{sentence_class}:{direction}:{ordinal}"
                sentence = f"Candidate {trace_id}."
                candidates.append(
                    AnchorCandidate(
                        trace_id=trace_id,
                        sentence_class=sentence_class,
                        direction=direction,
                        sentence_index=ordinal,
                        sentence_text=sentence,
                        char_start=100 + ordinal * 20,
                        char_end=100 + ordinal * 20 + len(sentence),
                        initial_side=initial_side,
                        final_flip=final_flip,
                    )
                )
    return candidates


def test_frozen_anchor_selection_is_balanced_distinct_and_order_invariant() -> None:
    candidates = _balanced_candidates()
    classes = ("accuracy", "anti_bias", "search_stopping")
    directions = ("above_good", "below_good")

    forward = select_frozen_anchors(
        candidates,
        sentence_classes=classes,
        directions=directions,
        seed="preregistered-v1",
    )
    reverse = select_frozen_anchors(
        reversed(candidates),
        sentence_classes=classes,
        directions=directions,
        seed="preregistered-v1",
    )

    assert len(forward.anchors) == 24
    assert len({anchor.trace_id for anchor in forward.anchors}) == 24
    assert forward == reverse
    assert len(forward.selection_hash) == 64
    for sentence_class in classes:
        for direction in directions:
            cell = [
                anchor
                for anchor in forward.anchors
                if anchor.sentence_class == sentence_class and anchor.direction == direction
            ]
            assert len(cell) == 4
            assert {(anchor.initial_side, anchor.final_flip) for anchor in cell} == {
                ("good", False),
                ("good", True),
                ("bad", False),
                ("bad", True),
            }
    validate_anchor_manifest(forward)


def test_manifest_validation_detects_hash_tampering_and_trace_reuse() -> None:
    manifest = select_frozen_anchors(
        _balanced_candidates(),
        sentence_classes=("accuracy", "anti_bias", "search_stopping"),
        directions=("above_good", "below_good"),
    )

    for corrupted in (
        replace(manifest, selection_hash="0" * 64),
        replace(
            manifest,
            anchors=(
                *manifest.anchors[:-1],
                replace(manifest.anchors[-1], trace_id=manifest.anchors[0].trace_id),
            ),
        ),
    ):
        try:
            validate_anchor_manifest(corrupted)
        except AnchorSelectionError:
            pass
        else:
            raise AssertionError("corrupted manifest unexpectedly validated")
