from __future__ import annotations

from typing import Any

import pytest

from model_forensics.adjudication import (
    AdjudicationRequest,
    BlindedAdjudicationCase,
    JudgeProvenance,
)
from model_forensics.estimate_spans import collect_first_estimate_span
from model_forensics.io import stable_hash
from model_forensics.lens_positions import LensPositionError, build_lens_position_row
from model_forensics.token_spans import CompletionTokenMap, token_stream_manifest


class CharacterTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


class FixedSpanCaller:
    not_for_primary_inference = False

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(provider="test", model_id="external-span-judge")

    def complete(self, request: AdjudicationRequest) -> str:
        del request
        return '{"status":"KNOWN","source":"trace","quote":"39,000,000","occurrence":1}'


def _fixture() -> tuple[dict[str, Any], dict[str, Any], Any, str]:
    raw = (
        "<think>Initial estimate 39,000,000. Accuracy should remain objective."
        "</think>Final answer: 42,000,000."
    )
    prompt_ids = (ord("P"),)
    completion_ids = tuple(ord(character) for character in raw)
    token_streams = token_stream_manifest(
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
    )
    token_map = CompletionTokenMap.from_manifest(
        tokenizer=CharacterTokenizer(),
        raw_text=raw,
        token_streams=token_streams,
    )
    reasoning = token_map.sections.reasoning
    answer = token_map.sections.answer
    rollout: dict[str, Any] = {
        "run_id": "trace-1",
        "task": "giraffe",
        "reasoning": reasoning,
        "answer": answer,
        "raw_text": raw,
        "token_streams": token_streams,
    }
    rollout["record_hash"] = stable_hash(rollout)

    sentence = "Accuracy should remain objective."
    char_start = reasoning.index(sentence)
    char_end = char_start + len(sentence)
    anchor_span = token_map.map_reasoning_span(
        char_start,
        char_end,
        expected_text=sentence,
    )
    anchor = {
        "anchor_id": "anchor-1",
        "trace_id": "trace-1",
        "direction": "above_good",
        "sentence_text": sentence,
        "char_start": char_start,
        "char_end": char_end,
        "provenance": {"token_span": anchor_span.as_dict()},
    }
    question = "How many black spots are on all living giraffes?"
    return rollout, anchor, CharacterTokenizer(), question


def test_position_builder_uses_exact_original_stream_and_blind_span() -> None:
    rollout, anchor, tokenizer, question = _fixture()
    case = BlindedAdjudicationCase(
        task_question=question,
        trace=rollout["reasoning"],
        answer=rollout["answer"],
    )
    span_record, _ = collect_first_estimate_span(case, FixedSpanCaller())
    row = build_lens_position_row(
        rollout=rollout,
        anchor=anchor,
        first_estimate_record=span_record,
        tokenizer=tokenizer,
        task_question=question,
        anchor_manifest_hash="sha256:anchors",
    )
    assert tuple(row["position_order"]) == (
        "prompt_end",
        "first_estimate_pre",
        "anchor_pre",
        "anchor_post",
        "final_answer_pre",
    )
    assert row["position_indices"]["prompt_end"] == 0
    assert row["position_indices"]["first_estimate_pre"] < row["position_indices"]["anchor_pre"]
    assert row["position_indices"]["anchor_pre"] < row["position_indices"]["anchor_post"]
    assert row["position_indices"]["anchor_post"] < row["position_indices"]["final_answer_pre"]
    assert row["good_side_direction"] == 1
    assert row["causal_claim"] is False
    assert row["record_hash"].startswith("sha256:")


def test_position_builder_rejects_tampered_anchor_or_nonprimary_span() -> None:
    rollout, anchor, tokenizer, question = _fixture()
    case = BlindedAdjudicationCase(question, rollout["reasoning"], rollout["answer"])
    span_record, _ = collect_first_estimate_span(case, FixedSpanCaller())

    tampered = {**anchor, "sentence_text": "Accuracy should remain subjective."}
    with pytest.raises((LensPositionError, ValueError)):
        build_lens_position_row(
            rollout=rollout,
            anchor=tampered,
            first_estimate_record=span_record,
            tokenizer=tokenizer,
            task_question=question,
            anchor_manifest_hash="sha256:anchors",
        )
