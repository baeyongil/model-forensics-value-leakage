from __future__ import annotations

import pytest

from model_forensics.token_spans import (
    CompletionTokenMap,
    TokenSpanMappingError,
    locate_completion_sections,
    token_stream_hash,
    token_stream_manifest,
    validate_token_stream_manifest,
)


class PieceTokenizer:
    """Small prefix-stable decoder with whitespace-bearing token fixtures."""

    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_locates_both_real_qwen_completion_formats_without_offset_loss() -> None:
    prompt_opened = (
        "1. Start with **41 million** giraffes.\n"
        "2. Adjust downward.\n"
        "</think>\n\n"
        "The estimate is 39,500,000."
    )
    sections = locate_completion_sections(prompt_opened)
    assert sections.reasoning == ("1. Start with **41 million** giraffes.\n2. Adjust downward.")
    assert (
        prompt_opened[sections.reasoning_char_start : sections.reasoning_char_end]
        == sections.reasoning
    )
    assert sections.answer == "The estimate is 39,500,000."
    assert sections.opening_think_in_completion is False
    assert sections.closing_think_in_completion is True

    completion_opened = (
        "  <think>\n- **Initial:** 41 million.\n- Revised: 39 million.\n</think>\n\n39,000,000  "
    )
    sections = locate_completion_sections(completion_opened)
    assert sections.reasoning == "- **Initial:** 41 million.\n- Revised: 39 million."
    assert sections.answer == "39,000,000"
    assert sections.opening_think_in_completion is True


def test_maps_markdown_list_sentence_to_original_completion_tokens() -> None:
    pieces = {
        1: "1.",
        2: " Start",
        3: " with",
        4: " **",
        5: "41",
        6: " million",
        7: "**",
        8: " giraffes.",
        9: "\n2.",
        10: " Adjust",
        11: " downward.",
        12: "\n",
        13: "</think>",
        14: "\n\n",
        15: "The",
        16: " estimate",
        17: " is",
        18: " 39,500,000.",
    }
    token_ids = tuple(pieces)
    raw_text = "".join(pieces.values())
    mapper = CompletionTokenMap(
        tokenizer=PieceTokenizer(pieces),
        raw_text=raw_text,
        completion_token_ids=token_ids,
    )
    sentence = "2. Adjust downward."
    start = mapper.sections.reasoning.index(sentence)
    span = mapper.map_reasoning_span(start, start + len(sentence), expected_text=sentence)

    assert span.token_ids == (9, 10, 11)
    assert span.leading_envelope_text == "\n"
    assert span.trailing_envelope_text == ""
    assert span.text == sentence
    assert span.sequence_token_span(4) == (12, 15)
    assert span.completion_token_ids_hash == token_stream_hash(token_ids, stream="completion")
    assert span.as_dict()["round_trip_verified"] is True


def test_maps_sentence_when_completion_contains_opening_think_tag() -> None:
    pieces = {
        101: "<think>",
        102: "\n-",
        103: " **Initial:**",
        104: " 41",
        105: " million.",
        106: "\n-",
        107: " Revised:",
        108: " 39",
        109: " million.",
        110: "\n",
        111: "</think>",
        112: "\n\n",
        113: "39,000,000",
    }
    raw_text = "".join(pieces.values())
    mapper = CompletionTokenMap(
        tokenizer=PieceTokenizer(pieces),
        raw_text=raw_text,
        completion_token_ids=tuple(pieces),
    )
    sentence = "- Revised: 39 million."
    start = mapper.sections.reasoning.index(sentence)
    span = mapper.map_reasoning_span(start, start + len(sentence), expected_text=sentence)

    assert span.token_ids == (106, 107, 108, 109)
    assert span.leading_envelope_text == "\n"
    assert raw_text[span.completion_char_start : span.completion_char_end] == sentence


def test_mapping_uses_prompt_context_and_recorded_special_token_filtering() -> None:
    class ContextualTokenizer:
        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert clean_up_tokenization_spaces is False
            pieces = {10: "PROMPT:", 20: " answer", 99: "" if skip_special_tokens else "<eos>"}
            return "".join(pieces[token_id] for token_id in token_ids)

    streams = token_stream_manifest(prompt_token_ids=(10,), completion_token_ids=(20, 99))
    mapper = CompletionTokenMap.from_manifest(
        tokenizer=ContextualTokenizer(),
        raw_text=" answer",
        token_streams=streams,
    )
    span = mapper.map_completion_span(1, 7, expected_text="answer")

    assert span.token_ids == (20,)
    assert span.leading_envelope_text == " "
    assert span.trailing_envelope_text == ""
    assert span.sequence_token_span(1) == (1, 2)


def test_strict_mapping_rejects_decode_mismatch_ambiguity_and_wrong_sentence() -> None:
    with pytest.raises(TokenSpanMappingError, match="duplicate"):
        locate_completion_sections("<think>a</think><think>b</think>")

    tokenizer = PieceTokenizer({1: "a", 2: "b"})
    with pytest.raises(TokenSpanMappingError, match="do not decode exactly"):
        CompletionTokenMap(tokenizer=tokenizer, raw_text="ac", completion_token_ids=(1, 2))

    mapper = CompletionTokenMap(tokenizer=tokenizer, raw_text="ab", completion_token_ids=(1, 2))
    with pytest.raises(TokenSpanMappingError, match="expected text"):
        mapper.map_completion_span(0, 1, expected_text="z")

    class NonMonotoneTokenizer:
        def decode(self, token_ids: list[int], **_: object) -> str:
            return {(1,): "x", (1, 2): "ab", (): ""}[tuple(token_ids)]

    mapper = CompletionTokenMap(
        tokenizer=NonMonotoneTokenizer(), raw_text="ab", completion_token_ids=(1, 2)
    )
    with pytest.raises(TokenSpanMappingError, match="not a prefix"):
        mapper.map_completion_span(0, 1, expected_text="a")


def test_token_stream_manifest_is_exact_domain_separated_and_stable() -> None:
    first = token_stream_manifest(prompt_token_ids=(1, 2), completion_token_ids=(1, 2))
    second = token_stream_manifest(prompt_token_ids=[1, 2], completion_token_ids=[1, 2])
    assert first == second
    assert first["prompt_token_ids"] == [1, 2]
    assert first["completion_token_ids"] == [1, 2]
    assert first["prompt_token_ids_hash"] != first["completion_token_ids_hash"]
    assert first["combined_token_stream_hash"].startswith("sha256:")
    assert validate_token_stream_manifest(first, require_both=True) == ((1, 2), (1, 2))

    tampered = {**first, "completion_token_ids": [1, 3]}
    with pytest.raises(TokenSpanMappingError, match="failed validation"):
        validate_token_stream_manifest(tampered, require_both=True)
