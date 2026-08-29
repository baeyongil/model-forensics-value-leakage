"""Audited mapping from generated text offsets to the exact generated tokens.

Retokenizing a decoded completion is not a valid way to recover the tokens that
the model consumed: whitespace-sensitive tokenizers can produce a different
segmentation.  This module instead treats the backend's original token IDs as
authoritative and locates character boundaries by decoding prefixes of that
same immutable stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_forensics.io import stable_hash

OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"
TOKEN_SPAN_SCHEMA_VERSION = "1"


class TokenSpanMappingError(ValueError):
    """Raised when text cannot be mapped to original tokens without guessing."""


def token_stream_hash(token_ids: Sequence[int], *, stream: str) -> str:
    """Hash an ordered token stream with an explicit domain separator."""

    normalized = _normalize_token_ids(token_ids)
    if not stream or not isinstance(stream, str):
        raise ValueError("stream must be a non-empty string")
    return stable_hash(
        {
            "schema_version": TOKEN_SPAN_SCHEMA_VERSION,
            "stream": stream,
            "token_ids": list(normalized),
        }
    )


@dataclass(frozen=True, slots=True)
class CompletionSections:
    """Exact, trimmed section offsets into one raw model completion."""

    raw_text: str
    reasoning: str
    answer: str
    reasoning_char_start: int
    reasoning_char_end: int
    answer_char_start: int
    answer_char_end: int
    opening_think_in_completion: bool
    closing_think_in_completion: bool

    def __post_init__(self) -> None:
        length = len(self.raw_text)
        for start, end, name in (
            (self.reasoning_char_start, self.reasoning_char_end, "reasoning"),
            (self.answer_char_start, self.answer_char_end, "answer"),
        ):
            if not 0 <= start <= end <= length:
                raise TokenSpanMappingError(f"invalid {name} character offsets")
        if self.raw_text[self.reasoning_char_start : self.reasoning_char_end] != self.reasoning:
            raise TokenSpanMappingError("reasoning offsets do not round-trip")
        if self.raw_text[self.answer_char_start : self.answer_char_end] != self.answer:
            raise TokenSpanMappingError("answer offsets do not round-trip")

    def as_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "answer": self.answer,
            "reasoning_char_start": self.reasoning_char_start,
            "reasoning_char_end": self.reasoning_char_end,
            "answer_char_start": self.answer_char_start,
            "answer_char_end": self.answer_char_end,
            "opening_think_in_completion": self.opening_think_in_completion,
            "closing_think_in_completion": self.closing_think_in_completion,
        }


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def locate_completion_sections(raw_text: str) -> CompletionSections:
    """Locate Qwen thinking and answer sections without normalizing the text.

    Qwen/vLLM can expose either of two valid completion forms:

    * ``<think>... </think> answer`` when the opening tag was generated, or
    * ``... </think> answer`` when the chat template placed ``<think>`` in the
      prompt and generation therefore begins inside the thinking section.

    A tag-free completion is treated as an answer-only completion.  Duplicate,
    reordered, or non-leading opening tags are rejected because choosing one
    occurrence would make downstream sentence offsets ambiguous.
    """

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    open_count = raw_text.count(OPEN_THINK)
    close_count = raw_text.count(CLOSE_THINK)
    if open_count > 1 or close_count > 1:
        raise TokenSpanMappingError("completion contains duplicate thinking tags")

    open_at = raw_text.find(OPEN_THINK)
    close_at = raw_text.find(CLOSE_THINK)
    has_open = open_at >= 0
    has_close = close_at >= 0

    if has_open and raw_text[:open_at].strip():
        raise TokenSpanMappingError("opening <think> tag is not at the completion start")
    if has_open and has_close and close_at < open_at + len(OPEN_THINK):
        raise TokenSpanMappingError("thinking tags are out of order")

    if has_open:
        reasoning_region_start = open_at + len(OPEN_THINK)
        reasoning_region_end = close_at if has_close else len(raw_text)
    elif has_close:
        # The opening tag lives in the rendered prompt.  Character zero is the
        # first generated character inside the reasoning block.
        reasoning_region_start = 0
        reasoning_region_end = close_at
    else:
        reasoning_region_start = 0
        reasoning_region_end = 0

    if has_close:
        answer_region_start = close_at + len(CLOSE_THINK)
        answer_region_end = len(raw_text)
    elif has_open:
        answer_region_start = len(raw_text)
        answer_region_end = len(raw_text)
    else:
        answer_region_start = 0
        answer_region_end = len(raw_text)

    reasoning_start, reasoning_end = _trimmed_bounds(
        raw_text, reasoning_region_start, reasoning_region_end
    )
    answer_start, answer_end = _trimmed_bounds(raw_text, answer_region_start, answer_region_end)

    return CompletionSections(
        raw_text=raw_text,
        reasoning=raw_text[reasoning_start:reasoning_end],
        answer=raw_text[answer_start:answer_end],
        reasoning_char_start=reasoning_start,
        reasoning_char_end=reasoning_end,
        answer_char_start=answer_start,
        answer_char_end=answer_end,
        opening_think_in_completion=has_open,
        closing_think_in_completion=has_close,
    )


@dataclass(frozen=True, slots=True)
class OriginalTokenSpan:
    """A character span covered by an audited slice of original token IDs.

    ``token_start`` and ``token_end`` are half-open offsets into the generated
    completion token stream.  The decoded token envelope can include leading or
    trailing characters when the requested sentence begins or ends inside a
    whitespace-bearing token; those exact strings are recorded rather than
    silently retokenizing the sentence.
    """

    section: str
    section_char_start: int
    section_char_end: int
    completion_char_start: int
    completion_char_end: int
    token_start: int
    token_end: int
    token_envelope_char_start: int
    token_envelope_char_end: int
    text: str
    leading_envelope_text: str
    trailing_envelope_text: str
    token_ids: tuple[int, ...]
    token_ids_hash: str
    completion_token_ids_hash: str
    round_trip_verified: bool = True

    def __post_init__(self) -> None:
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise TokenSpanMappingError("token span must be non-empty and ordered")
        if len(self.token_ids) != self.token_end - self.token_start:
            raise TokenSpanMappingError("token span length does not match token_ids")
        if self.completion_char_end <= self.completion_char_start:
            raise TokenSpanMappingError("character span must be non-empty and ordered")
        if not self.round_trip_verified:
            raise TokenSpanMappingError("unverified token spans cannot be materialized")

    def sequence_token_span(self, prompt_token_count: int) -> tuple[int, int]:
        """Return this span's half-open offsets in prompt-plus-completion space."""

        if isinstance(prompt_token_count, bool) or not isinstance(prompt_token_count, int):
            raise TypeError("prompt_token_count must be an integer")
        if prompt_token_count < 0:
            raise ValueError("prompt_token_count must be non-negative")
        return (
            prompt_token_count + self.token_start,
            prompt_token_count + self.token_end,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TOKEN_SPAN_SCHEMA_VERSION,
            "section": self.section,
            "section_char_start": self.section_char_start,
            "section_char_end": self.section_char_end,
            "completion_char_start": self.completion_char_start,
            "completion_char_end": self.completion_char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_envelope_char_start": self.token_envelope_char_start,
            "token_envelope_char_end": self.token_envelope_char_end,
            "text": self.text,
            "leading_envelope_text": self.leading_envelope_text,
            "trailing_envelope_text": self.trailing_envelope_text,
            "token_ids": list(self.token_ids),
            "token_ids_hash": self.token_ids_hash,
            "completion_token_ids_hash": self.completion_token_ids_hash,
            "round_trip_verified": self.round_trip_verified,
        }


def _normalize_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise TypeError("token_ids must be a sequence of integers")
    normalized = tuple(token_ids)
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in normalized):
        raise TypeError("token_ids must contain only integers")
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("token_ids must be non-negative")
    return normalized


def _decode(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool,
) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise TokenSpanMappingError("tokenizer must expose a decode method")
    try:
        decoded = decoder(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:  # tokenizer implementations expose varied exceptions
        raise TokenSpanMappingError(f"tokenizer decode failed: {exc}") from exc
    if not isinstance(decoded, str):
        raise TokenSpanMappingError("tokenizer.decode must return a string")
    return decoded


class CompletionTokenMap:
    """Lazy decode-prefix index over one immutable completion token stream."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        raw_text: str,
        completion_token_ids: Sequence[int],
        prompt_token_ids: Sequence[int] = (),
        skip_special_tokens: bool = True,
    ):
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a bool")
        self.tokenizer = tokenizer
        self.raw_text = raw_text
        self.prompt_token_ids = _normalize_token_ids(prompt_token_ids)
        self.token_ids = _normalize_token_ids(completion_token_ids)
        self.skip_special_tokens = skip_special_tokens
        self.sections = locate_completion_sections(raw_text)
        self.prompt_token_ids_hash = token_stream_hash(self.prompt_token_ids, stream="prompt")
        self.completion_token_ids_hash = token_stream_hash(self.token_ids, stream="completion")
        self._prefix_cache: dict[int, str] = {0: ""}
        self._decoded_prompt = _decode(
            self.tokenizer,
            self.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        decoded = self._decode_combined_completion_prefix(len(self.token_ids))
        if decoded != raw_text:
            raise TokenSpanMappingError(
                "original prompt-plus-completion token IDs do not decode exactly to raw_text "
                "under the recorded detokenization settings"
            )
        self._prefix_cache[len(self.token_ids)] = decoded

    @classmethod
    def from_manifest(
        cls,
        *,
        tokenizer: Any,
        raw_text: str,
        token_streams: Mapping[str, Any],
        skip_special_tokens: bool = True,
    ) -> CompletionTokenMap:
        """Build a map only after validating a persisted token-stream manifest."""

        prompt, completion = validate_token_stream_manifest(token_streams, require_both=True)
        assert prompt is not None  # established by require_both
        assert completion is not None  # established by require_both
        return cls(
            tokenizer=tokenizer,
            raw_text=raw_text,
            completion_token_ids=completion,
            prompt_token_ids=prompt,
            skip_special_tokens=skip_special_tokens,
        )

    def _decode_combined_completion_prefix(self, count: int) -> str:
        combined = _decode(
            self.tokenizer,
            (*self.prompt_token_ids, *self.token_ids[:count]),
            skip_special_tokens=self.skip_special_tokens,
        )
        if not combined.startswith(self._decoded_prompt):
            raise TokenSpanMappingError(
                "decoding completion tokens changed the decoded prompt prefix"
            )
        return combined[len(self._decoded_prompt) :]

    def _decode_prefix(self, count: int) -> str:
        if not 0 <= count <= len(self.token_ids):
            raise TokenSpanMappingError("token prefix boundary is out of range")
        if count not in self._prefix_cache:
            decoded = self._decode_combined_completion_prefix(count)
            if not self.raw_text.startswith(decoded):
                raise TokenSpanMappingError(
                    "a decoded token prefix is not a prefix of raw_text; "
                    "character offsets are not recoverable without guessing"
                )
            self._prefix_cache[count] = decoded
        return self._prefix_cache[count]

    def _prefix_length(self, count: int) -> int:
        return len(self._decode_prefix(count))

    def _rightmost_boundary_at_or_before(self, character: int) -> int:
        low = 0
        high = len(self.token_ids)
        while low < high:
            middle = (low + high + 1) // 2
            if self._prefix_length(middle) <= character:
                low = middle
            else:
                high = middle - 1
        return low

    def _leftmost_boundary_at_or_after(self, character: int) -> int:
        low = 0
        high = len(self.token_ids)
        while low < high:
            middle = (low + high) // 2
            if self._prefix_length(middle) >= character:
                high = middle
            else:
                low = middle + 1
        return low

    def map_completion_span(
        self,
        char_start: int,
        char_end: int,
        *,
        expected_text: str | None = None,
        section: str = "completion",
        section_char_start: int | None = None,
        section_char_end: int | None = None,
    ) -> OriginalTokenSpan:
        """Map a non-empty completion character span to covering original tokens."""

        for name, value in (("char_start", char_start), ("char_end", char_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= char_start < char_end <= len(self.raw_text):
            raise TokenSpanMappingError("character span is empty or outside raw_text")

        text = self.raw_text[char_start:char_end]
        if expected_text is not None and text != expected_text:
            raise TokenSpanMappingError("expected text does not match the requested character span")

        token_start = self._rightmost_boundary_at_or_before(char_start)
        token_end = self._leftmost_boundary_at_or_after(char_end)
        if token_start >= token_end:
            raise TokenSpanMappingError("character span does not map to a non-empty token span")

        envelope_start = self._prefix_length(token_start)
        envelope_end = self._prefix_length(token_end)
        if not envelope_start <= char_start < char_end <= envelope_end:
            raise TokenSpanMappingError("decoded token envelope does not contain character span")

        # Check the adjacent boundaries explicitly.  This catches a non-monotone
        # decoder even when the binary search happened not to probe that point.
        if token_start < len(self.token_ids):
            after_start = self._prefix_length(token_start + 1)
            if after_start <= char_start:
                raise TokenSpanMappingError(
                    "token-start boundary is not the rightmost valid boundary"
                )
        if token_end > 0:
            before_end = self._prefix_length(token_end - 1)
            if before_end >= char_end:
                raise TokenSpanMappingError("token-end boundary is not the leftmost valid boundary")

        prefix_start = self._decode_prefix(token_start)
        prefix_end = self._decode_prefix(token_end)
        envelope_text = self.raw_text[envelope_start:envelope_end]
        if prefix_end[len(prefix_start) :] != envelope_text:
            raise TokenSpanMappingError(
                "decoded token envelope failed strict round-trip validation"
            )

        span_token_ids = self.token_ids[token_start:token_end]
        relative_start = char_start if section_char_start is None else section_char_start
        relative_end = char_end if section_char_end is None else section_char_end
        return OriginalTokenSpan(
            section=section,
            section_char_start=relative_start,
            section_char_end=relative_end,
            completion_char_start=char_start,
            completion_char_end=char_end,
            token_start=token_start,
            token_end=token_end,
            token_envelope_char_start=envelope_start,
            token_envelope_char_end=envelope_end,
            text=text,
            leading_envelope_text=self.raw_text[envelope_start:char_start],
            trailing_envelope_text=self.raw_text[char_end:envelope_end],
            token_ids=span_token_ids,
            token_ids_hash=token_stream_hash(span_token_ids, stream="completion_span"),
            completion_token_ids_hash=self.completion_token_ids_hash,
        )

    def map_reasoning_span(
        self,
        char_start: int,
        char_end: int,
        *,
        expected_text: str | None = None,
    ) -> OriginalTokenSpan:
        """Map half-open offsets relative to the trimmed reasoning section."""

        for name, value in (("char_start", char_start), ("char_end", char_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= char_start < char_end <= len(self.sections.reasoning):
            raise TokenSpanMappingError("reasoning span is empty or outside the reasoning section")
        completion_start = self.sections.reasoning_char_start + char_start
        completion_end = self.sections.reasoning_char_start + char_end
        return self.map_completion_span(
            completion_start,
            completion_end,
            expected_text=expected_text,
            section="reasoning",
            section_char_start=char_start,
            section_char_end=char_end,
        )


def token_stream_manifest(
    *,
    prompt_token_ids: Sequence[int] | None,
    completion_token_ids: Sequence[int] | None,
) -> dict[str, Any]:
    """Serialize exact generation streams and domain-separated content hashes."""

    prompt = None if prompt_token_ids is None else _normalize_token_ids(prompt_token_ids)
    completion = (
        None if completion_token_ids is None else _normalize_token_ids(completion_token_ids)
    )
    return {
        "schema_version": TOKEN_SPAN_SCHEMA_VERSION,
        "prompt_token_ids": None if prompt is None else list(prompt),
        "completion_token_ids": None if completion is None else list(completion),
        "prompt_token_ids_hash": (
            None if prompt is None else token_stream_hash(prompt, stream="prompt")
        ),
        "completion_token_ids_hash": (
            None if completion is None else token_stream_hash(completion, stream="completion")
        ),
        "combined_token_stream_hash": (
            None
            if prompt is None or completion is None
            else stable_hash(
                {
                    "schema_version": TOKEN_SPAN_SCHEMA_VERSION,
                    "prompt_token_ids_hash": token_stream_hash(prompt, stream="prompt"),
                    "completion_token_ids_hash": token_stream_hash(completion, stream="completion"),
                }
            )
        ),
    }


def validate_token_stream_manifest(
    manifest: Mapping[str, Any],
    *,
    require_both: bool = False,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Reject persisted token provenance whose IDs and hashes disagree."""

    if not isinstance(manifest, Mapping):
        raise TypeError("token stream manifest must be a mapping")
    if manifest.get("schema_version") != TOKEN_SPAN_SCHEMA_VERSION:
        raise TokenSpanMappingError("unsupported token stream manifest schema")
    prompt_raw = manifest.get("prompt_token_ids")
    completion_raw = manifest.get("completion_token_ids")
    if prompt_raw is not None and not isinstance(prompt_raw, Sequence):
        raise TokenSpanMappingError("prompt_token_ids must be a sequence or null")
    if completion_raw is not None and not isinstance(completion_raw, Sequence):
        raise TokenSpanMappingError("completion_token_ids must be a sequence or null")
    prompt = None if prompt_raw is None else _normalize_token_ids(prompt_raw)
    completion = None if completion_raw is None else _normalize_token_ids(completion_raw)
    if require_both and (prompt is None or completion is None):
        raise TokenSpanMappingError("both exact token streams are required")

    expected = token_stream_manifest(
        prompt_token_ids=prompt,
        completion_token_ids=completion,
    )
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise TokenSpanMappingError(f"token stream manifest field {key!r} failed validation")
    return prompt, completion


__all__ = [
    "CLOSE_THINK",
    "OPEN_THINK",
    "CompletionSections",
    "CompletionTokenMap",
    "OriginalTokenSpan",
    "TokenSpanMappingError",
    "locate_completion_sections",
    "token_stream_hash",
    "token_stream_manifest",
    "validate_token_stream_manifest",
]
