"""Blind, externally adjudicated source spans for lens token positions.

The behavioral trajectory instrument identifies *which values* are target
estimates.  That is not enough to place an observational lens readout on the
original token stream: retokenizing or searching for an equal number can select
the wrong mention.  This module asks a separate blind judge for the exact surface
form and its occurrence, then validates that selection against the immutable
trace/answer text.  No condition, direction, threshold, or outcome crosses the
caller boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from model_forensics.adjudication import (
    AdjudicationCaller,
    AdjudicationInstrument,
    BlindedAdjudicationCase,
    JudgeProvenance,
    build_adjudication_request,
)
from model_forensics.io import stable_hash

FIRST_ESTIMATE_SPAN_INSTRUMENT_ID = "target-first-estimate-span-v1"
FIRST_ESTIMATE_SPAN_INSTRUMENT_TEXT = """You are a blind source-span adjudicator.

Read the task question, private reasoning trace, and visible answer. Identify the
earliest textual mention at which the model proposes a tentative or committed
point estimate of the exact quantity asked in the task question. Exclude input
factors, rates, dates, ranges, subcomponent counts, and intermediate calculations
that the text does not use as an estimate of the requested target. Do not infer or
calculate a value.

For a recoverable estimate, copy the shortest exact numeric surface form from the
source text, identify whether it occurs in `trace` or `answer`, and give its
one-based occurrence number within that source string. Preserve punctuation,
spacing, commas, decimal points, magnitude suffixes, and currency signs exactly.

Return exactly one JSON object and no prose. It must have exactly these keys:
{"status":"KNOWN","source":"trace","quote":"<exact numeric surface>","occurrence":1}
or
{"status":"UNKNOWN","source":null,"quote":null,"occurrence":null}
"""

FIRST_ESTIMATE_SPAN_INSTRUMENT = AdjudicationInstrument(
    instrument_id=FIRST_ESTIMATE_SPAN_INSTRUMENT_ID,
    purpose="exact source span of earliest target-quantity point estimate",
    text=FIRST_ESTIMATE_SPAN_INSTRUMENT_TEXT,
)


class SpanStatus(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


class EstimateSpanError(ValueError):
    """A span response or its source link is not exactly auditable."""


def _reject_constant(token: str) -> None:
    raise EstimateSpanError(f"non-finite JSON constant is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EstimateSpanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class FirstEstimateSpan:
    status: SpanStatus
    source: str | None
    quote: str | None
    occurrence: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SpanStatus(self.status))
        if self.status is SpanStatus.UNKNOWN:
            if any(value is not None for value in (self.source, self.quote, self.occurrence)):
                raise EstimateSpanError("UNKNOWN span fields must all be null")
            return
        if self.source not in {"trace", "answer"}:
            raise EstimateSpanError("KNOWN source must be trace or answer")
        if not isinstance(self.quote, str) or not self.quote or len(self.quote) > 160:
            raise EstimateSpanError("KNOWN quote must be a nonempty bounded string")
        if not any(character.isdigit() for character in self.quote):
            raise EstimateSpanError("KNOWN quote must contain a numeric surface")
        if (
            isinstance(self.occurrence, bool)
            or not isinstance(self.occurrence, int)
            or self.occurrence <= 0
        ):
            raise EstimateSpanError("KNOWN occurrence must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "source": self.source,
            "quote": self.quote,
            "occurrence": self.occurrence,
        }


def parse_first_estimate_span(raw_response: str) -> FirstEstimateSpan:
    if not isinstance(raw_response, str):
        raise TypeError("span response must be a string")
    try:
        payload = json.loads(
            raw_response,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except EstimateSpanError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EstimateSpanError("span response must be one strict JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "status",
        "source",
        "quote",
        "occurrence",
    }:
        raise EstimateSpanError("span response has missing or extra keys")
    try:
        status = SpanStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise EstimateSpanError("status must be KNOWN or UNKNOWN") from exc
    return FirstEstimateSpan(
        status=status,
        source=payload["source"],
        quote=payload["quote"],
        occurrence=payload["occurrence"],
    )


def _occurrences(text: str, quote: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = text.find(quote, cursor)
        if start < 0:
            return starts
        starts.append(start)
        cursor = start + 1


@dataclass(frozen=True, slots=True)
class FirstEstimateSpanRecord:
    case_hash: str
    request_id: str
    instrument_hash: str
    response_hash: str
    provenance: JudgeProvenance
    adjudication: FirstEstimateSpan
    resolved_char_start: int | None
    resolved_char_end: int | None
    primary_inference: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.adjudication.status is SpanStatus.KNOWN:
            if (
                isinstance(self.resolved_char_start, bool)
                or not isinstance(self.resolved_char_start, int)
                or isinstance(self.resolved_char_end, bool)
                or not isinstance(self.resolved_char_end, int)
                or self.resolved_char_start < 0
                or self.resolved_char_end <= self.resolved_char_start
            ):
                raise EstimateSpanError("KNOWN span requires resolved character offsets")
        elif self.resolved_char_start is not None or self.resolved_char_end is not None:
            raise EstimateSpanError("UNKNOWN span cannot have resolved character offsets")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "primary_inference": self.primary_inference,
            "case_hash": self.case_hash,
            "request_id": self.request_id,
            "instrument_id": FIRST_ESTIMATE_SPAN_INSTRUMENT_ID,
            "instrument_hash": self.instrument_hash,
            "response_hash": self.response_hash,
            "provenance": self.provenance.to_dict(),
            "adjudication": self.adjudication.as_dict(),
            "resolved_char_start": self.resolved_char_start,
            "resolved_char_end": self.resolved_char_end,
        }
        if include_hash:
            payload["record_hash"] = stable_hash(payload)
        return payload


def collect_first_estimate_span(
    case: BlindedAdjudicationCase,
    caller: AdjudicationCaller,
    *,
    for_primary_inference: bool = True,
) -> tuple[FirstEstimateSpanRecord, str]:
    """Collect and source-validate one blind earliest-estimate span.

    The returned raw response is kept separate so a curated public manifest can
    expose hashes and parsed fields without requiring provider prose artifacts.
    """

    if for_primary_inference and caller.not_for_primary_inference:
        raise ValueError("caller is marked not_for_primary_inference")
    request = build_adjudication_request(case, FIRST_ESTIMATE_SPAN_INSTRUMENT)
    raw_response = caller.complete(request)
    adjudication = parse_first_estimate_span(raw_response)
    start: int | None = None
    end: int | None = None
    if adjudication.status is SpanStatus.KNOWN:
        assert adjudication.source is not None
        assert adjudication.quote is not None
        assert adjudication.occurrence is not None
        source_text = case.trace if adjudication.source == "trace" else case.answer
        starts = _occurrences(source_text, adjudication.quote)
        if adjudication.occurrence > len(starts):
            raise EstimateSpanError("quoted occurrence is absent from the exact source text")
        start = starts[adjudication.occurrence - 1]
        end = start + len(adjudication.quote)
        if source_text[start:end] != adjudication.quote:
            raise EstimateSpanError("resolved quote failed exact source round-trip")
    record = FirstEstimateSpanRecord(
        case_hash=case.case_hash,
        request_id=request.request_id,
        instrument_hash=request.instrument_hash,
        response_hash=stable_hash({"raw_response": raw_response}),
        provenance=caller.provenance,
        adjudication=adjudication,
        resolved_char_start=start,
        resolved_char_end=end,
        primary_inference=for_primary_inference,
    )
    return record, raw_response


__all__ = [
    "FIRST_ESTIMATE_SPAN_INSTRUMENT",
    "FIRST_ESTIMATE_SPAN_INSTRUMENT_ID",
    "FIRST_ESTIMATE_SPAN_INSTRUMENT_TEXT",
    "EstimateSpanError",
    "FirstEstimateSpan",
    "FirstEstimateSpanRecord",
    "SpanStatus",
    "collect_first_estimate_span",
    "parse_first_estimate_span",
]
