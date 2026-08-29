from __future__ import annotations

import pytest

from model_forensics.adjudication import (
    AdjudicationRequest,
    BlindedAdjudicationCase,
    JudgeProvenance,
)
from model_forensics.estimate_spans import (
    EstimateSpanError,
    SpanStatus,
    collect_first_estimate_span,
    parse_first_estimate_span,
)


class SpanCaller:
    not_for_primary_inference = False

    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(provider="test", model_id="external-span-judge")

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        return self.response


def _case() -> BlindedAdjudicationCase:
    return BlindedAdjudicationCase(
        task_question="How many black spots are on all living giraffes?",
        trace="Inputs include 117,000 animals. Estimate 39m, then revise 39m later.",
        answer="Final answer: 42m.",
    )


def test_blind_span_is_source_linked_and_occurrence_disambiguated() -> None:
    caller = SpanCaller('{"status":"KNOWN","source":"trace","quote":"39m","occurrence":2}')
    record, raw = collect_first_estimate_span(_case(), caller)
    assert raw == caller.response
    assert record.adjudication.status is SpanStatus.KNOWN
    assert _case().trace[record.resolved_char_start : record.resolved_char_end] == "39m"
    assert record.resolved_char_start == _case().trace.rfind("39m")
    assert len(caller.requests) == 1
    assert set(caller.requests[0].user_payload) == {"task_question", "trace", "answer"}
    assert record.primary_inference is True
    assert record.to_dict()["record_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"status":"KNOWN","source":"trace","quote":"39m","occurrence":1,"x":0}',
        '{"status":"KNOWN","source":"trace","quote":"39m","occurrence":true}',
        '{"status":"KNOWN","source":"trace","quote":"estimate","occurrence":1}',
        '{"status":"UNKNOWN","source":"trace","quote":null,"occurrence":null}',
    ],
)
def test_span_parser_rejects_malformed_or_unverifiable_schema(raw: str) -> None:
    with pytest.raises(EstimateSpanError):
        parse_first_estimate_span(raw)


def test_absent_occurrence_and_smoke_primary_are_rejected() -> None:
    caller = SpanCaller('{"status":"KNOWN","source":"answer","quote":"42m","occurrence":2}')
    with pytest.raises(EstimateSpanError, match="absent"):
        collect_first_estimate_span(_case(), caller)

    caller.not_for_primary_inference = True
    with pytest.raises(ValueError, match="not_for_primary"):
        collect_first_estimate_span(_case(), caller)
