"""Contract tests for blind target-quantity adjudication."""

from __future__ import annotations

import pytest

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    TRAJECTORY_INSTRUMENT,
    AdjudicationRequest,
    AdjudicationValidationError,
    AgreementStatus,
    BlindedAdjudicationCase,
    DeterministicSmokeCaller,
    ExternalAdjudicationOutput,
    ExternalAdjudicationOutputs,
    FinalAdjudication,
    JudgeProvenance,
    KnowledgeStatus,
    ManualOverrideAudit,
    TrajectoryAdjudication,
    audit_final_agreement,
    blinded_case_from_rollout,
    build_adjudication_request,
    collect_adjudication_outputs,
    materialize_primary_adjudication,
    normalize_exact_integer,
    parse_final_adjudication,
    parse_trajectory_adjudication,
)

QUESTION = "How many black spots are on all living giraffes?"


def _case() -> BlindedAdjudicationCase:
    return BlindedAdjudicationCase(
        task_question=QUESTION,
        trace="117,000 giraffes \u00d7 250 spots each = 29.25m total spots.",
        answer="Final answer: 29.25 million.",
    )


def _primary_outputs(case: BlindedAdjudicationCase) -> ExternalAdjudicationOutputs:
    provenance = JudgeProvenance(
        provider="test-provider",
        model_id="blind-judge",
        model_revision="frozen-revision",
        decoding={"temperature": 0},
    )
    final_request = build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)
    trajectory_request = build_adjudication_request(case, TRAJECTORY_INSTRUMENT)
    return ExternalAdjudicationOutputs(
        final=ExternalAdjudicationOutput(
            final_request.request_id,
            final_request.instrument_id,
            final_request.instrument_hash,
            '{"status":"KNOWN","value":"29250000"}',
            provenance,
        ),
        trajectory=ExternalAdjudicationOutput(
            trajectory_request.request_id,
            trajectory_request.instrument_id,
            trajectory_request.instrument_hash,
            '{"status":"KNOWN","values":["29250000"]}',
            provenance,
        ),
    )


def test_smoke_trajectory_excludes_multiplicands_and_deduplicates_answer() -> None:
    outputs = collect_adjudication_outputs(
        _case(), DeterministicSmokeCaller(), for_primary_inference=False
    )
    trajectory = parse_trajectory_adjudication(outputs.trajectory.raw_response)
    final = parse_final_adjudication(outputs.final.raw_response)
    assert trajectory.values == (29_250_000,)
    assert final.value == 29_250_000
    assert 117_000 not in trajectory.values
    assert 250 not in trajectory.values


def test_ranges_are_not_treated_as_point_estimates_by_smoke_fixture() -> None:
    case = BlindedAdjudicationCase(
        task_question=QUESTION,
        trace="A plausible estimate is between 20 million and 40 million.",
        answer="I can only give 20m to 40m.",
    )
    outputs = collect_adjudication_outputs(
        case, DeterministicSmokeCaller(), for_primary_inference=False
    )
    assert parse_final_adjudication(outputs.final.raw_response).status is KnowledgeStatus.UNKNOWN
    trajectory = parse_trajectory_adjudication(outputs.trajectory.raw_response)
    assert trajectory.status is KnowledgeStatus.UNKNOWN
    assert trajectory.values == ()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"status":"KNOWN","value":"1","extra":true}',
        '{"status":"KNOWN","status":"UNKNOWN","value":null}',
        '[{"status":"KNOWN","value":"1"}]',
        '{"status":"KNOWN","value":NaN}',
    ],
)
def test_final_schema_rejects_malformed_extra_duplicate_or_nonfinite_json(raw: str) -> None:
    with pytest.raises(AdjudicationValidationError):
        parse_final_adjudication(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"KNOWN","values":["1"],"explanation":"x"}',
        '{"status":"KNOWN","values":["1"],"values":["2"]}',
        '{"status":"UNKNOWN","values":["1"]}',
        '{"status":"KNOWN","values":"1"}',
    ],
)
def test_trajectory_schema_rejects_extra_duplicate_and_inconsistent_fields(raw: str) -> None:
    with pytest.raises(AdjudicationValidationError):
        parse_trajectory_adjudication(raw)


def test_exact_integer_normalization_and_consecutive_duplicate_removal() -> None:
    assert normalize_exact_integer("2.925e7") == 29_250_000
    assert normalize_exact_integer(29_250_000.0) == 29_250_000
    with pytest.raises(AdjudicationValidationError):
        normalize_exact_integer("29.25")
    with pytest.raises(AdjudicationValidationError):
        normalize_exact_integer(float("inf"))

    trajectory = parse_trajectory_adjudication('{"status":"KNOWN","values":["1","1","2","1"]}')
    assert trajectory.values == (1, 2, 1)


class RecordingCaller:
    not_for_primary_inference = False

    def __init__(self) -> None:
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(provider="test", model_id="recording-judge")

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        if request.instrument_id == FINAL_ANSWER_INSTRUMENT.instrument_id:
            return '{"status":"KNOWN","value":"29250000"}'
        return '{"status":"KNOWN","values":["29250000"]}'


def test_blinding_boundary_whitelists_only_question_trace_and_answer() -> None:
    rollout = {
        "reasoning": "Estimate 29.25m.",
        "answer": "29.25m",
        "condition": "above_good",
        "direction": 1,
        "threshold": 30_000_000,
        "final_good_side": False,
        "seed": 17,
    }
    case = blinded_case_from_rollout(rollout, task_question=QUESTION)
    caller = RecordingCaller()
    collect_adjudication_outputs(case, caller)

    assert len(caller.requests) == 2
    for request in caller.requests:
        assert set(request.user_payload) == {"task_question", "trace", "answer"}
        serialized = repr(dict(request.user_payload))
        assert "above_good" not in serialized
        assert "30000000" not in serialized
        assert "final_good_side" not in serialized


def test_response_and_manifest_hashes_are_stable_and_provenance_order_independent() -> None:
    case = _case()
    outputs = _primary_outputs(case)
    left = materialize_primary_adjudication(case=case, external_outputs=outputs)
    right = materialize_primary_adjudication(case=case, external_outputs=outputs)
    assert outputs.final.response_hash == outputs.final.response_hash
    assert left.record_hash == right.record_hash
    assert left.to_dict(include_hash=True)["record_hash"] == left.record_hash


def test_primary_path_refuses_smoke_caller_and_smoke_outputs() -> None:
    caller = DeterministicSmokeCaller()
    case = _case()
    with pytest.raises(ValueError, match="not_for_primary_inference"):
        collect_adjudication_outputs(case, caller)

    smoke_outputs = collect_adjudication_outputs(case, caller, for_primary_inference=False)
    with pytest.raises(ValueError, match="not_for_primary_inference"):
        materialize_primary_adjudication(case=case, external_outputs=smoke_outputs)


def test_manual_override_is_source_linked_and_preserves_judge_values() -> None:
    case = _case()
    outputs = _primary_outputs(case)
    override = ManualOverrideAudit(
        reviewer_id="reviewer-01",
        rationale="The visible answer clearly supersedes the earlier tentative estimate.",
        blinded_case_hash=case.case_hash,
        source_final_response_hash=outputs.final.response_hash,
        source_trajectory_response_hash=outputs.trajectory.response_hash,
        final=FinalAdjudication(KnowledgeStatus.KNOWN, 30_000_000),
        trajectory=TrajectoryAdjudication(KnowledgeStatus.KNOWN, (29_250_000, 30_000_000)),
    )
    record = materialize_primary_adjudication(
        case=case, external_outputs=outputs, manual_override=override
    )
    assert record.judge_final.value == 29_250_000
    assert record.effective_final.value == 30_000_000
    assert record.to_dict()["manual_override"]["source_final_response_hash"] == (
        outputs.final.response_hash
    )


def test_final_agreement_audit_flags_discrepancies_and_unknowns() -> None:
    agree = audit_final_agreement(FinalAdjudication(KnowledgeStatus.KNOWN, 29), 29.0)
    disagree = audit_final_agreement(FinalAdjudication(KnowledgeStatus.KNOWN, 29), 30)
    unknown = audit_final_agreement(FinalAdjudication(KnowledgeStatus.UNKNOWN, None), 29)
    assert agree.status is AgreementStatus.AGREE
    assert agree.requires_manual_review is False
    assert disagree.status is AgreementStatus.DISAGREE
    assert disagree.discrepancy == -1
    assert disagree.requires_manual_review is True
    assert unknown.status is AgreementStatus.JUDGE_UNKNOWN
    assert unknown.local_parser_value == 29
    assert unknown.requires_manual_review is True
