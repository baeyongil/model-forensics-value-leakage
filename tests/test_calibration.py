from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    TRAJECTORY_INSTRUMENT,
    AdjudicationRequest,
    BlindedAdjudicationCase,
    JudgeProvenance,
    build_adjudication_request,
    parse_final_adjudication,
)
from model_forensics.calibration import (
    ConsensusGateError,
    DualFinalConsensusCaller,
    FinalOnlyCheckpoint,
    FinalOnlyJudgment,
    QualityGateError,
    apply_all_final_consensus,
    collect_independent_final_judgments,
    evaluate_adjudication_quality,
    freeze_consensus_baseline_threshold,
)
from model_forensics.io import read_json, read_jsonl, sha256_file, stable_hash
from model_forensics.prompts import QUESTIONS, Task


class FinalOnlyFakeCaller:
    not_for_primary_inference = False

    def __init__(
        self,
        values: dict[str, int | None],
        *,
        model_id: str = "independent-frontier-judge",
    ) -> None:
        self._values = values
        self._model_id = model_id
        self.requests: list[AdjudicationRequest] = []
        self._calls = 0

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="fake-provider",
            model_id=self._model_id,
            model_revision="route-snapshot",
            caller_version="fake-v1",
            decoding={"temperature": 0, "response_format": "json_object"},
            metadata={
                "input_tokens": 100 + self._calls,
                "output_tokens": 7,
                "charged_cost_usd": 0.001,
                "api_key": "must-never-be-written",
            },
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        self._calls += 1
        answer = request.user_payload["answer"]
        value = self._values[answer]
        if value is None:
            return json.dumps({"status": "UNKNOWN", "value": None})
        return json.dumps({"status": "KNOWN", "value": str(value)})


def _behavior_row(index: int, *, value: int | None = None, condition: str = "baseline"):
    final = 100 + index if value is None else value
    return {
        "run_id": f"run-{index:03d}",
        "task": "chicago_coffee",
        "condition": condition,
        "direction": 0 if condition == "baseline" else 1,
        "seed": index,
        "prompt_hash": stable_hash({"prompt": index}),
        "reasoning": f"Intermediate factor 20. My target estimate is {final}.",
        "answer": f"Final answer: {final}",
        "final_estimate": final,
        "final_measurement_valid": True,
        "trajectory_values": [final - 1, final],
        "trajectory_measurement_valid": True,
        "adjudication_primary_inference": True,
        "valid": True,
        "record_hash": stable_hash({"old": index}),
    }


def _judgment(
    unit: str | dict,
    value: int | None,
    *,
    suffix: str = "",
) -> FinalOnlyJudgment:
    if isinstance(unit, dict):
        unit_id = str(unit.get("run_id", unit.get("resample_id")))
        trace = str(unit.get("trace", unit.get("reasoning", unit.get("full_trace", ""))))
        case_hash = BlindedAdjudicationCase(
            QUESTIONS[Task(str(unit["task"]))], trace, str(unit.get("answer", ""))
        ).case_hash
    else:
        unit_id = unit
        case_hash = stable_hash({"case": unit_id})
    status = "UNKNOWN" if value is None else "KNOWN"
    raw = json.dumps({"status": status, "value": None if value is None else str(value)})
    return FinalOnlyJudgment(
        unit_id=unit_id,
        case_hash=case_hash,
        request_id=stable_hash({"request": unit_id}),
        instrument_hash=stable_hash("instrument"),
        raw_response=raw,
        status=status,
        value=value,
        public_provenance={
            "provider": "fake",
            "model_id": f"judge{suffix}",
            "model_revision": "snapshot",
            "caller_version": "test",
            "decoding": {"temperature": 0},
        },
        usage={"input_tokens": 10, "output_tokens": 2, "charged_cost_usd": 0.001},
    )


def test_collects_only_blinded_final_and_checkpoints_secret_safe_hashes(tmp_path: Path) -> None:
    rows = [_behavior_row(0), _behavior_row(1)]
    values = {row["answer"]: int(row["final_estimate"]) for row in rows}
    caller = FinalOnlyFakeCaller(values)
    checkpoint = FinalOnlyCheckpoint(tmp_path / "checkpoint")

    batch = collect_independent_final_judgments(
        rows,
        caller=caller,
        on_judgment=checkpoint.append,
    )

    assert len(batch.records) == 2
    assert all(request.instrument_id == "target-final-v1" for request in caller.requests)
    assert all(
        set(request.user_payload) == {"task_question", "trace", "answer"}
        for request in caller.requests
    )
    assert all("condition" not in request.user_payload for request in caller.requests)

    raw_path = checkpoint.directory / "independent_final_raw.jsonl"
    usage_path = checkpoint.directory / "independent_final_usage.jsonl"
    manifest_path = checkpoint.directory / "independent_final_manifest.jsonl"
    checkpoint_manifest = read_json(checkpoint.directory / "checkpoint_manifest.json")
    assert len(read_jsonl(raw_path)) == 2
    assert len(read_jsonl(usage_path)) == 2
    assert len(read_jsonl(manifest_path)) == 2
    assert checkpoint_manifest["completed_count"] == 2
    assert checkpoint_manifest["artifacts"]["raw"]["sha256"] == sha256_file(raw_path)
    assert "must-never-be-written" not in repr(read_jsonl(usage_path))
    assert "must-never-be-written" not in repr(
        read_json(checkpoint.directory / "checkpoint_manifest.json")
    )
    assert all(row["usage_hash"].startswith("sha256:") for row in read_jsonl(usage_path))
    assert all(row["response_hash"].startswith("sha256:") for row in read_jsonl(raw_path))


def test_all_final_exact_consensus_clears_every_non_known_consensus() -> None:
    rows = [_behavior_row(index) for index in range(4)]
    rows[2]["final_estimate"] = None
    rows[2]["final_measurement_valid"] = False
    rows[2]["valid"] = False
    independent = [
        _judgment(rows[0], 100),
        _judgment(rows[1], 999),
        _judgment(rows[2], None),
        _judgment(rows[3], None),
    ]

    result = apply_all_final_consensus(
        rows,
        independent,
        minimum_exact_agreement=0.50,
    )

    assert result.summary["scope"] == "all_final_outcomes"
    assert result.summary["exact_status_value_agreements"] == 2
    assert result.summary["exact_status_value_agreement_rate"] == pytest.approx(0.5)
    assert result.summary["known_consensus_count"] == 1
    assert result.summary["gate_passed"] is True
    assert result.rows[0]["final_estimate"] == 100
    for row in result.rows[1:]:
        assert row["final_estimate"] is None
        assert row["final_measurement_valid"] is False
        assert row["trajectory_measurement_valid"] is False
        assert row["valid"] is False
    assert result.rows[2]["final_consensus"]["exact_status_value_agreement"] is True
    assert result.rows[2]["final_consensus"]["known_consensus"] is False
    assert result.rows[2]["invalid_reason"] == "independent_final_unknown"


def test_consensus_gate_passes_at_ninety_percent_and_fails_below() -> None:
    rows = [_behavior_row(index) for index in range(10)]
    nine = [
        _judgment(row, int(row["final_estimate"]) + int(index == 9))
        for index, row in enumerate(rows)
    ]
    passed = apply_all_final_consensus(rows, nine, minimum_exact_agreement=0.90)
    assert passed.summary["exact_status_value_agreement_rate"] == pytest.approx(0.90)
    assert passed.summary["gate_passed"] is True

    eight = [
        _judgment(row, int(row["final_estimate"]) + int(index >= 8))
        for index, row in enumerate(rows)
    ]
    with pytest.raises(ConsensusGateError, match=r"0\.800"):
        apply_all_final_consensus(rows, eight, minimum_exact_agreement=0.90)


def test_missing_independent_all_final_row_counts_against_fail_closed_gate() -> None:
    rows = [_behavior_row(index) for index in range(10)]
    independent = [_judgment(row, int(row["final_estimate"])) for row in rows[:8]]
    result = apply_all_final_consensus(
        rows,
        independent,
        minimum_exact_agreement=0.90,
        enforce_gate=False,
    )
    assert result.summary["missing_independent_count"] == 2
    assert result.summary["gate_passed"] is False
    assert result.rows[-1]["invalid_reason"] == "missing_independent_final"


def test_quality_gates_are_separate_for_baseline_and_treatment_and_inclusive() -> None:
    rows = []
    for offset, condition in ((0, "baseline"), (100, "above_good")):
        for index in range(20):
            row = _behavior_row(offset + index, condition=condition)
            row["final_consensus_valid"] = True
            row["final_consensus"] = {
                "known_consensus": True,
                "exact_status_value_agreement": True,
            }
            rows.append(row)
    # Exactly one missing final/trajectory gives both rates 19/20 = .95.
    rows[0]["final_consensus_valid"] = False
    rows[0]["final_estimate"] = None
    rows[0]["final_measurement_valid"] = False
    rows[0]["trajectory_measurement_valid"] = False

    report = evaluate_adjudication_quality(
        rows,
        minimum_final_known_rate=0.95,
        minimum_trajectory_final_consistency=0.95,
    )
    assert report["gate_passed"] is True
    baseline = next(item for item in report["phases"] if item["phase"] == "baseline")
    assert baseline["final_known_rate"] == pytest.approx(0.95)
    assert baseline["trajectory_final_consistency_rate"] == pytest.approx(0.95)

    rows[1]["trajectory_values"] = [2, 998]
    rows[1]["trajectory_measurement_valid"] = False
    with pytest.raises(QualityGateError, match="baseline"):
        evaluate_adjudication_quality(
            rows,
            minimum_final_known_rate=0.95,
            minimum_trajectory_final_consistency=0.95,
            enforce=True,
        )


def test_task_condition_cell_gate_prevents_aggregate_masking() -> None:
    rows = []
    for offset, (task, condition) in enumerate(
        (("giraffe", "above_good"), ("chicago_coffee", "below_good"))
    ):
        for index in range(20):
            row = _behavior_row(offset * 100 + index, condition=condition)
            row["task"] = task
            row["final_consensus_valid"] = True
            row["final_consensus"] = {
                "known_consensus": True,
                "exact_status_value_agreement": True,
            }
            rows.append(row)
    # The treatment aggregate is exactly 95%, but both missing units are
    # concentrated in one 20-row scientific contrast cell (90%).
    for row in rows[:2]:
        row["final_consensus_valid"] = False
        row["final_estimate"] = None
        row["final_measurement_valid"] = False
        row["trajectory_measurement_valid"] = False

    report = evaluate_adjudication_quality(
        rows,
        required_phases=("treatment",),
        minimum_final_known_rate=0.95,
        minimum_trajectory_final_consistency=0.95,
    )

    assert report["phases"][0]["final_known_rate"] == pytest.approx(0.95)
    assert report["phases"][0]["gate_passed"] is True
    assert report["cell_scope"] == "task_x_condition"
    by_cell = {(row["task"], row["condition"]): row for row in report["cells"]}
    assert by_cell[("giraffe", "above_good")]["final_known_rate"] == pytest.approx(0.90)
    assert by_cell[("giraffe", "above_good")]["gate_passed"] is False
    assert by_cell[("chicago_coffee", "below_good")]["gate_passed"] is True
    assert report["gate_passed"] is False


def test_chicago_threshold_freezes_only_after_consensus_and_baseline_gates() -> None:
    rows = []
    for index in range(30):
        row = _behavior_row(index, value=index + 1)
        row["final_consensus_valid"] = True
        row["final_consensus"] = {
            "known_consensus": True,
            "exact_status_value_agreement": True,
        }
        rows.append(row)

    frozen = freeze_consensus_baseline_threshold(rows, task="chicago_coffee")
    assert frozen["threshold"] == pytest.approx(15.5)
    assert frozen["source_count"] == 30
    assert frozen["quality_gate"]["gate_passed"] is True
    assert frozen["manifest_hash"] == stable_hash(
        {key: value for key, value in frozen.items() if key != "manifest_hash"}
    )

    rows[0].pop("final_consensus_valid")
    rows[0].pop("final_consensus")
    with pytest.raises(QualityGateError, match="consensus"):
        freeze_consensus_baseline_threshold(rows, task="chicago_coffee")


def test_resampling_consensus_clears_confirmatory_eligibility() -> None:
    row = {
        "resample_id": "resample-1",
        "task": "giraffe",
        "condition": "above_good",
        "threshold": 100,
        "final_estimate": 120,
        "final_good_side": True,
        "final_measurement_valid": True,
        "intervention_eligible": True,
        "primary_eligible": True,
        "confirmatory_eligible": True,
        "analysis_tier": "confirmatory",
        "outcome_adjudication_primary_inference": True,
        "outcome_adjudication": {"status": "KNOWN", "value": 120},
    }
    result = apply_all_final_consensus(
        [row],
        [_judgment("resample-1", 121)],
        minimum_exact_agreement=0,
        id_field="resample_id",
    )
    calibrated = result.rows[0]
    assert calibrated["final_estimate"] is None
    assert calibrated["final_good_side"] is None
    assert calibrated["intervention_eligible"] is True
    assert calibrated["primary_eligible"] is False
    assert calibrated["confirmatory_eligible"] is False
    assert calibrated["analysis_tier"] == "outcome_unmeasured"
    assert calibrated["outcome_adjudication"]["value"] == 120


def test_dual_final_caller_returns_known_only_on_exact_known_consensus() -> None:
    case_a = BlindedAdjudicationCase("How many?", "Estimate 42.", "answer-a")
    case_b = BlindedAdjudicationCase("How many?", "Estimate 43.", "answer-b")
    primary = FinalOnlyFakeCaller(
        {"answer-a": 42, "answer-b": 43},
        model_id="primary-opus",
    )
    independent = FinalOnlyFakeCaller(
        {"answer-a": 42, "answer-b": 44},
        model_id="independent-gemini",
    )
    audits: list[dict] = []
    caller = DualFinalConsensusCaller(
        primary,
        independent,
        minimum_exact_agreement=0.50,
        minimum_known_consensus_rate=0.50,
        on_audit=lambda audit, _primary, _independent: audits.append(dict(audit)),
    )

    agreed = caller.complete(build_adjudication_request(case_a, FINAL_ANSWER_INSTRUMENT))
    disagreed = caller.complete(build_adjudication_request(case_b, FINAL_ANSWER_INSTRUMENT))

    assert parse_final_adjudication(agreed).value == 42
    assert parse_final_adjudication(disagreed).value is None
    assert len(primary.requests) == len(independent.requests) == len(audits) == 2
    summary = caller.require_quality_gates(expected_count=2)
    assert summary["exact_status_value_agreement_rate"] == pytest.approx(0.5)
    assert summary["known_consensus_rate"] == pytest.approx(0.5)
    assert caller.provenance.metadata["trajectory_scope"].startswith("primary_route_only")
    with pytest.raises(ValueError, match="final instrument"):
        caller.complete(build_adjudication_request(case_a, TRAJECTORY_INSTRUMENT))
