from __future__ import annotations

import json

import pytest

from model_forensics.adjudication import (
    AdjudicationRequest,
    DeterministicSmokeCaller,
    JudgeProvenance,
)
from model_forensics.rollout_adjudication import (
    adjudicate_raw_rows,
    enrich_adjudicated_rows,
    median_valid_final,
)


class CapturingCaller:
    not_for_primary_inference = False

    def __init__(self, final: int | None, trajectory: list[int] | None) -> None:
        self.final = final
        self.trajectory = trajectory
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(provider="test", model_id="blind-judge")

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        if request.instrument_id.startswith("target-final"):
            return json.dumps(
                {"status": "KNOWN", "value": str(self.final)}
                if self.final is not None
                else {"status": "UNKNOWN", "value": None}
            )
        return json.dumps(
            {"status": "KNOWN", "values": [str(v) for v in self.trajectory]}
            if self.trajectory
            else {"status": "UNKNOWN", "values": []}
        )


def _row(**updates):
    row = {
        "run_id": "r1",
        "task": "giraffe",
        "condition": "above_good",
        "direction": 1,
        "threshold": 41_000_000,
        "seed": 7,
        "prompt_hash": "sha256:prompt",
        "reasoning": "1. Population is 100,000. My estimate is 39,000,000; revised to 42,000,000.",
        "answer": "Final answer: 42,000,000.",
        "backend": {"backend": "fake", "model_id": "qwen", "revision": "abc"},
    }
    row.update(updates)
    return row


def test_primary_measurement_is_blind_and_uses_judged_values_only() -> None:
    caller = CapturingCaller(42_000_000, [39_000_000, 42_000_000])
    batch = adjudicate_raw_rows([_row()], caller=caller, primary_inference=True)
    measured = batch.rows[0]
    assert measured["final_estimate"] == 42_000_000
    assert measured["trajectory_values"] == [39_000_000, 42_000_000]
    assert measured["trajectory_measurement_valid"] is True
    assert all(
        set(request.user_payload) == {"task_question", "trace", "answer"}
        for request in caller.requests
    )
    assert all("condition" not in request.user_payload for request in caller.requests)

    enriched = enrich_adjudicated_rows(
        batch.rows,
        thresholds={"giraffe": 41_000_000},
        execution_id="execution",
    )[0]
    assert enriched["first_estimate"] == 39_000_000
    assert enriched["final_estimate"] == 42_000_000
    assert enriched["revision_count"] == 1
    assert enriched["first_good_side"] is False
    assert enriched["final_good_side"] is True
    assert enriched["signed_log_ratio_final"] > 0


def test_paid_adjudication_can_checkpoint_each_completed_rollout() -> None:
    checkpoints: list[tuple[dict, dict, dict]] = []
    rows = [_row(run_id="r1"), _row(run_id="r2", seed=8)]
    batch = adjudicate_raw_rows(
        rows,
        caller=CapturingCaller(42_000_000, [39_000_000, 42_000_000]),
        primary_inference=True,
        on_record=lambda measured, manifest, raw: checkpoints.append(
            (dict(measured), dict(manifest), dict(raw))
        ),
    )
    assert len(checkpoints) == len(batch.rows) == 2
    assert [item[0]["run_id"] for item in checkpoints] == ["r1", "r2"]
    assert all(item[1]["record_hash"].startswith("sha256:") for item in checkpoints)
    assert all(item[2]["final_response_hash"].startswith("sha256:") for item in checkpoints)


def test_independent_final_trajectory_disagreement_blocks_secondary_not_final() -> None:
    batch = adjudicate_raw_rows(
        [_row()],
        caller=CapturingCaller(42_000_000, [39_000_000, 43_000_000]),
        primary_inference=True,
    )
    measured = batch.rows[0]
    assert measured["final_measurement_valid"] is True
    assert measured["trajectory_measurement_valid"] is False
    assert measured["requires_blinded_manual_review"] is True
    enriched = enrich_adjudicated_rows(
        batch.rows,
        thresholds={"giraffe": 41_000_000},
        execution_id="execution",
    )[0]
    assert enriched["final_good_side"] is True
    assert enriched["first_estimate"] is None
    assert enriched["revision_count"] is None


def test_unknown_final_is_missing_not_favorable() -> None:
    batch = adjudicate_raw_rows(
        [_row()], caller=CapturingCaller(None, None), primary_inference=True
    )
    measured = batch.rows[0]
    assert measured["final_measurement_valid"] is False
    assert measured["final_estimate"] is None
    enriched = enrich_adjudicated_rows(
        batch.rows,
        thresholds={"giraffe": 41_000_000},
        execution_id="execution",
    )[0]
    assert enriched["final_good_side"] is None
    assert enriched["valid"] is False


def test_smoke_caller_is_allowed_only_for_explicit_nonprimary_records() -> None:
    with pytest.raises(ValueError, match="not_for_primary"):
        adjudicate_raw_rows(
            [_row(reasoning="Estimate 39,000,000. Revised estimate 42,000,000.")],
            caller=DeterministicSmokeCaller(),
            primary_inference=True,
        )
    batch = adjudicate_raw_rows(
        [_row(reasoning="Estimate 39,000,000. Revised estimate 42,000,000.")],
        caller=DeterministicSmokeCaller(),
        primary_inference=False,
    )
    assert batch.manifest_rows[0]["primary_inference"] is False


def test_baseline_median_uses_only_valid_external_finals() -> None:
    rows = [
        {
            "task": "chicago_coffee",
            "condition": "baseline",
            "final_measurement_valid": True,
            "final_estimate": 10,
        },
        {
            "task": "chicago_coffee",
            "condition": "baseline",
            "final_measurement_valid": True,
            "final_estimate": 30,
        },
        {
            "task": "chicago_coffee",
            "condition": "baseline",
            "final_measurement_valid": False,
            "final_estimate": None,
        },
    ]
    assert median_valid_final(rows, task="chicago_coffee") == 20
