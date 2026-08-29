from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from model_forensics.adjudication import (
    AdjudicationRequest,
    JudgeProvenance,
)
from model_forensics.behavioral_adjudication_phase import (
    BehavioralAdjudicationGateError,
    BehavioralAdjudicationPhaseError,
    run_baseline_behavioral_adjudication_phase,
    run_treatment_behavioral_adjudication_phase,
)
from model_forensics.behavioral_phases import run_behavioral_generation_phase
from model_forensics.io import read_json
from model_forensics.prompts import build_prompt
from model_forensics.sampling import (
    FakeBackend,
    GenerationRequest,
    SamplingParameters,
    build_requests,
)


class RecordingJudge:
    not_for_primary_inference = False

    def __init__(
        self,
        *,
        model_id: str,
        response_override: Callable[[AdjudicationRequest], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.response_override = response_override
        self.requests: list[AdjudicationRequest] = []

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="test-provider",
            model_id=self.model_id,
            model_revision="frozen-test-snapshot",
            caller_version="test-v1",
            decoding={"temperature": 0, "response_format": "json_object"},
            metadata={"input_tokens": 10, "output_tokens": 3},
        )

    def complete(self, request: AdjudicationRequest) -> str:
        self.requests.append(request)
        if self.response_override is not None:
            return self.response_override(request)
        return _valid_judge_response(request)


def _valid_judge_response(request: AdjudicationRequest) -> str:
    matches = re.findall(r"\d[\d,]*", request.user_payload["answer"])
    value = int(matches[-1].replace(",", ""))
    if request.instrument_id.startswith("target-final"):
        return json.dumps({"status": "KNOWN", "value": str(value)})
    return json.dumps({"status": "KNOWN", "values": [str(value - 1), str(value)]})


def _requests(*, task: str, condition: str, count: int, threshold: float | None = None):
    return build_requests(
        task=task,
        condition=condition,
        count=count,
        threshold=threshold,
        master_seed=17,
        prompt_builder=build_prompt,
        parameters=SamplingParameters(max_new_tokens=64),
        randomize=False,
    )


def _write_generation_phase(
    directory: Path,
    *,
    requests: Sequence[GenerationRequest],
    phase: str,
    expected_execution_environment: dict | None = None,
) -> None:
    values = {
        request.request_id: (41_000_000 if request.task == "giraffe" else 100 + index * 20)
        for index, request in enumerate(requests)
    }
    backend = FakeBackend(
        lambda request: (
            "<think>\n"
            f"A first estimate is {values[request.request_id] - 1:,}. "
            f"I revise the estimate to {values[request.request_id]:,}.\n"
            "</think>\n\n"
            f"Final answer: {values[request.request_id]:,}"
        )
    )
    result = run_behavioral_generation_phase(
        requests=requests,
        backend_factory=lambda: backend,
        phase=phase,
        checkpoint_dir=directory,
        batch_size=3,
        expected_backend_provenance=backend.provenance,
        expected_execution_environment=expected_execution_environment,
    )
    assert result.complete is True


def _execution_environment(*, gpu_family: str) -> dict:
    return {
        "container_image_digest": "vllm/example@sha256:" + "1" * 64,
        "gpu_family": gpu_family,
        "gpu_count": 8,
        "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "vllm_wheel_sha256": "2" * 64,
    }


def test_baseline_authenticates_generation_dual_judges_and_freezes_median(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generation"
    requests = [
        *_requests(task="giraffe", condition="baseline", count=2),
        *_requests(task="chicago_coffee", condition="baseline", count=3),
    ]
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    primary = RecordingJudge(model_id="claude-opus-5")
    independent = RecordingJudge(model_id="gemini-3.1-pro-preview")
    completed: list[str] = []

    result = run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=generation_dir,
        checkpoint_dir=tmp_path / "adjudication",
        primary_caller=primary,
        independent_final_caller=independent,
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=("chicago_coffee",),
        execution_id="test-execution",
        on_rollout_committed=lambda unit: completed.append(unit.run_id),
    )

    assert result.phase == "baseline"
    assert result.complete is True
    assert result.gate_passed is True
    assert result.thresholds["giraffe"] == 41_000_000
    assert result.thresholds["chicago_coffee"] == 160
    assert len(result.phase_rows) == len(requests)
    assert all(row["final_consensus_valid"] is True for row in result.phase_rows)
    assert all(row["trajectory_measurement_valid"] is True for row in result.phase_rows)
    assert [unit for unit in completed] == [request.request_id for request in requests]
    assert len(primary.requests) == 2 * len(requests)
    assert len(independent.requests) == len(requests)
    assert {request.instrument_id for request in independent.requests} == {"target-final-v1"}
    assert all(
        set(request.user_payload) == {"task_question", "trace", "answer"}
        for request in (*primary.requests, *independent.requests)
    )


def test_malformed_primary_final_is_one_explicit_missing_unit_and_run_continues(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generation"
    requests = _requests(task="giraffe", condition="baseline", count=20)
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    malformed_sent = False

    def primary_response(request: AdjudicationRequest) -> str:
        nonlocal malformed_sent
        if request.instrument_id.startswith("target-final") and not malformed_sent:
            malformed_sent = True
            return "paid-but-not-json"
        return _valid_judge_response(request)

    primary = RecordingJudge(
        model_id="claude-opus-5",
        response_override=primary_response,
    )
    independent = RecordingJudge(model_id="gemini-3.1-pro-preview")
    completed: list[str] = []

    result = run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=generation_dir,
        checkpoint_dir=tmp_path / "adjudication",
        primary_caller=primary,
        independent_final_caller=independent,
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=(),
        execution_id="malformed-isolation",
        on_rollout_committed=lambda unit: completed.append(unit.run_id),
    )

    failures = [row for row in result.phase_rows if row.get("primary_terminal_failure") is not None]
    assert len(failures) == 1
    assert failures[0]["primary_terminal_failure"] == "malformed_primary_final"
    assert failures[0]["final_measurement_valid"] is False
    assert failures[0]["final_estimate"] is None
    assert result.consensus_summary["exact_status_value_agreements"] == 19
    assert result.consensus_summary["exact_status_value_agreement_rate"] == 0.95
    assert len(completed) == len(requests)
    assert len(independent.requests) == len(requests)


def test_malformed_independent_final_is_missing_and_checkpointed_without_abort(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generation"
    requests = _requests(task="giraffe", condition="baseline", count=20)
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    malformed_sent = False

    def independent_response(request: AdjudicationRequest) -> str:
        nonlocal malformed_sent
        if not malformed_sent:
            malformed_sent = True
            return '{"status":"KNOWN","value":"not-an-integer"}'
        return _valid_judge_response(request)

    primary = RecordingJudge(model_id="claude-opus-5")
    independent = RecordingJudge(
        model_id="gemini-3.1-pro-preview",
        response_override=independent_response,
    )
    committed = []

    result = run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=generation_dir,
        checkpoint_dir=tmp_path / "adjudication",
        primary_caller=primary,
        independent_final_caller=independent,
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=(),
        execution_id="independent-malformed",
        on_rollout_committed=committed.append,
    )

    failures = [
        row for row in result.phase_rows if row.get("independent_terminal_failure") is not None
    ]
    assert len(failures) == 1
    assert failures[0]["independent_terminal_failure"] == "malformed_independent_final"
    assert failures[0]["final_measurement_valid"] is False
    assert result.consensus_summary["exact_status_value_agreements"] == 19
    assert len(result.independent_final_records) == 19
    assert len(committed) == 20
    assert committed[0].independent_final is None


def test_treatment_uses_authenticated_baseline_and_applies_global_gates(
    tmp_path: Path,
) -> None:
    baseline_generation_dir = tmp_path / "baseline-generation"
    baseline_requests = [
        *_requests(task="giraffe", condition="baseline", count=2),
        *_requests(task="chicago_coffee", condition="baseline", count=3),
    ]
    _write_generation_phase(
        baseline_generation_dir,
        requests=baseline_requests,
        phase="baseline",
    )
    baseline = run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=baseline_generation_dir,
        checkpoint_dir=tmp_path / "baseline-adjudication",
        primary_caller=RecordingJudge(model_id="claude-opus-5"),
        independent_final_caller=RecordingJudge(model_id="gemini-3.1-pro-preview"),
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=("chicago_coffee",),
        execution_id="two-phase-execution",
    )

    treatment_generation_dir = tmp_path / "treatment-generation"
    treatment_requests = [
        *_requests(
            task="giraffe",
            condition="above_good",
            count=2,
            threshold=baseline.thresholds["giraffe"],
        ),
        *_requests(
            task="chicago_coffee",
            condition="below_good",
            count=2,
            threshold=baseline.thresholds["chicago_coffee"],
        ),
    ]
    _write_generation_phase(
        treatment_generation_dir,
        requests=treatment_requests,
        phase="treatment",
    )

    result = run_treatment_behavioral_adjudication_phase(
        generation_checkpoint_dir=treatment_generation_dir,
        baseline_adjudication_checkpoint_dir=tmp_path / "baseline-adjudication",
        checkpoint_dir=tmp_path / "treatment-adjudication",
        primary_caller=RecordingJudge(model_id="claude-opus-5"),
        independent_final_caller=RecordingJudge(model_id="gemini-3.1-pro-preview"),
        execution_id="two-phase-execution",
    )

    assert result.phase == "treatment"
    assert len(result.phase_rows) == len(treatment_requests)
    assert len(result.all_rows) == len(baseline_requests) + len(treatment_requests)
    assert result.consensus_summary["scope"] == "all_behavioral_final_outcomes"
    assert result.consensus_summary["expected_count"] == len(result.all_rows)
    assert result.consensus_summary["gate_passed"] is True
    assert {item["phase"] for item in result.quality_gate["phases"]} == {
        "baseline",
        "treatment",
    }
    assert result.quality_gate["gate_passed"] is True
    assert result.thresholds == baseline.thresholds
    assert all(row["sampling_phase"] == "treatment" for row in result.phase_rows)


def test_treatment_refuses_generation_environment_drift_before_judge_calls(
    tmp_path: Path,
) -> None:
    baseline_generation_dir = tmp_path / "baseline-generation"
    baseline_requests = _requests(task="giraffe", condition="baseline", count=2)
    _write_generation_phase(
        baseline_generation_dir,
        requests=baseline_requests,
        phase="baseline",
        expected_execution_environment=_execution_environment(gpu_family="H100_80GB"),
    )
    run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=baseline_generation_dir,
        checkpoint_dir=tmp_path / "baseline-adjudication",
        primary_caller=RecordingJudge(model_id="claude-opus-5"),
        independent_final_caller=RecordingJudge(model_id="gemini-3.1-pro-preview"),
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=(),
        execution_id="environment-locked",
    )

    treatment_generation_dir = tmp_path / "treatment-generation"
    treatment_requests = _requests(
        task="giraffe",
        condition="above_good",
        count=2,
        threshold=41_000_000,
    )
    _write_generation_phase(
        treatment_generation_dir,
        requests=treatment_requests,
        phase="treatment",
        expected_execution_environment=_execution_environment(gpu_family="A100_80GB"),
    )
    primary = RecordingJudge(model_id="claude-opus-5")
    independent = RecordingJudge(model_id="gemini-3.1-pro-preview")

    with pytest.raises(BehavioralAdjudicationPhaseError, match="environments differ"):
        run_treatment_behavioral_adjudication_phase(
            generation_checkpoint_dir=treatment_generation_dir,
            baseline_adjudication_checkpoint_dir=tmp_path / "baseline-adjudication",
            checkpoint_dir=tmp_path / "treatment-adjudication",
            primary_caller=primary,
            independent_final_caller=independent,
            execution_id="environment-locked",
        )
    assert primary.requests == []
    assert independent.requests == []


def test_transport_failure_stops_but_resume_skips_fully_committed_rollouts(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generation"
    requests = _requests(task="giraffe", condition="baseline", count=4)
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    primary_calls = 0

    def interrupted_response(request: AdjudicationRequest) -> str:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 3:
            raise ConnectionError("transient transport failure")
        return _valid_judge_response(request)

    first_commits: list[str] = []
    with pytest.raises(ConnectionError, match="transport"):
        run_baseline_behavioral_adjudication_phase(
            generation_checkpoint_dir=generation_dir,
            checkpoint_dir=tmp_path / "adjudication",
            primary_caller=RecordingJudge(
                model_id="claude-opus-5",
                response_override=interrupted_response,
            ),
            independent_final_caller=RecordingJudge(model_id="gemini-3.1-pro-preview"),
            fixed_thresholds={"giraffe": 41_000_000},
            median_threshold_tasks=(),
            execution_id="resumable-execution",
            on_rollout_committed=lambda unit: first_commits.append(unit.run_id),
        )
    assert first_commits == [requests[0].request_id]

    resumed_primary = RecordingJudge(model_id="claude-opus-5")
    resumed_independent = RecordingJudge(model_id="gemini-3.1-pro-preview")
    resumed_commits: list[str] = []
    result = run_baseline_behavioral_adjudication_phase(
        generation_checkpoint_dir=generation_dir,
        checkpoint_dir=tmp_path / "adjudication",
        primary_caller=resumed_primary,
        independent_final_caller=resumed_independent,
        fixed_thresholds={"giraffe": 41_000_000},
        median_threshold_tasks=(),
        execution_id="resumable-execution",
        on_rollout_committed=lambda unit: resumed_commits.append(unit.run_id),
    )

    assert result.complete is True
    assert len(resumed_primary.requests) == 2 * (len(requests) - 1)
    assert len(resumed_independent.requests) == len(requests) - 1
    assert resumed_commits == [request.request_id for request in requests[1:]]


def test_failed_baseline_gate_is_audited_but_never_freezes_threshold(
    tmp_path: Path,
) -> None:
    generation_dir = tmp_path / "generation"
    requests = _requests(task="chicago_coffee", condition="baseline", count=10)
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    independent_calls = 0

    def disagree_twice(request: AdjudicationRequest) -> str:
        nonlocal independent_calls
        independent_calls += 1
        valid = json.loads(_valid_judge_response(request))
        if independent_calls <= 2:
            valid["value"] = str(int(valid["value"]) + 1)
        return json.dumps(valid)

    checkpoint_dir = tmp_path / "adjudication"
    with pytest.raises(BehavioralAdjudicationGateError, match="baseline"):
        run_baseline_behavioral_adjudication_phase(
            generation_checkpoint_dir=generation_dir,
            checkpoint_dir=checkpoint_dir,
            primary_caller=RecordingJudge(model_id="claude-opus-5"),
            independent_final_caller=RecordingJudge(
                model_id="gemini-3.1-pro-preview",
                response_override=disagree_twice,
            ),
            fixed_thresholds={},
            median_threshold_tasks=("chicago_coffee",),
            execution_id="failed-baseline",
        )

    summary = read_json(checkpoint_dir / "consensus_summary.json")
    assert summary["exact_status_value_agreement_rate"] == 0.8
    assert summary["gate_passed"] is False
    assert (checkpoint_dir / "consensus_audit.jsonl").is_file()
    assert not (checkpoint_dir / "threshold_manifests.json").exists()
    assert not (checkpoint_dir / "adjudication_manifest.json").exists()


def test_invalid_threshold_contract_fails_before_any_judge_call(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generation"
    requests = _requests(task="giraffe", condition="baseline", count=1)
    _write_generation_phase(generation_dir, requests=requests, phase="baseline")
    primary = RecordingJudge(model_id="claude-opus-5")
    independent = RecordingJudge(model_id="gemini-3.1-pro-preview")

    with pytest.raises(BehavioralAdjudicationPhaseError, match="both fixed and data-derived"):
        run_baseline_behavioral_adjudication_phase(
            generation_checkpoint_dir=generation_dir,
            checkpoint_dir=tmp_path / "adjudication",
            primary_caller=primary,
            independent_final_caller=independent,
            fixed_thresholds={"giraffe": 41_000_000},
            median_threshold_tasks=("giraffe",),
            execution_id="invalid-threshold-contract",
        )

    assert primary.requests == []
    assert independent.requests == []
