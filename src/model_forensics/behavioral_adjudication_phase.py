"""CPU/API-only behavioral adjudication phases.

The public entry points authenticate completed GPU generation artifacts before
presenting any blinded case to injected, provider-neutral judge callers.  The
primary route measures the final and trajectory; a distinct independent route
measures every final.  Only exact known final consensus is analysis-usable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    FINAL_INSTRUMENT_ID,
    TRAJECTORY_INSTRUMENT_ID,
    AdjudicationCaller,
    AdjudicationRequest,
    AdjudicationValidationError,
    BlindedAdjudicationCase,
    build_adjudication_request,
    parse_final_adjudication,
    parse_trajectory_adjudication,
)
from model_forensics.behavioral_phases import (
    BehavioralPhaseError,
    load_behavioral_generation_phase,
    validate_behavioral_generation_environment_identity,
)
from model_forensics.calibration import (
    FinalOnlyJudgment,
    apply_all_final_consensus,
    evaluate_adjudication_quality,
    freeze_consensus_baseline_threshold,
)
from model_forensics.io import (
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)
from model_forensics.prompts import QUESTIONS, Task
from model_forensics.rollout_adjudication import (
    adjudicate_raw_rows,
    enrich_adjudicated_rows,
)

BEHAVIORAL_ADJUDICATION_PROTOCOL = "behavioral-cpu-adjudication-v1"


class BehavioralAdjudicationPhaseError(RuntimeError):
    """A CPU adjudication phase cannot safely continue or enter analysis."""


class BehavioralAdjudicationGateError(BehavioralAdjudicationPhaseError):
    """Completed adjudication failed a preregistered measurement-quality gate."""


class _TerminalMalformedIsolatingCaller:
    """Replace only malformed completed responses with audited UNKNOWN sentinels.

    Transport, budget, checkpoint-integrity, and arbitrary provider exceptions
    are deliberately not caught: those failures stop the phase so a resume can
    retry safely.  Only a response body that was returned but violates a frozen
    instrument contract is terminal for that measurement unit.
    """

    def __init__(self, caller: AdjudicationCaller) -> None:
        self._caller = caller
        self.raw_by_instrument: dict[str, str] = {}
        self.failures: dict[str, str] = {}

    @property
    def not_for_primary_inference(self) -> bool:
        return self._caller.not_for_primary_inference

    @property
    def provenance(self):  # type: ignore[no-untyped-def]
        return self._caller.provenance

    def complete(self, request: AdjudicationRequest) -> str:
        raw = self._caller.complete(request)
        self.raw_by_instrument[request.instrument_id] = raw
        try:
            if request.instrument_id == FINAL_INSTRUMENT_ID:
                parse_final_adjudication(raw)
            elif request.instrument_id == TRAJECTORY_INSTRUMENT_ID:
                parse_trajectory_adjudication(raw)
            else:  # pragma: no cover - guarded by the upstream adjudication helper
                raise ValueError(f"unknown adjudication instrument: {request.instrument_id}")
        except AdjudicationValidationError:
            self.failures[request.instrument_id] = "malformed_judge_response"
            if request.instrument_id == FINAL_INSTRUMENT_ID:
                return '{"status":"UNKNOWN","value":null}'
            return '{"status":"UNKNOWN","values":[]}'
        return raw


def _apply_primary_terminal_failure_audit(
    measured: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_row: Mapping[str, Any],
    caller: _TerminalMalformedIsolatingCaller,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Restore paid raw bodies and mark surrogate UNKNOWNs as contract failures."""

    row = dict(measured)
    manifest_row = dict(manifest)
    raw = dict(raw_row)
    failure_names: list[str] = []
    for instrument_id, label in (
        (FINAL_INSTRUMENT_ID, "final"),
        (TRAJECTORY_INSTRUMENT_ID, "trajectory"),
    ):
        actual = caller.raw_by_instrument.get(instrument_id)
        if actual is None:  # pragma: no cover - upstream helper always makes both calls
            raise BehavioralAdjudicationPhaseError(f"primary caller omitted the {label} response")
        raw[f"{label}_response"] = actual
        raw[f"{label}_response_hash"] = stable_hash({"raw_response": actual})
        if instrument_id not in caller.failures:
            continue
        failure_names.append(f"malformed_primary_{label}")
        instrument = dict(manifest_row[f"{label}_instrument"])
        instrument["response_hash"] = raw[f"{label}_response_hash"]
        instrument["response_contract_valid"] = False
        manifest_row[f"{label}_instrument"] = instrument
        if label == "final":
            manifest_row["judge_final"] = {"status": "MALFORMED", "value": None}
            manifest_row["effective_final"] = {"status": "MALFORMED", "value": None}
        else:
            manifest_row["judge_trajectory"] = {"status": "MALFORMED", "values": []}
            manifest_row["effective_trajectory"] = {"status": "MALFORMED", "values": []}

    if failure_names:
        row.pop("record_hash", None)
        row["primary_contract_failures"] = failure_names
        row["primary_terminal_failure"] = (
            failure_names[0] if len(failure_names) == 1 else "multiple_malformed_primary_responses"
        )
        row["record_hash"] = stable_hash(row)
        manifest_row.pop("record_hash", None)
        manifest_row["terminal_contract_failures"] = failure_names
        manifest_row["scientific_disposition"] = "explicit_missing_no_parser_fallback"
        manifest_row["record_hash"] = stable_hash(manifest_row)
        raw.pop("record_hash", None)
        raw["terminal_contract_failures"] = failure_names
        raw["record_hash"] = stable_hash(raw)
    return row, manifest_row, raw


@dataclass(frozen=True, slots=True)
class BehavioralAdjudicationUnit:
    """One fully checkpointed rollout's primary and independent measurements."""

    run_id: str
    measured_row: Mapping[str, Any]
    primary_manifest: Mapping[str, Any]
    primary_raw: Mapping[str, Any]
    independent_final: FinalOnlyJudgment | None
    independent_raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BehavioralAdjudicationPhase:
    phase: str
    phase_rows: tuple[dict[str, Any], ...]
    all_rows: tuple[dict[str, Any], ...]
    primary_rows: tuple[dict[str, Any], ...]
    primary_manifest_rows: tuple[dict[str, Any], ...]
    primary_raw_rows: tuple[dict[str, Any], ...]
    independent_final_records: tuple[FinalOnlyJudgment, ...]
    consensus_audit_rows: tuple[dict[str, Any], ...]
    consensus_summary: Mapping[str, Any]
    quality_gate: Mapping[str, Any]
    thresholds: Mapping[str, float]
    threshold_manifests: Mapping[str, Mapping[str, Any]]
    manifest: Mapping[str, Any]
    complete: bool
    gate_passed: bool


@dataclass(frozen=True, slots=True)
class _PhaseMeasurements:
    primary_rows: tuple[dict[str, Any], ...]
    primary_manifests: tuple[dict[str, Any], ...]
    primary_raw: tuple[dict[str, Any], ...]
    independent_records: tuple[FinalOnlyJudgment, ...]
    consensus_independent_records: tuple[FinalOnlyJudgment, ...]


def _adjudication_plan_payload(
    *,
    phase: str,
    generation: Any,
    primary_caller: AdjudicationCaller,
    independent_final_caller: AdjudicationCaller,
    execution_id: str,
    minimum_exact_agreement: float,
    minimum_final_known_rate: float,
    minimum_trajectory_final_consistency: float,
    threshold_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_ADJUDICATION_PROTOCOL,
        "phase": phase,
        "execution_id": execution_id,
        "generation_plan_hash": generation.plan["plan_hash"],
        "generation_manifest_hash": generation.manifest["manifest_hash"],
        "generation_environment_identity_hash": generation.manifest[
            "shared_generation_environment_hash"
        ],
        "run_ids": [row["run_id"] for row in generation.rows],
        "source_record_hashes": [row["record_hash"] for row in generation.rows],
        "primary_route": _route_identity(primary_caller),
        "independent_final_route": _route_identity(independent_final_caller),
        "minimum_exact_agreement": float(minimum_exact_agreement),
        "minimum_final_known_rate": float(minimum_final_known_rate),
        "minimum_trajectory_final_consistency": float(minimum_trajectory_final_consistency),
        "threshold_contract": dict(threshold_contract),
    }
    payload["plan_hash"] = stable_hash(payload)
    return payload


def _freeze_adjudication_plan(directory: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = directory / "adjudication_plan.json"
    expected = dict(payload)
    if path.exists():
        observed = read_json(path)
        if not isinstance(observed, dict) or observed != expected:
            raise BehavioralAdjudicationPhaseError("frozen behavioral adjudication plan mismatch")
        return observed
    write_json(path, expected)
    return expected


def _record_hash_valid(row: Mapping[str, Any]) -> bool:
    return row.get("record_hash") == stable_hash(_without_hash(row, "record_hash"))


@dataclass(slots=True)
class _CheckpointState:
    primary_rows: list[dict[str, Any]]
    primary_manifests: list[dict[str, Any]]
    primary_raw: list[dict[str, Any]]
    independent_records: list[FinalOnlyJudgment]
    independent_manifests: list[dict[str, Any]]
    independent_raw: list[dict[str, Any]]
    independent_usage: list[dict[str, Any]]
    consensus_independent_records: list[FinalOnlyJudgment]


def _load_checkpoint_state(
    directory: Path,
    source_rows: Sequence[Mapping[str, Any]],
) -> _CheckpointState:
    paths = {
        "primary_rows": directory / "primary_rows.jsonl",
        "primary_manifests": directory / "primary_manifest.jsonl",
        "primary_raw": directory / "primary_raw.jsonl",
        "independent_manifests": directory / "independent_final_manifest.jsonl",
        "independent_raw": directory / "independent_final_raw.jsonl",
        "independent_usage": directory / "independent_final_usage.jsonl",
    }
    existing = {name for name, path in paths.items() if path.exists()}
    if not existing:
        return _CheckpointState([], [], [], [], [], [], [], [])
    if existing != set(paths):
        raise BehavioralAdjudicationPhaseError(
            "behavioral adjudication checkpoint bundle is incomplete"
        )
    loaded = {name: read_jsonl(path) for name, path in paths.items()}
    lengths = {len(rows) for rows in loaded.values()}
    if len(lengths) != 1:
        raise BehavioralAdjudicationPhaseError(
            "behavioral adjudication checkpoint lengths disagree"
        )
    completed = lengths.pop()
    if completed > len(source_rows):
        raise BehavioralAdjudicationPhaseError(
            "behavioral adjudication checkpoint exceeds source inventory"
        )

    records: list[FinalOnlyJudgment] = []
    consensus_records: list[FinalOnlyJudgment] = []
    for index in range(completed):
        source = source_rows[index]
        primary_row = loaded["primary_rows"][index]
        primary_manifest = loaded["primary_manifests"][index]
        primary_raw = loaded["primary_raw"][index]
        independent_manifest = loaded["independent_manifests"][index]
        independent_raw = loaded["independent_raw"][index]
        independent_usage = loaded["independent_usage"][index]
        entries = (
            primary_row,
            primary_manifest,
            primary_raw,
            independent_manifest,
            independent_raw,
            independent_usage,
        )
        if not all(_record_hash_valid(entry) for entry in entries):
            raise BehavioralAdjudicationPhaseError(
                f"behavioral adjudication checkpoint record hash mismatch at {index}"
            )
        run_id = str(source["run_id"])
        if (
            primary_row.get("run_id") != run_id
            or primary_manifest.get("run_id") != run_id
            or primary_raw.get("run_id") != run_id
            or independent_manifest.get("unit_id") != run_id
            or independent_raw.get("unit_id") != run_id
            or independent_usage.get("unit_id") != run_id
            or primary_row.get("source_generation_record_hash") != source.get("record_hash")
        ):
            raise BehavioralAdjudicationPhaseError(
                f"behavioral adjudication checkpoint source mismatch at {index}"
            )
        if independent_manifest.get("terminal_contract_failure") is not None:
            if independent_manifest.get("terminal_contract_failure") != (
                "malformed_independent_final"
            ):
                raise BehavioralAdjudicationPhaseError(
                    "unknown terminal independent-final checkpoint disposition"
                )
            continue
        try:
            record = FinalOnlyJudgment(
                unit_id=run_id,
                case_hash=str(independent_manifest["case_hash"]),
                request_id=str(independent_manifest["request_id"]),
                instrument_hash=str(independent_manifest["instrument_hash"]),
                raw_response=str(independent_raw["raw_response"]),
                status=str(independent_manifest["status"]),
                value=independent_manifest.get("value"),
                public_provenance=independent_manifest["public_provenance"],
                usage=independent_usage.get("usage", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BehavioralAdjudicationPhaseError(
                f"cannot reconstruct independent-final checkpoint at {index}"
            ) from exc
        if record.manifest_dict() != independent_manifest:
            raise BehavioralAdjudicationPhaseError(
                f"independent-final checkpoint content mismatch at {index}"
            )
        records.append(record)
        primary_failures = primary_row.get("primary_contract_failures", ())
        if "malformed_primary_final" not in primary_failures:
            consensus_records.append(record)

    return _CheckpointState(
        primary_rows=[dict(row) for row in loaded["primary_rows"]],
        primary_manifests=[dict(row) for row in loaded["primary_manifests"]],
        primary_raw=[dict(row) for row in loaded["primary_raw"]],
        independent_records=records,
        independent_manifests=[dict(row) for row in loaded["independent_manifests"]],
        independent_raw=[dict(row) for row in loaded["independent_raw"]],
        independent_usage=[dict(row) for row in loaded["independent_usage"]],
        consensus_independent_records=consensus_records,
    )


def _adjudicate_generation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    directory: Path,
    primary_caller: AdjudicationCaller,
    independent_final_caller: AdjudicationCaller,
    on_rollout_committed: Callable[[BehavioralAdjudicationUnit], None] | None,
) -> _PhaseMeasurements:
    state = _load_checkpoint_state(directory, rows)
    primary_rows = state.primary_rows
    primary_manifests = state.primary_manifests
    primary_raw = state.primary_raw
    independent_records = state.independent_records
    independent_manifests = state.independent_manifests
    independent_raw_rows = state.independent_raw
    independent_usage_rows = state.independent_usage
    consensus_independent_records = state.consensus_independent_records
    for source in rows[len(primary_rows) :]:
        isolating_primary = _TerminalMalformedIsolatingCaller(primary_caller)
        primary_batch = adjudicate_raw_rows(
            [source],
            caller=isolating_primary,
            primary_inference=True,
        )
        measured_row, primary_manifest, primary_raw_row = _apply_primary_terminal_failure_audit(
            primary_batch.rows[0],
            primary_batch.manifest_rows[0],
            primary_batch.raw_judge_rows[0],
            isolating_primary,
        )
        measured_row.pop("record_hash", None)
        measured_row["source_generation_record_hash"] = source["record_hash"]
        measured_row["record_hash"] = stable_hash(measured_row)
        (
            independent_record,
            independent_manifest,
            independent_raw,
            independent_usage,
        ) = _collect_independent_final_safely(
            measured_row,
            caller=independent_final_caller,
        )
        if independent_record is None:
            measured_row.pop("record_hash", None)
            measured_row["independent_terminal_failure"] = "malformed_independent_final"
            measured_row["record_hash"] = stable_hash(measured_row)
        primary_rows.append(measured_row)
        primary_manifests.append(primary_manifest)
        primary_raw.append(primary_raw_row)
        independent_manifests.append(independent_manifest)
        independent_raw_rows.append(independent_raw)
        independent_usage_rows.append(independent_usage)
        if independent_record is not None:
            independent_records.append(independent_record)
        if independent_record is not None and FINAL_INSTRUMENT_ID not in isolating_primary.failures:
            consensus_independent_records.append(independent_record)
        _checkpoint_completed_units(
            directory,
            primary_rows=primary_rows,
            primary_manifests=primary_manifests,
            primary_raw=primary_raw,
            independent_manifests=independent_manifests,
            independent_raw=independent_raw_rows,
            independent_usage=independent_usage_rows,
        )
        if on_rollout_committed is not None:
            on_rollout_committed(
                BehavioralAdjudicationUnit(
                    run_id=str(source["run_id"]),
                    measured_row=primary_rows[-1],
                    primary_manifest=primary_manifests[-1],
                    primary_raw=primary_raw[-1],
                    independent_final=independent_record,
                    independent_raw=independent_raw,
                )
            )
    return _PhaseMeasurements(
        primary_rows=tuple(primary_rows),
        primary_manifests=tuple(primary_manifests),
        primary_raw=tuple(primary_raw),
        independent_records=tuple(independent_records),
        consensus_independent_records=tuple(consensus_independent_records),
    )


def _route_identity(caller: AdjudicationCaller) -> dict[str, Any]:
    provenance = caller.provenance
    return {
        "provider": provenance.provider,
        "model_id": provenance.model_id,
        "model_revision": provenance.model_revision,
        "caller_version": provenance.caller_version,
    }


def _validate_callers(
    primary: AdjudicationCaller,
    independent: AdjudicationCaller,
) -> None:
    if primary.not_for_primary_inference or independent.not_for_primary_inference:
        raise BehavioralAdjudicationPhaseError(
            "primary behavioral adjudication refuses non-primary judge routes"
        )
    if _route_identity(primary) == _route_identity(independent):
        raise BehavioralAdjudicationPhaseError(
            "primary and independent final judges must be distinct routes"
        )


def _checkpoint_completed_units(
    directory: Path,
    *,
    primary_rows: Sequence[Mapping[str, Any]],
    primary_manifests: Sequence[Mapping[str, Any]],
    primary_raw: Sequence[Mapping[str, Any]],
    independent_manifests: Sequence[Mapping[str, Any]],
    independent_raw: Sequence[Mapping[str, Any]],
    independent_usage: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(directory / "primary_rows.jsonl", primary_rows)
    write_jsonl(directory / "primary_manifest.jsonl", primary_manifests)
    write_jsonl(directory / "primary_raw.jsonl", primary_raw)
    write_jsonl(
        directory / "independent_final_manifest.jsonl",
        independent_manifests,
    )
    write_jsonl(
        directory / "independent_final_raw.jsonl",
        independent_raw,
    )
    write_jsonl(
        directory / "independent_final_usage.jsonl",
        independent_usage,
    )


def _blinded_case(row: Mapping[str, Any]) -> BlindedAdjudicationCase:
    try:
        question = QUESTIONS[Task(str(row.get("task")))]
    except ValueError as exc:
        raise BehavioralAdjudicationPhaseError(
            f"unsupported behavioral task: {row.get('task')!r}"
        ) from exc
    trace = row.get("trace", row.get("reasoning", ""))
    answer = row.get("answer", "")
    if not isinstance(trace, str) or not isinstance(answer, str):
        raise BehavioralAdjudicationPhaseError("behavioral trace and answer must be strings")
    return BlindedAdjudicationCase(question, trace, answer)


def _collect_independent_final_safely(
    row: Mapping[str, Any],
    *,
    caller: AdjudicationCaller,
) -> tuple[FinalOnlyJudgment | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return a valid record, or a terminal malformed audit without synthesizing one."""

    unit_id = str(row.get("run_id", ""))
    if not unit_id:
        raise BehavioralAdjudicationPhaseError("every behavioral row requires run_id")
    case = _blinded_case(row)
    request = build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)
    raw_response = caller.complete(request)
    provenance = caller.provenance
    response_hash = stable_hash({"raw_response": raw_response})
    try:
        parsed = parse_final_adjudication(raw_response)
    except AdjudicationValidationError:
        manifest: dict[str, Any] = {
            "protocol_version": BEHAVIORAL_ADJUDICATION_PROTOCOL,
            "unit_id": unit_id,
            "case_hash": case.case_hash,
            "request_id": request.request_id,
            "instrument_hash": request.instrument_hash,
            "response_hash": response_hash,
            "status": "MALFORMED",
            "value": None,
            "public_provenance": _route_identity(caller),
            "scientific_disposition": "explicit_missing_no_parser_fallback",
            "terminal_contract_failure": "malformed_independent_final",
        }
        manifest["record_hash"] = stable_hash(manifest)
        raw = {
            "unit_id": unit_id,
            "case_hash": case.case_hash,
            "request_id": request.request_id,
            "instrument_hash": request.instrument_hash,
            "raw_response": raw_response,
            "response_hash": response_hash,
            "terminal_contract_failure": "malformed_independent_final",
        }
        raw["record_hash"] = stable_hash(raw)
        usage = {
            "unit_id": unit_id,
            "request_id": request.request_id,
            "public_provenance": _route_identity(caller),
            "usage": {},
            "terminal_contract_failure": "malformed_independent_final",
        }
        usage["record_hash"] = stable_hash(usage)
        return None, manifest, raw, usage

    record = FinalOnlyJudgment(
        unit_id=unit_id,
        case_hash=case.case_hash,
        request_id=request.request_id,
        instrument_hash=request.instrument_hash,
        raw_response=raw_response,
        status=parsed.status.value,
        value=parsed.value,
        public_provenance=provenance.to_dict(),
        usage=provenance.metadata,
    )
    return record, record.manifest_dict(), record.raw_dict(), record.usage_dict()


def _fixed_threshold_manifest(task: str, threshold: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_ADJUDICATION_PROTOCOL,
        "task": task,
        "threshold_rule": "fixed_external_reference",
        "threshold": threshold,
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def run_baseline_behavioral_adjudication_phase(
    *,
    generation_checkpoint_dir: str | Path,
    checkpoint_dir: str | Path,
    primary_caller: AdjudicationCaller,
    independent_final_caller: AdjudicationCaller,
    fixed_thresholds: Mapping[str, float],
    median_threshold_tasks: Sequence[str],
    execution_id: str,
    minimum_exact_agreement: float = 0.90,
    minimum_final_known_rate: float = 0.95,
    minimum_trajectory_final_consistency: float = 0.95,
    on_rollout_committed: Callable[[BehavioralAdjudicationUnit], None] | None = None,
) -> BehavioralAdjudicationPhase:
    """Adjudicate an authenticated baseline and then freeze its thresholds."""

    generation = load_behavioral_generation_phase(generation_checkpoint_dir)
    if generation.plan.get("phase") != "baseline" or any(
        row.get("condition") != "baseline" for row in generation.rows
    ):
        raise BehavioralAdjudicationPhaseError(
            "baseline adjudication requires an authenticated baseline generation phase"
        )
    if not execution_id:
        raise BehavioralAdjudicationPhaseError("execution_id must be nonempty")
    tasks = {str(row["task"]) for row in generation.rows}
    fixed = {str(task): float(value) for task, value in fixed_thresholds.items()}
    median_tasks = tuple(str(task) for task in median_threshold_tasks)
    if len(set(median_tasks)) != len(median_tasks) or any(not task for task in median_tasks):
        raise BehavioralAdjudicationPhaseError(
            "data-derived threshold tasks must be unique and nonempty"
        )
    if any(not math.isfinite(value) or value <= 0 for value in fixed.values()):
        raise BehavioralAdjudicationPhaseError("fixed thresholds must be positive and finite")
    if set(fixed).intersection(median_tasks):
        raise BehavioralAdjudicationPhaseError(
            "a task cannot have both fixed and data-derived threshold rules"
        )
    if tasks != set(fixed).union(median_tasks):
        raise BehavioralAdjudicationPhaseError(
            "threshold rules must cover exactly the authenticated baseline tasks"
        )
    _validate_callers(primary_caller, independent_final_caller)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _freeze_adjudication_plan(
        directory,
        _adjudication_plan_payload(
            phase="baseline",
            generation=generation,
            primary_caller=primary_caller,
            independent_final_caller=independent_final_caller,
            execution_id=execution_id,
            minimum_exact_agreement=minimum_exact_agreement,
            minimum_final_known_rate=minimum_final_known_rate,
            minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
            threshold_contract={
                "fixed_thresholds": {
                    str(task): float(value) for task, value in fixed_thresholds.items()
                },
                "median_threshold_tasks": [str(task) for task in median_threshold_tasks],
            },
        ),
    )

    measurements = _adjudicate_generation_rows(
        generation.rows,
        directory=directory,
        primary_caller=primary_caller,
        independent_final_caller=independent_final_caller,
        on_rollout_committed=on_rollout_committed,
    )

    consensus = apply_all_final_consensus(
        measurements.primary_rows,
        measurements.consensus_independent_records,
        minimum_exact_agreement=minimum_exact_agreement,
        enforce_gate=False,
    )
    quality = evaluate_adjudication_quality(
        consensus.rows,
        minimum_exact_agreement=minimum_exact_agreement,
        minimum_final_known_rate=minimum_final_known_rate,
        minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
        required_phases=("baseline",),
        enforce=False,
    )
    write_jsonl(directory / "consensus_audit.jsonl", consensus.audit_rows)
    write_json(directory / "consensus_summary.json", consensus.summary)
    write_json(directory / "quality_gate.json", quality)
    if not consensus.summary["gate_passed"] or not quality["gate_passed"]:
        raise BehavioralAdjudicationGateError(
            "baseline exact-consensus or external measurement-quality gate failed closed"
        )

    thresholds = dict(fixed)
    threshold_manifests: dict[str, Mapping[str, Any]] = {
        task: _fixed_threshold_manifest(task, value) for task, value in fixed.items()
    }
    for task in median_tasks:
        frozen = freeze_consensus_baseline_threshold(
            consensus.rows,
            task=task,
            minimum_final_known_rate=minimum_final_known_rate,
            minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
        )
        thresholds[task] = float(frozen["threshold"])
        threshold_manifests[task] = frozen

    phase_rows = enrich_adjudicated_rows(
        consensus.rows,
        thresholds=thresholds,
        execution_id=execution_id,
    )
    consensus_rows_path = write_jsonl(directory / "consensus_rows.jsonl", phase_rows)
    consensus_audit_path = write_jsonl(directory / "consensus_audit.jsonl", consensus.audit_rows)
    quality_path = write_json(directory / "quality_gate.json", quality)
    thresholds_path = write_json(
        directory / "threshold_manifests.json", dict(sorted(threshold_manifests.items()))
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_ADJUDICATION_PROTOCOL,
        "phase": "baseline",
        "complete": True,
        "execution_id": execution_id,
        "generation_plan_hash": generation.plan["plan_hash"],
        "generation_manifest_hash": generation.manifest["manifest_hash"],
        "generation_environment_identity": dict(
            generation.manifest["shared_generation_environment"]
        ),
        "generation_environment_identity_hash": generation.manifest[
            "shared_generation_environment_hash"
        ],
        "row_count": len(phase_rows),
        "run_ids_hash": stable_hash([row["run_id"] for row in phase_rows]),
        "phase_rows_hash": stable_hash(phase_rows),
        "primary_route": _route_identity(primary_caller),
        "independent_final_route": _route_identity(independent_final_caller),
        "consensus_summary": dict(consensus.summary),
        "quality_gate": dict(quality),
        "thresholds": dict(sorted(thresholds.items())),
        "artifacts": {
            "consensus_rows": {
                "path": consensus_rows_path.name,
                "sha256": sha256_file(consensus_rows_path),
            },
            "consensus_audit": {
                "path": consensus_audit_path.name,
                "sha256": sha256_file(consensus_audit_path),
            },
            "quality_gate": {"path": quality_path.name, "sha256": sha256_file(quality_path)},
            "threshold_manifests": {
                "path": thresholds_path.name,
                "sha256": sha256_file(thresholds_path),
            },
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "adjudication_manifest.json", manifest)
    return BehavioralAdjudicationPhase(
        phase="baseline",
        phase_rows=tuple(phase_rows),
        all_rows=tuple(phase_rows),
        primary_rows=measurements.primary_rows,
        primary_manifest_rows=measurements.primary_manifests,
        primary_raw_rows=measurements.primary_raw,
        independent_final_records=measurements.independent_records,
        consensus_audit_rows=consensus.audit_rows,
        consensus_summary=consensus.summary,
        quality_gate=quality,
        thresholds=thresholds,
        threshold_manifests=threshold_manifests,
        manifest=manifest,
        complete=True,
        gate_passed=True,
    )


@dataclass(frozen=True, slots=True)
class _AuthenticatedBaseline:
    rows: tuple[dict[str, Any], ...]
    audit_rows: tuple[dict[str, Any], ...]
    quality_gate: Mapping[str, Any]
    thresholds: Mapping[str, float]
    threshold_manifests: Mapping[str, Mapping[str, Any]]
    manifest: Mapping[str, Any]


def _without_hash(value: Mapping[str, Any], field: str = "manifest_hash") -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _authenticated_artifact_path(
    directory: Path,
    artifacts: Mapping[str, Any],
    name: str,
) -> Path:
    entry = artifacts.get(name)
    if not isinstance(entry, Mapping):
        raise BehavioralAdjudicationPhaseError(
            f"baseline adjudication manifest lacks {name!r} artifact"
        )
    relative = entry.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise BehavioralAdjudicationPhaseError(
            f"baseline adjudication manifest has unsafe {name!r} path"
        )
    path = directory / relative
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise BehavioralAdjudicationPhaseError(
            f"baseline adjudication {name!r} artifact hash mismatch"
        )
    return path


def _load_authenticated_baseline(
    checkpoint_dir: str | Path,
) -> _AuthenticatedBaseline:
    directory = Path(checkpoint_dir)
    manifest_path = directory / "adjudication_manifest.json"
    if not manifest_path.is_file():
        raise BehavioralAdjudicationPhaseError(
            "treatment requires a completed baseline adjudication manifest"
        )
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_hash") != stable_hash(_without_hash(manifest))
        or manifest.get("protocol_version") != BEHAVIORAL_ADJUDICATION_PROTOCOL
        or manifest.get("phase") != "baseline"
        or manifest.get("complete") is not True
    ):
        raise BehavioralAdjudicationPhaseError(
            "baseline adjudication manifest identity or content hash mismatch"
        )
    environment = manifest.get("generation_environment_identity")
    try:
        authenticated_environment = validate_behavioral_generation_environment_identity(
            environment if isinstance(environment, Mapping) else {}
        )
    except BehavioralPhaseError as exc:
        raise BehavioralAdjudicationPhaseError(
            "baseline generation environment identity is invalid"
        ) from exc
    if manifest.get("generation_environment_identity_hash") != authenticated_environment.get(
        "identity_hash"
    ):
        raise BehavioralAdjudicationPhaseError(
            "baseline generation environment identity hash mismatch"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise BehavioralAdjudicationPhaseError(
            "baseline adjudication artifact inventory is malformed"
        )
    rows = read_jsonl(_authenticated_artifact_path(directory, artifacts, "consensus_rows"))
    audits = read_jsonl(_authenticated_artifact_path(directory, artifacts, "consensus_audit"))
    quality = read_json(_authenticated_artifact_path(directory, artifacts, "quality_gate"))
    threshold_manifests = read_json(
        _authenticated_artifact_path(directory, artifacts, "threshold_manifests")
    )
    if len(rows) != manifest.get("row_count") or stable_hash(rows) != manifest.get(
        "phase_rows_hash"
    ):
        raise BehavioralAdjudicationPhaseError("baseline adjudication row inventory mismatch")
    if stable_hash([row.get("run_id") for row in rows]) != manifest.get("run_ids_hash"):
        raise BehavioralAdjudicationPhaseError("baseline adjudication run ID inventory mismatch")
    for row in rows:
        if row.get("record_hash") != stable_hash(_without_hash(row, "record_hash")):
            raise BehavioralAdjudicationPhaseError("baseline adjudication row record hash mismatch")
    for audit in audits:
        if audit.get("record_hash") != stable_hash(_without_hash(audit, "record_hash")):
            raise BehavioralAdjudicationPhaseError("baseline consensus audit record hash mismatch")
    if not isinstance(quality, Mapping) or quality != manifest.get("quality_gate"):
        raise BehavioralAdjudicationPhaseError(
            "baseline quality artifact disagrees with its manifest"
        )
    consensus_summary = manifest.get("consensus_summary")
    if (
        not isinstance(consensus_summary, Mapping)
        or consensus_summary.get("gate_passed") is not True
        or quality.get("gate_passed") is not True
    ):
        raise BehavioralAdjudicationPhaseError(
            "treatment refuses a baseline that did not pass frozen gates"
        )
    if not isinstance(threshold_manifests, Mapping):
        raise BehavioralAdjudicationPhaseError("baseline threshold manifest bundle is malformed")
    thresholds_source = manifest.get("thresholds")
    if not isinstance(thresholds_source, Mapping):
        raise BehavioralAdjudicationPhaseError("baseline thresholds are missing")
    thresholds = {str(task): float(value) for task, value in thresholds_source.items()}
    if set(thresholds) != {str(task) for task in threshold_manifests}:
        raise BehavioralAdjudicationPhaseError("baseline threshold inventory mismatch")
    for task, source in threshold_manifests.items():
        if (
            not isinstance(source, Mapping)
            or source.get("manifest_hash") != stable_hash(_without_hash(source))
            or float(source.get("threshold")) != thresholds[str(task)]
        ):
            raise BehavioralAdjudicationPhaseError(
                f"baseline threshold manifest is invalid for {task!r}"
            )
    return _AuthenticatedBaseline(
        rows=tuple(rows),
        audit_rows=tuple(audits),
        quality_gate=dict(quality),
        thresholds=thresholds,
        threshold_manifests={str(task): dict(value) for task, value in threshold_manifests.items()},
        manifest=manifest,
    )


def _global_consensus_summary(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    minimum_exact_agreement: float,
    audit_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count_fields = (
        "expected_count",
        "independent_count",
        "exact_status_value_agreements",
        "known_consensus_count",
        "missing_independent_count",
        "independent_unknown_count",
        "disagreement_count",
    )
    counts = {
        field: int(baseline.get(field, 0)) + int(treatment.get(field, 0)) for field in count_fields
    }
    expected = counts["expected_count"]
    agreement_rate = counts["exact_status_value_agreements"] / expected if expected else 0.0
    known_rate = counts["known_consensus_count"] / expected if expected else 0.0
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": baseline.get("protocol_version"),
        "scope": "all_behavioral_final_outcomes",
        **counts,
        "exact_status_value_agreement_rate": agreement_rate,
        "minimum_exact_status_value_agreement": float(minimum_exact_agreement),
        "known_consensus_rate": known_rate,
        "gate_passed": bool(expected and agreement_rate >= minimum_exact_agreement),
        "consensus_rows_hash": stable_hash(rows),
        "audit_rows_hash": stable_hash(audit_rows),
        "phase_summary_hashes": {
            "baseline": baseline.get("manifest_hash"),
            "treatment": treatment.get("manifest_hash"),
        },
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def run_treatment_behavioral_adjudication_phase(
    *,
    generation_checkpoint_dir: str | Path,
    baseline_adjudication_checkpoint_dir: str | Path,
    checkpoint_dir: str | Path,
    primary_caller: AdjudicationCaller,
    independent_final_caller: AdjudicationCaller,
    execution_id: str,
    minimum_exact_agreement: float = 0.90,
    minimum_final_known_rate: float = 0.95,
    minimum_trajectory_final_consistency: float = 0.95,
    on_rollout_committed: Callable[[BehavioralAdjudicationUnit], None] | None = None,
) -> BehavioralAdjudicationPhase:
    """Adjudicate treatment rows and apply preregistered global behavioral gates."""

    baseline = _load_authenticated_baseline(baseline_adjudication_checkpoint_dir)
    generation = load_behavioral_generation_phase(generation_checkpoint_dir)
    if generation.plan.get("phase") != "treatment" or any(
        row.get("condition") == "baseline" for row in generation.rows
    ):
        raise BehavioralAdjudicationPhaseError(
            "treatment adjudication requires an authenticated treatment generation phase"
        )
    if not execution_id or baseline.manifest.get("execution_id") != execution_id:
        raise BehavioralAdjudicationPhaseError(
            "treatment execution_id must match the authenticated baseline"
        )
    treatment_environment = generation.manifest.get("shared_generation_environment")
    baseline_environment = baseline.manifest.get("generation_environment_identity")
    if (
        not isinstance(treatment_environment, Mapping)
        or not isinstance(baseline_environment, Mapping)
        or dict(treatment_environment) != dict(baseline_environment)
        or generation.manifest.get("shared_generation_environment_hash")
        != baseline.manifest.get("generation_environment_identity_hash")
    ):
        raise BehavioralAdjudicationPhaseError(
            "baseline and treatment behavioral generation environments differ"
        )
    _validate_callers(primary_caller, independent_final_caller)
    if _route_identity(primary_caller) != baseline.manifest.get("primary_route") or _route_identity(
        independent_final_caller
    ) != baseline.manifest.get("independent_final_route"):
        raise BehavioralAdjudicationPhaseError(
            "treatment judge routes must match the frozen baseline routes"
        )
    baseline_consensus = baseline.manifest["consensus_summary"]
    baseline_quality = baseline.quality_gate
    if (
        float(baseline_consensus["minimum_exact_status_value_agreement"])
        != float(minimum_exact_agreement)
        or float(baseline_quality["minimum_exact_status_value_agreement"])
        != float(minimum_exact_agreement)
        or float(baseline_quality["minimum_final_known_rate"]) != float(minimum_final_known_rate)
        or float(baseline_quality["minimum_trajectory_final_consistency"])
        != float(minimum_trajectory_final_consistency)
    ):
        raise BehavioralAdjudicationPhaseError(
            "treatment quality thresholds must match the frozen baseline thresholds"
        )
    for row in generation.rows:
        task = str(row.get("task"))
        if task not in baseline.thresholds or float(row.get("threshold")) != float(
            baseline.thresholds[task]
        ):
            raise BehavioralAdjudicationPhaseError(
                f"treatment generation threshold disagrees with baseline for {task!r}"
            )

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _freeze_adjudication_plan(
        directory,
        _adjudication_plan_payload(
            phase="treatment",
            generation=generation,
            primary_caller=primary_caller,
            independent_final_caller=independent_final_caller,
            execution_id=execution_id,
            minimum_exact_agreement=minimum_exact_agreement,
            minimum_final_known_rate=minimum_final_known_rate,
            minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
            threshold_contract={
                "baseline_adjudication_manifest_hash": baseline.manifest["manifest_hash"],
                "thresholds": dict(sorted(baseline.thresholds.items())),
            },
        ),
    )
    measurements = _adjudicate_generation_rows(
        generation.rows,
        directory=directory,
        primary_caller=primary_caller,
        independent_final_caller=independent_final_caller,
        on_rollout_committed=on_rollout_committed,
    )
    treatment_consensus = apply_all_final_consensus(
        measurements.primary_rows,
        measurements.consensus_independent_records,
        minimum_exact_agreement=minimum_exact_agreement,
        enforce_gate=False,
    )
    phase_rows = enrich_adjudicated_rows(
        treatment_consensus.rows,
        thresholds=baseline.thresholds,
        execution_id=execution_id,
    )
    all_rows = [*baseline.rows, *phase_rows]
    all_audits = [*baseline.audit_rows, *treatment_consensus.audit_rows]
    global_summary = _global_consensus_summary(
        baseline_consensus,
        treatment_consensus.summary,
        minimum_exact_agreement=minimum_exact_agreement,
        audit_rows=all_audits,
        rows=all_rows,
    )
    quality = evaluate_adjudication_quality(
        all_rows,
        minimum_exact_agreement=minimum_exact_agreement,
        minimum_final_known_rate=minimum_final_known_rate,
        minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
        required_phases=("baseline", "treatment"),
        enforce=False,
    )

    phase_rows_path = write_jsonl(directory / "consensus_rows.jsonl", phase_rows)
    all_rows_path = write_jsonl(directory / "all_behavioral_rows.jsonl", all_rows)
    audit_path = write_jsonl(directory / "consensus_audit.jsonl", all_audits)
    quality_path = write_json(directory / "quality_gate.json", quality)
    threshold_path = write_json(
        directory / "threshold_manifests.json",
        dict(sorted(baseline.threshold_manifests.items())),
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": BEHAVIORAL_ADJUDICATION_PROTOCOL,
        "phase": "treatment",
        "complete": True,
        "execution_id": execution_id,
        "generation_plan_hash": generation.plan["plan_hash"],
        "generation_manifest_hash": generation.manifest["manifest_hash"],
        "generation_environment_identity": dict(treatment_environment),
        "generation_environment_identity_hash": generation.manifest[
            "shared_generation_environment_hash"
        ],
        "baseline_adjudication_manifest_hash": baseline.manifest["manifest_hash"],
        "row_count": len(phase_rows),
        "all_row_count": len(all_rows),
        "run_ids_hash": stable_hash([row["run_id"] for row in phase_rows]),
        "phase_rows_hash": stable_hash(phase_rows),
        "all_rows_hash": stable_hash(all_rows),
        "primary_route": _route_identity(primary_caller),
        "independent_final_route": _route_identity(independent_final_caller),
        "consensus_summary": global_summary,
        "treatment_consensus_summary": dict(treatment_consensus.summary),
        "quality_gate": quality,
        "thresholds": dict(sorted(baseline.thresholds.items())),
        "artifacts": {
            "consensus_rows": {
                "path": phase_rows_path.name,
                "sha256": sha256_file(phase_rows_path),
            },
            "all_rows": {"path": all_rows_path.name, "sha256": sha256_file(all_rows_path)},
            "consensus_audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
            "quality_gate": {"path": quality_path.name, "sha256": sha256_file(quality_path)},
            "threshold_manifests": {
                "path": threshold_path.name,
                "sha256": sha256_file(threshold_path),
            },
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "adjudication_manifest.json", manifest)
    gate_passed = bool(global_summary["gate_passed"] and quality["gate_passed"])
    if not gate_passed:
        raise BehavioralAdjudicationGateError(
            "global behavioral exact-consensus or phase quality gate failed closed"
        )
    return BehavioralAdjudicationPhase(
        phase="treatment",
        phase_rows=tuple(phase_rows),
        all_rows=tuple(all_rows),
        primary_rows=measurements.primary_rows,
        primary_manifest_rows=measurements.primary_manifests,
        primary_raw_rows=measurements.primary_raw,
        independent_final_records=measurements.independent_records,
        consensus_audit_rows=tuple(all_audits),
        consensus_summary=global_summary,
        quality_gate=quality,
        thresholds=baseline.thresholds,
        threshold_manifests=baseline.threshold_manifests,
        manifest=manifest,
        complete=True,
        gate_passed=True,
    )


__all__ = [
    "BEHAVIORAL_ADJUDICATION_PROTOCOL",
    "BehavioralAdjudicationGateError",
    "BehavioralAdjudicationPhase",
    "BehavioralAdjudicationPhaseError",
    "BehavioralAdjudicationUnit",
    "run_baseline_behavioral_adjudication_phase",
    "run_treatment_behavioral_adjudication_phase",
]
