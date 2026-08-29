"""Primary rollout measurement from blinded external adjudication.

Raw numeric parsing is retained only as a disagreement diagnostic.  Every value
used by behavioral inference comes from the frozen blind final/trajectory
instruments (or a separately audited manual override).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from model_forensics.adjudication import (
    AdjudicationCaller,
    KnowledgeStatus,
    audit_final_agreement,
    blinded_case_from_rollout,
    collect_adjudication_outputs,
    materialize_adjudication,
)
from model_forensics.io import assert_unique, stable_hash
from model_forensics.parsing import parse_trajectory, select_final_estimate
from model_forensics.prompts import QUESTIONS, Task, is_good_outcome
from model_forensics.statistics import signed_log_ratio


class RolloutAdjudicationError(RuntimeError):
    """A rollout cannot safely enter the primary analysis table."""


@dataclass(frozen=True)
class AdjudicatedBatch:
    rows: tuple[dict[str, Any], ...]
    manifest_rows: tuple[dict[str, Any], ...]
    raw_judge_rows: tuple[dict[str, Any], ...]


def _question(task: str) -> str:
    try:
        task_value = Task(task)
    except ValueError as exc:
        raise RolloutAdjudicationError(f"unsupported adjudication task: {task}") from exc
    return QUESTIONS[task_value]


def _local_final(row: Mapping[str, Any]) -> float | None:
    estimate = select_final_estimate(str(row.get("answer", "")), source="answer")
    return None if estimate is None else estimate.value


def adjudicate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    caller: AdjudicationCaller,
    primary_inference: bool,
    on_record: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], None]
    | None = None,
) -> AdjudicatedBatch:
    """Blindly judge raw rows without exposing any experimental metadata."""

    measured: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    for source in rows:
        run_id = str(source.get("run_id", ""))
        if not run_id:
            raise RolloutAdjudicationError("every rollout requires run_id")
        task = str(source.get("task", ""))
        case = blinded_case_from_rollout(source, task_question=_question(task))
        outputs = collect_adjudication_outputs(
            case,
            caller,
            for_primary_inference=primary_inference,
        )
        record = materialize_adjudication(
            case=case,
            external_outputs=outputs,
            primary_inference=primary_inference,
        )
        final_known = record.effective_final.status is KnowledgeStatus.KNOWN
        trajectory_known = record.effective_trajectory.status is KnowledgeStatus.KNOWN
        final_value = record.effective_final.value if final_known else None
        trajectory_values = tuple(record.effective_trajectory.values) if trajectory_known else ()
        trajectory_final_matches = bool(
            final_value is not None and trajectory_values and trajectory_values[-1] == final_value
        )
        if final_value is None:
            trajectory_reason = "final_unknown"
        elif not trajectory_values:
            trajectory_reason = "trajectory_unknown"
        elif not trajectory_final_matches:
            trajectory_reason = "trajectory_final_disagrees_with_final_instrument"
        else:
            trajectory_reason = None

        local_audit = audit_final_agreement(record.effective_final, _local_final(source))
        row = dict(source)
        row.pop("record_hash", None)
        row.update(
            {
                "adjudication_case_hash": case.case_hash,
                "adjudication_primary_inference": primary_inference,
                "final_estimate": final_value,
                "trajectory_values": list(trajectory_values),
                "final_measurement_valid": final_value is not None,
                "trajectory_measurement_valid": trajectory_reason is None,
                "trajectory_invalid_reason": trajectory_reason,
                "requires_blinded_manual_review": (
                    trajectory_reason is not None or local_audit.requires_manual_review
                ),
                "local_parser_final_audit": local_audit.to_dict(),
                "invalid_reason": None if final_value is not None else "judge_final_unknown",
                "valid": final_value is not None,
            }
        )
        row["record_hash"] = stable_hash(row)
        measured.append(row)

        manifest = record.to_dict(include_hash=True)
        manifest["run_id"] = run_id
        manifest["record_hash"] = stable_hash(
            {key: value for key, value in manifest.items() if key != "record_hash"}
        )
        manifests.append(manifest)
        raw_outputs.append(
            {
                "run_id": run_id,
                "blinded_case_hash": case.case_hash,
                "final_request_id": outputs.final.request_id,
                "final_response": outputs.final.raw_response,
                "final_response_hash": outputs.final.response_hash,
                "trajectory_request_id": outputs.trajectory.request_id,
                "trajectory_response": outputs.trajectory.raw_response,
                "trajectory_response_hash": outputs.trajectory.response_hash,
                "primary_inference": primary_inference,
            }
        )
        raw_outputs[-1]["record_hash"] = stable_hash(raw_outputs[-1])
        if on_record is not None:
            on_record(measured[-1], manifests[-1], raw_outputs[-1])

    assert_unique(measured, "run_id")
    assert_unique(manifests, "run_id")
    assert_unique(raw_outputs, "run_id")
    return AdjudicatedBatch(tuple(measured), tuple(manifests), tuple(raw_outputs))


def median_valid_final(rows: Sequence[Mapping[str, Any]], *, task: str) -> float:
    """Return the frozen baseline median from externally judged final values."""

    values = [
        float(row["final_estimate"])
        for row in rows
        if row.get("task") == task
        and row.get("condition") == "baseline"
        and row.get("final_measurement_valid") is True
        and row.get("final_estimate") is not None
    ]
    if not values:
        raise RolloutAdjudicationError(f"task {task!r} has no valid judged baseline finals")
    value = float(statistics.median(values))
    if not math.isfinite(value) or value <= 0:
        raise RolloutAdjudicationError("judged baseline median must be positive and finite")
    return value


def enrich_adjudicated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
    execution_id: str,
) -> list[dict[str, Any]]:
    """Derive outcomes exclusively from judged final and trajectory values."""

    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row.pop("record_hash", None)
        task = str(row["task"])
        if task not in thresholds:
            raise RolloutAdjudicationError(f"missing frozen threshold for task {task!r}")
        threshold = float(thresholds[task])
        if not math.isfinite(threshold) or threshold <= 0:
            raise RolloutAdjudicationError("threshold must be positive and finite")
        condition = str(row["condition"])
        direction = int(row["direction"])
        final = row.get("final_estimate")
        final_value = float(final) if final is not None else None

        trajectory = None
        if row.get("trajectory_measurement_valid") is True:
            trajectory = parse_trajectory(
                str(row.get("reasoning", "")),
                str(row.get("answer", "")),
                threshold,
                condition,
                adjudicated_estimates=row.get("trajectory_values", ()),
                review_notes="blind external trajectory instrument",
            )
        first = trajectory.features.first_estimate if trajectory is not None else None
        first_good = is_good_outcome(condition, first, threshold) if first is not None else None
        final_good = (
            is_good_outcome(condition, final_value, threshold) if final_value is not None else None
        )
        first_to_final_flip = (
            bool(first_good != final_good)
            if first_good is not None and final_good is not None
            else None
        )
        signed_first = None
        signed_final = None
        if direction in {-1, 1}:
            if first is not None and first > 0:
                signed_first = signed_log_ratio(first, threshold, direction)
            if final_value is not None and final_value > 0:
                signed_final = signed_log_ratio(final_value, threshold, direction)

        features = trajectory.features if trajectory is not None else None
        model_hash = stable_hash(
            {
                "model_id": row.get("backend", {}).get("model_id"),
                "revision": row.get("backend", {}).get("revision"),
                "backend": row.get("backend", {}).get("backend"),
            }
        )
        row.update(
            {
                "schema_version": 1,
                "threshold": threshold,
                "trace": str(row.get("reasoning", "")),
                "trajectory": trajectory.to_dict() if trajectory is not None else None,
                "first_estimate": first,
                "trajectory_final_estimate": (
                    features.final_estimate if features is not None else None
                ),
                "revision_count": features.revision_count if features is not None else None,
                "first_good_side_crossing_index": (
                    features.first_good_side_crossing_index if features is not None else None
                ),
                "stopped_after_first_good_side_crossing": (
                    features.stopped_after_first_good_side_crossing
                    if features is not None
                    else None
                ),
                "revisions_after_good": (
                    features.revisions_after_good if features is not None else None
                ),
                "first_good_side": first_good,
                "final_good_side": final_good,
                "first_to_final_flip": first_to_final_flip,
                "signed_log_ratio_first": signed_first,
                "signed_log_ratio_final": signed_final,
                "model_hash": model_hash,
                "provenance": {
                    "execution_id": execution_id,
                    "model_id": row.get("backend", {}).get("model_id"),
                    "provider": row.get("backend", {}).get("backend"),
                    "seed": int(row["seed"]),
                    "prompt_hash": row["prompt_hash"],
                    "model_hash": model_hash,
                    "model_revision": row.get("backend", {}).get("revision"),
                    "request_id": row["run_id"],
                    "measurement": "blind_external_adjudication",
                },
            }
        )
        row["record_hash"] = stable_hash(row)
        enriched.append(row)
    assert_unique(enriched, "run_id")
    return enriched


__all__ = [
    "AdjudicatedBatch",
    "RolloutAdjudicationError",
    "adjudicate_raw_rows",
    "enrich_adjudicated_rows",
    "median_valid_final",
]
