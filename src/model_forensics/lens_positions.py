"""Construct the five preregistered lens positions from exact source evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_forensics.adjudication import BlindedAdjudicationCase, blinded_case_from_rollout
from model_forensics.estimate_spans import (
    FIRST_ESTIMATE_SPAN_INSTRUMENT_ID,
    FirstEstimateSpanRecord,
    SpanStatus,
)
from model_forensics.io import stable_hash
from model_forensics.token_spans import CompletionTokenMap

POSITION_ORDER = (
    "prompt_end",
    "first_estimate_pre",
    "anchor_pre",
    "anchor_post",
    "final_answer_pre",
)
POSITION_MANIFEST_SCHEMA_VERSION = 1


class LensPositionError(ValueError):
    """Exact token evidence is missing, mismatched, or ambiguous."""


def _validate_content_hash(row: Mapping[str, Any], *, label: str) -> None:
    recorded = row.get("record_hash")
    if not isinstance(recorded, str):
        raise LensPositionError(f"{label} lacks record_hash")
    unhashed = {key: value for key, value in row.items() if key != "record_hash"}
    if stable_hash(unhashed) != recorded:
        raise LensPositionError(f"{label} record_hash mismatch")


def _pre_token_index(prompt_count: int, completion_token_start: int) -> int:
    if prompt_count <= 0:
        raise LensPositionError("prompt token stream must be nonempty")
    if completion_token_start < 0:
        raise LensPositionError("completion token boundary cannot be negative")
    if completion_token_start == 0:
        return prompt_count - 1
    return prompt_count + completion_token_start - 1


def _assert_anchor_span_matches(
    actual: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    required = (
        "section",
        "section_char_start",
        "section_char_end",
        "completion_char_start",
        "completion_char_end",
        "token_start",
        "token_end",
        "text",
        "token_ids_hash",
        "completion_token_ids_hash",
        "round_trip_verified",
    )
    missing = [key for key in required if key not in frozen]
    if missing:
        raise LensPositionError(f"frozen anchor token span lacks {missing}")
    if any(actual[key] != frozen[key] for key in required):
        raise LensPositionError("recomputed anchor token span differs from frozen provenance")


def _case_for_rollout(
    rollout: Mapping[str, Any],
    *,
    task_question: str,
) -> BlindedAdjudicationCase:
    return blinded_case_from_rollout(rollout, task_question=task_question)


def build_lens_position_row(
    *,
    rollout: Mapping[str, Any],
    anchor: Mapping[str, Any],
    first_estimate_record: FirstEstimateSpanRecord,
    tokenizer: Any,
    task_question: str,
    anchor_manifest_hash: str,
) -> dict[str, Any]:
    """Build one position row without retokenizing or numerically re-parsing.

    `*_pre` is the residual after the token immediately preceding the selected
    span. `anchor_post` is the residual after the anchor's final original token.
    """

    _validate_content_hash(rollout, label="rollout")
    trace_id = str(anchor.get("trace_id", ""))
    if not trace_id or trace_id != str(rollout.get("run_id", "")):
        raise LensPositionError("anchor and rollout trace IDs disagree")
    if not isinstance(anchor_manifest_hash, str) or not anchor_manifest_hash:
        raise LensPositionError("anchor manifest hash is required")
    case = _case_for_rollout(rollout, task_question=task_question)
    if first_estimate_record.case_hash != case.case_hash:
        raise LensPositionError("first-estimate adjudication belongs to another blind case")
    if not first_estimate_record.primary_inference:
        raise LensPositionError("primary lens positions require an external primary span record")
    if first_estimate_record.adjudication.status is not SpanStatus.KNOWN:
        raise LensPositionError("first-estimate span is UNKNOWN")

    raw_text = rollout.get("raw_text")
    token_streams = rollout.get("token_streams")
    if not isinstance(raw_text, str) or not isinstance(token_streams, Mapping):
        raise LensPositionError("rollout lacks raw_text or exact token streams")
    token_map = CompletionTokenMap.from_manifest(
        tokenizer=tokenizer,
        raw_text=raw_text,
        token_streams=token_streams,
        skip_special_tokens=True,
    )
    if token_map.sections.reasoning != case.trace or token_map.sections.answer != case.answer:
        raise LensPositionError("adjudicated trace/answer differs from decoded token sections")

    provenance = anchor.get("provenance")
    if not isinstance(provenance, Mapping):
        raise LensPositionError("anchor lacks frozen provenance")
    frozen_anchor_span = provenance.get("token_span")
    if not isinstance(frozen_anchor_span, Mapping):
        raise LensPositionError("anchor provenance lacks exact token_span")
    anchor_text = anchor.get("sentence_text")
    anchor_start = anchor.get("char_start")
    anchor_end = anchor.get("char_end")
    if (
        not isinstance(anchor_text, str)
        or isinstance(anchor_start, bool)
        or not isinstance(anchor_start, int)
        or isinstance(anchor_end, bool)
        or not isinstance(anchor_end, int)
    ):
        raise LensPositionError("anchor lacks exact reasoning character span")
    anchor_span = token_map.map_reasoning_span(
        anchor_start,
        anchor_end,
        expected_text=anchor_text,
    )
    _assert_anchor_span_matches(anchor_span.as_dict(), frozen_anchor_span)

    adjudicated = first_estimate_record.adjudication
    assert adjudicated.source is not None and adjudicated.quote is not None
    assert first_estimate_record.resolved_char_start is not None
    assert first_estimate_record.resolved_char_end is not None
    if adjudicated.source == "trace":
        first_span = token_map.map_reasoning_span(
            first_estimate_record.resolved_char_start,
            first_estimate_record.resolved_char_end,
            expected_text=adjudicated.quote,
        )
    else:
        completion_start = (
            token_map.sections.answer_char_start + first_estimate_record.resolved_char_start
        )
        completion_end = (
            token_map.sections.answer_char_start + first_estimate_record.resolved_char_end
        )
        first_span = token_map.map_completion_span(
            completion_start,
            completion_end,
            expected_text=adjudicated.quote,
            section="answer",
            section_char_start=first_estimate_record.resolved_char_start,
            section_char_end=first_estimate_record.resolved_char_end,
        )

    if not token_map.sections.answer:
        raise LensPositionError("final_answer_pre is undefined for an empty answer")
    answer_start = token_map.sections.answer_char_start
    answer_first_token = token_map.map_completion_span(
        answer_start,
        answer_start + 1,
        expected_text=raw_text[answer_start : answer_start + 1],
        section="answer",
        section_char_start=0,
        section_char_end=1,
    )

    prompt_count = len(token_map.prompt_token_ids)
    completion_count = len(token_map.token_ids)
    positions = {
        "prompt_end": prompt_count - 1,
        "first_estimate_pre": _pre_token_index(prompt_count, first_span.token_start),
        "anchor_pre": _pre_token_index(prompt_count, anchor_span.token_start),
        "anchor_post": prompt_count + anchor_span.token_end - 1,
        "final_answer_pre": _pre_token_index(prompt_count, answer_first_token.token_start),
    }
    sequence_length = prompt_count + completion_count
    if set(positions) != set(POSITION_ORDER) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < sequence_length
        for index in positions.values()
    ):
        raise LensPositionError("computed lens position is outside the exact token stream")

    direction = anchor.get("direction")
    if direction == "above_good":
        good_side_direction = 1
    elif direction == "below_good":
        good_side_direction = -1
    else:
        raise LensPositionError("anchor has an unknown incentive direction")

    payload: dict[str, Any] = {
        "schema_version": POSITION_MANIFEST_SCHEMA_VERSION,
        "trace_id": trace_id,
        "anchor_id": anchor.get("anchor_id"),
        "anchor_manifest_hash": anchor_manifest_hash,
        "rollout_record_hash": rollout["record_hash"],
        "first_estimate_span_record_hash": first_estimate_record.to_dict()["record_hash"],
        "first_estimate_span_instrument_id": FIRST_ESTIMATE_SPAN_INSTRUMENT_ID,
        "first_estimate_span_primary_inference": first_estimate_record.primary_inference,
        "prompt_token_ids_hash": token_streams["prompt_token_ids_hash"],
        "completion_token_ids_hash": token_streams["completion_token_ids_hash"],
        "combined_token_stream_hash": token_streams["combined_token_stream_hash"],
        "position_order": list(POSITION_ORDER),
        "position_indices": positions,
        "position_evidence": {
            "first_estimate": first_span.as_dict(),
            "anchor": anchor_span.as_dict(),
            "answer_first_token": answer_first_token.as_dict(),
        },
        "good_side_direction": good_side_direction,
        "causal_claim": False,
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "POSITION_MANIFEST_SCHEMA_VERSION",
    "POSITION_ORDER",
    "LensPositionError",
    "build_lens_position_row",
]
