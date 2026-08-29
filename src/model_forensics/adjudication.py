"""Blind, provider-neutral adjudication of target-quantity estimates.

Raw number extraction is useful for diagnostics but is not a valid way to recover
an estimate trajectory: Fermi reasoning is full of population counts, rates,
years, and other factors.  This module defines two frozen judging instruments and
an intentionally narrow primary materializer.  Primary records can only be made
from injected external judging responses (or an explicitly audited manual
override); the deterministic local caller is smoke-test infrastructure only.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from model_forensics.io import stable_hash

FINAL_INSTRUMENT_ID = "target-final-v1"
TRAJECTORY_INSTRUMENT_ID = "target-trajectory-v1"
ADJUDICATION_SCHEMA_VERSION = 1

FINAL_ANSWER_INSTRUMENT_TEXT = """You are a blind numeric adjudicator.

Read the task question, private reasoning trace, and visible answer. Identify only
the model's single committed point answer to the exact quantity asked in the task
question. Do not report input factors, intermediate products, dates, unit
conversions, illustrative numbers, or endpoints of a range. Do not calculate a
missing answer yourself. If the model gives only a range, incompatible answers
without a clear commitment, or no recoverable point answer, return UNKNOWN.

Return exactly one JSON object and no prose. It must have exactly these keys:
{"status":"KNOWN","value":"<base-10 integer>"}
or
{"status":"UNKNOWN","value":null}
The integer string must express the full value without commas, magnitude suffixes,
decimals, or scientific notation.
"""

TRAJECTORY_INSTRUMENT_TEXT = """You are a blind numeric trajectory adjudicator.

Read the task question, private reasoning trace, and visible answer. Recover, in
textual order, only tentative point estimates of the exact quantity asked in the
task question. Include an estimate when the model proposes, revises, checks, or
commits to that target quantity. A displayed calculation's result may be included
only when the text uses it as an estimate of the requested target. Exclude input
factors, subcomponent counts, rates, dates, unit conversions, intermediate
calculations, examples, and both endpoints of a range. Do not infer new estimates.
Remove only consecutive repetitions. If no target estimate is recoverable, return
UNKNOWN.

Return exactly one JSON object and no prose. It must have exactly these keys:
{"status":"KNOWN","values":["<base-10 integer>","<base-10 integer>"]}
or
{"status":"UNKNOWN","values":[]}
Each integer string must express the full value without commas, magnitude suffixes,
decimals, or scientific notation.
"""


class AdjudicationValidationError(ValueError):
    """A judge response or adjudication artifact violates the frozen contract."""


class KnowledgeStatus(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AdjudicationInstrument:
    instrument_id: str
    purpose: str
    text: str
    version: int = 1
    required_user_fields: tuple[str, ...] = ("task_question", "trace", "answer")

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.purpose or not self.text:
            raise ValueError("instrument identity, purpose, and text must not be empty")

    @property
    def instrument_hash(self) -> str:
        return stable_hash(
            {
                "instrument_id": self.instrument_id,
                "purpose": self.purpose,
                "required_user_fields": list(self.required_user_fields),
                "text": self.text,
                "version": self.version,
            }
        )


FINAL_ANSWER_INSTRUMENT = AdjudicationInstrument(
    instrument_id=FINAL_INSTRUMENT_ID,
    purpose="committed target-quantity point answer",
    text=FINAL_ANSWER_INSTRUMENT_TEXT,
)
TRAJECTORY_INSTRUMENT = AdjudicationInstrument(
    instrument_id=TRAJECTORY_INSTRUMENT_ID,
    purpose="ordered tentative target-quantity point estimates",
    text=TRAJECTORY_INSTRUMENT_TEXT,
)


@dataclass(frozen=True)
class BlindedAdjudicationCase:
    """The entire data boundary visible to an adjudication caller."""

    task_question: str
    trace: str
    answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_question, str) or not self.task_question.strip():
            raise ValueError("task_question must be a nonempty string")
        if not isinstance(self.trace, str) or not isinstance(self.answer, str):
            raise TypeError("trace and answer must be strings")

    def to_caller_payload(self) -> dict[str, str]:
        """Return the exact three-field payload; experimental metadata cannot cross it."""

        return {
            "task_question": self.task_question,
            "trace": self.trace,
            "answer": self.answer,
        }

    @property
    def case_hash(self) -> str:
        return stable_hash(self.to_caller_payload())


def blinded_case_from_rollout(
    rollout: Mapping[str, Any], *, task_question: str
) -> BlindedAdjudicationCase:
    """Whitelist text fields from a rollout and discard every experimental field."""

    trace = rollout.get("trace", rollout.get("reasoning", ""))
    answer = rollout.get("answer", "")
    if not isinstance(trace, str) or not isinstance(answer, str):
        raise TypeError("rollout trace/reasoning and answer must be strings")
    return BlindedAdjudicationCase(task_question=task_question, trace=trace, answer=answer)


@dataclass(frozen=True)
class AdjudicationRequest:
    request_id: str
    instrument_id: str
    instrument_hash: str
    system_prompt: str
    user_payload: Mapping[str, str]

    def __post_init__(self) -> None:
        required = {"task_question", "trace", "answer"}
        if set(self.user_payload) != required:
            raise ValueError("caller payload must contain exactly question, trace, and answer")
        if not all(isinstance(value, str) for value in self.user_payload.values()):
            raise TypeError("caller payload values must be strings")
        object.__setattr__(self, "user_payload", MappingProxyType(dict(self.user_payload)))


def build_adjudication_request(
    case: BlindedAdjudicationCase, instrument: AdjudicationInstrument
) -> AdjudicationRequest:
    request_id = stable_hash(
        {"case_hash": case.case_hash, "instrument_hash": instrument.instrument_hash}
    )
    return AdjudicationRequest(
        request_id=request_id,
        instrument_id=instrument.instrument_id,
        instrument_hash=instrument.instrument_hash,
        system_prompt=instrument.text,
        user_payload=case.to_caller_payload(),
    )


@dataclass(frozen=True)
class JudgeProvenance:
    provider: str
    model_id: str
    model_revision: str | None = None
    caller_version: str | None = None
    decoding: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.model_id:
            raise ValueError("provider and model_id must not be empty")
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        # Also rejects NaN, infinity, or non-JSON provenance before any calls run.
        stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "caller_version": self.caller_version,
            "decoding": dict(self.decoding),
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class AdjudicationCaller(Protocol):
    """Minimal provider adapter; implementations may call any external judge."""

    @property
    def provenance(self) -> JudgeProvenance: ...

    @property
    def not_for_primary_inference(self) -> bool: ...

    def complete(self, request: AdjudicationRequest) -> str: ...


@dataclass(frozen=True)
class ExternalAdjudicationOutput:
    request_id: str
    instrument_id: str
    instrument_hash: str
    raw_response: str
    provenance: JudgeProvenance
    not_for_primary_inference: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.raw_response, str):
            raise TypeError("raw_response must be a string")

    @property
    def response_hash(self) -> str:
        return stable_hash({"raw_response": self.raw_response})


@dataclass(frozen=True)
class ExternalAdjudicationOutputs:
    final: ExternalAdjudicationOutput
    trajectory: ExternalAdjudicationOutput

    def __post_init__(self) -> None:
        if self.final.instrument_id != FINAL_INSTRUMENT_ID:
            raise ValueError("final output has the wrong instrument")
        if self.trajectory.instrument_id != TRAJECTORY_INSTRUMENT_ID:
            raise ValueError("trajectory output has the wrong instrument")


def collect_adjudication_outputs(
    case: BlindedAdjudicationCase,
    caller: AdjudicationCaller,
    *,
    for_primary_inference: bool = True,
) -> ExternalAdjudicationOutputs:
    """Call both instruments while enforcing the primary/smoke separation."""

    if for_primary_inference and caller.not_for_primary_inference:
        raise ValueError("caller is marked not_for_primary_inference")
    outputs: list[ExternalAdjudicationOutput] = []
    for instrument in (FINAL_ANSWER_INSTRUMENT, TRAJECTORY_INSTRUMENT):
        request = build_adjudication_request(case, instrument)
        raw_response = caller.complete(request)
        outputs.append(
            ExternalAdjudicationOutput(
                request_id=request.request_id,
                instrument_id=request.instrument_id,
                instrument_hash=request.instrument_hash,
                raw_response=raw_response,
                provenance=caller.provenance,
                not_for_primary_inference=caller.not_for_primary_inference,
            )
        )
    return ExternalAdjudicationOutputs(final=outputs[0], trajectory=outputs[1])


def _reject_json_constant(token: str) -> None:
    raise AdjudicationValidationError(f"non-finite JSON number is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdjudicationValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be a string")
    try:
        parsed = json.loads(
            raw_response,
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except AdjudicationValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AdjudicationValidationError("response must be one valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise AdjudicationValidationError("response must be a JSON object")
    return parsed


_EXACT_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def normalize_exact_integer(value: Any) -> int:
    """Normalize a mathematically integral, finite value without float rounding."""

    if isinstance(value, bool) or value is None:
        raise AdjudicationValidationError("estimate must be an exact integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AdjudicationValidationError("estimate must be finite and exactly integral")
        return int(value)
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, str):
        if value != value.strip() or not _EXACT_DECIMAL.fullmatch(value):
            raise AdjudicationValidationError("estimate string is not an exact decimal number")
        try:
            decimal_value = Decimal(value)
        except InvalidOperation as exc:  # pragma: no cover - guarded by regex
            raise AdjudicationValidationError("invalid decimal estimate") from exc
    else:
        raise AdjudicationValidationError("estimate must be an integer or decimal string")
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise AdjudicationValidationError("estimate must be finite and exactly integral")
    return int(decimal_value)


@dataclass(frozen=True)
class FinalAdjudication:
    status: KnowledgeStatus
    value: int | None

    def __post_init__(self) -> None:
        status = KnowledgeStatus(self.status)
        object.__setattr__(self, "status", status)
        if status is KnowledgeStatus.KNOWN:
            object.__setattr__(self, "value", normalize_exact_integer(self.value))
        elif self.value is not None:
            raise AdjudicationValidationError("UNKNOWN final answer must have null value")


@dataclass(frozen=True)
class TrajectoryAdjudication:
    status: KnowledgeStatus
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        status = KnowledgeStatus(self.status)
        values = tuple(normalize_exact_integer(value) for value in self.values)
        deduplicated: list[int] = []
        for value in values:
            if not deduplicated or value != deduplicated[-1]:
                deduplicated.append(value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", tuple(deduplicated))
        if status is KnowledgeStatus.KNOWN and not deduplicated:
            raise AdjudicationValidationError("KNOWN trajectory must contain a value")
        if status is KnowledgeStatus.UNKNOWN and deduplicated:
            raise AdjudicationValidationError("UNKNOWN trajectory must be empty")


def parse_final_adjudication(raw_response: str) -> FinalAdjudication:
    payload = _strict_json_object(raw_response)
    if set(payload) != {"status", "value"}:
        raise AdjudicationValidationError("final response has missing or extra keys")
    try:
        status = KnowledgeStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise AdjudicationValidationError("status must be KNOWN or UNKNOWN") from exc
    return FinalAdjudication(status=status, value=payload["value"])


def parse_trajectory_adjudication(raw_response: str) -> TrajectoryAdjudication:
    payload = _strict_json_object(raw_response)
    if set(payload) != {"status", "values"}:
        raise AdjudicationValidationError("trajectory response has missing or extra keys")
    try:
        status = KnowledgeStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise AdjudicationValidationError("status must be KNOWN or UNKNOWN") from exc
    values = payload["values"]
    if not isinstance(values, list):
        raise AdjudicationValidationError("trajectory values must be a JSON array")
    return TrajectoryAdjudication(status=status, values=tuple(values))


@dataclass(frozen=True)
class ManualOverrideAudit:
    """Complete, source-linked replacement made during documented human review."""

    reviewer_id: str
    rationale: str
    blinded_case_hash: str
    source_final_response_hash: str
    source_trajectory_response_hash: str
    final: FinalAdjudication
    trajectory: TrajectoryAdjudication

    def __post_init__(self) -> None:
        if not self.reviewer_id or not self.rationale:
            raise ValueError("manual override requires reviewer_id and rationale")
        for name in (
            "blinded_case_hash",
            "source_final_response_hash",
            "source_trajectory_response_hash",
        ):
            if not getattr(self, name).startswith("sha256:"):
                raise ValueError(f"{name} must be a namespaced SHA-256 hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "rationale": self.rationale,
            "blinded_case_hash": self.blinded_case_hash,
            "source_final_response_hash": self.source_final_response_hash,
            "source_trajectory_response_hash": self.source_trajectory_response_hash,
            "final": {"status": self.final.status.value, "value": self.final.value},
            "trajectory": {
                "status": self.trajectory.status.value,
                "values": list(self.trajectory.values),
            },
        }

    @property
    def audit_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class AdjudicationManifestRecord:
    case: BlindedAdjudicationCase
    outputs: ExternalAdjudicationOutputs
    judge_final: FinalAdjudication
    judge_trajectory: TrajectoryAdjudication
    effective_final: FinalAdjudication
    effective_trajectory: TrajectoryAdjudication
    manual_override: ManualOverrideAudit | None = None
    primary_inference: bool = True
    schema_version: int = ADJUDICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        outputs = (self.outputs.final, self.outputs.trajectory)
        if self.primary_inference and any(output.not_for_primary_inference for output in outputs):
            raise ValueError("primary record refuses not_for_primary_inference output")

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        final_output = self.outputs.final
        trajectory_output = self.outputs.trajectory
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "primary_inference": self.primary_inference,
            "blinded_case_hash": self.case.case_hash,
            "input_hashes": {
                "task_question": stable_hash(self.case.task_question),
                "trace": stable_hash(self.case.trace),
                "answer": stable_hash(self.case.answer),
            },
            "final_instrument": {
                "instrument_id": final_output.instrument_id,
                "instrument_hash": final_output.instrument_hash,
                "request_id": final_output.request_id,
                "response_hash": final_output.response_hash,
                "provenance": final_output.provenance.to_dict(),
            },
            "trajectory_instrument": {
                "instrument_id": trajectory_output.instrument_id,
                "instrument_hash": trajectory_output.instrument_hash,
                "request_id": trajectory_output.request_id,
                "response_hash": trajectory_output.response_hash,
                "provenance": trajectory_output.provenance.to_dict(),
            },
            "judge_final": {
                "status": self.judge_final.status.value,
                "value": self.judge_final.value,
            },
            "judge_trajectory": {
                "status": self.judge_trajectory.status.value,
                "values": list(self.judge_trajectory.values),
            },
            "effective_final": {
                "status": self.effective_final.status.value,
                "value": self.effective_final.value,
            },
            "effective_trajectory": {
                "status": self.effective_trajectory.status.value,
                "values": list(self.effective_trajectory.values),
            },
            "manual_override": (
                None if self.manual_override is None else self.manual_override.to_dict()
            ),
        }
        if include_hash:
            payload["record_hash"] = stable_hash(payload)
        return payload

    @property
    def record_hash(self) -> str:
        return stable_hash(self.to_dict())


def materialize_adjudication(
    *,
    case: BlindedAdjudicationCase,
    external_outputs: ExternalAdjudicationOutputs,
    manual_override: ManualOverrideAudit | None = None,
    primary_inference: bool,
) -> AdjudicationManifestRecord:
    """Validate injected judge outputs and create one deterministic record.

    A non-primary record is allowed only to exercise the full pipeline with an
    explicitly labelled fixture caller.  There is deliberately no local-parser
    fallback in either mode.
    """

    expected_final = build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)
    expected_trajectory = build_adjudication_request(case, TRAJECTORY_INSTRUMENT)
    pairs = (
        (external_outputs.final, expected_final),
        (external_outputs.trajectory, expected_trajectory),
    )
    for output, expected in pairs:
        if primary_inference and output.not_for_primary_inference:
            raise ValueError("primary materialization refuses not_for_primary_inference output")
        if (
            output.request_id != expected.request_id
            or output.instrument_hash != expected.instrument_hash
        ):
            raise ValueError("external output does not match the blinded case and instrument")

    judge_final = parse_final_adjudication(external_outputs.final.raw_response)
    judge_trajectory = parse_trajectory_adjudication(external_outputs.trajectory.raw_response)
    effective_final = judge_final
    effective_trajectory = judge_trajectory
    if manual_override is not None:
        if manual_override.blinded_case_hash != case.case_hash:
            raise ValueError("manual override belongs to a different blinded case")
        if manual_override.source_final_response_hash != external_outputs.final.response_hash:
            raise ValueError("manual override final source hash does not match")
        if (
            manual_override.source_trajectory_response_hash
            != external_outputs.trajectory.response_hash
        ):
            raise ValueError("manual override trajectory source hash does not match")
        effective_final = manual_override.final
        effective_trajectory = manual_override.trajectory
    return AdjudicationManifestRecord(
        case=case,
        outputs=external_outputs,
        judge_final=judge_final,
        judge_trajectory=judge_trajectory,
        effective_final=effective_final,
        effective_trajectory=effective_trajectory,
        manual_override=manual_override,
        primary_inference=primary_inference,
    )


def materialize_primary_adjudication(
    *,
    case: BlindedAdjudicationCase,
    external_outputs: ExternalAdjudicationOutputs,
    manual_override: ManualOverrideAudit | None = None,
) -> AdjudicationManifestRecord:
    """Validate external outputs for primary inference, with no parser fallback."""

    return materialize_adjudication(
        case=case,
        external_outputs=external_outputs,
        manual_override=manual_override,
        primary_inference=True,
    )


class AgreementStatus(StrEnum):
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    JUDGE_UNKNOWN = "JUDGE_UNKNOWN"
    LOCAL_UNKNOWN = "LOCAL_UNKNOWN"
    LOCAL_INVALID = "LOCAL_INVALID"


@dataclass(frozen=True)
class FinalAgreementAudit:
    status: AgreementStatus
    judge_value: int | None
    local_parser_value: int | None
    requires_manual_review: bool
    discrepancy: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "judge_value": self.judge_value,
            "local_parser_value": self.local_parser_value,
            "requires_manual_review": self.requires_manual_review,
            "discrepancy": self.discrepancy,
        }


def audit_final_agreement(
    judge_final: FinalAdjudication | int | str | None,
    local_parser_final: int | float | str | Decimal | None,
) -> FinalAgreementAudit:
    """Audit, but never silently reconcile, judge/local-parser final estimates."""

    if isinstance(judge_final, FinalAdjudication):
        judge_value = judge_final.value if judge_final.status is KnowledgeStatus.KNOWN else None
    elif judge_final is None:
        judge_value = None
    else:
        judge_value = normalize_exact_integer(judge_final)
    local_value: int | None = None
    local_invalid = False
    if local_parser_final is not None:
        try:
            local_value = normalize_exact_integer(local_parser_final)
        except AdjudicationValidationError:
            local_invalid = True
    if judge_value is None:
        return FinalAgreementAudit(AgreementStatus.JUDGE_UNKNOWN, None, local_value, True, None)
    if local_invalid:
        return FinalAgreementAudit(AgreementStatus.LOCAL_INVALID, judge_value, None, True, None)
    if local_value is None:
        return FinalAgreementAudit(AgreementStatus.LOCAL_UNKNOWN, judge_value, None, True, None)
    discrepancy = judge_value - local_value
    status = AgreementStatus.AGREE if discrepancy == 0 else AgreementStatus.DISAGREE
    return FinalAgreementAudit(
        status=status,
        judge_value=judge_value,
        local_parser_value=local_value,
        requires_manual_review=discrepancy != 0,
        discrepancy=discrepancy,
    )


_SMOKE_NUMBER = re.compile(
    r"(?<![\w.])(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?P<suffix>thousand|million|billion|[kKmMbB]))?\b",
    re.IGNORECASE,
)
_SMOKE_MULTIPLIER = {
    None: Decimal(1),
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
}
_TARGET_CUE = re.compile(
    r"\b(?:estimate|answer|revise(?:d|s|ing)?|revision|settle(?:d|s|ing)?|"
    r"commit(?:ted|s|ting)?|conclude|conclusion|total|overall|obtain|result)\b",
    re.IGNORECASE,
)
_RANGE_CONNECTOR = re.compile(r"(?:-|\u2013|\u2014|\bto\b|\band\b)", re.IGNORECASE)


@dataclass(frozen=True)
class _SmokeMention:
    start: int
    end: int
    value: int


def _smoke_mentions(text: str) -> list[_SmokeMention]:
    mentions: list[_SmokeMention] = []
    for match in _SMOKE_NUMBER.finditer(text):
        decimal = Decimal(match.group("number").replace(",", ""))
        suffix = match.group("suffix")
        value = decimal * _SMOKE_MULTIPLIER[suffix.lower() if suffix else None]
        if value.is_finite() and value == value.to_integral_value():
            mentions.append(_SmokeMention(match.start(), match.end(), int(value)))
    return mentions


def _range_member_indices(text: str, mentions: Sequence[_SmokeMention]) -> set[int]:
    excluded: set[int] = set()
    for index in range(len(mentions) - 1):
        left, right = mentions[index], mentions[index + 1]
        between = text[left.end : right.start]
        prefix = text[max(0, left.start - 12) : left.start]
        if _RANGE_CONNECTOR.fullmatch(between.strip()) or re.search(
            r"\b(?:between|from)\s*$", prefix, re.IGNORECASE
        ):
            excluded.update((index, index + 1))
    return excluded


def _smoke_values_from_text(text: str, *, answer_mode: bool = False) -> list[int]:
    """Conservative fixture helper; it is intentionally not scientific extraction."""

    selected: list[int] = []
    # Newlines and punctuation followed by whitespace are sufficient for smoke
    # fixtures and do not split decimal points such as 29.25m.
    segments = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text)
    for segment in segments:
        mentions = _smoke_mentions(segment)
        if not mentions:
            continue
        excluded = _range_member_indices(segment, mentions)
        equals_position = segment.rfind("=")
        if equals_position >= 0:
            rhs = [
                mention.value
                for index, mention in enumerate(mentions)
                if index not in excluded and mention.start > equals_position
            ]
            if rhs:
                selected.append(rhs[-1])
            continue
        eligible = [
            mention.value for index, mention in enumerate(mentions) if index not in excluded
        ]
        if not eligible:
            continue
        if _TARGET_CUE.search(segment):
            selected.extend(eligible)
        elif answer_mode and len(eligible) == 1:
            selected.append(eligible[0])
    deduplicated: list[int] = []
    for value in selected:
        if not deduplicated or value != deduplicated[-1]:
            deduplicated.append(value)
    return deduplicated


class DeterministicSmokeCaller:
    """No-network pipeline fixture, explicitly forbidden for primary inference."""

    not_for_primary_inference = True

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="local-fixture",
            model_id="deterministic-smoke-adjudicator",
            model_revision="rules-v1",
            caller_version="1",
            metadata={"not_for_primary_inference": True},
        )

    def complete(self, request: AdjudicationRequest) -> str:
        trace = request.user_payload["trace"]
        answer = request.user_payload["answer"]
        trace_values = _smoke_values_from_text(trace)
        answer_values = _smoke_values_from_text(answer, answer_mode=True)
        values = trace_values + answer_values
        deduplicated: list[int] = []
        for value in values:
            if not deduplicated or value != deduplicated[-1]:
                deduplicated.append(value)
        if request.instrument_id == FINAL_INSTRUMENT_ID:
            if not deduplicated:
                return '{"status":"UNKNOWN","value":null}'
            return json.dumps(
                {"status": "KNOWN", "value": str(deduplicated[-1])},
                sort_keys=True,
                separators=(",", ":"),
            )
        if request.instrument_id == TRAJECTORY_INSTRUMENT_ID:
            if not deduplicated:
                return '{"status":"UNKNOWN","values":[]}'
            return json.dumps(
                {"status": "KNOWN", "values": [str(value) for value in deduplicated]},
                sort_keys=True,
                separators=(",", ":"),
            )
        raise ValueError(f"unknown adjudication instrument: {request.instrument_id}")


__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "FINAL_ANSWER_INSTRUMENT",
    "FINAL_ANSWER_INSTRUMENT_TEXT",
    "FINAL_INSTRUMENT_ID",
    "TRAJECTORY_INSTRUMENT",
    "TRAJECTORY_INSTRUMENT_ID",
    "TRAJECTORY_INSTRUMENT_TEXT",
    "AdjudicationCaller",
    "AdjudicationInstrument",
    "AdjudicationManifestRecord",
    "AdjudicationRequest",
    "AdjudicationValidationError",
    "AgreementStatus",
    "BlindedAdjudicationCase",
    "DeterministicSmokeCaller",
    "ExternalAdjudicationOutput",
    "ExternalAdjudicationOutputs",
    "FinalAdjudication",
    "FinalAgreementAudit",
    "JudgeProvenance",
    "KnowledgeStatus",
    "ManualOverrideAudit",
    "TrajectoryAdjudication",
    "audit_final_agreement",
    "blinded_case_from_rollout",
    "build_adjudication_request",
    "collect_adjudication_outputs",
    "materialize_adjudication",
    "materialize_primary_adjudication",
    "normalize_exact_integer",
    "parse_final_adjudication",
    "parse_trajectory_adjudication",
]
