"""All-final independent adjudication, exact consensus, and quality gates.

The primary judge and the independent judge see the same frozen, blinded final-
answer instrument.  Only exact ``status`` + ``value`` agreement on a known value
is analysis-usable.  Unknowns, missing independent judgments, and disagreements
remain explicit missing data; they are never reconciled by a local parser.

This module is provider-neutral.  It accepts the small :class:`AdjudicationCaller`
protocol and writes only strict numeric responses, public route identity, and an
allow-listed usage summary.  In particular, arbitrary provider metadata is not
copied into checkpoints.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    AdjudicationCaller,
    BlindedAdjudicationCase,
    JudgeProvenance,
    KnowledgeStatus,
    build_adjudication_request,
    normalize_exact_integer,
    parse_final_adjudication,
)
from model_forensics.io import assert_unique, sha256_file, stable_hash, write_json, write_jsonl
from model_forensics.prompts import QUESTIONS, Task

FINAL_CONSENSUS_PROTOCOL = "all-final-exact-consensus-v1"
QUALITY_GATE_PROTOCOL = "external-adjudication-quality-v1"
CHECKPOINT_SCHEMA_VERSION = 1


class ConsensusGateError(RuntimeError):
    """The preregistered independent-final agreement gate failed closed."""


class QualityGateError(RuntimeError):
    """A preregistered aggregate measurement-quality gate failed closed."""


class DualFinalConsensusCaller:
    """Adjudication-caller adapter returning only exact known two-route consensus.

    This adapter is intentionally final-only.  The primary route remains solely
    responsible for trajectory extraction elsewhere in the pipeline.
    """

    def __init__(
        self,
        primary: AdjudicationCaller,
        independent: AdjudicationCaller,
        *,
        minimum_exact_agreement: float = 0.90,
        minimum_known_consensus_rate: float = 0.95,
        on_audit: Callable[[Mapping[str, Any], FinalOnlyJudgment, FinalOnlyJudgment], None]
        | None = None,
    ) -> None:
        for name, value in (
            ("minimum_exact_agreement", minimum_exact_agreement),
            ("minimum_known_consensus_rate", minimum_known_consensus_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        primary_public = _public_provenance(primary.provenance)
        independent_public = _public_provenance(independent.provenance)
        if (
            primary_public["provider"],
            primary_public["model_id"],
            primary_public["model_revision"],
        ) == (
            independent_public["provider"],
            independent_public["model_id"],
            independent_public["model_revision"],
        ):
            raise ValueError("primary and independent final routes must be distinct")
        self._primary = primary
        self._independent = independent
        self._minimum_exact_agreement = float(minimum_exact_agreement)
        self._minimum_known_consensus_rate = float(minimum_known_consensus_rate)
        self._on_audit = on_audit
        self._audits: list[dict[str, Any]] = []
        self._primary_records: list[FinalOnlyJudgment] = []
        self._independent_records: list[FinalOnlyJudgment] = []
        self._request_ids: set[str] = set()

    @property
    def not_for_primary_inference(self) -> bool:
        return bool(
            self._primary.not_for_primary_inference or self._independent.not_for_primary_inference
        )

    @property
    def provenance(self) -> JudgeProvenance:
        primary = _public_provenance(self._primary.provenance)
        independent = _public_provenance(self._independent.provenance)
        summary = self.summary()
        latest_hash = self._audits[-1]["record_hash"] if self._audits else None
        return JudgeProvenance(
            provider="dual_route_exact_consensus",
            model_id=f"{primary['model_id']}||{independent['model_id']}",
            model_revision=None,
            caller_version=FINAL_CONSENSUS_PROTOCOL,
            decoding={"temperature": 0, "response_format": "json_object"},
            metadata={
                "primary_route": primary,
                "independent_route": independent,
                "trajectory_scope": "primary_route_only_outside_this_final_only_adapter",
                "minimum_exact_agreement": self._minimum_exact_agreement,
                "minimum_known_consensus_rate": self._minimum_known_consensus_rate,
                "completed_count": summary["expected_count"],
                "exact_status_value_agreement_rate": summary["exact_status_value_agreement_rate"],
                "known_consensus_rate": summary["known_consensus_rate"],
                "latest_consensus_record_hash": latest_hash,
            },
        )

    @property
    def audit_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._audits)

    @property
    def primary_records(self) -> tuple[FinalOnlyJudgment, ...]:
        return tuple(self._primary_records)

    @property
    def independent_records(self) -> tuple[FinalOnlyJudgment, ...]:
        return tuple(self._independent_records)

    def summary(self) -> dict[str, Any]:
        total = len(self._audits)
        exact = sum(bool(row["exact_status_value_agreement"]) for row in self._audits)
        known = sum(bool(row["known_consensus"]) for row in self._audits)
        exact_rate = exact / total if total else 0.0
        known_rate = known / total if total else 0.0
        payload: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": FINAL_CONSENSUS_PROTOCOL,
            "scope": "all_resampling_final_outcomes",
            "expected_count": total,
            "exact_status_value_agreements": exact,
            "exact_status_value_agreement_rate": exact_rate,
            "minimum_exact_status_value_agreement": self._minimum_exact_agreement,
            "known_consensus_count": known,
            "known_consensus_rate": known_rate,
            "minimum_known_consensus_rate": self._minimum_known_consensus_rate,
            "agreement_gate_passed": bool(total and exact_rate >= self._minimum_exact_agreement),
            "known_consensus_gate_passed": bool(
                total and known_rate >= self._minimum_known_consensus_rate
            ),
            "audit_rows_hash": stable_hash(self._audits),
            "primary_record_hashes": [record.record_hash for record in self._primary_records],
            "independent_record_hashes": [
                record.record_hash for record in self._independent_records
            ],
        }
        payload["gate_passed"] = bool(
            payload["agreement_gate_passed"] and payload["known_consensus_gate_passed"]
        )
        payload["manifest_hash"] = stable_hash(payload)
        return payload

    def require_quality_gates(self, *, expected_count: int | None = None) -> dict[str, Any]:
        summary = self.summary()
        if expected_count is not None and summary["expected_count"] != expected_count:
            raise ConsensusGateError(
                "dual-final caller did not adjudicate every expected final: "
                f"{summary['expected_count']} != {expected_count}"
            )
        if not summary["agreement_gate_passed"]:
            raise ConsensusGateError(
                "all-final exact status+value agreement gate failed closed: "
                f"{summary['exact_status_value_agreement_rate']:.3f} < "
                f"{self._minimum_exact_agreement:.3f}"
            )
        if not summary["known_consensus_gate_passed"]:
            raise QualityGateError(
                "all-final known consensus rate gate failed closed: "
                f"{summary['known_consensus_rate']:.3f} < "
                f"{self._minimum_known_consensus_rate:.3f}"
            )
        return summary

    def complete(self, request: Any) -> str:
        if request.instrument_id != FINAL_ANSWER_INSTRUMENT.instrument_id:
            raise ValueError("dual-final consensus caller accepts only the frozen final instrument")
        if self.not_for_primary_inference:
            raise ValueError("dual-final primary consensus refuses non-primary routes")
        if request.request_id in self._request_ids:
            raise ValueError("dual-final consensus caller refuses duplicate request IDs")
        primary_raw = self._primary.complete(request)
        primary_parsed = parse_final_adjudication(primary_raw)
        primary_provenance = self._primary.provenance
        independent_raw = self._independent.complete(request)
        independent_parsed = parse_final_adjudication(independent_raw)
        independent_provenance = self._independent.provenance
        case_hash = stable_hash(dict(request.user_payload))
        primary_record = FinalOnlyJudgment(
            unit_id=request.request_id,
            case_hash=case_hash,
            request_id=request.request_id,
            instrument_hash=request.instrument_hash,
            raw_response=primary_raw,
            status=primary_parsed.status.value,
            value=primary_parsed.value,
            public_provenance=_public_provenance(primary_provenance),
            usage=_public_usage(primary_provenance),
        )
        independent_record = FinalOnlyJudgment(
            unit_id=request.request_id,
            case_hash=case_hash,
            request_id=request.request_id,
            instrument_hash=request.instrument_hash,
            raw_response=independent_raw,
            status=independent_parsed.status.value,
            value=independent_parsed.value,
            public_provenance=_public_provenance(independent_provenance),
            usage=_public_usage(independent_provenance),
        )
        exact = bool(
            primary_parsed.status is independent_parsed.status
            and primary_parsed.value == independent_parsed.value
        )
        known = bool(exact and primary_parsed.status is KnowledgeStatus.KNOWN)
        audit = {
            "protocol_version": FINAL_CONSENSUS_PROTOCOL,
            "request_id": request.request_id,
            "case_hash": case_hash,
            "instrument_hash": request.instrument_hash,
            "primary_record_hash": primary_record.record_hash,
            "independent_record_hash": independent_record.record_hash,
            "primary": {
                "status": primary_parsed.status.value,
                "value": primary_parsed.value,
            },
            "independent": {
                "status": independent_parsed.status.value,
                "value": independent_parsed.value,
            },
            "exact_status_value_agreement": exact,
            "known_consensus": known,
            "returned_status": "KNOWN" if known else "UNKNOWN",
            "returned_value": primary_parsed.value if known else None,
        }
        audit["record_hash"] = stable_hash(audit)
        self._primary_records.append(primary_record)
        self._independent_records.append(independent_record)
        self._audits.append(audit)
        self._request_ids.add(request.request_id)
        if self._on_audit is not None:
            self._on_audit(audit, primary_record, independent_record)
        if known:
            return primary_raw
        return json.dumps(
            {"status": "UNKNOWN", "value": None},
            separators=(",", ":"),
            sort_keys=True,
        )


_PUBLIC_DECODING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "response_format",
        "explicit_reasoning",
        "seed",
        "preflight_input_bound",
    }
)
_PUBLIC_USAGE_FIELDS = frozenset(
    {
        "calls_completed",
        "purpose",
        "logical_request_hash",
        "provider_response_id_hash",
        "request_id_hash",
        "response_model",
        "response_provider",
        "attempts_used",
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "reported_cost_usd",
        "computed_cost_usd",
        "charged_cost_usd",
        "cost_usd",
        "preflight_upper_bound_usd",
        "api_total_usd",
        "replayed_from_checkpoint",
        "paid_response_checkpoint_hash",
    }
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.split(":", 1)[1]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _public_provenance(value: JudgeProvenance | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, JudgeProvenance):
        source = value.to_dict()
    elif isinstance(value, Mapping):
        source = dict(value)
    else:  # pragma: no cover - protocol/type guard
        raise TypeError("judge provenance must be JudgeProvenance or a mapping")
    decoding_source = source.get("decoding", {})
    decoding = (
        {str(key): item for key, item in decoding_source.items() if key in _PUBLIC_DECODING_FIELDS}
        if isinstance(decoding_source, Mapping)
        else {}
    )
    public = {
        "provider": source.get("provider"),
        "model_id": source.get("model_id"),
        "model_revision": source.get("model_revision"),
        "caller_version": source.get("caller_version"),
        "decoding": decoding,
    }
    if not isinstance(public["provider"], str) or not public["provider"]:
        raise ValueError("judge provenance requires a public provider")
    if not isinstance(public["model_id"], str) or not public["model_id"]:
        raise ValueError("judge provenance requires a public model_id")
    stable_hash(public)
    return public


def _public_usage(value: JudgeProvenance | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, JudgeProvenance):
        source: Mapping[str, Any] = value.metadata
    else:
        raw_metadata = value.get("metadata", {})
        source = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    usage = {str(key): item for key, item in source.items() if key in _PUBLIC_USAGE_FIELDS}
    stable_hash(usage)
    return usage


def _task_question(task: Any) -> str:
    try:
        normalized = Task(str(task))
    except ValueError as exc:
        raise ValueError(f"unsupported final-adjudication task: {task!r}") from exc
    return QUESTIONS[normalized]


def _case_from_row(row: Mapping[str, Any]) -> BlindedAdjudicationCase:
    trace = row.get("trace", row.get("reasoning", row.get("full_trace", "")))
    answer = row.get("answer", "")
    if not isinstance(trace, str) or not isinstance(answer, str):
        raise TypeError("final-adjudication trace and answer must be strings")
    return BlindedAdjudicationCase(
        task_question=_task_question(row.get("task")),
        trace=trace,
        answer=answer,
    )


def _resolve_id_field(rows: Sequence[Mapping[str, Any]], requested: str | None) -> str:
    if requested is not None:
        if not requested:
            raise ValueError("id_field cannot be empty")
        return requested
    if all("run_id" in row for row in rows):
        return "run_id"
    if all("resample_id" in row for row in rows):
        return "resample_id"
    raise ValueError("cannot infer a common run_id or resample_id field")


@dataclass(frozen=True, slots=True)
class FinalOnlyJudgment:
    """One strict final-only response plus secret-safe audit metadata."""

    unit_id: str
    case_hash: str
    request_id: str
    instrument_hash: str
    raw_response: str
    status: str
    value: int | None
    public_provenance: Mapping[str, Any]
    usage: Mapping[str, Any]
    protocol_version: str = FINAL_CONSENSUS_PROTOCOL

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise ValueError("final-only judgment requires a nonempty unit_id")
        for name in ("case_hash", "request_id", "instrument_hash"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a namespaced SHA-256 hash")
        parsed = parse_final_adjudication(self.raw_response)
        status = KnowledgeStatus(self.status)
        value = None if self.value is None else normalize_exact_integer(self.value)
        if parsed.status is not status or parsed.value != value:
            raise ValueError("stored final-only status/value does not match strict raw response")
        public_provenance = _public_provenance(self.public_provenance)
        usage = {str(key): item for key, item in self.usage.items() if key in _PUBLIC_USAGE_FIELDS}
        stable_hash(usage)
        object.__setattr__(self, "status", status.value)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "public_provenance", public_provenance)
        object.__setattr__(self, "usage", usage)

    @property
    def response_hash(self) -> str:
        return stable_hash({"raw_response": self.raw_response})

    @property
    def provenance_hash(self) -> str:
        return stable_hash(dict(self.public_provenance))

    @property
    def usage_hash(self) -> str:
        return stable_hash(dict(self.usage))

    def manifest_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "protocol_version": self.protocol_version,
            "unit_id": self.unit_id,
            "case_hash": self.case_hash,
            "request_id": self.request_id,
            "instrument_hash": self.instrument_hash,
            "response_hash": self.response_hash,
            "status": self.status,
            "value": self.value,
            "public_provenance": dict(self.public_provenance),
            "provenance_hash": self.provenance_hash,
            "usage_hash": self.usage_hash,
        }
        if include_hash:
            payload["record_hash"] = stable_hash(payload)
        return payload

    @property
    def record_hash(self) -> str:
        return stable_hash(self.manifest_dict(include_hash=False))

    def raw_dict(self) -> dict[str, Any]:
        payload = {
            "unit_id": self.unit_id,
            "case_hash": self.case_hash,
            "request_id": self.request_id,
            "instrument_hash": self.instrument_hash,
            "raw_response": self.raw_response,
            "response_hash": self.response_hash,
        }
        payload["record_hash"] = stable_hash(payload)
        return payload

    def usage_dict(self) -> dict[str, Any]:
        payload = {
            "unit_id": self.unit_id,
            "request_id": self.request_id,
            "public_provenance": dict(self.public_provenance),
            "provenance_hash": self.provenance_hash,
            "usage": dict(self.usage),
            "usage_hash": self.usage_hash,
        }
        payload["record_hash"] = stable_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class FinalOnlyBatch:
    records: tuple[FinalOnlyJudgment, ...]

    def __post_init__(self) -> None:
        identifiers = [record.unit_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("final-only batch contains duplicate unit IDs")

    def summary(self) -> dict[str, Any]:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "protocol_version": FINAL_CONSENSUS_PROTOCOL,
            "scope": "all_final_outcomes",
            "record_count": len(self.records),
            "unit_ids_hash": stable_hash([record.unit_id for record in self.records]),
            "record_hashes": [record.record_hash for record in self.records],
            "instrument_hashes": sorted({record.instrument_hash for record in self.records}),
            "judge_provenance_hashes": sorted({record.provenance_hash for record in self.records}),
        }
        payload["manifest_hash"] = stable_hash(payload)
        return payload


def collect_independent_final_judgments(
    rows: Sequence[Mapping[str, Any]],
    *,
    caller: AdjudicationCaller,
    id_field: str | None = None,
    primary_inference: bool = True,
    on_judgment: Callable[[FinalOnlyJudgment], None] | None = None,
) -> FinalOnlyBatch:
    """Judge every supplied final with only question, trace, and answer visible."""

    if primary_inference and caller.not_for_primary_inference:
        raise ValueError("primary consensus refuses a not_for_primary_inference caller")
    resolved_id_field = _resolve_id_field(rows, id_field)
    seen: set[str] = set()
    records: list[FinalOnlyJudgment] = []
    for source in rows:
        unit_id = source.get(resolved_id_field)
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"every row requires nonempty {resolved_id_field}")
        if unit_id in seen:
            raise ValueError(f"duplicate {resolved_id_field}: {unit_id}")
        seen.add(unit_id)
        case = _case_from_row(source)
        request = build_adjudication_request(case, FINAL_ANSWER_INSTRUMENT)
        raw_response = caller.complete(request)
        parsed = parse_final_adjudication(raw_response)
        provenance = caller.provenance
        record = FinalOnlyJudgment(
            unit_id=unit_id,
            case_hash=case.case_hash,
            request_id=request.request_id,
            instrument_hash=request.instrument_hash,
            raw_response=raw_response,
            status=parsed.status.value,
            value=parsed.value,
            public_provenance=_public_provenance(provenance),
            usage=_public_usage(provenance),
        )
        records.append(record)
        if on_judgment is not None:
            on_judgment(record)
    return FinalOnlyBatch(tuple(records))


class FinalOnlyCheckpoint:
    """Durably rewrite a compact audit bundle after each paid final judgment."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: list[FinalOnlyJudgment] = []

    @property
    def records(self) -> tuple[FinalOnlyJudgment, ...]:
        return tuple(self._records)

    def append(self, record: FinalOnlyJudgment) -> None:
        if any(existing.unit_id == record.unit_id for existing in self._records):
            raise ValueError(f"duplicate final-only checkpoint ID: {record.unit_id}")
        self._records.append(record)
        self.flush()

    def flush(self) -> dict[str, Any]:
        batch = FinalOnlyBatch(tuple(self._records))
        raw_path = write_jsonl(
            self.directory / "independent_final_raw.jsonl",
            (record.raw_dict() for record in self._records),
        )
        usage_path = write_jsonl(
            self.directory / "independent_final_usage.jsonl",
            (record.usage_dict() for record in self._records),
        )
        records_path = write_jsonl(
            self.directory / "independent_final_manifest.jsonl",
            (record.manifest_dict() for record in self._records),
        )
        payload = {
            **batch.summary(),
            "completed_count": len(self._records),
            "artifacts": {
                "raw": {"path": raw_path.name, "sha256": sha256_file(raw_path)},
                "usage": {"path": usage_path.name, "sha256": sha256_file(usage_path)},
                "manifest": {"path": records_path.name, "sha256": sha256_file(records_path)},
            },
        }
        payload.pop("manifest_hash", None)
        payload["manifest_hash"] = stable_hash(payload)
        write_json(self.directory / "checkpoint_manifest.json", payload)
        return payload


def _primary_final(row: Mapping[str, Any]) -> tuple[KnowledgeStatus, int | None]:
    valid = row.get("final_measurement_valid")
    value = row.get("final_estimate")
    if valid is None:
        valid = value is not None
    if type(valid) is not bool:
        raise ValueError("final_measurement_valid must be boolean when present")
    if valid != (value is not None):
        raise ValueError("primary final validity does not match final_estimate availability")
    if not valid:
        return KnowledgeStatus.UNKNOWN, None
    return KnowledgeStatus.KNOWN, normalize_exact_integer(value)


def _consensus_reason(
    primary_status: KnowledgeStatus,
    primary_value: int | None,
    independent: FinalOnlyJudgment | None,
) -> tuple[bool, bool, str | None]:
    if independent is None:
        return False, False, "missing_independent_final"
    secondary_status = KnowledgeStatus(independent.status)
    exact = primary_status is secondary_status and primary_value == independent.value
    known_consensus = bool(exact and secondary_status is KnowledgeStatus.KNOWN)
    if known_consensus:
        return True, True, None
    if secondary_status is KnowledgeStatus.UNKNOWN:
        return exact, False, "independent_final_unknown"
    if primary_status is KnowledgeStatus.UNKNOWN:
        return exact, False, "primary_final_unknown"
    return exact, False, "final_judge_disagreement"


@dataclass(frozen=True, slots=True)
class ConsensusBatch:
    rows: tuple[dict[str, Any], ...]
    audit_rows: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]


def apply_all_final_consensus(
    rows: Sequence[Mapping[str, Any]],
    independent: Sequence[FinalOnlyJudgment],
    *,
    minimum_exact_agreement: float = 0.90,
    id_field: str | None = None,
    enforce_gate: bool = True,
) -> ConsensusBatch:
    """Replace primary outcomes with exact two-route known consensus outcomes.

    The aggregate gate denominator is every supplied final.  A missing independent
    row therefore counts against agreement.  Exact ``UNKNOWN``/``UNKNOWN`` status
    agreement is reported as agreement, but still produces missing outcome data.
    """

    if (
        isinstance(minimum_exact_agreement, bool)
        or not isinstance(minimum_exact_agreement, (int, float))
        or not math.isfinite(float(minimum_exact_agreement))
        or not 0 <= minimum_exact_agreement <= 1
    ):
        raise ValueError("minimum_exact_agreement must be finite and in [0, 1]")
    resolved_id_field = _resolve_id_field(rows, id_field)
    assert_unique(rows, resolved_id_field)
    independent_by_id: dict[str, FinalOnlyJudgment] = {}
    for record in independent:
        if record.unit_id in independent_by_id:
            raise ValueError(f"duplicate independent final ID: {record.unit_id}")
        independent_by_id[record.unit_id] = record
    expected_ids = {str(row[resolved_id_field]) for row in rows}
    extra = set(independent_by_id) - expected_ids
    if extra:
        raise ValueError(f"independent judgments contain unexpected IDs: {sorted(extra)!r}")

    calibrated: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    exact_count = 0
    known_count = 0
    missing_count = 0
    secondary_unknown_count = 0
    disagreement_count = 0
    for source in rows:
        unit_id = str(source[resolved_id_field])
        primary_status, primary_value = _primary_final(source)
        secondary = independent_by_id.get(unit_id)
        if (
            secondary is not None
            and "task" in source
            and ("trace" in source or "reasoning" in source or "full_trace" in source)
        ):
            expected_case_hash = _case_from_row(source).case_hash
            if secondary.case_hash != expected_case_hash:
                raise ValueError(f"independent final belongs to a different case: {unit_id}")
        exact, known_consensus, reason = _consensus_reason(
            primary_status,
            primary_value,
            secondary,
        )
        exact_count += int(exact)
        known_count += int(known_consensus)
        missing_count += int(secondary is None)
        secondary_unknown_count += int(
            secondary is not None and secondary.status == KnowledgeStatus.UNKNOWN.value
        )
        disagreement_count += int(secondary is not None and not exact)
        audit = {
            "protocol_version": FINAL_CONSENSUS_PROTOCOL,
            "unit_id": unit_id,
            "primary": {"status": primary_status.value, "value": primary_value},
            "independent": (
                None
                if secondary is None
                else {"status": secondary.status, "value": secondary.value}
            ),
            "independent_record_hash": None if secondary is None else secondary.record_hash,
            "exact_status_value_agreement": exact,
            "known_consensus": known_consensus,
            "missing_reason": reason,
        }
        audit["record_hash"] = stable_hash(audit)
        audits.append(audit)

        row = dict(source)
        row.pop("record_hash", None)
        row["final_consensus"] = dict(audit)
        row["final_consensus_valid"] = known_consensus
        row["independent_final_status"] = None if secondary is None else secondary.status
        row["independent_final_estimate"] = None if secondary is None else secondary.value
        row["independent_final_record_hash"] = None if secondary is None else secondary.record_hash
        if not known_consensus:
            row["final_estimate"] = None
            row["final_measurement_valid"] = False
            if "final_good_side" in row:
                row["final_good_side"] = None
            if "signed_log_ratio_final" in row:
                row["signed_log_ratio_final"] = None
            if "trajectory_measurement_valid" in row:
                row["trajectory_measurement_valid"] = False
                row["trajectory_invalid_reason"] = "final_consensus_missing"
            if "requires_blinded_manual_review" in row:
                row["requires_blinded_manual_review"] = True
            if "valid" in row:
                row["valid"] = False
            if "primary_eligible" in row:
                row["primary_eligible"] = False
            if "confirmatory_eligible" in row:
                row["confirmatory_eligible"] = False
            if "analysis_tier" in row:
                row["analysis_tier"] = "outcome_unmeasured"
            row["invalid_reason"] = reason
        else:
            row["final_estimate"] = primary_value
            row["final_measurement_valid"] = True
            if "valid" in row:
                row["valid"] = True
            row["invalid_reason"] = None
        row["record_hash"] = stable_hash(row)
        calibrated.append(row)

    total = len(rows)
    agreement_rate = exact_count / total if total else 0.0
    gate_passed = bool(total and agreement_rate >= minimum_exact_agreement)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": FINAL_CONSENSUS_PROTOCOL,
        "scope": "all_final_outcomes",
        "id_field": resolved_id_field,
        "expected_count": total,
        "independent_count": len(independent),
        "exact_status_value_agreements": exact_count,
        "exact_status_value_agreement_rate": agreement_rate,
        "minimum_exact_status_value_agreement": float(minimum_exact_agreement),
        "known_consensus_count": known_count,
        "known_consensus_rate": known_count / total if total else 0.0,
        "missing_independent_count": missing_count,
        "independent_unknown_count": secondary_unknown_count,
        "disagreement_count": disagreement_count,
        "gate_passed": gate_passed,
        "consensus_rows_hash": stable_hash(calibrated),
        "audit_rows_hash": stable_hash(audits),
        "independent_records_hash": stable_hash([record.record_hash for record in independent]),
    }
    summary["manifest_hash"] = stable_hash(summary)
    result = ConsensusBatch(tuple(calibrated), tuple(audits), summary)
    if enforce_gate and not gate_passed:
        raise ConsensusGateError(
            "all-final exact status+value agreement gate failed closed: "
            f"{agreement_rate:.3f} < {minimum_exact_agreement:.3f}"
        )
    return result


def _quality_phase(row: Mapping[str, Any]) -> str:
    return "baseline" if row.get("condition") == "baseline" else "treatment"


def _trajectory_consistent(row: Mapping[str, Any], final_value: int | None) -> bool:
    if final_value is None or row.get("trajectory_measurement_valid") is not True:
        return False
    values = row.get("trajectory_values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return False
    try:
        return normalize_exact_integer(values[-1]) == final_value
    except (TypeError, ValueError):
        return False


def evaluate_adjudication_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_exact_agreement: float = 0.90,
    minimum_final_known_rate: float = 0.95,
    minimum_trajectory_final_consistency: float = 0.95,
    required_phases: Sequence[str] = ("baseline", "treatment"),
    enforce: bool = False,
) -> dict[str, Any]:
    """Evaluate final-known and trajectory/final consistency by phase and cell.

    Both rates use every row in the phase as their denominator.  This conservative
    definition means an unknown trajectory cannot disappear from the consistency
    denominator after the fact.  The same gates are also required separately in
    every task x condition cell, preventing a high-quality cell from masking a
    failed scientific contrast cell in an aggregate phase rate.
    """

    for name, value in (
        ("minimum_exact_agreement", minimum_exact_agreement),
        ("minimum_final_known_rate", minimum_final_known_rate),
        ("minimum_trajectory_final_consistency", minimum_trajectory_final_consistency),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
    phases = tuple(str(phase) for phase in required_phases)
    if not phases or any(not phase for phase in phases) or len(set(phases)) != len(phases):
        raise ValueError("required_phases must be unique and nonempty")

    phase_rows = {phase: [] for phase in phases}
    for row in rows:
        phase = _quality_phase(row)
        if phase in phase_rows:
            phase_rows[phase].append(row)

    def measurement_report(
        selected: Sequence[Mapping[str, Any]],
        *,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        total = len(selected)
        final_known = 0
        consistent = 0
        consensus_marked = 0
        exact_agreements = 0
        for row in selected:
            consensus_valid = row.get("final_consensus_valid") is True and isinstance(
                row.get("final_consensus"), Mapping
            )
            consensus_marked += int("final_consensus_valid" in row and "final_consensus" in row)
            final_consensus = row.get("final_consensus")
            exact_agreements += int(
                isinstance(final_consensus, Mapping)
                and final_consensus.get("exact_status_value_agreement") is True
            )
            final_value: int | None = None
            if consensus_valid and row.get("final_measurement_valid") is True:
                try:
                    final_value = normalize_exact_integer(row.get("final_estimate"))
                except (TypeError, ValueError):
                    final_value = None
            final_known += int(final_value is not None)
            consistent += int(_trajectory_consistent(row, final_value))
        final_rate = final_known / total if total else 0.0
        consistency_rate = consistent / total if total else 0.0
        exact_rate = exact_agreements / total if total else 0.0
        return {
            **dict(identity),
            "row_count": total,
            "consensus_marked_count": consensus_marked,
            "exact_status_value_agreement_count": exact_agreements,
            "exact_status_value_agreement_rate": exact_rate,
            "final_known_count": final_known,
            "final_known_rate": final_rate,
            "trajectory_final_consistent_count": consistent,
            "trajectory_final_consistency_rate": consistency_rate,
            "gate_passed": bool(
                total
                and consensus_marked == total
                and exact_rate >= minimum_exact_agreement
                and final_rate >= minimum_final_known_rate
                and consistency_rate >= minimum_trajectory_final_consistency
            ),
        }

    reports: list[dict[str, Any]] = []
    all_passed = True
    for phase in phases:
        selected = phase_rows[phase]
        report = measurement_report(selected, identity={"phase": phase})
        all_passed = all_passed and bool(report["gate_passed"])
        reports.append(report)

    cell_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if _quality_phase(row) not in phase_rows:
            continue
        task = row.get("task")
        condition = row.get("condition")
        if not isinstance(task, str) or not task or not isinstance(condition, str) or not condition:
            raise ValueError("quality-gated rows require nonempty task and condition")
        cell_rows.setdefault((task, condition), []).append(row)
    cell_reports = [
        measurement_report(
            cell_rows[(task, condition)],
            identity={
                "task": task,
                "condition": condition,
                "phase": _quality_phase(cell_rows[(task, condition)][0]),
            },
        )
        for task, condition in sorted(cell_rows)
    ]
    all_passed = (
        all_passed
        and bool(cell_reports)
        and all(bool(report["gate_passed"]) for report in cell_reports)
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": QUALITY_GATE_PROTOCOL,
        "rate_denominator": "all_rows_in_each_phase_or_task_x_condition_cell",
        "minimum_exact_status_value_agreement": float(minimum_exact_agreement),
        "minimum_final_known_rate": float(minimum_final_known_rate),
        "minimum_trajectory_final_consistency": float(minimum_trajectory_final_consistency),
        "phases": reports,
        "cell_scope": "task_x_condition",
        "cells": cell_reports,
        "gate_passed": all_passed,
        "source_rows_hash": stable_hash(rows),
    }
    payload["manifest_hash"] = stable_hash(payload)
    if enforce and not all_passed:
        failed_phases = [report["phase"] for report in reports if not report["gate_passed"]]
        failed_cells = [
            f"{report['task']}:{report['condition']}"
            for report in cell_reports
            if not report["gate_passed"]
        ]
        failed = ", ".join([*failed_phases, *failed_cells])
        raise QualityGateError(f"external adjudication quality gate failed closed: {failed}")
    return payload


def freeze_consensus_baseline_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    minimum_final_known_rate: float = 0.95,
    minimum_trajectory_final_consistency: float = 0.95,
) -> dict[str, Any]:
    """Freeze a data-derived baseline median only after consensus quality passes."""

    baseline = [
        row for row in rows if row.get("task") == task and row.get("condition") == "baseline"
    ]
    if not baseline:
        raise QualityGateError(f"task {task!r} has no baseline rows")
    if any("final_consensus_valid" not in row or "final_consensus" not in row for row in baseline):
        raise QualityGateError("baseline threshold freeze requires all-final consensus records")
    quality = evaluate_adjudication_quality(
        baseline,
        minimum_final_known_rate=minimum_final_known_rate,
        minimum_trajectory_final_consistency=minimum_trajectory_final_consistency,
        required_phases=("baseline",),
        enforce=True,
    )
    values = [
        normalize_exact_integer(row["final_estimate"])
        for row in baseline
        if row.get("final_consensus_valid") is True and row.get("final_estimate") is not None
    ]
    if not values:  # pragma: no cover - implied by the >0 quality gate
        raise QualityGateError("no known consensus baseline finals remain")
    threshold = float(statistics.median(values))
    if not math.isfinite(threshold) or threshold <= 0:
        raise QualityGateError("consensus baseline median must be positive and finite")
    id_field = _resolve_id_field(baseline, None)
    source_ids = [
        str(row[id_field])
        for row in baseline
        if row.get("final_consensus_valid") is True and row.get("final_estimate") is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": FINAL_CONSENSUS_PROTOCOL,
        "task": task,
        "condition": "baseline",
        "threshold_rule": "median_of_known_exact_final_consensus",
        "threshold": threshold,
        "eligible_baseline_count": len(baseline),
        "source_count": len(values),
        "source_ids": source_ids,
        "source_ids_hash": stable_hash(source_ids),
        "source_rows_hash": stable_hash(baseline),
        "quality_gate": quality,
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "FINAL_CONSENSUS_PROTOCOL",
    "QUALITY_GATE_PROTOCOL",
    "ConsensusBatch",
    "ConsensusGateError",
    "DualFinalConsensusCaller",
    "FinalOnlyBatch",
    "FinalOnlyCheckpoint",
    "FinalOnlyJudgment",
    "QualityGateError",
    "apply_all_final_consensus",
    "collect_independent_final_judgments",
    "evaluate_adjudication_quality",
    "freeze_consensus_baseline_threshold",
]
