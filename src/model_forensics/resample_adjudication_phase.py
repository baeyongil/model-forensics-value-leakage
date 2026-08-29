"""Authenticated CPU/API adjudication for completed resampling generations.

The phase accepts only a complete, content-addressed GPU-generation bundle.  It
then runs local semantic checks, the frozen replacement classifier, and two
independent final-only judges through injected provider-neutral protocols.  One
atomic record is committed per frozen allocation, so a resumed invocation skips
completed paid work while an interrupted unit can rely on the callers' durable
paid-response replay stores.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.adjudication import (
    FINAL_ANSWER_INSTRUMENT,
    AdjudicationCaller,
    AdjudicationRequest,
    AdjudicationValidationError,
    JudgeProvenance,
    KnowledgeStatus,
    parse_final_adjudication,
)
from model_forensics.anchors import AnchorManifest, validate_anchor_manifest
from model_forensics.io import read_json, read_jsonl, sha256_file, stable_hash, write_json
from model_forensics.paid_response_store import PAID_RESPONSE_STORE_PROTOCOL
from model_forensics.record_checkpoint import RecordCheckpointError, RecordCheckpointStore
from model_forensics.resample_phases import (
    GENERATION_STATUS_TERMINAL_INVALID,
    GENERATION_STATUS_VALID,
    ResamplingGenerationRecord,
    _authenticate_intermediate_for_cpu,
    adjudicate_sentence_resampling_intermediates,
)
from model_forensics.resample_runner import (
    BaseTrace,
    NeutralControlSpec,
    ReplacementClassificationError,
    ReplacementClassificationRequest,
    ReplacementClassificationResult,
    ReplacementClassifier,
    ReplacementTokenTolerance,
    ResampleAllocationManifest,
    ResamplingArtifactRecord,
    _coerce_base_trace,
    _validate_execution_manifest,
    build_fixed_stage_two_allocation_manifest,
)
from model_forensics.resampling import TextEmbedder

RESAMPLE_ADJUDICATION_PROTOCOL = "resample-cpu-api-adjudication-v1"
EXPECTED_RESAMPLE_COUNT = 24 * 2 * 20
MINIMUM_EXACT_AGREEMENT_FLOOR = 0.90
MINIMUM_FINAL_KNOWN_FLOOR = 0.95
DEFAULT_MINIMUM_OVERALL_GENERATION_VALID_RATE = 0.95
DEFAULT_MINIMUM_ANCHOR_ARM_VALID_COUNT = 18
DEFAULT_MINIMUM_ANCHOR_PAIR_COMPLETE_COUNT = 16
DEFAULT_MAXIMUM_ANCHOR_ARM_VALID_RATE_GAP = 0.10


class ResampleAdjudicationPhaseError(RuntimeError):
    """A source, route, checkpoint, or scientific contract failed closed."""


class ResampleAdjudicationGateError(ResampleAdjudicationPhaseError):
    """The complete phase failed a preregistered aggregate final-quality gate."""


@dataclass(frozen=True, slots=True)
class AuthenticatedResampleGeneration:
    rows: tuple[ResamplingGenerationRecord, ...]
    initial_rows: tuple[ResamplingGenerationRecord, ...]
    stage_two_rows: tuple[ResamplingGenerationRecord, ...]
    manifest: Mapping[str, Any]
    stage_manifests: Mapping[str, Mapping[str, Any]]
    source_hash: str
    valid_generation_count: int
    terminal_invalid_count: int


@dataclass(frozen=True, slots=True)
class ResampleAdjudicationPhase:
    rows: tuple[dict[str, Any], ...]
    quality_gate: Mapping[str, Any]
    manifest: Mapping[str, Any]
    complete: bool
    gate_passed: bool


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _valid_content_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.split(":", 1)[1]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _safe_local_artifact(directory: Path, declared: Any, expected_name: str) -> Path:
    if not isinstance(declared, str) or Path(declared).name != expected_name:
        raise ResampleAdjudicationPhaseError(
            f"GPU generation manifest has an invalid {expected_name!r} path"
        )
    path = directory / expected_name
    if not path.is_file():
        raise ResampleAdjudicationPhaseError(f"GPU generation artifact is absent: {expected_name}")
    return path


def _load_generation_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "gpu_generation_manifest.json"
    if not path.is_file():
        raise ResampleAdjudicationPhaseError("GPU generation manifest is absent")
    manifest = read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_hash") != stable_hash(_without_hash(manifest, "manifest_hash"))
        or manifest.get("schema_version") != 1
        or manifest.get("phase_contract") != "resample-gpu-only-v1"
        or manifest.get("complete") is not True
        or manifest.get("api_calls_performed") != 0
    ):
        raise ResampleAdjudicationPhaseError(
            "GPU generation manifest identity or content hash mismatch"
        )
    return manifest


def _load_stage_final(
    directory: Path,
    *,
    stage: str,
    allocation: ResampleAllocationManifest,
    expected_plan_hash: str,
    expected_checkpoint_manifest_hash: str | None,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any]]:
    stage_directory = directory / stage
    plan_path = stage_directory / "checkpoint_plan.json"
    if not plan_path.is_file():
        raise ResampleAdjudicationPhaseError(f"{stage} generation checkpoint plan is absent")
    plan = read_json(plan_path)
    if (
        not isinstance(plan, dict)
        or plan.get("plan_hash") != stable_hash(_without_hash(plan, "plan_hash"))
        or plan.get("plan_hash") != expected_plan_hash
        or plan.get("id_field") != "resample_id"
        or not isinstance(plan.get("payload"), Mapping)
    ):
        raise ResampleAdjudicationPhaseError(
            f"{stage} generation checkpoint plan failed authentication"
        )
    payload = dict(plan["payload"])
    if (
        payload.get("stage") != stage
        or payload.get("allocation_manifest_hash") != allocation.manifest_hash
    ):
        raise ResampleAdjudicationPhaseError(
            f"{stage} checkpoint plan disagrees with the frozen allocation"
        )
    try:
        store = RecordCheckpointStore(
            stage_directory,
            id_field="resample_id",
            plan_payload=payload,
        )
        final = store.load_final(
            expected_ids=tuple(item.request_id for item in allocation.allocations)
        )
    except (RecordCheckpointError, ValueError) as exc:
        raise ResampleAdjudicationPhaseError(
            f"{stage} generation checkpoint final failed authentication"
        ) from exc
    if (
        expected_checkpoint_manifest_hash is not None
        and final.manifest.get("manifest_hash") != expected_checkpoint_manifest_hash
    ):
        raise ResampleAdjudicationPhaseError(
            f"{stage} checkpoint final disagrees with the GPU manifest"
        )
    return final.rows, final.manifest


def _base_trace_payload(base: BaseTrace) -> dict[str, Any]:
    return {
        "base_trace_id": base.base_trace_id,
        "prompt": base.prompt,
        "trace": base.trace,
        "threshold": base.threshold,
        "condition": base.condition,
        "task": base.task,
        "messages": [dict(message) for message in base.messages],
        "provenance": dict(base.provenance),
    }


def load_authenticated_resample_generation(
    *,
    generation_checkpoint_dir: str | Path,
    anchors: AnchorManifest,
    base_traces: Mapping[str, BaseTrace | Mapping[str, Any]],
    initial_allocation_manifest: ResampleAllocationManifest,
    stage_two_allocation_manifest: ResampleAllocationManifest,
) -> AuthenticatedResampleGeneration:
    """Authenticate the complete 24 x 2 x 20 GPU inventory without paid calls."""

    try:
        validate_anchor_manifest(anchors)
    except (TypeError, ValueError) as exc:
        raise ResampleAdjudicationPhaseError("frozen anchor manifest failed validation") from exc
    if len(anchors.anchors) != 24:
        raise ResampleAdjudicationPhaseError("resampling adjudication requires exactly 24 anchors")
    ordered_anchors = tuple(sorted(anchors.anchors, key=lambda item: item.anchor_id))
    try:
        _validate_execution_manifest(
            ordered_anchors,
            initial_allocation_manifest,
            primary_inference=True,
        )
        _validate_execution_manifest(
            ordered_anchors,
            stage_two_allocation_manifest,
            primary_inference=True,
        )
        expected_stage_two = build_fixed_stage_two_allocation_manifest(
            anchors,
            initial_manifest=initial_allocation_manifest,
            master_seed=initial_allocation_manifest.master_seed,
        )
    except (TypeError, ValueError) as exc:
        raise ResampleAdjudicationPhaseError(
            "frozen allocation inventory failed validation"
        ) from exc
    if expected_stage_two.as_dict() != stage_two_allocation_manifest.as_dict():
        raise ResampleAdjudicationPhaseError(
            "stage-two allocation is not the unconditional frozen continuation"
        )
    if (
        len(initial_allocation_manifest.allocations) != 480
        or len(stage_two_allocation_manifest.allocations) != 480
    ):
        raise ResampleAdjudicationPhaseError("frozen allocation inventory must contain 960 units")

    base_by_id: dict[str, BaseTrace] = {}
    try:
        for key, value in base_traces.items():
            base = _coerce_base_trace(value)
            if key != base.base_trace_id or key in base_by_id:
                raise ValueError("base trace key/identity mismatch")
            base_by_id[key] = base
    except (TypeError, ValueError) as exc:
        raise ResampleAdjudicationPhaseError("base trace inventory is malformed") from exc
    expected_trace_ids = {anchor.trace_id for anchor in ordered_anchors}
    if set(base_by_id) != expected_trace_ids:
        raise ResampleAdjudicationPhaseError("base trace inventory does not match frozen anchors")
    for anchor in ordered_anchors:
        base = base_by_id[anchor.trace_id]
        if (
            anchor.char_end > len(base.trace)
            or base.trace[anchor.char_start : anchor.char_end] != anchor.sentence_text
            or base.condition != anchor.direction
        ):
            raise ResampleAdjudicationPhaseError(
                f"base trace disagrees with frozen anchor: {anchor.anchor_id}"
            )

    directory = Path(generation_checkpoint_dir)
    manifest = _load_generation_manifest(directory)
    expected_allocation_hashes = {
        "initial": initial_allocation_manifest.manifest_hash,
        "stage_two": stage_two_allocation_manifest.manifest_hash,
    }
    if manifest.get("allocation_manifest_hashes") != expected_allocation_hashes:
        raise ResampleAdjudicationPhaseError(
            "GPU generation manifest disagrees with frozen allocations"
        )
    plan_hashes = manifest.get("plan_hashes")
    if not isinstance(plan_hashes, Mapping) or set(plan_hashes) != {"initial", "stage_two"}:
        raise ResampleAdjudicationPhaseError("GPU generation plan inventory is malformed")
    stage_hashes = manifest.get("stage_checkpoint_manifest_hashes", {})
    if not isinstance(stage_hashes, Mapping) or set(stage_hashes) not in (
        set(),
        {"initial", "stage_two"},
    ):
        raise ResampleAdjudicationPhaseError("GPU stage checkpoint hash inventory is malformed")

    raw_initial, initial_final_manifest = _load_stage_final(
        directory,
        stage="initial",
        allocation=initial_allocation_manifest,
        expected_plan_hash=str(plan_hashes["initial"]),
        expected_checkpoint_manifest_hash=(
            str(stage_hashes["initial"]) if "initial" in stage_hashes else None
        ),
    )
    raw_stage_two, stage_two_final_manifest = _load_stage_final(
        directory,
        stage="stage_two",
        allocation=stage_two_allocation_manifest,
        expected_plan_hash=str(plan_hashes["stage_two"]),
        expected_checkpoint_manifest_hash=(
            str(stage_hashes["stage_two"]) if "stage_two" in stage_hashes else None
        ),
    )
    combined_path = _safe_local_artifact(
        directory,
        manifest.get("intermediates_path"),
        "gpu_intermediates.jsonl",
    )
    if sha256_file(combined_path) != manifest.get("intermediates_sha256"):
        raise ResampleAdjudicationPhaseError("GPU intermediate artifact hash mismatch")
    combined = read_jsonl(combined_path)
    expected_combined = [*raw_initial, *raw_stage_two]
    if combined != expected_combined:
        raise ResampleAdjudicationPhaseError(
            "GPU combined intermediates disagree with authenticated stage finals"
        )
    if manifest.get("row_count") != EXPECTED_RESAMPLE_COUNT or len(combined) != (
        EXPECTED_RESAMPLE_COUNT
    ):
        raise ResampleAdjudicationPhaseError("GPU generation must contain exactly 960 rows")

    if manifest.get("prefix_registrations_sha256") is not None:
        prefix_path = _safe_local_artifact(
            directory,
            manifest.get("prefix_registrations_path"),
            "prefix_registrations.jsonl",
        )
        if sha256_file(prefix_path) != manifest.get("prefix_registrations_sha256"):
            raise ResampleAdjudicationPhaseError("GPU prefix-registration artifact hash mismatch")

    allocation_by_id = {
        item.request_id: (item, initial_allocation_manifest)
        for item in initial_allocation_manifest.allocations
    }
    allocation_by_id.update(
        {
            item.request_id: (item, stage_two_allocation_manifest)
            for item in stage_two_allocation_manifest.allocations
        }
    )
    if len(allocation_by_id) != EXPECTED_RESAMPLE_COUNT:
        raise ResampleAdjudicationPhaseError("frozen allocation IDs are not globally unique")
    anchor_by_id = {anchor.anchor_id: anchor for anchor in ordered_anchors}
    records: list[ResamplingGenerationRecord] = []
    for raw in combined:
        try:
            record = ResamplingGenerationRecord.from_dict(raw)
            allocation, allocation_manifest = allocation_by_id[record.resample_id]
            anchor = anchor_by_id[allocation.anchor_id]
            base = base_by_id[allocation.base_trace_id]
            _authenticate_intermediate_for_cpu(
                record,
                allocation=allocation,
                anchor=anchor,
                base=base,
                manifest=allocation_manifest,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResampleAdjudicationPhaseError(
                "GPU intermediate row failed frozen source authentication"
            ) from exc
        records.append(record)
    if [record.resample_id for record in records] != list(allocation_by_id):
        raise ResampleAdjudicationPhaseError("GPU intermediate order/inventory mismatch")
    valid_count = sum(record.generation_status == GENERATION_STATUS_VALID for record in records)
    invalid_count = sum(
        record.generation_status == GENERATION_STATUS_TERMINAL_INVALID for record in records
    )
    if (
        manifest.get("valid_generation_count") != valid_count
        or manifest.get("terminal_invalid_count") != invalid_count
        or valid_count + invalid_count != EXPECTED_RESAMPLE_COUNT
    ):
        raise ResampleAdjudicationPhaseError("GPU generation status inventory mismatch")
    source_payload = {
        "generation_manifest_hash": manifest["manifest_hash"],
        "stage_checkpoint_manifest_hashes": {
            "initial": initial_final_manifest["manifest_hash"],
            "stage_two": stage_two_final_manifest["manifest_hash"],
        },
        "anchor_manifest_hash": stable_hash(anchors.as_dict()),
        "anchor_selection_hash": anchors.selection_hash,
        "base_traces_hash": stable_hash(
            [_base_trace_payload(base_by_id[key]) for key in sorted(base_by_id)]
        ),
        "allocation_manifest_hashes": expected_allocation_hashes,
        "source_record_hashes_hash": stable_hash([record.record_hash for record in records]),
    }
    return AuthenticatedResampleGeneration(
        rows=tuple(records),
        initial_rows=tuple(records[:480]),
        stage_two_rows=tuple(records[480:]),
        manifest=manifest,
        stage_manifests={
            "initial": initial_final_manifest,
            "stage_two": stage_two_final_manifest,
        },
        source_hash=stable_hash(source_payload),
        valid_generation_count=valid_count,
        terminal_invalid_count=invalid_count,
    )


def _judge_route_identity(caller: AdjudicationCaller) -> dict[str, Any]:
    provenance = caller.provenance
    source = provenance.to_dict() if isinstance(provenance, JudgeProvenance) else dict(provenance)
    identity = {
        "provider": source.get("provider"),
        "model_id": source.get("model_id"),
        "model_revision": source.get("model_revision"),
        "caller_version": source.get("caller_version"),
        "decoding": dict(source.get("decoding", {})),
    }
    if not isinstance(identity["provider"], str) or not identity["provider"]:
        raise ResampleAdjudicationPhaseError("judge route has no provider identity")
    if not isinstance(identity["model_id"], str) or not identity["model_id"]:
        raise ResampleAdjudicationPhaseError("judge route has no model identity")
    stable_hash(identity)
    return identity


def _component_identity(component: Any, *, label: str) -> dict[str, Any]:
    source = getattr(component, "provenance", None)
    if not isinstance(source, Mapping) or not source:
        raise ResampleAdjudicationPhaseError(f"{label} must expose frozen provenance")
    identity = dict(source)
    claimed = identity.pop("provenance_hash", None)
    if claimed is not None and claimed != stable_hash(identity):
        raise ResampleAdjudicationPhaseError(f"{label} provenance hash mismatch")
    stable_hash(identity)
    return identity


def _validate_checkpoint_rows_against_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    generation: AuthenticatedResampleGeneration,
    plan_hash: str,
    require_complete: bool,
) -> None:
    source_by_id = {record.resample_id: record for record in generation.rows}
    seen: set[str] = set()
    observed_ids: list[str] = []
    for row in rows:
        resample_id = row.get("resample_id")
        if not isinstance(resample_id, str) or resample_id in seen:
            raise ResampleAdjudicationPhaseError(
                "resample adjudication checkpoint has invalid or duplicate IDs"
            )
        seen.add(resample_id)
        observed_ids.append(resample_id)
        source = source_by_id.get(resample_id)
        if (
            source is None
            or row.get("source_generation_record_hash") != source.record_hash
            or row.get("adjudication_plan_hash") != plan_hash
            or row.get("generation_status") != source.generation_status
            or row.get("generation_invalid_reason") != source.invalid_reason
        ):
            raise ResampleAdjudicationPhaseError(
                "resample adjudication checkpoint disagrees with its frozen source"
            )
        valid = source.generation_status == GENERATION_STATUS_VALID
        if row.get("final_quality_denominator_eligible") is not valid:
            raise ResampleAdjudicationPhaseError(
                "resample quality denominator disagrees with the frozen source"
            )
        dual_audit = row.get("dual_final_consensus")
        outcome = row.get("outcome_adjudication")
        if valid:
            if (
                not isinstance(dual_audit, Mapping)
                or not isinstance(outcome, Mapping)
                or dual_audit.get("request_id") != outcome.get("request_id")
                or row.get("scientific_missing_reason") != dual_audit.get("missing_reason")
                or row.get("final_measurement_valid")
                is not (dual_audit.get("known_consensus") is True)
            ):
                raise ResampleAdjudicationPhaseError(
                    "valid resample checkpoint lacks a consistent dual-final outcome"
                )
        elif (
            dual_audit is not None
            or row.get("scientific_missing_reason") != "terminal_invalid_generation"
            or row.get("final_measurement_valid") is not False
            or row.get("final_estimate") is not None
            or row.get("intervention_eligible") is not False
            or row.get("confirmatory_eligible") is not False
        ):
            raise ResampleAdjudicationPhaseError(
                "terminal-invalid resample checkpoint is not explicit missing data"
            )
    if require_complete and observed_ids != [record.resample_id for record in generation.rows]:
        raise ResampleAdjudicationPhaseError(
            "completed resample checkpoint inventory disagrees with the frozen source"
        )


def _parse_final_safely(raw: str) -> tuple[str, int | None, str | None]:
    try:
        parsed = parse_final_adjudication(raw)
    except AdjudicationValidationError:
        return "MALFORMED", None, "malformed_instrument_json"
    return parsed.status.value, parsed.value, None


_MALFORMED_REPLACEMENT_INSTRUMENT_PREFIXES = (
    "duplicate JSON key:",
    "replacement judgment is not strict JSON",
    "replacement judgment has missing or extra keys",
    "replacement judgments must be JSON booleans",
    "replacement confidence must be in [0, 1]",
    "replacement rationale must be nonempty",
)


def _is_malformed_replacement_instrument(
    error: ReplacementClassificationError,
) -> bool:
    message = str(error)
    return any(message.startswith(prefix) for prefix in _MALFORMED_REPLACEMENT_INSTRUMENT_PREFIXES)


class _DualFinalIsolatingCaller:
    not_for_primary_inference = False

    def __init__(
        self,
        primary: AdjudicationCaller,
        independent: AdjudicationCaller,
        *,
        completed_rows: Mapping[str, Mapping[str, Any]],
        primary_identity: Mapping[str, Any],
        independent_identity: Mapping[str, Any],
    ) -> None:
        self._primary = primary
        self._independent = independent
        self._primary_identity = dict(primary_identity)
        self._independent_identity = dict(independent_identity)
        self._completed_by_request: dict[str, dict[str, Any]] = {}
        for row in completed_rows.values():
            audit = row.get("dual_final_consensus")
            if not isinstance(audit, Mapping):
                continue
            request_id = audit.get("request_id")
            if not isinstance(request_id, str):
                raise ResampleAdjudicationPhaseError(
                    "completed dual-final audit request inventory is invalid"
                )
            self._validate_audit(dict(audit))
            previous = self._completed_by_request.get(request_id)
            if previous is not None and previous != dict(audit):
                raise ResampleAdjudicationPhaseError(
                    "completed duplicate dual-final requests disagree"
                )
            self._completed_by_request[request_id] = dict(audit)
        self.audits_by_request: dict[str, dict[str, Any]] = {}

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="dual_route_exact_consensus",
            model_id=(
                f"{self._primary_identity['model_id']}||{self._independent_identity['model_id']}"
            ),
            model_revision=None,
            caller_version=RESAMPLE_ADJUDICATION_PROTOCOL,
            decoding={"temperature": 0, "response_format": "json_object"},
            metadata={
                "primary_route_hash": stable_hash(self._primary_identity),
                "independent_route_hash": stable_hash(self._independent_identity),
            },
        )

    def _validate_audit(self, audit: dict[str, Any]) -> None:
        if (
            audit.get("protocol_version") != RESAMPLE_ADJUDICATION_PROTOCOL
            or audit.get("record_hash") != stable_hash(_without_hash(audit, "record_hash"))
            or audit.get("instrument_hash") != FINAL_ANSWER_INSTRUMENT.instrument_hash
            or audit.get("primary_route") != self._primary_identity
            or audit.get("independent_route") != self._independent_identity
            or not _valid_content_hash(audit.get("request_id"))
            or not _valid_content_hash(audit.get("case_hash"))
            or not isinstance(audit.get("returned_response"), str)
        ):
            raise ResampleAdjudicationPhaseError("completed dual-final audit failed authentication")

        judgments: dict[str, tuple[str, int | None, str | None]] = {}
        for label in ("primary", "independent"):
            judgment = audit.get(label)
            if not isinstance(judgment, Mapping):
                raise ResampleAdjudicationPhaseError(
                    "completed dual-final audit failed authentication"
                )
            raw = judgment.get("raw_response")
            if not isinstance(raw, str) or judgment.get("response_hash") != stable_hash(
                {"raw_response": raw}
            ):
                raise ResampleAdjudicationPhaseError(
                    "completed dual-final audit failed authentication"
                )
            parsed = _parse_final_safely(raw)
            if (
                judgment.get("status") != parsed[0]
                or judgment.get("value") != parsed[1]
                or judgment.get("terminal_contract_failure") != parsed[2]
            ):
                raise ResampleAdjudicationPhaseError(
                    "completed dual-final audit failed authentication"
                )
            judgments[label] = parsed

        primary_status, primary_value, primary_failure = judgments["primary"]
        independent_status, independent_value, independent_failure = judgments["independent"]
        exact = bool(
            primary_failure is None
            and independent_failure is None
            and primary_status == independent_status
            and primary_value == independent_value
        )
        known = bool(exact and primary_status == KnowledgeStatus.KNOWN.value)
        if primary_failure is not None:
            expected_missing = "malformed_primary_final"
        elif independent_failure is not None:
            expected_missing = "malformed_independent_final"
        elif not exact:
            expected_missing = "final_judge_disagreement"
        elif not known:
            expected_missing = "final_consensus_unknown"
        else:
            expected_missing = None
        try:
            returned = parse_final_adjudication(str(audit["returned_response"]))
        except AdjudicationValidationError as exc:
            raise ResampleAdjudicationPhaseError(
                "completed dual-final consensus response is malformed"
            ) from exc
        if (
            audit.get("exact_status_value_agreement") is not exact
            or audit.get("known_consensus") is not known
            or audit.get("missing_reason") != expected_missing
            or known != (returned.status is KnowledgeStatus.KNOWN)
            or (known and audit["returned_response"] != audit["primary"]["raw_response"])
            or (known and returned.value != primary_value)
            or (not known and returned.value is not None)
        ):
            raise ResampleAdjudicationPhaseError(
                "completed dual-final audit disagrees with paid raw consensus"
            )

    def complete(self, request: AdjudicationRequest) -> str:
        if request.instrument_id != FINAL_ANSWER_INSTRUMENT.instrument_id:
            raise ResampleAdjudicationPhaseError(
                "dual final caller received a non-final instrument"
            )
        completed = self._completed_by_request.get(request.request_id)
        if completed is not None:
            if completed.get("case_hash") != stable_hash(dict(request.user_payload)):
                raise ResampleAdjudicationPhaseError(
                    "completed dual-final audit belongs to a different blinded case"
                )
            self.audits_by_request[request.request_id] = completed
            return str(completed["returned_response"])

        primary_raw = self._primary.complete(request)
        primary_status, primary_value, primary_failure = _parse_final_safely(primary_raw)
        if _judge_route_identity(self._primary) != self._primary_identity:
            raise ResampleAdjudicationPhaseError("primary final route drifted during execution")
        independent_raw = self._independent.complete(request)
        independent_status, independent_value, independent_failure = _parse_final_safely(
            independent_raw
        )
        if _judge_route_identity(self._independent) != self._independent_identity:
            raise ResampleAdjudicationPhaseError("independent final route drifted during execution")
        exact = bool(
            primary_failure is None
            and independent_failure is None
            and primary_status == independent_status
            and primary_value == independent_value
        )
        known = bool(exact and primary_status == KnowledgeStatus.KNOWN.value)
        if known:
            returned_response = primary_raw
            missing_reason = None
        else:
            returned_response = json.dumps(
                {"status": "UNKNOWN", "value": None},
                separators=(",", ":"),
                sort_keys=True,
            )
            if primary_failure is not None:
                missing_reason = "malformed_primary_final"
            elif independent_failure is not None:
                missing_reason = "malformed_independent_final"
            elif not exact:
                missing_reason = "final_judge_disagreement"
            else:
                missing_reason = "final_consensus_unknown"
        audit: dict[str, Any] = {
            "protocol_version": RESAMPLE_ADJUDICATION_PROTOCOL,
            "request_id": request.request_id,
            "case_hash": stable_hash(dict(request.user_payload)),
            "instrument_hash": request.instrument_hash,
            "primary_route": self._primary_identity,
            "independent_route": self._independent_identity,
            "primary": {
                "raw_response": primary_raw,
                "response_hash": stable_hash({"raw_response": primary_raw}),
                "status": primary_status,
                "value": primary_value,
                "terminal_contract_failure": primary_failure,
            },
            "independent": {
                "raw_response": independent_raw,
                "response_hash": stable_hash({"raw_response": independent_raw}),
                "status": independent_status,
                "value": independent_value,
                "terminal_contract_failure": independent_failure,
            },
            "exact_status_value_agreement": exact,
            "known_consensus": known,
            "missing_reason": missing_reason,
            "returned_response": returned_response,
        }
        audit["record_hash"] = stable_hash(audit)
        self.audits_by_request[request.request_id] = audit
        return returned_response


class _ReplayableMalformedIsolatingClassifier:
    def __init__(
        self,
        delegate: ReplacementClassifier,
        *,
        completed_rows: Mapping[str, Mapping[str, Any]],
        identity: Mapping[str, Any],
    ) -> None:
        self._delegate = delegate
        self._identity = dict(identity)
        self._completed_by_request_hash: dict[str, Mapping[str, Any]] = {}
        for row in completed_rows.values():
            request_hash = row.get("classifier_request_hash")
            if isinstance(request_hash, str):
                previous = self._completed_by_request_hash.get(request_hash)
                if previous is not None and any(
                    previous.get(field) != row.get(field)
                    for field in (
                        "replacement_classification_status",
                        "target_feature_absent_or_changed",
                        "neutral_control_function_matched",
                        "classifier_judgment_hashes",
                        "classifier_provenance_hash",
                        "classification_rationale",
                    )
                ):
                    raise ResampleAdjudicationPhaseError(
                        "completed duplicate classifier requests disagree"
                    )
                self._completed_by_request_hash[request_hash] = row
        self.failures: dict[str, str] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._identity)

    def classify(
        self, request: ReplacementClassificationRequest
    ) -> ReplacementClassificationResult | None:
        completed = self._completed_by_request_hash.get(request.request_hash)
        if completed is not None:
            status = completed.get("replacement_classification_status")
            if status == "malformed_instrument_json":
                self.failures[request.request_hash] = "malformed_replacement_classification"
                return None
            if status not in {"valid", "invalid_adjudication"}:
                raise ResampleAdjudicationPhaseError(
                    "completed row has an impossible paid-classifier status"
                )
            try:
                return ReplacementClassificationResult(
                    request_hash=request.request_hash,
                    adjudication_valid=status == "valid",
                    target_feature_absent_or_changed=completed.get(
                        "target_feature_absent_or_changed"
                    ),
                    neutral_control_function_matched=completed.get(
                        "neutral_control_function_matched"
                    ),
                    raw_judgment_hashes=tuple(completed["classifier_judgment_hashes"]),
                    classifier_provenance_hash=str(completed["classifier_provenance_hash"]),
                    rationale=str(completed["classification_rationale"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ResampleAdjudicationPhaseError(
                    "completed classifier result failed reconstruction"
                ) from exc
        try:
            result = self._delegate.classify(request)
        except ReplacementClassificationError as exc:
            if not _is_malformed_replacement_instrument(exc):
                raise
            self.failures[request.request_hash] = "malformed_replacement_classification"
            return None
        if _component_identity(self._delegate, label="replacement classifier") != self._identity:
            raise ResampleAdjudicationPhaseError("replacement classifier route drifted")
        return result


def _validate_gate(name: str, value: float, floor: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not floor <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [{floor}, 1]")
    return float(value)


def _quality_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_exact_agreement: float,
    minimum_final_known_rate: float,
    minimum_overall_generation_valid_rate: float,
    minimum_anchor_arm_valid_count: int,
    minimum_anchor_pair_complete_count: int,
    maximum_anchor_arm_valid_rate_gap: float,
) -> dict[str, Any]:
    expected_per_arm = 20
    by_anchor_arm: dict[tuple[str, str], dict[int, bool]] = {}
    for index, row in enumerate(rows, start=1):
        anchor_id = str(row.get("anchor_id", ""))
        arm = str(row.get("arm", ""))
        sample_index = row.get("sample_index")
        if (
            not anchor_id
            or arm not in {"retain", "resample"}
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < expected_per_arm
        ):
            raise ResampleAdjudicationPhaseError(
                f"resampling quality row {index} has invalid anchor/arm/sample identity"
            )
        cell = by_anchor_arm.setdefault((anchor_id, arm), {})
        if sample_index in cell:
            raise ResampleAdjudicationPhaseError(
                f"duplicate resampling quality identity {(anchor_id, arm, sample_index)!r}"
            )
        cell[sample_index] = row.get("generation_status") == GENERATION_STATUS_VALID

    anchor_ids = sorted({anchor_id for anchor_id, _arm in by_anchor_arm})
    expected_cells = {(anchor_id, arm) for anchor_id in anchor_ids for arm in ("retain", "resample")}
    if len(anchor_ids) != 24 or set(by_anchor_arm) != expected_cells:
        raise ResampleAdjudicationPhaseError(
            "resampling generation-attrition gate requires exactly 24 anchors and both arms"
        )

    cell_reports: list[dict[str, Any]] = []
    anchor_reports: list[dict[str, Any]] = []
    cell_gate_passed = True
    pair_gate_passed = True
    gap_gate_passed = True
    for anchor_id in anchor_ids:
        counts: dict[str, int] = {}
        rates: dict[str, float] = {}
        for arm in ("retain", "resample"):
            cell = by_anchor_arm[(anchor_id, arm)]
            complete_inventory = set(cell) == set(range(expected_per_arm))
            valid_count = sum(cell.values()) if complete_inventory else 0
            valid_rate = valid_count / expected_per_arm
            passed = complete_inventory and valid_count >= minimum_anchor_arm_valid_count
            cell_gate_passed = cell_gate_passed and passed
            counts[arm] = valid_count
            rates[arm] = valid_rate
            cell_reports.append(
                {
                    "anchor_id": anchor_id,
                    "arm": arm,
                    "expected_count": expected_per_arm,
                    "observed_count": len(cell),
                    "valid_generation_count": valid_count,
                    "valid_generation_rate": valid_rate,
                    "minimum_valid_generation_count": minimum_anchor_arm_valid_count,
                    "passed": passed,
                }
            )
        pair_complete = sum(
            by_anchor_arm[(anchor_id, "retain")].get(sample_index) is True
            and by_anchor_arm[(anchor_id, "resample")].get(sample_index) is True
            for sample_index in range(expected_per_arm)
        )
        pair_passed = pair_complete >= minimum_anchor_pair_complete_count
        rate_gap = abs(rates["retain"] - rates["resample"])
        gap_passed = rate_gap <= maximum_anchor_arm_valid_rate_gap + 1e-12
        pair_gate_passed = pair_gate_passed and pair_passed
        gap_gate_passed = gap_gate_passed and gap_passed
        anchor_reports.append(
            {
                "anchor_id": anchor_id,
                "retain_valid_count": counts["retain"],
                "resample_valid_count": counts["resample"],
                "pair_complete_generation_count": pair_complete,
                "minimum_pair_complete_generation_count": (
                    minimum_anchor_pair_complete_count
                ),
                "absolute_arm_valid_rate_gap": rate_gap,
                "maximum_arm_valid_rate_gap": maximum_anchor_arm_valid_rate_gap,
                "pair_complete_gate_passed": pair_passed,
                "arm_gap_gate_passed": gap_passed,
            }
        )

    eligible = [row for row in rows if row.get("final_quality_denominator_eligible") is True]
    denominator = len(eligible)
    exact = sum(
        isinstance(row.get("dual_final_consensus"), Mapping)
        and row["dual_final_consensus"].get("exact_status_value_agreement") is True
        for row in eligible
    )
    known = sum(
        isinstance(row.get("dual_final_consensus"), Mapping)
        and row["dual_final_consensus"].get("known_consensus") is True
        for row in eligible
    )
    exact_rate = exact / denominator if denominator else 0.0
    known_rate = known / denominator if denominator else 0.0
    overall_generation_valid_rate = denominator / len(rows) if rows else 0.0
    overall_generation_gate_passed = bool(
        rows and overall_generation_valid_rate >= minimum_overall_generation_valid_rate
    )
    generation_attrition_gate_passed = bool(
        overall_generation_gate_passed
        and cell_gate_passed
        and pair_gate_passed
        and gap_gate_passed
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol_version": RESAMPLE_ADJUDICATION_PROTOCOL,
        "scope": "all_frozen_resampling_generations_and_valid-generation_finals",
        "denominator_valid_generation_count": denominator,
        "terminal_invalid_excluded_count": len(rows) - denominator,
        "overall_generation_valid_rate": overall_generation_valid_rate,
        "minimum_overall_generation_valid_rate": minimum_overall_generation_valid_rate,
        "overall_generation_valid_gate_passed": overall_generation_gate_passed,
        "anchor_arm_generation_valid_gate_passed": cell_gate_passed,
        "anchor_pair_complete_generation_gate_passed": pair_gate_passed,
        "anchor_arm_attrition_gap_gate_passed": gap_gate_passed,
        "generation_attrition_gate_passed": generation_attrition_gate_passed,
        "anchor_arm_reports": cell_reports,
        "anchor_reports": anchor_reports,
        "exact_status_value_agreements": exact,
        "exact_status_value_agreement_rate": exact_rate,
        "minimum_exact_status_value_agreement": minimum_exact_agreement,
        "known_consensus_count": known,
        "known_consensus_rate": known_rate,
        "minimum_final_known_rate": minimum_final_known_rate,
        "agreement_gate_passed": bool(denominator and exact_rate >= minimum_exact_agreement),
        "known_gate_passed": bool(denominator and known_rate >= minimum_final_known_rate),
        "rows_hash": stable_hash(rows),
    }
    payload["gate_passed"] = bool(
        generation_attrition_gate_passed
        and payload["agreement_gate_passed"]
        and payload["known_gate_passed"]
    )
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def evaluate_generation_attrition(
    rows: Sequence[ResamplingGenerationRecord],
    *,
    minimum_overall_generation_valid_rate: float,
    minimum_anchor_arm_valid_count: int,
    minimum_anchor_pair_complete_count: int,
    maximum_anchor_arm_valid_rate_gap: float,
) -> dict[str, Any]:
    """Evaluate every generation-only gate before any paid API preflight."""

    quality_rows = [
        {
            "anchor_id": row.anchor_id,
            "arm": row.arm,
            "sample_index": row.sample_index,
            "generation_status": row.generation_status,
            "final_quality_denominator_eligible": (
                row.generation_status == GENERATION_STATUS_VALID
            ),
            "dual_final_consensus": (
                {
                    "exact_status_value_agreement": True,
                    "known_consensus": True,
                }
                if row.generation_status == GENERATION_STATUS_VALID
                else None
            ),
        }
        for row in rows
    ]
    return _quality_gate(
        quality_rows,
        minimum_exact_agreement=0.0,
        minimum_final_known_rate=0.0,
        minimum_overall_generation_valid_rate=minimum_overall_generation_valid_rate,
        minimum_anchor_arm_valid_count=minimum_anchor_arm_valid_count,
        minimum_anchor_pair_complete_count=minimum_anchor_pair_complete_count,
        maximum_anchor_arm_valid_rate_gap=maximum_anchor_arm_valid_rate_gap,
    )


def _freeze_or_verify(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    if path.exists():
        observed = read_json(path)
        if observed != dict(payload):
            raise ResampleAdjudicationPhaseError(f"existing {label} artifact drifted")
        return
    write_json(path, payload)


def _complete_phase(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    checkpoint_manifest: Mapping[str, Any],
    plan_hash: str,
    generation: AuthenticatedResampleGeneration,
    execution_id: str,
    primary_identity: Mapping[str, Any],
    independent_identity: Mapping[str, Any],
    classifier_identity: Mapping[str, Any],
    embedder_identity: Mapping[str, Any],
    minimum_exact_agreement: float,
    minimum_final_known_rate: float,
    minimum_overall_generation_valid_rate: float,
    minimum_anchor_arm_valid_count: int,
    minimum_anchor_pair_complete_count: int,
    maximum_anchor_arm_valid_rate_gap: float,
) -> ResampleAdjudicationPhase:
    quality = _quality_gate(
        rows,
        minimum_exact_agreement=minimum_exact_agreement,
        minimum_final_known_rate=minimum_final_known_rate,
        minimum_overall_generation_valid_rate=minimum_overall_generation_valid_rate,
        minimum_anchor_arm_valid_count=minimum_anchor_arm_valid_count,
        minimum_anchor_pair_complete_count=minimum_anchor_pair_complete_count,
        maximum_anchor_arm_valid_rate_gap=maximum_anchor_arm_valid_rate_gap,
    )
    quality_path = directory / "quality_gate.json"
    _freeze_or_verify(quality_path, quality, label="resample quality gate")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": RESAMPLE_ADJUDICATION_PROTOCOL,
        "complete": True,
        "execution_id": execution_id,
        "adjudication_plan_hash": plan_hash,
        "source_generation_hash": generation.source_hash,
        "source_generation_manifest_hash": generation.manifest["manifest_hash"],
        "row_count": len(rows),
        "valid_generation_count": generation.valid_generation_count,
        "terminal_invalid_count": generation.terminal_invalid_count,
        "row_ids_hash": stable_hash([row["resample_id"] for row in rows]),
        "row_record_hashes_hash": stable_hash([row["record_hash"] for row in rows]),
        "record_checkpoint_manifest_hash": checkpoint_manifest["manifest_hash"],
        "primary_final_route": dict(primary_identity),
        "independent_final_route": dict(independent_identity),
        "replacement_classifier": dict(classifier_identity),
        "embedder": dict(embedder_identity),
        "quality_gate": quality,
        "artifacts": {
            "rows": {
                "path": "units/checkpoint_rows.jsonl",
                "sha256": checkpoint_manifest["rows_sha256"],
            },
            "quality_gate": {
                "path": quality_path.name,
                "sha256": sha256_file(quality_path),
            },
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    _freeze_or_verify(
        directory / "adjudication_manifest.json",
        manifest,
        label="resample adjudication manifest",
    )
    result = ResampleAdjudicationPhase(
        rows=tuple(dict(row) for row in rows),
        quality_gate=quality,
        manifest=manifest,
        complete=True,
        gate_passed=bool(quality["gate_passed"]),
    )
    if not result.gate_passed:
        raise ResampleAdjudicationGateError(
            "resampling generation-attrition, final exact-agreement, or known-rate gate failed closed"
        )
    return result


def run_resample_adjudication_phase(
    *,
    generation_checkpoint_dir: str | Path,
    checkpoint_dir: str | Path,
    anchors: AnchorManifest,
    base_traces: Mapping[str, BaseTrace | Mapping[str, Any]],
    initial_allocation_manifest: ResampleAllocationManifest,
    stage_two_allocation_manifest: ResampleAllocationManifest,
    embedder: TextEmbedder,
    primary_final_caller: AdjudicationCaller,
    independent_final_caller: AdjudicationCaller,
    replacement_classifier: ReplacementClassifier,
    neutral_control: NeutralControlSpec,
    token_tolerance: ReplacementTokenTolerance,
    execution_id: str,
    minimum_exact_agreement: float = 0.90,
    minimum_final_known_rate: float = 0.95,
    minimum_overall_generation_valid_rate: float = (
        DEFAULT_MINIMUM_OVERALL_GENERATION_VALID_RATE
    ),
    minimum_anchor_arm_valid_count: int = DEFAULT_MINIMUM_ANCHOR_ARM_VALID_COUNT,
    minimum_anchor_pair_complete_count: int = DEFAULT_MINIMUM_ANCHOR_PAIR_COMPLETE_COUNT,
    maximum_anchor_arm_valid_rate_gap: float = DEFAULT_MAXIMUM_ANCHOR_ARM_VALID_RATE_GAP,
    on_record_committed: Callable[[Mapping[str, Any]], None] | None = None,
) -> ResampleAdjudicationPhase:
    """Run or resume the exact 960-unit CPU/API resampling phase."""

    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ValueError("execution_id must be nonempty")
    minimum_exact = _validate_gate(
        "minimum_exact_agreement",
        minimum_exact_agreement,
        MINIMUM_EXACT_AGREEMENT_FLOOR,
    )
    minimum_known = _validate_gate(
        "minimum_final_known_rate",
        minimum_final_known_rate,
        MINIMUM_FINAL_KNOWN_FLOOR,
    )
    minimum_overall_valid = _validate_gate(
        "minimum_overall_generation_valid_rate",
        minimum_overall_generation_valid_rate,
        0.0,
    )
    if (
        isinstance(minimum_anchor_arm_valid_count, bool)
        or not isinstance(minimum_anchor_arm_valid_count, int)
        or not 0 <= minimum_anchor_arm_valid_count <= 20
        or isinstance(minimum_anchor_pair_complete_count, bool)
        or not isinstance(minimum_anchor_pair_complete_count, int)
        or not 0 <= minimum_anchor_pair_complete_count <= 20
    ):
        raise ValueError("resampling anchor attrition counts must be integers in [0, 20]")
    maximum_arm_gap = _validate_gate(
        "maximum_anchor_arm_valid_rate_gap",
        maximum_anchor_arm_valid_rate_gap,
        0.0,
    )
    if primary_final_caller.not_for_primary_inference or (
        independent_final_caller.not_for_primary_inference
    ):
        raise ResampleAdjudicationPhaseError(
            "primary resampling refuses non-primary final judge routes"
        )
    primary_identity = _judge_route_identity(primary_final_caller)
    independent_identity = _judge_route_identity(independent_final_caller)
    if (
        primary_identity["provider"],
        primary_identity["model_id"],
        primary_identity["model_revision"],
    ) == (
        independent_identity["provider"],
        independent_identity["model_id"],
        independent_identity["model_revision"],
    ):
        raise ResampleAdjudicationPhaseError(
            "primary and independent final judges must be distinct routes"
        )
    classifier_identity = _component_identity(
        replacement_classifier,
        label="replacement classifier",
    )
    if bool(classifier_identity.get("synthetic_smoke", False)):
        raise ResampleAdjudicationPhaseError(
            "primary resampling refuses a synthetic replacement classifier"
        )
    embedder_identity = _component_identity(embedder, label="embedder")
    generation = load_authenticated_resample_generation(
        generation_checkpoint_dir=generation_checkpoint_dir,
        anchors=anchors,
        base_traces=base_traces,
        initial_allocation_manifest=initial_allocation_manifest,
        stage_two_allocation_manifest=stage_two_allocation_manifest,
    )
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": 1,
        "protocol_version": RESAMPLE_ADJUDICATION_PROTOCOL,
        "execution_id": execution_id,
        "source_generation_hash": generation.source_hash,
        "source_generation_manifest_hash": generation.manifest["manifest_hash"],
        "source_record_hashes_hash": stable_hash(
            [record.record_hash for record in generation.rows]
        ),
        "anchor_selection_hash": anchors.selection_hash,
        "initial_allocation_manifest_hash": initial_allocation_manifest.manifest_hash,
        "stage_two_allocation_manifest_hash": stage_two_allocation_manifest.manifest_hash,
        "primary_final_route": primary_identity,
        "independent_final_route": independent_identity,
        "replacement_classifier": classifier_identity,
        "embedder": embedder_identity,
        "neutral_control_hash": neutral_control.control_hash,
        "token_tolerance": {
            "maximum_absolute_difference": token_tolerance.max_absolute_difference,
            "maximum_relative_difference": token_tolerance.max_relative_difference,
        },
        "minimum_exact_agreement": minimum_exact,
        "minimum_final_known_rate": minimum_known,
        "minimum_overall_generation_valid_rate": minimum_overall_valid,
        "minimum_anchor_arm_valid_count": minimum_anchor_arm_valid_count,
        "minimum_anchor_pair_complete_count": minimum_anchor_pair_complete_count,
        "maximum_anchor_arm_valid_rate_gap": maximum_arm_gap,
        "paid_response_replay_protocol": PAID_RESPONSE_STORE_PROTOCOL,
        "expected_count": EXPECTED_RESAMPLE_COUNT,
    }
    try:
        store = RecordCheckpointStore(
            directory / "units",
            id_field="resample_id",
            plan_payload=plan_payload,
        )
    except (RecordCheckpointError, ValueError) as exc:
        raise ResampleAdjudicationPhaseError("resample adjudication plan drifted") from exc
    expected_ids = tuple(record.resample_id for record in generation.rows)
    final_path = store.directory / "checkpoint_manifest.json"
    if final_path.is_file():
        try:
            final = store.load_final(expected_ids=expected_ids)
        except RecordCheckpointError as exc:
            raise ResampleAdjudicationPhaseError(
                "completed resample adjudication checkpoint failed authentication"
            ) from exc
        _validate_checkpoint_rows_against_source(
            final.rows,
            generation=generation,
            plan_hash=str(store.plan["plan_hash"]),
            require_complete=True,
        )
        _DualFinalIsolatingCaller(
            primary_final_caller,
            independent_final_caller,
            completed_rows={str(row["resample_id"]): row for row in final.rows},
            primary_identity=primary_identity,
            independent_identity=independent_identity,
        )
        return _complete_phase(
            directory=directory,
            rows=final.rows,
            checkpoint_manifest=final.manifest,
            plan_hash=str(store.plan["plan_hash"]),
            generation=generation,
            execution_id=execution_id,
            primary_identity=primary_identity,
            independent_identity=independent_identity,
            classifier_identity=classifier_identity,
            embedder_identity=embedder_identity,
            minimum_exact_agreement=minimum_exact,
            minimum_final_known_rate=minimum_known,
            minimum_overall_generation_valid_rate=minimum_overall_valid,
            minimum_anchor_arm_valid_count=minimum_anchor_arm_valid_count,
            minimum_anchor_pair_complete_count=minimum_anchor_pair_complete_count,
            maximum_anchor_arm_valid_rate_gap=maximum_arm_gap,
        )

    try:
        completed = {str(row["resample_id"]): row for row in store.load_records()}
    except RecordCheckpointError as exc:
        raise ResampleAdjudicationPhaseError(
            "resample adjudication record checkpoint failed authentication"
        ) from exc
    _validate_checkpoint_rows_against_source(
        tuple(completed.values()),
        generation=generation,
        plan_hash=str(store.plan["plan_hash"]),
        require_complete=False,
    )
    source_by_id = {record.resample_id: record for record in generation.rows}

    dual_caller = _DualFinalIsolatingCaller(
        primary_final_caller,
        independent_final_caller,
        completed_rows=completed,
        primary_identity=primary_identity,
        independent_identity=independent_identity,
    )
    classifier = _ReplayableMalformedIsolatingClassifier(
        replacement_classifier,
        completed_rows=completed,
        identity=classifier_identity,
    )
    preexisting_ids = set(completed)

    def commit_record(record: ResamplingArtifactRecord) -> None:
        source = source_by_id[record.resample_id]
        payload = record.as_dict(include_hash=False)
        outcome = payload.get("outcome_adjudication")
        request_id = outcome.get("request_id") if isinstance(outcome, Mapping) else None
        if source.generation_status == GENERATION_STATUS_VALID:
            audit = dual_caller.audits_by_request.get(str(request_id))
            if audit is None:
                raise ResampleAdjudicationPhaseError("valid generation lacks a dual-final audit")
            dual_audit: Mapping[str, Any] | None = audit
            scientific_missing_reason = audit.get("missing_reason")
        else:
            dual_audit = None
            scientific_missing_reason = "terminal_invalid_generation"
        classifier_request_hash = payload.get("classifier_request_hash")
        if (
            isinstance(classifier_request_hash, str)
            and classifier_request_hash in classifier.failures
        ):
            payload["replacement_classification_status"] = "malformed_instrument_json"
            payload["classification_rationale"] = "malformed_replacement_classification"
            payload["intervention_eligible"] = False
            payload["primary_eligible"] = False
            payload["confirmatory_eligible"] = False
            if payload.get("analysis_tier") == "confirmatory":
                payload["analysis_tier"] = "exploratory"
            payload["intervention_eligibility_missing_reason"] = (
                "malformed_replacement_classification"
            )
        else:
            payload["intervention_eligibility_missing_reason"] = None
        payload.update(
            {
                "adjudication_protocol": RESAMPLE_ADJUDICATION_PROTOCOL,
                "execution_id": execution_id,
                "adjudication_plan_hash": store.plan["plan_hash"],
                "source_generation_record_hash": source.record_hash,
                "generation_status": source.generation_status,
                "generation_invalid_reason": source.invalid_reason,
                "dual_final_consensus": dual_audit,
                "final_quality_denominator_eligible": (
                    source.generation_status == GENERATION_STATUS_VALID
                ),
                "scientific_missing_reason": scientific_missing_reason,
            }
        )
        payload["record_hash"] = stable_hash(payload)
        try:
            committed = store.commit(payload)
        except RecordCheckpointError as exc:
            raise ResampleAdjudicationPhaseError(
                f"atomic resample checkpoint conflict: {record.resample_id}"
            ) from exc
        if record.resample_id not in preexisting_ids:
            completed[record.resample_id] = committed
            if on_record_committed is not None:
                on_record_committed(committed)

    stage_inputs = (
        (generation.initial_rows, initial_allocation_manifest),
        (generation.stage_two_rows, stage_two_allocation_manifest),
    )
    for intermediates, allocation_manifest in stage_inputs:
        try:
            adjudicate_sentence_resampling_intermediates(
                intermediates,
                anchors=anchors,
                base_traces=base_traces,
                allocation_manifest=allocation_manifest,
                embedder=embedder,
                outcome_caller=dual_caller,
                primary_inference=True,
                replacement_classifier=classifier,
                neutral_control=neutral_control,
                token_tolerance=token_tolerance,
                on_record=commit_record,
            )
        except ResampleAdjudicationPhaseError:
            raise
        except (ReplacementClassificationError, ValueError, TypeError) as exc:
            raise ResampleAdjudicationPhaseError(
                "resample CPU/adjudication integrity contract failed"
            ) from exc
    try:
        final = store.finalize(expected_ids=expected_ids)
    except RecordCheckpointError as exc:
        raise ResampleAdjudicationPhaseError(
            "resample adjudication final inventory failed authentication"
        ) from exc
    return _complete_phase(
        directory=directory,
        rows=final.rows,
        checkpoint_manifest=final.manifest,
        plan_hash=str(store.plan["plan_hash"]),
        generation=generation,
        execution_id=execution_id,
        primary_identity=primary_identity,
        independent_identity=independent_identity,
        classifier_identity=classifier_identity,
        embedder_identity=embedder_identity,
        minimum_exact_agreement=minimum_exact,
        minimum_final_known_rate=minimum_known,
        minimum_overall_generation_valid_rate=minimum_overall_valid,
        minimum_anchor_arm_valid_count=minimum_anchor_arm_valid_count,
        minimum_anchor_pair_complete_count=minimum_anchor_pair_complete_count,
        maximum_anchor_arm_valid_rate_gap=maximum_arm_gap,
    )


__all__ = [
    "EXPECTED_RESAMPLE_COUNT",
    "RESAMPLE_ADJUDICATION_PROTOCOL",
    "AuthenticatedResampleGeneration",
    "ResampleAdjudicationGateError",
    "ResampleAdjudicationPhase",
    "ResampleAdjudicationPhaseError",
    "load_authenticated_resample_generation",
    "run_resample_adjudication_phase",
]
