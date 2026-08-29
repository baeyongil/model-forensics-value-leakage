"""Fail-closed, parser-independent command boundary for primary lens execution.

The behavioral rollout artifact does not currently contain an auditable token
span for the *first* estimate.  This module therefore refuses to infer one by
searching for numbers.  Before the real lens job can run, an upstream blinded
span-adjudication step must provide a frozen position record for each selected
anchor.  :data:`POSITION_RECORD_SCHEMA` documents that required record.

The command validates all 24 source rollouts, their frozen anchors, exact token
streams, and all span-derived named positions before it calls any runtime
factory.  It then runs the ordered 4B smoke and the at-most-two-attempt 122B
gate, writes the compatibility manifest even on gate failure, executes J/R
readouts, and atomically writes canonical rows and an execution manifest.

There is intentionally no argparse dependency here.  A CLI or RunPod entry
point can call :func:`run_frozen_lens_command_from_files` without duplicating
any scientific validation logic.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from model_forensics.anchors import (
    AnchorManifest,
    FrozenAnchor,
    validate_anchor_manifest,
)
from model_forensics.estimate_spans import FIRST_ESTIMATE_SPAN_INSTRUMENT_ID
from model_forensics.io import (
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
)
from model_forensics.lens import EXPECTED_MODEL_ID, FITTED_LAYERS, LoadedLens, ModelRuntime
from model_forensics.lens_positions import (
    POSITION_MANIFEST_SCHEMA_VERSION,
    POSITION_ORDER,
)
from model_forensics.lens_runner import (
    COMPATIBILITY_ATTEMPT_PREFIX_STREAM,
    PRIMARY_MODEL_REVISION,
    CompatibilityGateError,
    LensCompatibilityManifest,
    LensProbeDesign,
    LensTraceInput,
    SameForwardBackend,
    execute_lens_traces,
    run_4b_compatibility_smoke,
    run_122b_preflight,
    run_ordered_compatibility_gate,
    write_compatibility_manifest,
)
from model_forensics.schemas import CanonicalRecord
from model_forensics.token_spans import (
    token_stream_hash,
    validate_token_stream_manifest,
)

# Machine-readable documentation for the upstream span-adjudication handoff.
# Values in angle brackets describe types or required provenance rather than
# literal strings.  ``record_hash`` is stable_hash(record without record_hash).
POSITION_RECORD_SCHEMA: Mapping[str, Any] = {
    "schema_version": POSITION_MANIFEST_SCHEMA_VERSION,
    "trace_id": "<selected rollout run_id>",
    "anchor_id": "<frozen anchor_id>",
    "anchor_manifest_hash": "sha256:<canonical full anchor-manifest hash>",
    "rollout_record_hash": "sha256:<canonical rollout hash>",
    "first_estimate_span_record_hash": "sha256:<blind span record>",
    "first_estimate_span_instrument_id": FIRST_ESTIMATE_SPAN_INSTRUMENT_ID,
    "first_estimate_span_primary_inference": True,
    "prompt_token_ids_hash": "sha256:<exact prompt token stream>",
    "completion_token_ids_hash": "sha256:<exact completion token stream>",
    "combined_token_stream_hash": "sha256:<exact prompt+completion stream hash>",
    "position_order": list(POSITION_ORDER),
    "position_indices": {
        name: "<zero-based combined-stream token index>" for name in POSITION_ORDER
    },
    "position_evidence": {
        "first_estimate": "<OriginalTokenSpan emitted from blind span adjudication>",
        "anchor": "<exact token_span frozen in anchor provenance>",
        "answer_first_token": "<OriginalTokenSpan from the completion-section boundary>",
    },
    "good_side_direction": "<-1 or +1 from frozen incentive direction>",
    "causal_claim": False,
    "record_hash": "sha256:<stable hash of every preceding field>",
}


class LensCommandInputError(ValueError):
    """A frozen input is absent, unauthenticated, or internally inconsistent."""


class LensCommandGateError(RuntimeError):
    """The ordered compatibility gate failed after writing its audit manifest."""

    def __init__(
        self,
        message: str,
        *,
        compatibility_manifest: LensCompatibilityManifest,
        failure_manifest: LensFailureManifest | None = None,
    ) -> None:
        super().__init__(message)
        self.compatibility_manifest = compatibility_manifest
        self.failure_manifest = failure_manifest


@dataclass(frozen=True, slots=True)
class SmokeRuntimeBundle:
    """4B runtime created lazily only after every frozen input validates."""

    runtime: ModelRuntime
    backend: SameForwardBackend
    post_release: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class PrimaryRuntimeBundle:
    """122B runtime, matched J/R lenses, and one same-forward backend."""

    runtime: ModelRuntime
    lenses: tuple[LoadedLens, LoadedLens]
    backend: SameForwardBackend

    def __post_init__(self) -> None:
        if {lens.lens_type for lens in self.lenses} != {"J", "R"}:
            raise ValueError("primary bundle requires exactly one J and one R lens")


@dataclass(frozen=True, slots=True)
class CompatibilityPrefixes:
    """Content-addressed preflight inputs, frozen before runtime loading."""

    four_b_token_ids: tuple[int, ...]
    four_b_token_ids_hash: str
    primary_trace_id: str
    primary_full_token_ids: tuple[int, ...]
    primary_full_token_ids_hash: str
    primary_short_token_ids: tuple[int, ...]
    primary_short_token_ids_hash: str

    def __post_init__(self) -> None:
        for name in ("four_b_token_ids", "primary_full_token_ids", "primary_short_token_ids"):
            values = tuple(getattr(self, name))
            if not values:
                raise LensCommandInputError(f"{name} must be non-empty")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise LensCommandInputError(f"{name} must contain non-negative integers")
            object.__setattr__(self, name, values)
        if not self.primary_trace_id:
            raise LensCommandInputError("primary_trace_id must be non-empty")
        expected_hashes = {
            "four_b_token_ids_hash": token_stream_hash(
                self.four_b_token_ids, stream="4b_compatibility_prefix"
            ),
            "primary_full_token_ids_hash": token_stream_hash(
                self.primary_full_token_ids, stream="122b_compatibility_full_prefix"
            ),
            "primary_short_token_ids_hash": token_stream_hash(
                self.primary_short_token_ids, stream="122b_compatibility_short_prefix"
            ),
        }
        for name, expected in expected_hashes.items():
            if getattr(self, name) != expected:
                raise LensCommandInputError(f"{name} does not authenticate its exact token IDs")
        if len(self.primary_short_token_ids) >= len(self.primary_full_token_ids):
            raise LensCommandInputError("short primary preflight must be strictly shorter")
        if self.primary_full_token_ids[: len(self.primary_short_token_ids)] != (
            self.primary_short_token_ids
        ):
            raise LensCommandInputError("short primary preflight must be an exact prefix")

    @classmethod
    def freeze(
        cls,
        *,
        four_b_token_ids: Sequence[int],
        primary_trace_id: str,
        primary_full_token_ids: Sequence[int],
        primary_short_token_ids: Sequence[int],
    ) -> CompatibilityPrefixes:
        four_b = tuple(four_b_token_ids)
        full = tuple(primary_full_token_ids)
        short = tuple(primary_short_token_ids)
        return cls(
            four_b_token_ids=four_b,
            four_b_token_ids_hash=token_stream_hash(four_b, stream="4b_compatibility_prefix"),
            primary_trace_id=primary_trace_id,
            primary_full_token_ids=full,
            primary_full_token_ids_hash=token_stream_hash(
                full, stream="122b_compatibility_full_prefix"
            ),
            primary_short_token_ids=short,
            primary_short_token_ids_hash=token_stream_hash(
                short, stream="122b_compatibility_short_prefix"
            ),
        )

    def to_manifest(self) -> CompatibilityPrefixManifest:
        def entry(token_ids: Sequence[int]) -> dict[str, Any]:
            values = tuple(token_ids)
            return {
                "token_count": len(values),
                "token_ids_hash": token_stream_hash(
                    values, stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
                ),
            }

        return CompatibilityPrefixManifest(
            primary_trace_id=self.primary_trace_id,
            four_b=entry(self.four_b_token_ids),
            primary_full=entry(self.primary_full_token_ids),
            primary_short=entry(self.primary_short_token_ids),
        )


@dataclass(frozen=True, slots=True)
class CompatibilityPrefixManifest(CanonicalRecord):
    """Exact, content-addressed prefixes used by every compatibility attempt."""

    primary_trace_id: str
    four_b: Mapping[str, Any]
    primary_full: Mapping[str, Any]
    primary_short: Mapping[str, Any]
    source_policy: str = "4b_pinned_text_and_first_manifest_ordered_primary_trace"
    shortening_policy: str = "strict_prefix_length_frozen_before_runtime"
    protocol_version: str = "lens-compatibility-prefixes-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.primary_trace_id:
            raise LensCommandInputError("compatibility prefix manifest trace ID is empty")
        for name in ("four_b", "primary_full", "primary_short"):
            entry = dict(getattr(self, name))
            if set(entry) != {"token_count", "token_ids_hash"}:
                raise LensCommandInputError(f"{name} prefix evidence has a noncanonical schema")
            if (
                isinstance(entry["token_count"], bool)
                or not isinstance(entry["token_count"], int)
                or entry["token_count"] <= 0
                or not isinstance(entry["token_ids_hash"], str)
                or not entry["token_ids_hash"].startswith("sha256:")
            ):
                raise LensCommandInputError(f"{name} prefix evidence is invalid")
            object.__setattr__(self, name, entry)


@dataclass(frozen=True, slots=True)
class LensCommandPaths:
    rollouts: Path
    anchor_manifest: Path
    position_manifest: Path
    lens_records: Path
    compatibility_prefix_manifest: Path
    compatibility_manifest: Path
    execution_manifest: Path
    failure_manifest: Path

    def __post_init__(self) -> None:
        normalized = [Path(getattr(self, name)).expanduser().resolve() for name in self.__slots__]
        if len(set(normalized)) != len(normalized):
            raise ValueError("lens command input and output paths must be distinct")
        for name, value in zip(self.__slots__, normalized, strict=True):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ValidatedLensInputs:
    traces: tuple[LensTraceInput, ...]
    anchor_manifest_hash: str
    anchor_selection_hash: str
    position_manifest_hash: str
    rollout_manifest_hash: str


@dataclass(frozen=True, slots=True)
class LensExecutionManifest(CanonicalRecord):
    """Content-addressed linkage from frozen inputs to canonical lens rows."""

    anchor_manifest_hash: str
    anchor_selection_hash: str
    position_manifest_hash: str
    rollout_manifest_hash: str
    compatibility_prefix_manifest_hash: str
    compatibility_prefix_manifest_sha256: str
    compatibility_manifest_hash: str
    primary_model_revision: str
    trace_count: int
    record_count: int
    layers: tuple[int, ...]
    lens_records_sha256: str
    lens_records_path: str
    record_schema_version: int
    probe_design_manifest_hash: str
    probe_design_manifest_sha256: str
    candidate_probe_manifest_hash: str
    candidate_probe_manifest_sha256: str
    probe_protocol_version: str
    probe_cell_count: int
    eligible_probe_cell_count: int
    ineligible_probe_cell_count: int
    eligible_record_count: int
    ineligible_record_count: int
    analysis_forward_rule: str
    release_authorization_manifest_hash: str | None = None
    release_authorization_manifest_sha256: str | None = None
    evidence_scope: str = "observational_readout"
    causal_claim: bool = False
    schema_version: int = 3


@dataclass(frozen=True, slots=True)
class LensFailureManifest(CanonicalRecord):
    """Authenticated alternate root after both bounded 122B attempts fail."""

    primary_model_id: str
    primary_model_revision: str
    anchor_manifest_hash: str
    anchor_selection_hash: str
    position_manifest_hash: str
    rollout_manifest_hash: str
    probe_design_manifest_hash: str
    probe_design_manifest_sha256: str
    candidate_probe_manifest_hash: str
    candidate_probe_manifest_sha256: str
    probe_protocol_version: str
    compatibility_prefix_manifest_hash: str
    compatibility_prefix_manifest_sha256: str
    compatibility_manifest_hash: str
    compatibility_manifest_sha256: str
    release_authorization_manifest_hash: str | None = None
    release_authorization_manifest_sha256: str | None = None
    status: str = "primary_122b_lens_unavailable"
    failure_stage: str = "ordered_122b_compatibility_gate"
    failure_policy: str = "two_bounded_version_fixed_attempts_then_behavior_only"
    attempt_count_122b: int = 2
    attempt_strategies: tuple[str, str] = (
        "version_fixed_full_prefix",
        "version_fixed_shortened_prefix",
    )
    all_122b_attempts_failed: bool = True
    lens_records_absent: bool = True
    execution_manifest_absent: bool = True
    analysis_mode: str = "behavior_only"
    lens_evidence_status: str = "unavailable_not_zero"
    lens_claim_eligibility: bool = False
    fallback_27b_policy: str = "methodology_support_only_not_122b_substitute"
    fallback_27b_used_as_primary: bool = False
    causal_claim: bool = False
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.primary_model_id != EXPECTED_MODEL_ID:
            raise ValueError("failure manifest must bind the primary 122B model")
        if self.primary_model_revision != PRIMARY_MODEL_REVISION:
            raise ValueError("failure manifest must bind the primary 122B revision")
        if not (
            self.attempt_count_122b == 2
            and self.all_122b_attempts_failed
            and self.lens_records_absent
            and self.execution_manifest_absent
        ):
            raise ValueError("failure manifest may authenticate only the frozen two-failure root")
        if self.fallback_27b_used_as_primary or self.lens_claim_eligibility or self.causal_claim:
            raise ValueError("failure manifest cannot promote fallback or lens claims")


@dataclass(frozen=True, slots=True)
class LensCommandResult:
    records_written: int
    traces_analyzed: int
    compatibility_manifest: LensCompatibilityManifest
    execution_manifest: LensExecutionManifest
    lens_records_path: Path
    compatibility_prefix_manifest_path: Path
    compatibility_manifest_path: Path
    execution_manifest_path: Path


def required_position_record_schema() -> dict[str, Any]:
    """Return a detached copy suitable for docs or a producer contract."""

    # JSON round-tripping would be needless; recurse only over the small schema.
    def copy_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): copy_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_value(item) for item in value]
        return value

    return copy_value(POSITION_RECORD_SCHEMA)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LensCommandInputError(f"{context} must be a mapping")
    return value


def _require_hash(value: Any, *, context: str, namespaced: bool = True) -> str:
    if not isinstance(value, str):
        raise LensCommandInputError(f"{context} must be a hash string")
    digest = value.removeprefix("sha256:") if namespaced else value
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise LensCommandInputError(f"{context} must contain a lowercase SHA-256 digest")
    if namespaced and not value.startswith("sha256:"):
        raise LensCommandInputError(f"{context} must use the sha256: namespace")
    return value


def _validate_record_hash(row: Mapping[str, Any], *, context: str) -> str:
    observed = _require_hash(row.get("record_hash"), context=f"{context}.record_hash")
    unhashed = {key: value for key, value in row.items() if key != "record_hash"}
    if stable_hash(unhashed) != observed:
        raise LensCommandInputError(f"{context} record_hash does not match its contents")
    return observed


def _parse_anchor_manifest(payload: Mapping[str, Any]) -> AnchorManifest:
    try:
        raw_anchors = payload["anchors"]
        if not isinstance(raw_anchors, Sequence) or isinstance(raw_anchors, (str, bytes)):
            raise TypeError("anchors must be a sequence")
        anchors = tuple(
            FrozenAnchor(
                anchor_id=str(_require_mapping(row, context="anchor")["anchor_id"]),
                trace_id=str(row["trace_id"]),
                sentence_class=str(row["sentence_class"]),
                direction=str(row["direction"]),
                sentence_index=int(row["sentence_index"]),
                sentence_text=str(row["sentence_text"]),
                char_start=int(row["char_start"]),
                char_end=int(row["char_end"]),
                initial_side=str(row["initial_side"]),
                final_flip=row["final_flip"],
                provenance=_require_mapping(row.get("provenance"), context="anchor.provenance"),
            )
            for row in raw_anchors
        )
        manifest = AnchorManifest(
            anchors=anchors,
            sentence_classes=tuple(str(value) for value in payload["sentence_classes"]),
            directions=tuple(str(value) for value in payload["directions"]),
            per_cell=int(payload["per_cell"]),
            seed=str(payload["seed"]),
            selection_hash=str(payload["selection_hash"]),
            schema_version=str(payload.get("schema_version", "")),
        )
        validate_anchor_manifest(manifest)
    except LensCommandInputError:
        raise
    except Exception as exc:
        raise LensCommandInputError(f"invalid frozen anchor manifest: {exc}") from exc
    return manifest


def _validated_anchor_manifest_hash(payload: Mapping[str, Any]) -> str:
    """Authenticate the full augmented anchor manifest, not only selection."""

    declared = payload.get("manifest_hash")
    if declared is None:
        return stable_hash(payload)
    observed = _require_hash(declared, context="anchor_manifest.manifest_hash")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if stable_hash(unhashed) != observed:
        raise LensCommandInputError("anchor_manifest.manifest_hash does not match its contents")
    return observed


def _validated_span(
    source: Any,
    *,
    completion_ids: tuple[int, ...],
    completion_hash: str,
    context: str,
) -> tuple[int, int, Mapping[str, Any]]:
    span = _require_mapping(source, context=context)
    required = {
        "token_start",
        "token_end",
        "token_ids",
        "token_ids_hash",
        "completion_token_ids_hash",
        "round_trip_verified",
    }
    missing = sorted(required - set(span))
    if missing:
        raise LensCommandInputError(f"{context} is missing {missing}")
    start = span["token_start"]
    end = span["token_end"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(completion_ids)
    ):
        raise LensCommandInputError(f"{context} has an invalid completion-token range")
    raw_ids = span["token_ids"]
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise LensCommandInputError(f"{context}.token_ids must be a sequence")
    try:
        span_ids = tuple(int(value) for value in raw_ids)
    except (TypeError, ValueError) as exc:
        raise LensCommandInputError(f"{context}.token_ids must be integers") from exc
    if span_ids != completion_ids[start:end]:
        raise LensCommandInputError(f"{context} does not match the exact completion tokens")
    if span.get("token_ids_hash") != token_stream_hash(span_ids, stream="completion_span"):
        raise LensCommandInputError(f"{context}.token_ids_hash failed validation")
    if span.get("completion_token_ids_hash") != completion_hash:
        raise LensCommandInputError(f"{context} refers to a different completion stream")
    if span.get("round_trip_verified") is not True:
        raise LensCommandInputError(f"{context} lacks strict token/text round-trip verification")
    return start, end, span


def _position_record_for_trace(
    *,
    row: Mapping[str, Any],
    anchor: FrozenAnchor,
    position: Mapping[str, Any],
    anchor_manifest_hash: str,
) -> LensTraceInput:
    trace_id = anchor.trace_id
    _validate_record_hash(position, context=f"position[{trace_id}]")
    if position.get("schema_version") != POSITION_MANIFEST_SCHEMA_VERSION:
        raise LensCommandInputError(f"position[{trace_id}] has an unsupported schema version")
    if position.get("trace_id") != trace_id or position.get("anchor_id") != anchor.anchor_id:
        raise LensCommandInputError(f"position[{trace_id}] does not identify its frozen anchor")
    if position.get("rollout_record_hash") != row["record_hash"]:
        raise LensCommandInputError(f"position[{trace_id}] refers to a different rollout hash")
    if position.get("anchor_manifest_hash") != anchor_manifest_hash:
        raise LensCommandInputError(f"position[{trace_id}] refers to a different anchor manifest")
    _require_hash(
        position.get("first_estimate_span_record_hash"),
        context=f"position[{trace_id}].first_estimate_span_record_hash",
    )
    if position.get("first_estimate_span_instrument_id") != FIRST_ESTIMATE_SPAN_INSTRUMENT_ID:
        raise LensCommandInputError(
            f"position[{trace_id}] lacks the frozen blind first-estimate instrument"
        )
    if position.get("first_estimate_span_primary_inference") is not True:
        raise LensCommandInputError(
            f"position[{trace_id}] first-estimate span is not approved for primary inference"
        )

    token_streams = _require_mapping(
        row.get("token_streams"), context=f"rollout[{trace_id}].token_streams"
    )
    try:
        prompt_ids, completion_ids = validate_token_stream_manifest(
            token_streams, require_both=True
        )
    except Exception as exc:
        raise LensCommandInputError(
            f"rollout[{trace_id}] exact token stream failed validation: {exc}"
        ) from exc
    assert prompt_ids is not None and completion_ids is not None
    if not prompt_ids or not completion_ids:
        raise LensCommandInputError(f"rollout[{trace_id}] has an empty exact token stream")
    combined_hash = token_streams["combined_token_stream_hash"]
    if position.get("combined_token_stream_hash") != combined_hash:
        raise LensCommandInputError(f"position[{trace_id}] refers to different token streams")
    if position.get("prompt_token_ids_hash") != token_streams["prompt_token_ids_hash"]:
        raise LensCommandInputError(f"position[{trace_id}] refers to a different prompt stream")
    if position.get("completion_token_ids_hash") != token_streams["completion_token_ids_hash"]:
        raise LensCommandInputError(f"position[{trace_id}] refers to a different completion stream")
    completion_hash = str(token_streams["completion_token_ids_hash"])

    if position.get("position_order") != list(POSITION_ORDER):
        raise LensCommandInputError(f"position[{trace_id}] changed the preregistered order")
    evidence = _require_mapping(
        position.get("position_evidence"), context=f"position[{trace_id}].position_evidence"
    )
    if set(evidence) != {"first_estimate", "anchor", "answer_first_token"}:
        raise LensCommandInputError(
            f"position[{trace_id}] must authenticate first-estimate, anchor, and answer spans"
        )
    first_start, _, first_span = _validated_span(
        evidence["first_estimate"],
        completion_ids=completion_ids,
        completion_hash=completion_hash,
        context=f"position[{trace_id}].position_evidence.first_estimate",
    )
    if first_span.get("section") not in {"reasoning", "answer"}:
        raise LensCommandInputError(f"position[{trace_id}] first-estimate span has wrong section")

    anchor_start, anchor_end, anchor_span = _validated_span(
        evidence["anchor"],
        completion_ids=completion_ids,
        completion_hash=completion_hash,
        context=f"position[{trace_id}].position_evidence.anchor",
    )
    provenance = _require_mapping(anchor.provenance, context=f"anchor[{trace_id}].provenance")
    frozen_span = _require_mapping(
        provenance.get("token_span"), context=f"anchor[{trace_id}].provenance.token_span"
    )
    if dict(anchor_span) != dict(frozen_span):
        raise LensCommandInputError(f"position[{trace_id}] anchor span changed after selection")
    if provenance.get("source_rollout_hash") != row["record_hash"]:
        raise LensCommandInputError(f"anchor[{trace_id}] refers to a different rollout")
    if provenance.get("completion_token_ids_hash") != completion_hash:
        raise LensCommandInputError(f"anchor[{trace_id}] refers to a different completion")

    final_start, _, final_span = _validated_span(
        evidence["answer_first_token"],
        completion_ids=completion_ids,
        completion_hash=completion_hash,
        context=f"position[{trace_id}].position_evidence.answer_first_token",
    )
    if final_span.get("section") != "answer":
        raise LensCommandInputError(f"position[{trace_id}] final answer span has wrong section")

    prompt_count = len(prompt_ids)
    expected_positions = {
        "prompt_end": prompt_count - 1,
        "first_estimate_pre": prompt_count + first_start - 1,
        "anchor_pre": prompt_count + anchor_start - 1,
        "anchor_post": prompt_count + anchor_end - 1,
        "final_answer_pre": prompt_count + final_start - 1,
    }
    named = _require_mapping(
        position.get("position_indices"), context=f"position[{trace_id}].position_indices"
    )
    if dict(named) != expected_positions:
        raise LensCommandInputError(
            f"position[{trace_id}] named positions do not match authenticated token spans"
        )
    if anchor_end > final_start or first_start > final_start:
        raise LensCommandInputError(
            f"position[{trace_id}] first estimate/anchor must not follow the final answer"
        )

    condition = row.get("condition")
    if condition != anchor.direction or condition not in {"above_good", "below_good"}:
        raise LensCommandInputError(
            f"rollout[{trace_id}] condition disagrees with anchor direction"
        )
    good_side_direction: Literal[-1, 1] = 1 if condition == "above_good" else -1
    if position.get("good_side_direction") != good_side_direction:
        raise LensCommandInputError(f"position[{trace_id}] good-side sign disagrees with anchor")
    if position.get("causal_claim") is not False:
        raise LensCommandInputError(f"position[{trace_id}] may not label lens evidence causal")
    if row.get("direction") not in {None, good_side_direction}:
        raise LensCommandInputError(
            f"rollout[{trace_id}] numeric direction disagrees with condition"
        )
    backend = _require_mapping(row.get("backend"), context=f"rollout[{trace_id}].backend")
    if backend.get("model_id") != PRIMARY_MODEL_PIN_ID:
        raise LensCommandInputError(f"rollout[{trace_id}] is not from the primary 122B model")
    observed_revision = backend.get("model_revision", backend.get("revision"))
    if observed_revision != PRIMARY_MODEL_REVISION:
        raise LensCommandInputError(f"rollout[{trace_id}] model revision is not frozen")
    if row.get("synthetic_smoke") is True:
        raise LensCommandInputError("synthetic smoke rollouts cannot enter primary lens analysis")

    return LensTraceInput.from_token_stream_manifest(
        trace_id=trace_id,
        token_streams=token_streams,
        position_indices=expected_positions,
        good_side_direction=good_side_direction,
    )


# Kept separate from the runner constant to make the input error explicit and
# avoid accepting a model ID merely because its geometry happens to match.
PRIMARY_MODEL_PIN_ID = "Qwen/Qwen3.5-122B-A10B"


def validate_frozen_lens_inputs(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    anchor_manifest: Mapping[str, Any],
    position_records: Sequence[Mapping[str, Any]],
) -> ValidatedLensInputs:
    """Authenticate all frozen joins without loading a model or parsing numbers."""

    manifest = _parse_anchor_manifest(anchor_manifest)
    anchor_manifest_hash = _validated_anchor_manifest_hash(anchor_manifest)
    rollout_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rollouts, start=1):
        trace_id = row.get("run_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise LensCommandInputError(f"rollout row {index} lacks run_id")
        if trace_id in rollout_by_id:
            raise LensCommandInputError(f"duplicate rollout run_id: {trace_id}")
        _validate_record_hash(row, context=f"rollout[{trace_id}]")
        rollout_by_id[trace_id] = row

    position_by_trace: dict[str, Mapping[str, Any]] = {}
    for index, position in enumerate(position_records, start=1):
        trace_id = position.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise LensCommandInputError(f"position row {index} lacks trace_id")
        if trace_id in position_by_trace:
            raise LensCommandInputError(f"duplicate position trace_id: {trace_id}")
        position_by_trace[trace_id] = position

    selected_ids = {anchor.trace_id for anchor in manifest.anchors}
    if set(position_by_trace) != selected_ids:
        raise LensCommandInputError(
            "position manifest must contain exactly one row for every selected anchor; "
            f"missing={sorted(selected_ids - set(position_by_trace))}, "
            f"extra={sorted(set(position_by_trace) - selected_ids)}"
        )
    missing_rollouts = selected_ids - set(rollout_by_id)
    if missing_rollouts:
        raise LensCommandInputError(f"selected rollouts are absent: {sorted(missing_rollouts)}")

    traces = tuple(
        _position_record_for_trace(
            row=rollout_by_id[anchor.trace_id],
            anchor=anchor,
            position=position_by_trace[anchor.trace_id],
            anchor_manifest_hash=anchor_manifest_hash,
        )
        for anchor in manifest.anchors
    )
    return ValidatedLensInputs(
        traces=traces,
        anchor_manifest_hash=anchor_manifest_hash,
        anchor_selection_hash=manifest.selection_hash,
        position_manifest_hash=stable_hash(
            [position_by_trace[trace.trace_id]["record_hash"] for trace in traces]
        ),
        rollout_manifest_hash=stable_hash(
            [rollout_by_id[trace.trace_id]["record_hash"] for trace in traces]
        ),
    )


def _validate_compatibility_prefixes(
    prefixes: CompatibilityPrefixes,
    traces: Sequence[LensTraceInput],
) -> None:
    by_id = {trace.trace_id: trace for trace in traces}
    source = by_id.get(prefixes.primary_trace_id)
    if source is None:
        raise LensCommandInputError("primary compatibility trace is not a selected frozen trace")
    expected = source.sequence_token_ids[: max(source.position_indices.values()) + 1]
    if prefixes.primary_full_token_ids != expected:
        raise LensCommandInputError(
            "full primary compatibility input must equal the authenticated analysis forward"
        )


def validate_probe_design(
    probe_design: LensProbeDesign,
    validated: ValidatedLensInputs,
) -> None:
    """Bind the frozen probe universe and every causal cell to frozen inputs."""

    expected_links = {
        "anchor_manifest_hash": validated.anchor_manifest_hash,
        "anchor_selection_hash": validated.anchor_selection_hash,
        "rollout_manifest_hash": validated.rollout_manifest_hash,
        "position_manifest_hash": validated.position_manifest_hash,
    }
    if probe_design.model_id != EXPECTED_MODEL_ID or any(
        getattr(probe_design, field) != expected for field, expected in expected_links.items()
    ):
        raise LensCommandInputError("probe design disagrees with the frozen inputs")
    expected_keys = {
        (trace.trace_id, position, concept)
        for trace in validated.traces
        for position in POSITION_ORDER
        for concept in probe_design.concepts
    }
    observed_keys = {
        (cell.trace_id, cell.position_name, cell.concept) for cell in probe_design.cells
    }
    if observed_keys != expected_keys or len(probe_design.cells) != len(expected_keys):
        raise LensCommandInputError("probe design cell inventory disagrees with frozen inputs")
    traces = {trace.trace_id: trace for trace in validated.traces}
    for cell in probe_design.cells:
        trace = traces[cell.trace_id]
        token_index = trace.position_indices[cell.position_name]
        causal_ids = trace.sequence_token_ids[: token_index + 1]
        if (
            cell.token_index != token_index
            or cell.causal_prefix_token_count != len(causal_ids)
            or cell.causal_prefix_token_ids_hash
            != token_stream_hash(causal_ids, stream="lens_causal_prefix")
        ):
            raise LensCommandInputError(
                "probe design causal-prefix evidence disagrees with frozen inputs"
            )


def _is_authenticated_two_attempt_primary_failure(
    manifest: LensCompatibilityManifest,
) -> bool:
    smoke = [attempt for attempt in manifest.attempts if attempt.stage == "4b_smoke"]
    primary = [attempt for attempt in manifest.attempts if attempt.stage == "122b_preflight"]
    return (
        not manifest.primary_ready
        and len(smoke) == 1
        and smoke[0].status == "passed"
        and len(primary) == 2
        and tuple(attempt.ordinal for attempt in primary) == (1, 2)
        and tuple(attempt.strategy for attempt in primary)
        == ("version_fixed_full_prefix", "version_fixed_shortened_prefix")
        and all(attempt.status == "failed" for attempt in primary)
    )


def validate_compatibility_prefix_manifest(
    prefix_manifest: CompatibilityPrefixManifest,
    compatibility_manifest: LensCompatibilityManifest,
) -> None:
    """Bind every compatibility attempt to its exact frozen token prefix."""

    expected_by_attempt = {
        ("4b_smoke", "pinned_text_only_single_forward"): prefix_manifest.four_b,
        ("122b_preflight", "version_fixed_full_prefix"): prefix_manifest.primary_full,
        (
            "122b_preflight",
            "version_fixed_shortened_prefix",
        ): prefix_manifest.primary_short,
    }
    seen: set[tuple[str, str]] = set()
    for attempt in compatibility_manifest.attempts:
        identity = (attempt.stage, attempt.strategy)
        expected = expected_by_attempt.get(identity)
        if expected is None or identity in seen:
            raise LensCommandInputError(
                "compatibility manifest contains an unknown or duplicate attempt"
            )
        seen.add(identity)
        if (
            attempt.prefix_token_count != expected["token_count"]
            or attempt.prefix_token_ids_hash != expected["token_ids_hash"]
        ):
            raise LensCommandInputError(
                "compatibility attempt disagrees with its exact prefix manifest"
            )


def validate_lens_failure_manifest(
    payload: Mapping[str, Any],
    *,
    compatibility_manifest: LensCompatibilityManifest,
    compatibility_manifest_sha256: str,
    compatibility_prefix_manifest: CompatibilityPrefixManifest,
    compatibility_prefix_manifest_sha256: str,
    release_authorization_manifest_hash: str | None = None,
    release_authorization_manifest_sha256: str | None = None,
    validated: ValidatedLensInputs,
    probe_design: LensProbeDesign,
    probe_design_manifest_sha256: str,
    lens_records_path: str | Path,
    execution_manifest_path: str | Path,
) -> LensFailureManifest:
    """Validate the sole behavior-only alternate root, including absent outputs."""

    if not _is_authenticated_two_attempt_primary_failure(compatibility_manifest):
        raise LensCommandInputError(
            "failure manifest requires exactly two authenticated failed 122B attempts"
        )
    for value, context in (
        (compatibility_manifest_sha256, "compatibility_manifest_sha256"),
        (
            compatibility_prefix_manifest_sha256,
            "compatibility_prefix_manifest_sha256",
        ),
        (probe_design_manifest_sha256, "probe_design_manifest_sha256"),
    ):
        _require_hash(value, context=context, namespaced=False)
    if (release_authorization_manifest_hash is None) != (
        release_authorization_manifest_sha256 is None
    ):
        raise LensCommandInputError(
            "release authorization hash and SHA-256 must be supplied together"
        )
    if release_authorization_manifest_hash is not None:
        _require_hash(
            release_authorization_manifest_hash,
            context="release_authorization_manifest_hash",
        )
        assert release_authorization_manifest_sha256 is not None
        _require_hash(
            release_authorization_manifest_sha256,
            context="release_authorization_manifest_sha256",
            namespaced=False,
        )
    validate_compatibility_prefix_manifest(
        compatibility_prefix_manifest, compatibility_manifest
    )
    validate_probe_design(probe_design, validated)
    if Path(lens_records_path).expanduser().resolve().exists() or Path(
        execution_manifest_path
    ).expanduser().resolve().exists():
        raise LensCommandInputError("failure root requires absent lens and execution artifacts")
    expected = LensFailureManifest(
        primary_model_id=EXPECTED_MODEL_ID,
        primary_model_revision=PRIMARY_MODEL_REVISION,
        anchor_manifest_hash=validated.anchor_manifest_hash,
        anchor_selection_hash=validated.anchor_selection_hash,
        position_manifest_hash=validated.position_manifest_hash,
        rollout_manifest_hash=validated.rollout_manifest_hash,
        probe_design_manifest_hash=probe_design.manifest_hash,
        probe_design_manifest_sha256=probe_design_manifest_sha256,
        candidate_probe_manifest_hash=probe_design.candidate_probe_manifest_hash,
        candidate_probe_manifest_sha256=probe_design.candidate_probe_manifest_sha256,
        probe_protocol_version=probe_design.protocol_version,
        compatibility_prefix_manifest_hash=compatibility_prefix_manifest.record_hash,
        compatibility_prefix_manifest_sha256=compatibility_prefix_manifest_sha256,
        compatibility_manifest_hash=compatibility_manifest.record_hash,
        compatibility_manifest_sha256=compatibility_manifest_sha256,
        release_authorization_manifest_hash=release_authorization_manifest_hash,
        release_authorization_manifest_sha256=release_authorization_manifest_sha256,
    )
    if dict(payload) != expected.to_dict(include_hash=True):
        raise LensCommandInputError("lens failure manifest disagrees with authenticated evidence")
    return expected


def run_frozen_lens_command(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    anchor_manifest: Mapping[str, Any],
    position_records: Sequence[Mapping[str, Any]],
    compatibility_prefixes: CompatibilityPrefixes,
    probe_design: LensProbeDesign,
    probe_design_manifest_sha256: str | None = None,
    smoke_runtime_factory: Callable[[], SmokeRuntimeBundle],
    primary_runtime_factory: Callable[[], PrimaryRuntimeBundle],
    lens_records_path: str | Path,
    compatibility_manifest_path: str | Path,
    execution_manifest_path: str | Path,
    compatibility_prefix_manifest_path: str | Path | None = None,
    failure_manifest_path: str | Path | None = None,
    release_authorization_manifest_hash: str | None = None,
    release_authorization_manifest_sha256: str | None = None,
    layers: Sequence[int] = FITTED_LAYERS,
) -> LensCommandResult:
    """Validate, gate, execute, and persist the real observational lens job."""

    validated = validate_frozen_lens_inputs(
        rollouts=rollouts,
        anchor_manifest=anchor_manifest,
        position_records=position_records,
    )
    _validate_compatibility_prefixes(compatibility_prefixes, validated.traces)
    validate_probe_design(probe_design, validated)
    if probe_design_manifest_sha256 is not None:
        _require_hash(
            probe_design_manifest_sha256,
            context="probe_design_manifest_sha256",
            namespaced=False,
        )
    if (release_authorization_manifest_hash is None) != (
        release_authorization_manifest_sha256 is None
    ):
        raise LensCommandInputError(
            "release authorization hash and SHA-256 must be supplied together"
        )
    if release_authorization_manifest_hash is not None:
        _require_hash(
            release_authorization_manifest_hash,
            context="release_authorization_manifest_hash",
        )
        assert release_authorization_manifest_sha256 is not None
        _require_hash(
            release_authorization_manifest_sha256,
            context="release_authorization_manifest_sha256",
            namespaced=False,
        )
    output = Path(lens_records_path).expanduser().resolve()
    compatibility_path = Path(compatibility_manifest_path).expanduser().resolve()
    execution_path = Path(execution_manifest_path).expanduser().resolve()
    prefix_manifest_path = (
        Path(compatibility_prefix_manifest_path).expanduser().resolve()
        if compatibility_prefix_manifest_path is not None
        else compatibility_path.with_name("lens_compatibility_prefix_manifest.json")
    )
    failure_path = (
        Path(failure_manifest_path).expanduser().resolve()
        if failure_manifest_path is not None
        else compatibility_path.with_name("lens_failure_manifest.json")
    )
    if len({output, prefix_manifest_path, compatibility_path, execution_path, failure_path}) != 5:
        raise LensCommandInputError("lens output paths must be distinct")
    if output.exists() or execution_path.exists():
        raise LensCommandInputError(
            "lens records and execution manifest must be absent before primary execution"
        )
    if failure_path.exists():
        raise LensCommandInputError(
            "an authenticated prior 122B lens-failure root forbids additional primary attempts"
        )
    prefix_manifest = compatibility_prefixes.to_manifest()
    write_json(prefix_manifest_path, prefix_manifest.to_dict(include_hash=True))
    prefix_manifest_sha256 = sha256_file(prefix_manifest_path)
    normalized_layers = tuple(layers)

    primary_state: dict[str, PrimaryRuntimeBundle] = {}

    def run_smoke(ids: Sequence[int]) -> Mapping[str, Any]:
        bundle = smoke_runtime_factory()
        if not isinstance(bundle, SmokeRuntimeBundle):
            raise TypeError("smoke_runtime_factory must return SmokeRuntimeBundle")
        post_release = bundle.post_release
        try:
            return run_4b_compatibility_smoke(
                bundle.runtime,
                token_ids=ids,
                backend=bundle.backend,
            )
        finally:
            bundle = None
            gc.collect()
            if post_release is not None:
                post_release()

    def run_primary(ids: Sequence[int]) -> Mapping[str, Any]:
        if "bundle" not in primary_state:
            bundle = primary_runtime_factory()
            if not isinstance(bundle, PrimaryRuntimeBundle):
                raise TypeError("primary_runtime_factory must return PrimaryRuntimeBundle")
            primary_state["bundle"] = bundle
        bundle = primary_state["bundle"]
        return run_122b_preflight(
            bundle.runtime,
            bundle.lenses,
            token_ids=ids,
            backend=bundle.backend,
            probe_design=probe_design,
        )

    try:
        compatibility = run_ordered_compatibility_gate(
            four_b_prefix_token_ids=compatibility_prefixes.four_b_token_ids,
            primary_prefix_token_ids=compatibility_prefixes.primary_full_token_ids,
            shortened_primary_prefix_token_ids=compatibility_prefixes.primary_short_token_ids,
            four_b_runner=run_smoke,
            primary_runner=run_primary,
        )
    except CompatibilityGateError as exc:
        write_compatibility_manifest(compatibility_path, exc.manifest)
        failure: LensFailureManifest | None = None
        if _is_authenticated_two_attempt_primary_failure(exc.manifest):
            failure = LensFailureManifest(
                primary_model_id=EXPECTED_MODEL_ID,
                primary_model_revision=PRIMARY_MODEL_REVISION,
                anchor_manifest_hash=validated.anchor_manifest_hash,
                anchor_selection_hash=validated.anchor_selection_hash,
                position_manifest_hash=validated.position_manifest_hash,
                rollout_manifest_hash=validated.rollout_manifest_hash,
                probe_design_manifest_hash=probe_design.manifest_hash,
                probe_design_manifest_sha256=(
                    probe_design_manifest_sha256
                    or probe_design.manifest_hash.removeprefix("sha256:")
                ),
                candidate_probe_manifest_hash=probe_design.candidate_probe_manifest_hash,
                candidate_probe_manifest_sha256=probe_design.candidate_probe_manifest_sha256,
                probe_protocol_version=probe_design.protocol_version,
                compatibility_prefix_manifest_hash=prefix_manifest.record_hash,
                compatibility_prefix_manifest_sha256=prefix_manifest_sha256,
                compatibility_manifest_hash=exc.manifest.record_hash,
                compatibility_manifest_sha256=sha256_file(compatibility_path),
                release_authorization_manifest_hash=(
                    release_authorization_manifest_hash
                ),
                release_authorization_manifest_sha256=(
                    release_authorization_manifest_sha256
                ),
            )
            write_json(failure_path, failure.to_dict(include_hash=True))
        raise LensCommandGateError(
            str(exc),
            compatibility_manifest=exc.manifest,
            failure_manifest=failure,
        ) from exc
    write_compatibility_manifest(compatibility_path, compatibility)

    bundle = primary_state.get("bundle")
    if bundle is None:  # pragma: no cover - guaranteed by a passed primary gate
        raise RuntimeError("passed compatibility gate produced no primary runtime")
    records = execute_lens_traces(
        validated.traces,
        runtime=bundle.runtime,
        lenses=bundle.lenses,
        backend=bundle.backend,
        probe_design=probe_design,
        output_path=output,
        layers=normalized_layers,
    )
    eligible_cells = sum(cell.probe_eligible for cell in probe_design.cells)
    rows_per_cell = 2 * len(normalized_layers)
    manifest = LensExecutionManifest(
        anchor_manifest_hash=validated.anchor_manifest_hash,
        anchor_selection_hash=validated.anchor_selection_hash,
        position_manifest_hash=validated.position_manifest_hash,
        rollout_manifest_hash=validated.rollout_manifest_hash,
        compatibility_prefix_manifest_hash=prefix_manifest.record_hash,
        compatibility_prefix_manifest_sha256=prefix_manifest_sha256,
        compatibility_manifest_hash=compatibility.record_hash,
        primary_model_revision=PRIMARY_MODEL_REVISION,
        trace_count=len(validated.traces),
        record_count=len(records),
        layers=normalized_layers,
        lens_records_sha256=sha256_file(output),
        lens_records_path=str(output),
        record_schema_version=2,
        probe_design_manifest_hash=probe_design.manifest_hash,
        probe_design_manifest_sha256=(
            probe_design_manifest_sha256
            or probe_design.manifest_hash.removeprefix("sha256:")
        ),
        candidate_probe_manifest_hash=probe_design.candidate_probe_manifest_hash,
        candidate_probe_manifest_sha256=probe_design.candidate_probe_manifest_sha256,
        probe_protocol_version=probe_design.protocol_version,
        probe_cell_count=len(probe_design.cells),
        eligible_probe_cell_count=eligible_cells,
        ineligible_probe_cell_count=len(probe_design.cells) - eligible_cells,
        eligible_record_count=eligible_cells * rows_per_cell,
        ineligible_record_count=(len(probe_design.cells) - eligible_cells) * rows_per_cell,
        analysis_forward_rule="max_authenticated_position_inclusive",
        release_authorization_manifest_hash=release_authorization_manifest_hash,
        release_authorization_manifest_sha256=release_authorization_manifest_sha256,
    )
    write_json(execution_path, manifest.to_dict(include_hash=True))
    return LensCommandResult(
        records_written=len(records),
        traces_analyzed=len(validated.traces),
        compatibility_manifest=compatibility,
        execution_manifest=manifest,
        lens_records_path=output,
        compatibility_prefix_manifest_path=prefix_manifest_path,
        compatibility_manifest_path=compatibility_path,
        execution_manifest_path=execution_path,
    )


def run_frozen_lens_command_from_files(
    paths: LensCommandPaths,
    *,
    compatibility_prefixes: CompatibilityPrefixes,
    probe_design: LensProbeDesign,
    probe_design_manifest_sha256: str | None = None,
    release_authorization_manifest_hash: str | None = None,
    release_authorization_manifest_sha256: str | None = None,
    smoke_runtime_factory: Callable[[], SmokeRuntimeBundle],
    primary_runtime_factory: Callable[[], PrimaryRuntimeBundle],
    layers: Sequence[int] = FITTED_LAYERS,
) -> LensCommandResult:
    """File-backed adapter for CLI/RunPod callers; still contains no parser."""

    for path in (paths.rollouts, paths.anchor_manifest, paths.position_manifest):
        if not path.is_file():
            raise LensCommandInputError(f"required frozen lens input is absent: {path}")
    anchor_payload = read_json(paths.anchor_manifest)
    if not isinstance(anchor_payload, Mapping):
        raise LensCommandInputError("anchor manifest file must contain an object")
    return run_frozen_lens_command(
        rollouts=read_jsonl(paths.rollouts),
        anchor_manifest=anchor_payload,
        position_records=read_jsonl(paths.position_manifest),
        compatibility_prefixes=compatibility_prefixes,
        probe_design=probe_design,
        probe_design_manifest_sha256=probe_design_manifest_sha256,
        release_authorization_manifest_hash=release_authorization_manifest_hash,
        release_authorization_manifest_sha256=release_authorization_manifest_sha256,
        smoke_runtime_factory=smoke_runtime_factory,
        primary_runtime_factory=primary_runtime_factory,
        lens_records_path=paths.lens_records,
        compatibility_prefix_manifest_path=paths.compatibility_prefix_manifest,
        compatibility_manifest_path=paths.compatibility_manifest,
        execution_manifest_path=paths.execution_manifest,
        failure_manifest_path=paths.failure_manifest,
        layers=layers,
    )


__all__ = [
    "POSITION_RECORD_SCHEMA",
    "CompatibilityPrefixManifest",
    "CompatibilityPrefixes",
    "LensCommandGateError",
    "LensCommandInputError",
    "LensCommandPaths",
    "LensCommandResult",
    "LensExecutionManifest",
    "LensFailureManifest",
    "PrimaryRuntimeBundle",
    "SmokeRuntimeBundle",
    "ValidatedLensInputs",
    "required_position_record_schema",
    "run_frozen_lens_command",
    "run_frozen_lens_command_from_files",
    "validate_compatibility_prefix_manifest",
    "validate_frozen_lens_inputs",
    "validate_lens_failure_manifest",
    "validate_probe_design",
]
