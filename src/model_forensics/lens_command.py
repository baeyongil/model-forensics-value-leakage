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
from model_forensics.lens import FITTED_LAYERS, LoadedLens, ModelRuntime
from model_forensics.lens_positions import (
    POSITION_MANIFEST_SCHEMA_VERSION,
    POSITION_ORDER,
)
from model_forensics.lens_runner import (
    PRIMARY_MODEL_REVISION,
    CompatibilityGateError,
    LensCompatibilityManifest,
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
    ) -> None:
        super().__init__(message)
        self.compatibility_manifest = compatibility_manifest


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


@dataclass(frozen=True, slots=True)
class LensCommandPaths:
    rollouts: Path
    anchor_manifest: Path
    position_manifest: Path
    lens_records: Path
    compatibility_manifest: Path
    execution_manifest: Path

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
    compatibility_manifest_hash: str
    primary_model_revision: str
    trace_count: int
    record_count: int
    layers: tuple[int, ...]
    lens_records_sha256: str
    lens_records_path: str
    evidence_scope: str = "observational_readout"
    causal_claim: bool = False
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class LensCommandResult:
    records_written: int
    traces_analyzed: int
    compatibility_manifest: LensCompatibilityManifest
    execution_manifest: LensExecutionManifest
    lens_records_path: Path
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
    if prefixes.primary_full_token_ids != source.sequence_token_ids:
        raise LensCommandInputError(
            "full primary compatibility input must equal its selected rollout token stream"
        )


def run_frozen_lens_command(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    anchor_manifest: Mapping[str, Any],
    position_records: Sequence[Mapping[str, Any]],
    compatibility_prefixes: CompatibilityPrefixes,
    smoke_runtime_factory: Callable[[], SmokeRuntimeBundle],
    primary_runtime_factory: Callable[[], PrimaryRuntimeBundle],
    lens_records_path: str | Path,
    compatibility_manifest_path: str | Path,
    execution_manifest_path: str | Path,
    layers: Sequence[int] = FITTED_LAYERS,
) -> LensCommandResult:
    """Validate, gate, execute, and persist the real observational lens job."""

    validated = validate_frozen_lens_inputs(
        rollouts=rollouts,
        anchor_manifest=anchor_manifest,
        position_records=position_records,
    )
    _validate_compatibility_prefixes(compatibility_prefixes, validated.traces)
    output = Path(lens_records_path).expanduser().resolve()
    compatibility_path = Path(compatibility_manifest_path).expanduser().resolve()
    execution_path = Path(execution_manifest_path).expanduser().resolve()
    if len({output, compatibility_path, execution_path}) != 3:
        raise LensCommandInputError("lens output paths must be distinct")
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
        raise LensCommandGateError(str(exc), compatibility_manifest=exc.manifest) from exc
    write_compatibility_manifest(compatibility_path, compatibility)

    bundle = primary_state.get("bundle")
    if bundle is None:  # pragma: no cover - guaranteed by a passed primary gate
        raise RuntimeError("passed compatibility gate produced no primary runtime")
    records = execute_lens_traces(
        validated.traces,
        runtime=bundle.runtime,
        lenses=bundle.lenses,
        backend=bundle.backend,
        output_path=output,
        layers=normalized_layers,
    )
    manifest = LensExecutionManifest(
        anchor_manifest_hash=validated.anchor_manifest_hash,
        anchor_selection_hash=validated.anchor_selection_hash,
        position_manifest_hash=validated.position_manifest_hash,
        rollout_manifest_hash=validated.rollout_manifest_hash,
        compatibility_manifest_hash=compatibility.record_hash,
        primary_model_revision=PRIMARY_MODEL_REVISION,
        trace_count=len(validated.traces),
        record_count=len(records),
        layers=normalized_layers,
        lens_records_sha256=sha256_file(output),
        lens_records_path=str(output),
    )
    write_json(execution_path, manifest.to_dict(include_hash=True))
    return LensCommandResult(
        records_written=len(records),
        traces_analyzed=len(validated.traces),
        compatibility_manifest=compatibility,
        execution_manifest=manifest,
        lens_records_path=output,
        compatibility_manifest_path=compatibility_path,
        execution_manifest_path=execution_path,
    )


def run_frozen_lens_command_from_files(
    paths: LensCommandPaths,
    *,
    compatibility_prefixes: CompatibilityPrefixes,
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
        smoke_runtime_factory=smoke_runtime_factory,
        primary_runtime_factory=primary_runtime_factory,
        lens_records_path=paths.lens_records,
        compatibility_manifest_path=paths.compatibility_manifest,
        execution_manifest_path=paths.execution_manifest,
        layers=layers,
    )


__all__ = [
    "POSITION_RECORD_SCHEMA",
    "CompatibilityPrefixes",
    "LensCommandGateError",
    "LensCommandInputError",
    "LensCommandPaths",
    "LensCommandResult",
    "LensExecutionManifest",
    "PrimaryRuntimeBundle",
    "SmokeRuntimeBundle",
    "ValidatedLensInputs",
    "required_position_record_schema",
    "run_frozen_lens_command",
    "run_frozen_lens_command_from_files",
    "validate_frozen_lens_inputs",
]
