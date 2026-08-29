"""Pinned, same-forward GPU execution for the preregistered J/R lenses.

This module is the only place where the observational lens job is allowed to
load a Hugging Face model or download lens weights.  Importing it is safe in a
CPU-only environment: heavyweight dependencies are imported lazily and every
GPU-facing operation supports dependency injection for deterministic tests.

The central invariant is that a trace is forwarded exactly once.  Residuals
from that one pass are then supplied to both the standard Jacobian (J) and
RelP (R) transports.  Calling ``JacobianLens.apply`` twice would violate this
invariant and is intentionally not used here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

from model_forensics.io import stable_hash, write_json, write_jsonl
from model_forensics.lens import (
    ARTIFACT_SOURCE_LAYERS,
    DEFAULT_CONCEPT_WORDS,
    EXPECTED_D_MODEL,
    EXPECTED_MODEL_ID,
    EXPECTED_N_LAYERS,
    EXPECTED_TARGET_LAYER,
    FITTED_LAYERS,
    ConceptTokenIds,
    ConceptValidationError,
    LensExecutionError,
    LensRecord,
    LoadedLens,
    ModelRuntime,
    OptionalDependencyError,
    PositionMappingError,
    ProvenanceError,
    layer_band,
    load_local_lens,
    signed_mean_logit_contrasts,
)
from model_forensics.schemas import CanonicalRecord
from model_forensics.token_spans import (
    token_stream_hash,
    token_stream_manifest,
    validate_token_stream_manifest,
)

TRANSFORMERS_REVISION = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
PRIMARY_MODEL_REVISION = "dc4d348443bc740c68e2d77492492c11606384d5"
PRIMARY_LENS_REPOSITORY = "camilablank/workspace-lenses"
PRIMARY_LENS_REVISION = "d740106d1e0f95456dc8718fba2895e9c8ffd6ef"
PRIMARY_MAX_SEQUENCE_TOKENS = 65_536
SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
SMOKE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SMOKE_D_MODEL = 2560
SMOKE_N_LAYERS = 32
COMPATIBILITY_ATTEMPT_PREFIX_STREAM = "lens_compatibility_attempt_prefix"

POSITION_ORDER = (
    "prompt_end",
    "first_estimate_pre",
    "anchor_pre",
    "anchor_post",
    "final_answer_pre",
)

FROZEN_PROBE_TOKEN_IDS: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "direction": {
        "positive": (38453, 5096, 68304),
        "negative": (43001, 17921, 89495),
    },
    "valence": {
        "positive": (22957, 10631, 22135),
        "negative": (26857, 32243, 15866),
    },
    "epistemic": {
        "positive": (16050, 65441, 86775),
        "negative": (46073, 77507, 83484),
    },
}


class CompatibilityGateError(LensExecutionError):
    """Raised when the ordered 4B/122B compatibility gate does not pass."""

    def __init__(self, message: str, manifest: LensCompatibilityManifest) -> None:
        super().__init__(message)
        self.manifest = manifest


class PrimaryCompatibilityFailure(CompatibilityGateError):
    """The two explicitly limited 122B attempts both failed."""


@dataclass(frozen=True, slots=True)
class ProbeCollision(CanonicalRecord):
    """One frozen probe found in a cell's exact causal prefix."""

    polarity: Literal["positive", "negative"]
    word: str
    token_id: int
    exact_token_id_present: bool
    lexical_word_present: bool

    def __post_init__(self) -> None:
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("probe collision polarity must be positive or negative")
        if not self.word or self.token_id < 0:
            raise ValueError("probe collision word/token ID is invalid")
        if not (self.exact_token_id_present or self.lexical_word_present):
            raise ValueError("a probe collision needs token-ID or lexical evidence")


@dataclass(frozen=True, slots=True)
class ProbeCellEligibility(CanonicalRecord):
    """Outcome-blind eligibility of one trace x position x concept cell."""

    trace_id: str
    position_name: str
    concept: str
    token_index: int
    causal_prefix_token_count: int
    causal_prefix_token_ids_hash: str
    probe_eligible: bool
    probe_ineligibility_reason: str | None
    collisions: tuple[ProbeCollision, ...]
    collision_evidence_hash: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collisions", tuple(self.collisions))
        if not self.trace_id or self.position_name not in POSITION_ORDER or not self.concept:
            raise ValueError("probe cell identity is invalid")
        if self.token_index < 0 or self.causal_prefix_token_count != self.token_index + 1:
            raise ValueError("probe cell causal-prefix boundary is invalid")
        if self.probe_eligible:
            if self.collisions or self.probe_ineligibility_reason is not None:
                raise ValueError("eligible probe cells may not contain collision evidence")
            if self.collision_evidence_hash is not None:
                raise ValueError("eligible probe cells may not have a collision hash")
        else:
            if self.probe_ineligibility_reason != "causal_prefix_probe_collision":
                raise ValueError("ineligible probe cells require the frozen collision reason")
            if not self.collisions or self.collision_evidence_hash is None:
                raise ValueError("ineligible probe cells require collision evidence")


@dataclass(frozen=True, slots=True)
class LensProbeDesign:
    """One fixed probe universe plus a causal, cell-level eligibility mask."""

    model_id: str
    tokenizer_id: str
    tokenizer_revision: str
    candidate_probe_manifest_hash: str
    candidate_probe_manifest_sha256: str
    anchor_manifest_hash: str
    anchor_selection_hash: str
    rollout_manifest_hash: str
    position_manifest_hash: str
    concepts: Mapping[str, ConceptTokenIds]
    cells: tuple[ProbeCellEligibility, ...]
    schema_version: int = 1
    protocol_version: str = "fixed-common-probes-causal-cell-eligibility-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", dict(self.concepts))
        object.__setattr__(self, "cells", tuple(self.cells))
        if set(self.concepts) != set(DEFAULT_CONCEPT_WORDS):
            raise ValueError("probe design concepts disagree with the frozen universe")
        for concept, probes in self.concepts.items():
            if not probes.positive_ids or not probes.negative_ids:
                raise ConceptValidationError(
                    f"concept {concept!r} has empty polarity coverage"
                )
        identities = [
            (cell.trace_id, cell.position_name, cell.concept) for cell in self.cells
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("probe design contains duplicate cells")

    def cell_for(self, trace_id: str, position_name: str, concept: str) -> ProbeCellEligibility:
        matches = [
            cell
            for cell in self.cells
            if (cell.trace_id, cell.position_name, cell.concept)
            == (trace_id, position_name, concept)
        ]
        if len(matches) != 1:
            raise ProvenanceError(
                f"probe design has {len(matches)} cells for {trace_id}/{position_name}/{concept}"
            )
        return matches[0]

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        concepts = {
            concept: {
                "positive_words": list(probes.positive_words),
                "positive_token_ids": list(probes.positive_ids),
                "negative_words": list(probes.negative_words),
                "negative_token_ids": list(probes.negative_ids),
            }
            for concept, probes in self.concepts.items()
        }
        cells = [cell.to_dict(include_hash=True) for cell in self.cells]
        eligible = sum(cell.probe_eligible for cell in self.cells)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "trust_remote_code": False,
            "candidate_probe_manifest_hash": self.candidate_probe_manifest_hash,
            "candidate_probe_manifest_sha256": self.candidate_probe_manifest_sha256,
            "anchor_manifest_hash": self.anchor_manifest_hash,
            "anchor_selection_hash": self.anchor_selection_hash,
            "rollout_manifest_hash": self.rollout_manifest_hash,
            "position_manifest_hash": self.position_manifest_hash,
            "position_order": list(POSITION_ORDER),
            "causal_prefix_rule": "combined_token_ids_zero_through_position_inclusive",
            "collision_checks": [
                "exact_token_id",
                "decoded_casefolded_lexical_word_boundary",
            ],
            "collision_action": "whole_trace_position_concept_cell_ineligible",
            "individual_probe_filtering": False,
            "empty_polarity_policy": "abort_before_any_model_forward",
            "forward_input_rule": (
                "combined_token_ids_zero_through_max_authenticated_position_inclusive"
            ),
            "selection_inputs": [
                "frozen_probe_candidates",
                "exact_combined_token_ids",
                "authenticated_position_indices",
            ],
            "forbidden_selection_inputs": [
                "final_estimate",
                "final_good_side",
                "resampling_outcomes",
                "lens_logits",
            ],
            "concepts": concepts,
            "cells": cells,
            "cell_count": len(cells),
            "eligible_cell_count": eligible,
            "ineligible_cell_count": len(cells) - eligible,
        }
        if include_hash:
            payload["manifest_hash"] = stable_hash(payload)
        return payload

    @property
    def manifest_hash(self) -> str:
        return str(self.to_manifest(include_hash=True)["manifest_hash"])


@dataclass(frozen=True, slots=True)
class ModelPin:
    """Immutable model identity and text-decoder geometry."""

    model_id: str
    revision: str
    d_model: int
    n_layers: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("model ID and revision must be non-empty")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("model revision must be a lowercase 40-character commit")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("model geometry must be positive")


PRIMARY_MODEL_PIN = ModelPin(
    EXPECTED_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
)
SMOKE_MODEL_PIN = ModelPin(
    SMOKE_MODEL_ID,
    SMOKE_MODEL_REVISION,
    SMOKE_D_MODEL,
    SMOKE_N_LAYERS,
)


@dataclass(frozen=True, slots=True)
class LensArtifactPin:
    """One content-addressed lens file in a pinned Hub repository."""

    lens_type: Literal["J", "R"]
    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.lens_type not in {"J", "R"}:
            raise ValueError("lens_type must be J or R")
        if not self.repository or not self.filename:
            raise ValueError("lens repository and filename must be non-empty")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("lens revision must be a lowercase 40-character commit")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("lens SHA-256 must be lowercase hexadecimal")
        if self.size_bytes <= 0:
            raise ValueError("lens size must be positive")


PRIMARY_LENS_PINS = (
    LensArtifactPin(
        lens_type="J",
        repository=PRIMARY_LENS_REPOSITORY,
        revision=PRIMARY_LENS_REVISION,
        filename="qwen3.5-122b-a10b/j-lens/lens.pt",
        sha256="961bbb0e1e9362dda57812080fd3cb860bba891799ddbd4ebd89088540ed7842",
        size_bytes=887_106_103,
    ),
    LensArtifactPin(
        lens_type="R",
        repository=PRIMARY_LENS_REPOSITORY,
        revision=PRIMARY_LENS_REVISION,
        filename="qwen3.5-122b-a10b/r-lens/lens.pt",
        sha256="8093ae27ac100ce5ba18e6b4525c1cc394d1c32c03b0607b0cf3b5ebcd7bf2ee",
        size_bytes=887_106_295,
    ),
)


@dataclass(frozen=True, slots=True)
class PinnedModelRuntime:
    """A validated text-only HF runtime and its non-secret load provenance."""

    pin: ModelPin
    runtime: ModelRuntime
    model_class: str
    tokenizer_class: str
    device_map: Mapping[str, str | int]
    cuda_device_count: int


@dataclass(frozen=True, slots=True)
class VerifiedLensPair:
    """Matched local J/R handles after size, hash, and metadata validation."""

    j: LoadedLens
    r: LoadedLens
    pins: tuple[LensArtifactPin, LensArtifactPin]

    @property
    def handles(self) -> tuple[LoadedLens, LoadedLens]:
        return (self.j, self.r)


@dataclass(frozen=True, slots=True)
class CapturedActivations:
    """Position-selected residuals from exactly one model forward."""

    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    residuals: Mapping[int, Any]
    forward_count: int = 1


class SameForwardBackend(Protocol):
    """Small injectable surface used by CPU fakes and the actual GPU backend."""

    def capture_once(
        self,
        runtime: ModelRuntime,
        *,
        input_token_ids: Sequence[int],
        layers: Sequence[int],
        positions: Sequence[int],
    ) -> CapturedActivations: ...

    def transport_and_unembed(
        self,
        runtime: ModelRuntime,
        lens: LoadedLens,
        *,
        layer: int,
        residual: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class LensTraceInput:
    """Exact generation streams and pre-audited positions for one base trace."""

    trace_id: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    position_indices: Mapping[str, int]
    good_side_direction: Literal[-1, 1]
    prompt_token_ids_hash: str
    completion_token_ids_hash: str
    combined_token_stream_hash: str

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise ValueError("trace_id must be non-empty")
        for name in ("prompt_token_ids", "completion_token_ids"):
            ids = tuple(getattr(self, name))
            if not ids:
                raise ValueError(f"{name} must be non-empty")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids
            ):
                raise TypeError(f"{name} must contain non-negative integer IDs")
            object.__setattr__(self, name, ids)
        if self.good_side_direction not in {-1, 1}:
            raise ValueError("good_side_direction must be -1 or +1")
        _validate_named_positions(self.position_indices, len(self.sequence_token_ids))
        expected = token_stream_manifest(
            prompt_token_ids=self.prompt_token_ids,
            completion_token_ids=self.completion_token_ids,
        )
        for key in (
            "prompt_token_ids_hash",
            "completion_token_ids_hash",
            "combined_token_stream_hash",
        ):
            if getattr(self, key) != expected[key]:
                raise ProvenanceError(f"trace {self.trace_id}: {key} does not match token IDs")
        object.__setattr__(self, "position_indices", dict(self.position_indices))

    @property
    def sequence_token_ids(self) -> tuple[int, ...]:
        return (*self.prompt_token_ids, *self.completion_token_ids)

    @classmethod
    def from_token_stream_manifest(
        cls,
        *,
        trace_id: str,
        token_streams: Mapping[str, Any],
        position_indices: Mapping[str, int],
        good_side_direction: Literal[-1, 1],
    ) -> LensTraceInput:
        prompt, completion = validate_token_stream_manifest(token_streams, require_both=True)
        assert prompt is not None and completion is not None
        return cls(
            trace_id=trace_id,
            prompt_token_ids=prompt,
            completion_token_ids=completion,
            position_indices=dict(position_indices),
            good_side_direction=good_side_direction,
            prompt_token_ids_hash=str(token_streams["prompt_token_ids_hash"]),
            completion_token_ids_hash=str(token_streams["completion_token_ids_hash"]),
            combined_token_stream_hash=str(token_streams["combined_token_stream_hash"]),
        )


@dataclass(frozen=True, slots=True)
class CompatibilityAttempt(CanonicalRecord):
    ordinal: int
    stage: Literal["4b_smoke", "122b_preflight"]
    strategy: str
    model_id: str
    model_revision: str
    prefix_token_count: int
    prefix_token_ids_hash: str
    status: Literal["passed", "failed"]
    details: Mapping[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.ordinal <= 0 or self.prefix_token_count <= 0:
            raise ValueError("attempt ordinal and prefix length must be positive")
        if not (
            isinstance(self.prefix_token_ids_hash, str)
            and self.prefix_token_ids_hash.startswith("sha256:")
            and len(self.prefix_token_ids_hash) == len("sha256:") + 64
        ):
            raise ValueError("attempt prefix_token_ids_hash must be a namespaced SHA-256")
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True, slots=True)
class LensCompatibilityManifest(CanonicalRecord):
    """Outcome of the mandatory gate; it can never claim a 27B substitution."""

    attempts: tuple[CompatibilityAttempt, ...]
    primary_ready: bool
    transformers_revision: str = TRANSFORMERS_REVISION
    jlens_revision: str = JLENS_REVISION
    maximum_122b_attempts: int = 2
    fallback_model_used: bool = False
    fallback_policy: str = "27B_methodology_support_only_not_122B_substitute"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.fallback_model_used:
            raise ValueError("the primary compatibility manifest cannot substitute a fallback")
        primary_attempts = [item for item in self.attempts if item.stage == "122b_preflight"]
        if len(primary_attempts) > self.maximum_122b_attempts:
            raise ValueError("manifest exceeds the frozen two-attempt 122B limit")
        if self.primary_ready and not primary_attempts:
            raise ValueError("primary_ready requires a passed 122B preflight")
        if self.primary_ready and primary_attempts[-1].status != "passed":
            raise ValueError("primary_ready disagrees with the last 122B attempt")


def _validate_named_positions(position_indices: Mapping[str, int], sequence_length: int) -> None:
    observed = set(position_indices)
    expected = set(POSITION_ORDER)
    if observed != expected:
        raise PositionMappingError(
            f"named positions must match {POSITION_ORDER}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for name in POSITION_ORDER:
        value = position_indices[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < sequence_length
        ):
            raise PositionMappingError(
                f"position {name!r}={value!r} is outside sequence length {sequence_length}"
            )


def _as_ids(encoded: Any, *, word: str) -> tuple[int, ...]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, Sequence)
        and not isinstance(encoded, (str, bytes))
        and len(encoded) == 1
        and isinstance(encoded[0], Sequence)
        and not isinstance(encoded[0], (str, bytes))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise ConceptValidationError(f"tokenizer returned invalid IDs for {word!r}")
    try:
        return tuple(int(value) for value in encoded)
    except (TypeError, ValueError) as exc:
        raise ConceptValidationError(f"tokenizer returned non-integer IDs for {word!r}") from exc


def _encode_probe(tokenizer: Any, word: str) -> tuple[int, ...]:
    try:
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(word, add_special_tokens=False)
        else:
            encoded = tokenizer(word, add_special_tokens=False)
    except Exception as exc:
        raise ConceptValidationError(f"tokenizer could not encode {word!r}: {exc}") from exc
    return _as_ids(encoded, word=word)


def _decode_exact_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise ConceptValidationError("tokenizer must expose decode for prefix collision checks")
    try:
        decoded = decoder(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise ConceptValidationError(f"tokenizer could not decode exact input IDs: {exc}") from exc
    if not isinstance(decoded, str):
        raise ConceptValidationError("tokenizer.decode must return text")
    return decoded


def freeze_prefix_absent_probes(
    tokenizer: Any,
    *,
    exact_prefix_token_ids: Sequence[int],
    concept_words: Mapping[str, Mapping[str, Sequence[str]]] = DEFAULT_CONCEPT_WORDS,
    frozen_token_ids: Mapping[str, Mapping[str, Sequence[int]]] = FROZEN_PROBE_TOKEN_IDS,
) -> dict[str, ConceptTokenIds]:
    """Verify the preregistered IDs, then filter copied probes outcome-blindly.

    The model input is never retokenized.  Token collisions are checked against
    the exact persisted stream, while a decode-only lexical check also catches
    a copied word represented by a context-dependent token ID.
    """

    exact_ids = tuple(exact_prefix_token_ids)
    if not exact_ids:
        raise ConceptValidationError("exact prefix token stream must be non-empty")
    prefix_set = set(exact_ids)
    decoded_folded = _decode_exact_tokens(tokenizer, exact_ids).casefold()
    seen_ids: set[int] = set()
    result: dict[str, ConceptTokenIds] = {}

    if set(concept_words) != set(frozen_token_ids):
        raise ProvenanceError("concept names disagree with the frozen probe-ID manifest")
    for concept, polarities in concept_words.items():
        words_by_polarity: dict[str, tuple[str, ...]] = {}
        ids_by_polarity: dict[str, tuple[int, ...]] = {}
        for polarity in ("positive", "negative"):
            words = tuple(polarities.get(polarity, ()))
            declared = tuple(frozen_token_ids[concept].get(polarity, ()))
            if not words or len(words) != len(declared):
                raise ProvenanceError(
                    f"{concept}/{polarity} words disagree with frozen token-ID count"
                )
            retained: list[tuple[str, int]] = []
            for word, expected_id in zip(words, declared, strict=True):
                observed = _encode_probe(tokenizer, word)
                if len(observed) != 1:
                    raise ConceptValidationError(
                        f"probe {word!r} must be one token; observed {len(observed)}"
                    )
                token_id = observed[0]
                if token_id != expected_id:
                    raise ProvenanceError(
                        f"probe {word!r} token ID changed: expected {expected_id}, observed {token_id}"
                    )
                if token_id in seen_ids:
                    raise ConceptValidationError(f"frozen probe token ID {token_id} is reused")
                seen_ids.add(token_id)
                lexical = word.strip().casefold()
                copied_lexically = bool(lexical) and _contains_lexical_word(decoded_folded, lexical)
                if token_id not in prefix_set and not copied_lexically:
                    retained.append((word, token_id))
            if not retained:
                raise ConceptValidationError(
                    f"concept {concept!r} has no prefix-absent {polarity} probe"
                )
            words_by_polarity[polarity] = tuple(item[0] for item in retained)
            ids_by_polarity[polarity] = tuple(item[1] for item in retained)
        result[concept] = ConceptTokenIds(
            positive_ids=ids_by_polarity["positive"],
            negative_ids=ids_by_polarity["negative"],
            positive_words=words_by_polarity["positive"],
            negative_words=words_by_polarity["negative"],
        )
    return result


def _contains_lexical_word(text: str, word: str) -> bool:
    """Unicode-friendly word-boundary check without a regex dependency."""

    start = 0
    while True:
        start = text.find(word, start)
        if start < 0:
            return False
        end = start + len(word)
        before_word = start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_")
        after_word = end == len(text) or not (text[end].isalnum() or text[end] == "_")
        if before_word and after_word:
            return True
        start += 1


def causal_probe_collisions(
    tokenizer: Any,
    *,
    causal_token_ids: Sequence[int],
    probes: ConceptTokenIds,
) -> tuple[ProbeCollision, ...]:
    """Recompute exact-ID and lexical collisions from one causal prefix."""

    causal_ids = tuple(int(value) for value in causal_token_ids)
    prefix_set = set(causal_ids)
    decoded_folded = _decode_exact_tokens(tokenizer, causal_ids).casefold()
    collisions: list[ProbeCollision] = []
    for polarity, words, token_ids in (
        ("positive", probes.positive_words, probes.positive_ids),
        ("negative", probes.negative_words, probes.negative_ids),
    ):
        for word, token_id in zip(words, token_ids, strict=True):
            lexical = word.strip().casefold()
            exact_present = token_id in prefix_set
            lexical_present = bool(lexical) and _contains_lexical_word(
                decoded_folded, lexical
            )
            if exact_present or lexical_present:
                collisions.append(
                    ProbeCollision(
                        polarity=polarity,  # type: ignore[arg-type]
                        word=word,
                        token_id=token_id,
                        exact_token_id_present=exact_present,
                        lexical_word_present=lexical_present,
                    )
                )
    return tuple(collisions)


def _fixed_probe_universe(
    tokenizer: Any,
    *,
    concept_words: Mapping[str, Mapping[str, Sequence[str]]],
    frozen_token_ids: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, ConceptTokenIds]:
    """Verify the complete candidate universe without adapting it to any trace."""

    if set(concept_words) != set(frozen_token_ids):
        raise ProvenanceError("concept names disagree with the frozen probe-ID manifest")
    seen_ids: set[int] = set()
    concepts: dict[str, ConceptTokenIds] = {}
    for concept, polarities in concept_words.items():
        words_by_polarity: dict[str, tuple[str, ...]] = {}
        ids_by_polarity: dict[str, tuple[int, ...]] = {}
        for polarity in ("positive", "negative"):
            words = tuple(polarities.get(polarity, ()))
            declared = tuple(frozen_token_ids[concept].get(polarity, ()))
            if not words or len(words) != len(declared):
                raise ConceptValidationError(
                    f"concept {concept!r} has empty or mismatched {polarity} coverage"
                )
            verified: list[int] = []
            for word, expected_id in zip(words, declared, strict=True):
                observed = _encode_probe(tokenizer, word)
                if len(observed) != 1:
                    raise ConceptValidationError(
                        f"probe {word!r} must be one token; observed {len(observed)}"
                    )
                token_id = observed[0]
                if token_id != expected_id:
                    raise ProvenanceError(
                        f"probe {word!r} token ID changed: expected {expected_id}, observed {token_id}"
                    )
                if token_id in seen_ids:
                    raise ConceptValidationError(f"frozen probe token ID {token_id} is reused")
                seen_ids.add(token_id)
                verified.append(token_id)
            words_by_polarity[polarity] = words
            ids_by_polarity[polarity] = tuple(verified)
        concepts[concept] = ConceptTokenIds(
            positive_ids=ids_by_polarity["positive"],
            negative_ids=ids_by_polarity["negative"],
            positive_words=words_by_polarity["positive"],
            negative_words=words_by_polarity["negative"],
        )
    return concepts


def freeze_causal_probe_design(
    tokenizer: Any,
    *,
    traces: Sequence[LensTraceInput],
    candidate_probe_manifest_hash: str,
    candidate_probe_manifest_sha256: str,
    anchor_manifest_hash: str,
    anchor_selection_hash: str,
    rollout_manifest_hash: str,
    position_manifest_hash: str,
    concept_words: Mapping[str, Mapping[str, Sequence[str]]] = DEFAULT_CONCEPT_WORDS,
    frozen_token_ids: Mapping[str, Mapping[str, Sequence[int]]] = FROZEN_PROBE_TOKEN_IDS,
    model_id: str = EXPECTED_MODEL_ID,
    tokenizer_revision: str = PRIMARY_MODEL_REVISION,
) -> LensProbeDesign:
    """Freeze one common universe and a prefix-local collision mask.

    Eligibility for a cell sees exactly the activation's causal prefix.  A
    collision invalidates the entire concept cell; individual probes are never
    removed or reweighted.
    """

    if not traces:
        raise ValueError("at least one frozen trace is required for the probe design")
    if len({trace.trace_id for trace in traces}) != len(traces):
        raise ValueError("probe-design trace IDs must be unique")
    concepts = _fixed_probe_universe(
        tokenizer,
        concept_words=concept_words,
        frozen_token_ids=frozen_token_ids,
    )
    cells: list[ProbeCellEligibility] = []
    for trace in traces:
        sequence_ids = trace.sequence_token_ids
        for position_name in POSITION_ORDER:
            token_index = trace.position_indices[position_name]
            causal_ids = sequence_ids[: token_index + 1]
            causal_hash = token_stream_hash(causal_ids, stream="lens_causal_prefix")
            for concept, probes in concepts.items():
                collisions = causal_probe_collisions(
                    tokenizer,
                    causal_token_ids=causal_ids,
                    probes=probes,
                )
                collision_hash = None
                if collisions:
                    collision_hash = stable_hash([item.to_dict() for item in collisions])
                cells.append(
                    ProbeCellEligibility(
                        trace_id=trace.trace_id,
                        position_name=position_name,
                        concept=concept,
                        token_index=token_index,
                        causal_prefix_token_count=len(causal_ids),
                        causal_prefix_token_ids_hash=causal_hash,
                        probe_eligible=not collisions,
                        probe_ineligibility_reason=(
                            None if not collisions else "causal_prefix_probe_collision"
                        ),
                        collisions=collisions,
                        collision_evidence_hash=collision_hash,
                    )
                )
    return LensProbeDesign(
        model_id=model_id,
        tokenizer_id=model_id,
        tokenizer_revision=tokenizer_revision,
        candidate_probe_manifest_hash=candidate_probe_manifest_hash,
        candidate_probe_manifest_sha256=candidate_probe_manifest_sha256,
        anchor_manifest_hash=anchor_manifest_hash,
        anchor_selection_hash=anchor_selection_hash,
        rollout_manifest_hash=rollout_manifest_hash,
        position_manifest_hash=position_manifest_hash,
        concepts=concepts,
        cells=tuple(cells),
    )


def _model_geometry(hf_model: Any) -> tuple[int | None, int | None]:
    config = getattr(hf_model, "config", None)
    if config is None:
        return None, None
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else config
    return (
        getattr(text_config, "hidden_size", None),
        getattr(text_config, "num_hidden_layers", None),
    )


def installed_vcs_revision(package: str) -> str | None:
    """Read a PEP-610 VCS commit without importing the target package."""

    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs = payload.get("vcs_info")
    if isinstance(vcs, Mapping) and isinstance(vcs.get("commit_id"), str):
        return str(vcs["commit_id"])
    archive = payload.get("archive_info")
    if isinstance(archive, Mapping) and isinstance(archive.get("hash"), str):
        return str(archive["hash"]).removeprefix("sha256=")
    return None


def verify_software_revisions(
    *,
    revision_reader: Callable[[str], str | None] = installed_vcs_revision,
) -> dict[str, str]:
    """Require the two preregistered source commits before GPU execution."""

    expected = {"transformers": TRANSFORMERS_REVISION, "jlens": JLENS_REVISION}
    observed: dict[str, str] = {}
    for package, revision in expected.items():
        value = revision_reader(package)
        if value != revision:
            raise ProvenanceError(
                f"{package} must be installed from {revision}; observed {value!r}"
            )
        observed[package] = value
    return observed


def _optional_module(module: ModuleType | Any | None, name: str) -> Any:
    if module is not None:
        return module
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise OptionalDependencyError(f"{name} is required for the GPU lens job") from exc


def _location_is_gpu(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value >= 0
    if isinstance(value, str):
        folded = value.casefold()
        return folded.startswith("cuda") or folded.isdigit()
    return False


def _assert_no_offload(hf_model: Any) -> dict[str, str | int]:
    raw_map = getattr(hf_model, "hf_device_map", None)
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ProvenanceError("primary model must expose a non-empty hf_device_map")
    device_map = {str(name): value for name, value in raw_map.items()}
    bad = {name: value for name, value in device_map.items() if not _location_is_gpu(value)}
    if bad:
        raise ProvenanceError(f"CPU/disk/meta offload is forbidden; observed device map {bad}")

    for iterator_name in ("named_parameters", "named_buffers"):
        iterator = getattr(hf_model, iterator_name, None)
        if not callable(iterator):
            continue
        for name, value in iterator():
            device = str(getattr(value, "device", "unknown")).casefold()
            if not device.startswith("cuda"):
                raise ProvenanceError(
                    f"CPU/disk/meta offload is forbidden; {iterator_name} {name!r} is on {device}"
                )
    return device_map


def load_pinned_text_runtime(
    pin: ModelPin,
    *,
    required_cuda_devices: int,
    per_gpu_memory_gib: int,
    transformers_module: ModuleType | Any | None = None,
    torch_module: ModuleType | Any | None = None,
    jlens_module: ModuleType | Any | None = None,
    verify_dependencies: bool = True,
    revision_reader: Callable[[str], str | None] = installed_vcs_revision,
) -> PinnedModelRuntime:
    """Load only the Qwen text CausalLM at a pinned revision in BF16.

    ``AutoModelForCausalLM`` activates Transformers' Qwen3.5 VLM-compatibility
    route, which swaps the outer multimodal config for its text config and does
    not instantiate the vision tower.  The returned class and device placement
    are checked rather than inferred from the requested auto class.
    """

    if required_cuda_devices <= 0 or per_gpu_memory_gib <= 0:
        raise ValueError("CUDA device and memory requirements must be positive")
    if verify_dependencies:
        verify_software_revisions(revision_reader=revision_reader)
    transformers = _optional_module(transformers_module, "transformers")
    torch = _optional_module(torch_module, "torch")
    jlens = _optional_module(jlens_module, "jlens")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise LensExecutionError("CUDA is required for model/lens execution")
    cuda_count = int(cuda.device_count())
    if cuda_count < required_cuda_devices:
        raise LensExecutionError(
            f"need at least {required_cuda_devices} CUDA devices; observed {cuda_count}"
        )

    max_memory = {index: f"{per_gpu_memory_gib}GiB" for index in range(cuda_count)}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pin.model_id,
        revision=pin.revision,
        trust_remote_code=False,
        use_fast=True,
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        pin.model_id,
        revision=pin.revision,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        offload_state_dict=False,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model_class = type(hf_model).__name__
    if not model_class.endswith("ForCausalLM"):
        raise ProvenanceError(
            f"text-only AutoModelForCausalLM path was not used; observed {model_class}"
        )
    if hasattr(hf_model, "visual") or hasattr(getattr(hf_model, "model", None), "visual"):
        raise ProvenanceError("loaded runtime unexpectedly contains a vision tower")
    observed_d_model, observed_n_layers = _model_geometry(hf_model)
    if (observed_d_model, observed_n_layers) != (pin.d_model, pin.n_layers):
        raise ProvenanceError(
            "loaded text geometry disagrees with pin: "
            f"expected {(pin.d_model, pin.n_layers)}, "
            f"observed {(observed_d_model, observed_n_layers)}"
        )
    device_map = _assert_no_offload(hf_model)
    try:
        wrapped = jlens.from_hf(hf_model, tokenizer, compile=False)
    except Exception as exc:
        raise LensExecutionError(f"pinned jlens.from_hf compatibility failed: {exc}") from exc
    if (getattr(wrapped, "d_model", None), getattr(wrapped, "n_layers", None)) != (
        pin.d_model,
        pin.n_layers,
    ):
        raise ProvenanceError("wrapped jlens runtime reports incompatible geometry")
    runtime = ModelRuntime(
        model_id=pin.model_id,
        model=wrapped,
        tokenizer=tokenizer,
        device_map=device_map,
        compile=False,
    )
    return PinnedModelRuntime(
        pin=pin,
        runtime=runtime,
        model_class=model_class,
        tokenizer_class=type(tokenizer).__name__,
        device_map=device_map,
        cuda_device_count=cuda_count,
    )


def download_and_load_lens_pair(
    pins: Sequence[LensArtifactPin],
    *,
    cache_dir: str | Path,
    downloader: Callable[..., str] | None = None,
    lens_loader: Callable[..., LoadedLens] = load_local_lens,
    checkpoint_reader: Callable[[Path], Mapping[str, Any]] | None = None,
    jlens_module: ModuleType | Any | None = None,
) -> VerifiedLensPair:
    """Download only pinned files, verify size/hash/metadata, and load J/R."""

    by_type = {pin.lens_type: pin for pin in pins}
    if len(pins) != 2 or set(by_type) != {"J", "R"}:
        raise ProvenanceError("exactly one J and one R artifact pin are required")
    j_pin, r_pin = by_type["J"], by_type["R"]
    if (j_pin.repository, j_pin.revision) != (r_pin.repository, r_pin.revision):
        raise ProvenanceError("J and R artifacts must share one pinned repository revision")
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise OptionalDependencyError("huggingface_hub is required to download lenses") from exc
        downloader = hf_hub_download

    loaded: dict[str, LoadedLens] = {}
    for lens_type in ("J", "R"):
        pin = by_type[lens_type]
        raw_path = downloader(
            repo_id=pin.repository,
            filename=pin.filename,
            revision=pin.revision,
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
        )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ProvenanceError(f"downloaded {lens_type}-lens is not a file: {path}")
        observed_size = path.stat().st_size
        if observed_size != pin.size_bytes:
            raise ProvenanceError(
                f"{lens_type}-lens size mismatch: expected {pin.size_bytes}, observed {observed_size}"
            )
        handle = lens_loader(
            path,
            lens_type=lens_type,
            expected_sha256=pin.sha256,
            checkpoint_reader=checkpoint_reader,
            jlens_module=jlens_module,
        )
        if handle.lens_type != lens_type or handle.provenance.file_sha256 != pin.sha256:
            raise ProvenanceError(f"loaded {lens_type}-lens disagrees with its artifact pin")
        loaded[lens_type] = handle

    j_handle, r_handle = loaded["J"], loaded["R"]
    for handle in (j_handle, r_handle):
        provenance = handle.provenance
        if (
            provenance.model_id != EXPECTED_MODEL_ID
            or provenance.d_model != EXPECTED_D_MODEL
            or provenance.target_layer != EXPECTED_TARGET_LAYER
            or provenance.source_layers != ARTIFACT_SOURCE_LAYERS
        ):
            raise ProvenanceError(f"{handle.lens_type}-lens geometry is incompatible")
    return VerifiedLensPair(j_handle, r_handle, (j_pin, r_pin))


class JlensTorchSameForwardBackend:
    """Pinned jlens/torch backend that captures once and transports twice."""

    def __init__(
        self,
        *,
        torch_module: ModuleType | Any | None = None,
        activation_recorder_cls: type[Any] | None = None,
    ) -> None:
        self._torch = _optional_module(torch_module, "torch")
        if activation_recorder_cls is None:
            try:
                hooks = importlib.import_module("jlens.hooks")
            except ImportError as exc:
                raise OptionalDependencyError("pinned jlens.hooks is required") from exc
            activation_recorder_cls = hooks.ActivationRecorder
        self._recorder_cls = activation_recorder_cls

    def capture_once(
        self,
        runtime: ModelRuntime,
        *,
        input_token_ids: Sequence[int],
        layers: Sequence[int],
        positions: Sequence[int],
    ) -> CapturedActivations:
        ids = tuple(input_token_ids)
        normalized_layers = tuple(layers)
        normalized_positions = tuple(positions)
        if not ids or not normalized_layers or not normalized_positions:
            raise LensExecutionError("capture requires tokens, layers, and positions")
        if max(normalized_positions) >= len(ids) or min(normalized_positions) < 0:
            raise PositionMappingError("capture position is outside the exact input stream")
        input_ids = self._torch.tensor(
            [list(ids)], dtype=self._torch.long, device=runtime.model.input_device
        )
        try:
            with self._torch.inference_mode():
                with self._recorder_cls(runtime.model.layers, at=normalized_layers) as recorder:
                    runtime.model.forward(input_ids)
                    selected: dict[int, Any] = {}
                    for layer in normalized_layers:
                        full = recorder.activations[layer][0]
                        index = self._torch.tensor(
                            list(normalized_positions),
                            dtype=self._torch.long,
                            device=full.device,
                        )
                        selected[layer] = full.index_select(0, index).detach().float()
        except Exception as exc:
            raise LensExecutionError(f"single-forward activation capture failed: {exc}") from exc
        return CapturedActivations(ids, normalized_positions, selected, forward_count=1)

    def transport_and_unembed(
        self,
        runtime: ModelRuntime,
        lens: LoadedLens,
        *,
        layer: int,
        residual: Any,
    ) -> Any:
        try:
            transported = lens.lens.transport(residual, layer)
            return runtime.model.unembed(transported).float().cpu()
        except Exception as exc:
            raise LensExecutionError(
                f"{lens.lens_type}-lens transport/unembed failed at layer {layer}: {exc}"
            ) from exc


def _validate_primary_runtime(runtime: ModelRuntime) -> None:
    if runtime.model_id != EXPECTED_MODEL_ID:
        raise ProvenanceError(
            f"primary runtime must be {EXPECTED_MODEL_ID}; observed {runtime.model_id}"
        )
    if (getattr(runtime.model, "d_model", None), getattr(runtime.model, "n_layers", None)) != (
        EXPECTED_D_MODEL,
        EXPECTED_N_LAYERS,
    ):
        raise ProvenanceError("primary runtime geometry is incompatible")
    if runtime.compile:
        raise ProvenanceError("primary sharded runtime must use compile=False")
    if isinstance(runtime.device_map, Mapping):
        bad = {
            name: value for name, value in runtime.device_map.items() if not _location_is_gpu(value)
        }
        if bad:
            raise ProvenanceError(f"primary runtime contains forbidden offload devices: {bad}")
    else:
        raise ProvenanceError("primary runtime must retain its explicit GPU device map")


def _validated_lens_handles(lenses: Sequence[LoadedLens]) -> dict[str, LoadedLens]:
    by_type: dict[str, LoadedLens] = {}
    for handle in lenses:
        if handle.lens_type in by_type:
            raise ProvenanceError(f"duplicate {handle.lens_type}-lens")
        provenance = handle.provenance
        if (
            provenance.model_id != EXPECTED_MODEL_ID
            or provenance.d_model != EXPECTED_D_MODEL
            or provenance.target_layer != EXPECTED_TARGET_LAYER
            or provenance.source_layers != ARTIFACT_SOURCE_LAYERS
        ):
            raise ProvenanceError(f"{handle.lens_type}-lens provenance is incompatible")
        by_type[handle.lens_type] = handle
    if set(by_type) != {"J", "R"}:
        raise ProvenanceError("same-forward analysis requires exactly one J and one R lens")
    return by_type


def _token_prefix_sha256(token_ids: Sequence[int]) -> str:
    canonical = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def analyze_trace_same_forward(
    trace: LensTraceInput,
    *,
    runtime: ModelRuntime,
    lenses: Sequence[LoadedLens],
    backend: SameForwardBackend,
    probe_design: LensProbeDesign,
    layers: Sequence[int] = FITTED_LAYERS,
) -> list[LensRecord]:
    """Read both lenses from one exact-token forward and emit long-form rows."""

    _validate_primary_runtime(runtime)
    by_type = _validated_lens_handles(lenses)
    normalized_layers = tuple(layers)
    if (
        not normalized_layers
        or tuple(sorted(set(normalized_layers))) != normalized_layers
        or not set(normalized_layers).issubset(FITTED_LAYERS)
    ):
        raise LensExecutionError("layers must be unique, sorted, and within fitted layers 4..46")
    if probe_design.model_id != runtime.model_id:
        raise ProvenanceError("probe design model disagrees with the primary runtime")
    full_ids = trace.sequence_token_ids
    forward_end = max(trace.position_indices.values()) + 1
    exact_ids = full_ids[:forward_end]
    if len(exact_ids) > PRIMARY_MAX_SEQUENCE_TOKENS:
        raise LensExecutionError(
            f"trace has {len(exact_ids)} tokens; primary cap is {PRIMARY_MAX_SEQUENCE_TOKENS}"
        )
    _validate_named_positions(trace.position_indices, len(exact_ids))
    concepts = probe_design.concepts
    ordered_positions = tuple(trace.position_indices[name] for name in POSITION_ORDER)
    captured = backend.capture_once(
        runtime,
        input_token_ids=exact_ids,
        layers=normalized_layers,
        positions=ordered_positions,
    )
    if captured.forward_count != 1:
        raise LensExecutionError(
            f"same-forward invariant violated: observed {captured.forward_count} forwards"
        )
    if captured.input_token_ids != exact_ids or captured.positions != ordered_positions:
        raise ProvenanceError("activation capture disagrees with exact token/position manifest")
    if set(captured.residuals) != set(normalized_layers):
        raise LensExecutionError("activation capture omitted or added requested layers")

    records: list[LensRecord] = []
    prefix_digest = _token_prefix_sha256(exact_ids)
    forward_hash = token_stream_hash(exact_ids, stream="lens_forward_input")
    probe_design_hash = probe_design.manifest_hash
    eligibility_by_cell = {
        (cell.position_name, cell.concept): cell
        for cell in probe_design.cells
        if cell.trace_id == trace.trace_id
    }
    if len(eligibility_by_cell) != len(POSITION_ORDER) * len(concepts):
        raise ProvenanceError("probe design lacks the complete trace cell inventory")
    for lens_type in ("J", "R"):
        handle = by_type[lens_type]
        for layer in normalized_layers:
            logits_rows = backend.transport_and_unembed(
                runtime,
                handle,
                layer=layer,
                residual=captured.residuals[layer],
            )
            for offset, position_name in enumerate(POSITION_ORDER):
                try:
                    logits = logits_rows[offset]
                except (IndexError, KeyError, TypeError) as exc:
                    raise LensExecutionError(
                        f"{lens_type}-lens layer {layer} omitted position row {offset}"
                    ) from exc
                for contrast, token_ids in concepts.items():
                    eligibility = eligibility_by_cell[(position_name, contrast)]
                    expected_prefix = full_ids[: trace.position_indices[position_name] + 1]
                    expected_prefix_hash = token_stream_hash(
                        expected_prefix, stream="lens_causal_prefix"
                    )
                    if (
                        eligibility.token_index != trace.position_indices[position_name]
                        or eligibility.causal_prefix_token_count != len(expected_prefix)
                        or eligibility.causal_prefix_token_ids_hash != expected_prefix_hash
                    ):
                        raise ProvenanceError(
                            "probe eligibility disagrees with the authenticated causal prefix"
                        )
                    raw_value: float | None = None
                    signed_value: float | None = None
                    if eligibility.probe_eligible:
                        raw_value = signed_mean_logit_contrasts(
                            logits, {contrast: token_ids}
                        )[contrast]
                        if not math.isfinite(raw_value):
                            raise LensExecutionError("lens contrast is non-finite")
                        signed_value = (
                            raw_value * trace.good_side_direction
                            if contrast == "direction"
                            else raw_value
                        )
                    records.append(
                        LensRecord(
                            trace_id=trace.trace_id,
                            prefix_sha256=prefix_digest,
                            model_id=runtime.model_id,
                            lens_type=lens_type,  # type: ignore[arg-type]
                            lens_file_sha256=handle.provenance.file_sha256,
                            target_layer=handle.provenance.target_layer,
                            layer=layer,
                            layer_band=layer_band(layer),
                            position_name=position_name,
                            token_index=trace.position_indices[position_name],
                            contrast=contrast,
                            raw_mean_logit_contrast=raw_value,
                            signed_mean_logit_contrast=signed_value,
                            good_side_direction=trace.good_side_direction,
                            positive_token_ids=token_ids.positive_ids,
                            negative_token_ids=token_ids.negative_ids,
                            probe_design_hash=probe_design_hash,
                            probe_eligibility_record_hash=eligibility.record_hash,
                            probe_eligible=eligibility.probe_eligible,
                            probe_ineligibility_reason=eligibility.probe_ineligibility_reason,
                            collision_evidence_hash=eligibility.collision_evidence_hash,
                            causal_prefix_token_ids_hash=expected_prefix_hash,
                            causal_prefix_token_count=len(expected_prefix),
                            forward_input_token_ids_hash=forward_hash,
                            forward_input_token_count=len(exact_ids),
                        )
                    )
    return records


def canonical_lens_record(record: LensRecord) -> dict[str, Any]:
    """Serialize a ``LensRecord`` with an explicit schema and content hash."""

    payload = asdict(record)
    payload["schema_version"] = 2
    payload["record_hash"] = stable_hash(payload)
    return payload


def execute_lens_traces(
    traces: Sequence[LensTraceInput],
    *,
    runtime: ModelRuntime,
    lenses: Sequence[LoadedLens],
    backend: SameForwardBackend,
    probe_design: LensProbeDesign,
    output_path: str | Path | None = None,
    layers: Sequence[int] = FITTED_LAYERS,
) -> list[LensRecord]:
    """Execute all frozen traces in order and optionally atomically write JSONL."""

    if not traces:
        raise ValueError("at least one lens trace is required")
    if len({trace.trace_id for trace in traces}) != len(traces):
        raise ValueError("lens trace IDs must be unique")
    records: list[LensRecord] = []
    for trace in traces:
        records.extend(
            analyze_trace_same_forward(
                trace,
                runtime=runtime,
                lenses=lenses,
                backend=backend,
                probe_design=probe_design,
                layers=layers,
            )
        )
    if output_path is not None:
        write_jsonl(output_path, (canonical_lens_record(record) for record in records))
    return records


def run_4b_compatibility_smoke(
    runtime: ModelRuntime,
    *,
    token_ids: Sequence[int],
    backend: SameForwardBackend,
) -> Mapping[str, Any]:
    """One cheap forward that validates Qwen3.5 text layout before 122B load."""

    if runtime.model_id != SMOKE_MODEL_ID:
        raise ProvenanceError(f"4B smoke requires {SMOKE_MODEL_ID}")
    if (getattr(runtime.model, "d_model", None), getattr(runtime.model, "n_layers", None)) != (
        SMOKE_D_MODEL,
        SMOKE_N_LAYERS,
    ):
        raise ProvenanceError("4B smoke runtime geometry is incompatible")
    ids = tuple(token_ids)
    if not ids or len(ids) > 256:
        raise LensExecutionError("4B smoke prefix must contain 1..256 exact tokens")
    layers = (0, SMOKE_N_LAYERS - 1)
    captured = backend.capture_once(
        runtime,
        input_token_ids=ids,
        layers=layers,
        positions=(len(ids) - 1,),
    )
    if captured.forward_count != 1 or set(captured.residuals) != set(layers):
        raise LensExecutionError("4B smoke did not capture both boundary layers once")
    return {
        "captured_layers": list(layers),
        "d_model": SMOKE_D_MODEL,
        "forward_count": captured.forward_count,
        "n_layers": SMOKE_N_LAYERS,
        "token_stream_hash": token_stream_hash(ids, stream="4b_compatibility_prefix"),
    }


def run_122b_preflight(
    runtime: ModelRuntime,
    lenses: Sequence[LoadedLens],
    *,
    token_ids: Sequence[int],
    backend: SameForwardBackend,
    probe_design: LensProbeDesign | None = None,
) -> Mapping[str, Any]:
    """One exact-prefix forward followed by both transports at two layers."""

    _validate_primary_runtime(runtime)
    by_type = _validated_lens_handles(lenses)
    ids = tuple(token_ids)
    if not ids or len(ids) > PRIMARY_MAX_SEQUENCE_TOKENS:
        raise LensExecutionError(
            f"122B preflight prefix must contain 1..{PRIMARY_MAX_SEQUENCE_TOKENS} tokens"
        )
    layers = (FITTED_LAYERS[0], FITTED_LAYERS[-1])
    concepts = (
        probe_design.concepts
        if probe_design is not None
        else freeze_prefix_absent_probes(
            runtime.tokenizer,
            exact_prefix_token_ids=ids,
        )
    )
    captured = backend.capture_once(
        runtime,
        input_token_ids=ids,
        layers=layers,
        positions=(len(ids) - 1,),
    )
    if captured.forward_count != 1:
        raise LensExecutionError("122B preflight violated the single-forward invariant")
    for lens_type in ("J", "R"):
        for layer in layers:
            rows = backend.transport_and_unembed(
                runtime,
                by_type[lens_type],
                layer=layer,
                residual=captured.residuals[layer],
            )
            try:
                row = rows[0]
                contrasts = signed_mean_logit_contrasts(row, concepts)
                if any(not math.isfinite(value) for value in contrasts.values()):
                    raise ValueError("non-finite probe contrast")
            except Exception as exc:
                raise LensExecutionError(
                    f"{lens_type}-lens preflight readout failed at layer {layer}: {exc}"
                ) from exc
    return {
        "captured_layers": list(layers),
        "d_model": EXPECTED_D_MODEL,
        "forward_count": captured.forward_count,
        "lens_file_sha256": {
            lens_type: by_type[lens_type].provenance.file_sha256 for lens_type in ("J", "R")
        },
        "lens_n_prompts": {
            lens_type: by_type[lens_type].provenance.metadata.get("n_prompts")
            for lens_type in ("J", "R")
        },
        "lens_types": ["J", "R"],
        "n_layers": EXPECTED_N_LAYERS,
        "token_stream_hash": token_stream_hash(ids, stream="122b_preflight_prefix"),
    }


def run_ordered_compatibility_gate(
    *,
    four_b_prefix_token_ids: Sequence[int],
    primary_prefix_token_ids: Sequence[int],
    shortened_primary_prefix_token_ids: Sequence[int],
    four_b_runner: Callable[[Sequence[int]], Mapping[str, Any]],
    primary_runner: Callable[[Sequence[int]], Mapping[str, Any]],
) -> LensCompatibilityManifest:
    """Run 4B first, then at most two limited 122B attempts.

    The second primary attempt changes only prefix length.  It is invalid for
    callers to pass different software/model revisions through ``primary_runner``;
    those identities are fixed in the resulting manifest and must be verified
    by :func:`verify_software_revisions` and :func:`load_pinned_text_runtime`.
    No fallback model is invoked here.
    """

    full_ids = tuple(primary_prefix_token_ids)
    short_ids = tuple(shortened_primary_prefix_token_ids)
    smoke_ids = tuple(four_b_prefix_token_ids)
    if not smoke_ids or not full_ids or not short_ids:
        raise ValueError("all compatibility prefixes must be non-empty")
    if len(short_ids) >= len(full_ids):
        raise ValueError("the second 122B prefix must be strictly shorter")
    if full_ids[: len(short_ids)] != short_ids:
        raise ValueError("the shortened 122B input must be an exact prefix of the full input")
    attempts: list[CompatibilityAttempt] = []

    try:
        details = dict(four_b_runner(smoke_ids))
    except Exception as exc:
        attempts.append(
            CompatibilityAttempt(
                ordinal=1,
                stage="4b_smoke",
                strategy="pinned_text_only_single_forward",
                model_id=SMOKE_MODEL_ID,
                model_revision=SMOKE_MODEL_REVISION,
                prefix_token_count=len(smoke_ids),
                prefix_token_ids_hash=token_stream_hash(
                    smoke_ids, stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
                ),
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )
        manifest = LensCompatibilityManifest(tuple(attempts), primary_ready=False)
        raise CompatibilityGateError("4B compatibility smoke failed", manifest) from exc
    attempts.append(
        CompatibilityAttempt(
            ordinal=1,
            stage="4b_smoke",
            strategy="pinned_text_only_single_forward",
            model_id=SMOKE_MODEL_ID,
            model_revision=SMOKE_MODEL_REVISION,
            prefix_token_count=len(smoke_ids),
            prefix_token_ids_hash=token_stream_hash(
                smoke_ids, stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
            ),
            status="passed",
            details=details,
        )
    )

    strategies = (
        ("version_fixed_full_prefix", full_ids),
        ("version_fixed_shortened_prefix", short_ids),
    )
    for primary_ordinal, (strategy, ids) in enumerate(strategies, start=1):
        try:
            details = dict(primary_runner(ids))
        except Exception as exc:
            attempts.append(
                CompatibilityAttempt(
                    ordinal=primary_ordinal,
                    stage="122b_preflight",
                    strategy=strategy,
                    model_id=EXPECTED_MODEL_ID,
                    model_revision=PRIMARY_MODEL_REVISION,
                    prefix_token_count=len(ids),
                    prefix_token_ids_hash=token_stream_hash(
                        ids, stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
                    ),
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        attempts.append(
            CompatibilityAttempt(
                ordinal=primary_ordinal,
                stage="122b_preflight",
                strategy=strategy,
                model_id=EXPECTED_MODEL_ID,
                model_revision=PRIMARY_MODEL_REVISION,
                prefix_token_count=len(ids),
                prefix_token_ids_hash=token_stream_hash(
                    ids, stream=COMPATIBILITY_ATTEMPT_PREFIX_STREAM
                ),
                status="passed",
                details=details,
            )
        )
        return LensCompatibilityManifest(tuple(attempts), primary_ready=True)

    manifest = LensCompatibilityManifest(tuple(attempts), primary_ready=False)
    raise PrimaryCompatibilityFailure(
        "both limited 122B compatibility attempts failed; 27B was not substituted",
        manifest,
    )


def write_compatibility_manifest(
    path: str | Path,
    manifest: LensCompatibilityManifest,
) -> Path:
    """Atomically write the compatibility manifest including its record hash."""

    return write_json(path, manifest.to_dict(include_hash=True))


__all__ = [
    "COMPATIBILITY_ATTEMPT_PREFIX_STREAM",
    "FROZEN_PROBE_TOKEN_IDS",
    "JLENS_REVISION",
    "POSITION_ORDER",
    "PRIMARY_LENS_PINS",
    "PRIMARY_LENS_REPOSITORY",
    "PRIMARY_LENS_REVISION",
    "PRIMARY_MAX_SEQUENCE_TOKENS",
    "PRIMARY_MODEL_PIN",
    "PRIMARY_MODEL_REVISION",
    "SMOKE_MODEL_ID",
    "SMOKE_MODEL_PIN",
    "SMOKE_MODEL_REVISION",
    "TRANSFORMERS_REVISION",
    "CapturedActivations",
    "CompatibilityAttempt",
    "CompatibilityGateError",
    "JlensTorchSameForwardBackend",
    "LensArtifactPin",
    "LensCompatibilityManifest",
    "LensProbeDesign",
    "LensTraceInput",
    "ModelPin",
    "PinnedModelRuntime",
    "PrimaryCompatibilityFailure",
    "ProbeCellEligibility",
    "ProbeCollision",
    "SameForwardBackend",
    "VerifiedLensPair",
    "analyze_trace_same_forward",
    "canonical_lens_record",
    "causal_probe_collisions",
    "download_and_load_lens_pair",
    "execute_lens_traces",
    "freeze_causal_probe_design",
    "freeze_prefix_absent_probes",
    "installed_vcs_revision",
    "load_pinned_text_runtime",
    "run_4b_compatibility_smoke",
    "run_122b_preflight",
    "run_ordered_compatibility_gate",
    "verify_software_revisions",
    "write_compatibility_manifest",
]
