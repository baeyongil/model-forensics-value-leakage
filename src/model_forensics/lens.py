"""Validated, optional wrappers for observational J/R-lens readouts.

The module deliberately has no import-time dependency on ``torch``,
``transformers``, or the official ``jlens`` package.  GPU callers inject model
loading, while local artifact loading is hash- and provenance-gated before the
official lens deserializer is called.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

EXPECTED_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"
EXPECTED_D_MODEL = 3072
EXPECTED_TARGET_LAYER = 46
EXPECTED_N_LAYERS = 48
ARTIFACT_SOURCE_LAYERS = tuple(range(0, EXPECTED_TARGET_LAYER + 1))
FITTED_LAYERS = tuple(range(4, EXPECTED_TARGET_LAYER + 1))
LAYER_TERTILES: Mapping[str, tuple[int, ...]] = {
    "early": tuple(range(4, 19)),
    "middle": tuple(range(19, 33)),
    "late": tuple(range(33, 47)),
}

# Leading spaces make the intended word-boundary tokenization explicit for a
# decoder tokenizer.  Every runtime must still validate these against the
# exact pinned tokenizer; a model revision can change tokenization.
DEFAULT_CONCEPT_WORDS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "direction": {
        "positive": (" upward", " increase", " inflate"),
        "negative": (" downward", " decrease", " deflate"),
    },
    "valence": {
        "positive": (" beneficial", " helpful", " charity"),
        "negative": (" harmful", " damaging", " penalty"),
    },
    "epistemic": {
        "positive": (" objective", " impartial", " truthful"),
        "negative": (" biased", " deceptive", " skewed"),
    },
}


class LensError(RuntimeError):
    """Base class for lens-pipeline failures with actionable messages."""


class OptionalDependencyError(LensError):
    """Raised when an optional GPU/lens dependency was not supplied or installed."""


class ProvenanceError(LensError):
    """Raised when a model or lens artifact is incompatible with the preregistration."""


class ConceptValidationError(LensError):
    """Raised when a probe word is multi-token, ambiguous, or copied from its prefix."""


class PositionMappingError(LensError):
    """Raised when preregistered token positions cannot be mapped unambiguously."""


class LensExecutionError(LensError):
    """Raised when a validated lens readout cannot be computed safely."""


@dataclass(frozen=True)
class LensProvenance:
    """Validated identity and geometry of one immutable local lens artifact."""

    model_id: str
    d_model: int
    target_layer: int
    source_layers: tuple[int, ...]
    file_sha256: str
    artifact_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedLens:
    """An official ``jlens.JacobianLens`` paired with validated provenance."""

    lens_type: Literal["J", "R"]
    lens: Any
    provenance: LensProvenance


@dataclass(frozen=True)
class ModelRuntime:
    """A preloaded HF model wrapped by official ``jlens.from_hf``."""

    model_id: str
    model: Any
    tokenizer: Any
    device_map: str | Mapping[str, Any]
    compile: bool


@dataclass(frozen=True)
class LensRecord:
    """One observational concept contrast at one lens/layer/token position."""

    trace_id: str
    prefix_sha256: str
    model_id: str
    lens_type: Literal["J", "R"]
    lens_file_sha256: str
    target_layer: int
    layer: int
    layer_band: Literal["early", "middle", "late"]
    position_name: str
    token_index: int
    contrast: str
    raw_mean_logit_contrast: float | None
    signed_mean_logit_contrast: float | None
    good_side_direction: Literal[-1, 1]
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]
    probe_design_hash: str | None = None
    probe_eligibility_record_hash: str | None = None
    probe_eligible: bool = True
    probe_ineligibility_reason: str | None = None
    collision_evidence_hash: str | None = None
    causal_prefix_token_ids_hash: str | None = None
    causal_prefix_token_count: int | None = None
    forward_input_token_ids_hash: str | None = None
    forward_input_token_count: int | None = None
    evidence_scope: str = field(default="observational_readout", init=False)
    causal_claim: bool = field(default=False, init=False)


@dataclass(frozen=True)
class ConceptTokenIds:
    """Validated positive/negative token IDs for one preregistered construct."""

    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]
    positive_words: tuple[str, ...]
    negative_words: tuple[str, ...]


def fitted_layer_tertiles() -> dict[str, tuple[int, ...]]:
    """Return the fixed, data-independent tertiles over fitted layers 4..46."""

    return {name: tuple(layers) for name, layers in LAYER_TERTILES.items()}


def layer_band(layer: int) -> Literal["early", "middle", "late"]:
    """Return the preregistered tertile for one fitted source layer."""

    for name, layers in LAYER_TERTILES.items():
        if layer in layers:
            return name  # type: ignore[return-value]
    raise PositionMappingError(
        f"layer {layer!r} is not a fitted source layer in {FITTED_LAYERS[0]}..{FITTED_LAYERS[-1]}"
    )


def map_named_positions(
    *,
    sequence_length: int,
    prompt_end: int,
    first_estimate_start: int,
    anchor_start: int,
    anchor_end: int,
    final_answer_start: int,
) -> dict[str, int]:
    """Map preregistered token boundaries to readout token indices.

    ``prompt_end`` is already an inclusive token index.  All ``*_start``
    values and ``anchor_end`` are token-boundary indices, with
    ``anchor_end`` exclusive.  A ``*_pre`` readout therefore uses
    ``start - 1``; ``anchor_post`` uses the final token inside the anchor.
    """

    named = {
        "sequence_length": sequence_length,
        "prompt_end": prompt_end,
        "first_estimate_start": first_estimate_start,
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "final_answer_start": final_answer_start,
    }
    for name, value in named.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise PositionMappingError(f"{name} must be an integer; observed {value!r}")
    if sequence_length <= 0:
        raise PositionMappingError("sequence_length must be positive")
    if not 0 <= prompt_end < sequence_length:
        raise PositionMappingError("prompt_end is outside the token sequence")
    for name, start in (
        ("first_estimate_start", first_estimate_start),
        ("anchor_start", anchor_start),
        ("final_answer_start", final_answer_start),
    ):
        if not 1 <= start < sequence_length:
            raise PositionMappingError(f"{name} must leave an in-range preceding token")
        if start <= prompt_end:
            raise PositionMappingError(f"{name} must occur after prompt_end")
    if not anchor_start < anchor_end <= sequence_length:
        raise PositionMappingError(
            "anchor must have a non-empty [anchor_start, anchor_end) token span"
        )
    if anchor_end > final_answer_start:
        raise PositionMappingError("anchor must end no later than final_answer_start")
    if first_estimate_start > final_answer_start:
        raise PositionMappingError("first estimate must not begin after the final answer")

    mapped = {
        "prompt_end": prompt_end,
        "first_estimate_pre": first_estimate_start - 1,
        "anchor_pre": anchor_start - 1,
        "anchor_post": anchor_end - 1,
        "final_answer_pre": final_answer_start - 1,
    }
    if any(index < 0 or index >= sequence_length for index in mapped.values()):
        raise PositionMappingError(
            f"mapped positions are outside sequence length {sequence_length}"
        )
    return mapped


def _finite_logit(logits: Sequence[Any] | Any, token_id: int) -> float:
    try:
        vocab_size = len(logits)
    except TypeError as exc:
        raise LensExecutionError("logits must be a one-dimensional vocabulary vector") from exc
    if token_id < 0 or token_id >= vocab_size:
        raise LensExecutionError(
            f"probe token id {token_id} is outside the logit vocabulary of size {vocab_size}"
        )
    value = logits[token_id]
    if hasattr(value, "item"):
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LensExecutionError(f"logit at token id {token_id} is not scalar") from exc
    if not math.isfinite(numeric):
        raise LensExecutionError(f"logit at token id {token_id} is not finite")
    return numeric


def signed_mean_logit_contrasts(
    logits: Sequence[Any] | Any,
    concepts: Mapping[str, ConceptTokenIds],
) -> dict[str, float]:
    """Compute ``mean(positive logits) - mean(negative logits)`` per concept."""

    contrasts: dict[str, float] = {}
    for concept_name, token_ids in concepts.items():
        if not token_ids.positive_ids or not token_ids.negative_ids:
            raise LensExecutionError(
                f"concept {concept_name!r} has an empty positive or negative token set"
            )
        positive = [_finite_logit(logits, token_id) for token_id in token_ids.positive_ids]
        negative = [_finite_logit(logits, token_id) for token_id in token_ids.negative_ids]
        contrasts[concept_name] = sum(positive) / len(positive) - sum(negative) / len(negative)
    return contrasts


def _model_geometry(hf_model: Any) -> tuple[str | None, int | None, int | None]:
    config = getattr(hf_model, "config", None)
    if config is None:
        return None, None, None
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else config
    model_id = (
        getattr(hf_model, "name_or_path", None)
        or getattr(config, "_name_or_path", None)
        or getattr(text_config, "_name_or_path", None)
    )
    return (
        model_id,
        getattr(text_config, "hidden_size", None),
        getattr(text_config, "num_hidden_layers", None),
    )


def build_model_runtime(
    *,
    model_loader: Callable[..., tuple[Any, Any]] | None,
    model_id: str = EXPECTED_MODEL_ID,
    device_map: str | Mapping[str, Any] = "auto",
    compile: bool = False,
    jlens_module: ModuleType | Any | None = None,
) -> ModelRuntime:
    """Load through an injected callable and wrap with official ``jlens``.

    There is intentionally no default Hugging Face loader: invoking this
    module can never start a 122B model download by accident.  The caller's
    loader receives the pinned model ID and ``device_map``; for the
    preregistered multi-GPU path this is ``"auto"``.  Official jlens warns that
    compilation and automatic sharding are incompatible, so that combination
    is rejected before loading.
    """

    if model_id != EXPECTED_MODEL_ID:
        raise ProvenanceError(
            f"model_id must be pinned to {EXPECTED_MODEL_ID!r}; observed {model_id!r}"
        )
    if device_map == "auto" and compile:
        raise LensExecutionError("multi-GPU device_map='auto' requires compile=False")
    if model_loader is None:
        raise OptionalDependencyError(
            "model_loader must be injected; this module intentionally has no download path"
        )

    try:
        loaded = model_loader(model_id, device_map=device_map)
    except Exception as exc:
        raise LensExecutionError(f"injected model_loader failed: {exc}") from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise LensExecutionError("model_loader must return exactly (hf_model, tokenizer)")
    hf_model, tokenizer = loaded

    observed_id, observed_d_model, observed_n_layers = _model_geometry(hf_model)
    if observed_id != EXPECTED_MODEL_ID:
        raise ProvenanceError(
            f"loaded model_id must be {EXPECTED_MODEL_ID!r}; observed {observed_id!r}"
        )
    if observed_d_model != EXPECTED_D_MODEL:
        raise ProvenanceError(
            f"loaded model d_model must be {EXPECTED_D_MODEL}; observed {observed_d_model!r}"
        )
    if observed_n_layers != EXPECTED_N_LAYERS:
        raise ProvenanceError(
            f"loaded model must have {EXPECTED_N_LAYERS} layers so target/source layer "
            f"{EXPECTED_TARGET_LAYER} is compatible; observed {observed_n_layers!r}"
        )

    jlens_api = _optional_jlens(jlens_module)
    try:
        wrapped = jlens_api.from_hf(hf_model, tokenizer, compile=compile)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise LensExecutionError(f"official jlens.from_hf failed: {exc}") from exc
    if getattr(wrapped, "d_model", None) != EXPECTED_D_MODEL:
        raise ProvenanceError("wrapped jlens model reports an incompatible d_model")
    if getattr(wrapped, "n_layers", None) != EXPECTED_N_LAYERS:
        raise ProvenanceError("wrapped jlens model reports an incompatible layer count")

    return ModelRuntime(
        model_id=model_id,
        model=wrapped,
        tokenizer=tokenizer,
        device_map=device_map,
        compile=compile,
    )


_POSITION_ORDER = (
    "prompt_end",
    "first_estimate_pre",
    "anchor_pre",
    "anchor_post",
    "final_answer_pre",
)


def _position_row(rows: Any, offset: int, *, layer: int, lens_type: str) -> Any:
    try:
        return rows[offset]
    except (IndexError, KeyError, TypeError) as exc:
        raise LensExecutionError(
            f"{lens_type}-lens layer {layer} omitted requested position row {offset}"
        ) from exc


def run_lens_analysis(
    *,
    trace_id: str,
    prefix: str,
    runtime: ModelRuntime,
    lenses: Sequence[LoadedLens],
    position_indices: Mapping[str, int],
    good_side_direction: Literal[-1, 1],
    concept_words: Mapping[str, Mapping[str, Sequence[str]]] = DEFAULT_CONCEPT_WORDS,
    layers: Sequence[int] = FITTED_LAYERS,
    max_seq_len: int = 262_144,
) -> list[LensRecord]:
    """Apply a matched J/R pair and emit observational long-form contrasts.

    This function intentionally performs no hypothesis adjudication and no
    causal labeling.  Sentence-resampling, not a lens readout, supplies causal
    evidence elsewhere in the pipeline.
    """

    if not trace_id.strip():
        raise LensExecutionError("trace_id must be non-empty")
    if good_side_direction not in {-1, 1}:
        raise LensExecutionError("good_side_direction must be -1 or +1")
    if runtime.model_id != EXPECTED_MODEL_ID:
        raise ProvenanceError(
            f"runtime model_id must be {EXPECTED_MODEL_ID!r}; observed {runtime.model_id!r}"
        )
    if getattr(runtime.model, "d_model", None) != EXPECTED_D_MODEL:
        raise ProvenanceError("runtime model has an incompatible d_model")
    if getattr(runtime.model, "n_layers", None) != EXPECTED_N_LAYERS:
        raise ProvenanceError("runtime model has an incompatible layer count")

    by_type: dict[str, LoadedLens] = {}
    for handle in lenses:
        if handle.lens_type in by_type:
            raise LensExecutionError(f"duplicate {handle.lens_type}-lens handle")
        by_type[handle.lens_type] = handle
    if set(by_type) != {"J", "R"}:
        raise LensExecutionError("analysis requires both J and R lens handles")

    observed_position_names = set(position_indices)
    expected_position_names = set(_POSITION_ORDER)
    if observed_position_names != expected_position_names:
        missing = sorted(expected_position_names - observed_position_names)
        extra = sorted(observed_position_names - expected_position_names)
        raise PositionMappingError(
            f"named positions must match {_POSITION_ORDER}; missing={missing}, extra={extra}"
        )
    ordered_positions: list[int] = []
    for name in _POSITION_ORDER:
        index = position_indices[name]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PositionMappingError(f"position {name!r} has invalid token index {index!r}")
        ordered_positions.append(index)

    normalized_layers = tuple(layers)
    if not normalized_layers:
        raise LensExecutionError("at least one fitted layer is required")
    if (
        len(set(normalized_layers)) != len(normalized_layers)
        or tuple(sorted(normalized_layers)) != normalized_layers
    ):
        raise LensExecutionError("layers must be unique and sorted")
    unknown_layers = sorted(set(normalized_layers) - set(FITTED_LAYERS))
    if unknown_layers:
        raise LensExecutionError(f"layers are outside fitted range 4..46: {unknown_layers}")

    prefix_ids = _encode(runtime.tokenizer, prefix)
    if not prefix_ids:
        raise LensExecutionError("analyzed prefix tokenized to an empty sequence")
    if max(ordered_positions) >= len(prefix_ids):
        raise PositionMappingError(
            f"position index {max(ordered_positions)} exceeds prefix length {len(prefix_ids)}"
        )
    if max_seq_len < len(prefix_ids):
        raise LensExecutionError(
            f"max_seq_len={max_seq_len} would truncate the {len(prefix_ids)}-token prefix"
        )
    concepts = validate_concept_tokens(
        runtime.tokenizer,
        prefixes=(prefix,),
        concept_words=concept_words,
    )

    prefix_digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    records: list[LensRecord] = []
    for lens_type in ("J", "R"):
        handle = by_type[lens_type]
        provenance = handle.provenance
        if (
            provenance.model_id != EXPECTED_MODEL_ID
            or provenance.d_model != EXPECTED_D_MODEL
            or provenance.target_layer != EXPECTED_TARGET_LAYER
            or provenance.source_layers != ARTIFACT_SOURCE_LAYERS
        ):
            raise ProvenanceError(f"{lens_type}-lens handle has incompatible provenance")
        try:
            lens_logits, _model_logits, _input_ids = handle.lens.apply(
                runtime.model,
                prefix,
                layers=normalized_layers,
                positions=ordered_positions,
                max_seq_len=max_seq_len,
                use_jacobian=True,
            )
        except Exception as exc:
            raise LensExecutionError(f"{lens_type}-lens apply failed: {exc}") from exc
        if not isinstance(lens_logits, Mapping):
            raise LensExecutionError(f"{lens_type}-lens apply returned non-mapping logits")

        for layer in normalized_layers:
            if layer not in lens_logits:
                raise LensExecutionError(f"{lens_type}-lens output omitted layer {layer}")
            rows = lens_logits[layer]
            for position_offset, position_name in enumerate(_POSITION_ORDER):
                logits = _position_row(
                    rows,
                    position_offset,
                    layer=layer,
                    lens_type=lens_type,
                )
                contrasts = signed_mean_logit_contrasts(logits, concepts)
                for contrast_name, value in contrasts.items():
                    token_ids = concepts[contrast_name]
                    aligned_value = (
                        value * good_side_direction if contrast_name == "direction" else value
                    )
                    records.append(
                        LensRecord(
                            trace_id=trace_id,
                            prefix_sha256=prefix_digest,
                            model_id=runtime.model_id,
                            lens_type=lens_type,  # type: ignore[arg-type]
                            lens_file_sha256=provenance.file_sha256,
                            target_layer=provenance.target_layer,
                            layer=layer,
                            layer_band=layer_band(layer),
                            position_name=position_name,
                            token_index=position_indices[position_name],
                            contrast=contrast_name,
                            raw_mean_logit_contrast=value,
                            signed_mean_logit_contrast=aligned_value,
                            good_side_direction=good_side_direction,
                            positive_token_ids=token_ids.positive_ids,
                            negative_token_ids=token_ids.negative_ids,
                        )
                    )
    return records


def _as_token_ids(encoded: Any, *, context: str) -> tuple[int, ...]:
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
        raise ConceptValidationError(f"tokenizer returned invalid input_ids for {context}")
    try:
        return tuple(int(token_id) for token_id in encoded)
    except (TypeError, ValueError) as exc:
        raise ConceptValidationError(
            f"tokenizer returned non-integer input_ids for {context}"
        ) from exc


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
        else:
            encoded = tokenizer(text, add_special_tokens=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConceptValidationError(f"tokenizer could not encode {text!r}: {exc}") from exc
    return _as_token_ids(encoded, context=repr(text))


def validate_concept_tokens(
    tokenizer: Any,
    *,
    prefixes: Sequence[str],
    concept_words: Mapping[str, Mapping[str, Sequence[str]]] = DEFAULT_CONCEPT_WORDS,
) -> dict[str, ConceptTokenIds]:
    """Validate probes and retain only words absent from every analyzed prefix.

    The candidate vocabulary is frozen before results.  For each trace, probes
    copied in the analyzed prefix are removed without looking at outcomes; each
    polarity must retain at least one word.  Both token-ID and case-insensitive
    lexical membership are checked.  The lexical check catches a concept at the
    beginning of a prefix, where a decoder tokenizer may use a different
    non-space-prefixed token.
    """

    tokenized: dict[str, ConceptTokenIds] = {}
    all_word_ids: dict[int, tuple[str, str]] = {}
    for concept_name, polarities in concept_words.items():
        positive_words = tuple(polarities.get("positive", ()))
        negative_words = tuple(polarities.get("negative", ()))
        if not positive_words or not negative_words:
            raise ConceptValidationError(
                f"concept {concept_name!r} needs non-empty positive and negative words"
            )

        ids_by_polarity: dict[str, tuple[int, ...]] = {}
        for polarity, words in (
            ("positive", positive_words),
            ("negative", negative_words),
        ):
            collected: list[int] = []
            for word in words:
                ids = _encode(tokenizer, word)
                if len(ids) != 1:
                    raise ConceptValidationError(
                        f"probe word {word!r} for {concept_name}/{polarity} must encode "
                        f"to exactly one token; observed {len(ids)}"
                    )
                token_id = ids[0]
                previous = all_word_ids.get(token_id)
                if previous is not None:
                    raise ConceptValidationError(
                        f"probe token id {token_id} is reused by {previous[0]}/{previous[1]} "
                        f"and {concept_name}/{word.strip()}"
                    )
                all_word_ids[token_id] = (concept_name, word.strip())
                collected.append(token_id)
            ids_by_polarity[polarity] = tuple(collected)

        tokenized[concept_name] = ConceptTokenIds(
            positive_ids=ids_by_polarity["positive"],
            negative_ids=ids_by_polarity["negative"],
            positive_words=positive_words,
            negative_words=negative_words,
        )

    copied_probe_ids: set[int] = set()
    copied_lexical_words: set[str] = set()
    lexical_words = {
        word.strip().casefold(): word
        for polarities in concept_words.values()
        for words in polarities.values()
        for word in words
    }
    for prefix in prefixes:
        prefix_ids = set(_encode(tokenizer, prefix))
        copied_probe_ids.update(prefix_ids & set(all_word_ids))
        folded = prefix.casefold()
        copied_lexical_words.update(
            word for word in lexical_words if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", folded)
        )

    filtered: dict[str, ConceptTokenIds] = {}
    for concept_name, probes in tokenized.items():
        positive = tuple(
            (token_id, word)
            for token_id, word in zip(probes.positive_ids, probes.positive_words, strict=True)
            if token_id not in copied_probe_ids
            and word.strip().casefold() not in copied_lexical_words
        )
        negative = tuple(
            (token_id, word)
            for token_id, word in zip(probes.negative_ids, probes.negative_words, strict=True)
            if token_id not in copied_probe_ids
            and word.strip().casefold() not in copied_lexical_words
        )
        if not positive or not negative:
            raise ConceptValidationError(
                f"concept {concept_name!r} has no prefix-absent probe in one polarity"
            )
        filtered[concept_name] = ConceptTokenIds(
            positive_ids=tuple(item[0] for item in positive),
            negative_ids=tuple(item[0] for item in negative),
            positive_words=tuple(item[1] for item in positive),
            negative_words=tuple(item[1] for item in negative),
        )

    return filtered


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of a local file without loading it at once."""

    artifact = Path(path)
    if not artifact.is_file():
        raise ProvenanceError(f"lens artifact is not a local file: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file_sha256(path: Path, expected_sha256: str) -> str:
    declared_digest = expected_sha256.lower()
    if len(declared_digest) != 64 or any(c not in "0123456789abcdef" for c in declared_digest):
        raise ProvenanceError("expected SHA256 must be a 64-character hexadecimal digest")
    actual_digest = sha256_file(path)
    if actual_digest != declared_digest:
        raise ProvenanceError(
            f"lens SHA256 mismatch: expected {declared_digest}, observed {actual_digest}"
        )
    return actual_digest


def _optional_jlens(jlens_module: ModuleType | Any | None) -> Any:
    if jlens_module is not None:
        return jlens_module
    try:
        return importlib.import_module("jlens")
    except ImportError as exc:
        raise OptionalDependencyError(
            "official 'jlens' is required for lens loading; install the GPU extra "
            "or inject jlens_module for a controlled runtime"
        ) from exc


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise OptionalDependencyError(
            "torch is required to inspect local lens provenance; inject "
            "checkpoint_reader in dependency-light environments"
        ) from exc
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ProvenanceError("lens checkpoint must be a mapping")
    return checkpoint


def _validated_provenance(
    path: Path,
    *,
    lens_type: Literal["J", "R"],
    file_sha256: str,
    checkpoint: Mapping[str, Any],
) -> LensProvenance:
    embedded = checkpoint.get("provenance")
    if not isinstance(embedded, Mapping):
        raise ProvenanceError("lens checkpoint has no mapping-valued provenance")

    model_id = embedded.get("model_id")
    if model_id != EXPECTED_MODEL_ID:
        raise ProvenanceError(f"lens model_id must be {EXPECTED_MODEL_ID!r}; observed {model_id!r}")

    d_model = checkpoint.get("d_model")
    if d_model != EXPECTED_D_MODEL:
        raise ProvenanceError(f"lens d_model must be {EXPECTED_D_MODEL}; observed {d_model!r}")

    target_layer = embedded.get("target_layer")
    if target_layer != EXPECTED_TARGET_LAYER:
        raise ProvenanceError(
            f"lens target_layer must be {EXPECTED_TARGET_LAYER}; observed {target_layer!r}"
        )

    try:
        source_layers = tuple(int(layer) for layer in checkpoint["source_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProvenanceError("lens source_layers must be an integer sequence") from exc
    if source_layers != ARTIFACT_SOURCE_LAYERS:
        raise ProvenanceError(
            "lens source_layers must exactly cover artifact layers "
            f"{ARTIFACT_SOURCE_LAYERS[0]}..{ARTIFACT_SOURCE_LAYERS[-1]}; observed "
            f"{source_layers[0] if source_layers else None}.."
            f"{source_layers[-1] if source_layers else None} ({len(source_layers)} layers)"
        )

    skip_first = embedded.get("skip_first")
    if skip_first != FITTED_LAYERS[0]:
        raise ProvenanceError(
            f"lens skip_first must be {FITTED_LAYERS[0]}; observed {skip_first!r}"
        )

    n_prompts = checkpoint.get("n_prompts")
    if isinstance(n_prompts, bool) or not isinstance(n_prompts, int) or n_prompts <= 0:
        raise ProvenanceError(f"lens n_prompts must be a positive integer; observed {n_prompts!r}")
    config_json = embedded.get("config_json")
    try:
        fit_config = json.loads(config_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("lens provenance config_json must be valid JSON") from exc
    if not isinstance(fit_config, Mapping):
        raise ProvenanceError("lens provenance config_json must decode to a mapping")
    expected_estimator = "standard" if lens_type == "J" else "relp"
    if fit_config.get("estimator") != expected_estimator:
        raise ProvenanceError(
            f"{lens_type}-lens estimator must be {expected_estimator!r}; "
            f"observed {fit_config.get('estimator')!r}"
        )

    return LensProvenance(
        model_id=model_id,
        d_model=d_model,
        target_layer=target_layer,
        source_layers=source_layers,
        file_sha256=file_sha256,
        artifact_path=str(path),
        metadata={
            "n_prompts": n_prompts,
            "dataset_id": embedded.get("dataset_id"),
            "target_layer": target_layer,
            "skip_first": skip_first,
            "fit_config": dict(fit_config),
            "t_max": embedded.get("t_max"),
            "docs_consumed": embedded.get("docs_consumed"),
            "weighting": embedded.get("weighting"),
            "corpus_mode": embedded.get("corpus_mode"),
        },
    )


def load_local_lens(
    path: str | Path,
    *,
    lens_type: Literal["J", "R"] | str,
    expected_sha256: str,
    checkpoint_reader: Callable[[Path], Mapping[str, Any]] | None = None,
    jlens_module: ModuleType | Any | None = None,
) -> LoadedLens:
    """Validate then load a local J/R artifact with the official ``jlens`` class.

    No Hub path is accepted, so this helper cannot implicitly download a lens.
    The file digest and embedded metadata are checked before deserialization by
    ``JacobianLens.load``.
    """

    normalized_type = str(lens_type).upper()
    if normalized_type not in {"J", "R"}:
        raise ProvenanceError(f"lens_type must be 'J' or 'R'; observed {lens_type!r}")

    artifact = Path(path).expanduser().resolve()
    file_sha256 = _validate_file_sha256(artifact, expected_sha256)
    reader = checkpoint_reader or _read_checkpoint
    checkpoint = reader(artifact)
    if not isinstance(checkpoint, Mapping):
        raise ProvenanceError("checkpoint_reader must return a mapping")
    provenance = _validated_provenance(
        artifact,
        lens_type=normalized_type,  # type: ignore[arg-type]
        file_sha256=file_sha256,
        checkpoint=checkpoint,
    )

    jlens_api = _optional_jlens(jlens_module)
    try:
        lens = jlens_api.JacobianLens.load(str(artifact))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LensError(f"official jlens failed to load {artifact}: {exc}") from exc

    loaded_d_model = getattr(lens, "d_model", None)
    loaded_sources = tuple(getattr(lens, "source_layers", ()))
    if loaded_d_model != provenance.d_model or loaded_sources != provenance.source_layers:
        raise ProvenanceError(
            "official jlens object disagrees with validated checkpoint geometry: "
            f"d_model={loaded_d_model!r}, source_layers={loaded_sources!r}"
        )

    return LoadedLens(
        lens_type=normalized_type,  # type: ignore[arg-type]
        lens=lens,
        provenance=provenance,
    )
