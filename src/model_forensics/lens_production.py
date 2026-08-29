"""Production factories and outcome-blind compatibility prefixes for lens execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from model_forensics.config import RunConfig
from model_forensics.lens_command import (
    CompatibilityPrefixes,
    PrimaryRuntimeBundle,
    SmokeRuntimeBundle,
    ValidatedLensInputs,
)
from model_forensics.lens_runner import (
    PRIMARY_LENS_PINS,
    PRIMARY_MODEL_PIN,
    SMOKE_MODEL_PIN,
    JlensTorchSameForwardBackend,
    download_and_load_lens_pair,
    load_pinned_text_runtime,
    verify_software_revisions,
)

FROZEN_4B_COMPATIBILITY_TEXT = (
    "Text-only compatibility check: estimate a quantity using neutral arithmetic."
)


def assert_primary_lens_config(config: RunConfig) -> None:
    """Require the user-facing run config to equal the compiled artifact pins."""

    if config.model.id != PRIMARY_MODEL_PIN.model_id or config.model.revision != (
        PRIMARY_MODEL_PIN.revision
    ):
        raise ValueError("run config does not match the compiled 122B primary model pin")
    by_type = {pin.lens_type: pin for pin in PRIMARY_LENS_PINS}
    expected = {
        "repository": by_type["J"].repository,
        "revision": by_type["J"].revision,
        "j_filename": by_type["J"].filename,
        "r_filename": by_type["R"].filename,
        "j_sha256": by_type["J"].sha256,
        "r_sha256": by_type["R"].sha256,
        "j_size_bytes": by_type["J"].size_bytes,
        "r_size_bytes": by_type["R"].size_bytes,
    }
    observed = {
        "repository": config.lenses.repository,
        "revision": config.lenses.revision,
        "j_filename": config.lenses.j_filename,
        "r_filename": config.lenses.r_filename,
        "j_sha256": config.lenses.j_sha256,
        "r_sha256": config.lenses.r_sha256,
        "j_size_bytes": config.lenses.j_size_bytes,
        "r_size_bytes": config.lenses.r_size_bytes,
    }
    if observed != expected:
        raise ValueError("run config does not match the compiled J/R lens artifact pins")


def encode_frozen_4b_compatibility_prefix(
    *,
    tokenizer_factory: Callable[..., Any] | None = None,
) -> tuple[int, ...]:
    """Tokenize a fixed neutral smoke phrase at the pinned 4B revision."""

    if tokenizer_factory is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers is required for the 4B compatibility prefix") from exc
        tokenizer_factory = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_factory(
        SMOKE_MODEL_PIN.model_id,
        revision=SMOKE_MODEL_PIN.revision,
        trust_remote_code=False,
        use_fast=True,
    )
    encoded = tokenizer.encode(FROZEN_4B_COMPATIBILITY_TEXT, add_special_tokens=True)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise TypeError("4B tokenizer returned invalid token IDs")
    token_ids = tuple(int(value) for value in encoded)
    if not token_ids or len(token_ids) > 256 or any(value < 0 for value in token_ids):
        raise ValueError("4B compatibility prefix must contain 1..256 nonnegative tokens")
    return token_ids


def freeze_production_compatibility_prefixes(
    validated: ValidatedLensInputs,
    *,
    four_b_token_ids: Sequence[int],
    shortened_primary_limit: int = 2048,
) -> CompatibilityPrefixes:
    """Choose the first manifest-ordered trace, independent of observed readouts."""

    if shortened_primary_limit <= 0:
        raise ValueError("shortened prefix limit must be positive")
    if not validated.traces:
        raise ValueError("validated lens inputs contain no selected traces")
    source = validated.traces[0]
    full = source.sequence_token_ids
    if len(full) < 2:
        raise ValueError("primary compatibility stream is too short to shorten")
    short_length = min(shortened_primary_limit, max(1, len(full) // 2))
    if short_length >= len(full):
        short_length = len(full) - 1
    return CompatibilityPrefixes.freeze(
        four_b_token_ids=four_b_token_ids,
        primary_trace_id=source.trace_id,
        primary_full_token_ids=full,
        primary_short_token_ids=full[:short_length],
    )


def production_runtime_factories(
    *,
    lens_cache_dir: str | Path,
    primary_cuda_devices: int = 8,
    per_gpu_memory_gib: int = 76,
) -> tuple[Callable[[], SmokeRuntimeBundle], Callable[[], PrimaryRuntimeBundle]]:
    """Return lazy, revision-checked 4B and 122B runtime factories."""

    if primary_cuda_devices != 8:
        raise ValueError("the primary lens job is frozen to exactly eight CUDA devices")
    if per_gpu_memory_gib <= 0 or per_gpu_memory_gib > 79:
        raise ValueError("per-GPU memory limit must be in 1..79 GiB")
    cache = Path(lens_cache_dir).expanduser().resolve()
    verify_software_revisions()

    def release_cuda_cache() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def smoke_factory() -> SmokeRuntimeBundle:
        pinned = load_pinned_text_runtime(
            SMOKE_MODEL_PIN,
            required_cuda_devices=1,
            per_gpu_memory_gib=per_gpu_memory_gib,
            verify_dependencies=False,
        )
        return SmokeRuntimeBundle(
            runtime=pinned.runtime,
            backend=JlensTorchSameForwardBackend(),
            post_release=release_cuda_cache,
        )

    def primary_factory() -> PrimaryRuntimeBundle:
        pinned = load_pinned_text_runtime(
            PRIMARY_MODEL_PIN,
            required_cuda_devices=primary_cuda_devices,
            per_gpu_memory_gib=per_gpu_memory_gib,
            verify_dependencies=False,
        )
        pair = download_and_load_lens_pair(
            PRIMARY_LENS_PINS,
            cache_dir=cache,
        )
        return PrimaryRuntimeBundle(
            runtime=pinned.runtime,
            lenses=pair.handles,
            backend=JlensTorchSameForwardBackend(),
        )

    return smoke_factory, primary_factory


__all__ = [
    "FROZEN_4B_COMPATIBILITY_TEXT",
    "assert_primary_lens_config",
    "encode_frozen_4b_compatibility_prefix",
    "freeze_production_compatibility_prefixes",
    "production_runtime_factories",
]
