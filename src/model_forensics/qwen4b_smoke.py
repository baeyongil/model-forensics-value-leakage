"""Bounded, opt-in Qwen3.5-4B GPU smoke for rollout and raw-prefix paths.

Importing this module performs no network access, model loading, or CUDA setup.
Only :func:`run_qwen4b_prefix_smoke` enters the GPU path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash, write_json
from model_forensics.prompts import build_prompt
from model_forensics.resample_runner import RawPrefixGenerationRequest
from model_forensics.sampling import (
    SamplingParameters,
    VLLMOfflineBackend,
    build_requests,
    materialize_rollout_rows,
)
from model_forensics.token_spans import CompletionTokenMap, locate_completion_sections
from model_forensics.vllm_prefix import VLLMRawPrefixBackend

SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
SMOKE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SMOKE_SEED = 20260829


class Qwen4BSmokeError(RuntimeError):
    """The real-model compatibility smoke failed an exactness gate."""


def _validate_bounds(
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    rollout_max_tokens: int,
    continuation_max_tokens: int,
) -> None:
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or not 1 <= tensor_parallel_size <= 8
    ):
        raise ValueError("tensor_parallel_size must be in 1..8")
    if (
        isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or not 1024 <= max_model_len <= 8192
    ):
        raise ValueError("4B smoke max_model_len must be in 1024..8192")
    if (
        isinstance(rollout_max_tokens, bool)
        or not isinstance(rollout_max_tokens, int)
        or not 128 <= rollout_max_tokens <= 2048
    ):
        raise ValueError("rollout_max_tokens must be in 128..2048")
    if (
        isinstance(continuation_max_tokens, bool)
        or not isinstance(continuation_max_tokens, int)
        or not 16 <= continuation_max_tokens <= 512
    ):
        raise ValueError("continuation_max_tokens must be in 16..512")
    if rollout_max_tokens + continuation_max_tokens >= max_model_len:
        raise ValueError("smoke generation bounds leave no safe prompt/context margin")


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU-only path
        raise Qwen4BSmokeError("Torch is unavailable in the GPU environment") from exc
    if not torch.cuda.is_available():  # pragma: no cover - GPU-only path
        raise Qwen4BSmokeError("Qwen4B smoke requires a CUDA GPU")


def _decode_visible_completion_prefix(
    tokenizer: Any,
    prompt_token_ids: tuple[int, ...],
    completion_token_ids: tuple[int, ...],
    count: int,
) -> str:
    kwargs = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    decoded_prompt = tokenizer.decode(list(prompt_token_ids), **kwargs)
    decoded_combined = tokenizer.decode(
        [*prompt_token_ids, *completion_token_ids[:count]],
        **kwargs,
    )
    if not decoded_combined.startswith(decoded_prompt):
        raise Qwen4BSmokeError("completion decoding changed the rendered prompt")
    return decoded_combined[len(decoded_prompt) :]


def _choose_exact_reasoning_boundary(
    *,
    tokenizer: Any,
    prompt_token_ids: tuple[int, ...],
    completion_token_ids: tuple[int, ...],
    reasoning: str,
) -> tuple[int, str, int, str]:
    candidates: list[tuple[int, str]] = []
    for count in range(1, len(completion_token_ids) + 1):
        visible = _decode_visible_completion_prefix(
            tokenizer,
            prompt_token_ids,
            completion_token_ids,
            count,
        )
        if visible and len(visible) < len(reasoning) and reasoning.startswith(visible):
            candidates.append((count, visible))
    if len(candidates) < 2:
        raise Qwen4BSmokeError("full rollout exposed fewer than two reasoning token boundaries")
    target = max(8, min(64, len(reasoning) // 3))
    prefix_count, raw_prefix = min(
        candidates[:-1],
        key=lambda item: (abs(len(item[1]) - target), item[0]),
    )
    next_count, next_prefix = next(
        item for item in candidates if item[0] > prefix_count and len(item[1]) > len(raw_prefix)
    )
    forced_text = next_prefix[len(raw_prefix) :]
    if not forced_text:
        raise Qwen4BSmokeError("next original token boundary produced no visible continuation")
    return prefix_count, raw_prefix, next_count, forced_text


def run_qwen4b_prefix_smoke(
    output_path: str | Path,
    *,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    rollout_max_tokens: int = 1024,
    continuation_max_tokens: int = 256,
) -> dict[str, Any]:
    """Run exactly one 4B rollout and one exact raw-prefix continuation.

    This is a compatibility gate, not an experimental sample.  It never calls a
    paid API and cannot be pointed at the 122B checkpoint through function
    arguments.  The only potentially networked operation is loading the pinned
    public 4B model after this function is explicitly invoked.
    """

    _validate_bounds(
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        rollout_max_tokens=rollout_max_tokens,
        continuation_max_tokens=continuation_max_tokens,
    )
    _require_cuda()

    prefix_backend = VLLMRawPrefixBackend(
        model_id=SMOKE_MODEL_ID,
        revision=SMOKE_MODEL_REVISION,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        parameters=SamplingParameters(max_new_tokens=continuation_max_tokens),
        require_registered_prefixes=True,
        use_tqdm=True,
    )
    rollout_backend = VLLMOfflineBackend(
        model_id=SMOKE_MODEL_ID,
        revision=SMOKE_MODEL_REVISION,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        llm_factory=lambda **_: prefix_backend.llm,
    )
    rollout_request = build_requests(
        task="giraffe",
        condition="baseline",
        count=1,
        threshold=None,
        master_seed=SMOKE_SEED,
        prompt_builder=build_prompt,
        parameters=SamplingParameters(max_new_tokens=rollout_max_tokens),
        randomize=False,
    )[0]
    rollout_result = rollout_backend.generate([rollout_request])[0]
    if rollout_result.prompt_token_ids is None or rollout_result.completion_token_ids is None:
        raise Qwen4BSmokeError("full rollout omitted exact token streams")
    prompt_ids = tuple(rollout_result.prompt_token_ids)
    completion_ids = tuple(rollout_result.completion_token_ids)
    sections = locate_completion_sections(rollout_result.raw_text)
    if not sections.reasoning or not sections.answer:
        raise Qwen4BSmokeError(
            "bounded full rollout did not finish both reasoning and answer sections"
        )
    if sections.reasoning_char_start != 0:
        raise Qwen4BSmokeError(
            "full rollout did not begin directly inside the prompt-opened thinking block"
        )
    CompletionTokenMap(
        tokenizer=prefix_backend.tokenizer,
        raw_text=rollout_result.raw_text,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        skip_special_tokens=True,
    )
    prefix_count, raw_prefix, next_count, forced_text = _choose_exact_reasoning_boundary(
        tokenizer=prefix_backend.tokenizer,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        reasoning=sections.reasoning,
    )
    messages = ({"role": "user", "content": rollout_request.prompt},)
    registration = prefix_backend.register_generated_prefix(
        messages=messages,
        raw_completion_text=rollout_result.raw_text,
        original_prompt_token_ids=prompt_ids,
        original_completion_token_ids=completion_ids,
        raw_thinking_prefix=raw_prefix,
    )
    exact_prefix_ids = tuple(prefix_backend.encode_prefix(messages, raw_prefix))
    expected_prefix_ids = prompt_ids + completion_ids[:prefix_count]
    if exact_prefix_ids != expected_prefix_ids:
        raise Qwen4BSmokeError("registered raw prefix differs from original rollout IDs")
    forced_ids = tuple(prefix_backend.encode_continuation(forced_text))
    decoded_before = prefix_backend.tokenizer.decode(
        list(exact_prefix_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_after = prefix_backend.tokenizer.decode(
        [*exact_prefix_ids, *forced_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_after != decoded_before + forced_text:
        raise Qwen4BSmokeError("forced append changed the immutable raw prefix")

    prefix_request = RawPrefixGenerationRequest(
        request_id=stable_hash(
            {
                "smoke": "qwen4b_raw_prefix",
                "rollout_request_id": rollout_request.request_id,
                "prefix_token_count": len(exact_prefix_ids),
                "seed": SMOKE_SEED,
            }
        ).split(":", 1)[1][:24],
        anchor_id="qwen4b-smoke-anchor",
        base_trace_id=rollout_request.request_id,
        arm="resample",
        sample_index=0,
        seed=SMOKE_SEED,
        messages=messages,
        conditioning_text=raw_prefix,
        prompt_token_ids=exact_prefix_ids,
        common_prefix_token_count=len(exact_prefix_ids),
    )
    prefix_result = prefix_backend.generate([prefix_request])[0]
    if prefix_result.prompt_token_ids != exact_prefix_ids:
        raise Qwen4BSmokeError("raw-prefix generation consumed different prompt IDs")
    completion_metadata = prefix_result.backend_metadata.get("completion_token_ids")
    if not isinstance(completion_metadata, list) or not completion_metadata:
        raise Qwen4BSmokeError("raw-prefix generation omitted completion token IDs")
    generated_completion_ids = tuple(completion_metadata)
    CompletionTokenMap(
        tokenizer=prefix_backend.tokenizer,
        raw_text=prefix_result.generated_text,
        prompt_token_ids=exact_prefix_ids,
        completion_token_ids=generated_completion_ids,
        skip_special_tokens=True,
    )

    rollout_row = materialize_rollout_rows(
        [rollout_request],
        [rollout_result],
        backend_provenance=rollout_backend.provenance,
    )[0]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "scope": "one_full_rollout_plus_one_raw_prefix_continuation",
        "experimental_sample": False,
        "paid_api_calls": 0,
        "bounds": {
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "rollout_count": 1,
            "rollout_max_tokens": rollout_max_tokens,
            "raw_prefix_continuation_count": 1,
            "continuation_max_tokens": continuation_max_tokens,
        },
        "model": {
            "id": SMOKE_MODEL_ID,
            "revision": SMOKE_MODEL_REVISION,
        },
        "full_rollout": {
            "request_id": rollout_request.request_id,
            "seed": rollout_request.seed,
            "record_hash": rollout_row["record_hash"],
            "finish_reason": rollout_result.finish_reason,
            "reasoning_present": True,
            "answer_present": True,
            "prompt_token_count": len(prompt_ids),
            "completion_token_count": len(completion_ids),
            "token_streams": rollout_row["token_streams"],
        },
        "registered_prefix": {
            **registration.as_dict(),
            "original_completion_boundary": prefix_count,
            "exact_original_ids_reused": exact_prefix_ids == expected_prefix_ids,
        },
        "forced_append_check": {
            "next_original_boundary": next_count,
            "forced_text_hash": stable_hash(forced_text),
            "forced_token_ids_hash": stable_hash(list(forced_ids)),
            "immutable_prefix_preserved": True,
        },
        "raw_prefix_continuation": {
            "request_id": prefix_request.request_id,
            "seed": prefix_request.seed,
            "finish_reason": prefix_result.finish_reason,
            "prompt_token_count": prefix_result.prompt_tokens,
            "completion_token_count": prefix_result.completion_tokens,
            "prompt_ids_exact": prefix_result.prompt_token_ids == exact_prefix_ids,
            "prompt_token_ids_hash": prefix_result.backend_metadata["prompt_token_ids_hash"],
            "completion_token_ids_hash": prefix_result.backend_metadata[
                "completion_token_ids_hash"
            ],
            "combined_token_stream_hash": prefix_result.backend_metadata[
                "combined_token_stream_hash"
            ],
        },
        "backend_provenance": dict(prefix_backend.provenance),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(output_path, manifest)
    return manifest


__all__ = [
    "SMOKE_MODEL_ID",
    "SMOKE_MODEL_REVISION",
    "SMOKE_SEED",
    "Qwen4BSmokeError",
    "run_qwen4b_prefix_smoke",
]
