"""Bounded, non-primary Qwen3.5-4B GPU integration gate.

Importing this module performs no network access, model loading, or CUDA setup.
Only :func:`run_qwen4b_prefix_smoke` enters the GPU path. The resulting rows are
explicitly synthetic/non-primary and can never be consumed as research evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash, write_json
from model_forensics.lens import DEFAULT_CONCEPT_WORDS
from model_forensics.lens_runner import (
    POSITION_ORDER,
    LensTraceInput,
    freeze_causal_probe_design,
)
from model_forensics.parsing import parse_trajectory
from model_forensics.prompts import build_prompt
from model_forensics.resample_runner import RawPrefixGenerationRequest
from model_forensics.sampling import (
    SamplingParameters,
    VLLMOfflineBackend,
    build_requests,
    materialize_rollout_rows,
)
from model_forensics.token_spans import (
    CompletionTokenMap,
    locate_completion_sections,
    token_stream_manifest,
)
from model_forensics.vllm_prefix import VLLMRawPrefixBackend

SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
SMOKE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SMOKE_SEED = 20260829
SMOKE_PROTOCOL = "qwen4b-bounded-integration-smoke-v2"
SMOKE_SCOPE = "bounded_nonprimary_qwen4b_integration_smoke"

_FIXTURE_REASONING = (
    "A first estimate is 36 million cups. "
    "I will calculate from physical factors rather than target a preferred side. "
    "After revising one factor, I obtain 42 million cups."
)
_FIXTURE_ANSWER = "Final answer: 42 million cups per weekday."
_FIXTURE_ANCHOR = (
    "I will calculate from physical factors rather than target a preferred side."
)


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
    if rollout_max_tokens + 2 * continuation_max_tokens >= max_model_len:
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


def _fixture_evidence(tokenizer: Any) -> dict[str, Any]:
    """Exercise deterministic parsing/trajectory and exact span mapping."""

    trajectory = parse_trajectory(
        _FIXTURE_REASONING,
        _FIXTURE_ANSWER,
        threshold=41_000_000,
        condition="above_good",
    )
    features = trajectory.features
    if (
        features.first_estimate != 36_000_000
        or features.final_estimate != 42_000_000
        or features.revision_count != 1
        or features.first_good_side_crossing_index != 1
        or features.stopped_after_first_good_side_crossing is not True
    ):
        raise Qwen4BSmokeError("deterministic parser/trajectory fixture changed")

    encoded = tokenizer.encode(_FIXTURE_REASONING, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise Qwen4BSmokeError("4B tokenizer could not encode the anchor fixture")
    fixture_ids = tuple(int(value) for value in encoded)
    mapper = CompletionTokenMap(
        tokenizer=tokenizer,
        raw_text=_FIXTURE_REASONING,
        completion_token_ids=fixture_ids,
        skip_special_tokens=True,
    )
    anchor_start = _FIXTURE_REASONING.index(_FIXTURE_ANCHOR)
    anchor_span = mapper.map_completion_span(
        anchor_start,
        anchor_start + len(_FIXTURE_ANCHOR),
        expected_text=_FIXTURE_ANCHOR,
        section="synthetic_fixture_reasoning",
        section_char_start=anchor_start,
        section_char_end=anchor_start + len(_FIXTURE_ANCHOR),
    )
    payload: dict[str, Any] = {
        "fixture_kind": "deterministic_nonprimary_local_contract",
        "synthetic": True,
        "primary_eligible": False,
        "reasoning_hash": stable_hash(_FIXTURE_REASONING),
        "answer_hash": stable_hash(_FIXTURE_ANSWER),
        "trajectory": trajectory.to_dict(include_hash=True),
        "anchor": {
            "sentence_hash": stable_hash(_FIXTURE_ANCHOR),
            "character_start": anchor_start,
            "character_end": anchor_start + len(_FIXTURE_ANCHOR),
            "token_span": anchor_span.as_dict(),
        },
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _continuation_request(
    *,
    rollout_request_id: str,
    messages: tuple[dict[str, str], ...],
    arm: str,
    conditioning_text: str,
    prompt_token_ids: tuple[int, ...],
    common_prefix_token_count: int,
) -> RawPrefixGenerationRequest:
    return RawPrefixGenerationRequest(
        request_id=stable_hash(
            {
                "smoke": "qwen4b_raw_prefix",
                "rollout_request_id": rollout_request_id,
                "arm": arm,
                "conditioning_text_hash": stable_hash(conditioning_text),
                "prompt_token_ids": list(prompt_token_ids),
                "seed": SMOKE_SEED,
            }
        ).split(":", 1)[1][:24],
        anchor_id="qwen4b-smoke-anchor",
        base_trace_id=rollout_request_id,
        arm=arm,
        sample_index=0,
        seed=SMOKE_SEED,
        messages=messages,
        conditioning_text=conditioning_text,
        prompt_token_ids=prompt_token_ids,
        common_prefix_token_count=common_prefix_token_count,
    )


def _continuation_evidence(
    result: Any,
    *,
    expected_ids: tuple[int, ...],
) -> dict[str, Any]:
    if result.prompt_token_ids != expected_ids:
        raise Qwen4BSmokeError("raw-prefix generation consumed different prompt IDs")
    metadata = result.backend_metadata
    completion_ids = metadata.get("completion_token_ids")
    if not isinstance(completion_ids, list) or not completion_ids:
        raise Qwen4BSmokeError("raw-prefix generation omitted completion token IDs")
    return {
        "request_id": result.request_id,
        "seed": SMOKE_SEED,
        "finish_reason": result.finish_reason,
        "prompt_token_count": result.prompt_tokens,
        "completion_token_count": result.completion_tokens,
        "prompt_ids_exact": result.prompt_token_ids == expected_ids,
        "prompt_token_ids_hash": metadata["prompt_token_ids_hash"],
        "completion_token_ids_hash": metadata["completion_token_ids_hash"],
        "combined_token_stream_hash": metadata["combined_token_stream_hash"],
    }


def _probe_grid_evidence(
    *,
    tokenizer: Any,
    prompt_ids: tuple[int, ...],
    completion_ids: tuple[int, ...],
    prefix_count: int,
    next_count: int,
    answer_first_token: int,
) -> dict[str, Any]:
    prompt_count = len(prompt_ids)
    positions = {
        "prompt_end": prompt_count - 1,
        "first_estimate_pre": prompt_count,
        "anchor_pre": prompt_count + prefix_count - 1,
        "anchor_post": prompt_count + next_count - 1,
        "final_answer_pre": prompt_count + answer_first_token - 1,
    }
    streams = token_stream_manifest(
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
    )
    trace = LensTraceInput.from_token_stream_manifest(
        trace_id="qwen4b-smoke-structural-trace",
        token_streams=streams,
        position_indices=positions,
        good_side_direction=1,
    )
    candidates = {
        "scope": "compiled_probe_candidates_for_structural_4b_smoke",
        "model": SMOKE_MODEL_ID,
        "revision": SMOKE_MODEL_REVISION,
    }
    candidate_hash = stable_hash(candidates)
    design = freeze_causal_probe_design(
        tokenizer,
        traces=(trace,),
        candidate_probe_manifest_hash=candidate_hash,
        candidate_probe_manifest_sha256=candidate_hash.removeprefix("sha256:"),
        anchor_manifest_hash=stable_hash({"smoke_anchor": True}),
        anchor_selection_hash=stable_hash({"selection": "synthetic_nonprimary"}),
        rollout_manifest_hash=stable_hash(streams),
        position_manifest_hash=stable_hash(positions),
        model_id=SMOKE_MODEL_ID,
        tokenizer_revision=SMOKE_MODEL_REVISION,
    )
    design_manifest = design.to_manifest(include_hash=True)
    expected_cells = len(POSITION_ORDER) * len(DEFAULT_CONCEPT_WORDS)
    if len(design.cells) != expected_cells:
        raise Qwen4BSmokeError("4B structural probe grid is incomplete")
    payload: dict[str, Any] = {
        "scope": "probe_design_and_grid_shape_only",
        "synthetic": True,
        "primary_eligible": False,
        "position_order": list(POSITION_ORDER),
        "concepts": sorted(DEFAULT_CONCEPT_WORDS),
        "probe_design_manifest_hash": design.manifest_hash,
        "probe_design_serialized_manifest_hash": design_manifest["manifest_hash"],
        "probe_cell_count": len(design.cells),
        "eligible_probe_cell_count": sum(cell.probe_eligible for cell in design.cells),
        "ineligible_probe_cell_count": sum(not cell.probe_eligible for cell in design.cells),
        "probe_cell_record_hashes_hash": stable_hash(
            [cell.record_hash for cell in design.cells]
        ),
        "transport_boundary": {
            "status": "not_executable_without_matched_4b_j_and_r_lenses",
            "vllm_generation_runtime_exposes_model_runtime_contract": False,
            "matched_4b_lens_artifact_count": 0,
            "activation_transport_executed": False,
            "fabricated_lens_record_count": 0,
            "would_require_lens_types": ["J", "R"],
            "would_require_layer_count_at_least": 1,
            "would_require_grid_record_count_at_least": 2 * expected_cells,
            "policy": "fail_boundary_explicitly_never_fabricate_observational_evidence",
        },
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def run_qwen4b_prefix_smoke(
    output_path: str | Path,
    *,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    rollout_max_tokens: int = 1024,
    continuation_max_tokens: int = 256,
) -> dict[str, Any]:
    """Run one rollout, two exact-prefix arms, and bounded local contracts.

    This is a compatibility gate, not an experimental sample. It never calls a
    paid API and cannot be pointed at the 122B checkpoint through arguments.
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
    token_map = CompletionTokenMap(
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
    retained_prefix = raw_prefix + forced_text
    messages = ({"role": "user", "content": rollout_request.prompt},)
    resample_registration = prefix_backend.register_generated_prefix(
        messages=messages,
        raw_completion_text=rollout_result.raw_text,
        original_prompt_token_ids=prompt_ids,
        original_completion_token_ids=completion_ids,
        raw_thinking_prefix=raw_prefix,
    )
    retain_registration = prefix_backend.register_generated_prefix(
        messages=messages,
        raw_completion_text=rollout_result.raw_text,
        original_prompt_token_ids=prompt_ids,
        original_completion_token_ids=completion_ids,
        raw_thinking_prefix=retained_prefix,
    )
    resample_prefix_ids = tuple(prefix_backend.encode_prefix(messages, raw_prefix))
    expected_resample_ids = prompt_ids + completion_ids[:prefix_count]
    if resample_prefix_ids != expected_resample_ids:
        raise Qwen4BSmokeError("registered resample prefix differs from original rollout IDs")
    forced_ids = tuple(prefix_backend.encode_continuation(forced_text))
    decoded_before = prefix_backend.tokenizer.decode(
        list(resample_prefix_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_after = prefix_backend.tokenizer.decode(
        [*resample_prefix_ids, *forced_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_after != decoded_before + forced_text:
        raise Qwen4BSmokeError("forced append changed the immutable raw prefix")
    retain_prefix_ids = tuple(prefix_backend.encode_prefix(messages, retained_prefix))
    expected_retain_ids = prompt_ids + completion_ids[:next_count]
    if retain_prefix_ids != expected_retain_ids:
        raise Qwen4BSmokeError("registered retain prefix differs from original rollout IDs")

    requests = (
        _continuation_request(
            rollout_request_id=rollout_request.request_id,
            messages=messages,
            arm="retain",
            conditioning_text=retained_prefix,
            prompt_token_ids=retain_prefix_ids,
            common_prefix_token_count=len(resample_prefix_ids),
        ),
        _continuation_request(
            rollout_request_id=rollout_request.request_id,
            messages=messages,
            arm="resample",
            conditioning_text=raw_prefix,
            prompt_token_ids=resample_prefix_ids,
            common_prefix_token_count=len(resample_prefix_ids),
        ),
    )
    results = prefix_backend.generate(requests)
    by_id = {result.request_id: result for result in results}
    if set(by_id) != {request.request_id for request in requests}:
        raise Qwen4BSmokeError("retain/resample continuation results are incomplete")
    continuation_rows = {
        request.arm: _continuation_evidence(
            by_id[request.request_id],
            expected_ids=request.prompt_token_ids,
        )
        for request in requests
    }

    answer_first = token_map.map_completion_span(
        sections.answer_char_start,
        sections.answer_char_start + 1,
        expected_text=rollout_result.raw_text[
            sections.answer_char_start : sections.answer_char_start + 1
        ],
        section="answer",
        section_char_start=0,
        section_char_end=1,
    )
    fixture = _fixture_evidence(prefix_backend.tokenizer)
    probe_grid = _probe_grid_evidence(
        tokenizer=prefix_backend.tokenizer,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        prefix_count=prefix_count,
        next_count=next_count,
        answer_first_token=answer_first.token_start,
    )

    rollout_row = materialize_rollout_rows(
        [rollout_request],
        [rollout_result],
        backend_provenance=rollout_backend.provenance,
    )[0]
    rollout_provenance = dict(rollout_backend.provenance)
    prefix_provenance = dict(prefix_backend.provenance)
    for provenance in (rollout_provenance, prefix_provenance):
        if (
            provenance.get("model_revision") != SMOKE_MODEL_REVISION
            or provenance.get("tokenizer_revision") != SMOKE_MODEL_REVISION
            or not isinstance(provenance.get("chat_template_hash"), str)
        ):
            raise Qwen4BSmokeError("4B model/tokenizer/chat-template provenance is incomplete")
    if rollout_provenance["chat_template_hash"] != prefix_provenance["chat_template_hash"]:
        raise Qwen4BSmokeError("rollout and prefix backends used different chat templates")

    handoff: dict[str, Any] = {
        "protocol_version": "nonprimary-analysis-evidence-handoff-v1",
        "status": "contract_exercised_not_research_evidence",
        "synthetic": True,
        "primary_eligible": False,
        "analysis_ingest_allowed": False,
        "behavioral_rollout_record_hash": rollout_row["record_hash"],
        "parser_trajectory_fixture_hash": fixture["record_hash"],
        "probe_grid_contract_hash": probe_grid["record_hash"],
        "required_primary_evidence_intentionally_absent": [
            "122b_model_identity",
            "paid_blind_final_adjudication",
            "selected_24_trace_anchor_manifest",
            "matched_122b_j_and_r_lens_transports",
        ],
    }
    handoff["record_hash"] = stable_hash(handoff)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol_version": SMOKE_PROTOCOL,
        "status": "passed",
        "scope": SMOKE_SCOPE,
        "experimental_sample": False,
        "primary_eligible": False,
        "synthetic_analysis_fixture": True,
        "paid_api_calls": 0,
        "bounds": {
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "rollout_count": 1,
            "rollout_max_tokens": rollout_max_tokens,
            "raw_prefix_continuation_count": 2,
            "continuation_max_tokens_each": continuation_max_tokens,
            "model_id_is_not_overridable": True,
        },
        "model": {"id": SMOKE_MODEL_ID, "revision": SMOKE_MODEL_REVISION},
        "tokenizer_chat_template": {
            "tokenizer_id": prefix_provenance["tokenizer_id"],
            "tokenizer_revision": prefix_provenance["tokenizer_revision"],
            "tokenizer_class": prefix_provenance["tokenizer_class"],
            "tokenizer_vocab_size": prefix_provenance["tokenizer_vocab_size"],
            "chat_template_hash": prefix_provenance["chat_template_hash"],
            "chat_template_revision": prefix_provenance["chat_template_revision"],
            "chat_template_kwargs": prefix_provenance["chat_template_kwargs"],
            "rollout_and_prefix_template_match": True,
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
            "primary_eligible": False,
        },
        "registered_prefixes": {
            "resample": {
                **resample_registration.as_dict(),
                "original_completion_boundary": prefix_count,
                "exact_original_ids_reused": resample_prefix_ids == expected_resample_ids,
            },
            "retain": {
                **retain_registration.as_dict(),
                "original_completion_boundary": next_count,
                "exact_original_ids_reused": retain_prefix_ids == expected_retain_ids,
            },
            "common_prefix_token_count": len(resample_prefix_ids),
        },
        # Retain the old singular aliases for an auditable schema migration.
        "registered_prefix": {
            **resample_registration.as_dict(),
            "original_completion_boundary": prefix_count,
            "exact_original_ids_reused": resample_prefix_ids == expected_resample_ids,
        },
        "forced_append_check": {
            "next_original_boundary": next_count,
            "forced_text_hash": stable_hash(forced_text),
            "forced_token_ids_hash": stable_hash(list(forced_ids)),
            "immutable_prefix_preserved": True,
        },
        "raw_prefix_continuations": continuation_rows,
        "raw_prefix_continuation": continuation_rows["resample"],
        "deterministic_local_fixture": fixture,
        "lens_probe_grid": probe_grid,
        "analysis_evidence_handoff": handoff,
        "backend_provenance": prefix_provenance,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(output_path, manifest)
    return manifest


__all__ = [
    "SMOKE_MODEL_ID",
    "SMOKE_MODEL_REVISION",
    "SMOKE_PROTOCOL",
    "SMOKE_SCOPE",
    "SMOKE_SEED",
    "Qwen4BSmokeError",
    "run_qwen4b_prefix_smoke",
]
