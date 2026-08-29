from __future__ import annotations

import pytest

from model_forensics.lens import DEFAULT_CONCEPT_WORDS
from model_forensics.lens_command import LensTraceInput, ValidatedLensInputs
from model_forensics.lens_production import (
    FROZEN_4B_COMPATIBILITY_TEXT,
    encode_frozen_4b_compatibility_prefix,
    freeze_production_compatibility_prefixes,
    freeze_production_probe_design,
)
from model_forensics.lens_runner import FROZEN_PROBE_TOKEN_IDS
from model_forensics.token_spans import token_stream_manifest


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == FROZEN_4B_COMPATIBILITY_TEXT
        assert add_special_tokens is True
        return [7, 8, 9]


def _trace(trace_id: str, completion: list[int]) -> LensTraceInput:
    streams = token_stream_manifest(prompt_token_ids=[1, 2], completion_token_ids=completion)
    return LensTraceInput.from_token_stream_manifest(
        trace_id=trace_id,
        token_streams=streams,
        position_indices={
            "prompt_end": 1,
            "first_estimate_pre": 2,
            "anchor_pre": 3,
            "anchor_post": 4,
            "final_answer_pre": 5,
        },
        good_side_direction=1,
    )


def test_frozen_4b_prefix_uses_exact_model_revision_arguments() -> None:
    observed = {}

    def factory(*args, **kwargs):
        observed.update(args=args, kwargs=kwargs)
        return Tokenizer()

    assert encode_frozen_4b_compatibility_prefix(tokenizer_factory=factory) == (7, 8, 9)
    assert observed["kwargs"]["trust_remote_code"] is False
    assert len(observed["kwargs"]["revision"]) == 40


def test_primary_prefix_choice_is_manifest_ordered_and_shortened() -> None:
    first = _trace("first", [3, 4, 5, 6, 999])
    second = _trace("second", [3, 4, 5, 6, 7])
    validated = ValidatedLensInputs(
        traces=(first, second),
        anchor_manifest_hash="sha256:" + "a" * 64,
        anchor_selection_hash="b" * 64,
        position_manifest_hash="sha256:" + "c" * 64,
        rollout_manifest_hash="sha256:" + "d" * 64,
    )
    prefixes = freeze_production_compatibility_prefixes(
        validated,
        four_b_token_ids=[7, 8, 9],
        shortened_primary_limit=2,
    )
    assert prefixes.primary_trace_id == "first"
    assert prefixes.primary_full_token_ids == first.sequence_token_ids[:6]
    assert prefixes.primary_short_token_ids == first.sequence_token_ids[:2]


def test_production_probe_design_uses_only_the_pinned_primary_tokenizer() -> None:
    trace = _trace("first", [3, 4, 5, 6])
    validated = ValidatedLensInputs(
        traces=(trace,),
        anchor_manifest_hash="sha256:" + "a" * 64,
        anchor_selection_hash="sha256:" + "b" * 64,
        position_manifest_hash="sha256:" + "c" * 64,
        rollout_manifest_hash="sha256:" + "d" * 64,
    )
    encodings = {
        word: [token_id]
        for concept, polarities in DEFAULT_CONCEPT_WORDS.items()
        for polarity, words in polarities.items()
        for word, token_id in zip(
            words, FROZEN_PROBE_TOKEN_IDS[concept][polarity], strict=True
        )
    }

    class PrimaryTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return encodings[text]

        def decode(self, token_ids: list[int], **kwargs) -> str:
            del token_ids, kwargs
            return "neutral arithmetic"

    observed = {}

    def factory(*args, **kwargs):
        observed.update(args=args, kwargs=kwargs)
        return PrimaryTokenizer()

    design = freeze_production_probe_design(
        validated,
        candidate_probe_manifest_hash="sha256:" + "e" * 64,
        candidate_probe_manifest_sha256="f" * 64,
        tokenizer_factory=factory,
    )
    assert design.to_manifest()["cell_count"] == 15
    assert observed["args"] == ("Qwen/Qwen3.5-122B-A10B",)
    assert observed["kwargs"]["revision"] == design.tokenizer_revision
    assert observed["kwargs"]["trust_remote_code"] is False


def test_production_factories_reject_non_eight_gpu_primary() -> None:
    from model_forensics.lens_production import production_runtime_factories

    with pytest.raises(ValueError, match="exactly eight"):
        production_runtime_factories(lens_cache_dir="cache", primary_cuda_devices=4)
