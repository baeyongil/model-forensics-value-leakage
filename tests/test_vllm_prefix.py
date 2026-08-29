from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from model_forensics.lens import DEFAULT_CONCEPT_WORDS
from model_forensics.lens_runner import FROZEN_PROBE_TOKEN_IDS
from model_forensics.qwen4b_smoke import (
    SMOKE_MODEL_ID,
    SMOKE_MODEL_REVISION,
    _fixture_evidence,
    _probe_grid_evidence,
    run_qwen4b_prefix_smoke,
)
from model_forensics.resample_runner import RawPrefixGenerationRequest
from model_forensics.sampling import SamplingParameters
from model_forensics.vllm_prefix import (
    PrefixRegistrationError,
    PrefixTokenIdentityError,
    VLLMPrefixBackendError,
    VLLMRawPrefixBackend,
)

REVISION = "a" * 40
MESSAGES = ({"role": "user", "content": "Q"},)
RENDERED = "CHAT[user:Q]<think>\n"


class CharacterTokenizer:
    vocab_size = 256

    def __init__(self, *, bad_suffix: bool = False) -> None:
        self.bad_suffix = bad_suffix

    def get_chat_template(self) -> str:
        return "frozen-qwen-template"

    def apply_chat_template(self, messages, **kwargs) -> str:
        assert messages == [dict(MESSAGES[0])]
        assert kwargs == {
            "chat_template": "frozen-qwen-template",
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        return "CHAT[user:Q]assistant\n" if self.bad_suffix else RENDERED

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


class FakeSamplingParams:
    instances: ClassVar[list[FakeSamplingParams]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class FakeTokensPrompt(dict):
    pass


class FakeLLM:
    def __init__(self, tokenizer: CharacterTokenizer, *, corrupt_prompt: bool = False) -> None:
        self.tokenizer = tokenizer
        self.corrupt_prompt = corrupt_prompt
        self.generate_calls: list[dict[str, object]] = []

    def get_tokenizer(self) -> CharacterTokenizer:
        return self.tokenizer

    def generate(self, prompts, sampling_params, **kwargs):
        self.generate_calls.append(
            {"prompts": prompts, "sampling_params": sampling_params, **kwargs}
        )
        outputs = []
        generated = " tail</think>42"
        for index, prompt in enumerate(prompts):
            prompt_ids = tuple(prompt["prompt_token_ids"])
            if self.corrupt_prompt:
                prompt_ids = (*prompt_ids, 999)
            candidate = SimpleNamespace(
                text=generated,
                token_ids=tuple(ord(character) for character in generated),
                finish_reason="stop",
                stop_reason=248046,
            )
            outputs.append(
                SimpleNamespace(
                    request_id=f"engine-{index}",
                    prompt_token_ids=prompt_ids,
                    outputs=[candidate],
                )
            )
        return outputs


def _backend(
    *,
    tokenizer: CharacterTokenizer | None = None,
    llm: FakeLLM | None = None,
    require_registered_prefixes: bool = True,
) -> VLLMRawPrefixBackend:
    tokenizer = tokenizer or CharacterTokenizer()
    llm = llm or FakeLLM(tokenizer)
    FakeSamplingParams.instances.clear()
    return VLLMRawPrefixBackend(
        model_id="Qwen/Qwen3.5-4B",
        revision=REVISION,
        tensor_parallel_size=1,
        max_model_len=4096,
        parameters=SamplingParameters(max_new_tokens=64),
        require_registered_prefixes=require_registered_prefixes,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params_factory=FakeSamplingParams,
        tokens_prompt_factory=FakeTokensPrompt,
        use_tqdm=False,
    )


def _register(backend: VLLMRawPrefixBackend, raw_prefix: str = "Alpha "):
    raw_completion = "Alpha beta.</think>42"
    return backend.register_generated_prefix(
        messages=MESSAGES,
        raw_completion_text=raw_completion,
        original_prompt_token_ids=tuple(ord(character) for character in RENDERED),
        original_completion_token_ids=tuple(ord(character) for character in raw_completion),
        raw_thinking_prefix=raw_prefix,
    )


def test_registered_prefix_reuses_original_ids_and_forced_append_never_retokenizes() -> None:
    backend = _backend()
    registration = _register(backend)
    exact_prefix = tuple(backend.encode_prefix(MESSAGES, "Alpha "))
    expected = tuple(ord(character) for character in RENDERED + "Alpha ")

    assert exact_prefix == expected
    assert registration.completion_prefix_token_count == len("Alpha ")
    assert registration.exact_prefix_token_ids_hash.startswith("sha256:")
    forced = tuple(backend.encode_continuation("beta."))
    assert forced == tuple(ord(character) for character in "beta.")
    assert (
        backend.tokenizer.decode(
            [*exact_prefix, *forced],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        == RENDERED + "Alpha beta."
    )

    # Later runner calls measure generated replacement length without modifying
    # the already-consumed intervention prefix.
    assert tuple(backend.encode_continuation("replacement.")) == tuple(
        ord(character) for character in "replacement."
    )
    assert backend.provenance["registered_prefix_count"] == 1
    assert backend.provenance["text_reencoded_prefix_count"] == 0


def test_unregistered_prefix_fails_closed_unless_smoke_fallback_is_explicit() -> None:
    backend = _backend()
    with pytest.raises(PrefixRegistrationError, match="not registered"):
        backend.encode_prefix(MESSAGES, "Alpha ")

    fallback = _backend(require_registered_prefixes=False)
    assert tuple(fallback.encode_prefix(MESSAGES, "Alpha ")) == tuple(
        ord(character) for character in RENDERED + "Alpha "
    )
    assert fallback.provenance["text_reencoded_prefix_count"] == 1
    assert fallback.provenance["synthetic_smoke"] is True


def test_registration_rejects_non_boundary_and_wrong_template_prompt_ids() -> None:
    backend = _backend()
    with pytest.raises(PrefixRegistrationError, match="rendered chat template"):
        backend.register_generated_prefix(
            messages=MESSAGES,
            raw_completion_text="Alpha beta.</think>42",
            original_prompt_token_ids=(1, 2),
            original_completion_token_ids=tuple(
                ord(character) for character in "Alpha beta.</think>42"
            ),
            raw_thinking_prefix="Alpha ",
        )

    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            # Keep the rendered prompt character-wise but merge "ab" in the
            # completion, making the requested "a" boundary impossible.
            if text == RENDERED:
                return super().encode(text, add_special_tokens=add_special_tokens)
            if text == "ab</think>42":
                return [900, *super().encode("</think>42", add_special_tokens=False)]
            return super().encode(text, add_special_tokens=add_special_tokens)

        def decode(self, token_ids: list[int], **kwargs) -> str:
            pieces = []
            for token_id in token_ids:
                pieces.append("ab" if token_id == 900 else chr(token_id))
            return "".join(pieces)

    tokenizer = MergingTokenizer()
    backend = _backend(tokenizer=tokenizer, llm=FakeLLM(tokenizer))
    with pytest.raises(PrefixRegistrationError, match="inside an original completion token"):
        backend.register_generated_prefix(
            messages=MESSAGES,
            raw_completion_text="ab</think>42",
            original_prompt_token_ids=tuple(ord(character) for character in RENDERED),
            original_completion_token_ids=(
                900,
                *(ord(character) for character in "</think>42"),
            ),
            raw_thinking_prefix="a",
        )


def test_generate_uses_tokens_prompt_and_preserves_exact_ids_seed_and_sampling() -> None:
    tokenizer = CharacterTokenizer()
    llm = FakeLLM(tokenizer)
    backend = _backend(tokenizer=tokenizer, llm=llm)
    _register(backend)
    exact_prefix = tuple(backend.encode_prefix(MESSAGES, "Alpha "))
    backend.encode_continuation("beta.")
    request = RawPrefixGenerationRequest(
        request_id="request-1",
        anchor_id="anchor-1",
        base_trace_id="trace-1",
        arm="resample",
        sample_index=0,
        seed=20260829,
        messages=MESSAGES,
        conditioning_text="Alpha ",
        prompt_token_ids=exact_prefix,
        common_prefix_token_count=len(exact_prefix),
    )
    result = backend.generate([request])[0]

    call = llm.generate_calls[0]
    assert isinstance(call["prompts"][0], FakeTokensPrompt)
    assert call["prompts"][0] == {"prompt_token_ids": list(exact_prefix)}
    params = call["sampling_params"][0].kwargs
    assert params["seed"] == 20260829
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["top_k"] == 20
    assert params["presence_penalty"] == 1.5
    assert params["max_tokens"] == 64
    assert call["use_tqdm"] is False

    assert result.prompt_token_ids == exact_prefix
    assert result.prompt_tokens == len(exact_prefix)
    assert result.completion_tokens == len(" tail</think>42")
    assert result.backend_metadata["prompt_token_ids"] == list(exact_prefix)
    assert result.backend_metadata["completion_token_ids"] == [
        ord(character) for character in " tail</think>42"
    ]
    assert result.backend_metadata["seed"] == 20260829
    assert result.backend_metadata["combined_token_stream_hash"].startswith("sha256:")
    assert backend.provenance["model_revision"] == REVISION
    assert backend.provenance["chat_template_hash"].startswith("sha256:")
    assert "secret" not in repr(backend.provenance).lower()


def test_generate_rejects_consumed_prompt_identity_break() -> None:
    tokenizer = CharacterTokenizer()
    backend = _backend(tokenizer=tokenizer, llm=FakeLLM(tokenizer, corrupt_prompt=True))
    _register(backend)
    exact_prefix = tuple(backend.encode_prefix(MESSAGES, "Alpha "))
    backend.encode_continuation("beta.")
    request = RawPrefixGenerationRequest(
        request_id="request-1",
        anchor_id="anchor-1",
        base_trace_id="trace-1",
        arm="retain",
        sample_index=0,
        seed=7,
        messages=MESSAGES,
        conditioning_text="Alpha beta.",
        prompt_token_ids=exact_prefix,
        common_prefix_token_count=len(exact_prefix),
    )
    with pytest.raises(PrefixTokenIdentityError, match="consumed different prefix"):
        backend.generate([request])


def test_bad_chat_template_and_unpinned_revision_are_rejected() -> None:
    with pytest.raises(ValueError, match="pinned"):
        VLLMRawPrefixBackend(
            model_id="Qwen/Qwen3.5-4B",
            revision="main",
            tensor_parallel_size=1,
            max_model_len=4096,
            llm=FakeLLM(CharacterTokenizer()),
        )

    tokenizer = CharacterTokenizer(bad_suffix=True)
    backend = _backend(tokenizer=tokenizer, llm=FakeLLM(tokenizer))
    with pytest.raises(VLLMPrefixBackendError, match="expected <think>"):
        backend.encode_prefix(MESSAGES, "Alpha ")


def test_real_4b_smoke_is_fixed_and_rejects_unbounded_work_before_gpu_setup(tmp_path) -> None:
    assert SMOKE_MODEL_ID == "Qwen/Qwen3.5-4B"
    assert SMOKE_MODEL_REVISION == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    with pytest.raises(ValueError, match="rollout_max_tokens"):
        run_qwen4b_prefix_smoke(
            tmp_path / "must-not-exist.json",
            rollout_max_tokens=4096,
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_nonprimary_fixture_exercises_parser_trajectory_and_exact_anchor_mapping() -> None:
    evidence = _fixture_evidence(CharacterTokenizer())

    assert evidence["synthetic"] is True
    assert evidence["primary_eligible"] is False
    assert evidence["trajectory"]["features"]["first_estimate"] == 36_000_000
    assert evidence["trajectory"]["features"]["final_estimate"] == 42_000_000
    assert evidence["anchor"]["token_span"]["round_trip_verified"] is True


class ProbeTokenizer(CharacterTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.probes = {
            word: FROZEN_PROBE_TOKEN_IDS[concept][polarity][index]
            for concept, polarities in DEFAULT_CONCEPT_WORDS.items()
            for polarity, words in polarities.items()
            for index, word in enumerate(words)
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text in self.probes:
            return [self.probes[text]]
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_4b_probe_grid_exercises_full_shape_and_states_transport_boundary() -> None:
    evidence = _probe_grid_evidence(
        tokenizer=ProbeTokenizer(),
        prompt_ids=(ord("p"), ord("q")),
        completion_ids=tuple(ord(character) for character in "abcdef"),
        prefix_count=2,
        next_count=3,
        answer_first_token=5,
    )

    assert evidence["probe_cell_count"] == 15
    assert evidence["primary_eligible"] is False
    boundary = evidence["transport_boundary"]
    assert boundary["activation_transport_executed"] is False
    assert boundary["fabricated_lens_record_count"] == 0
    assert boundary["status"] == "not_executable_without_matched_4b_j_and_r_lenses"
