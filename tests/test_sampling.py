from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from model_forensics.sampling import (
    FakeBackend,
    GenerationResult,
    SamplingParameters,
    VLLMOfflineBackend,
    build_requests,
    materialize_rollout_rows,
    split_thinking_response,
)


def _prompt_builder(task: str, condition: str, threshold: float | None) -> str:
    return f"task={task}; condition={condition}; threshold={threshold}"


def test_build_requests_is_deterministic_unique_and_direction_aware() -> None:
    kwargs = dict(
        task="giraffe",
        condition="above_good",
        count=12,
        threshold=41_000_000,
        master_seed=7,
        prompt_builder=_prompt_builder,
        parameters=SamplingParameters(),
    )
    left = build_requests(**kwargs)
    right = build_requests(**kwargs)
    assert left == right
    assert len({request.request_id for request in left}) == 12
    assert len({request.seed for request in left}) == 12
    assert {request.direction for request in left} == {1}


def test_incentive_conditions_require_threshold() -> None:
    try:
        build_requests(
            task="giraffe",
            condition="below_good",
            count=1,
            threshold=None,
            master_seed=1,
            prompt_builder=_prompt_builder,
            parameters=SamplingParameters(),
        )
    except ValueError as exc:
        assert "requires a threshold" in str(exc)
    else:
        raise AssertionError("expected threshold validation")


def test_split_thinking_and_fake_backend_materialization() -> None:
    reasoning, answer = split_thinking_response("<think>work\n</think>\n42")
    assert reasoning == "work"
    assert answer == "42"
    completion_reasoning, completion_answer = split_thinking_response("work\n</think>\n42")
    assert completion_reasoning == "work"
    assert completion_answer == "42"
    requests = build_requests(
        task="coffee",
        condition="baseline",
        count=2,
        threshold=None,
        master_seed=5,
        prompt_builder=_prompt_builder,
        parameters=SamplingParameters(),
    )
    backend = FakeBackend()
    rows = materialize_rollout_rows(
        requests,
        backend.generate(requests),
        backend_provenance=backend.provenance,
    )
    assert len(rows) == 2
    assert all(row["record_hash"].startswith("sha256:") for row in rows)
    assert all(row["reasoning"] and row["answer"] for row in rows)


def test_vllm_constructor_forces_text_only_pinned_loading() -> None:
    observed: dict[str, object] = {}

    class FakeLLM:
        pass

    def factory(**kwargs):
        observed.update(kwargs)
        return FakeLLM()

    VLLMOfflineBackend(
        model_id="Qwen/Qwen3.5-122B-A10B",
        revision="a" * 40,
        tensor_parallel_size=8,
        max_model_len=65_536,
        llm_factory=factory,
    )
    assert observed["revision"] == "a" * 40
    assert observed["tokenizer"] == "Qwen/Qwen3.5-122B-A10B"
    assert observed["tokenizer_revision"] == "a" * 40
    assert observed["tensor_parallel_size"] == 8
    assert observed["language_model_only"] is True
    assert observed["trust_remote_code"] is False


def test_generation_result_requires_usage_to_match_exact_token_streams() -> None:
    result = GenerationResult(
        request_id="request",
        raw_text="answer",
        prompt_token_ids=[1, 2],  # type: ignore[arg-type]
        completion_token_ids=[3],  # type: ignore[arg-type]
    )
    assert result.prompt_token_ids == (1, 2)
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 1

    with pytest.raises(ValueError, match="disagrees"):
        GenerationResult(
            request_id="bad",
            raw_text="answer",
            prompt_tokens=3,
            prompt_token_ids=(1, 2),
        )


def test_vllm_results_retain_exact_ids_and_template_provenance(monkeypatch) -> None:
    captured_chat: dict[str, object] = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=FakeSamplingParams))

    class FakeTokenizer:
        vocab_size = 248_077

        def get_chat_template(self) -> str:
            return "{% if enable_thinking %}<think>{% endif %}"

    candidate = SimpleNamespace(
        text="reasoning\n</think>\n42",
        finish_reason="stop",
        token_ids=[300, 301, 302],
    )
    output = SimpleNamespace(
        outputs=[candidate],
        prompt_token_ids=[100, 101],
        request_id="engine-0",
    )

    class FakeLLM:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

        def chat(self, conversations, params, **kwargs):
            captured_chat.update({"conversations": conversations, "params": params, **kwargs})
            return [output]

    backend = VLLMOfflineBackend(
        model_id="Qwen/Qwen3.5-122B-A10B",
        revision="b" * 40,
        tensor_parallel_size=8,
        max_model_len=65_536,
        llm_factory=lambda **_: FakeLLM(),
    )
    request = build_requests(
        task="giraffe",
        condition="baseline",
        count=1,
        threshold=None,
        master_seed=7,
        prompt_builder=_prompt_builder,
        parameters=SamplingParameters(max_new_tokens=32),
        randomize=False,
    )[0]
    result = backend.generate([request])[0]
    rows = materialize_rollout_rows([request], [result], backend_provenance=backend.provenance)

    assert result.prompt_token_ids == (100, 101)
    assert result.completion_token_ids == (300, 301, 302)
    assert rows[0]["token_streams"]["prompt_token_ids"] == [100, 101]
    assert rows[0]["token_streams"]["completion_token_ids"] == [300, 301, 302]
    assert rows[0]["token_streams"]["combined_token_stream_hash"].startswith("sha256:")
    assert rows[0]["backend_result"]["engine_request_id"] == "engine-0"
    assert backend.provenance["model_revision"] == "b" * 40
    assert backend.provenance["tokenizer_revision"] == "b" * 40
    assert backend.provenance["chat_template_hash"].startswith("sha256:")
    assert backend.provenance["chat_template_kwargs"] == {"enable_thinking": True}
    assert backend.provenance["detokenization_kwargs"] == {
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
    }
    assert captured_chat["chat_template_kwargs"] == {"enable_thinking": True}
    sampling_kwargs = captured_chat["params"][0].kwargs  # type: ignore[index,union-attr]
    assert sampling_kwargs["skip_special_tokens"] is True
