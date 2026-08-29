"""Deterministic full-rollout generation with injectable local backends."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from model_forensics.io import stable_hash
from model_forensics.token_spans import locate_completion_sections, token_stream_manifest


@dataclass(frozen=True)
class SamplingParameters:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_new_tokens: int = 16384
    stop: tuple[str, ...] = ("<|im_end|>",)


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    task: str
    condition: str
    direction: int
    threshold: float | None
    seed: int
    prompt: str
    prompt_hash: str
    parameters: SamplingParameters
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    raw_text: str
    finish_reason: str = "stop"
    stop_reason: int | str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    completion_token_ids: tuple[int, ...] | None = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("prompt_token_ids", "completion_token_ids"):
            token_ids = getattr(self, field_name)
            if token_ids is None:
                continue
            normalized = tuple(token_ids)
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in normalized
            ):
                raise TypeError(f"{field_name} must contain only integers")
            if any(token_id < 0 for token_id in normalized):
                raise ValueError(f"{field_name} must contain only non-negative token IDs")
            object.__setattr__(self, field_name, normalized)

        for count_name, ids_name in (
            ("prompt_tokens", "prompt_token_ids"),
            ("completion_tokens", "completion_token_ids"),
        ):
            count = getattr(self, count_name)
            token_ids = getattr(self, ids_name)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError(f"{count_name} must be a non-negative integer or None")
            if token_ids is not None:
                if count is not None and count != len(token_ids):
                    raise ValueError(f"{count_name} disagrees with exact {ids_name}")
                if count is None:
                    object.__setattr__(self, count_name, len(token_ids))


class GenerationBackend(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...


PromptBuilder = Callable[[str, str, float | None], str]


def condition_direction(condition: str) -> int:
    if condition == "above_good":
        return 1
    if condition == "below_good":
        return -1
    if condition in {"baseline", "threshold_only"}:
        return 0
    raise ValueError(f"unknown condition: {condition}")


def build_requests(
    *,
    task: str,
    condition: str,
    count: int,
    threshold: float | None,
    master_seed: int,
    prompt_builder: PromptBuilder,
    parameters: SamplingParameters,
    randomize: bool = True,
) -> list[GenerationRequest]:
    """Build a unique, deterministic request manifest for one condition."""

    if count <= 0:
        raise ValueError("count must be positive")
    if condition not in {"baseline", "threshold_only"} and threshold is None:
        raise ValueError(f"{condition} requires a threshold")
    if condition == "threshold_only" and threshold is None:
        raise ValueError("threshold_only requires a threshold")

    prompt = prompt_builder(task, condition, threshold)
    prompt_hash = stable_hash({"task": task, "condition": condition, "prompt": prompt})
    direction = condition_direction(condition)
    rng = random.Random(stable_hash([master_seed, task, condition]))
    seeds: set[int] = set()
    while len(seeds) < count:
        seeds.add(rng.randrange(1, 2**31 - 1))

    requests = []
    for index, seed in enumerate(sorted(seeds)):
        identifier_payload = {
            "task": task,
            "condition": condition,
            "index": index,
            "seed": seed,
            "prompt_hash": prompt_hash,
        }
        requests.append(
            GenerationRequest(
                request_id=stable_hash(identifier_payload).split(":", 1)[1][:24],
                task=task,
                condition=condition,
                direction=direction,
                threshold=threshold,
                seed=seed,
                prompt=prompt,
                prompt_hash=prompt_hash,
                parameters=parameters,
            )
        )
    if randomize:
        random.Random(master_seed ^ 0x5F3759DF).shuffle(requests)
    return requests


def split_thinking_response(text: str) -> tuple[str, str]:
    """Separate Qwen's public ``<think>`` block from the visible answer."""

    sections = locate_completion_sections(text)
    return sections.reasoning, sections.answer


def materialize_rollout_rows(
    requests: Sequence[GenerationRequest],
    results: Sequence[GenerationResult],
    *,
    backend_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {result.request_id: result for result in results}
    if set(by_id) != {request.request_id for request in requests}:
        missing = {request.request_id for request in requests} - set(by_id)
        extra = set(by_id) - {request.request_id for request in requests}
        raise ValueError(
            f"request/result mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    rows: list[dict[str, Any]] = []
    for request in requests:
        result = by_id[request.request_id]
        sections = locate_completion_sections(result.raw_text)
        reasoning, answer = sections.reasoning, sections.answer
        row = {
            "run_id": request.request_id,
            "task": request.task,
            "condition": request.condition,
            "direction": request.direction,
            "threshold": request.threshold,
            "seed": request.seed,
            "prompt": request.prompt,
            "prompt_hash": request.prompt_hash,
            "sampling": asdict(request.parameters),
            "reasoning": reasoning,
            "answer": answer,
            "raw_text": result.raw_text,
            "finish_reason": result.finish_reason,
            "stop_reason": result.stop_reason,
            "completion_sections": sections.as_dict(),
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            "token_streams": token_stream_manifest(
                prompt_token_ids=result.prompt_token_ids,
                completion_token_ids=result.completion_token_ids,
            ),
            "backend": dict(backend_provenance),
            "backend_result": dict(result.backend_metadata),
        }
        row["record_hash"] = stable_hash(row)
        rows.append(row)
    return rows


class FakeBackend:
    """A no-network backend for deterministic pipeline smoke tests."""

    def __init__(self, response_factory: Callable[[GenerationRequest], str] | None = None) -> None:
        self._factory = response_factory or self._default_response

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"backend": "fake", "model_id": "deterministic-smoke", "revision": "local"}

    @staticmethod
    def _default_response(request: GenerationRequest) -> str:
        base = 1_000_000 + request.seed % 100_000
        return (
            "<think>\n"
            f"A first estimate is {base:,}. I will use the calculation rather than the threshold. "
            f"After checking one factor, I obtain {base + 10_000:,}.\n"
            "</think>\n\n"
            f"{base + 10_000:,}"
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [
            GenerationResult(
                request_id=request.request_id,
                raw_text=self._factory(request),
                completion_tokens=64,
                prompt_tokens=max(1, len(request.prompt.split())),
                backend_metadata={"seed": request.seed},
            )
            for request in requests
        ]


class VLLMOfflineBackend:
    """Thin optional adapter around ``vllm.LLM.chat``.

    Importing this module does not import vLLM. The large-model environment is
    responsible for installing a Qwen3.5-compatible vLLM revision and for
    passing a fully pinned model revision.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tensor_parallel_size: int,
        max_model_len: int,
        dtype: str = "bfloat16",
        llm_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not revision:
            raise ValueError("a pinned model revision is required")
        if llm_factory is None:
            try:
                from vllm import LLM
            except ImportError as exc:  # pragma: no cover - GPU-only path
                raise RuntimeError(
                    "install a Qwen3.5-compatible vLLM build on the GPU host"
                ) from exc
            llm_factory = LLM
        self.model_id = model_id
        self.revision = revision
        self.tokenizer_id = model_id
        self.tokenizer_revision = revision
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._engine_kwargs = {
            "trust_remote_code": False,
            "language_model_only": True,
        }
        self._chat_template_kwargs = {"enable_thinking": True}
        self._detokenization_kwargs = {
            "skip_special_tokens": True,
            "spaces_between_special_tokens": True,
        }
        self._llm = llm_factory(
            model=model_id,
            revision=revision,
            tokenizer=model_id,
            tokenizer_revision=revision,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            **self._engine_kwargs,
        )
        self._provenance = self._build_provenance()

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._provenance)

    def _build_provenance(self) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "backend": "vllm_offline",
            "model_id": self.model_id,
            "model_revision": self.revision,
            # Kept for compatibility with earlier manifests.
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "engine_kwargs": dict(self._engine_kwargs),
            "engine_kwargs_hash": stable_hash(self._engine_kwargs),
            "generation_api": "vllm.LLM.chat",
            "chat_template_kwargs": dict(self._chat_template_kwargs),
            "chat_template_kwargs_hash": stable_hash(self._chat_template_kwargs),
            "detokenization_kwargs": dict(self._detokenization_kwargs),
            "detokenization_kwargs_hash": stable_hash(self._detokenization_kwargs),
        }
        for package in ("vllm", "transformers"):
            try:
                provenance[f"{package}_version"] = version(package)
            except PackageNotFoundError:
                continue

        get_tokenizer = getattr(self._llm, "get_tokenizer", None)
        if not callable(get_tokenizer):
            return provenance
        try:
            tokenizer = get_tokenizer()
        except Exception:  # pragma: no cover - defensive against backend internals
            return provenance

        provenance["tokenizer_class"] = (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        )
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int) and not isinstance(vocab_size, bool):
            provenance["tokenizer_vocab_size"] = vocab_size

        chat_template: Any = None
        get_chat_template = getattr(tokenizer, "get_chat_template", None)
        if callable(get_chat_template):
            try:
                chat_template = get_chat_template()
            except Exception:  # pragma: no cover - tokenizer-version dependent
                chat_template = None
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, (str, list, dict)):
            provenance["chat_template_hash"] = stable_hash({"chat_template": chat_template})
            provenance["chat_template_revision"] = self.tokenizer_revision
        return provenance

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        try:
            from vllm import SamplingParams
        except ImportError as exc:  # pragma: no cover - GPU-only path
            raise RuntimeError("vLLM is unavailable") from exc

        conversations = [[{"role": "user", "content": request.prompt}] for request in requests]
        params = [
            SamplingParams(
                temperature=request.parameters.temperature,
                top_p=request.parameters.top_p,
                top_k=request.parameters.top_k,
                min_p=request.parameters.min_p,
                presence_penalty=request.parameters.presence_penalty,
                repetition_penalty=request.parameters.repetition_penalty,
                max_tokens=request.parameters.max_new_tokens,
                stop=list(request.parameters.stop),
                seed=request.seed,
                **self._detokenization_kwargs,
            )
            for request in requests
        ]
        outputs = self._llm.chat(
            conversations,
            params,
            use_tqdm=True,
            chat_template_kwargs=dict(self._chat_template_kwargs),
        )
        results: list[GenerationResult] = []
        for request, output in zip(requests, outputs, strict=True):
            candidate = output.outputs[0]
            if output.prompt_token_ids is None:
                raise RuntimeError("vLLM did not return the exact rendered prompt token IDs")
            if candidate.token_ids is None:
                raise RuntimeError("vLLM did not return the exact completion token IDs")
            prompt_token_ids = tuple(output.prompt_token_ids)
            completion_token_ids = tuple(candidate.token_ids)
            if not prompt_token_ids:
                raise RuntimeError("vLLM returned an empty rendered prompt token stream")
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in (*prompt_token_ids, *completion_token_ids)
            ):
                raise RuntimeError("vLLM returned non-integer token IDs")
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    raw_text=candidate.text,
                    finish_reason=str(candidate.finish_reason),
                    stop_reason=getattr(candidate, "stop_reason", None),
                    prompt_tokens=len(prompt_token_ids),
                    completion_tokens=len(completion_token_ids),
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    backend_metadata={
                        "engine_request_id": str(output.request_id),
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "prompt_token_ids_hash": token_stream_manifest(
                            prompt_token_ids=prompt_token_ids,
                            completion_token_ids=None,
                        )["prompt_token_ids_hash"],
                        "completion_token_ids_hash": token_stream_manifest(
                            prompt_token_ids=None,
                            completion_token_ids=completion_token_ids,
                        )["completion_token_ids_hash"],
                    },
                )
            )
        return results


def batched(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
