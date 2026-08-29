"""Production vLLM adapter for exact-token sentence-resampling prefixes.

The module imports neither vLLM nor Torch at import time.  A GPU process may
inject an already-loaded ``vllm.LLM`` instance, allowing behavioral sampling and
raw-prefix continuation to share one model allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from model_forensics.io import stable_hash
from model_forensics.resample_runner import (
    RawPrefixGenerationRequest,
    RawPrefixGenerationResult,
)
from model_forensics.sampling import SamplingParameters
from model_forensics.token_spans import CompletionTokenMap, token_stream_manifest

QWEN_THINKING_PROMPT_SUFFIX = "<think>\n"


class VLLMPrefixBackendError(RuntimeError):
    """A vLLM or tokenizer invariant needed for exact intervention failed."""


class PrefixRegistrationError(VLLMPrefixBackendError):
    """An original generated prefix cannot be authenticated at a token boundary."""


class PrefixTokenIdentityError(VLLMPrefixBackendError):
    """A forced append or vLLM generation changed an immutable prompt prefix."""


def _commit_revision(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.lower()
    if len(normalized) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a pinned 40- or 64-character commit hash")
    return normalized


def _token_ids(values: Sequence[int], *, name: str, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of token IDs")
    normalized = tuple(values)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized):
        raise TypeError(f"{name} must contain only integers")
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} must contain only nonnegative IDs")
    return normalized


def _messages(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        if set(message) != {"role", "content"}:
            raise ValueError("text-only prefix messages require exactly role and content")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"message {index} role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string")
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


def _decode(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise VLLMPrefixBackendError("tokenizer does not expose decode")
    try:
        text = decoder(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise VLLMPrefixBackendError(f"tokenizer decode failed: {exc}") from exc
    if not isinstance(text, str):
        raise VLLMPrefixBackendError("tokenizer.decode returned a non-string")
    return text


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise VLLMPrefixBackendError("tokenizer does not expose encode")
    try:
        encoded = encoder(text, add_special_tokens=False)
    except Exception as exc:
        raise VLLMPrefixBackendError(f"tokenizer encode failed: {exc}") from exc
    return _token_ids(encoded, name="encoded token stream", allow_empty=(text == ""))


def _resolve_chat_template(tokenizer: Any) -> str:
    template: Any = None
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        try:
            template = getter()
        except Exception as exc:
            raise VLLMPrefixBackendError(
                f"could not resolve tokenizer chat template: {exc}"
            ) from exc
    if template is None:
        template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise VLLMPrefixBackendError("tokenizer has no concrete chat template")
    return template


@dataclass(frozen=True, slots=True)
class ExactPrefixRegistration:
    """Secret-free evidence linking a text prefix to original generation IDs."""

    registration_key: str
    registration_hash: str
    messages_hash: str
    raw_prefix_hash: str
    original_prompt_token_ids_hash: str
    original_completion_token_ids_hash: str
    exact_prefix_token_ids_hash: str
    prompt_token_count: int
    completion_prefix_token_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLLMRawPrefixBackend:
    """Exact-token ``RawPrefixGenerationBackend`` for pinned Qwen checkpoints.

    Production use defaults to ``require_registered_prefixes=True``.  Call
    :meth:`register_generated_prefix` with the original full-rollout token IDs
    before requesting an anchor prefix.  This prevents decoded model output from
    being silently retokenized.  An explicit fallback is available only for
    bounded compatibility smoke tests.

    ``encode_prefix`` arms exactly one subsequent ``encode_continuation`` call.
    The latter appends newly encoded tokens and verifies by contextual decoding
    that the immutable prefix was not changed.  Later calls are standalone token-
    length measurements, matching ``run_sentence_resampling`` call order.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tensor_parallel_size: int,
        max_model_len: int,
        dtype: str = "bfloat16",
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        parameters: SamplingParameters | None = None,
        require_registered_prefixes: bool = True,
        llm: Any | None = None,
        tokenizer: Any | None = None,
        llm_factory: Callable[..., Any] | None = None,
        sampling_params_factory: Callable[..., Any] | None = None,
        tokens_prompt_factory: Callable[..., Any] | None = None,
        use_tqdm: bool = True,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be non-empty")
        if isinstance(tensor_parallel_size, bool) or tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if isinstance(max_model_len, bool) or max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if type(require_registered_prefixes) is not bool or type(use_tqdm) is not bool:
            raise TypeError("prefix policy and tqdm flags must be bools")

        self.model_id = model_id
        self.revision = _commit_revision(revision, name="model revision")
        self.tokenizer_id = tokenizer_id or model_id
        self.tokenizer_revision = _commit_revision(
            tokenizer_revision or revision,
            name="tokenizer revision",
        )
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.parameters = parameters or SamplingParameters()
        self.require_registered_prefixes = require_registered_prefixes
        self._use_tqdm = use_tqdm
        self._sampling_params_factory = sampling_params_factory
        self._tokens_prompt_factory = tokens_prompt_factory
        self._reencoded_prefix_count = 0
        self._registrations: dict[str, tuple[tuple[int, ...], ExactPrefixRegistration]] = {}
        self._pending_append_prefix: tuple[int, ...] | None = None

        injected_llm = llm is not None
        if llm is not None and llm_factory is not None:
            raise ValueError("pass either llm or llm_factory, not both")
        if llm is None:
            if llm_factory is None:
                try:
                    from vllm import LLM
                except ImportError as exc:  # pragma: no cover - GPU-only path
                    raise RuntimeError("install the pinned Qwen-compatible vLLM build") from exc
                llm_factory = LLM
            llm = llm_factory(
                model=model_id,
                revision=self.revision,
                tokenizer=self.tokenizer_id,
                tokenizer_revision=self.tokenizer_revision,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                dtype=dtype,
                trust_remote_code=False,
                language_model_only=True,
            )
        self._llm = llm
        self._llm_reused = injected_llm

        if tokenizer is None:
            get_tokenizer = getattr(self._llm, "get_tokenizer", None)
            if not callable(get_tokenizer):
                raise VLLMPrefixBackendError("LLM does not expose get_tokenizer")
            tokenizer = get_tokenizer()
        self._tokenizer = tokenizer
        self._chat_template = _resolve_chat_template(tokenizer)
        self._chat_template_hash = stable_hash({"chat_template": self._chat_template})

    @property
    def llm(self) -> Any:
        """Return the loaded instance so a full-rollout backend can reuse it."""

        return self._llm

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def provenance(self) -> Mapping[str, Any]:
        packages: dict[str, str] = {}
        for package in ("vllm", "transformers"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                continue
        vocab_size = getattr(self._tokenizer, "vocab_size", None)
        return {
            "backend": "vllm_raw_prefix",
            "backend_version": "exact-prefix-v1",
            "model_id": self.model_id,
            "model_revision": self.revision,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "language_model_only": True,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_class": (
                f"{type(self._tokenizer).__module__}.{type(self._tokenizer).__qualname__}"
            ),
            "tokenizer_vocab_size": (
                vocab_size
                if isinstance(vocab_size, int) and not isinstance(vocab_size, bool)
                else None
            ),
            "chat_template_hash": self._chat_template_hash,
            "chat_template_revision": self.tokenizer_revision,
            "chat_template_kwargs": {
                "add_generation_prompt": True,
                "enable_thinking": True,
                "tokenize": False,
            },
            "sampling": asdict(self.parameters),
            "seed_policy": "request_manifest_seed",
            "detokenization": {
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
            "require_registered_prefixes": self.require_registered_prefixes,
            "synthetic_smoke": not self.require_registered_prefixes,
            "registered_prefix_count": len(self._registrations),
            "text_reencoded_prefix_count": self._reencoded_prefix_count,
            "llm_reused": self._llm_reused,
            "packages": packages,
        }

    def _render_prompt(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[dict[str, str], ...], str]:
        normalized = _messages(messages)
        renderer = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(renderer):
            raise VLLMPrefixBackendError("tokenizer does not expose apply_chat_template")
        try:
            rendered = renderer(
                list(normalized),
                chat_template=self._chat_template,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception as exc:
            raise VLLMPrefixBackendError(f"chat-template rendering failed: {exc}") from exc
        if not isinstance(rendered, str):
            raise VLLMPrefixBackendError("chat template returned non-text output")
        if not rendered.endswith(QWEN_THINKING_PROMPT_SUFFIX):
            raise VLLMPrefixBackendError(
                "Qwen chat template did not end inside the expected <think> block"
            )
        return normalized, rendered

    def _registration_key(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> tuple[str, tuple[dict[str, str], ...], str]:
        if not isinstance(raw_thinking_prefix, str):
            raise TypeError("raw_thinking_prefix must be a string")
        normalized, rendered = self._render_prompt(messages)
        key = stable_hash(
            {
                "messages": list(normalized),
                "raw_thinking_prefix": raw_thinking_prefix,
                "chat_template_hash": self._chat_template_hash,
                "model_revision": self.revision,
                "tokenizer_revision": self.tokenizer_revision,
            }
        )
        return key, normalized, rendered

    def register_generated_prefix(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        raw_completion_text: str,
        original_prompt_token_ids: Sequence[int],
        original_completion_token_ids: Sequence[int],
        raw_thinking_prefix: str,
    ) -> ExactPrefixRegistration:
        """Authenticate one prefix directly from a full rollout's original IDs."""

        if not isinstance(raw_completion_text, str):
            raise TypeError("raw_completion_text must be a string")
        if not raw_completion_text.startswith(raw_thinking_prefix):
            raise PrefixRegistrationError(
                "raw thinking prefix is not an exact prefix of the generated completion"
            )
        key, normalized_messages, rendered = self._registration_key(
            messages,
            raw_thinking_prefix,
        )
        prompt_ids = _token_ids(original_prompt_token_ids, name="original prompt token IDs")
        completion_ids = _token_ids(
            original_completion_token_ids,
            name="original completion token IDs",
            allow_empty=True,
        )
        rendered_ids = _encode(self._tokenizer, rendered)
        if rendered_ids != prompt_ids:
            raise PrefixRegistrationError(
                "original prompt IDs do not match the pinned rendered chat template"
            )

        mapper = CompletionTokenMap(
            tokenizer=self._tokenizer,
            raw_text=raw_completion_text,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            skip_special_tokens=True,
        )
        if raw_thinking_prefix:
            span = mapper.map_completion_span(
                0,
                len(raw_thinking_prefix),
                expected_text=raw_thinking_prefix,
            )
            if span.token_envelope_char_end != len(raw_thinking_prefix):
                raise PrefixRegistrationError(
                    "requested raw prefix ends inside an original completion token"
                )
            completion_prefix_count = span.token_end
        else:
            completion_prefix_count = 0
        exact_ids = prompt_ids + completion_ids[:completion_prefix_count]
        decoded_prompt = _decode(self._tokenizer, prompt_ids, skip_special_tokens=True)
        decoded_exact = _decode(self._tokenizer, exact_ids, skip_special_tokens=True)
        if not decoded_exact.startswith(decoded_prompt):
            raise PrefixRegistrationError("completion tokens changed the decoded prompt prefix")
        if decoded_exact[len(decoded_prompt) :] != raw_thinking_prefix:
            raise PrefixRegistrationError("registered prefix failed exact contextual decoding")

        streams = token_stream_manifest(
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )
        exact_hash = token_stream_manifest(
            prompt_token_ids=exact_ids,
            completion_token_ids=None,
        )["prompt_token_ids_hash"]
        payload = {
            "registration_key": key,
            "messages_hash": stable_hash(list(normalized_messages)),
            "raw_prefix_hash": stable_hash(raw_thinking_prefix),
            "original_prompt_token_ids_hash": streams["prompt_token_ids_hash"],
            "original_completion_token_ids_hash": streams["completion_token_ids_hash"],
            "exact_prefix_token_ids_hash": exact_hash,
            "prompt_token_count": len(prompt_ids),
            "completion_prefix_token_count": completion_prefix_count,
        }
        registration = ExactPrefixRegistration(
            **payload,
            registration_hash=stable_hash(payload),
        )
        existing = self._registrations.get(key)
        if existing is not None and existing[0] != exact_ids:
            raise PrefixRegistrationError("registration key already maps to different token IDs")
        self._registrations[key] = (exact_ids, registration)
        return registration

    def encode_prefix(
        self,
        messages: Sequence[Mapping[str, Any]],
        raw_thinking_prefix: str,
    ) -> Sequence[int]:
        key, _, rendered = self._registration_key(messages, raw_thinking_prefix)
        registered = self._registrations.get(key)
        if registered is not None:
            token_ids = registered[0]
        else:
            if self.require_registered_prefixes:
                raise PrefixRegistrationError(
                    "prefix was not registered from original full-rollout token IDs"
                )
            token_ids = _encode(self._tokenizer, rendered + raw_thinking_prefix)
            if _decode(self._tokenizer, token_ids, skip_special_tokens=False) != (
                rendered + raw_thinking_prefix
            ):
                raise PrefixRegistrationError("text-reencoded prefix failed exact round-trip")
            self._reencoded_prefix_count += 1
        self._pending_append_prefix = token_ids
        return token_ids

    def encode_continuation(self, raw_text: str) -> Sequence[int]:
        if not isinstance(raw_text, str):
            raise TypeError("raw continuation must be a string")
        if not raw_text:
            raise ValueError("raw continuation must be non-empty")
        continuation_ids = _encode(self._tokenizer, raw_text)
        if self._pending_append_prefix is None:
            if _decode(self._tokenizer, continuation_ids, skip_special_tokens=False) != raw_text:
                raise PrefixTokenIdentityError("standalone continuation failed exact round-trip")
            return continuation_ids

        prefix_ids = self._pending_append_prefix
        decoded_prefix = _decode(self._tokenizer, prefix_ids, skip_special_tokens=False)
        decoded_combined = _decode(
            self._tokenizer,
            prefix_ids + continuation_ids,
            skip_special_tokens=False,
        )
        if not decoded_combined.startswith(decoded_prefix):
            raise PrefixTokenIdentityError("forced tokens changed the decoded immutable prefix")
        if decoded_combined[len(decoded_prefix) :] != raw_text:
            raise PrefixTokenIdentityError(
                "forced continuation does not decode exactly after the immutable prefix"
            )
        self._pending_append_prefix = None
        return continuation_ids

    def _vllm_factories(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        sampling_factory = self._sampling_params_factory
        prompt_factory = self._tokens_prompt_factory
        if sampling_factory is None:
            try:
                from vllm import SamplingParams
            except ImportError as exc:  # pragma: no cover - GPU-only path
                raise RuntimeError("vLLM SamplingParams is unavailable") from exc
            sampling_factory = SamplingParams
        if prompt_factory is None:
            try:
                from vllm.inputs import TokensPrompt
            except ImportError as exc:  # pragma: no cover - GPU-only path
                raise RuntimeError("vLLM TokensPrompt is unavailable") from exc
            prompt_factory = TokensPrompt
        return sampling_factory, prompt_factory

    def generate(
        self,
        requests: Sequence[RawPrefixGenerationRequest],
    ) -> Sequence[RawPrefixGenerationResult]:
        if not requests:
            return []
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("raw-prefix request IDs must be unique")
        sampling_factory, prompt_factory = self._vllm_factories()
        prompt_objects: list[Any] = []
        sampling_objects: list[Any] = []
        sampling_base = asdict(self.parameters)
        for request in requests:
            prompt_ids = _token_ids(request.prompt_token_ids, name="request prompt token IDs")
            if len(prompt_ids) + self.parameters.max_new_tokens > self.max_model_len:
                raise VLLMPrefixBackendError(
                    f"request {request.request_id} plus max_new_tokens exceeds max_model_len"
                )
            prompt_objects.append(prompt_factory(prompt_token_ids=list(prompt_ids)))
            sampling_objects.append(
                sampling_factory(
                    temperature=self.parameters.temperature,
                    top_p=self.parameters.top_p,
                    top_k=self.parameters.top_k,
                    min_p=self.parameters.min_p,
                    presence_penalty=self.parameters.presence_penalty,
                    repetition_penalty=self.parameters.repetition_penalty,
                    max_tokens=self.parameters.max_new_tokens,
                    stop=list(self.parameters.stop),
                    seed=request.seed,
                    n=1,
                    detokenize=True,
                    include_stop_str_in_output=False,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=True,
                )
            )
        outputs = self._llm.generate(
            prompt_objects,
            sampling_objects,
            use_tqdm=self._use_tqdm,
        )
        if len(outputs) != len(requests):
            raise VLLMPrefixBackendError("vLLM returned the wrong number of prefix outputs")

        results: list[RawPrefixGenerationResult] = []
        for request, output in zip(requests, outputs, strict=True):
            candidates = getattr(output, "outputs", None)
            if not isinstance(candidates, Sequence) or len(candidates) != 1:
                raise VLLMPrefixBackendError("vLLM must return exactly one candidate per request")
            candidate = candidates[0]
            observed_prompt = getattr(output, "prompt_token_ids", None)
            observed_completion = getattr(candidate, "token_ids", None)
            if observed_prompt is None or observed_completion is None:
                raise VLLMPrefixBackendError("vLLM omitted exact prompt or completion token IDs")
            prompt_ids = _token_ids(observed_prompt, name="consumed prompt token IDs")
            completion_ids = _token_ids(
                observed_completion,
                name="generated completion token IDs",
                allow_empty=True,
            )
            if prompt_ids != tuple(request.prompt_token_ids):
                raise PrefixTokenIdentityError(
                    f"vLLM consumed different prefix IDs for request {request.request_id}"
                )
            generated_text = getattr(candidate, "text", None)
            if not isinstance(generated_text, str):
                raise VLLMPrefixBackendError("vLLM returned non-text completion output")
            streams = token_stream_manifest(
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
            )
            per_request_sampling = {
                **sampling_base,
                "stop": list(self.parameters.stop),
                "seed": request.seed,
                "n": 1,
                "detokenize": True,
                "include_stop_str_in_output": False,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": True,
            }
            results.append(
                RawPrefixGenerationResult(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    prompt_token_ids=prompt_ids,
                    finish_reason=str(getattr(candidate, "finish_reason", "unknown")),
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(completion_ids),
                    backend_metadata={
                        "engine_request_id": str(getattr(output, "request_id", "")),
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "seed": request.seed,
                        "sampling": per_request_sampling,
                        **streams,
                    },
                )
            )
        return results


__all__ = [
    "QWEN_THINKING_PROMPT_SUFFIX",
    "ExactPrefixRegistration",
    "PrefixRegistrationError",
    "PrefixTokenIdentityError",
    "VLLMPrefixBackendError",
    "VLLMRawPrefixBackend",
]
