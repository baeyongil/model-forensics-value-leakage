"""Secret-safe, budget-guarded external JSON judging providers.

One generic OpenRouter client owns transport, retry, budget, and usage logic.
Thin adapters expose the provider-neutral adjudication and blinded sentence-
classification contracts without leaking provider details into either frozen
scientific instrument.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from model_forensics.adjudication import (
    AdjudicationRequest,
    JudgeProvenance,
)
from model_forensics.budget import CostEntry, CostLedger
from model_forensics.classification import ModelProvenance
from model_forensics.io import canonical_json, stable_hash
from model_forensics.paid_response_store import PaidResponseStore

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_MESSAGE_OVERHEAD_TOKEN_BOUND = 64


class ProviderError(RuntimeError):
    """External provider returned an unusable response."""


@dataclass(frozen=True)
class TokenPrice:
    """Frozen USD prices per million input/output tokens."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or input_tokens < 0
            or output_tokens < 0
        ):
            raise ValueError("token counts must be nonnegative integers")
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class HTTPResult:
    status: int
    body: Mapping[str, Any]


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], HTTPResult]


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> HTTPResult:
    request = urllib.request.Request(
        url,
        data=canonical_json(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned non-JSON HTTP {status}") from exc
    if not isinstance(body, Mapping):
        raise ProviderError(f"provider returned a non-object HTTP {status} response")
    return HTTPResult(status=status, body=body)


_PURPOSE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def _deterministic_seed(request_id: str) -> int:
    """Preserve hex-ID seeds and safely support arbitrary opaque request IDs."""

    suffix = request_id.split(":", 1)[-1]
    if len(suffix) >= 8 and all(character in "0123456789abcdefABCDEF" for character in suffix[:8]):
        return int(suffix[:8], 16)
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)


def _validate_json_object(content: str) -> None:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ProviderError(f"provider JSON contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ProviderError(f"provider JSON contains non-finite number {value!r}")

    try:
        decoded = json.loads(
            content,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ProviderError("provider completion is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ProviderError("provider completion must be one JSON object")


class OpenRouterJSONClient:
    """Generic secret-safe, budgeted JSON-object completion client.

    Only opaque request IDs and non-secret model configuration reach provenance
    and the ledger.  Successful provider responses are cost-accounted before
    their JSON shape is validated, so malformed paid responses are not silently
    omitted from the budget record.
    """

    def __init__(
        self,
        *,
        model_id: str,
        price: TokenPrice,
        ledger: CostLedger,
        model_revision: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        endpoint: str = OPENROUTER_ENDPOINT,
        max_output_tokens: int = 512,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if max_output_tokens <= 0 or max_attempts <= 0 or timeout_seconds <= 0:
            raise ValueError("output limit, attempts, and timeout must be positive")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"required secret environment variable is unset: {api_key_env}")
        self._api_key = api_key
        self._model_id = model_id
        self._model_revision = model_revision
        self._price = price
        self._ledger = ledger
        self._endpoint = endpoint
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._paid_response_store = paid_response_store
        self._last_metadata: dict[str, Any] = {"calls_completed": 0}
        self._audit_records: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    @property
    def decoding(self) -> dict[str, Any]:
        return {
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "response_format": "json_object",
            "preflight_input_bound": "one_token_per_character_plus_64_per_message",
        }

    @property
    def pricing(self) -> dict[str, float]:
        return {
            "input_per_million": float(self._price.input_per_million),
            "output_per_million": float(self._price.output_per_million),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    @property
    def audit_records(self) -> tuple[dict[str, Any], ...]:
        """Return non-secret per-call usage records in completion order."""

        return tuple(dict(record) for record in self._audit_records)

    @property
    def audit_record_count(self) -> int:
        return len(self._audit_records)

    def audit_records_since(self, index: int) -> tuple[dict[str, Any], ...]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(self._audit_records)
        ):
            raise ValueError("audit record index is out of range")
        return tuple(dict(record) for record in self._audit_records[index:])

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "model_id": self._model_id,
            "model_revision": self._model_revision,
            "client_version": "openrouter-json-client-v2",
            "decoding": self.decoding,
            "pricing_usd_per_million_tokens": self.pricing,
            "metadata": self.metadata,
        }

    @staticmethod
    def _usage(body: Mapping[str, Any]) -> tuple[int, int, float | None]:
        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            raise ProviderError("provider response omitted auditable usage")
        try:
            raw_input_tokens = usage["prompt_tokens"]
            raw_output_tokens = usage["completion_tokens"]
        except KeyError as exc:
            raise ProviderError("provider usage token counts are missing or invalid") from exc
        if (
            isinstance(raw_input_tokens, bool)
            or not isinstance(raw_input_tokens, int)
            or isinstance(raw_output_tokens, bool)
            or not isinstance(raw_output_tokens, int)
        ):
            raise ProviderError("provider usage token counts are missing or invalid")
        input_tokens = raw_input_tokens
        output_tokens = raw_output_tokens
        if input_tokens < 0 or output_tokens < 0:
            raise ProviderError("provider usage token counts must be nonnegative")
        raw_cost = usage.get("cost")
        if raw_cost is None:
            return input_tokens, output_tokens, None
        if isinstance(raw_cost, bool):
            raise ProviderError("provider usage cost is invalid")
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise ProviderError("provider usage cost is invalid") from exc
        if not math.isfinite(cost) or cost < 0:
            raise ProviderError("provider usage cost must be finite and nonnegative")
        return input_tokens, output_tokens, cost

    def _preflight_cost(
        self,
        *,
        request_id: str,
        purpose: str,
        system_prompt: str | None,
        user_content: str,
        reservation_id: str,
    ) -> float:
        # One UTF-8 character per token is deliberately conservative for these
        # English-only instruments and protects the cap before the request leaves.
        message_count = 1 + int(bool(system_prompt))
        input_upper_bound = (
            len(system_prompt or "")
            + len(user_content)
            + _MESSAGE_OVERHEAD_TOKEN_BOUND * message_count
        )
        predicted = self._price.cost(input_upper_bound, self._max_output_tokens)
        request_hash = stable_hash(request_id)
        self._ledger.reserve(
            reservation_id,
            CostEntry(
                kind="api",
                amount_usd=predicted,
                description=f"preflight OpenRouter {purpose} {request_hash}",
                status="estimated",
            ),
        )
        return predicted

    def complete_json(
        self,
        *,
        request_id: str,
        user_content: str,
        purpose: str,
        system_prompt: str | None = None,
    ) -> str:
        """Return one paid JSON-object completion under a hard budget gate."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty opaque string")
        if not isinstance(user_content, str) or not user_content:
            raise ValueError("user_content must be a non-empty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if not isinstance(purpose, str) or _PURPOSE_RE.fullmatch(purpose) is None:
            raise ValueError("purpose must be a short lowercase identifier")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
            "seed": _deterministic_seed(request_id),
        }
        store_key = PaidResponseStore.key(
            request_id=request_id,
            model_id=self._model_id,
            purpose=purpose,
        )
        request_fingerprint = PaidResponseStore.fingerprint(
            endpoint=self._endpoint,
            model_id=self._model_id,
            purpose=purpose,
            system_prompt=system_prompt,
            user_content=user_content,
            decoding=self.decoding,
        )
        reservation_id = stable_hash(
            {"protocol": "openrouter-api-reservation-v1", "store_key": store_key}
        )
        cached = (
            self._paid_response_store.load(
                key=store_key,
                request_fingerprint=request_fingerprint,
            )
            if self._paid_response_store is not None
            else None
        )
        predicted = self._preflight_cost(
            request_id=request_id,
            purpose=purpose,
            system_prompt=system_prompt,
            user_content=user_content,
            reservation_id=reservation_id,
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/baeyongil/model-forensics-value-leakage",
            "X-Title": "Model Forensics Value Leakage",
        }
        result: HTTPResult | None = None
        attempts_used = 0
        replayed_from_checkpoint = cached is not None
        checkpoint_record_hash: str | None = None
        if cached is not None:
            result = HTTPResult(
                status=int(cached["http_status"]),
                body=cached["response_body"],
            )
            checkpoint_record_hash = str(cached["record_hash"])
        else:
            for attempt in range(1, self._max_attempts + 1):
                attempts_used = attempt
                try:
                    result = self._transport(
                        self._endpoint,
                        headers,
                        payload,
                        self._timeout_seconds,
                    )
                except (ConnectionError, TimeoutError, urllib.error.URLError):
                    if attempt >= self._max_attempts:
                        raise ProviderError(
                            f"provider transport failed after {attempt} attempts"
                        ) from None
                    self._sleep(min(2 ** (attempt - 1), 4))
                    continue
                if result.status not in {408, 409, 425, 429} and result.status < 500:
                    break
                if attempt < self._max_attempts:
                    self._sleep(min(2 ** (attempt - 1), 4))
        if result is None:  # pragma: no cover - loop invariant
            raise ProviderError("provider transport produced no response")
        if result.status < 200 or result.status >= 300:
            error = result.body.get("error")
            error_type = error.get("type") if isinstance(error, Mapping) else None
            raise ProviderError(
                f"provider HTTP {result.status}; error_type={error_type or 'unknown'}"
            )

        # Persist the complete paid body before usage/content parsing.  A
        # malformed paid response therefore remains a terminal, replayable
        # artifact rather than triggering an accidental second charge.
        if cached is None and self._paid_response_store is not None:
            committed = self._paid_response_store.commit(
                key=store_key,
                request_fingerprint=request_fingerprint,
                logical_request_hash=stable_hash(request_id),
                model_id=self._model_id,
                purpose=purpose,
                http_status=result.status,
                response_body=result.body,
            )
            checkpoint_record_hash = str(committed["record_hash"])

        input_tokens, output_tokens, reported_cost = self._usage(result.body)
        computed_cost = self._price.cost(input_tokens, output_tokens)
        incurred = reported_cost if reported_cost is not None else computed_cost
        request_hash = stable_hash(request_id)
        totals = self._ledger.settle_reservation(
            reservation_id,
            CostEntry(
                kind="api",
                amount_usd=incurred,
                description=f"OpenRouter {purpose} {request_hash}",
            ),
        )
        response_id = result.body.get("id")
        self._last_metadata = {
            "calls_completed": int(self._last_metadata["calls_completed"]) + 1,
            "purpose": purpose,
            "logical_request_hash": request_hash,
            "provider_response_id_hash": (stable_hash(str(response_id)) if response_id else None),
            # Backward-compatible name used by the original adjudication adapter.
            "request_id_hash": stable_hash(str(response_id)) if response_id else None,
            "response_model": (
                result.body.get("model") if isinstance(result.body.get("model"), str) else None
            ),
            "response_provider": (
                result.body.get("provider")
                if isinstance(result.body.get("provider"), str)
                else None
            ),
            "attempts_used": attempts_used,
            "replayed_from_checkpoint": replayed_from_checkpoint,
            "paid_response_checkpoint_hash": checkpoint_record_hash,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reported_cost_usd": reported_cost,
            "computed_cost_usd": computed_cost,
            "charged_cost_usd": incurred,
            "preflight_upper_bound_usd": predicted,
            "api_total_usd": totals["api"],
        }
        self._audit_records.append(dict(self._last_metadata))

        choices = result.body.get("choices")
        try:
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("provider response omitted choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider returned empty completion content")
        stripped = content.strip()
        _validate_json_object(stripped)
        return stripped


class OpenRouterAdjudicationCaller:
    """Adapter preserving the provider-neutral adjudication caller contract."""

    not_for_primary_inference = False

    def __init__(
        self,
        *,
        model_id: str,
        price: TokenPrice,
        ledger: CostLedger,
        model_revision: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        endpoint: str = OPENROUTER_ENDPOINT,
        max_output_tokens: int = 512,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
    ) -> None:
        self._client = OpenRouterJSONClient(
            model_id=model_id,
            model_revision=model_revision,
            price=price,
            ledger=ledger,
            api_key_env=api_key_env,
            endpoint=endpoint,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            transport=transport,
            sleep=sleep,
            paid_response_store=paid_response_store,
        )

    @property
    def provenance(self) -> JudgeProvenance:
        return JudgeProvenance(
            provider="openrouter",
            model_id=self._client.model_id,
            model_revision=self._client.model_revision,
            caller_version="openrouter-json-v1",
            decoding=self._client.decoding,
            metadata={
                "pricing_usd_per_million_tokens": self._client.pricing,
                **self._client.metadata,
            },
        )

    def complete(self, request: AdjudicationRequest) -> str:
        return self._client.complete_json(
            request_id=request.request_id,
            system_prompt=request.system_prompt,
            user_content=canonical_json(dict(request.user_payload)),
            purpose="adjudication",
        )


class OpenRouterClassificationCaller:
    """OpenRouter adapter for the frozen blinded anchor-classifier prompt."""

    not_for_primary_inference = False

    def __init__(
        self,
        *,
        model_id: str,
        price: TokenPrice,
        ledger: CostLedger,
        model_revision: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        endpoint: str = OPENROUTER_ENDPOINT,
        max_output_tokens: int = 256,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
    ) -> None:
        self._client = OpenRouterJSONClient(
            model_id=model_id,
            model_revision=model_revision,
            price=price,
            ledger=ledger,
            api_key_env=api_key_env,
            endpoint=endpoint,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            transport=transport,
            sleep=sleep,
            paid_response_store=paid_response_store,
        )
        self._last_audit_metadata: dict[str, Any] = {}
        self._audit_records: list[dict[str, Any]] = []

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            provider="openrouter",
            model_id=self._client.model_id,
            model_revision=self._client.model_revision,
            caller_version="openrouter-classification-json-v1",
            external=True,
        )

    @property
    def usage_metadata(self) -> dict[str, Any]:
        return {
            "pricing_usd_per_million_tokens": self._client.pricing,
            **self._client.metadata,
            **self._last_audit_metadata,
        }

    @property
    def usage_records(self) -> tuple[dict[str, Any], ...]:
        """Return one secret-free, judgment-linked record per completed call."""

        return tuple(dict(record) for record in self._audit_records)

    def __call__(
        self,
        *,
        prompt: str,
        judgment_id: str,
        input_hash: str,
        prompt_hash: str,
    ) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("classification prompt must be non-empty")
        for name, digest in (
            ("judgment_id", judgment_id),
            ("input_hash", input_hash),
            ("prompt_hash", prompt_hash),
        ):
            if not isinstance(digest, str) or _HEX_DIGEST_RE.fullmatch(digest) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        observed_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(observed_prompt_hash, prompt_hash):
            raise ProviderError("classification prompt hash mismatch")
        blinded_marker = "Blinded input:\n"
        if blinded_marker not in prompt:
            raise ProviderError("classification prompt omitted the blinded input marker")
        visible_input = prompt.rsplit(blinded_marker, maxsplit=1)[1]
        observed_input_hash = hashlib.sha256(visible_input.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(observed_input_hash, input_hash):
            raise ProviderError("classification blinded-input hash mismatch")

        audit_metadata = {
            "judgment_id_hash": stable_hash(judgment_id),
            "input_hash": input_hash,
            "prompt_hash": prompt_hash,
        }
        self._last_audit_metadata = audit_metadata
        audit_count_before = self._client.audit_record_count
        try:
            response = self._client.complete_json(
                request_id=judgment_id,
                user_content=prompt,
                purpose="classification",
            )
        finally:
            for usage in self._client.audit_records_since(audit_count_before):
                self._audit_records.append(
                    {
                        "pricing_usd_per_million_tokens": self._client.pricing,
                        **usage,
                        **audit_metadata,
                    }
                )
        return response


__all__ = [
    "OPENROUTER_ENDPOINT",
    "HTTPResult",
    "OpenRouterAdjudicationCaller",
    "OpenRouterClassificationCaller",
    "OpenRouterJSONClient",
    "ProviderError",
    "TokenPrice",
]
