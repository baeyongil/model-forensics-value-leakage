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
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from model_forensics.adjudication import (
    AdjudicationRequest,
    JudgeProvenance,
)
from model_forensics.budget import BudgetExceeded, CostEntry, CostLedger
from model_forensics.classification import ModelProvenance
from model_forensics.io import canonical_json, stable_hash
from model_forensics.paid_response_store import PaidResponseStore

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_MESSAGE_OVERHEAD_TOKEN_BOUND = 64


def _ceil_usd_six(value: float) -> float:
    """Match ledger precision without rounding a conservative amount downward."""

    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


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
PaidResponseStoreIdentity = tuple[str, int, int, int, int, int, int]


def _api_reservation_id(
    *,
    store_key: str,
    store_identity: PaidResponseStoreIdentity | None,
) -> str:
    return stable_hash(
        {
            "protocol": "openrouter-api-reservation-v2",
            "store_key": store_key,
            "paid_response_store_identity": (
                list(store_identity) if store_identity is not None else None
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class OpenRouterRequestSpec:
    """One exact, secret-free OpenRouter request used for phase budgeting.

    The object contains the actual system and user strings so the phase gate and
    :class:`OpenRouterJSONClient` price precisely the same payload.  Only hashes
    of those strings enter the public-ish paid plan.
    """

    route: str
    model_id: str
    price: TokenPrice
    request_id: str
    purpose: str
    user_content: str
    paid_response_store: PaidResponseStore
    system_prompt: str | None = None
    model_revision: str | None = None
    endpoint: str = OPENROUTER_ENDPOINT
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.route, str) or not self.route:
            raise ValueError("request route must be non-empty")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("request model_id must be non-empty")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty opaque string")
        if not isinstance(self.user_content, str) or not self.user_content:
            raise ValueError("user_content must be non-empty")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if not isinstance(self.purpose, str) or _PURPOSE_RE.fullmatch(self.purpose) is None:
            raise ValueError("purpose must be a short lowercase identifier")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")

    @property
    def decoding(self) -> dict[str, Any]:
        return _openrouter_decoding(self.max_output_tokens)

    @property
    def store_key(self) -> str:
        return PaidResponseStore.key(
            request_id=self.request_id,
            model_id=self.model_id,
            purpose=self.purpose,
        )

    @property
    def request_fingerprint(self) -> str:
        return PaidResponseStore.fingerprint(
            endpoint=self.endpoint,
            model_id=self.model_id,
            purpose=self.purpose,
            system_prompt=self.system_prompt,
            user_content=self.user_content,
            decoding=self.decoding,
        )

    @property
    def reservation_id(self) -> str:
        return _api_reservation_id(
            store_key=self.store_key,
            store_identity=self.paid_response_store.identity(),
        )

    @property
    def input_token_upper_bound(self) -> int:
        return _openrouter_input_token_upper_bound(
            system_prompt=self.system_prompt,
            user_content=self.user_content,
        )

    @property
    def conservative_cost_usd(self) -> float:
        # CostLedger normalizes each individual reservation to six decimals.
        # Summing those exact normalized amounts makes the phase boundary agree
        # with the later per-call budget gates, including equality at the cap.
        return _ceil_usd_six(self.price.cost(self.input_token_upper_bound, self.max_output_tokens))

    @property
    def request_manifest(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "purpose": self.purpose,
            "logical_request_hash": stable_hash(self.request_id),
            "store_key": self.store_key,
            "request_fingerprint": self.request_fingerprint,
            "payload_hash": stable_hash(
                {
                    "system_prompt": self.system_prompt,
                    "user_content": self.user_content,
                }
            ),
            "decoding": self.decoding,
            "pricing_usd_per_million_tokens": {
                "input": float(self.price.input_per_million),
                "output": float(self.price.output_per_million),
            },
            "input_token_upper_bound": self.input_token_upper_bound,
            "max_output_tokens": self.max_output_tokens,
            "conservative_cost_usd": self.conservative_cost_usd,
            "reservation_id": self.reservation_id,
        }


@dataclass(frozen=True, slots=True)
class OpenRouterPhasePreflight:
    """Authenticated cost-to-completion result for one whole paid API phase."""

    phase: str
    requests: tuple[OpenRouterRequestSpec, ...]
    paid_response_store_identities: tuple[PaidResponseStoreIdentity, ...]
    manifest: Mapping[str, Any]

    @property
    def manifest_hash(self) -> str:
        value = self.manifest.get("manifest_hash")
        if not isinstance(value, str):  # pragma: no cover - constructor invariant
            raise ProviderError("API completion preflight lacks a manifest hash")
        return value

    def assert_manifest(self, expected: Mapping[str, Any]) -> None:
        """Reject a modified or substituted inventory before any provider call."""

        observed = dict(expected)
        supplied_hash = observed.pop("manifest_hash", None)
        if supplied_hash != stable_hash(observed) or dict(expected) != dict(self.manifest):
            raise ProviderError("API completion inventory manifest mismatch")


def _request_identity(spec: OpenRouterRequestSpec) -> str:
    return stable_hash(
        {
            "protocol": "openrouter-exact-dispatch-identity-v1",
            "request_manifest": spec.request_manifest,
        }
    )


def _paid_response_store_identity(store: PaidResponseStore) -> PaidResponseStoreIdentity:
    try:
        return store.identity()
    except (OSError, RuntimeError) as exc:
        raise ProviderError("paid-response store identity is unavailable") from exc


@dataclass(frozen=True, slots=True)
class OpenRouterDispatchAuthorization:
    request_identity: str
    frozen_status: str
    paid_response_store_identity: PaidResponseStoreIdentity


class OpenRouterDispatchGuard:
    """Consume only exact logical calls frozen by a whole-phase preflight."""

    def __init__(self, preflight: OpenRouterPhasePreflight) -> None:
        preflight.assert_manifest(preflight.manifest)
        request_rows = [spec.request_manifest for spec in preflight.requests]
        if preflight.manifest.get("full_inventory_hash") != stable_hash(
            {"phase": preflight.phase, "logical_request_manifests": request_rows}
        ):
            raise ProviderError("dispatch guard request inventory disagrees with preflight")
        self.phase = preflight.phase
        cached = preflight.manifest.get("authenticated_cached_request_identities")
        pending = preflight.manifest.get("pending_request_identities")
        if not isinstance(cached, list) or not isinstance(pending, list):
            raise ProviderError("dispatch guard preflight statuses are absent")
        cached_set = set(cached)
        pending_set = set(pending)
        if cached_set & pending_set:
            raise ProviderError("dispatch guard preflight statuses overlap")
        expected_identities = {_request_identity(spec) for spec in preflight.requests}
        if cached_set | pending_set != expected_identities:
            raise ProviderError("dispatch guard preflight statuses are incomplete")
        if len(preflight.paid_response_store_identities) != len(preflight.requests):
            raise ProviderError("dispatch guard store inventory is incomplete")
        store_identity_hashes = [
            stable_hash(
                {
                    "protocol": "paid-response-store-identity-v2",
                    "identity": list(identity),
                }
            )
            for identity in preflight.paid_response_store_identities
        ]
        if preflight.manifest.get("paid_response_store_identities_hash") != stable_hash(
            store_identity_hashes
        ):
            raise ProviderError("dispatch guard store inventory disagrees with preflight")
        self._remaining: dict[str, list[OpenRouterDispatchAuthorization]] = {}
        for spec, store_identity in zip(
            preflight.requests,
            preflight.paid_response_store_identities,
            strict=True,
        ):
            identity = _request_identity(spec)
            if identity in cached_set:
                status = "authenticated_paid_response"
            elif identity in pending_set:
                status = "pending"
            else:
                raise ProviderError("dispatch guard request lacks a frozen preflight status")
            self._remaining.setdefault(identity, []).append(
                OpenRouterDispatchAuthorization(
                    request_identity=identity,
                    frozen_status=status,
                    paid_response_store_identity=store_identity,
                )
            )
        self._lock = threading.Lock()

    def authorize(self, spec: OpenRouterRequestSpec) -> OpenRouterDispatchAuthorization:
        identity = _request_identity(spec)
        with self._lock:
            remaining = self._remaining.get(identity)
            if not remaining:
                raise ProviderError(
                    "provider request is not present in the authorized phase inventory"
                )
            authorization = remaining[-1]
            if (
                _paid_response_store_identity(spec.paid_response_store)
                != authorization.paid_response_store_identity
            ):
                raise ProviderError(
                    "provider request paid-response store identity changed after preflight"
                )
            remaining.pop()
        return authorization


def _openrouter_decoding(max_output_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": "json_object",
        "preflight_input_bound": "one_token_per_utf8_byte_plus_64_per_message",
    }


def _openrouter_input_token_upper_bound(
    *,
    system_prompt: str | None,
    user_content: str,
) -> int:
    message_count = 1 + int(bool(system_prompt))
    return (
        len((system_prompt or "").encode("utf-8"))
        + len(user_content.encode("utf-8"))
        + _MESSAGE_OVERHEAD_TOKEN_BOUND * message_count
    )


def _reservation_description(spec: OpenRouterRequestSpec) -> str:
    return f"preflight OpenRouter {spec.purpose} {stable_hash(spec.request_id)}"


def preflight_openrouter_phase(
    *,
    phase: str,
    requests: Sequence[OpenRouterRequestSpec],
    ledger: CostLedger,
) -> OpenRouterPhasePreflight:
    """Gate the complete remaining phase before its first provider request.

    Every exact request is fingerprinted first.  Only a checkpoint that passes
    :meth:`PaidResponseStore.load` for that exact fingerprint is excluded.  An
    exact duplicate invocation within the same route/store is counted once as a
    future paid transport because the first response will durably satisfy all
    later identical invocations.
    """

    if not isinstance(phase, str) or not phase:
        raise ValueError("API phase must be non-empty")
    frozen = tuple(requests)
    paid_response_store_identities = tuple(
        _paid_response_store_identity(spec.paid_response_store) for spec in frozen
    )
    reservation_store_identities: dict[str, PaidResponseStoreIdentity] = {}
    for spec, store_identity in zip(
        frozen,
        paid_response_store_identities,
        strict=True,
    ):
        previous = reservation_store_identities.setdefault(
            spec.reservation_id,
            store_identity,
        )
        if previous != store_identity:
            raise ProviderError(
                "one API reservation identity maps to multiple paid-response stores"
            )
    paid_response_store_identity_hashes = [
        stable_hash(
            {
                "protocol": "paid-response-store-identity-v2",
                "identity": list(identity),
            }
        )
        for identity in paid_response_store_identities
    ]
    request_rows = [spec.request_manifest for spec in frozen]
    full_inventory_hash = stable_hash({"phase": phase, "logical_request_manifests": request_rows})

    unique: dict[tuple[str, str], OpenRouterRequestSpec] = {}
    invocation_counts: Counter[tuple[str, str]] = Counter()
    route_store_directories: dict[str, Path] = {}
    for spec in frozen:
        directory = Path(_paid_response_store_identity(spec.paid_response_store)[0])
        previous_directory = route_store_directories.setdefault(spec.route, directory)
        if previous_directory != directory:
            raise ProviderError("one API inventory route maps to multiple response stores")
        key = (spec.route, spec.store_key)
        previous = unique.get(key)
        if previous is not None and previous.request_manifest != spec.request_manifest:
            raise ProviderError("API inventory store-key collision has different exact payloads")
        unique.setdefault(key, spec)
        invocation_counts[key] += 1

    cached_rows: list[dict[str, Any]] = []
    pending_specs: list[OpenRouterRequestSpec] = []
    unique_rows: list[dict[str, Any]] = []
    for key, spec in unique.items():
        cached = spec.paid_response_store.load(
            key=spec.store_key,
            request_fingerprint=spec.request_fingerprint,
        )
        row = {
            **spec.request_manifest,
            "logical_invocation_count": invocation_counts[key],
            "status": "authenticated_paid_response" if cached is not None else "pending",
            "paid_response_checkpoint_hash": (
                str(cached["record_hash"]) if cached is not None else None
            ),
        }
        unique_rows.append(row)
        if cached is None:
            pending_specs.append(spec)
        else:
            cached_rows.append(row)

    pending_bound = round(sum(spec.conservative_cost_usd for spec in pending_specs), 6)
    reservations = tuple(
        (
            spec.reservation_id,
            CostEntry(
                kind="api",
                amount_usd=spec.conservative_cost_usd,
                description=_reservation_description(spec),
                status="estimated",
            ),
        )
        for spec in pending_specs
    )
    try:
        reservation = ledger.reserve_batch(
            reservations,
            required_existing_entry_ids=tuple(
                spec.reservation_id for spec in unique.values() if spec not in pending_specs
            ),
            reject_incurred_entry_ids=tuple(spec.reservation_id for spec in pending_specs),
        )
    except BudgetExceeded as exc:
        raise BudgetExceeded(f"whole API phase cannot be reserved atomically: {exc}") from exc
    except ValueError as exc:
        raise ProviderError(str(exc)) from exc

    document = ledger.document()
    entries_by_id = {
        str(entry["entry_id"]): entry
        for entry in document["entries"]
        if isinstance(entry.get("entry_id"), str)
    }
    pending_ids = {spec.reservation_id for spec in pending_specs}
    for spec in unique.values():
        entry = entries_by_id.get(spec.reservation_id)
        if entry is None:  # pragma: no cover - atomic reservation invariant
            raise ProviderError("API request lacks its matching ledger reservation")
        if spec.reservation_id in pending_ids:
            if (
                entry.get("status") != "estimated"
                or entry.get("kind") != "api"
                or float(entry.get("amount_usd", -1)) != spec.conservative_cost_usd
                or entry.get("description") != _reservation_description(spec)
            ):
                raise ProviderError(
                    "existing API reservation differs from the exact pending request"
                )
            continue
        request_hash = stable_hash(spec.request_id)
        estimated_matches = bool(
            entry.get("status") == "estimated"
            and entry.get("kind") == "api"
            and float(entry.get("amount_usd", -1)) == spec.conservative_cost_usd
            and entry.get("description") == _reservation_description(spec)
        )
        incurred_matches = bool(
            entry.get("status") == "incurred"
            and entry.get("kind") == "api"
            and entry.get("description") == f"OpenRouter {spec.purpose} {request_hash}"
        )
        if not (estimated_matches or incurred_matches):
            raise ProviderError("authenticated paid response has a mismatched ledger reservation")

    created_ids = set(reservation.created_entry_ids)
    additional = round(
        sum(
            spec.conservative_cost_usd
            for spec in pending_specs
            if spec.reservation_id in created_ids
        ),
        6,
    )
    covered = round(pending_bound - additional, 6)
    projected_api = reservation.committed_after["api"]
    projected_total = reservation.committed_after["total"]

    route_names = sorted({spec.route for spec in frozen})
    per_route: dict[str, Any] = {}
    for route in route_names:
        route_invocations = [spec for spec in frozen if spec.route == route]
        route_unique = [row for row in unique_rows if row["route"] == route]
        route_pending = [row for row in route_unique if row["status"] == "pending"]
        per_route[route] = {
            "logical_invocation_count": len(route_invocations),
            "unique_request_count": len(route_unique),
            "authenticated_cached_count": len(route_unique) - len(route_pending),
            "pending_request_count": len(route_pending),
            "pending_conservative_usd": round(
                sum(float(row["conservative_cost_usd"]) for row in route_pending), 6
            ),
        }

    pending_rows = [row for row in unique_rows if row["status"] == "pending"]
    unique_request_identities = sorted(_request_identity(spec) for spec in unique.values())
    pending_request_identities = sorted(_request_identity(spec) for spec in pending_specs)
    cached_request_identities = sorted(
        set(unique_request_identities) - set(pending_request_identities)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "openrouter-phase-cost-to-completion-v1",
        "phase": phase,
        "logical_invocation_count": len(frozen),
        "unique_request_count": len(unique_rows),
        "authenticated_cached_count": len(cached_rows),
        "pending_request_count": len(pending_rows),
        "full_inventory_hash": full_inventory_hash,
        "unique_inventory_hash": stable_hash(unique_rows),
        "pending_inventory_hash": stable_hash(pending_rows),
        "paid_response_store_identities_hash": stable_hash(paid_response_store_identity_hashes),
        "unique_request_identities": unique_request_identities,
        "pending_request_identities": pending_request_identities,
        "authenticated_cached_request_identities": cached_request_identities,
        "per_route": per_route,
        "conservative_pending_usd": pending_bound,
        "covered_by_existing_reservations_usd": covered,
        "additional_commitment_required_usd": additional,
        "ledger": {
            "document_hash": stable_hash(document),
            "document_before_hash": reservation.document_before_hash,
            "document_after_hash": reservation.document_after_hash,
            "incurred_before_usd": reservation.incurred_before,
            "committed_before_usd": reservation.committed_before,
            "committed_after_reservation_usd": reservation.committed_after,
            "created_reservation_ids_hash": stable_hash(sorted(reservation.created_entry_ids)),
            "covered_reservation_ids_hash": stable_hash(sorted(reservation.covered_entry_ids)),
            "hard_stops_usd": {
                "api": float(ledger.limits.api),
                "total": float(ledger.limits.total),
            },
            "remaining_before_usd": {
                "api": round(float(ledger.limits.api) - reservation.committed_before["api"], 6),
                "total": round(
                    float(ledger.limits.total) - reservation.committed_before["total"], 6
                ),
            },
            "projected_after_completion_usd": {
                "api": projected_api,
                "total": projected_total,
            },
        },
    }
    payload["manifest_hash"] = stable_hash(payload)
    if (
        tuple(_paid_response_store_identity(spec.paid_response_store) for spec in frozen)
        != paid_response_store_identities
    ):
        raise ProviderError("paid-response store identity changed during API preflight")
    return OpenRouterPhasePreflight(
        phase=phase,
        requests=frozen,
        paid_response_store_identities=paid_response_store_identities,
        manifest=payload,
    )


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
        max_attempts: int = 1,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
        dispatch_guard: OpenRouterDispatchGuard | None = None,
        dispatch_route: str | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if max_output_tokens <= 0 or max_attempts <= 0 or timeout_seconds <= 0:
            raise ValueError("output limit, attempts, and timeout must be positive")
        if max_attempts != 1:
            raise ValueError("automatic paid-provider retries are disabled; max_attempts must be 1")
        if (dispatch_guard is None) != (dispatch_route is None):
            raise ValueError("dispatch guard and route must be supplied together")
        if dispatch_route is not None and not dispatch_route:
            raise ValueError("dispatch route must be non-empty")
        if dispatch_guard is not None and paid_response_store is None:
            raise ValueError("guarded production dispatch requires a paid-response store")
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
        self._dispatch_guard = dispatch_guard
        self._dispatch_route = dispatch_route
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
        return _openrouter_decoding(self._max_output_tokens)

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
        # One token per UTF-8 byte is a tokenizer-independent upper bound for
        # arbitrary Unicode payloads under byte-fallback vocabularies.
        input_upper_bound = _openrouter_input_token_upper_bound(
            system_prompt=system_prompt,
            user_content=user_content,
        )
        predicted = _ceil_usd_six(self._price.cost(input_upper_bound, self._max_output_tokens))
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
        reservation_id = _api_reservation_id(
            store_key=store_key,
            store_identity=(
                self._paid_response_store.identity()
                if self._paid_response_store is not None
                else None
            ),
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/baeyongil/model-forensics-value-leakage",
            "X-Title": "Model Forensics Value Leakage",
        }
        dispatch_authorization: OpenRouterDispatchAuthorization | None = None
        if self._dispatch_guard is not None:
            assert self._dispatch_route is not None
            assert self._paid_response_store is not None
            dispatch_authorization = self._dispatch_guard.authorize(
                OpenRouterRequestSpec(
                    route=self._dispatch_route,
                    model_id=self._model_id,
                    model_revision=self._model_revision,
                    price=self._price,
                    request_id=request_id,
                    purpose=purpose,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    endpoint=self._endpoint,
                    max_output_tokens=self._max_output_tokens,
                    paid_response_store=self._paid_response_store,
                )
            )
        claim = (
            self._paid_response_store.request_claim(
                key=store_key, request_fingerprint=request_fingerprint
            )
            if self._paid_response_store is not None
            else nullcontext()
        )
        with claim:
            return self._complete_json_claimed(
                request_id=request_id,
                user_content=user_content,
                purpose=purpose,
                system_prompt=system_prompt,
                payload=payload,
                headers=headers,
                store_key=store_key,
                request_fingerprint=request_fingerprint,
                reservation_id=reservation_id,
                dispatch_authorization=dispatch_authorization,
            )

    def _complete_json_claimed(
        self,
        *,
        request_id: str,
        user_content: str,
        purpose: str,
        system_prompt: str | None,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        store_key: str,
        request_fingerprint: str,
        reservation_id: str,
        dispatch_authorization: OpenRouterDispatchAuthorization | None,
    ) -> str:
        store = self._paid_response_store
        if (
            dispatch_authorization is not None
            and store is not None
            and _paid_response_store_identity(store)
            != dispatch_authorization.paid_response_store_identity
        ):
            raise ProviderError(
                "provider request paid-response store identity changed after authorization"
            )
        cached = (
            store.load(key=store_key, request_fingerprint=request_fingerprint)
            if store is not None
            else None
        )
        if (
            dispatch_authorization is not None
            and dispatch_authorization.frozen_status == "authenticated_paid_response"
            and cached is None
        ):
            raise ProviderError(
                "provider request was frozen as cached but its checkpoint disappeared"
            )
        uncertain = (
            store.load_uncertain_attempt(key=store_key, request_fingerprint=request_fingerprint)
            if store is not None
            else None
        )
        if cached is None and uncertain is not None:
            raise ProviderError("uncertain paid attempt requires reconciliation before resume")
        if cached is not None and uncertain is not None and store is not None:
            store.resolve_uncertain_attempt(
                key=store_key,
                request_fingerprint=request_fingerprint,
                expected_record_hash=str(uncertain["record_hash"]),
            )
        predicted = self._preflight_cost(
            request_id=request_id,
            purpose=purpose,
            system_prompt=system_prompt,
            user_content=user_content,
            reservation_id=reservation_id,
        )
        attempts_used = 0
        checkpoint_record_hash: str | None = None
        if cached is not None:
            result = HTTPResult(status=int(cached["http_status"]), body=cached["response_body"])
            checkpoint_record_hash = str(cached["record_hash"])
        else:
            attempts_used = 1
            marker = (
                store.mark_uncertain_attempt(
                    key=store_key,
                    request_fingerprint=request_fingerprint,
                    logical_request_hash=stable_hash(request_id),
                    model_id=self._model_id,
                    purpose=purpose,
                    reservation_id=reservation_id,
                )
                if store is not None
                else None
            )
            try:
                result = self._transport(self._endpoint, headers, payload, self._timeout_seconds)
            except Exception:
                raise ProviderError(
                    "provider transport left an uncertain paid attempt; reconcile before resume"
                ) from None
        if cached is None and store is not None:
            committed = store.commit(
                key=store_key,
                request_fingerprint=request_fingerprint,
                logical_request_hash=stable_hash(request_id),
                model_id=self._model_id,
                purpose=purpose,
                http_status=result.status,
                response_body=result.body,
            )
            checkpoint_record_hash = str(committed["record_hash"])
            assert marker is not None
            store.resolve_uncertain_attempt(
                key=store_key,
                request_fingerprint=request_fingerprint,
                expected_record_hash=str(marker["record_hash"]),
            )
        if result.status < 200 or result.status >= 300:
            error = result.body.get("error")
            error_type = error.get("type") if isinstance(error, Mapping) else None
            raise ProviderError(
                f"provider HTTP {result.status}; error_type={error_type or 'unknown'}"
            )

        input_tokens, output_tokens, reported_cost = self._usage(result.body)
        computed_cost = self._price.cost(input_tokens, output_tokens)
        incurred = _ceil_usd_six(
            max(reported_cost, computed_cost) if reported_cost is not None else computed_cost
        )
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
            "provider_response_id_hash": stable_hash(str(response_id)) if response_id else None,
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
            "replayed_from_checkpoint": cached is not None,
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
        max_attempts: int = 1,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
        dispatch_guard: OpenRouterDispatchGuard | None = None,
        dispatch_route: str | None = None,
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
            dispatch_guard=dispatch_guard,
            dispatch_route=dispatch_route,
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
        max_attempts: int = 1,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        paid_response_store: PaidResponseStore | None = None,
        dispatch_guard: OpenRouterDispatchGuard | None = None,
        dispatch_route: str | None = None,
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
            dispatch_guard=dispatch_guard,
            dispatch_route=dispatch_route,
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
    "OpenRouterDispatchGuard",
    "OpenRouterJSONClient",
    "OpenRouterPhasePreflight",
    "OpenRouterRequestSpec",
    "ProviderError",
    "TokenPrice",
    "preflight_openrouter_phase",
]
