"""Independent construction of the paid-run approval bindings.

The expected contract is derived from five frozen sources: the run config, the
preregistration, the software/GPU lock, a fresh content-addressed RunPod quote
lock, and a fresh content-addressed API route quote lock.  It must never be
reconstructed from the approval document it is supposed to validate.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
)

from model_forensics.approval import (
    GPU_COMMAND_PHASES,
    PHASE_CONTRACT_VERSION,
    ApiQuoteBinding,
    ApprovalBindings,
    GpuBinding,
    GpuPhaseRuntimeAllocation,
    GpuQuote,
    RouteBinding,
    SpendingCaps,
)
from model_forensics.config import RunConfig
from model_forensics.io import stable_hash

GPU_QUOTE_LOCK_FILENAME = "gpu_quote_lock.json"
API_ROUTE_QUOTE_LOCK_FILENAME = "api_route_quote_lock.json"
API_ROUTE_ROLES = (
    "primary_final_and_trajectory",
    "independent_final",
    "classifier_anthropic",
    "classifier_google",
)


class GpuQuoteLockError(ValueError):
    """The independently frozen GPU quote is malformed or has drifted."""


class ApiRouteQuoteLockError(ValueError):
    """The independently frozen API route quote is malformed or has drifted."""


class ApiQuotedRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal[
        "primary_final_and_trajectory",
        "independent_final",
        "classifier_anthropic",
        "classifier_google",
    ]
    model: str = Field(min_length=3, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]+$")
    input_usd_per_million_tokens: StrictFloat
    output_usd_per_million_tokens: StrictFloat

    @field_validator("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    @classmethod
    def require_positive_finite_price(cls, value: float, info: Any) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{info.field_name} must be finite and positive")
        return value


class ApiRouteQuoteLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    provider: Literal["openrouter"]
    source_url: str
    checked_at: datetime
    routes: tuple[ApiQuotedRoute, ...]
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("source_url")
    @classmethod
    def require_https_source(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("source_url must be a credential-free HTTPS URL")
        return value

    @field_validator("checked_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value

    @field_validator("routes")
    @classmethod
    def require_exact_routes(cls, value: tuple[ApiQuotedRoute, ...]) -> tuple[ApiQuotedRoute, ...]:
        if tuple(route.role for route in value) != API_ROUTE_ROLES:
            raise ValueError("routes must contain the exact four canonical roles in order")
        return value


class GpuQuoteLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    provider: Literal["runpod"]
    quote_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$")
    gpu_family: Literal["H100_80GB", "A100_80GB"]
    provider_gpu_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._:/()+-]+$",
    )
    cloud_type: Literal["SECURE"]
    allowed_cuda_versions: tuple[Literal["12.8"], ...]
    data_center_ids: tuple[str, ...]
    gpu_count: StrictInt
    container_disk_gb: StrictInt
    volume_disk_gb: StrictInt
    usd_per_gpu_hour: StrictFloat
    running_storage_usd_per_hour: StrictFloat
    quoted_at: datetime
    phase_runtime_allocations: tuple[GpuPhaseRuntimeAllocation, ...]
    source_url: str
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("gpu_count")
    @classmethod
    def require_eight_gpus(cls, value: int) -> int:
        if isinstance(value, bool) or value != 8:
            raise ValueError("primary quote must contain exactly eight GPUs")
        return value

    @field_validator("provider_gpu_id")
    @classmethod
    def require_matching_exact_provider_gpu_id(cls, value: str, info: Any) -> str:
        family = info.data.get("gpu_family")
        expected_token = "H100" if family == "H100_80GB" else "A100"
        if re.search(rf"(?:^|[^A-Z0-9]){expected_token}(?:$|[^A-Z0-9])", value, re.I) is None:
            raise ValueError("provider_gpu_id must match gpu_family")
        return value

    @field_validator("usd_per_gpu_hour", "running_storage_usd_per_hour")
    @classmethod
    def require_positive_finite(cls, value: float, info: Any) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{info.field_name} must be finite and positive")
        return value

    @field_validator("allowed_cuda_versions")
    @classmethod
    def require_cuda_12_8_only(
        cls, value: tuple[Literal["12.8"], ...]
    ) -> tuple[Literal["12.8"], ...]:
        if value != ("12.8",):
            raise ValueError("allowed_cuda_versions must contain exactly 12.8")
        return value

    @field_validator("data_center_ids")
    @classmethod
    def require_unique_data_centers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("data_center_ids must be nonempty and unique")
        if any(re.fullmatch(r"[A-Z0-9][A-Z0-9-]{2,31}", item) is None for item in value):
            raise ValueError("data_center_ids contain an invalid provider identifier")
        return value

    @field_validator("container_disk_gb", "volume_disk_gb")
    @classmethod
    def require_frozen_storage_sizes(cls, value: int, info: Any) -> int:
        expected = 50 if info.field_name == "container_disk_gb" else 650
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"{info.field_name} must equal {expected}")
        return value

    @field_validator("quoted_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quoted_at must be timezone-aware")
        return value

    @field_validator("phase_runtime_allocations")
    @classmethod
    def require_each_canonical_phase_once(
        cls, value: tuple[GpuPhaseRuntimeAllocation, ...]
    ) -> tuple[GpuPhaseRuntimeAllocation, ...]:
        if tuple(item.command_phase for item in value) != GPU_COMMAND_PHASES:
            raise ValueError(
                "phase_runtime_allocations must contain each canonical GPU phase exactly once"
            )
        return value

    @field_validator("source_url")
    @classmethod
    def require_https_source(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("source_url must be a credential-free HTTPS URL")
        return value


def _normalize_quote_timestamp(payload: dict[str, Any]) -> None:
    value = payload.get("quoted_at")
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return
    elif isinstance(value, datetime):
        timestamp = value
    else:
        return
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return
    payload["quoted_at"] = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def gpu_quote_lock_content_hash(value: GpuQuoteLock | Mapping[str, Any]) -> str:
    if isinstance(value, GpuQuoteLock):
        payload = value.model_dump(mode="json")
    else:
        payload = deepcopy(dict(value))
    payload.pop("content_hash", None)
    _normalize_quote_timestamp(payload)
    return stable_hash(payload)


def api_route_quote_lock_content_hash(
    value: ApiRouteQuoteLock | Mapping[str, Any],
) -> str:
    if isinstance(value, ApiRouteQuoteLock):
        payload = value.model_dump(mode="json")
    else:
        payload = deepcopy(dict(value))
    payload.pop("content_hash", None)
    raw_timestamp = payload.get("checked_at")
    if isinstance(raw_timestamp, str):
        try:
            timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        except ValueError:
            timestamp = None
    elif isinstance(raw_timestamp, datetime):
        timestamp = raw_timestamp
    else:
        timestamp = None
    if timestamp is not None and timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
        payload["checked_at"] = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return stable_hash(payload)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GpuQuoteLockError(f"duplicate JSON key in GPU quote lock: {key}")
        result[key] = value
    return result


def _unique_api_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApiRouteQuoteLockError(f"duplicate JSON key in API route quote lock: {key}")
        result[key] = value
    return result


def load_api_route_quote_lock(path: str | Path) -> ApiRouteQuoteLock:
    source = Path(path)
    if source.name != API_ROUTE_QUOTE_LOCK_FILENAME:
        raise ApiRouteQuoteLockError(
            f"API route quote lock must be named {API_ROUTE_QUOTE_LOCK_FILENAME}"
        )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_api_object)
    except FileNotFoundError as exc:
        raise ApiRouteQuoteLockError(f"missing API route quote lock: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiRouteQuoteLockError(f"cannot read API route quote lock: {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ApiRouteQuoteLockError("API route quote lock must be a JSON object")
    expected_hash = api_route_quote_lock_content_hash(raw)
    if raw.get("content_hash") != expected_hash:
        raise ApiRouteQuoteLockError("API route quote lock content hash mismatch")
    try:
        return ApiRouteQuoteLock.model_validate(raw)
    except ValidationError as exc:
        raise ApiRouteQuoteLockError(f"invalid API route quote lock schema: {exc}") from exc


def load_gpu_quote_lock(path: str | Path) -> GpuQuoteLock:
    source = Path(path)
    if source.name != GPU_QUOTE_LOCK_FILENAME:
        raise GpuQuoteLockError(f"GPU quote lock must be named {GPU_QUOTE_LOCK_FILENAME}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except FileNotFoundError as exc:
        raise GpuQuoteLockError(f"missing GPU quote lock: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuQuoteLockError(f"cannot read GPU quote lock: {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise GpuQuoteLockError("GPU quote lock must be a JSON object")
    expected_hash = gpu_quote_lock_content_hash(raw)
    if raw.get("content_hash") != expected_hash:
        raise GpuQuoteLockError("GPU quote lock content hash mismatch")
    try:
        return GpuQuoteLock.model_validate(raw)
    except ValidationError as exc:
        raise GpuQuoteLockError(f"invalid GPU quote lock schema: {exc}") from exc


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _route(
    *,
    role: str,
    provider: str,
    source: Mapping[str, Any],
) -> RouteBinding:
    return RouteBinding(
        role=role,
        provider=provider,
        model=str(source["model"]),
        input_usd_per_million_tokens=float(source["input_usd_per_million_tokens"]),
        output_usd_per_million_tokens=float(source["output_usd_per_million_tokens"]),
    )


def build_approval_bindings(
    *,
    config: RunConfig,
    preregistration: Mapping[str, Any],
    gpu_lock: Mapping[str, Any],
    quote_lock: GpuQuoteLock,
    api_quote_lock: ApiRouteQuoteLock,
) -> ApprovalBindings:
    """Derive the exact expected approval contract without reading approval data."""

    external = _mapping(preregistration.get("external_judging"), label="external_judging")
    provider = str(external.get("provider_gateway", ""))
    if provider != "openrouter":
        raise ValueError("external judging provider must be frozen as openrouter")
    primary = _mapping(
        external.get("high_volume_outcome_and_trajectory"),
        label="high_volume_outcome_and_trajectory",
    )
    calibration = _mapping(external.get("outcome_calibration"), label="outcome_calibration")
    semantic_raw = external.get("semantic_classification_routes")
    if not isinstance(semantic_raw, list) or len(semantic_raw) != 2:
        raise ValueError("exactly two semantic classification routes must be frozen")
    semantic = [_mapping(item, label="semantic route") for item in semantic_raw]
    semantic_by_role = {str(item.get("role")): item for item in semantic}
    expected_semantic_roles = {
        "strongest_anthropic_route",
        "independent_frontier_google_route",
    }
    if set(semantic_by_role) != expected_semantic_roles:
        raise ValueError("semantic classification route roles disagree with the frozen design")
    if calibration.get("primary_model") != primary.get("model"):
        raise ValueError("outcome primary model disagrees with the trajectory route")
    independent_model = str(calibration.get("independent_model", ""))
    independent_matches = [item for item in semantic if item.get("model") == independent_model]
    if len(independent_matches) != 1:
        raise ValueError("independent final model lacks one exact frozen price route")
    primary_semantic = semantic_by_role["strongest_anthropic_route"]
    if primary_semantic.get("model") != primary.get("model"):
        raise ValueError("Anthropic classifier route must match the primary frontier model")
    for price_field in (
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
    ):
        if float(primary_semantic[price_field]) != float(primary[price_field]):
            raise ValueError("primary route prices disagree across frozen roles")

    container = _mapping(gpu_lock.get("container_image"), label="container_image")
    repositories = _mapping(gpu_lock.get("source_repositories"), label="source_repositories")
    vllm = _mapping(repositories.get("vllm"), label="source_repositories.vllm")
    image_reference = str(container.get("reference", ""))
    wheel_sha256 = str(vllm.get("wheel_sha256", ""))
    preregistered_routes = (
        _route(role="primary_final_and_trajectory", provider=provider, source=primary),
        _route(
            role="independent_final",
            provider=provider,
            source=independent_matches[0],
        ),
        _route(
            role="classifier_anthropic",
            provider=provider,
            source=primary_semantic,
        ),
        _route(
            role="classifier_google",
            provider=provider,
            source=semantic_by_role["independent_frontier_google_route"],
        ),
    )
    quoted_routes = tuple(
        RouteBinding(
            role=route.role,
            provider=api_quote_lock.provider,
            model=route.model,
            input_usd_per_million_tokens=route.input_usd_per_million_tokens,
            output_usd_per_million_tokens=route.output_usd_per_million_tokens,
        )
        for route in api_quote_lock.routes
    )
    if quoted_routes != preregistered_routes:
        raise ValueError("API route quote does not exactly match the preregistered routes")
    return ApprovalBindings(
        phase_contract_version=PHASE_CONTRACT_VERSION,
        config_hash=stable_hash(config.model_dump(mode="json", exclude={"source_path"})),
        preregistration_hash=stable_hash(dict(preregistration)),
        gpu_lock_hash=stable_hash(dict(gpu_lock)),
        gpu=GpuBinding(
            family=quote_lock.gpu_family,
            provider_gpu_id=quote_lock.provider_gpu_id,
            cloud_type=quote_lock.cloud_type,
            allowed_cuda_versions=quote_lock.allowed_cuda_versions,
            data_center_ids=quote_lock.data_center_ids,
            count=quote_lock.gpu_count,
            container_disk_gb=quote_lock.container_disk_gb,
            volume_disk_gb=quote_lock.volume_disk_gb,
            quote=GpuQuote(
                provider=quote_lock.provider,
                quote_id=quote_lock.quote_id,
                usd_per_gpu_hour=quote_lock.usd_per_gpu_hour,
                running_storage_usd_per_hour=quote_lock.running_storage_usd_per_hour,
                quoted_at=quote_lock.quoted_at,
                source_url=quote_lock.source_url,
                content_hash=quote_lock.content_hash,
            ),
            phase_runtime_allocations=quote_lock.phase_runtime_allocations,
            container_image_digest=image_reference,
            vllm_wheel_sha256=wheel_sha256,
        ),
        api_quote=ApiQuoteBinding(
            provider=api_quote_lock.provider,
            source_url=api_quote_lock.source_url,
            checked_at=api_quote_lock.checked_at,
            content_hash=api_quote_lock.content_hash,
        ),
        caps_usd=SpendingCaps(
            gpu=float(config.execution.gpu_cost_hard_stop_usd),
            api=float(config.execution.api_cost_hard_stop_usd),
            total=float(config.execution.total_cost_hard_stop_usd),
        ),
        routes=quoted_routes,
    )


__all__ = [
    "API_ROUTE_QUOTE_LOCK_FILENAME",
    "API_ROUTE_ROLES",
    "GPU_QUOTE_LOCK_FILENAME",
    "ApiQuotedRoute",
    "ApiRouteQuoteLock",
    "ApiRouteQuoteLockError",
    "GpuQuoteLock",
    "GpuQuoteLockError",
    "api_route_quote_lock_content_hash",
    "build_approval_bindings",
    "gpu_quote_lock_content_hash",
    "load_api_route_quote_lock",
    "load_gpu_quote_lock",
]
