"""Offline, content-addressed approval gate for every paid execution phase.

This module deliberately has no provider SDK or network imports.  Call
``load_paid_run_approval`` and ``validate_paid_run_approval`` before constructing
an API client or loading a GPU model.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime, timedelta
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
    model_validator,
)

from model_forensics.io import stable_hash

APPROVAL_FILENAME = "paid_run_approval.json"
APPROVAL_SCHEMA_VERSION = 2
PHASE_CONTRACT_VERSION = "gpu-api-phase-split-v2"
MAX_GPU_QUOTE_AGE = timedelta(hours=6)
MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)
PAID_COMMAND_PHASES = frozenset(
    {
        "behavior_baseline_gpu",
        "behavior_baseline_api",
        "behavior_treatment_gpu",
        "behavior_treatment_api",
        "anchors_api",
        "resample_gpu",
        "resample_api",
        "positions_api",
        "lens_gpu",
    }
)
GPU_COMMAND_PHASES = (
    "behavior_baseline_gpu",
    "behavior_treatment_gpu",
    "resample_gpu",
    "lens_gpu",
)
API_COMMAND_PHASES = frozenset(phase for phase in PAID_COMMAND_PHASES if phase.endswith("_api"))
MAX_API_QUOTE_AGE = timedelta(hours=6)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_DIGEST_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,127}$")
_APPROVAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_PLACEHOLDER_RE = re.compile(
    r"(?:<[^>]+>|\$\{[^}]+\}|\b(?:changeme|dummy|example|placeholder|todo|tbd|unknown|pending)\b|"
    r"(?:replace|insert|enter)[_-]?(?:me|here|with[_-][a-z0-9_-]+)|"
    r"your[_-][a-z0-9_-]+)",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|"
    r"github_pat_[A-Za-z0-9_]{12,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|Bearer\s+\S+)",
    re.IGNORECASE,
)


class PaidRunApprovalError(ValueError):
    """Raised when paid execution has no exact, current user approval."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_placeholder(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    if _PLACEHOLDER_RE.search(value):
        raise ValueError(f"{field_name} contains a placeholder")
    return value


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _require_finite_positive(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return value


def _require_finite_nonnegative(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return value


def _reject_degenerate_digest(value: str, *, field_name: str) -> str:
    digest = value.rsplit("sha256:", maxsplit=1)[-1]
    if len(set(digest)) == 1:
        raise ValueError(f"{field_name} contains a placeholder digest")
    return value


class GpuQuote(_StrictModel):
    provider: str
    quote_id: str
    usd_per_gpu_hour: StrictFloat
    running_storage_usd_per_hour: StrictFloat
    quoted_at: datetime
    source_url: str
    content_hash: str

    @field_validator("provider", "quote_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        value = _reject_placeholder(value, field_name=info.field_name)
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} is not a valid identifier")
        return value

    @field_validator("usd_per_gpu_hour", "running_storage_usd_per_hour")
    @classmethod
    def validate_rate(cls, value: float, info: Any) -> float:
        return _require_finite_positive(value, field_name=info.field_name)

    @field_validator("quoted_at")
    @classmethod
    def validate_quoted_at(cls, value: datetime) -> datetime:
        return _require_aware(value, field_name="quoted_at")

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="source_url")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("source_url must be a credential-free HTTPS URL")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError("content_hash must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name="content_hash")


class GpuPhaseRuntimeAllocation(_StrictModel):
    command_phase: Literal[
        "behavior_baseline_gpu",
        "behavior_treatment_gpu",
        "resample_gpu",
        "lens_gpu",
    ]
    maximum_runtime_hours: StrictFloat

    @field_validator("maximum_runtime_hours")
    @classmethod
    def validate_runtime(cls, value: float) -> float:
        return _require_finite_positive(value, field_name="maximum_runtime_hours")


class GpuBinding(_StrictModel):
    family: str
    provider_gpu_id: str
    cloud_type: Literal["SECURE"]
    allowed_cuda_versions: tuple[str, ...]
    data_center_ids: tuple[str, ...]
    count: StrictInt = Field(ge=1)
    container_disk_gb: StrictInt
    volume_disk_gb: StrictInt
    quote: GpuQuote
    phase_runtime_allocations: tuple[GpuPhaseRuntimeAllocation, ...]
    container_image_digest: str
    vllm_wheel_sha256: str

    @field_validator("family", "provider_gpu_id")
    @classmethod
    def validate_gpu_identity(cls, value: str, info: Any) -> str:
        return _reject_placeholder(value, field_name=info.field_name)

    @field_validator("allowed_cuda_versions")
    @classmethod
    def validate_cuda_versions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("12.8",):
            raise ValueError("allowed_cuda_versions must be exactly ('12.8',)")
        return value

    @field_validator("data_center_ids")
    @classmethod
    def validate_data_center_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("data_center_ids must be nonempty and unique")
        for item in value:
            _reject_placeholder(item, field_name="data_center_ids")
        return value

    @field_validator("container_disk_gb", "volume_disk_gb", mode="before")
    @classmethod
    def reject_boolean_storage_size(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("storage sizes must be integers")
        return value

    @field_validator("container_disk_gb")
    @classmethod
    def validate_container_disk(cls, value: int) -> int:
        if value != 50:
            raise ValueError("container_disk_gb must equal the frozen 50 GB launch size")
        return value

    @field_validator("volume_disk_gb")
    @classmethod
    def validate_volume_disk(cls, value: int) -> int:
        if value != 650:
            raise ValueError("volume_disk_gb must equal the frozen 650 GB launch size")
        return value

    @field_validator("count", mode="before")
    @classmethod
    def reject_boolean_count(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("count must be an integer")
        return value

    @field_validator("phase_runtime_allocations")
    @classmethod
    def validate_phase_runtime_allocations(
        cls, value: tuple[GpuPhaseRuntimeAllocation, ...]
    ) -> tuple[GpuPhaseRuntimeAllocation, ...]:
        phases = tuple(item.command_phase for item in value)
        if phases != GPU_COMMAND_PHASES:
            raise ValueError(
                "phase_runtime_allocations must contain each canonical GPU phase exactly once"
            )
        return value

    @field_validator("container_image_digest")
    @classmethod
    def validate_container_digest(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="container_image_digest")
        if not _CONTAINER_DIGEST_RE.fullmatch(value):
            raise ValueError("container_image_digest must pin an image by sha256 digest")
        return _reject_degenerate_digest(value, field_name="container_image_digest")

    @field_validator("vllm_wheel_sha256")
    @classmethod
    def validate_wheel_hash(cls, value: str) -> str:
        if not _RAW_SHA256_RE.fullmatch(value):
            raise ValueError("vllm_wheel_sha256 must be 64 lowercase hexadecimal characters")
        return _reject_degenerate_digest(value, field_name="vllm_wheel_sha256")


class SpendingCaps(_StrictModel):
    gpu: StrictFloat
    api: StrictFloat
    total: StrictFloat

    @field_validator("gpu", "api", "total")
    @classmethod
    def validate_cap(cls, value: float, info: Any) -> float:
        return _require_finite_nonnegative(value, field_name=info.field_name)

    @model_validator(mode="after")
    def total_covers_categories(self) -> SpendingCaps:
        if self.total < self.gpu + self.api:
            raise ValueError("total cap must cover the GPU plus API caps")
        return self


class RouteBinding(_StrictModel):
    role: str
    provider: str
    model: str
    input_usd_per_million_tokens: StrictFloat
    output_usd_per_million_tokens: StrictFloat

    @field_validator("role", "provider", "model")
    @classmethod
    def validate_route_identifiers(cls, value: str, info: Any) -> str:
        value = _reject_placeholder(value, field_name=info.field_name)
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} is not a valid identifier")
        return value

    @field_validator("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    @classmethod
    def validate_price(cls, value: float, info: Any) -> float:
        return _require_finite_positive(value, field_name=info.field_name)


class ApiQuoteBinding(_StrictModel):
    provider: str
    source_url: str
    checked_at: datetime
    content_hash: str

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="provider")
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("provider is not a valid identifier")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="source_url")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("source_url must be a credential-free HTTPS URL")
        return value

    @field_validator("checked_at")
    @classmethod
    def validate_checked_at(cls, value: datetime) -> datetime:
        return _require_aware(value, field_name="checked_at")

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError("content_hash must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name="content_hash")


class ApprovalBindings(_StrictModel):
    phase_contract_version: Literal[PHASE_CONTRACT_VERSION]
    config_hash: str
    preregistration_hash: str
    gpu_lock_hash: str
    gpu: GpuBinding
    api_quote: ApiQuoteBinding
    caps_usd: SpendingCaps
    routes: tuple[RouteBinding, ...]

    @field_validator("phase_contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        return _reject_placeholder(value, field_name="phase_contract_version")

    @field_validator("config_hash", "preregistration_hash", "gpu_lock_hash")
    @classmethod
    def validate_namespaced_hash(cls, value: str, info: Any) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name=info.field_name)

    @field_validator("routes")
    @classmethod
    def validate_routes(cls, value: tuple[RouteBinding, ...]) -> tuple[RouteBinding, ...]:
        if not value:
            raise ValueError("routes must not be empty")
        roles = [route.role for route in value]
        if len(set(roles)) != len(roles):
            raise ValueError("route roles must be unique")
        return value

    @model_validator(mode="after")
    def projected_gpu_cost_fits_cap(self) -> ApprovalBindings:
        allocated_hours = sum(
            allocation.maximum_runtime_hours for allocation in self.gpu.phase_runtime_allocations
        )
        projected = (
            self.gpu.count * self.gpu.quote.usd_per_gpu_hour
            + self.gpu.quote.running_storage_usd_per_hour
        ) * allocated_hours
        if projected > self.caps_usd.gpu:
            raise ValueError("approved GPU phase runtime allocations would exceed the GPU cap")
        return self


class UserApproval(_StrictModel):
    approval_id: str
    approved_at: datetime

    @field_validator("approval_id")
    @classmethod
    def validate_approval_id(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="approval_id")
        if not _APPROVAL_ID_RE.fullmatch(value):
            raise ValueError("approval_id must be a non-secret identifier")
        if _SECRET_RE.search(value):
            raise ValueError("approval_id appears to contain a secret")
        return value

    @field_validator("approved_at")
    @classmethod
    def validate_approved_at(cls, value: datetime) -> datetime:
        return _require_aware(value, field_name="approved_at")


class PaidRunApproval(_StrictModel):
    schema_version: Literal[APPROVAL_SCHEMA_VERSION]
    bindings: ApprovalBindings
    allowed_command_phases: tuple[str, ...]
    user_approval: UserApproval
    content_hash: str

    @field_validator("allowed_command_phases")
    @classmethod
    def validate_phases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("allowed_command_phases must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("allowed_command_phases must be unique")
        for phase in value:
            _reject_placeholder(phase, field_name="command phase")
            if not _IDENTIFIER_RE.fullmatch(phase):
                raise ValueError("command phase is not a valid identifier")
            if phase not in PAID_COMMAND_PHASES:
                raise ValueError(f"command phase is not canonical: {phase}")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash_format(cls, value: str) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError("content_hash must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name="content_hash")


def approval_content_hash(value: PaidRunApproval | Mapping[str, Any]) -> str:
    """Hash every approval field except the self-referential ``content_hash``."""

    if isinstance(value, PaidRunApproval):
        payload = value.model_dump(mode="json")
    else:
        payload = deepcopy(dict(value))
    payload.pop("content_hash", None)
    _normalize_hash_timestamp(payload, ("bindings", "gpu", "quote", "quoted_at"))
    _normalize_hash_timestamp(payload, ("bindings", "api_quote", "checked_at"))
    _normalize_hash_timestamp(payload, ("user_approval", "approved_at"))
    return stable_hash(payload)


def _normalize_hash_timestamp(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    target: Any = payload
    try:
        for component in path[:-1]:
            target = target[component]
        value = target[path[-1]]
    except (KeyError, TypeError):
        return
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
    target[path[-1]] = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _mapping_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PaidRunApprovalError(f"duplicate JSON key in paid-run approval: {key}")
        value[key] = item
    return value


def load_paid_run_approval(path: str | Path) -> PaidRunApproval:
    """Load and authenticate a local approval document without network access."""

    source = Path(path)
    if source.name != APPROVAL_FILENAME:
        raise PaidRunApprovalError(f"approval file must be named {APPROVAL_FILENAME}")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_mapping_without_duplicate_keys,
        )
    except FileNotFoundError as exc:
        raise PaidRunApprovalError(f"missing paid-run approval: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidRunApprovalError(f"cannot read paid-run approval: {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PaidRunApprovalError("paid-run approval must be a JSON object")
    try:
        approval = PaidRunApproval.model_validate(raw)
    except ValidationError as exc:
        raise PaidRunApprovalError(f"invalid paid-run approval schema: {exc}") from exc
    expected_hash = approval_content_hash(raw)
    if approval.content_hash != expected_hash:
        raise PaidRunApprovalError("paid-run approval content hash mismatch")
    return approval


def validate_paid_run_approval(
    approval: PaidRunApproval,
    *,
    expected: ApprovalBindings,
    command_phase: str,
    now: datetime | None = None,
) -> PaidRunApproval:
    """Require an exact binding and an explicitly approved paid command phase."""

    current = now or datetime.now(UTC)
    _require_aware(current, field_name="now")
    if approval.content_hash != approval_content_hash(approval):
        raise PaidRunApprovalError("paid-run approval content hash mismatch")
    if approval.bindings != expected:
        raise PaidRunApprovalError("paid-run approval bindings do not match the frozen run")
    if command_phase not in approval.allowed_command_phases:
        raise PaidRunApprovalError(f"command phase is not approved: {command_phase}")

    quoted_at = approval.bindings.gpu.quote.quoted_at
    quote_age = current.astimezone(UTC) - quoted_at.astimezone(UTC)
    # Quote freshness controls GPU acquisition only.  Once a generation
    # artifact is synced and compute is stopped, expiring the hardware quote
    # must not force a scientifically irrelevant reapproval of the API-only
    # adjudication phase.
    if command_phase.endswith("_gpu"):
        if quote_age > MAX_GPU_QUOTE_AGE:
            raise PaidRunApprovalError("GPU quote is older than six hours")
        if quote_age < -MAX_FUTURE_CLOCK_SKEW:
            raise PaidRunApprovalError("GPU quote timestamp is in the future")

    api_checked_at = approval.bindings.api_quote.checked_at
    api_quote_age = current.astimezone(UTC) - api_checked_at.astimezone(UTC)
    if command_phase in API_COMMAND_PHASES:
        if api_quote_age > MAX_API_QUOTE_AGE:
            raise PaidRunApprovalError("API quote is older than six hours")
        if api_quote_age < -MAX_FUTURE_CLOCK_SKEW:
            raise PaidRunApprovalError("API quote timestamp is in the future")

    approved_at = approval.user_approval.approved_at
    if command_phase.endswith("_gpu") and approved_at < quoted_at:
        raise PaidRunApprovalError("user approval predates the GPU quote")
    if command_phase in API_COMMAND_PHASES and approved_at < api_checked_at:
        raise PaidRunApprovalError("user approval predates the API quote")
    if approved_at.astimezone(UTC) - current.astimezone(UTC) > MAX_FUTURE_CLOCK_SKEW:
        raise PaidRunApprovalError("user approval timestamp is in the future")
    return approval
