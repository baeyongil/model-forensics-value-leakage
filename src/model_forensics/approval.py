"""Offline, content-addressed approval gate for every paid execution phase.

This module deliberately has no provider SDK or network imports.  Call
``load_paid_run_approval`` and ``validate_paid_run_approval`` before constructing
an API client or loading a GPU model.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path, PurePosixPath
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
APPROVAL_SCHEMA_VERSION = 4
PHASE_CONTRACT_VERSION = "gpu-api-phase-split-v2"
PAID_RUN_REVIEW_PROTOCOL_VERSION = "paid-run-review-v2"
MAX_GPU_QUOTE_AGE = timedelta(hours=6)
MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)
PAID_COMMAND_PHASE_ORDER = (
    "behavior_baseline_gpu",
    "behavior_baseline_api",
    "behavior_treatment_gpu",
    "behavior_treatment_api",
    "anchors_api",
    "resample_gpu",
    "resample_api",
    "positions_api",
    "lens_gpu",
)
PAID_COMMAND_PHASES = frozenset(PAID_COMMAND_PHASE_ORDER)
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
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
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


def canonicalize_paid_command_phases(phases: Sequence[str]) -> tuple[str, ...]:
    """Return one deterministic, nonempty paid-command authorization scope."""

    values = tuple(phases)
    if not values:
        raise ValueError("paid command phase scope must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("paid command phase scope must be unique")
    for phase in values:
        _reject_placeholder(phase, field_name="command phase")
        if not _IDENTIFIER_RE.fullmatch(phase):
            raise ValueError("command phase is not a valid identifier")
        if phase not in PAID_COMMAND_PHASES:
            raise ValueError(f"command phase is not canonical: {phase}")
    selected = set(values)
    return tuple(phase for phase in PAID_COMMAND_PHASE_ORDER if phase in selected)


def _git_output(project_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "--no-optional-locks",
            *arguments,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise PaidRunApprovalError("project source checkout cannot be authenticated")
    return completed.stdout


def _project_relative_path(project_root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as exc:
        raise PaidRunApprovalError("mutable paid-run artifact is outside the project") from exc
    if not relative.parts:
        raise PaidRunApprovalError("project root cannot be a mutable paid-run artifact")
    return relative.as_posix()


def require_clean_source_commit(
    project_root: str | Path,
    *,
    mutable_paths: tuple[str | Path, ...] = (),
) -> str:
    """Authenticate one exact clean Git HEAD for a paid-operation boundary.

    Private ``.runpod`` state and explicitly named mutable accounting artifacts
    are not source. Every tracked change and every non-ignored untracked file
    outside those narrow locations fails closed. The complete Git observation
    is repeated so a commit or worktree transition during the check cannot
    yield a mixed snapshot.
    """

    root_input = Path(project_root)
    if root_input.is_symlink():
        raise PaidRunApprovalError("project source root must not be a symlink")
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        raise PaidRunApprovalError("project source root is unavailable") from exc
    if not root.is_dir():
        raise PaidRunApprovalError("project source root is not a directory")

    top_level = Path(
        os.fsdecode(_git_output(root, "rev-parse", "--show-toplevel")).strip()
    )
    try:
        authenticated_top_level = top_level.resolve(strict=True)
    except OSError as exc:
        raise PaidRunApprovalError("Git worktree root is unavailable") from exc
    if authenticated_top_level != root:
        raise PaidRunApprovalError("paid execution must run from the Git worktree root")

    mutable = {_project_relative_path(root, path) for path in mutable_paths}
    mutable_locks = {
        str(Path(path).with_name(f".{Path(path).name}.lock")) for path in mutable
    }

    def snapshot() -> tuple[str, frozenset[str], frozenset[str], frozenset[str]]:
        commit_before = os.fsdecode(
            _git_output(root, "rev-parse", "--verify", "HEAD^{commit}")
        ).strip()
        if _SOURCE_COMMIT_RE.fullmatch(commit_before) is None:
            raise PaidRunApprovalError("project source commit is malformed")
        tracked = frozenset(
            os.fsdecode(item)
            for item in _git_output(
                root,
                "diff",
                "--no-ext-diff",
                "--ignore-submodules=none",
                "--name-only",
                "-z",
                "HEAD",
                "--",
            ).split(b"\0")
            if item
        )
        untracked = frozenset(
            os.fsdecode(item)
            for item in _git_output(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ).split(b"\0")
            if item
        )
        index_flags = frozenset(
            os.fsdecode(item)
            for item in _git_output(root, "ls-files", "-v", "-z").split(b"\0")
            if item and (item[:1].islower() or item.startswith(b"S "))
        )
        commit_after = os.fsdecode(
            _git_output(root, "rev-parse", "--verify", "HEAD^{commit}")
        ).strip()
        if commit_after != commit_before:
            raise PaidRunApprovalError("project source commit changed during authentication")
        return commit_before, tracked, untracked, index_flags

    first = snapshot()
    second = snapshot()
    if first != second:
        raise PaidRunApprovalError("project source changed during authentication")
    commit, tracked, untracked, index_flags = second
    if index_flags:
        raise PaidRunApprovalError("project source index contains hidden worktree flags")
    if tracked - mutable:
        raise PaidRunApprovalError("tracked project source differs from the reviewed Git commit")
    unexpected_untracked = {
        item
        for item in untracked
        if item not in mutable | mutable_locks
        and item != ".runpod"
        and not item.startswith(".runpod/")
    }
    if unexpected_untracked:
        raise PaidRunApprovalError(
            "untracked project source is not included in the reviewed Git commit"
        )
    return commit


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
        # Match the paid reservation gate exactly: every phase is rounded
        # upward to ledger precision before the phase ceilings are summed.
        # Multiplying the aggregate hours and rounding only once can
        # understate the approved commitment by a few micro-dollars.
        from model_forensics.gpu_budget import approved_gpu_phase_maximum_usd

        projected = sum(
            approved_gpu_phase_maximum_usd(
                gpu_count=self.gpu.count,
                quote_hourly_per_gpu_usd=self.gpu.quote.usd_per_gpu_hour,
                running_storage_hourly_usd=self.gpu.quote.running_storage_usd_per_hour,
                approved_runtime_hours=allocation.maximum_runtime_hours,
            )
            for allocation in self.gpu.phase_runtime_allocations
        )
        if projected > self.caps_usd.gpu:
            raise ValueError("approved GPU phase runtime allocations would exceed the GPU cap")
        return self


class PaidRunReviewContextHashes(_StrictModel):
    config: str
    preregistration: str
    gpu_lock: str
    gpu_quote_lock: str
    api_quote_lock: str
    bindings: str

    @field_validator(
        "config",
        "preregistration",
        "gpu_lock",
        "gpu_quote_lock",
        "api_quote_lock",
        "bindings",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name=info.field_name)


class PaidRunReviewLedger(_StrictModel):
    path: str
    bytes_sha256: str
    document_hash: str
    byte_count: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = _reject_placeholder(value, field_name="ledger path")
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or not parsed.parts
            or parsed.as_posix() != value
            or "\\" in value
            or any(ord(character) < 0x20 for character in value)
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("ledger path must be a normalized project-relative path")
        return value

    @field_validator("bytes_sha256", "document_hash")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name=info.field_name)


class PaidRunReviewPhaseMaximum(_StrictModel):
    command_phase: Literal[
        "behavior_baseline_gpu",
        "behavior_treatment_gpu",
        "resample_gpu",
        "lens_gpu",
    ]
    maximum_usd: StrictFloat

    @field_validator("maximum_usd")
    @classmethod
    def validate_maximum(cls, value: float) -> float:
        return _require_finite_positive(value, field_name="maximum_usd")


class PaidRunReviewCostTotals(_StrictModel):
    gpu: StrictFloat
    api: StrictFloat
    storage: StrictFloat
    other: StrictFloat
    total: StrictFloat

    @field_validator("gpu", "api", "storage", "other", "total")
    @classmethod
    def validate_total(cls, value: float, info: Any) -> float:
        return _require_finite_nonnegative(value, field_name=info.field_name)

    @model_validator(mode="after")
    def categories_sum_to_total(self) -> PaidRunReviewCostTotals:
        expected = round(self.gpu + self.api + self.storage + self.other, 6)
        if abs(self.total - expected) > 1e-6:
            raise ValueError("review ledger category totals are inconsistent")
        return self


class PaidRunReviewCumulativeCost(_StrictModel):
    ledger_incurred: PaidRunReviewCostTotals
    ledger_committed: PaidRunReviewCostTotals
    future_gpu_phase_maxima_usd: StrictFloat
    gpu_worst_case_usd: StrictFloat
    gpu_safety_margin_fraction: StrictFloat
    gpu_safety_adjusted_ceiling_usd: StrictFloat
    gpu_safety_headroom_usd: StrictFloat
    gpu_hard_stop_headroom_usd: StrictFloat
    api_hard_stop_usd: StrictFloat
    total_worst_case_usd: StrictFloat
    total_hard_stop_headroom_usd: StrictFloat

    @field_validator(
        "future_gpu_phase_maxima_usd",
        "gpu_worst_case_usd",
        "gpu_safety_adjusted_ceiling_usd",
        "gpu_safety_headroom_usd",
        "gpu_hard_stop_headroom_usd",
        "api_hard_stop_usd",
        "total_worst_case_usd",
        "total_hard_stop_headroom_usd",
    )
    @classmethod
    def validate_cost(cls, value: float, info: Any) -> float:
        return _require_finite_nonnegative(value, field_name=info.field_name)

    @field_validator("gpu_safety_margin_fraction")
    @classmethod
    def validate_margin(cls, value: float) -> float:
        if not math.isfinite(value) or not 0 < value < 0.25:
            raise ValueError("gpu_safety_margin_fraction must be in (0, 0.25)")
        return value


class PaidRunReviewPayload(_StrictModel):
    protocol_version: Literal[PAID_RUN_REVIEW_PROTOCOL_VERSION]
    source_commit: str
    context_hashes: PaidRunReviewContextHashes
    ledger: PaidRunReviewLedger
    planned_command_phases: tuple[str, ...]
    phase_maxima_usd: tuple[PaidRunReviewPhaseMaximum, ...]
    caps_usd: SpendingCaps
    cumulative_cost: PaidRunReviewCumulativeCost

    @field_validator("source_commit")
    @classmethod
    def validate_source_commit(cls, value: str) -> str:
        if _SOURCE_COMMIT_RE.fullmatch(value) is None:
            raise ValueError("source_commit must be an exact lowercase Git commit")
        return value

    @field_validator("planned_command_phases")
    @classmethod
    def validate_planned_phases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = canonicalize_paid_command_phases(value)
        if value != canonical:
            raise ValueError("planned command phases must use canonical order")
        return value

    @field_validator("phase_maxima_usd")
    @classmethod
    def validate_phase_maxima(
        cls,
        value: tuple[PaidRunReviewPhaseMaximum, ...],
    ) -> tuple[PaidRunReviewPhaseMaximum, ...]:
        if tuple(item.command_phase for item in value) != GPU_COMMAND_PHASES:
            raise ValueError("review phase maxima must contain every canonical GPU phase in order")
        return value

    @model_validator(mode="after")
    def cumulative_values_match_review(self) -> PaidRunReviewPayload:
        planned = frozenset(self.planned_command_phases)
        future = round(
            sum(
                item.maximum_usd
                for item in self.phase_maxima_usd
                if item.command_phase in planned
            ),
            6,
        )
        cumulative = self.cumulative_cost
        incurred = cumulative.ledger_incurred
        committed = cumulative.ledger_committed
        for field in ("gpu", "api", "storage", "other", "total"):
            if getattr(incurred, field) > getattr(committed, field) + 1e-6:
                raise ValueError("review incurred ledger totals exceed committed totals")
        if committed.api > self.caps_usd.api + 1e-6:
            raise ValueError("review committed API total exceeds its hard stop")
        if committed.total > self.caps_usd.total + 1e-6:
            raise ValueError("review committed ledger total exceeds its hard stop")
        if abs(committed.gpu - incurred.gpu) > 1e-6:
            raise ValueError("review requires no outstanding GPU reservation")
        if abs(cumulative.future_gpu_phase_maxima_usd - future) > 1e-6:
            raise ValueError("review phase maxima do not match the cumulative GPU commitment")
        if abs(cumulative.gpu_worst_case_usd - (committed.gpu + future)) > 1e-6:
            raise ValueError("review cumulative GPU commitment is inconsistent")
        expected_safety_ceiling = float(
            (
                Decimal(str(self.caps_usd.gpu))
                * (Decimal("1") - Decimal(str(cumulative.gpu_safety_margin_fraction)))
            ).quantize(Decimal("0.000001"), rounding=ROUND_FLOOR)
        )
        if abs(cumulative.gpu_safety_adjusted_ceiling_usd - expected_safety_ceiling) > 1e-6:
            raise ValueError("review safety-adjusted GPU ceiling is inconsistent")
        if abs(
            cumulative.gpu_safety_headroom_usd
            - (expected_safety_ceiling - cumulative.gpu_worst_case_usd)
        ) > 1e-6:
            raise ValueError("review safety-adjusted GPU headroom is inconsistent")
        if abs(cumulative.api_hard_stop_usd - self.caps_usd.api) > 1e-6:
            raise ValueError("review API hard stop is inconsistent")
        if abs(
            cumulative.gpu_hard_stop_headroom_usd
            - (self.caps_usd.gpu - cumulative.gpu_worst_case_usd)
        ) > 1e-6:
            raise ValueError("review GPU hard-stop headroom is inconsistent")
        expected_total_worst = round(
            cumulative.gpu_worst_case_usd
            + self.caps_usd.api
            + committed.storage
            + committed.other,
            6,
        )
        if abs(cumulative.total_worst_case_usd - expected_total_worst) > 1e-6:
            raise ValueError("review cumulative total commitment is inconsistent")
        if abs(
            cumulative.total_hard_stop_headroom_usd
            - (self.caps_usd.total - cumulative.total_worst_case_usd)
        ) > 1e-6:
            raise ValueError("review total hard-stop headroom is inconsistent")
        return self


def paid_run_review_hash(value: PaidRunReviewPayload | Mapping[str, Any]) -> str:
    """Hash the complete user-visible paid-run review payload."""

    payload = value.model_dump(mode="json") if isinstance(value, PaidRunReviewPayload) else dict(value)
    return stable_hash(payload)


class PaidRunReview(_StrictModel):
    payload: PaidRunReviewPayload
    review_hash: str

    @field_validator("review_hash")
    @classmethod
    def validate_review_hash_format(cls, value: str) -> str:
        if not _HASH_RE.fullmatch(value):
            raise ValueError("review_hash must be a namespaced SHA-256 hash")
        return _reject_degenerate_digest(value, field_name="review_hash")

    @model_validator(mode="after")
    def content_hash_matches_payload(self) -> PaidRunReview:
        if self.review_hash != paid_run_review_hash(self.payload):
            raise ValueError("review_hash does not authenticate the review payload")
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
    review: PaidRunReview
    allowed_command_phases: tuple[str, ...]
    user_approval: UserApproval
    content_hash: str

    @field_validator("allowed_command_phases")
    @classmethod
    def validate_phases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = canonicalize_paid_command_phases(value)
        if value != canonical:
            raise ValueError("allowed_command_phases must use canonical order")
        return value

    @model_validator(mode="after")
    def approval_scope_matches_review(self) -> PaidRunApproval:
        if self.allowed_command_phases != self.review.payload.planned_command_phases:
            raise ValueError("approved command phases do not match the user-reviewed scope")
        return self

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


def _stable_approval_bytes(source: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError as exc:
        raise PaidRunApprovalError(f"missing paid-run approval: {source}") from exc
    except OSError as exc:
        raise PaidRunApprovalError(f"cannot safely open paid-run approval: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise PaidRunApprovalError(
                "paid-run approval must be an owned, non-linked regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != after.st_size:
            raise PaidRunApprovalError("paid-run approval changed while it was being read")
        try:
            current = source.lstat()
        except OSError as exc:
            raise PaidRunApprovalError("paid-run approval path changed while it was read") from exc
        if (
            current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
            or stat.S_ISLNK(current.st_mode)
        ):
            raise PaidRunApprovalError("paid-run approval path changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def load_paid_run_approval(path: str | Path) -> PaidRunApproval:
    """Load and authenticate a local approval document without network access."""

    source = Path(path)
    if source.name != APPROVAL_FILENAME:
        raise PaidRunApprovalError(f"approval file must be named {APPROVAL_FILENAME}")
    try:
        raw = json.loads(
            _stable_approval_bytes(source).decode("utf-8"),
            object_pairs_hook=_mapping_without_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    expected_source_commit: str | None = None,
    expected_ledger_path: str | None = None,
) -> PaidRunApproval:
    """Require an exact binding and an explicitly approved paid command phase."""

    current = now or datetime.now(UTC)
    _require_aware(current, field_name="now")
    try:
        revalidated = PaidRunApproval.model_validate(approval.model_dump(mode="python"))
    except ValidationError as exc:
        raise PaidRunApprovalError(f"invalid in-memory paid-run approval: {exc}") from exc
    if revalidated != approval:
        raise PaidRunApprovalError("in-memory paid-run approval changed during validation")
    if approval.content_hash != approval_content_hash(approval):
        raise PaidRunApprovalError("paid-run approval content hash mismatch")
    if approval.bindings != expected:
        raise PaidRunApprovalError("paid-run approval bindings do not match the frozen run")
    if command_phase not in approval.allowed_command_phases:
        raise PaidRunApprovalError(f"command phase is not approved: {command_phase}")

    review = approval.review.payload
    if review.planned_command_phases != approval.allowed_command_phases:
        raise PaidRunApprovalError("paid-run approval scope does not match the reviewed scope")
    expected_context_hashes = PaidRunReviewContextHashes(
        config=approval.bindings.config_hash,
        preregistration=approval.bindings.preregistration_hash,
        gpu_lock=approval.bindings.gpu_lock_hash,
        gpu_quote_lock=approval.bindings.gpu.quote.content_hash,
        api_quote_lock=approval.bindings.api_quote.content_hash,
        bindings=stable_hash(approval.bindings.model_dump(mode="json")),
    )
    if review.context_hashes != expected_context_hashes:
        raise PaidRunApprovalError("paid-run review context does not match the frozen bindings")
    if review.caps_usd != approval.bindings.caps_usd:
        raise PaidRunApprovalError("paid-run review caps do not match the frozen bindings")
    from model_forensics.gpu_budget import approved_gpu_phase_maximum_usd

    expected_phase_maxima = tuple(
        PaidRunReviewPhaseMaximum(
            command_phase=allocation.command_phase,
            maximum_usd=approved_gpu_phase_maximum_usd(
                gpu_count=approval.bindings.gpu.count,
                quote_hourly_per_gpu_usd=approval.bindings.gpu.quote.usd_per_gpu_hour,
                running_storage_hourly_usd=(
                    approval.bindings.gpu.quote.running_storage_usd_per_hour
                ),
                approved_runtime_hours=allocation.maximum_runtime_hours,
            ),
        )
        for allocation in approval.bindings.gpu.phase_runtime_allocations
    )
    if review.phase_maxima_usd != expected_phase_maxima:
        raise PaidRunApprovalError("paid-run review phase maxima do not match the frozen bindings")
    if expected_ledger_path is not None:
        try:
            normalized_ledger_path = PaidRunReviewLedger(
                path=expected_ledger_path,
                bytes_sha256=review.ledger.bytes_sha256,
                document_hash=review.ledger.document_hash,
                byte_count=review.ledger.byte_count,
            ).path
        except ValidationError as exc:
            raise PaidRunApprovalError("expected cumulative ledger path is malformed") from exc
        if review.ledger.path != normalized_ledger_path:
            raise PaidRunApprovalError(
                "paid-run review ledger path does not match the canonical runner ledger"
            )
    if expected_source_commit is not None:
        if _SOURCE_COMMIT_RE.fullmatch(expected_source_commit) is None:
            raise PaidRunApprovalError("expected source commit is malformed")
        if review.source_commit != expected_source_commit:
            raise PaidRunApprovalError("paid-run review source commit does not match the runner")

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
