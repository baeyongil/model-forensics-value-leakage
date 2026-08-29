"""Canonical, dependency-light schemas for rollout and trajectory artifacts.

The project uses Pydantic for configuration, but raw experimental artifacts need
to remain readable in minimal analysis environments.  These frozen dataclasses
therefore expose the familiar ``model_dump`` shape while depending only on the
standard library.  Hashes are computed from canonical JSON and never include the
hash field itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from model_forensics.io import canonical_json as _canonical_json
from model_forensics.io import stable_hash as _json_stable_hash

JsonScalar = type(None) | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _jsonable(value: Any) -> JsonValue:
    """Convert supported Python values to a deterministic JSON-compatible tree."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical artifacts cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical artifacts cannot contain NaN or infinity")
        return format(value.normalize(), "f")
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize dataclasses and JSON-like values with stable key ordering."""

    return _canonical_json(_jsonable(value))


def stable_hash(value: Any) -> str:
    """Return a namespaced SHA-256 digest of canonical content."""

    return _json_stable_hash(_jsonable(value))


def hash_text(text: str) -> str:
    """Hash text exactly as supplied, including whitespace and Unicode."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return stable_hash(text)


class CanonicalRecord:
    """Serialization helpers shared by all artifact dataclasses."""

    def to_dict(self, *, include_hash: bool = False) -> dict[str, JsonValue]:
        payload = _jsonable(self)
        if not isinstance(payload, dict):  # pragma: no cover - defensive invariant
            raise TypeError("canonical record must serialize to an object")
        if include_hash:
            payload = dict(payload)
            payload["record_hash"] = self.record_hash
        return payload

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, JsonValue]:
        """Small Pydantic-compatible adapter used by artifact writers."""

        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        payload = self.to_dict()
        if include is not None:
            allowed = set(include)
            payload = {key: value for key, value in payload.items() if key in allowed}
        if exclude is not None:
            denied = set(exclude)
            payload = {key: value for key, value in payload.items() if key not in denied}
        return payload

    @property
    def record_hash(self) -> str:
        """Stable content hash, excluding any externally stored hash field."""

        return stable_hash(self.to_dict())

    @property
    def stable_hash(self) -> str:
        """Alias retained for callers that use the schema terminology directly."""

        return self.record_hash


@dataclass(frozen=True)
class Provenance(CanonicalRecord):
    """Identity of one model invocation and the artifacts that produced it."""

    execution_id: str
    model_id: str
    provider: str
    seed: int
    prompt_hash: str
    model_revision: str | None = None
    model_hash: str | None = None
    config_hash: str | None = None
    code_revision: str | None = None
    request_id: str | None = None
    created_at_utc: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("execution_id", "model_id", "provider", "prompt_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.model_hash is None:
            object.__setattr__(
                self,
                "model_hash",
                stable_hash(
                    {
                        "model_id": self.model_id,
                        "provider": self.provider,
                        "model_revision": self.model_revision,
                    }
                ),
            )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TokenUsage(CanonicalRecord):
    """Provider-neutral token and cost accounting."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd must be nonnegative")


@dataclass(frozen=True)
class NumericEstimate(CanonicalRecord):
    """One normalized numeric mention and its character-level provenance."""

    value: float
    raw: str
    start: int
    end: int
    source: str = "trace"
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValueError("estimate value must be finite")
        if self.start < 0 or self.end < self.start:
            raise ValueError("estimate span must satisfy 0 <= start <= end")
        if self.end - self.start != len(self.raw):
            raise ValueError("estimate span length must equal raw text length")
        if not self.source:
            raise ValueError("estimate source must not be empty")


@dataclass(frozen=True)
class TrajectoryFeatures(CanonicalRecord):
    """Precomputed search/stopping summaries for an estimate sequence.

    ``first_good_side_crossing_index`` is zero-based in the consecutive-deduplicated
    revision sequence.  ``stopped_after_first_good_side_crossing`` is ``None`` when
    a prompt has no good side or the trajectory never reaches it.
    """

    first_estimate: float | None
    final_estimate: float | None
    revision_count: int
    first_good_side_crossing_index: int | None
    stopped_after_first_good_side_crossing: bool | None
    revisions_after_good: int | None
    estimate_count: int

    def __post_init__(self) -> None:
        if self.revision_count < 0 or self.estimate_count < 0:
            raise ValueError("trajectory counts must be nonnegative")
        if (
            self.first_good_side_crossing_index is not None
            and self.first_good_side_crossing_index < 0
        ):
            raise ValueError("first_good_side_crossing_index must be nonnegative")
        if self.revisions_after_good is not None and self.revisions_after_good < 0:
            raise ValueError("revisions_after_good must be nonnegative")

    @property
    def first_good_side_crossing(self) -> int | None:
        """Compact alias for the preregistered crossing-index field."""

        return self.first_good_side_crossing_index

    @property
    def stop_after_good(self) -> bool | None:
        """Compact alias for the preregistered stopping field."""

        return self.stopped_after_first_good_side_crossing


@dataclass(frozen=True)
class TrajectoryRecord(CanonicalRecord):
    """Ordered numeric mentions plus their deterministic derived features."""

    estimates: tuple[NumericEstimate, ...]
    features: TrajectoryFeatures
    parser_version: str = "numeric-v1"
    requires_manual_review: bool = False
    review_notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimates", tuple(self.estimates))


@dataclass(frozen=True)
class RolloutRecord(CanonicalRecord):
    """Canonical record for one complete model rollout."""

    run_id: str
    task: str
    condition: str
    threshold: float | None
    prompt: str
    trace: str
    answer: str
    provenance: Provenance
    trajectory: TrajectoryRecord
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    raw_response: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.task or not self.condition:
            raise ValueError("task and condition must not be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported rollout schema_version")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.raw_response is not None:
            object.__setattr__(self, "raw_response", dict(self.raw_response))


# Short aliases make the JSON schema terminology convenient at call sites.
Estimate = NumericEstimate
Trajectory = TrajectoryRecord
Rollout = RolloutRecord


__all__ = [
    "CanonicalRecord",
    "Estimate",
    "JsonValue",
    "NumericEstimate",
    "Provenance",
    "Rollout",
    "RolloutRecord",
    "TokenUsage",
    "Trajectory",
    "TrajectoryFeatures",
    "TrajectoryRecord",
    "canonical_json",
    "hash_text",
    "stable_hash",
]
