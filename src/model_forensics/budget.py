"""External-cost estimation and hard-stop enforcement."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from model_forensics.io import atomic_write_text

CostKind = Literal["gpu", "api", "storage", "other"]


@dataclass(frozen=True)
class CostEntry:
    kind: CostKind
    amount_usd: float
    description: str
    status: Literal["estimated", "incurred"] = "incurred"
    occurred_at: str = ""

    def normalized(self) -> CostEntry:
        try:
            amount = float(self.amount_usd)
        except (TypeError, ValueError) as exc:
            raise ValueError("cost must be finite and non-negative") from exc
        if isinstance(self.amount_usd, bool) or not math.isfinite(amount) or amount < 0:
            raise ValueError("cost must be finite and non-negative")
        if self.kind not in {"gpu", "api", "storage", "other"}:
            raise ValueError("cost kind is unsupported")
        if self.status not in {"estimated", "incurred"}:
            raise ValueError("cost status is unsupported")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("cost description must be non-empty")
        timestamp = self.occurred_at or datetime.now(UTC).isoformat()
        if not isinstance(timestamp, str):
            raise ValueError("cost timestamp must be an ISO-8601 string")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("cost timestamp must be ISO-8601") from exc
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            raise ValueError("cost timestamp must be timezone-aware")
        return CostEntry(
            kind=self.kind,
            amount_usd=round(amount, 6),
            description=self.description,
            status=self.status,
            occurred_at=timestamp,
        )


@dataclass(frozen=True)
class BudgetLimits:
    gpu: float = 220.0
    api: float = 100.0
    total: float = 325.0

    def __post_init__(self) -> None:
        values = (self.gpu, self.api, self.total)
        if any(
            isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0
            for value in values
        ):
            raise ValueError("budget limits must be finite and non-negative")
        if self.gpu <= 0 or self.total <= 0:
            raise ValueError("GPU and total budget limits must be positive")


class BudgetExceeded(RuntimeError):
    pass


class ReservationConflict(RuntimeError):
    """A one-use reservation identity or resource is already outstanding."""


@dataclass(frozen=True)
class ReservationSnapshot:
    """Atomic before/after totals produced by a unique reservation."""

    incurred_before: dict[str, float]
    committed_before: dict[str, float]
    committed_after: dict[str, float]


def estimate_gpu_cost(*, gpu_count: int, hourly_per_gpu: float, hours: float) -> float:
    if gpu_count <= 0 or hourly_per_gpu < 0 or hours < 0:
        raise ValueError("GPU count must be positive and rates/hours non-negative")
    return round(gpu_count * hourly_per_gpu * hours, 2)


class CostLedger:
    def __init__(self, path: str | Path, limits: BudgetLimits | None = None) -> None:
        self.path = Path(path)
        self.limits = limits or BudgetLimits()

    @contextmanager
    def _locked(self):  # type: ignore[no-untyped-def]
        """Serialize budget checks and writes across local worker processes."""

        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with os.fdopen(descriptor, "a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "currency": "USD",
                "hard_stops": asdict(self.limits),
                "entries": [],
            }
        value = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
            raise ValueError(f"invalid cost ledger: {self.path}")
        if value.get("schema_version") != 1 or value.get("currency") != "USD":
            raise ValueError(f"invalid cost ledger protocol: {self.path}")
        observed_limits = value.get("hard_stops")
        expected_limits = asdict(self.limits)
        if observed_limits != expected_limits:
            raise ValueError(f"cost ledger hard stops disagree with configured limits: {self.path}")
        entry_ids: list[str] = []
        required_entry_keys = {
            "kind",
            "amount_usd",
            "description",
            "status",
            "occurred_at",
        }
        for index, entry in enumerate(value["entries"]):
            if not isinstance(entry, dict):
                raise ValueError(f"cost ledger entry {index} must be a mapping: {self.path}")
            if not required_entry_keys.issubset(entry) or set(entry) - (
                required_entry_keys | {"entry_id"}
            ):
                raise ValueError(f"cost ledger entry {index} has an invalid schema: {self.path}")
            if entry["kind"] not in {"gpu", "api", "storage", "other"}:
                raise ValueError(f"cost ledger entry {index} has an invalid kind: {self.path}")
            amount = entry["amount_usd"]
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or float(amount) < 0
            ):
                raise ValueError(f"cost ledger entry {index} has an invalid amount: {self.path}")
            if not isinstance(entry["description"], str) or not entry["description"].strip():
                raise ValueError(
                    f"cost ledger entry {index} has an invalid description: {self.path}"
                )
            if entry["status"] not in {"estimated", "incurred"}:
                raise ValueError(f"cost ledger entry {index} has an invalid status: {self.path}")
            occurred_at = entry["occurred_at"]
            if not isinstance(occurred_at, str):
                raise ValueError(f"cost ledger entry {index} has an invalid timestamp: {self.path}")
            try:
                parsed_at = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"cost ledger entry {index} has an invalid timestamp: {self.path}"
                ) from exc
            if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
                raise ValueError(
                    f"cost ledger entry {index} timestamp must be timezone-aware: {self.path}"
                )
            entry_id = entry.get("entry_id")
            if entry_id is not None:
                if not isinstance(entry_id, str) or not entry_id:
                    raise ValueError(
                        f"cost ledger entry {index} has an invalid entry ID: {self.path}"
                    )
                entry_ids.append(entry_id)
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError(f"cost ledger contains duplicate entry IDs: {self.path}")
        self._assert_limits(self.totals(value, include_estimates=True))
        return value

    def document(self) -> dict[str, Any]:
        """Return a validated snapshot of the ledger."""

        with self._locked():
            return self._load_unlocked()

    @staticmethod
    def totals(document: dict, *, include_estimates: bool = False) -> dict[str, float]:
        totals = {"gpu": 0.0, "api": 0.0, "storage": 0.0, "other": 0.0}
        for entry in document["entries"]:
            if entry.get("status", "incurred") == "estimated" and not include_estimates:
                continue
            totals[entry["kind"]] += float(entry["amount_usd"])
        totals["total"] = sum(totals.values())
        return {key: round(value, 6) for key, value in totals.items()}

    def _assert_limits(self, totals: dict[str, float]) -> None:
        if totals["gpu"] > self.limits.gpu:
            raise BudgetExceeded(f"GPU cost ${totals['gpu']:.2f} exceeds ${self.limits.gpu:.2f}")
        if totals["api"] > self.limits.api:
            raise BudgetExceeded(f"API cost ${totals['api']:.2f} exceeds ${self.limits.api:.2f}")
        if totals["total"] > self.limits.total:
            raise BudgetExceeded(
                f"total cost ${totals['total']:.2f} exceeds ${self.limits.total:.2f}"
            )

    def append(self, entry: CostEntry) -> dict[str, float]:
        with self._locked():
            document = self._load_unlocked()
            normalized = entry.normalized()
            candidate = {**document, "entries": [*document["entries"], asdict(normalized)]}
            self._assert_limits(self.totals(candidate, include_estimates=True))
            atomic_write_text(
                self.path,
                yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True),
            )
            return self.totals(candidate)

    @staticmethod
    def _same_entry_content(left: dict[str, Any], right: dict[str, Any]) -> bool:
        fields = ("kind", "amount_usd", "description", "status")
        return all(left.get(field) == right.get(field) for field in fields)

    def reserve(self, entry_id: str, entry: CostEntry) -> dict[str, float]:
        """Atomically reserve a preflight estimate under an idempotency key."""

        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("reservation entry_id must be a non-empty string")
        if entry.status != "estimated":
            raise ValueError("a reservation must have estimated status")
        with self._locked():
            document = self._load_unlocked()
            normalized = asdict(entry.normalized())
            normalized["entry_id"] = entry_id
            existing = next(
                (item for item in document["entries"] if item.get("entry_id") == entry_id),
                None,
            )
            if existing is not None:
                if existing.get("status") == "incurred":
                    if existing.get("kind") != normalized["kind"]:
                        raise ValueError(
                            "reservation entry ID already settled under a different kind"
                        )
                    return self.totals(document, include_estimates=True)
                if not self._same_entry_content(existing, normalized):
                    raise ValueError("reservation entry ID already exists with different content")
                return self.totals(document, include_estimates=True)
            candidate = {**document, "entries": [*document["entries"], normalized]}
            totals = self.totals(candidate, include_estimates=True)
            self._assert_limits(totals)
            atomic_write_text(
                self.path,
                yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True),
            )
            return totals

    def reserve_once(
        self,
        entry_id: str,
        entry: CostEntry,
        *,
        maximum_totals: Mapping[str, float] | None = None,
        require_no_outstanding_kind: bool = False,
    ) -> ReservationSnapshot:
        """Atomically create a non-replayable reservation.

        Unlike :meth:`reserve`, this method deliberately rejects an existing
        ``entry_id`` even when its content is identical.  It is intended for
        resources, such as a GPU session, where treating a repeated preflight
        as an idempotent success could authorize a second paid launch.

        ``maximum_totals`` can impose stricter transaction-local ceilings than
        the ledger hard stops (for example, a safety-adjusted GPU ceiling).
        The optional outstanding-kind guard prevents overlapping reservations
        for the same paid resource class.
        """

        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("reservation entry_id must be a non-empty string")
        if entry.status != "estimated":
            raise ValueError("a reservation must have estimated status")
        ceilings: dict[str, float] = {}
        for key, raw_value in (maximum_totals or {}).items():
            if key not in {"gpu", "api", "storage", "other", "total"}:
                raise ValueError(f"unknown reservation total ceiling: {key}")
            if (
                isinstance(raw_value, bool)
                or not math.isfinite(float(raw_value))
                or float(raw_value) < 0
            ):
                raise ValueError("reservation total ceilings must be finite and non-negative")
            ceilings[key] = float(raw_value)

        normalized = asdict(entry.normalized())
        normalized["entry_id"] = entry_id
        with self._locked():
            document = self._load_unlocked()
            if any(item.get("entry_id") == entry_id for item in document["entries"]):
                raise ReservationConflict("one-use reservation entry ID has already been used")
            if require_no_outstanding_kind and any(
                item.get("kind") == normalized["kind"]
                and item.get("status", "incurred") == "estimated"
                for item in document["entries"]
            ):
                raise ReservationConflict(
                    f"an unsettled {normalized['kind']} reservation is already outstanding"
                )

            incurred_before = self.totals(document)
            committed_before = self.totals(document, include_estimates=True)
            candidate = {**document, "entries": [*document["entries"], normalized]}
            committed_after = self.totals(candidate, include_estimates=True)
            self._assert_limits(committed_after)
            for key, ceiling in ceilings.items():
                if committed_after[key] > ceiling + 1e-9:
                    raise BudgetExceeded(
                        f"reserved {key} total ${committed_after[key]:.6f} exceeds "
                        f"the transaction ceiling ${ceiling:.6f}"
                    )
            atomic_write_text(
                self.path,
                yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True),
            )
            return ReservationSnapshot(
                incurred_before=incurred_before,
                committed_before=committed_before,
                committed_after=committed_after,
            )

    def settle_reservation(self, entry_id: str, entry: CostEntry) -> dict[str, float]:
        """Replace one estimate with its exact incurred cost, idempotently."""

        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("reservation entry_id must be a non-empty string")
        if entry.status != "incurred":
            raise ValueError("a settled reservation must have incurred status")
        with self._locked():
            document = self._load_unlocked()
            matching = [
                (index, item)
                for index, item in enumerate(document["entries"])
                if item.get("entry_id") == entry_id
            ]
            if not matching:
                raise ValueError("cannot settle a missing reservation")
            index, existing = matching[0]
            normalized = asdict(entry.normalized())
            normalized["entry_id"] = entry_id
            if existing.get("status") == "incurred":
                if not self._same_entry_content(existing, normalized):
                    raise ValueError("settled entry ID already exists with different content")
                return self.totals(document)
            if existing.get("status") != "estimated":
                raise ValueError("reservation has an invalid status")
            if existing.get("kind") != normalized["kind"]:
                raise ValueError("settlement kind disagrees with reservation")
            entries = list(document["entries"])
            entries[index] = normalized
            candidate = {**document, "entries": entries}
            self._assert_limits(self.totals(candidate, include_estimates=True))
            atomic_write_text(
                self.path,
                yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True),
            )
            return self.totals(candidate)

    def assert_estimate_fits(self, entry: CostEntry) -> dict[str, float]:
        with self._locked():
            document = self._load_unlocked()
            candidate = {
                **document,
                "entries": [*document["entries"], asdict(entry.normalized())],
            }
            totals = self.totals(candidate, include_estimates=True)
            self._assert_limits(totals)
            return totals
