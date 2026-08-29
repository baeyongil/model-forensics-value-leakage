"""Transparent five-hour investigation-time accounting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

from model_forensics.io import atomic_write_text

TimeStatus = Literal["counted", "excluded"]


@dataclass(frozen=True)
class TimeEntry:
    category: str
    minutes: int
    description: str
    status: TimeStatus
    started_at: str = ""
    ended_at: str = ""
    elapsed_seconds: float | None = None

    def normalized(self) -> TimeEntry:
        if self.minutes <= 0:
            raise ValueError("minutes must be positive")
        started_at = self.started_at or datetime.now(UTC).isoformat()
        ended_at = self.ended_at or started_at
        started = _aware_datetime(started_at, field="started_at")
        ended = _aware_datetime(ended_at, field="ended_at")
        if ended < started:
            raise ValueError("ended_at must not precede started_at")
        elapsed_seconds = (
            float(self.elapsed_seconds)
            if self.elapsed_seconds is not None
            else (ended - started).total_seconds()
        )
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        return TimeEntry(
            category=self.category,
            minutes=int(self.minutes),
            description=self.description,
            status=self.status,
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            elapsed_seconds=elapsed_seconds,
        )


class InvestigationTimeExceeded(RuntimeError):
    pass


def _aware_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


class TimeLedger:
    def __init__(self, path: str | Path, *, counted_limit_minutes: int = 300) -> None:
        self.path = Path(path)
        self.counted_limit_minutes = counted_limit_minutes

    def _load(self) -> dict:
        value = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
            raise ValueError(f"invalid time ledger: {self.path}")
        return value

    def _write(self, document: dict) -> None:
        atomic_write_text(
            self.path,
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        )

    @staticmethod
    def _validate_category(document: dict, *, category: str, status: TimeStatus) -> None:
        allowed_counted = set(document.get("categories", {}).get("counted", {}))
        allowed_excluded = set(document.get("categories", {}).get("excluded_but_logged", []))
        allowed = allowed_counted if status == "counted" else allowed_excluded
        if category not in allowed:
            raise ValueError(f"category {category!r} is not preregistered for {status} time")

    @staticmethod
    def totals(document: dict) -> dict[str, int]:
        counted = sum(
            int(entry["minutes"]) for entry in document["entries"] if entry["status"] == "counted"
        )
        excluded = sum(
            int(entry["minutes"]) for entry in document["entries"] if entry["status"] == "excluded"
        )
        return {"counted": counted, "excluded": excluded, "wall_logged": counted + excluded}

    @staticmethod
    def category_accounting(document: dict) -> dict[str, dict[str, int]]:
        """Report planned allocations and actual use without rewriting elapsed time."""

        allocations = {
            str(category): int(minutes)
            for category, minutes in document.get("categories", {}).get("counted", {}).items()
        }
        usage = {category: 0 for category in allocations}
        for entry in document["entries"]:
            if entry.get("status") != "counted":
                continue
            category = str(entry.get("category"))
            usage[category] = usage.get(category, 0) + int(entry["minutes"])
        overages = {
            category: used - allocations[category]
            for category, used in usage.items()
            if category in allocations and used > allocations[category]
        }
        return {
            "allocations": allocations,
            "usage": usage,
            "overages": overages,
        }

    def _append_to_document(self, document: dict, entry: TimeEntry) -> dict[str, int]:
        normalized = entry.normalized()
        self._validate_category(
            document,
            category=normalized.category,
            status=normalized.status,
        )
        candidate = {**document, "entries": [*document["entries"], asdict(normalized)]}
        totals = self.totals(candidate)
        if totals["counted"] > self.counted_limit_minutes:
            raise InvestigationTimeExceeded(
                f"counted investigation time {totals['counted']} min exceeds "
                f"{self.counted_limit_minutes} min"
            )
        self._write(candidate)
        return totals

    def append(self, entry: TimeEntry) -> dict[str, int]:
        return self._append_to_document(self._load(), entry)

    def start(
        self,
        *,
        category: str,
        description: str,
        status: TimeStatus,
        started_at: str | None = None,
    ) -> dict:
        """Start one non-overlapping interval and persist it immediately."""

        document = self._load()
        if document.get("active_session") is not None:
            raise ValueError("the time ledger already has an active session")
        self._validate_category(document, category=category, status=status)
        if not description.strip():
            raise ValueError("description must not be blank")
        started = _aware_datetime(
            started_at or datetime.now(UTC).isoformat(),
            field="started_at",
        )
        active = {
            "category": category,
            "description": description,
            "status": status,
            "started_at": started.isoformat(),
        }
        self._write({**document, "active_session": active})
        return active

    def stop(self, *, ended_at: str | None = None) -> dict[str, int]:
        """Stop the active interval, conservatively rounding time up to minutes."""

        document = self._load()
        active = document.get("active_session")
        if not isinstance(active, dict):
            raise ValueError("the time ledger has no active session")
        started = _aware_datetime(str(active.get("started_at", "")), field="started_at")
        ended = _aware_datetime(
            ended_at or datetime.now(UTC).isoformat(),
            field="ended_at",
        )
        elapsed_seconds = (ended - started).total_seconds()
        if elapsed_seconds < 0:
            raise ValueError("ended_at must not precede the active start")
        minutes = max(1, math.ceil(elapsed_seconds / 60))
        without_active = dict(document)
        without_active.pop("active_session", None)
        # Reuse the same validation and cap logic without risking a partially
        # stopped session if the interval would exceed a frozen allocation.
        active_status = active.get("status")
        if active_status not in {"counted", "excluded"}:
            raise ValueError("active session has an invalid status")
        return self._append_to_document(
            without_active,
            TimeEntry(
                category=str(active["category"]),
                minutes=minutes,
                description=str(active["description"]),
                status=active_status,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                elapsed_seconds=elapsed_seconds,
            ),
        )

    def status(self) -> dict:
        document = self._load()
        return {
            "totals": self.totals(document),
            "category_accounting": self.category_accounting(document),
            "active_session": document.get("active_session"),
            "counted_limit_minutes": self.counted_limit_minutes,
        }
