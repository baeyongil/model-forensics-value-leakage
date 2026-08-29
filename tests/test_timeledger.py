from __future__ import annotations

import pytest
import yaml

from model_forensics.timeledger import InvestigationTimeExceeded, TimeEntry, TimeLedger


def _ledger(tmp_path):
    path = tmp_path / "time.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "entries": [],
                "categories": {
                    "counted": {"analysis": 10},
                    "excluded_but_logged": ["replication"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return TimeLedger(path, counted_limit_minutes=10)


def test_time_ledger_tracks_counted_and_excluded(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.append(TimeEntry("analysis", 5, "work", "counted"))["counted"] == 5
    totals = ledger.append(TimeEntry("replication", 20, "runtime", "excluded"))
    assert totals == {"counted": 5, "excluded": 20, "wall_logged": 25}


def test_time_ledger_reports_category_overrun_but_preserves_actual_time(tmp_path) -> None:
    seeded = _ledger(tmp_path)
    ledger = TimeLedger(seeded.path, counted_limit_minutes=20)
    totals = ledger.append(TimeEntry("analysis", 11, "actual overrun", "counted"))
    assert totals["counted"] == 11
    accounting = ledger.status()["category_accounting"]
    assert accounting["allocations"] == {"analysis": 10}
    assert accounting["usage"] == {"analysis": 11}
    assert accounting["overages"] == {"analysis": 1}


def test_time_ledger_records_actual_nonoverlapping_interval(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    active = ledger.start(
        category="analysis",
        description="frozen analysis",
        status="counted",
        started_at="2026-08-29T12:00:00+00:00",
    )
    assert active["started_at"] == "2026-08-29T12:00:00+00:00"
    with pytest.raises(ValueError, match="active session"):
        ledger.start(
            category="analysis",
            description="overlap",
            status="counted",
        )

    totals = ledger.stop(ended_at="2026-08-29T12:02:01+00:00")
    assert totals["counted"] == 3
    document = yaml.safe_load(ledger.path.read_text(encoding="utf-8"))
    assert "active_session" not in document
    assert document["entries"][0]["elapsed_seconds"] == 121.0
    assert document["entries"][0]["ended_at"] == "2026-08-29T12:02:01+00:00"


def test_time_ledger_failed_stop_preserves_active_session(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.start(
        category="analysis",
        description="too long",
        status="counted",
        started_at="2026-08-29T12:00:00+00:00",
    )
    with pytest.raises(InvestigationTimeExceeded):
        ledger.stop(ended_at="2026-08-29T12:11:00+00:00")
    assert ledger.status()["active_session"] is not None
