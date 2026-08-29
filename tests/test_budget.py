from __future__ import annotations

import pytest
import yaml

from model_forensics.budget import (
    BudgetExceeded,
    BudgetLimits,
    CostEntry,
    CostLedger,
    ReservationConflict,
    estimate_gpu_cost,
)


def test_gpu_estimate_matches_count_rate_and_hours() -> None:
    assert estimate_gpu_cost(gpu_count=8, hourly_per_gpu=3.29, hours=7.5) == 197.4


def test_ledger_enforces_category_and_total_hard_stops(tmp_path) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml", BudgetLimits(gpu=5, api=2, total=7))
    totals = ledger.append(CostEntry(kind="gpu", amount_usd=4.5, description="pilot"))
    assert totals["gpu"] == 4.5
    with pytest.raises(BudgetExceeded, match="GPU cost"):
        ledger.append(CostEntry(kind="gpu", amount_usd=1, description="over cap"))


def test_estimate_check_does_not_write(tmp_path) -> None:
    path = tmp_path / "cost.yaml"
    ledger = CostLedger(path)
    totals = ledger.assert_estimate_fits(
        CostEntry(kind="gpu", amount_usd=197.4, description="8xH100", status="estimated")
    )
    assert totals["total"] == 197.4
    assert not path.exists()


def test_reservation_is_idempotent_and_settles_exactly_once(tmp_path) -> None:
    path = tmp_path / "cost.yaml"
    ledger = CostLedger(path, BudgetLimits(gpu=5, api=2, total=7))
    reservation = CostEntry(
        kind="api",
        amount_usd=0.25,
        description="frozen request",
        status="estimated",
    )
    first = ledger.reserve("sha256:" + "a" * 64, reservation)
    second = ledger.reserve("sha256:" + "a" * 64, reservation)
    assert first == second
    assert len(ledger.document()["entries"]) == 1
    assert ledger.totals(ledger.document(), include_estimates=True)["api"] == 0.25

    incurred = CostEntry(kind="api", amount_usd=0.10, description="frozen request")
    settled = ledger.settle_reservation("sha256:" + "a" * 64, incurred)
    repeated = ledger.settle_reservation("sha256:" + "a" * 64, incurred)
    assert settled == repeated
    document = ledger.document()
    assert len(document["entries"]) == 1
    assert document["entries"][0]["status"] == "incurred"
    assert ledger.totals(document)["api"] == 0.10


def test_reservation_id_reuse_with_different_content_fails_closed(tmp_path) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml")
    entry_id = "sha256:" + "b" * 64
    ledger.reserve(
        entry_id,
        CostEntry(kind="api", amount_usd=0.2, description="first", status="estimated"),
    )
    with pytest.raises(ValueError, match="different content"):
        ledger.reserve(
            entry_id,
            CostEntry(kind="api", amount_usd=0.3, description="changed", status="estimated"),
        )


def test_batch_reservation_is_atomic_when_complete_inventory_exceeds_cap(tmp_path) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml", BudgetLimits(gpu=5, api=0.30, total=5.30))
    reservations = (
        (
            "sha256:" + "1" * 64,
            CostEntry(kind="api", amount_usd=0.20, description="first", status="estimated"),
        ),
        (
            "sha256:" + "2" * 64,
            CostEntry(kind="api", amount_usd=0.20, description="second", status="estimated"),
        ),
    )

    with pytest.raises(BudgetExceeded):
        ledger.reserve_batch(reservations)

    assert not ledger.path.exists()


def test_batch_reservation_returns_exact_created_and_covered_identities(tmp_path) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml")
    existing_id = "sha256:" + "3" * 64
    new_id = "sha256:" + "4" * 64
    existing = CostEntry(
        kind="api", amount_usd=0.10, description="existing", status="estimated"
    )
    new = CostEntry(kind="api", amount_usd=0.20, description="new", status="estimated")
    ledger.reserve(existing_id, existing)

    snapshot = ledger.reserve_batch(((existing_id, existing), (new_id, new)))

    assert snapshot.covered_entry_ids == (existing_id,)
    assert snapshot.created_entry_ids == (new_id,)
    assert snapshot.committed_before["api"] == 0.10
    assert snapshot.committed_after["api"] == 0.30


def test_loaded_ledger_cannot_silently_change_hard_stops(tmp_path) -> None:
    path = tmp_path / "cost.yaml"
    CostLedger(path, BudgetLimits(gpu=5, api=2, total=7)).append(
        CostEntry(kind="gpu", amount_usd=1, description="pilot")
    )
    with pytest.raises(ValueError, match="hard stops"):
        CostLedger(path, BudgetLimits(gpu=500, api=500, total=1000)).document()


def test_unique_reservation_returns_atomic_before_after_totals_and_cannot_replay(
    tmp_path,
) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml", BudgetLimits(gpu=10, api=2, total=12))
    ledger.append(CostEntry(kind="gpu", amount_usd=2, description="prior phase"))
    entry = CostEntry(
        kind="gpu",
        amount_usd=3,
        description="one paid launch",
        status="estimated",
    )
    snapshot = ledger.reserve_once(
        "sha256:" + "c" * 64,
        entry,
        maximum_totals={"gpu": 9},
        require_no_outstanding_kind=True,
    )

    assert snapshot.incurred_before["gpu"] == 2
    assert snapshot.committed_before["gpu"] == 2
    assert snapshot.committed_after["gpu"] == 5
    with pytest.raises(ReservationConflict, match="already been used"):
        ledger.reserve_once(
            "sha256:" + "c" * 64,
            entry,
            maximum_totals={"gpu": 9},
            require_no_outstanding_kind=True,
        )


def test_unique_reservation_rejects_outstanding_kind_and_stricter_ceiling(tmp_path) -> None:
    ledger = CostLedger(tmp_path / "cost.yaml", BudgetLimits(gpu=10, api=2, total=12))
    ledger.reserve_once(
        "sha256:" + "d" * 64,
        CostEntry(kind="gpu", amount_usd=3, description="first", status="estimated"),
    )
    with pytest.raises(ReservationConflict, match="already outstanding"):
        ledger.reserve_once(
            "sha256:" + "e" * 64,
            CostEntry(kind="gpu", amount_usd=1, description="second", status="estimated"),
            require_no_outstanding_kind=True,
        )

    other = CostLedger(tmp_path / "other.yaml", BudgetLimits(gpu=10, api=2, total=12))
    with pytest.raises(BudgetExceeded, match="transaction ceiling"):
        other.reserve_once(
            "sha256:" + "f" * 64,
            CostEntry(kind="gpu", amount_usd=3, description="too large", status="estimated"),
            maximum_totals={"gpu": 2.5},
        )
    assert not other.path.exists()


def test_canonical_ledger_rejects_nonfinite_or_schema_ambiguous_entries(tmp_path) -> None:
    path = tmp_path / "cost.yaml"
    base = {
        "schema_version": 1,
        "currency": "USD",
        "hard_stops": {"gpu": 220.0, "api": 100.0, "total": 325.0},
        "entries": [
            {
                "kind": "gpu",
                "amount_usd": float("nan"),
                "description": "corrupt",
                "status": "incurred",
                "occurred_at": "2026-08-29T12:00:00+00:00",
            }
        ],
    }
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid amount"):
        CostLedger(path).document()

    base["entries"][0]["amount_usd"] = 1
    base["entries"][0]["unexpected"] = "field"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema"):
        CostLedger(path).document()
