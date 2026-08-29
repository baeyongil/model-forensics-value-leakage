from __future__ import annotations

from pathlib import Path

import pytest

from model_forensics.io import read_json, stable_hash, write_json
from model_forensics.record_checkpoint import (
    RecordCheckpointError,
    RecordCheckpointStore,
)


def _row(identifier: str, value: int) -> dict:
    row = {"unit_id": identifier, "value": value}
    row["record_hash"] = stable_hash(row)
    return row


def test_record_store_is_atomic_idempotent_and_finalizes_in_frozen_order(
    tmp_path: Path,
) -> None:
    store = RecordCheckpointStore(
        tmp_path,
        id_field="unit_id",
        plan_payload={"phase": "gpu", "input_hash": stable_hash("input")},
    )
    rows = [_row("c", 3), _row("a", 1), _row("b", 2)]
    for row in rows:
        assert store.commit(row) == store.commit(row)
    assert {row["unit_id"] for row in store.load_records()} == {"a", "b", "c"}

    final = store.finalize(expected_ids=("a", "b", "c"))
    assert [row["unit_id"] for row in final.rows] == ["a", "b", "c"]
    assert final.manifest["complete"] is True
    assert store.load_final(expected_ids=("a", "b", "c")) == final


def test_record_store_rejects_tampering_drift_and_extra_inventory(tmp_path: Path) -> None:
    store = RecordCheckpointStore(
        tmp_path,
        id_field="unit_id",
        plan_payload={"phase": "gpu", "input_hash": stable_hash("input")},
    )
    store.commit(_row("a", 1))
    record_path = next((tmp_path / "records").glob("*.json"))
    tampered = read_json(record_path)
    tampered["value"] = 999
    write_json(record_path, tampered)
    with pytest.raises(RecordCheckpointError, match="record hash"):
        store.load_records()

    with pytest.raises(RecordCheckpointError, match="plan mismatch"):
        RecordCheckpointStore(
            tmp_path,
            id_field="unit_id",
            plan_payload={"phase": "gpu", "input_hash": stable_hash("changed")},
        )


def test_record_store_requires_exact_expected_inventory(tmp_path: Path) -> None:
    store = RecordCheckpointStore(
        tmp_path,
        id_field="unit_id",
        plan_payload={"phase": "gpu", "input_hash": stable_hash("input")},
    )
    store.commit(_row("a", 1))
    with pytest.raises(RecordCheckpointError, match="inventory"):
        store.finalize(expected_ids=("a", "b"))


def test_existing_final_manifest_is_authenticated(tmp_path: Path) -> None:
    store = RecordCheckpointStore(
        tmp_path,
        id_field="unit_id",
        plan_payload={"phase": "gpu", "input_hash": stable_hash("input")},
    )
    store.commit(_row("a", 1))
    store.finalize(expected_ids=("a",))
    path = tmp_path / "checkpoint_manifest.json"
    manifest = read_json(path)
    manifest["row_count"] = 2
    write_json(path, manifest)
    with pytest.raises(RecordCheckpointError, match="manifest hash"):
        store.load_final(expected_ids=("a",))
