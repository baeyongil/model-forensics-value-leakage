from __future__ import annotations

import json

import pytest

from model_forensics.io import (
    assert_unique,
    canonical_json,
    read_jsonl,
    stable_hash,
    write_jsonl,
)


def test_canonical_json_and_hash_are_order_independent() -> None:
    left = {"b": 2, "a": [3, 1]}
    right = {"a": [3, 1], "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert stable_hash(left) == stable_hash(right)


def test_jsonl_round_trip_is_canonical(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"z": 1, "a": "alpha"}, {"z": 2, "a": "beta"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first == rows[0]


def test_assert_unique_reports_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate run_id"):
        assert_unique([{"run_id": "x"}, {"run_id": "x"}], "run_id")
