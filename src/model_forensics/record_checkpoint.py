"""Generic per-record atomic checkpoints for large resumable phases."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.io import (
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)

RECORD_CHECKPOINT_PROTOCOL = "content-addressed-record-checkpoint-v1"


class RecordCheckpointError(RuntimeError):
    """A frozen checkpoint plan or record inventory is inconsistent."""


@dataclass(frozen=True)
class FinalizedCheckpoint:
    rows: tuple[dict[str, Any], ...]
    manifest: Mapping[str, Any]


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


@contextmanager
def _lock(path: Path):  # type: ignore[no-untyped-def]
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
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


class RecordCheckpointStore:
    """Persist one authenticated JSON object per logical unit."""

    def __init__(
        self,
        directory: str | Path,
        *,
        id_field: str,
        plan_payload: Mapping[str, Any],
    ) -> None:
        if not id_field or id_field == "record_hash":
            raise ValueError("id_field must be a non-hash field name")
        if "plan_hash" in plan_payload:
            raise ValueError("plan_payload must not contain plan_hash")
        self.directory = Path(directory)
        self.records_dir = self.directory / "records"
        self.id_field = id_field
        self._lock_path = self.directory / ".checkpoint.lock"
        plan: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": RECORD_CHECKPOINT_PROTOCOL,
            "id_field": id_field,
            "payload": dict(plan_payload),
        }
        plan["plan_hash"] = stable_hash(plan)
        self.plan = self._freeze_plan(plan)

    def _freeze_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        path = self.directory / "checkpoint_plan.json"
        with _lock(self._lock_path):
            if path.exists():
                observed = read_json(path)
                if not isinstance(observed, dict) or stable_hash(observed) != stable_hash(plan):
                    raise RecordCheckpointError("frozen checkpoint plan mismatch")
                return observed
            write_json(path, plan)
        return plan

    def _record_path(self, identifier: str) -> Path:
        digest = stable_hash({"id_field": self.id_field, "identifier": identifier}).split(
            ":", 1
        )[1]
        return self.records_dir / f"{digest}.json"

    def _validate_row(self, source: Any, *, path: Path | None = None) -> dict[str, Any]:
        if not isinstance(source, Mapping):
            raise RecordCheckpointError("checkpoint row must be an object")
        row = dict(source)
        identifier = row.get(self.id_field)
        if not isinstance(identifier, str) or not identifier:
            raise RecordCheckpointError(f"checkpoint row requires non-empty {self.id_field}")
        if row.get("record_hash") != stable_hash(_without_hash(row, "record_hash")):
            raise RecordCheckpointError("checkpoint record hash mismatch")
        if path is not None and path != self._record_path(identifier):
            raise RecordCheckpointError("checkpoint record filename disagrees with its ID")
        return row

    def commit(self, source: Mapping[str, Any]) -> dict[str, Any]:
        row = self._validate_row(source)
        identifier = str(row[self.id_field])
        path = self._record_path(identifier)
        with _lock(self._lock_path):
            if path.exists():
                observed = self._validate_row(read_json(path), path=path)
                if observed != row:
                    raise RecordCheckpointError(
                        "checkpoint ID already exists with different content"
                    )
                return observed
            write_json(path, row)
        return row

    def load_records(self) -> tuple[dict[str, Any], ...]:
        with _lock(self._lock_path):
            if not self.records_dir.exists():
                return ()
            rows = [
                self._validate_row(read_json(path), path=path)
                for path in sorted(self.records_dir.glob("*.json"))
            ]
        identifiers = [row[self.id_field] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise RecordCheckpointError("checkpoint contains duplicate record IDs")
        return tuple(rows)

    def finalize(self, *, expected_ids: Sequence[str]) -> FinalizedCheckpoint:
        expected = tuple(expected_ids)
        if not expected or len(expected) != len(set(expected)) or any(not item for item in expected):
            raise ValueError("expected_ids must be a non-empty unique sequence")
        by_id = {str(row[self.id_field]): row for row in self.load_records()}
        if set(by_id) != set(expected):
            missing = sorted(set(expected).difference(by_id))
            extra = sorted(set(by_id).difference(expected))
            raise RecordCheckpointError(
                f"checkpoint inventory mismatch; missing={missing}, extra={extra}"
            )
        rows = tuple(by_id[identifier] for identifier in expected)
        merged_path = self.directory / "checkpoint_rows.jsonl"
        manifest_path = self.directory / "checkpoint_manifest.json"
        with _lock(self._lock_path):
            if merged_path.exists():
                if read_jsonl(merged_path) != list(rows):
                    raise RecordCheckpointError("existing merged checkpoint rows mismatch")
            else:
                write_jsonl(merged_path, rows)
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "protocol_version": RECORD_CHECKPOINT_PROTOCOL,
                "complete": True,
                "plan_hash": self.plan["plan_hash"],
                "id_field": self.id_field,
                "row_count": len(rows),
                "expected_ids_hash": stable_hash(expected),
                "record_hashes_hash": stable_hash([row["record_hash"] for row in rows]),
                "rows_file": merged_path.name,
                "rows_sha256": sha256_file(merged_path),
            }
            manifest["manifest_hash"] = stable_hash(manifest)
            if manifest_path.exists():
                observed = read_json(manifest_path)
                if observed != manifest:
                    raise RecordCheckpointError("existing checkpoint manifest mismatch")
            else:
                write_json(manifest_path, manifest)
        return FinalizedCheckpoint(rows, manifest)

    def load_final(self, *, expected_ids: Sequence[str]) -> FinalizedCheckpoint:
        manifest_path = self.directory / "checkpoint_manifest.json"
        if not manifest_path.is_file():
            raise RecordCheckpointError("completed checkpoint manifest is absent")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, Mapping) or manifest.get("manifest_hash") != stable_hash(
            _without_hash(manifest, "manifest_hash")
        ):
            raise RecordCheckpointError("checkpoint manifest hash mismatch")
        expected = tuple(expected_ids)
        if (
            manifest.get("plan_hash") != self.plan.get("plan_hash")
            or manifest.get("id_field") != self.id_field
            or manifest.get("expected_ids_hash") != stable_hash(expected)
        ):
            raise RecordCheckpointError("checkpoint manifest disagrees with frozen plan/inventory")
        merged_path = self.directory / str(manifest.get("rows_file"))
        if not merged_path.is_file() or sha256_file(merged_path) != manifest.get("rows_sha256"):
            raise RecordCheckpointError("checkpoint merged rows hash mismatch")
        rows = tuple(self._validate_row(row) for row in read_jsonl(merged_path))
        if [row[self.id_field] for row in rows] != list(expected):
            raise RecordCheckpointError("checkpoint merged row order mismatch")
        if len(rows) != manifest.get("row_count") or stable_hash(
            [row["record_hash"] for row in rows]
        ) != manifest.get("record_hashes_hash"):
            raise RecordCheckpointError("checkpoint merged row inventory mismatch")
        current = {row[self.id_field]: row for row in self.load_records()}
        if list(current) and any(current.get(identifier) != row for identifier, row in zip(expected, rows, strict=True)):
            raise RecordCheckpointError("checkpoint records disagree with merged artifact")
        return FinalizedCheckpoint(rows, dict(manifest))


__all__ = [
    "RECORD_CHECKPOINT_PROTOCOL",
    "FinalizedCheckpoint",
    "RecordCheckpointError",
    "RecordCheckpointStore",
]
