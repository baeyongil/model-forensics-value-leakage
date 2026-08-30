from __future__ import annotations

import json
from pathlib import Path

import pytest

import model_forensics.runpod_lifecycle_state as state_module
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_contract import LIFECYCLE_PROTOCOL
from model_forensics.runpod_lifecycle_state import (
    RunpodLifecycleStateError,
    load_lifecycle_state,
)


def _state(path: Path) -> Path:
    spec = {"image": "runpod/image@sha256:" + "a" * 64, "gpu": {"count": 8}}
    authorization = {
        "phase": "behavior_treatment_gpu",
        "reservation_id": stable_hash({"reservation": "fixture"}),
        "reservation_record_hash": stable_hash({"receipt": "fixture"}),
        "session_hash": stable_hash({"session": "fixture"}),
        "approval_hash": stable_hash({"approval": "fixture"}),
        "bindings_hash": stable_hash({"bindings": "fixture"}),
        "gpu_lock_hash": stable_hash({"gpu_lock": "fixture"}),
        "quote_hash": stable_hash({"quote": "fixture"}),
        "immutable_spec_hash": stable_hash(spec),
        "launch_spec_hash": stable_hash({"launch": "fixture"}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 1.0,
        "approved_phase_maximum_usd": 10.0,
        "live_hourly_total_usd": 10.0,
    }
    payload = {
        "schema_version": 1,
        "protocol_version": LIFECYCLE_PROTOCOL,
        "operation": "rearmed",
        "updated_at": "2026-08-30T00:00:00Z",
        "immutable_spec": spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": "pod-fixture", "status": "RUNNING"},
    }
    payload["record_hash"] = stable_hash(payload)
    write_json(path, payload)
    return path


def test_lifecycle_reader_rejects_symlink(tmp_path: Path) -> None:
    private = tmp_path / ".runpod"
    real = _state(tmp_path / "real.json")
    private.mkdir()
    lifecycle = private / "pod_lifecycle.json"
    lifecycle.symlink_to(real)

    with pytest.raises(RunpodLifecycleStateError, match="non-symlink"):
        load_lifecycle_state(lifecycle)


def test_lifecycle_reader_detects_path_swap_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _state(tmp_path / ".runpod" / "pod_lifecycle.json")
    original_read = state_module.os.read
    swapped = False

    def swap_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            raw = lifecycle.read_bytes()
            lifecycle.replace(lifecycle.with_suffix(".original.json"))
            lifecycle.write_bytes(raw)
            swapped = True
        return original_read(descriptor, size)

    monkeypatch.setattr(state_module.os, "read", swap_then_read)
    with pytest.raises(RunpodLifecycleStateError, match="changed during"):
        load_lifecycle_state(lifecycle)
    assert swapped is True


def test_lifecycle_reader_enforces_size_cap(tmp_path: Path) -> None:
    lifecycle = tmp_path / ".runpod" / "pod_lifecycle.json"
    lifecycle.parent.mkdir()
    lifecycle.write_text(
        json.dumps({"padding": "x" * (2 * 1024 * 1024)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunpodLifecycleStateError, match="size limit"):
        load_lifecycle_state(lifecycle)


@pytest.mark.parametrize("malformation", ["hash", "phase"])
def test_lifecycle_reader_rejects_rehashed_malformed_historical_authorization(
    tmp_path: Path,
    malformation: str,
) -> None:
    lifecycle = _state(tmp_path / ".runpod" / "pod_lifecycle.json")
    state = json.loads(lifecycle.read_text(encoding="utf-8"))
    historical = dict(state["current_authorization"])
    historical["session_hash"] = stable_hash({"session": "historical"})
    historical["reservation_id"] = stable_hash({"reservation": "historical"})
    if malformation == "hash":
        historical["approval_hash"] = "sha256:not-a-valid-digest"
        error_pattern = r"historical\[0\].*hash"
    else:
        historical["phase"] = "arbitrary_paid_phase"
        error_pattern = r"historical\[0\].*phase"
    state["authorization_history"] = [historical]
    state["record_hash"] = stable_hash(
        {key: value for key, value in state.items() if key != "record_hash"}
    )
    write_json(lifecycle, state)

    with pytest.raises(RunpodLifecycleStateError, match=error_pattern):
        load_lifecycle_state(lifecycle)


def test_lifecycle_reader_rejects_history_that_reuses_current_identity(
    tmp_path: Path,
) -> None:
    lifecycle = _state(tmp_path / ".runpod" / "pod_lifecycle.json")
    state = json.loads(lifecycle.read_text(encoding="utf-8"))
    state["authorization_history"] = [dict(state["current_authorization"])]
    state["record_hash"] = stable_hash(
        {key: value for key, value in state.items() if key != "record_hash"}
    )
    write_json(lifecycle, state)

    with pytest.raises(RunpodLifecycleStateError, match="reuses a session, reservation"):
        load_lifecycle_state(lifecycle)
