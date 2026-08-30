from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import reserve_gpu_phase_budget
from model_forensics.io import write_json
from model_forensics.runpod_session_path import (
    RunpodSessionPathError,
    canonical_host_session_directory,
)


def _reservation(root: Path, phase: str = "behavior_treatment_gpu") -> Path:
    (root / ".runpod" / "reservations").mkdir(parents=True)
    ledger = CostLedger(
        root / "data" / "manifests" / "cost_ledger.yaml",
        BudgetLimits(gpu=220, api=100, total=325),
    )
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase=phase,
        session_id="canonical-session-path-fixture",
        approved_phase_maximum_usd=10,
        approved_maximum_runtime_hours=1,
        live_hourly_total_usd=10,
    )
    path = root / ".runpod" / "reservations" / f"{phase}.json"
    write_json(path, reservation.manifest())
    return path


def test_authenticated_reservation_derives_contained_lowercase_hex_path(
    tmp_path: Path,
) -> None:
    receipt = _reservation(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))

    result = canonical_host_session_directory(
        project_root=tmp_path,
        phase="behavior_treatment_gpu",
        reservation_path=receipt,
    )

    assert result == (
        tmp_path
        / ".runpod"
        / "sessions"
        / payload["session_hash"].removeprefix("sha256:")
    )
    assert not result.exists()


def test_invalid_reservation_fails_before_creating_session_directory(
    tmp_path: Path,
) -> None:
    receipt = _reservation(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["session_hash"] = "sha256:../" + "a" * 61
    write_json(receipt, payload)

    with pytest.raises(RunpodSessionPathError, match="not authenticated"):
        canonical_host_session_directory(
            project_root=tmp_path,
            phase="behavior_treatment_gpu",
            reservation_path=receipt,
        )
    assert not (tmp_path / ".runpod" / "sessions").exists()


def test_noncanonical_receipt_and_symlinked_sessions_root_fail_closed(
    tmp_path: Path,
) -> None:
    receipt = _reservation(tmp_path)
    copied = tmp_path / "receipt.json"
    copied.write_bytes(receipt.read_bytes())
    with pytest.raises(RunpodSessionPathError, match="not canonical"):
        canonical_host_session_directory(
            project_root=tmp_path,
            phase="behavior_treatment_gpu",
            reservation_path=copied,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".runpod" / "sessions").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(RunpodSessionPathError, match="sessions root is unsafe"):
        canonical_host_session_directory(
            project_root=tmp_path,
            phase="behavior_treatment_gpu",
            reservation_path=receipt,
        )
    assert list(outside.iterdir()) == []
