from __future__ import annotations

from pathlib import Path

import pytest

from model_forensics.io import stable_hash
from model_forensics.paid_phase_receipt import (
    PaidPhaseReceiptError,
    PaidPhaseReceiptStore,
)


def test_phase_receipt_is_idempotent_for_exact_same_approved_plan(tmp_path: Path) -> None:
    store = PaidPhaseReceiptStore(tmp_path)
    kwargs = {
        "command_phase": "behavior_baseline_gpu",
        "approval_content_hash": stable_hash("approval"),
        "approval_id_hash": stable_hash("approval-id"),
        "bindings_hash": stable_hash("bindings"),
        "plan_hash": stable_hash("plan"),
    }
    first = store.authorize(**kwargs)
    second = store.authorize(**kwargs)
    assert first == second
    assert first["receipt_hash"].startswith("sha256:")


def test_phase_receipt_blocks_alternate_plan_or_approval(tmp_path: Path) -> None:
    store = PaidPhaseReceiptStore(tmp_path)
    common = {
        "command_phase": "resample_gpu",
        "approval_content_hash": stable_hash("approval"),
        "approval_id_hash": stable_hash("approval-id"),
        "bindings_hash": stable_hash("bindings"),
        "plan_hash": stable_hash("plan"),
    }
    store.authorize(**common)
    for field in ("approval_content_hash", "plan_hash"):
        changed = dict(common)
        changed[field] = stable_hash(f"changed-{field}")
        with pytest.raises(PaidPhaseReceiptError, match="already authorized"):
            store.authorize(**changed)


def test_phase_receipt_rejects_noncanonical_phase_and_placeholder_hashes(
    tmp_path: Path,
) -> None:
    store = PaidPhaseReceiptStore(tmp_path)
    with pytest.raises(PaidPhaseReceiptError, match="canonical"):
        store.authorize(
            command_phase="custom",
            approval_content_hash=stable_hash("approval"),
            approval_id_hash=stable_hash("approval-id"),
            bindings_hash=stable_hash("bindings"),
            plan_hash=stable_hash("plan"),
        )
    with pytest.raises(PaidPhaseReceiptError, match="placeholder"):
        store.authorize(
            command_phase="lens_gpu",
            approval_content_hash="sha256:" + "0" * 64,
            approval_id_hash=stable_hash("approval-id"),
            bindings_hash=stable_hash("bindings"),
            plan_hash=stable_hash("plan"),
        )
