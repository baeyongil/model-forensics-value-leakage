"""Single-plan replay lock for each approved paid execution phase."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from model_forensics.approval import PAID_COMMAND_PHASES
from model_forensics.io import read_json, stable_hash, write_json

PAID_PHASE_RECEIPT_PROTOCOL = "paid-phase-single-plan-v1"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PaidPhaseReceiptError(RuntimeError):
    """A paid phase was already authorized under different immutable inputs."""


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


def _valid_hash(value: str, *, label: str) -> str:
    if _HASH_RE.fullmatch(value) is None:
        raise PaidPhaseReceiptError(f"{label} must be a namespaced SHA-256")
    digest = value.split(":", 1)[1]
    if len(set(digest)) == 1:
        raise PaidPhaseReceiptError(f"{label} contains a placeholder digest")
    return value


class PaidPhaseReceiptStore:
    """Persist one immutable authorization receipt per canonical paid phase."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.chmod(0o700)
        except OSError:  # pragma: no cover - platform permission model
            pass
        self._lock_path = self.directory / ".receipt.lock"

    def authorize(
        self,
        *,
        command_phase: str,
        approval_content_hash: str,
        approval_id_hash: str,
        bindings_hash: str,
        plan_hash: str,
    ) -> dict[str, Any]:
        if command_phase not in PAID_COMMAND_PHASES:
            raise PaidPhaseReceiptError("paid command phase is not canonical")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": PAID_PHASE_RECEIPT_PROTOCOL,
            "command_phase": command_phase,
            "approval_content_hash": _valid_hash(
                approval_content_hash, label="approval_content_hash"
            ),
            "approval_id_hash": _valid_hash(approval_id_hash, label="approval_id_hash"),
            "bindings_hash": _valid_hash(bindings_hash, label="bindings_hash"),
            "plan_hash": _valid_hash(plan_hash, label="plan_hash"),
        }
        payload["receipt_hash"] = stable_hash(payload)
        path = self.directory / f"{command_phase}.json"
        with _lock(self._lock_path):
            if path.exists():
                observed = read_json(path)
                if not isinstance(observed, dict) or observed.get("receipt_hash") != stable_hash(
                    {key: value for key, value in observed.items() if key != "receipt_hash"}
                ):
                    raise PaidPhaseReceiptError("existing paid phase receipt hash mismatch")
                if observed != payload:
                    raise PaidPhaseReceiptError(
                        f"paid phase {command_phase} was already authorized for another plan"
                    )
                return observed
            write_json(path, payload)
            try:
                path.chmod(0o600)
            except OSError:  # pragma: no cover
                pass
        return payload


__all__ = [
    "PAID_PHASE_RECEIPT_PROTOCOL",
    "PaidPhaseReceiptError",
    "PaidPhaseReceiptStore",
]
