"""Single-plan replay lock for each approved paid execution phase."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
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
        api_completion_preflight: Mapping[str, Any] | None = None,
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
        if api_completion_preflight is not None:
            completion = dict(api_completion_preflight)
            completion_hash = completion.get("manifest_hash")
            if not isinstance(completion_hash, str) or completion_hash != stable_hash(
                {key: value for key, value in completion.items() if key != "manifest_hash"}
            ):
                raise PaidPhaseReceiptError("API completion preflight hash mismatch")
            if completion.get("phase") != command_phase:
                raise PaidPhaseReceiptError("API completion preflight phase mismatch")
            logical_count = completion.get("logical_invocation_count")
            unique_count = completion.get("unique_request_count")
            cached_count = completion.get("authenticated_cached_count")
            pending_count = completion.get("pending_request_count")
            if (
                isinstance(logical_count, bool)
                or not isinstance(logical_count, int)
                or logical_count < 0
                or isinstance(unique_count, bool)
                or not isinstance(unique_count, int)
                or not 0 <= unique_count <= logical_count
                or isinstance(cached_count, bool)
                or not isinstance(cached_count, int)
                or not 0 <= cached_count <= unique_count
                or isinstance(pending_count, bool)
                or not isinstance(pending_count, int)
                or not 0 <= pending_count <= unique_count
                or cached_count + pending_count != unique_count
            ):
                raise PaidPhaseReceiptError("API completion preflight counts are invalid")
            for field in (
                "conservative_pending_usd",
                "additional_commitment_required_usd",
            ):
                value = completion.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                ):
                    raise PaidPhaseReceiptError(
                        f"API completion preflight {field} is invalid"
                    )
            ledger = completion.get("ledger")
            if not isinstance(ledger, Mapping):
                raise PaidPhaseReceiptError("API completion preflight ledger is invalid")
            per_route = completion.get("per_route")
            if not isinstance(per_route, Mapping) or any(
                not isinstance(route, str)
                or not route
                or not isinstance(counts, Mapping)
                for route, counts in per_route.items()
            ):
                raise PaidPhaseReceiptError("API completion preflight route counts are invalid")
            pending_identities = completion.get("pending_request_identities")
            if (
                not isinstance(pending_identities, list)
                or len(pending_identities) != pending_count
                or len(pending_identities) != len(set(pending_identities))
                or any(
                    not isinstance(identity, str) or _HASH_RE.fullmatch(identity) is None
                    for identity in pending_identities
                )
            ):
                raise PaidPhaseReceiptError(
                    "API completion preflight pending identities are invalid"
                )
            payload["api_completion_preflight"] = {
                "manifest_hash": _valid_hash(
                    completion_hash,
                    label="api_completion_preflight.manifest_hash",
                ),
                "full_inventory_hash": _valid_hash(
                    str(completion.get("full_inventory_hash")),
                    label="api_completion_preflight.full_inventory_hash",
                ),
                "logical_invocation_count": logical_count,
                "unique_request_count": unique_count,
                "authenticated_cached_count": cached_count,
                "pending_request_count": pending_count,
                "pending_inventory_hash": _valid_hash(
                    str(completion.get("pending_inventory_hash")),
                    label="api_completion_preflight.pending_inventory_hash",
                ),
                "pending_request_identities": list(pending_identities),
                "conservative_pending_usd": completion.get("conservative_pending_usd"),
                "additional_commitment_required_usd": completion.get(
                    "additional_commitment_required_usd"
                ),
                "per_route": {str(route): dict(counts) for route, counts in per_route.items()},
                "ledger": dict(ledger),
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
