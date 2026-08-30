"""Authenticated, ledger-neutral upgrade of one legacy GPU settlement.

This module exists for a narrow recovery case: a stopped RunPod session was
already reconciled in the canonical ledger using a v1 watchdog settlement,
then stronger external-stop evidence became available.  The upgrade preserves
the exact v1 bytes and replaces only the private settlement evidence.  It does
not call a provider and it never writes the cost ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.gpu_budget import (
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    GpuPhaseBudgetReservation,
    load_gpu_phase_budget_reservation,
)
from model_forensics.io import stable_hash
from model_forensics.runpod_recovery import load_external_stop_receipt

SETTLEMENT_V2_PROTOCOL = "cumulative-gpu-phase-settlement-v2"
LEGACY_SETTLEMENT_V1_FILENAME = "settlement.v1.json"
CANONICAL_SETTLEMENT_FILENAME = "settlement.json"
WATCHDOG_STATE_FILENAME = "runpod_watchdog.json"
EXTERNAL_STOP_RECEIPT_FILENAME = "external_stop_receipt.json"

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_WATCHDOG_PROTOCOL = "runpod-gpu-cost-watchdog-v2"
_AMOUNT_TOLERANCE_USD = 1e-6


class SettlementUpgradeError(RuntimeError):
    """The legacy settlement cannot be upgraded without losing provenance."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SettlementUpgradeError("authenticated JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SettlementUpgradeError(f"authenticated JSON contains non-finite value {value}")


def _decode_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SettlementUpgradeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SettlementUpgradeError(f"{label} must be a JSON object")
    return value


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_private_root(project_root: Path) -> Path:
    private_root = project_root / ".runpod"
    try:
        details = private_root.lstat()
    except OSError as exc:
        raise SettlementUpgradeError("private .runpod root is missing") from exc
    if not stat.S_ISDIR(details.st_mode) or private_root.is_symlink():
        raise SettlementUpgradeError("private .runpod root is unsafe")
    return private_root


def _private_path(
    path: str | Path,
    *,
    project_root: Path,
    label: str,
    must_exist: bool = True,
) -> Path:
    private_root = _require_private_root(project_root)
    supplied = Path(path)
    candidate = _absolute_without_symlink_resolution(
        supplied if supplied.is_absolute() else project_root / supplied
    )
    if not candidate.is_relative_to(private_root):
        raise SettlementUpgradeError(f"{label} must stay under private .runpod/")
    relative = candidate.relative_to(private_root)
    current = private_root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            details = current.lstat()
        except OSError as exc:
            raise SettlementUpgradeError(f"{label} parent is missing or unsafe") from exc
        if not stat.S_ISDIR(details.st_mode) or current.is_symlink():
            raise SettlementUpgradeError(f"{label} parent is unsafe")
    if must_exist:
        _require_private_file(candidate, label=label)
    elif candidate.is_symlink():
        raise SettlementUpgradeError(f"{label} path is an unsafe symlink")
    return candidate


def _require_private_file(path: Path, *, label: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SettlementUpgradeError(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or path.is_symlink()
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
    ):
        raise SettlementUpgradeError(f"{label} is not an exclusively owned regular file")
    return details


def _require_project_file(path: Path, *, project_root: Path, label: str) -> None:
    if not path.is_relative_to(project_root):
        raise SettlementUpgradeError(f"{label} is outside the project root")
    current = project_root
    for part in path.relative_to(project_root).parts[:-1]:
        current = current / part
        try:
            details = current.lstat()
        except OSError as exc:
            raise SettlementUpgradeError(f"{label} parent is missing or unsafe") from exc
        if not stat.S_ISDIR(details.st_mode) or current.is_symlink():
            raise SettlementUpgradeError(f"{label} parent is unsafe")
    _require_private_file(path, label=label)


def _read_private_bytes(path: Path, *, label: str) -> bytes:
    before = _require_private_file(path, label=label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SettlementUpgradeError(f"{label} is unreadable") from exc
    after = _require_private_file(path, label=label)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != after.st_size:
        raise SettlementUpgradeError(f"{label} changed while it was read")
    return raw


def _file_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _finite_amount(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise SettlementUpgradeError(f"{label} must be finite and non-negative")
    return float(value)


def _require_same_amount(observed: Any, expected: float, *, label: str) -> float:
    parsed = _finite_amount(observed, label=label)
    if abs(parsed - expected) > _AMOUNT_TOLERANCE_USD:
        raise SettlementUpgradeError(f"{label} disagrees with the legacy settlement")
    return parsed


def _validate_legacy_settlement(
    payload: dict[str, Any],
    *,
    reservation: GpuPhaseBudgetReservation,
) -> float:
    expected_keys = {
        "schema_version",
        "protocol_version",
        "phase",
        "reservation_id",
        "reservation_record_hash",
        "session_hash",
        "provider_incurred_usd",
        "watchdog_state_hash",
        "status",
        "record_hash",
    }
    if set(payload) != expected_keys:
        raise SettlementUpgradeError("legacy settlement has an unexpected schema")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_version") != GPU_PHASE_SETTLEMENT_PROTOCOL
        or payload.get("status") != "settled"
    ):
        raise SettlementUpgradeError("legacy settlement is incomplete or unsupported")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _HASH_RE.fullmatch(record_hash) is None
        or record_hash != stable_hash(unsigned)
    ):
        raise SettlementUpgradeError("legacy settlement record hash mismatch")
    reservation_manifest = reservation.manifest()
    identities = {
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation_manifest["record_hash"],
        "session_hash": reservation.session_hash,
    }
    for field, expected in identities.items():
        if payload.get(field) != expected:
            raise SettlementUpgradeError(
                f"legacy settlement {field} disagrees with the reservation"
            )
    watchdog_hash = payload.get("watchdog_state_hash")
    if not isinstance(watchdog_hash, str) or _HASH_RE.fullmatch(watchdog_hash) is None:
        raise SettlementUpgradeError("legacy settlement watchdog state hash is invalid")
    return _finite_amount(
        payload.get("provider_incurred_usd"),
        label="legacy provider-incurred amount",
    )


def _validate_watchdog(raw: bytes, *, expected_hash: str) -> dict[str, Any]:
    watchdog = _decode_json_bytes(raw, label="legacy watchdog state")
    if (
        watchdog.get("schema_version") != 2
        or watchdog.get("watchdog_version") != _WATCHDOG_PROTOCOL
        or watchdog.get("status") != "stopped_confirmed"
    ):
        raise SettlementUpgradeError("legacy watchdog state is not stopped_confirmed")
    if stable_hash(watchdog) != expected_hash:
        raise SettlementUpgradeError("legacy settlement watchdog state hash mismatch")
    return watchdog


def _validated_ledger_amount(
    *,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    expected_amount: float,
) -> float:
    document = ledger.document()
    matching = [
        entry
        for entry in document["entries"]
        if entry.get("entry_id") == reservation.reservation_id
    ]
    if len(matching) != 1:
        raise SettlementUpgradeError(
            "canonical ledger does not contain exactly one legacy reservation"
        )
    entry = matching[0]
    if (
        entry.get("kind") != "gpu"
        or entry.get("status") != "incurred"
        or entry.get("description") != reservation.description
    ):
        raise SettlementUpgradeError("canonical ledger incurred entry identity mismatch")
    return _require_same_amount(
        entry.get("amount_usd"),
        expected_amount,
        label="canonical ledger incurred amount",
    )


def _validate_external_receipt(
    path: Path,
    *,
    reservation: GpuPhaseBudgetReservation,
    expected_amount: float,
) -> dict[str, Any]:
    # Strict decoding rejects duplicate keys before the shared semantic loader.
    _decode_json_bytes(
        _read_private_bytes(path, label="external-stop receipt"),
        label="external-stop receipt",
    )
    try:
        payload = load_external_stop_receipt(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SettlementUpgradeError("external-stop receipt is not authenticated") from exc
    identities = {
        "session_hash": reservation.session_hash,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
    }
    for field, expected in identities.items():
        if payload.get(field) != expected:
            raise SettlementUpgradeError(
                f"external-stop receipt {field} disagrees with the reservation"
            )
    status = payload.get("billing_status")
    evidence_kind = payload.get("evidence_kind")
    if (status, evidence_kind) not in {
        ("pending", "provider_timestamps_conservative_ceiling"),
        ("final", "provider_billing_row"),
    }:
        raise SettlementUpgradeError("external-stop billing status/evidence pair is invalid")
    _require_same_amount(
        payload.get("settlement_amount_usd"),
        expected_amount,
        label="external-stop settlement amount",
    )
    billing = payload.get("billing_evidence")
    if not isinstance(billing, dict):  # also enforced by the shared loader
        raise SettlementUpgradeError("external-stop billing evidence is missing")
    amount_field = (
        "conservative_ceiling_usd" if status == "pending" else "provider_amount_usd"
    )
    _require_same_amount(
        billing.get(amount_field),
        expected_amount,
        label=f"external-stop {amount_field}",
    )
    return payload


def _encoded_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exact_archive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _read_private_bytes(path, label="legacy settlement archive") != raw:
            raise SettlementUpgradeError(
                "existing legacy settlement archive has different exact bytes"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            if path.is_symlink() or _read_private_bytes(
                path, label="legacy settlement archive"
            ) != raw:
                raise SettlementUpgradeError(
                    "legacy settlement archive was concurrently claimed"
                ) from exc
        path.chmod(0o600)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_if_unchanged(path: Path, *, expected: bytes, replacement: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.v2.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        if _read_private_bytes(path, label="canonical legacy settlement") != expected:
            raise SettlementUpgradeError("canonical legacy settlement changed before upgrade")
        os.replace(temporary_name, path)
        path.chmod(0o600)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _v2_payload(
    *,
    reservation: GpuPhaseBudgetReservation,
    legacy: dict[str, Any],
    legacy_raw: bytes,
    external: dict[str, Any],
    amount: float,
) -> dict[str, Any]:
    billing = external["billing_evidence"]
    provider_amount = billing.get("provider_amount_usd")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol_version": SETTLEMENT_V2_PROTOCOL,
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
        "provider_incurred_usd": provider_amount,
        "accounted_gpu_usd": amount,
        "billing_status": external["billing_status"],
        "evidence_kind": external["evidence_kind"],
        "external_stop_receipt_hash": external["record_hash"],
        "stop_evidence_hash": external["stop_evidence_hash"],
        "billing_evidence_hash": external["billing_evidence_hash"],
        "legacy_settlement_v1_record_hash": legacy["record_hash"],
        "legacy_settlement_v1_file_hash": _file_hash(legacy_raw),
        "legacy_watchdog_state_hash": legacy["watchdog_state_hash"],
        "status": "settled",
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def upgrade_legacy_gpu_settlement(
    *,
    project_root: str | Path,
    reservation_receipt_path: str | Path,
    cost_ledger_path: str | Path,
    watchdog_state_path: str | Path,
    external_stop_receipt_path: str | Path,
    settlement_path: str | Path,
    limits: BudgetLimits,
) -> dict[str, Any]:
    """Upgrade one canonical v1 settlement without touching provider or ledger state."""

    root = Path(project_root).resolve()
    private_root = _require_private_root(root)
    reservation_path = _private_path(
        reservation_receipt_path,
        project_root=root,
        label="GPU reservation receipt",
    )
    canonical_path = _private_path(
        settlement_path,
        project_root=root,
        label="canonical settlement",
    )
    external_path = _private_path(
        external_stop_receipt_path,
        project_root=root,
        label="external-stop receipt",
    )
    watchdog_path = _private_path(
        watchdog_state_path,
        project_root=root,
        label="legacy watchdog state",
    )
    if canonical_path.name != CANONICAL_SETTLEMENT_FILENAME:
        raise SettlementUpgradeError("canonical settlement path must end in settlement.json")
    session_dir = canonical_path.parent
    if (
        session_dir.parent.name != "sessions"
        or session_dir.parent.parent != private_root
        or _RAW_HASH_RE.fullmatch(session_dir.name) is None
    ):
        raise SettlementUpgradeError("canonical settlement is outside its exact session path")
    if external_path != session_dir / EXTERNAL_STOP_RECEIPT_FILENAME:
        raise SettlementUpgradeError("external-stop receipt path disagrees with the session")
    if watchdog_path != session_dir / WATCHDOG_STATE_FILENAME:
        raise SettlementUpgradeError("legacy watchdog path disagrees with the session")
    archive_path = _private_path(
        session_dir / LEGACY_SETTLEMENT_V1_FILENAME,
        project_root=root,
        label="legacy settlement archive",
        must_exist=False,
    )

    # Reservation loading validates its exact schema, content hash, and internal
    # accounting invariants.  Decode once strictly first to reject duplicate keys.
    _decode_json_bytes(
        _read_private_bytes(reservation_path, label="GPU reservation receipt"),
        label="GPU reservation receipt",
    )
    try:
        reservation = load_gpu_phase_budget_reservation(reservation_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SettlementUpgradeError("GPU reservation receipt is not authenticated") from exc
    if session_dir.name != reservation.session_hash.removeprefix("sha256:"):
        raise SettlementUpgradeError("settlement session path disagrees with the reservation")
    expected_reservation_path = (
        private_root / "reservations" / f"{reservation.phase}.json"
    )
    if reservation_path != expected_reservation_path:
        raise SettlementUpgradeError("GPU reservation receipt path drifted from its phase")

    supplied_ledger_path = Path(cost_ledger_path)
    ledger_path = _absolute_without_symlink_resolution(
        supplied_ledger_path
        if supplied_ledger_path.is_absolute()
        else root / supplied_ledger_path
    )
    expected_ledger_path = root / "data" / "manifests" / "cost_ledger.yaml"
    if ledger_path != expected_ledger_path:
        raise SettlementUpgradeError("cost ledger path is not the canonical project ledger")
    _require_project_file(
        ledger_path,
        project_root=root,
        label="canonical cost ledger",
    )

    current_raw = _read_private_bytes(canonical_path, label="canonical settlement")
    current = _decode_json_bytes(current_raw, label="canonical settlement")
    archive_raw = (
        _read_private_bytes(archive_path, label="legacy settlement archive")
        if archive_path.exists()
        else current_raw
    )
    legacy = _decode_json_bytes(archive_raw, label="legacy settlement archive")
    amount = _validate_legacy_settlement(legacy, reservation=reservation)

    watchdog_raw = _read_private_bytes(watchdog_path, label="legacy watchdog state")
    _validate_watchdog(watchdog_raw, expected_hash=str(legacy["watchdog_state_hash"]))
    ledger = CostLedger(ledger_path, limits)
    _validated_ledger_amount(
        ledger=ledger,
        reservation=reservation,
        expected_amount=amount,
    )
    external = _validate_external_receipt(
        external_path,
        reservation=reservation,
        expected_amount=amount,
    )
    expected_v2 = _v2_payload(
        reservation=reservation,
        legacy=legacy,
        legacy_raw=archive_raw,
        external=external,
        amount=amount,
    )
    expected_v2_raw = _encoded_json(expected_v2)

    if current.get("schema_version") == 2:
        if not archive_path.exists():
            raise SettlementUpgradeError("upgraded settlement lacks its exact v1 archive")
        if current != expected_v2 or current_raw != expected_v2_raw:
            raise SettlementUpgradeError("existing settlement v2 has different content")
        return expected_v2
    if current_raw != archive_raw:
        raise SettlementUpgradeError("canonical v1 settlement and archive bytes disagree")

    _write_exact_archive(archive_path, archive_raw)
    # Revalidate the ledger after the archive claim.  This catches any drift
    # before the only replacement performed by this operation.
    _validated_ledger_amount(
        ledger=ledger,
        reservation=reservation,
        expected_amount=amount,
    )
    _replace_if_unchanged(
        canonical_path,
        expected=current_raw,
        replacement=expected_v2_raw,
    )
    return expected_v2


__all__ = [
    "CANONICAL_SETTLEMENT_FILENAME",
    "LEGACY_SETTLEMENT_V1_FILENAME",
    "SETTLEMENT_V2_PROTOCOL",
    "SettlementUpgradeError",
    "upgrade_legacy_gpu_settlement",
]
