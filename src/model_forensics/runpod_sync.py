"""Build a claim-safe host-to-Pod bootstrap bundle.

The host re-arm watcher must own the current session directory before provider
start.  The remote bootstrap must create that same directory itself.  Copying
``.runpod/sessions`` wholesale therefore turns a valid host guard into a remote
duplicate-claim failure.  This module selects only the current lifecycle,
reservation, ledger, private approval locks, and evidence for *completed*
prior sessions; the current host-watch directory is deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from model_forensics.approval import (
    PaidRunApprovalError,
    load_paid_run_approval,
    validate_paid_run_approval,
)
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.execution_bindings import (
    ApiRouteQuoteLockError,
    GpuQuoteLockError,
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.gpu_budget import (
    GpuPhaseBudgetReservation,
    approved_gpu_phase_maximum_usd,
    load_gpu_phase_budget_reservation,
)
from model_forensics.io import sha256_file, stable_hash
from model_forensics.runpod_lifecycle_state import (
    authorization_from_state,
    load_lifecycle_state,
)
from model_forensics.runpod_no_start import NO_START_RECEIPT_FILENAME
from model_forensics.runpod_sessions import (
    LEGACY_SETTLEMENT_V1_FILENAME,
    WATCHDOG_STATE_FILENAME,
    validate_completed_runpod_sessions,
)
from model_forensics.runpod_watchdog import (
    HOST_REARM_ACK_FILENAME,
    WATCHDOG_VERSION,
    WatchdogError,
    validate_host_rearm_ack,
)

SYNC_PROTOCOL = "runpod-selective-bootstrap-sync-v1"
SOURCE_REPOSITORY_URL = (
    "https://github.com/baeyongil/model-forensics-value-leakage.git"
)
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACED_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PRIVATE_CONTROL_PATHS = (
    ".runpod/gpu_quote_lock.json",
    ".runpod/api_route_quote_lock.json",
    ".runpod/paid_run_approval.json",
)
_CURRENT_HOST_SESSION_FILES = frozenset(
    {
        "host_rearm_watchdog.json",
        HOST_REARM_ACK_FILENAME,
    }
)
_EXTERNAL_SESSION_FILES = (
    "external_stop_receipt.json",
    "settlement.json",
)
_WATCHDOG_SESSION_FILES = (
    "gpu_budget_bootstrap.json",
    WATCHDOG_STATE_FILENAME,
    "settlement.json",
)
_LEGACY_SETTLEMENT_FIELDS = frozenset(
    {
        "legacy_settlement_v1_record_hash",
        "legacy_settlement_v1_file_hash",
        "legacy_watchdog_state_hash",
    }
)
_SOURCE_PATHS = (
    "src",
    "scripts",
    "config",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "uv.lock",
)
_SOURCE_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_HOST_GUARD_MAX_AGE_SECONDS = 20.0
_SYNCABLE_RUNNING_OPERATIONS = frozenset({"rearmed"})
_HOST_LIMIT_KEYS = frozenset(
    {
        "gpu_hard_stop_usd",
        "global_safe_budget_usd",
        "safe_budget_usd",
        "safety_margin_fraction",
        "maximum_runtime_hours",
        "maximum_approved_hourly_total_usd",
        "maximum_approved_compute_hourly_usd",
        "maximum_approved_storage_hourly_usd",
        "prior_committed_gpu_usd",
    }
)
_HOST_DEADLINE_KEYS = frozenset(
    {
        "budget_deadline",
        "runtime_deadline",
        "effective_deadline",
        "limiting_reason",
        "remaining_seconds",
        "calculation_hourly_usd",
        "incurred_cost_usd",
    }
)
_LIVE_METADATA_KEYS = frozenset(
    {
        "provider_api",
        "provider_evidence_unavailable",
        "pod_id",
        "pod_name",
        "gpu_count",
        "provider_gpu_id",
        "gpu_display_name",
        "runtime_gpu_count",
        "machine_id_hash",
        "execution_identity_hash",
        "data_center_id",
        "cuda_version",
        "secure_cloud",
        "container_image",
        "container_disk_gb",
        "persistent_volume_disk_gb",
        "persistent_volume_mount_path",
        "ports",
        "global_networking_enabled",
        "ssh_ready",
        "direct_ssh_ready",
        "direct_ssh_endpoint_hash",
        "environment_verified",
        "desired_status",
        "cost_per_hr",
        "adjusted_cost_per_hr",
        "effective_hourly_usd",
        "last_started_at",
        "observed_at",
        "locked",
        "interruptible",
        "network_volume_attached",
    }
)


class RunpodSyncError(RuntimeError):
    """A selective sync would omit required evidence or copy a live claim."""


def _validated_source_commit(root: Path) -> str:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunpodSyncError("source checkout cannot be authenticated") from exc

    top_level = git("rev-parse", "--show-toplevel")
    if top_level.returncode != 0:
        raise RunpodSyncError("project root is not an authenticated git checkout")
    try:
        observed_root = Path(top_level.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RunpodSyncError("git checkout root is unavailable") from exc
    if observed_root != root:
        raise RunpodSyncError("project root is not the git checkout root")
    revision = git("rev-parse", "--verify", "HEAD")
    commit = revision.stdout.strip()
    if revision.returncode != 0 or _SOURCE_COMMIT_RE.fullmatch(commit) is None:
        raise RunpodSyncError("source commit is missing or malformed")
    origin = git("remote", "get-url", "origin")
    if origin.returncode != 0 or origin.stdout.strip() != SOURCE_REPOSITORY_URL:
        raise RunpodSyncError("source checkout origin is not the canonical public repository")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_SOURCE_PATHS,
    )
    if status.returncode != 0 or status.stdout:
        raise RunpodSyncError("tracked runner source is dirty or contains untracked code")
    return commit


def _regular_owned(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RunpodSyncError(f"{label} is missing or unsafe")
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
    ):
        raise RunpodSyncError(f"{label} must be an exclusively owned regular file")
    return path


def _relative_file(root: Path, path: Path, *, label: str) -> str:
    source = _regular_owned(path, label=label)
    try:
        relative = source.resolve().relative_to(root)
    except ValueError as exc:
        raise RunpodSyncError(f"{label} escapes the project root") from exc
    return relative.as_posix()


def _active_ledger_entry(
    ledger: CostLedger,
    *,
    reservation_id: str,
    expected_amount_usd: float,
    expected_description: str,
) -> None:
    document = ledger.document()
    matches = [
        item
        for item in document["entries"]
        if item.get("entry_id") == reservation_id
    ]
    if len(matches) != 1:
        raise RunpodSyncError("current reservation has no unique canonical ledger entry")
    entry = matches[0]
    amount = entry.get("amount_usd")
    if (
        entry.get("kind") != "gpu"
        or entry.get("status") != "estimated"
        or entry.get("description") != expected_description
        or isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
        or abs(float(amount) - expected_amount_usd) > 1e-6
    ):
        raise RunpodSyncError("current reservation ledger claim is not exact and active")
    active_gpu = [
        item
        for item in document["entries"]
        if item.get("kind") == "gpu" and item.get("status") == "estimated"
    ]
    if len(active_gpu) != 1 or active_gpu[0].get("entry_id") != reservation_id:
        raise RunpodSyncError("current reservation is not the sole active GPU commitment")


def _validate_private_controls(
    *,
    root: Path,
    state: Mapping[str, Any],
    phase: str,
    observed_at: datetime,
    source_commit: str,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any]]:
    gpu_quote_path = root / ".runpod" / "gpu_quote_lock.json"
    api_quote_path = root / ".runpod" / "api_route_quote_lock.json"
    approval_path = root / ".runpod" / "paid_run_approval.json"
    selected = (gpu_quote_path, api_quote_path, approval_path)
    records = {
        path: _stable_file_record(path, label="private approval control")
        for path in selected
    }
    try:
        gpu_quote = load_gpu_quote_lock(gpu_quote_path)
        api_quote = load_api_route_quote_lock(api_quote_path)
        approval = load_paid_run_approval(approval_path)
        validate_paid_run_approval(
            approval,
            expected=approval.bindings,
            command_phase=phase,
            now=observed_at,
            expected_source_commit=source_commit,
        )
        gpu_lock_path = _regular_owned(
            root / "config" / "gpu_lock.yaml",
            label="GPU/software lock",
        )
        gpu_lock_record = _stable_file_record(
            gpu_lock_path,
            label="GPU/software lock",
        )
        gpu_lock = yaml.safe_load(gpu_lock_path.read_text(encoding="utf-8"))
        _require_same_record(
            gpu_lock_record,
            _stable_file_record(gpu_lock_path, label="GPU/software lock"),
            label="GPU/software lock",
        )
    except (
        OSError,
        UnicodeDecodeError,
        yaml.YAMLError,
        PaidRunApprovalError,
        GpuQuoteLockError,
        ApiRouteQuoteLockError,
    ) as exc:
        raise RunpodSyncError("private approval controls are not authenticated") from exc
    current = state.get("current_authorization")
    if not isinstance(current, Mapping) or not isinstance(gpu_lock, Mapping):
        raise RunpodSyncError("private approval control binding is malformed")
    expected = {
        "approval_hash": approval.content_hash,
        "bindings_hash": stable_hash(approval.bindings.model_dump(mode="json")),
        "gpu_lock_hash": stable_hash(dict(gpu_lock)),
        "quote_hash": gpu_quote.content_hash,
    }
    if any(current.get(field) != value for field, value in expected.items()):
        raise RunpodSyncError("private approval controls disagree with lifecycle authorization")
    if (
        approval.bindings.gpu.quote.content_hash != gpu_quote.content_hash
        or approval.bindings.api_quote.content_hash != api_quote.content_hash
    ):
        raise RunpodSyncError("private quote locks disagree with paid approval")
    gpu = approval.bindings.gpu
    allocations = [
        item for item in gpu.phase_runtime_allocations if item.command_phase == phase
    ]
    spec = state.get("immutable_spec")
    spec_gpu = spec.get("gpu") if isinstance(spec, Mapping) else None
    live_rate = (
        gpu.count * gpu.quote.usd_per_gpu_hour
        + gpu.quote.running_storage_usd_per_hour
    )
    expected_maximum = (
        approved_gpu_phase_maximum_usd(
            gpu_count=gpu.count,
            quote_hourly_per_gpu_usd=gpu.quote.usd_per_gpu_hour,
            running_storage_hourly_usd=gpu.quote.running_storage_usd_per_hour,
            approved_runtime_hours=allocations[0].maximum_runtime_hours,
        )
        if len(allocations) == 1
        else None
    )
    if (
        len(allocations) != 1
        or not isinstance(spec, Mapping)
        or not isinstance(spec_gpu, Mapping)
        or spec.get("image") != gpu.container_image_digest
        or spec_gpu.get("id") != gpu.provider_gpu_id
        or spec_gpu.get("count") != gpu.count
        or abs(float(current.get("live_hourly_total_usd", -1)) - live_rate) > 1e-6
        or abs(
            float(current.get("approved_runtime_hours", -1))
            - allocations[0].maximum_runtime_hours
        )
        > 1e-9
        or expected_maximum is None
        or abs(float(current.get("approved_phase_maximum_usd", -1)) - expected_maximum)
        > 1e-6
    ):
        raise RunpodSyncError("private approval execution profile disagrees with lifecycle")
    for path, expected_record in records.items():
        _require_same_record(
            expected_record,
            _stable_file_record(path, label="private approval control"),
            label="private approval control",
        )
    return records, {
        "gpu_family": gpu.family,
        "provider_gpu_id": gpu.provider_gpu_id,
        "gpu_count": gpu.count,
        "allowed_data_center_ids": tuple(gpu.data_center_ids),
        "allowed_cuda_versions": tuple(gpu.allowed_cuda_versions),
        "container_image": gpu.container_image_digest,
        "container_disk_gb": gpu.container_disk_gb,
        "volume_disk_gb": gpu.volume_disk_gb,
        "approved_compute_hourly_usd": (
            gpu.count * gpu.quote.usd_per_gpu_hour
        ),
        "approved_storage_hourly_usd": gpu.quote.running_storage_usd_per_hour,
        "approved_total_hourly_usd": live_rate,
    }


def _prior_session_files(session: Path) -> tuple[Path, ...]:
    has_no_start = (session / NO_START_RECEIPT_FILENAME).exists()
    has_external = (session / "external_stop_receipt.json").exists()
    if has_no_start and has_external:
        raise RunpodSyncError("prior session has conflicting stop evidence")
    if has_no_start:
        names = [NO_START_RECEIPT_FILENAME, "settlement.json"]
        if (session / "gpu_budget_bootstrap.json").exists():
            names.append("gpu_budget_bootstrap.json")
    elif has_external:
        names = list(_EXTERNAL_SESSION_FILES)
        settlement_path = _regular_owned(
            session / "settlement.json",
            label="prior session settlement.json",
        )
        try:
            settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunpodSyncError("prior external settlement is unreadable") from exc
        if not isinstance(settlement, dict):
            raise RunpodSyncError("prior external settlement must be a JSON object")
        legacy_fields = _LEGACY_SETTLEMENT_FIELDS & set(settlement)
        if settlement.get("schema_version") == 2 and legacy_fields:
            if legacy_fields != _LEGACY_SETTLEMENT_FIELDS:
                raise RunpodSyncError(
                    "prior upgraded settlement legacy binding is incomplete"
                )
            names.extend(
                [LEGACY_SETTLEMENT_V1_FILENAME, WATCHDOG_STATE_FILENAME]
            )
        if (session / "gpu_budget_bootstrap.json").exists():
            names.append("gpu_budget_bootstrap.json")
    else:
        names = list(_WATCHDOG_SESSION_FILES)
    return tuple(
        _regular_owned(session / name, label=f"prior session {name}") for name in names
    )


def _validate_prior_subset(
    *,
    sessions: Mapping[str, tuple[Path, ...]],
    ledger: CostLedger,
) -> dict[Path, dict[str, Any]]:
    validated_records: dict[Path, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="runpod-prior-session-audit-") as temporary:
        audit_root = Path(temporary) / "sessions"
        audit_root.mkdir()
        for digest, files in sessions.items():
            destination = audit_root / digest
            destination.mkdir()
            for source in files:
                record = _stable_file_record(
                    source,
                    label="prior completed session evidence",
                )
                _copy_verified_source(
                    source=source,
                    target=destination / source.name,
                    expected_sha256=str(record["sha256"]),
                    expected_size=int(record["size_bytes"]),
                )
                validated_records[source] = record
        summaries = validate_completed_runpod_sessions(
            sessions_root=audit_root,
            ledger=ledger,
        )
    expected_reservations = [
        str(entry.get("entry_id"))
        for entry in ledger.document()["entries"]
        if entry.get("kind") == "gpu" and entry.get("status") == "incurred"
    ]
    observed_reservations = [str(summary["reservation_id"]) for summary in summaries]
    if (
        len(expected_reservations) != len(set(expected_reservations))
        or len(observed_reservations) != len(set(observed_reservations))
        or set(expected_reservations) != set(observed_reservations)
    ):
        raise RunpodSyncError(
            "completed prior sessions do not exactly cover incurred GPU ledger entries"
        )
    for source, expected in validated_records.items():
        _require_same_record(
            expected,
            _stable_file_record(source, label="prior completed session evidence"),
            label="prior completed session evidence",
        )
    return validated_records


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_nlink,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _stable_file_record(path: Path, *, label: str) -> dict[str, Any]:
    source = _regular_owned(path, label=label)
    before = source.lstat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise RunpodSyncError(f"{label} changed before fingerprinting")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = source.lstat()
    if (
        _file_identity(after) != _file_identity(opened)
        or _file_identity(current) != _file_identity(opened)
        or size != opened.st_size
    ):
        raise RunpodSyncError(f"{label} changed while fingerprinting")
    return {
        "sha256": f"sha256:{digest.hexdigest()}",
        "size_bytes": size,
    }


def _require_same_record(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(before) != dict(after):
        raise RunpodSyncError(f"{label} changed across semantic validation")


def _timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise RunpodSyncError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSyncError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunpodSyncError(f"{label} timestamp lacks a timezone")
    return parsed.astimezone(UTC)


def _validate_current_host_guard(
    *,
    session: Path,
    session_hash: str,
    phase: str,
    pod_id: str,
    observed_at: datetime,
    reservation: GpuPhaseBudgetReservation,
    execution_profile: Mapping[str, Any],
) -> dict[str, str]:
    if (session / "runpod_stop.request").exists():
        raise RunpodSyncError("current host watcher has a pending stop request")
    names = {item.name for item in session.iterdir()}
    if names != _CURRENT_HOST_SESSION_FILES:
        raise RunpodSyncError(
            "current host-watch session must contain exactly its acknowledgement and state"
        )
    acknowledgement_path = _regular_owned(
        session / HOST_REARM_ACK_FILENAME,
        label="current host re-arm acknowledgement",
    )
    try:
        acknowledgement_before = acknowledgement_path.lstat()
        acknowledgement_raw = acknowledgement_path.read_bytes()
        acknowledgement_after = acknowledgement_path.lstat()
        acknowledgement = json.loads(acknowledgement_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodSyncError("current host re-arm acknowledgement is unreadable") from exc
    if _file_identity(acknowledgement_before) != _file_identity(
        acknowledgement_after
    ):
        raise RunpodSyncError("current host re-arm acknowledgement changed while read")
    if not isinstance(acknowledgement, dict) or not isinstance(
        acknowledgement.get("lifecycle_before_hash"),
        str,
    ):
        raise RunpodSyncError("current host re-arm acknowledgement is malformed")
    try:
        validated_ack = validate_host_rearm_ack(
            acknowledgement_path,
            expected_session_hash=session_hash,
            expected_phase=phase,
            expected_lifecycle_hash=str(acknowledgement["lifecycle_before_hash"]),
            expected_pod_id=pod_id,
            observed_at=observed_at,
            maximum_age_seconds=600,
        )
    except (OSError, ValueError, WatchdogError) as exc:
        raise RunpodSyncError(
            "current host re-arm acknowledgement is not live and authenticated"
        ) from exc

    state_path = _regular_owned(
        session / "host_rearm_watchdog.json",
        label="current host watchdog state",
    )
    try:
        state_before = state_path.lstat()
        state_raw = state_path.read_bytes()
        state_after = state_path.lstat()
        state = json.loads(state_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodSyncError("current host watchdog state is unreadable") from exc
    if _file_identity(state_before) != _file_identity(state_after):
        raise RunpodSyncError("current host watchdog state changed while read")
    expected_state_keys = {
        "schema_version",
        "watchdog_version",
        "pod_id",
        "status",
        "armed_at",
        "updated_at",
        "live_metadata",
        "limits",
        "deadline",
        "stop_reason",
        "action",
        "deletion",
        "error",
    }
    if (
        not isinstance(state, dict)
        or set(state) != expected_state_keys
        or state.get("schema_version") != 2
        or state.get("watchdog_version") != WATCHDOG_VERSION
        or state.get("pod_id") != pod_id
        or state.get("status") != "armed"
        or state.get("stop_reason") is not None
        or state.get("action") != "stop_only_preserve_volume"
        or state.get("deletion") != "manual_after_verified_sync"
        or state.get("error") is not None
    ):
        raise RunpodSyncError("current host watchdog state is not safely armed")

    def number(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
        value = mapping.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RunpodSyncError(f"{label} {key} is malformed")
        return float(value)

    live_metadata = state.get("live_metadata")
    if (
        not isinstance(live_metadata, dict)
        or set(live_metadata) != _LIVE_METADATA_KEYS
        or live_metadata.get("provider_api") != "rest-v1"
        or live_metadata.get("pod_id") != pod_id
        or live_metadata.get("desired_status") != "RUNNING"
        or live_metadata.get("environment_verified") is not True
        or live_metadata.get("provider_gpu_id")
        != execution_profile.get("provider_gpu_id")
        or live_metadata.get("gpu_count") != execution_profile.get("gpu_count")
        or live_metadata.get("runtime_gpu_count")
        not in (None, execution_profile.get("gpu_count"))
        or live_metadata.get("data_center_id")
        not in execution_profile.get("allowed_data_center_ids", ())
        or live_metadata.get("cuda_version")
        not in (None, *execution_profile.get("allowed_cuda_versions", ()))
        or live_metadata.get("secure_cloud") is not True
        or live_metadata.get("container_image")
        != execution_profile.get("container_image")
        or live_metadata.get("container_disk_gb")
        != execution_profile.get("container_disk_gb")
        or live_metadata.get("persistent_volume_disk_gb")
        != execution_profile.get("volume_disk_gb")
        or live_metadata.get("persistent_volume_mount_path") != "/workspace"
        or live_metadata.get("ports") != ["22/tcp"]
        or live_metadata.get("global_networking_enabled") is not None
        or live_metadata.get("network_volume_attached") is not False
        or live_metadata.get("ssh_ready") is not True
        or live_metadata.get("direct_ssh_ready") is not True
        or live_metadata.get("locked") is not None
        or live_metadata.get("interruptible") is not None
    ):
        raise RunpodSyncError("current host watchdog live Pod binding is incomplete")
    for hash_field in (
        "machine_id_hash",
        "execution_identity_hash",
        "direct_ssh_endpoint_hash",
    ):
        value = live_metadata.get(hash_field)
        if not isinstance(value, str) or _NAMESPACED_HASH_RE.fullmatch(value) is None:
            raise RunpodSyncError("current host watchdog live identity is malformed")
    pod_name = live_metadata.get("pod_name")
    gpu_display_name = live_metadata.get("gpu_display_name")
    if (
        not isinstance(pod_name, str)
        or not pod_name
        or not isinstance(gpu_display_name, str)
        or not gpu_display_name
    ):
        raise RunpodSyncError("current host watchdog public metadata is malformed")
    unavailable = live_metadata.get("provider_evidence_unavailable")
    if unavailable != [
        "cuda_version",
        "global_networking_enabled",
        "interruptible",
        "locked",
        "runtime_gpu_count",
    ]:
        raise RunpodSyncError("current host watchdog evidence availability drifted")
    updated_at = _timestamp(state.get("updated_at"), label="host watchdog update")
    age = (observed_at - updated_at).total_seconds()
    if age < -5 or age > _HOST_GUARD_MAX_AGE_SECONDS:
        raise RunpodSyncError("current host watchdog state is stale or future-dated")
    acknowledged_at = _timestamp(
        validated_ack.get("acknowledged_at"),
        label="host acknowledgement",
    )
    armed_at = _timestamp(state.get("armed_at"), label="host watchdog arm")
    if armed_at < acknowledged_at or updated_at < armed_at:
        raise RunpodSyncError("current host watchdog chronology is invalid")

    limits = state.get("limits")
    if not isinstance(limits, dict) or set(limits) != _HOST_LIMIT_KEYS:
        raise RunpodSyncError("current host watchdog limits are malformed")
    expected_limits = {
        "gpu_hard_stop_usd": reservation.global_gpu_hard_stop_usd,
        "global_safe_budget_usd": reservation.safety_adjusted_gpu_ceiling_usd,
        "safe_budget_usd": reservation.remaining_safe_gpu_before_phase_usd,
        "safety_margin_fraction": reservation.safety_margin_fraction,
        "maximum_runtime_hours": reservation.maximum_safe_runtime_hours,
        "maximum_approved_hourly_total_usd": execution_profile.get(
            "approved_total_hourly_usd"
        ),
        "maximum_approved_compute_hourly_usd": execution_profile.get(
            "approved_compute_hourly_usd"
        ),
        "maximum_approved_storage_hourly_usd": execution_profile.get(
            "approved_storage_hourly_usd"
        ),
        "prior_committed_gpu_usd": reservation.prior_committed_gpu_usd,
    }
    for key, expected in expected_limits.items():
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isclose(
                number(limits, key, label="host watchdog limits"),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise RunpodSyncError("current host watchdog limits disagree with reservation")

    cost_per_hr = number(live_metadata, "cost_per_hr", label="live metadata")
    adjusted_cost = number(
        live_metadata,
        "adjusted_cost_per_hr",
        label="live metadata",
    )
    effective_compute = number(
        live_metadata,
        "effective_hourly_usd",
        label="live metadata",
    )
    approved_compute = float(execution_profile["approved_compute_hourly_usd"])
    if (
        cost_per_hr <= 0
        or adjusted_cost <= 0
        or not math.isclose(
            effective_compute,
            max(cost_per_hr, adjusted_cost),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or effective_compute > approved_compute + 1e-6
    ):
        raise RunpodSyncError("current host watchdog live rate is unsafe")
    started_at = _timestamp(
        live_metadata.get("last_started_at"),
        label="host watchdog provider start",
    )
    metadata_observed_at = _timestamp(
        live_metadata.get("observed_at"),
        label="host watchdog provider observation",
    )
    if (
        started_at > metadata_observed_at
        or abs((metadata_observed_at - updated_at).total_seconds()) > 1
    ):
        raise RunpodSyncError("current host watchdog provider chronology is invalid")
    deadline = state.get("deadline")
    if not isinstance(deadline, dict) or set(deadline) != _HOST_DEADLINE_KEYS:
        raise RunpodSyncError("current host watchdog deadline is malformed")
    calculation_rate = effective_compute + float(
        execution_profile["approved_storage_hourly_usd"]
    )
    if not math.isclose(
        number(deadline, "calculation_hourly_usd", label="host watchdog deadline"),
        calculation_rate,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RunpodSyncError("current host watchdog deadline rate is unauthenticated")
    expected_budget_deadline = started_at + timedelta(
        hours=reservation.remaining_safe_gpu_before_phase_usd / calculation_rate
    )
    expected_runtime_deadline = started_at + timedelta(
        hours=reservation.maximum_safe_runtime_hours
    )
    expected_effective_deadline = min(
        expected_budget_deadline,
        expected_runtime_deadline,
    )
    observed_budget_deadline = _timestamp(
        deadline.get("budget_deadline"),
        label="host watchdog budget deadline",
    )
    observed_runtime_deadline = _timestamp(
        deadline.get("runtime_deadline"),
        label="host watchdog runtime deadline",
    )
    observed_deadline = _timestamp(
        deadline.get("effective_deadline"),
        label="host watchdog deadline",
    )
    if (
        abs((observed_budget_deadline - expected_budget_deadline).total_seconds())
        > 1e-3
        or abs((observed_runtime_deadline - expected_runtime_deadline).total_seconds())
        > 1e-3
        or abs((observed_deadline - expected_effective_deadline).total_seconds())
        > 1e-3
        or observed_deadline <= observed_at
        or deadline.get("limiting_reason")
        != (
            "safe_budget"
            if expected_budget_deadline <= expected_runtime_deadline
            else "maximum_runtime"
        )
    ):
        raise RunpodSyncError("current host watchdog deadline is absent or elapsed")
    expected_remaining = round(
        max(0.0, (expected_effective_deadline - updated_at).total_seconds()),
        3,
    )
    expected_incurred = round(
        calculation_rate
        * math.ceil(max(0.0, (metadata_observed_at - started_at).total_seconds()) / 60)
        / 60,
        6,
    )
    if (
        not math.isclose(
            number(deadline, "remaining_seconds", label="host watchdog deadline"),
            expected_remaining,
            rel_tol=0.0,
            abs_tol=1e-3,
        )
        or not math.isclose(
            number(deadline, "incurred_cost_usd", label="host watchdog deadline"),
            expected_incurred,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RunpodSyncError("current host watchdog derived cost evidence is inconsistent")
    if (
        _file_identity(acknowledgement_path.lstat())
        != _file_identity(acknowledgement_after)
        or _file_identity(state_path.lstat()) != _file_identity(state_after)
    ):
        raise RunpodSyncError("current host guard changed during validation")
    process_identity_hash = validated_ack.get("watcher_process_identity_hash")
    direct_ssh_endpoint_hash = live_metadata.get("direct_ssh_endpoint_hash")
    if (
        not isinstance(process_identity_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(process_identity_hash) is None
        or not isinstance(direct_ssh_endpoint_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(direct_ssh_endpoint_hash) is None
    ):
        raise RunpodSyncError("current host guard stable identity is malformed")

    # A heartbeat necessarily changes ``updated_at``, provider ``observed_at``,
    # ``remaining_seconds``, and ``incurred_cost_usd``.  Bind every other
    # semantically validated field so revalidation accepts a fresh heartbeat
    # from the same watcher/Pod while still rejecting execution, limits, rate,
    # deadline, or endpoint drift.
    stable_live_metadata = {
        key: live_metadata[key]
        for key in sorted(_LIVE_METADATA_KEYS - {"observed_at"})
    }
    stable_live_metadata["last_started_at"] = started_at.isoformat()
    watchdog_invariant = {
        "schema_version": state["schema_version"],
        "watchdog_version": state["watchdog_version"],
        "pod_id_hash": stable_hash({"runpod_pod_id": pod_id}),
        "armed_at": armed_at.isoformat(),
        "action": state["action"],
        "deletion": state["deletion"],
        "live_metadata": stable_live_metadata,
        "limits": dict(limits),
        "deadline": {
            "budget_deadline": observed_budget_deadline.isoformat(),
            "runtime_deadline": observed_runtime_deadline.isoformat(),
            "effective_deadline": observed_deadline.isoformat(),
            "limiting_reason": deadline["limiting_reason"],
            "calculation_hourly_usd": deadline["calculation_hourly_usd"],
        },
    }
    return {
        "acknowledgement_file_hash": f"sha256:{hashlib.sha256(acknowledgement_raw).hexdigest()}",
        "acknowledgement_record_hash": str(validated_ack["record_hash"]),
        "watcher_process_identity_hash": process_identity_hash,
        "watchdog_invariant_hash": stable_hash(watchdog_invariant),
        "direct_ssh_endpoint_hash": direct_ssh_endpoint_hash,
    }


def _revalidate_plan_host_guard(
    *,
    root: Path,
    plan: Mapping[str, Any],
) -> None:
    session_hash = plan.get("session_hash")
    phase = plan.get("phase")
    expected_guard = plan.get("current_host_guard")
    source_commit = plan.get("source_commit")
    if (
        not isinstance(session_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(session_hash) is None
        or not isinstance(phase, str)
        or not isinstance(expected_guard, Mapping)
        or not isinstance(source_commit, str)
        or _SOURCE_COMMIT_RE.fullmatch(source_commit) is None
        or plan.get("source_repository_url") != SOURCE_REPOSITORY_URL
    ):
        raise RunpodSyncError("selective sync host-guard binding is malformed")
    current_time = datetime.now(UTC)
    created_at = _timestamp(plan.get("created_at"), label="selective sync creation")
    expires_at = _timestamp(plan.get("expires_at"), label="selective sync expiry")
    if (
        (expires_at - created_at).total_seconds() != 300
        or current_time < created_at - timedelta(seconds=60)
        or current_time > expires_at
    ):
        raise RunpodSyncError("selective sync plan is stale or has an invalid lifetime")
    lifecycle_path = _regular_owned(
        root / ".runpod" / "pod_lifecycle.json",
        label="lifecycle state",
    )
    state = load_lifecycle_state(lifecycle_path)
    authorization = authorization_from_state(state)
    pod = state.get("pod")
    if (
        state.get("operation") not in _SYNCABLE_RUNNING_OPERATIONS
        or not isinstance(pod, Mapping)
        or pod.get("status") != "RUNNING"
        or not isinstance(pod.get("id"), str)
        or authorization.session_hash != session_hash
        or authorization.phase != phase
        or state.get("record_hash") != plan.get("lifecycle_record_hash")
        or stable_hash({"runpod_pod_id": pod.get("id")})
        != plan.get("running_pod_id_hash")
    ):
        raise RunpodSyncError("selective sync lifecycle changed after planning")
    if _validated_source_commit(root) != source_commit:
        raise RunpodSyncError("selective sync source checkout changed after planning")
    reservation_path = _regular_owned(
        root / ".runpod" / "reservations" / f"{phase}.json",
        label="current reservation receipt",
    )
    reservation = load_gpu_phase_budget_reservation(reservation_path)
    if (
        reservation.session_hash != session_hash
        or reservation.reservation_id != authorization.reservation_id
        or reservation.manifest().get("record_hash")
        != plan.get("reservation_record_hash")
    ):
        raise RunpodSyncError("selective sync reservation changed after planning")
    _control_records, execution_profile = _validate_private_controls(
        root=root,
        state=state,
        phase=phase,
        observed_at=current_time,
        source_commit=source_commit,
    )
    current = (
        root
        / ".runpod"
        / "sessions"
        / session_hash.removeprefix("sha256:")
    )
    observed_guard = _validate_current_host_guard(
        session=current,
        session_hash=session_hash,
        phase=phase,
        pod_id=str(pod["id"]),
        observed_at=current_time,
        reservation=reservation,
        execution_profile=execution_profile,
    )
    if dict(expected_guard) != observed_guard:
        raise RunpodSyncError("selective sync host guard changed after planning")


def revalidate_selective_sync_plan(
    *,
    project_root: str | Path,
    plan: Mapping[str, Any],
) -> None:
    """Recheck the short-lived lifecycle and live host guard around transfer."""

    _revalidate_plan_host_guard(root=Path(project_root).resolve(), plan=plan)


def _copy_verified_source(
    *,
    source: Path,
    target: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    source = _regular_owned(source, label="selective sync source")
    before = source.lstat()
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    target_descriptor: int | None = None
    try:
        opened = os.fstat(source_descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise RunpodSyncError("selective sync source changed before copy")
        if opened.st_size != expected_size:
            raise RunpodSyncError("selective sync source size changed after planning")

        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        target_descriptor = os.open(target, target_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                view = view[written:]
        os.fchmod(target_descriptor, 0o600)
        os.fsync(target_descriptor)

        after = os.fstat(source_descriptor)
        current = source.lstat()
        if (
            _file_identity(after) != _file_identity(opened)
            or _file_identity(current) != _file_identity(opened)
        ):
            raise RunpodSyncError("selective sync source changed during copy")
        if copied != expected_size or f"sha256:{digest.hexdigest()}" != expected_sha256:
            raise RunpodSyncError("selective sync source changed after planning")
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(source_descriptor)

    copied_target = _regular_owned(target, label="selective sync destination")
    if (
        copied_target.stat().st_size != expected_size
        or f"sha256:{sha256_file(copied_target)}" != expected_sha256
    ):
        raise RunpodSyncError("selective sync destination verification failed")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest_durable(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = (
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def build_selective_sync_plan(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
    cost_ledger_path: str | Path,
    limits: BudgetLimits | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise RunpodSyncError("project root is not a directory")
    source_commit = _validated_source_commit(root)
    lifecycle_path = root / ".runpod" / "pod_lifecycle.json"
    lifecycle_record = _stable_file_record(
        lifecycle_path,
        label="lifecycle state",
    )
    state = load_lifecycle_state(_regular_owned(lifecycle_path, label="lifecycle state"))
    _require_same_record(
        lifecycle_record,
        _stable_file_record(lifecycle_path, label="lifecycle state"),
        label="lifecycle state",
    )
    authorization = authorization_from_state(state)
    if state.get("operation") not in _SYNCABLE_RUNNING_OPERATIONS:
        raise RunpodSyncError("selective sync requires an authenticated re-armed Pod")
    pod = state.get("pod")
    if not isinstance(pod, Mapping) or pod.get("status") != "RUNNING":
        raise RunpodSyncError("current lifecycle does not bind a running Pod")
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise RunpodSyncError("current lifecycle Pod identity is missing")
    current_time = (observed_at or datetime.now(UTC)).astimezone(UTC)
    control_records, execution_profile = _validate_private_controls(
        root=root,
        state=state,
        phase=phase,
        observed_at=current_time,
        source_commit=source_commit,
    )

    reservation_source = _regular_owned(
        Path(reservation_path).resolve(),
        label="current reservation receipt",
    )
    expected_reservation = (
        root / ".runpod" / "reservations" / f"{phase}.json"
    ).resolve()
    if reservation_source != expected_reservation:
        raise RunpodSyncError("current reservation path is not canonical for the phase")
    reservation_file_record = _stable_file_record(
        reservation_source,
        label="current reservation receipt",
    )
    reservation = load_gpu_phase_budget_reservation(reservation_source)
    _require_same_record(
        reservation_file_record,
        _stable_file_record(
            reservation_source,
            label="current reservation receipt",
        ),
        label="current reservation receipt",
    )
    reservation_record_hash = reservation.manifest()["record_hash"]
    if (
        reservation.phase != phase
        or authorization.phase != phase
        or authorization.session_hash != reservation.session_hash
        or authorization.reservation_id != reservation.reservation_id
        or authorization.reservation_record_hash != reservation_record_hash
    ):
        raise RunpodSyncError("lifecycle and current reservation identities disagree")

    ledger_source = _regular_owned(
        Path(cost_ledger_path).resolve(),
        label="canonical cost ledger",
    )
    expected_ledger = (root / "data" / "manifests" / "cost_ledger.yaml").resolve()
    if ledger_source != expected_ledger:
        raise RunpodSyncError("cost ledger path is not canonical")
    ledger_file_record = _stable_file_record(
        ledger_source,
        label="canonical cost ledger",
    )
    ledger = CostLedger(
        ledger_source,
        limits or BudgetLimits(gpu=220, api=100, total=325),
    )
    _active_ledger_entry(
        ledger,
        reservation_id=reservation.reservation_id,
        expected_amount_usd=reservation.approved_phase_maximum_usd,
        expected_description=reservation.description,
    )
    _require_same_record(
        ledger_file_record,
        _stable_file_record(ledger_source, label="canonical cost ledger"),
        label="canonical cost ledger",
    )

    current_digest = reservation.session_hash.removeprefix("sha256:")
    sessions_root = root / ".runpod" / "sessions"
    prior: dict[str, tuple[Path, ...]] = {}
    host_guard: dict[str, str] | None = None
    if sessions_root.exists():
        if sessions_root.is_symlink() or not sessions_root.is_dir():
            raise RunpodSyncError("private sessions root is unsafe")
        for session in sorted(sessions_root.iterdir(), key=lambda item: item.name):
            if session.is_symlink() or not session.is_dir() or _RAW_HASH_RE.fullmatch(
                session.name
            ) is None:
                raise RunpodSyncError("private session directory is malformed")
            if session.name == current_digest:
                host_guard = _validate_current_host_guard(
                    session=session,
                    session_hash=reservation.session_hash,
                    phase=phase,
                    pod_id=pod_id,
                    observed_at=current_time,
                    reservation=reservation,
                    execution_profile=execution_profile,
                )
                continue
            prior[session.name] = _prior_session_files(session)
    if host_guard is None:
        raise RunpodSyncError("current host-watch session is missing")
    prior_records = _validate_prior_subset(sessions=prior, ledger=ledger)

    validated_records = {
        lifecycle_path: lifecycle_record,
        reservation_source: reservation_file_record,
        ledger_source: ledger_file_record,
        **control_records,
        **prior_records,
    }

    selected = [
        _relative_file(root, lifecycle_path, label="lifecycle state"),
        _relative_file(root, reservation_source, label="current reservation receipt"),
        _relative_file(root, ledger_source, label="canonical cost ledger"),
    ]
    for relative in _PRIVATE_CONTROL_PATHS:
        selected.append(
            _relative_file(root, root / relative, label=f"private control {relative}")
        )
    for files in prior.values():
        selected.extend(
            _relative_file(root, source, label="prior completed session evidence")
            for source in files
        )
    selected = sorted(set(selected))
    if any(f".runpod/sessions/{current_digest}/" in path for path in selected):
        raise RunpodSyncError("current host-watch session leaked into the remote sync plan")
    file_records: list[dict[str, Any]] = []
    for relative in selected:
        source = root / relative
        record = _stable_file_record(source, label="selective sync inventory source")
        validated = validated_records.get(source)
        if validated is not None:
            _require_same_record(
                validated,
                record,
                label="semantically validated selective sync source",
            )
        file_records.append({"path": relative, **record})
    expires_at = current_time + timedelta(minutes=5)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": SYNC_PROTOCOL,
        "phase": phase,
        "session_hash": reservation.session_hash,
        "created_at": current_time.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "running_pod_id_hash": stable_hash({"runpod_pod_id": pod_id}),
        "source_commit": source_commit,
        "source_repository_url": SOURCE_REPOSITORY_URL,
        "lifecycle_record_hash": state["record_hash"],
        "reservation_record_hash": reservation_record_hash,
        "current_host_session_excluded": True,
        "current_host_guard": host_guard,
        "files": file_records,
    }
    plan["record_hash"] = stable_hash(plan)
    return plan


def materialize_selective_sync_bundle(
    *,
    project_root: str | Path,
    destination: str | Path,
    plan: Mapping[str, Any],
) -> Path:
    root = Path(project_root).resolve()
    private = root / ".runpod"
    if private.is_symlink() or not private.is_dir():
        raise RunpodSyncError("private .runpod root is missing or unsafe")
    private_details = private.lstat()
    if private_details.st_uid != os.getuid():
        raise RunpodSyncError("private .runpod root ownership is unsafe")
    allowed_root = private / "sync_bundles"
    if os.path.lexists(allowed_root):
        details = allowed_root.lstat()
        if (
            allowed_root.is_symlink()
            or not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.getuid()
        ):
            raise RunpodSyncError("private sync-bundle root is unsafe")
    else:
        allowed_root.mkdir(mode=0o700)
    session_hash = plan.get("session_hash")
    if (
        not isinstance(session_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(session_hash) is None
    ):
        raise RunpodSyncError("selective sync plan session hash is malformed")
    expected_output = allowed_root / session_hash.removeprefix("sha256:")
    output = Path(os.path.abspath(os.fspath(destination)))
    if output != expected_output or os.path.lexists(output):
        raise RunpodSyncError(
            "sync bundle destination must be the new session-bound sync-bundle path"
        )
    files = plan.get("files")
    if not isinstance(files, list):
        raise RunpodSyncError("selective sync plan has no file inventory")
    unsigned = {key: value for key, value in plan.items() if key != "record_hash"}
    if plan.get("record_hash") != stable_hash(unsigned):
        raise RunpodSyncError("selective sync plan hash mismatch")
    _revalidate_plan_host_guard(root=root, plan=plan)
    output.mkdir(parents=True, mode=0o700)
    try:
        for item in files:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise RunpodSyncError("selective sync file record is malformed")
            if (
                not isinstance(item["path"], str)
                or not isinstance(item["sha256"], str)
                or _NAMESPACED_HASH_RE.fullmatch(item["sha256"]) is None
                or isinstance(item["size_bytes"], bool)
                or not isinstance(item["size_bytes"], int)
                or item["size_bytes"] < 0
            ):
                raise RunpodSyncError("selective sync file record values are malformed")
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RunpodSyncError("selective sync file path is unsafe")
            target = output / relative
            _copy_verified_source(
                source=root / relative,
                target=target,
                expected_sha256=item["sha256"],
                expected_size=item["size_bytes"],
            )
        _revalidate_plan_host_guard(root=root, plan=plan)
        manifest = output / ".runpod" / "selective_sync_manifest.json"
        _write_manifest_durable(manifest, plan)
        _fsync_directory(output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output


__all__ = [
    "SOURCE_REPOSITORY_URL",
    "SYNC_PROTOCOL",
    "RunpodSyncError",
    "build_selective_sync_plan",
    "materialize_selective_sync_bundle",
    "revalidate_selective_sync_plan",
]
