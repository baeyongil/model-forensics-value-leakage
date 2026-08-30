"""Verify a selective RunPod bootstrap sync before any dependency install.

This module intentionally imports only the Python standard library.  A fresh
provider image can therefore authenticate the host-created selective-sync
manifest, its exact file inventory, and the live lifecycle/budget binding
before PyYAML or the project environment exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SYNC_PROTOCOL = "runpod-selective-bootstrap-sync-v1"
SOURCE_REPOSITORY_URL = (
    "https://github.com/baeyongil/model-forensics-value-leakage.git"
)
LIFECYCLE_PROTOCOL = "runpod-pod-lifecycle-v1"
RESERVATION_PROTOCOL = "cumulative-gpu-phase-budget-v1"
MANIFEST_RELATIVE_PATH = ".runpod/selective_sync_manifest.json"
MAXIMUM_VALIDITY_SECONDS = 300
MAXIMUM_FUTURE_SKEW_SECONDS = 60
MAXIMUM_MANIFEST_BYTES = 1024 * 1024
MAXIMUM_INVENTORY_FILE_BYTES = 64 * 1024 * 1024
MAXIMUM_INVENTORY_FILES = 4096

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_PHASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}\Z")
_CANONICAL_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)
_YAML_IMPLICIT_SCALARS = frozenset(
    {
        "null",
        "~",
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        ".nan",
        ".inf",
        "+.inf",
        "-.inf",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "created_at",
        "expires_at",
        "phase",
        "session_hash",
        "lifecycle_record_hash",
        "reservation_record_hash",
        "running_pod_id_hash",
        "source_commit",
        "source_repository_url",
        "current_host_session_excluded",
        "current_host_guard",
        "files",
        "record_hash",
    }
)
_HOST_GUARD_KEYS = frozenset(
    {
        "acknowledgement_file_hash",
        "acknowledgement_record_hash",
        "watcher_process_identity_hash",
        "watchdog_invariant_hash",
        "direct_ssh_endpoint_hash",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "acknowledged_existing_pod_id_hashes",
        "approval_hash",
        "approved_phase_maximum_usd",
        "approved_runtime_hours",
        "bindings_hash",
        "gpu_lock_hash",
        "immutable_spec_hash",
        "launch_spec_hash",
        "live_hourly_total_usd",
        "phase",
        "quote_hash",
        "reservation_id",
        "reservation_record_hash",
        "session_hash",
    }
)
_RESERVATION_FIELDS = frozenset(
    {
        "reservation_id",
        "phase",
        "session_hash",
        "approved_phase_maximum_usd",
        "approved_maximum_runtime_hours",
        "live_hourly_total_usd",
        "safety_margin_fraction",
        "global_gpu_hard_stop_usd",
        "safety_adjusted_gpu_ceiling_usd",
        "prior_incurred_gpu_usd",
        "prior_reserved_gpu_usd",
        "prior_committed_gpu_usd",
        "prior_committed_total_usd",
        "remaining_safe_gpu_before_phase_usd",
        "remaining_total_before_phase_usd",
        "maximum_safe_runtime_hours",
        "committed_gpu_after_reservation_usd",
        "committed_total_after_reservation_usd",
    }
)
_FIXED_SYNC_PATHS = frozenset(
    {
        ".runpod/pod_lifecycle.json",
        ".runpod/gpu_quote_lock.json",
        ".runpod/api_route_quote_lock.json",
        ".runpod/paid_run_approval.json",
        "data/manifests/cost_ledger.yaml",
    }
)
_PRIOR_SESSION_FILENAMES = frozenset(
    {
        "external_stop_receipt.json",
        "gpu_budget_bootstrap.json",
        "no_start_receipt.json",
        "runpod_watchdog.json",
        "settlement.json",
        "settlement.v1.json",
    }
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "anthropic_api_key",
        "api_key",
        "apikey",
        "client_secret",
        "gemini_api_key",
        "google_api_key",
        "hf_token",
        "openai_api_key",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "runpod_api_key",
        "secret",
        "token",
    }
)
_SECRET_TEXT_MARKERS = tuple(
    marker.encode("ascii")
    for marker in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "RUNPOD_API_KEY",
    )
)
_CREDENTIAL_VALUE_RE = re.compile(rb"(?:^|[^A-Za-z0-9])(?:hf_|sk-)[A-Za-z0-9_-]{20,}")
_SOURCE_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUNNER_PATHS = (
    "src",
    "scripts",
    "config",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements.lock",
    "Pipfile",
    "Pipfile.lock",
)


class RunpodSyncVerificationError(RuntimeError):
    """The remote selective sync is incomplete, stale, or unauthenticated."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunpodSyncVerificationError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> Any:
    raise RunpodSyncVerificationError("JSON contains a non-finite number")


def _load_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodSyncVerificationError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RunpodSyncVerificationError("record is not canonical JSON data") from exc


def _stable_hash(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _authenticated_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("record_hash"), str):
        raise RunpodSyncVerificationError(f"{label} is not an authenticated object")
    unsigned = {key: item for key, item in value.items() if key != "record_hash"}
    if value["record_hash"] != _stable_hash(unsigned):
        raise RunpodSyncVerificationError(f"{label} record hash does not authenticate")
    return value


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunpodSyncVerificationError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise RunpodSyncVerificationError(f"{label} is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunpodSyncVerificationError(f"{label} lacks a timezone")
    return parsed.astimezone(UTC)


def _validate_manifest_time(
    manifest: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> tuple[datetime, datetime]:
    created = _parse_timestamp(manifest.get("created_at"), label="manifest created_at")
    expires = _parse_timestamp(manifest.get("expires_at"), label="manifest expires_at")
    validity = (expires - created).total_seconds()
    if validity <= 0 or validity > MAXIMUM_VALIDITY_SECONDS:
        raise RunpodSyncVerificationError("manifest validity window is not short and positive")
    now = observed_at.astimezone(UTC)
    if (created - now).total_seconds() > MAXIMUM_FUTURE_SKEW_SECONDS:
        raise RunpodSyncVerificationError("manifest was created in the future")
    if now > expires:
        raise RunpodSyncVerificationError("manifest has expired")
    return created, expires


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunpodSyncVerificationError("source checkout could not be inspected") from exc
    if completed.returncode != 0:
        raise RunpodSyncVerificationError("source checkout Git inspection failed")
    return completed.stdout.strip()


def _validate_source_checkout(
    source_checkout: str | Path,
    *,
    expected_commit: str,
    expected_repository_url: str,
) -> Path:
    supplied = Path(source_checkout)
    if not supplied.is_absolute() or supplied.is_symlink() or not supplied.is_dir():
        raise RunpodSyncVerificationError(
            "source checkout must be an exact absolute real directory"
        )
    resolved = supplied.resolve()
    if resolved != supplied:
        raise RunpodSyncVerificationError(
            "source checkout path contains a symlink or non-canonical component"
        )
    top_level = _git_output(resolved, "rev-parse", "--show-toplevel")
    if Path(top_level).resolve() != resolved or Path(top_level) != resolved:
        raise RunpodSyncVerificationError("source checkout is not the exact Git root")
    observed_commit = _git_output(resolved, "rev-parse", "HEAD")
    if (
        _SOURCE_COMMIT_RE.fullmatch(observed_commit) is None
        or observed_commit != expected_commit
    ):
        raise RunpodSyncVerificationError("source checkout commit disagrees with manifest")
    observed_origin = _git_output(resolved, "remote", "get-url", "origin")
    if (
        expected_repository_url != SOURCE_REPOSITORY_URL
        or observed_origin != SOURCE_REPOSITORY_URL
    ):
        raise RunpodSyncVerificationError(
            "source checkout origin disagrees with the canonical repository"
        )
    try:
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(resolved),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--",
                *_RUNNER_PATHS,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunpodSyncVerificationError("source checkout cleanliness could not be inspected") from exc
    if dirty.returncode != 0:
        raise RunpodSyncVerificationError("source checkout cleanliness inspection failed")
    if dirty.stdout:
        raise RunpodSyncVerificationError("tracked runner source checkout is dirty")
    return resolved


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RunpodSyncVerificationError("inventory contains an unsafe path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(ord(character) < 0x20 for character in value)
    ):
        raise RunpodSyncVerificationError("inventory contains an unsafe path")
    return value


def _allowed_inventory_path(path: str, *, phase: str, current_digest: str) -> bool:
    if path in _FIXED_SYNC_PATHS:
        return True
    if path == f".runpod/reservations/{phase}.json":
        return True
    parts = PurePosixPath(path).parts
    if (
        len(parts) == 4
        and parts[:2] == (".runpod", "sessions")
        and _RAW_HASH_RE.fullmatch(parts[2]) is not None
        and parts[2] != current_digest
        and parts[3] in _PRIOR_SESSION_FILENAMES
    ):
        return True
    return False


def _prior_inventory_groups(records: Mapping[str, Any]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for path in records:
        parts = PurePosixPath(path).parts
        if len(parts) != 4 or parts[:2] != (".runpod", "sessions"):
            continue
        groups.setdefault(parts[2], set()).add(parts[3])
    for names in groups.values():
        has_external = "external_stop_receipt.json" in names
        has_no_start = "no_start_receipt.json" in names
        if has_external and has_no_start:
            raise RunpodSyncVerificationError("prior session has conflicting stop evidence")
        if has_no_start:
            required = {"no_start_receipt.json", "settlement.json"}
            allowed = required | {"gpu_budget_bootstrap.json"}
        elif has_external:
            required = {"external_stop_receipt.json", "settlement.json"}
            allowed = required | {
                "gpu_budget_bootstrap.json",
                "runpod_watchdog.json",
                "settlement.v1.json",
            }
        else:
            required = {
                "gpu_budget_bootstrap.json",
                "runpod_watchdog.json",
                "settlement.json",
            }
            allowed = required
        if not required.issubset(names) or not names.issubset(allowed):
            raise RunpodSyncVerificationError(
                "prior session inventory is incomplete or unexpected"
            )
    return groups


def _secure_regular(root: Path, relative: str, *, label: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        try:
            details = current.lstat()
        except OSError as exc:
            raise RunpodSyncVerificationError(f"{label} parent is missing") from exc
        if not stat.S_ISDIR(details.st_mode) or current.is_symlink():
            raise RunpodSyncVerificationError(f"{label} parent is unsafe")
    try:
        details = target.lstat()
    except OSError as exc:
        raise RunpodSyncVerificationError(f"{label} is missing") from exc
    if (
        target.is_symlink()
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
    ):
        raise RunpodSyncVerificationError(f"{label} is not a safe regular file")
    return target


def _read_verified_file(
    root: Path,
    *,
    relative: str,
    expected_hash: str,
    expected_size: int,
) -> bytes:
    path = _secure_regular(root, relative, label="inventoried file")
    before = path.lstat()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunpodSyncVerificationError("inventoried file could not be read") from exc
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RunpodSyncVerificationError("inventoried file changed while read")
    observed_hash = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if len(raw) != expected_size or observed_hash != expected_hash:
        raise RunpodSyncVerificationError("inventoried file hash or size does not match")
    if any(marker in raw for marker in _SECRET_TEXT_MARKERS) or _CREDENTIAL_VALUE_RE.search(raw):
        raise RunpodSyncVerificationError("inventoried file contains credential material")
    return raw


def _read_stable_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    before = path.lstat()
    if before.st_size > maximum_bytes:
        raise RunpodSyncVerificationError(f"{label} is unexpectedly large")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunpodSyncVerificationError(f"{label} could not be read") from exc
    after = path.lstat()
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise RunpodSyncVerificationError(f"{label} changed while read")
    return raw


def _expected_private_nodes(inventory_paths: set[str]) -> tuple[set[str], set[str]]:
    files = {
        path.removeprefix(".runpod/")
        for path in inventory_paths
        if path.startswith(".runpod/")
    }
    files.add("selective_sync_manifest.json")
    directories: set[str] = set()
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _validate_exact_private_tree(root: Path, *, inventory_paths: set[str]) -> None:
    private = root / ".runpod"
    try:
        details = private.lstat()
    except OSError as exc:
        raise RunpodSyncVerificationError("private sync root is missing") from exc
    if private.is_symlink() or not stat.S_ISDIR(details.st_mode):
        raise RunpodSyncVerificationError("private sync root is unsafe")
    expected_files, expected_directories = _expected_private_nodes(inventory_paths)
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    pending = [private]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise RunpodSyncVerificationError("private sync tree is unreadable") from exc
        for entry in entries:
            relative = Path(entry.path).relative_to(private).as_posix()
            if entry.is_symlink():
                raise RunpodSyncVerificationError("private sync tree contains a symlink")
            if entry.is_dir(follow_symlinks=False):
                observed_directories.add(relative)
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                observed_files.add(relative)
            else:
                raise RunpodSyncVerificationError("private sync tree contains a special file")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RunpodSyncVerificationError("private sync tree differs from the exact inventory")


def _walk_json(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in _SECRET_FIELD_NAMES:
                raise RunpodSyncVerificationError("inventoried JSON contains a secret field")
            _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            _walk_json(item)


def _validate_lifecycle(
    raw: bytes,
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lifecycle = _authenticated_record(
        _load_json_bytes(raw, label="lifecycle"),
        label="lifecycle",
    )
    _walk_json(lifecycle)
    expected_keys = {
        "schema_version",
        "protocol_version",
        "operation",
        "updated_at",
        "immutable_spec",
        "current_authorization",
        "authorization_history",
        "pod",
        "record_hash",
    }
    if set(lifecycle) != expected_keys or lifecycle.get("schema_version") != 1:
        raise RunpodSyncVerificationError("lifecycle schema is unsupported")
    if lifecycle.get("protocol_version") != LIFECYCLE_PROTOCOL:
        raise RunpodSyncVerificationError("lifecycle protocol is unsupported")
    if lifecycle.get("record_hash") != manifest.get("lifecycle_record_hash"):
        raise RunpodSyncVerificationError("manifest and lifecycle hashes disagree")
    if lifecycle.get("operation") != "rearmed":
        raise RunpodSyncVerificationError("lifecycle is not a completed re-arm")
    authorization = lifecycle.get("current_authorization")
    pod = lifecycle.get("pod")
    if not isinstance(authorization, dict) or set(authorization) != _AUTHORIZATION_KEYS:
        raise RunpodSyncVerificationError("current lifecycle authorization is malformed")
    if not isinstance(pod, dict) or pod.get("status") != "RUNNING":
        raise RunpodSyncVerificationError("lifecycle does not bind a running Pod")
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise RunpodSyncVerificationError("running Pod identity is malformed")
    if manifest.get("running_pod_id_hash") != _stable_hash({"runpod_pod_id": pod_id}):
        raise RunpodSyncVerificationError("manifest running Pod hash does not bind lifecycle")
    if (
        authorization.get("phase") != manifest.get("phase")
        or authorization.get("session_hash") != manifest.get("session_hash")
        or authorization.get("reservation_record_hash")
        != manifest.get("reservation_record_hash")
    ):
        raise RunpodSyncVerificationError("manifest and lifecycle authorization disagree")
    hash_fields = (
        "approval_hash",
        "bindings_hash",
        "gpu_lock_hash",
        "immutable_spec_hash",
        "launch_spec_hash",
        "quote_hash",
        "reservation_id",
        "reservation_record_hash",
        "session_hash",
    )
    if any(
        not isinstance(authorization.get(field), str)
        or _HASH_RE.fullmatch(str(authorization[field])) is None
        for field in hash_fields
    ):
        raise RunpodSyncVerificationError("lifecycle authorization hash is malformed")
    acknowledged = authorization.get("acknowledged_existing_pod_id_hashes")
    if not isinstance(acknowledged, list) or not all(
        isinstance(item, str) for item in acknowledged
    ):
        raise RunpodSyncVerificationError("lifecycle Pod acknowledgement is malformed")
    for field in (
        "approved_phase_maximum_usd",
        "approved_runtime_hours",
        "live_hourly_total_usd",
    ):
        value = authorization.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise RunpodSyncVerificationError("lifecycle authorization cost is malformed")
    if authorization["immutable_spec_hash"] != _stable_hash(lifecycle["immutable_spec"]):
        raise RunpodSyncVerificationError("lifecycle immutable specification hash disagrees")
    return lifecycle, authorization


def _validate_reservation(
    raw: bytes,
    *,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    reservation = _authenticated_record(
        _load_json_bytes(raw, label="reservation"),
        label="reservation",
    )
    _walk_json(reservation)
    if (
        reservation.get("schema_version") != 1
        or reservation.get("protocol_version") != RESERVATION_PROTOCOL
        or set(reservation)
        != {"schema_version", "protocol_version", "record_hash", *_RESERVATION_FIELDS}
    ):
        raise RunpodSyncVerificationError("reservation schema or protocol is unsupported")
    if reservation.get("record_hash") != manifest.get("reservation_record_hash"):
        raise RunpodSyncVerificationError("manifest and reservation hashes disagree")
    if (
        reservation.get("phase") != manifest.get("phase")
        or reservation.get("session_hash") != manifest.get("session_hash")
        or reservation.get("reservation_id") != authorization.get("reservation_id")
    ):
        raise RunpodSyncVerificationError("reservation identity binding disagrees")
    expected_id = _stable_hash(
        {
            "protocol": RESERVATION_PROTOCOL,
            "phase": manifest["phase"],
            "session_hash": manifest["session_hash"],
        }
    )
    if reservation.get("reservation_id") != expected_id:
        raise RunpodSyncVerificationError("reservation identity is not deterministic")
    numeric: dict[str, float] = {}
    for field in _RESERVATION_FIELDS - {"reservation_id", "phase", "session_hash"}:
        value = reservation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RunpodSyncVerificationError("reservation accounting is malformed")
        numeric[field] = float(value)
    positive = {
        "approved_phase_maximum_usd",
        "approved_maximum_runtime_hours",
        "live_hourly_total_usd",
        "safety_margin_fraction",
        "global_gpu_hard_stop_usd",
        "safety_adjusted_gpu_ceiling_usd",
        "remaining_safe_gpu_before_phase_usd",
        "remaining_total_before_phase_usd",
        "maximum_safe_runtime_hours",
        "committed_gpu_after_reservation_usd",
        "committed_total_after_reservation_usd",
    }
    if any(numeric[field] <= 0 for field in positive) or any(
        value < 0 for field, value in numeric.items() if field not in positive
    ):
        raise RunpodSyncVerificationError("reservation accounting is not positive")
    if numeric["safety_margin_fraction"] >= 0.25:
        raise RunpodSyncVerificationError("reservation safety margin is invalid")
    safe_ceiling = math.floor(
        numeric["global_gpu_hard_stop_usd"]
        * (1 - numeric["safety_margin_fraction"])
        * 1_000_000
    ) / 1_000_000
    equations = (
        (
            numeric["safety_adjusted_gpu_ceiling_usd"],
            safe_ceiling,
        ),
        (
            numeric["prior_reserved_gpu_usd"],
            numeric["prior_committed_gpu_usd"] - numeric["prior_incurred_gpu_usd"],
        ),
        (
            numeric["remaining_safe_gpu_before_phase_usd"],
            safe_ceiling - numeric["prior_committed_gpu_usd"],
        ),
        (
            numeric["committed_gpu_after_reservation_usd"],
            numeric["prior_committed_gpu_usd"]
            + numeric["approved_phase_maximum_usd"],
        ),
        (
            numeric["committed_total_after_reservation_usd"],
            numeric["prior_committed_total_usd"]
            + numeric["approved_phase_maximum_usd"],
        ),
    )
    if any(abs(left - right) > 1e-6 for left, right in equations):
        raise RunpodSyncVerificationError("reservation accounting is inconsistent")
    total_hard_stop = (
        numeric["prior_committed_total_usd"]
        + numeric["remaining_total_before_phase_usd"]
    )
    if (
        numeric["committed_gpu_after_reservation_usd"] > safe_ceiling + 1e-6
        or numeric["committed_total_after_reservation_usd"] > total_hard_stop + 1e-6
    ):
        raise RunpodSyncVerificationError("reservation exceeds a hard stop")
    safe_spend = min(
        numeric["approved_phase_maximum_usd"],
        numeric["remaining_safe_gpu_before_phase_usd"],
        numeric["remaining_total_before_phase_usd"],
    )
    expected_runtime = min(
        numeric["approved_maximum_runtime_hours"],
        safe_spend / numeric["live_hourly_total_usd"],
    )
    if abs(numeric["maximum_safe_runtime_hours"] - expected_runtime) > 1e-9:
        raise RunpodSyncVerificationError("reservation safe runtime is inconsistent")
    authorization_fields = {
        "approved_phase_maximum_usd": "approved_phase_maximum_usd",
        "approved_maximum_runtime_hours": "approved_runtime_hours",
        "live_hourly_total_usd": "live_hourly_total_usd",
    }
    for reservation_field, authorization_field in authorization_fields.items():
        if (
            abs(
                float(authorization[authorization_field])
                - numeric[reservation_field]
            )
            > 1e-9
        ):
            raise RunpodSyncVerificationError(
                "lifecycle and reservation authorization amounts disagree"
            )
    return reservation


def _canonical_yaml_scalar(raw: str, *, field: str) -> str:
    if not raw or raw != raw.strip() or any(ord(character) < 0x20 for character in raw):
        raise RunpodSyncVerificationError(f"ledger {field} scalar is malformed")
    if raw.startswith("'"):
        if len(raw) < 2 or not raw.endswith("'"):
            raise RunpodSyncVerificationError(f"ledger {field} quote is malformed")
        inner = raw[1:-1]
        result: list[str] = []
        index = 0
        while index < len(inner):
            if inner[index] != "'":
                result.append(inner[index])
                index += 1
            elif index + 1 < len(inner) and inner[index + 1] == "'":
                result.append("'")
                index += 2
            else:
                raise RunpodSyncVerificationError(f"ledger {field} quote is malformed")
        return "".join(result)
    if raw.startswith('"'):
        value = _load_json_bytes(raw.encode("utf-8"), label=f"ledger {field}")
        if not isinstance(value, str):
            raise RunpodSyncVerificationError(f"ledger {field} must be text")
        return value
    if (
        raw[0] in "!&*{}[],#|>@`"
        or raw.startswith(("- ", "? ", ": "))
        or " #" in raw
        or ": " in raw
        or raw.endswith(":")
        or raw.casefold() in _YAML_IMPLICIT_SCALARS
        or _CANONICAL_NUMBER_RE.fullmatch(raw) is not None
        or field == "occurred_at"
    ):
        raise RunpodSyncVerificationError(f"ledger {field} plain scalar is unsafe")
    return raw


def _canonical_number(raw: str, *, field: str) -> float:
    if _CANONICAL_NUMBER_RE.fullmatch(raw) is None:
        raise RunpodSyncVerificationError(f"ledger {field} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise RunpodSyncVerificationError(f"ledger {field} must be finite")
    return value


def _load_canonical_ledger(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunpodSyncVerificationError("ledger is not UTF-8") from exc
    if "\r" in text or "\t" in text or not text.endswith("\n"):
        raise RunpodSyncVerificationError("ledger encoding is non-canonical")
    lines = text.splitlines()
    if len(lines) < 7 or lines[:3] != [
        "schema_version: 1",
        "currency: USD",
        "hard_stops:",
    ]:
        raise RunpodSyncVerificationError("ledger header is non-canonical")
    hard_stops: dict[str, float] = {}
    for offset, name in enumerate(("gpu", "api", "total"), start=3):
        prefix = f"  {name}: "
        if not lines[offset].startswith(prefix):
            raise RunpodSyncVerificationError("ledger hard stops are non-canonical")
        hard_stops[name] = _canonical_number(
            lines[offset][len(prefix) :],
            field=f"hard_stops.{name}",
        )
    if lines[6] == "entries: []":
        if len(lines) != 7:
            raise RunpodSyncVerificationError("empty ledger has trailing content")
        return {"hard_stops": hard_stops, "entries": []}
    if lines[6] != "entries:":
        raise RunpodSyncVerificationError("ledger entries header is non-canonical")
    entries: list[dict[str, Any]] = []
    index = 7
    while index < len(lines):
        if not lines[index].startswith("- kind: "):
            raise RunpodSyncVerificationError("ledger entry boundary is non-canonical")
        entry: dict[str, Any] = {
            "kind": _canonical_yaml_scalar(lines[index][8:], field="kind")
        }
        index += 1
        for field in ("amount_usd", "description", "status", "occurred_at"):
            if index >= len(lines) or not lines[index].startswith(f"  {field}: "):
                raise RunpodSyncVerificationError("ledger entry is truncated or reordered")
            raw_value = lines[index][len(f"  {field}: ") :]
            entry[field] = (
                _canonical_number(raw_value, field=field)
                if field == "amount_usd"
                else _canonical_yaml_scalar(raw_value, field=field)
            )
            index += 1
        if index < len(lines) and lines[index].startswith("  entry_id: "):
            entry["entry_id"] = _canonical_yaml_scalar(
                lines[index][12:],
                field="entry_id",
            )
            index += 1
        entries.append(entry)
    return {"hard_stops": hard_stops, "entries": entries}


def _validate_ledger(
    raw: bytes,
    *,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
) -> None:
    ledger = _load_canonical_ledger(raw)
    reservation_id = authorization["reservation_id"]
    matches = [
        entry for entry in ledger["entries"] if entry.get("entry_id") == reservation_id
    ]
    if len(matches) != 1:
        raise RunpodSyncVerificationError("ledger lacks one unique current reservation")
    entry = matches[0]
    expected_description = (
        f"GPU phase {manifest['phase']} session {manifest['session_hash']}"
    )
    amount = float(reservation["approved_phase_maximum_usd"])
    if (
        entry.get("kind") != "gpu"
        or entry.get("status") != "estimated"
        or entry.get("description") != expected_description
        or abs(float(entry.get("amount_usd", -1)) - amount) > 1e-6
    ):
        raise RunpodSyncVerificationError("ledger current reservation is not exact and active")
    if abs(float(ledger["hard_stops"]["gpu"]) - float(reservation["global_gpu_hard_stop_usd"])) > 1e-6:
        raise RunpodSyncVerificationError("ledger and reservation GPU hard stops disagree")
    inferred_total_hard_stop = float(reservation["prior_committed_total_usd"]) + float(
        reservation["remaining_total_before_phase_usd"]
    )
    if abs(float(ledger["hard_stops"]["total"]) - inferred_total_hard_stop) > 1e-6:
        raise RunpodSyncVerificationError("ledger and reservation total hard stops disagree")
    committed_gpu = sum(
        float(item["amount_usd"])
        for item in ledger["entries"]
        if item.get("kind") == "gpu" and item.get("status") in {"estimated", "incurred"}
    )
    committed_total = sum(
        float(item["amount_usd"])
        for item in ledger["entries"]
        if item.get("status") in {"estimated", "incurred"}
    )
    if (
        abs(committed_gpu - float(reservation["committed_gpu_after_reservation_usd"]))
        > 1e-6
        or abs(
            committed_total - float(reservation["committed_total_after_reservation_usd"])
        )
        > 1e-6
    ):
        raise RunpodSyncVerificationError("ledger committed totals disagree with reservation")


def verify_selective_sync(
    *,
    project_root: str | Path,
    source_checkout: str | Path,
    manifest_path: str | Path | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Fail closed unless the remote selective sync is exact and live.

    The returned summary is deliberately secret-safe and contains only hashes,
    counts, phase, and expiry metadata.  In particular it never returns the raw
    provider Pod identifier found in the lifecycle artifact.
    """

    supplied_root = Path(project_root).absolute()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise RunpodSyncVerificationError("project root is missing or unsafe")
    root = supplied_root.resolve()
    expected_manifest = root / MANIFEST_RELATIVE_PATH
    supplied_manifest = (
        Path(manifest_path).absolute() if manifest_path is not None else expected_manifest
    )
    if supplied_manifest != expected_manifest:
        raise RunpodSyncVerificationError("sync manifest path is not canonical")
    manifest_path_verified = _secure_regular(
        root,
        MANIFEST_RELATIVE_PATH,
        label="selective sync manifest",
    )
    manifest_raw = _read_stable_file(
        manifest_path_verified,
        maximum_bytes=MAXIMUM_MANIFEST_BYTES,
        label="selective sync manifest",
    )
    manifest = _authenticated_record(
        _load_json_bytes(manifest_raw, label="selective sync manifest"),
        label="selective sync manifest",
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise RunpodSyncVerificationError("selective sync manifest schema is unexpected")
    if manifest.get("schema_version") != 1 or manifest.get("protocol_version") != SYNC_PROTOCOL:
        raise RunpodSyncVerificationError("selective sync manifest protocol is unsupported")
    phase = manifest.get("phase")
    session_hash = manifest.get("session_hash")
    source_commit = manifest.get("source_commit")
    source_repository_url = manifest.get("source_repository_url")
    if not isinstance(phase, str) or _PHASE_RE.fullmatch(phase) is None:
        raise RunpodSyncVerificationError("manifest phase is malformed")
    if not isinstance(session_hash, str) or _HASH_RE.fullmatch(session_hash) is None:
        raise RunpodSyncVerificationError("manifest session hash is malformed")
    if not isinstance(source_commit, str) or _SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise RunpodSyncVerificationError("manifest source commit is malformed")
    if source_repository_url != SOURCE_REPOSITORY_URL:
        raise RunpodSyncVerificationError("manifest source repository is not canonical")
    for field in (
        "lifecycle_record_hash",
        "reservation_record_hash",
        "running_pod_id_hash",
        "record_hash",
    ):
        if not isinstance(manifest.get(field), str) or _HASH_RE.fullmatch(
            str(manifest[field])
        ) is None:
            raise RunpodSyncVerificationError("manifest contains a malformed hash")
    if manifest.get("current_host_session_excluded") is not True:
        raise RunpodSyncVerificationError("manifest does not exclude the current host session")
    guard = manifest.get("current_host_guard")
    if not isinstance(guard, dict) or set(guard) != _HOST_GUARD_KEYS or any(
        not isinstance(value, str) or _HASH_RE.fullmatch(value) is None
        for value in guard.values()
    ):
        raise RunpodSyncVerificationError("manifest host guard binding is malformed")
    _walk_json(manifest)
    _, expires = _validate_manifest_time(
        manifest,
        observed_at=observed_at or datetime.now(UTC),
    )
    _validate_source_checkout(
        source_checkout,
        expected_commit=source_commit,
        expected_repository_url=source_repository_url,
    )

    inventory = manifest.get("files")
    if (
        not isinstance(inventory, list)
        or not inventory
        or len(inventory) > MAXIMUM_INVENTORY_FILES
    ):
        raise RunpodSyncVerificationError("manifest file inventory is missing")
    current_digest = session_hash.removeprefix("sha256:")
    records: dict[str, tuple[str, int]] = {}
    file_bytes: dict[str, bytes] = {}
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise RunpodSyncVerificationError("inventory file record is malformed")
        path = _safe_relative_path(item.get("path"))
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if path in records:
            raise RunpodSyncVerificationError("inventory contains a duplicate path")
        if (
            not isinstance(digest, str)
            or _HASH_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > MAXIMUM_INVENTORY_FILE_BYTES
        ):
            raise RunpodSyncVerificationError("inventory file identity is malformed")
        if not _allowed_inventory_path(path, phase=phase, current_digest=current_digest):
            raise RunpodSyncVerificationError("inventory contains an unapproved path")
        records[path] = (digest, size)
    mandatory = _FIXED_SYNC_PATHS | {f".runpod/reservations/{phase}.json"}
    if not mandatory.issubset(records):
        raise RunpodSyncVerificationError("inventory omits a mandatory control file")
    if any(
        path.startswith(f".runpod/sessions/{current_digest}/") for path in records
    ):
        raise RunpodSyncVerificationError("inventory includes the current host session")
    _prior_inventory_groups(records)
    current_session = root / ".runpod" / "sessions" / current_digest
    if os.path.lexists(current_session):
        raise RunpodSyncVerificationError("current host session exists on the remote")
    _validate_exact_private_tree(root, inventory_paths=set(records))

    for path, (digest, size) in records.items():
        file_bytes[path] = _read_verified_file(
            root,
            relative=path,
            expected_hash=digest,
            expected_size=size,
        )
        if path.endswith(".json"):
            _walk_json(_load_json_bytes(file_bytes[path], label="inventoried JSON"))
    lifecycle_raw = file_bytes[".runpod/pod_lifecycle.json"]
    _, authorization = _validate_lifecycle(lifecycle_raw, manifest=manifest)
    reservation_relative = f".runpod/reservations/{phase}.json"
    reservation = _validate_reservation(
        file_bytes[reservation_relative],
        manifest=manifest,
        authorization=authorization,
    )
    _validate_ledger(
        file_bytes["data/manifests/cost_ledger.yaml"],
        manifest=manifest,
        authorization=authorization,
        reservation=reservation,
    )
    if os.path.lexists(current_session):
        raise RunpodSyncVerificationError("current host session appeared during verification")
    _validate_exact_private_tree(root, inventory_paths=set(records))
    final_manifest = _read_stable_file(
        _secure_regular(
            root,
            MANIFEST_RELATIVE_PATH,
            label="selective sync manifest",
        ),
        maximum_bytes=MAXIMUM_MANIFEST_BYTES,
        label="selective sync manifest",
    )
    if final_manifest != manifest_raw:
        raise RunpodSyncVerificationError("selective sync manifest changed during verification")
    return {
        "schema_version": 1,
        "protocol_version": SYNC_PROTOCOL,
        "phase": phase,
        "session_hash": session_hash,
        "running_pod_id_hash": manifest["running_pod_id_hash"],
        "manifest_record_hash": manifest["record_hash"],
        "source_commit": source_commit,
        "file_count": len(records),
        "expires_at": expires.isoformat().replace("+00:00", "Z"),
        "current_host_session_absent": True,
        "passed": True,
    }


__all__ = [
    "MANIFEST_RELATIVE_PATH",
    "MAXIMUM_FUTURE_SKEW_SECONDS",
    "MAXIMUM_INVENTORY_FILES",
    "MAXIMUM_INVENTORY_FILE_BYTES",
    "MAXIMUM_MANIFEST_BYTES",
    "MAXIMUM_VALIDITY_SECONDS",
    "SYNC_PROTOCOL",
    "RunpodSyncVerificationError",
    "verify_selective_sync",
]
