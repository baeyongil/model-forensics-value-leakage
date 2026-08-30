"""Provider-free, crash-resumable rotation of private paid-run controls.

Rotation is intentionally capability-free: this module imports no provider
client and performs no network operation.  It will archive paid controls only
while the canonical cost ledger has no outstanding GPU estimate, every local
RunPod session has authenticated schema-v2 closure evidence, and the canonical
lifecycle says exactly ``stopped`` / ``EXITED``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from model_forensics.approval import PaidRunApproval, approval_content_hash
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.execution_bindings import (
    ApiRouteQuoteLock,
    GpuQuoteLock,
    api_route_quote_lock_content_hash,
    gpu_quote_lock_content_hash,
)
from model_forensics.io import stable_hash
from model_forensics.runpod_lifecycle_state import (
    RunpodLifecycleStateError,
    load_lifecycle_state,
)
from model_forensics.runpod_sessions import (
    RunpodSessionError,
    validate_completed_runpod_sessions,
)

ROTATION_PROTOCOL = "private-paid-bundle-rotation-v1"
COMPLETION_PROTOCOL = "private-paid-bundle-rotation-completion-v1"
ARCHIVE_RELATIVE_ROOT = PurePosixPath(".runpod/archive/paid-bundles")
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "rotation_complete.json"
_BUNDLE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,95}\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAXIMUM_CONTROL_BYTES = 8 * 1024 * 1024
_MAXIMUM_MANIFEST_BYTES = 256 * 1024
_REQUIRED_CONTROL_PATHS = (
    PurePosixPath(".runpod/gpu_quote_lock.json"),
    PurePosixPath(".runpod/api_route_quote_lock.json"),
)
_OPTIONAL_CONTROL_PATHS = (
    PurePosixPath(".runpod/paid_run_approval.json"),
    PurePosixPath(".runpod/specs/gpu_quote_spec.json"),
    PurePosixPath(".runpod/specs/api_route_quote_spec.json"),
)
_CONTROL_PATHS = _REQUIRED_CONTROL_PATHS + _OPTIONAL_CONTROL_PATHS


class PaidBundleRotationError(RuntimeError):
    """The private paid bundle cannot be rotated safely."""


@dataclass(frozen=True, slots=True)
class _ControlSnapshot:
    source_path: PurePosixPath
    archive_path: PurePosixPath
    sha256: str
    size_bytes: int
    content: bytes

    def manifest_record(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path.as_posix(),
            "archive_path": self.archive_path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


class _LedgerSnapshot:
    """Read-only ledger view used by the existing session validator."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self._document = dict(document)

    def document(self) -> dict[str, Any]:
        return self._document


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PaidBundleRotationError(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _decode_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaidBundleRotationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PaidBundleRotationError(f"{label} must be a JSON object")
    return value


def _file_identity(details: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_nlink,
    )


def _require_owned_directory(path: Path, *, label: str) -> Path:
    try:
        details = path.lstat()
    except OSError as exc:
        raise PaidBundleRotationError(f"{label} is missing or unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
    ):
        raise PaidBundleRotationError(f"{label} must be an owned real directory")
    return path


def _require_no_symlink_components(*, root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PaidBundleRotationError(f"{label} escapes the project root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise PaidBundleRotationError(f"{label} path contains a symlink")


def _mkdir_private(path: Path, *, parent_label: str) -> None:
    _require_owned_directory(path.parent, label=parent_label)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        _require_owned_directory(path, label=str(path))
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_owned_regular(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = _MAXIMUM_CONTROL_BYTES,
    allowed_link_counts: frozenset[int] = frozenset({1}),
    allow_empty: bool = False,
) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise PaidBundleRotationError(f"{label} is missing or unreadable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink not in allowed_link_counts
        or (before.st_size <= 0 and not allow_empty)
        or before.st_size > maximum_bytes
    ):
        raise PaidBundleRotationError(
            f"{label} must be an owned regular file with an allowed link count"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PaidBundleRotationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise PaidBundleRotationError(f"{label} changed before authenticated read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise PaidBundleRotationError(f"{label} exceeds its safe size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise PaidBundleRotationError(f"{label} changed during authenticated read") from exc
    content = b"".join(chunks)
    if (
        _file_identity(after) != _file_identity(opened)
        or _file_identity(current) != _file_identity(opened)
        or len(content) != opened.st_size
    ):
        raise PaidBundleRotationError(f"{label} changed during authenticated read")
    return content


def _optional_owned_regular(path: Path, *, label: str) -> bytes | None:
    if not os.path.lexists(path):
        return None
    return _read_owned_regular(path, label=label)


def _validate_control_payload(relative: PurePosixPath, content: bytes) -> None:
    raw = _decode_json_object(content, label=relative.as_posix())
    try:
        if relative == PurePosixPath(".runpod/gpu_quote_lock.json"):
            if raw.get("content_hash") != gpu_quote_lock_content_hash(raw):
                raise PaidBundleRotationError("GPU quote lock content hash mismatch")
            GpuQuoteLock.model_validate(raw)
        elif relative == PurePosixPath(".runpod/api_route_quote_lock.json"):
            if raw.get("content_hash") != api_route_quote_lock_content_hash(raw):
                raise PaidBundleRotationError("API route quote lock content hash mismatch")
            ApiRouteQuoteLock.model_validate(raw)
        elif relative == PurePosixPath(".runpod/paid_run_approval.json"):
            if raw.get("content_hash") != approval_content_hash(raw):
                raise PaidBundleRotationError("paid-run approval content hash mismatch")
            PaidRunApproval.model_validate(raw)
        elif relative == PurePosixPath(".runpod/specs/gpu_quote_spec.json"):
            if "content_hash" in raw:
                raise PaidBundleRotationError("GPU quote spec must omit content_hash")
            GpuQuoteLock.model_validate(
                {**raw, "content_hash": gpu_quote_lock_content_hash(raw)}
            )
        elif relative == PurePosixPath(".runpod/specs/api_route_quote_spec.json"):
            if "content_hash" in raw:
                raise PaidBundleRotationError("API route quote spec must omit content_hash")
            ApiRouteQuoteLock.model_validate(
                {**raw, "content_hash": api_route_quote_lock_content_hash(raw)}
            )
        else:  # pragma: no cover - caller inventory is closed above
            raise PaidBundleRotationError("unexpected paid-control path")
    except ValidationError as exc:
        raise PaidBundleRotationError(
            f"{relative.as_posix()} has an invalid strict schema"
        ) from exc


def _snapshot_controls(root: Path, *, require_required: bool) -> list[_ControlSnapshot]:
    specs = root / ".runpod" / "specs"
    if os.path.lexists(specs):
        _require_owned_directory(specs, label="private quote-spec directory")
    snapshots: list[_ControlSnapshot] = []
    for relative in _CONTROL_PATHS:
        content = _optional_owned_regular(
            root / relative.as_posix(),
            label=relative.as_posix(),
        )
        if content is None:
            if require_required and relative in _REQUIRED_CONTROL_PATHS:
                raise PaidBundleRotationError(
                    f"required canonical paid control is missing: {relative.as_posix()}"
                )
            continue
        _validate_control_payload(relative, content)
        archive_relative = PurePosixPath("files") / relative.relative_to(".runpod")
        snapshots.append(
            _ControlSnapshot(
                source_path=relative,
                archive_path=archive_relative,
                sha256="sha256:" + hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                content=content,
            )
        )
    snapshots.sort(key=lambda item: item.source_path.as_posix())
    return snapshots


def _bundle_content_hash(records: list[dict[str, Any]]) -> str:
    return stable_hash({"protocol_version": ROTATION_PROTOCOL, "files": records})


def _validate_bundle_id(bundle_id: str) -> str:
    if _BUNDLE_ID_RE.fullmatch(bundle_id) is None or bundle_id in {".", ".."}:
        raise PaidBundleRotationError(
            "bundle ID must be 3-96 lowercase letters, digits, dots, underscores, or hyphens"
        )
    return bundle_id


def _manifest_payload(*, bundle_id: str, snapshots: list[_ControlSnapshot]) -> dict[str, Any]:
    records = [item.manifest_record() for item in snapshots]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": ROTATION_PROTOCOL,
        "bundle_id": bundle_id,
        "bundle_content_hash": _bundle_content_hash(records),
        "files": records,
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_manifest(payload: Any, *, expected_bundle_id: str) -> list[dict[str, Any]]:
    expected_keys = {
        "schema_version",
        "protocol_version",
        "bundle_id",
        "bundle_content_hash",
        "files",
        "record_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise PaidBundleRotationError("rotation manifest has an unexpected schema")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_version") != ROTATION_PROTOCOL
        or payload.get("bundle_id") != expected_bundle_id
        or not isinstance(payload.get("record_hash"), str)
        or _HASH_RE.fullmatch(str(payload["record_hash"])) is None
        or payload.get("record_hash") != stable_hash(unsigned)
    ):
        raise PaidBundleRotationError("rotation manifest authentication failed")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PaidBundleRotationError("rotation manifest has no file inventory")
    records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_archives: set[str] = set()
    allowed = {item.as_posix() for item in _CONTROL_PATHS}
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {
            "source_path",
            "archive_path",
            "sha256",
            "size_bytes",
        }:
            raise PaidBundleRotationError("rotation manifest file record is malformed")
        source = raw.get("source_path")
        archive = raw.get("archive_path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            not isinstance(source, str)
            or source not in allowed
            or not isinstance(archive, str)
            or archive
            != (PurePosixPath("files") / PurePosixPath(source).relative_to(".runpod")).as_posix()
            or not isinstance(digest, str)
            or _HASH_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > _MAXIMUM_CONTROL_BYTES
            or source in seen_sources
            or archive in seen_archives
        ):
            raise PaidBundleRotationError("rotation manifest file record is unsafe")
        seen_sources.add(source)
        seen_archives.add(archive)
        records.append(dict(raw))
    if [item["source_path"] for item in records] != sorted(seen_sources):
        raise PaidBundleRotationError("rotation manifest inventory order is not canonical")
    if not {item.as_posix() for item in _REQUIRED_CONTROL_PATHS}.issubset(seen_sources):
        raise PaidBundleRotationError("rotation manifest omits a required quote lock")
    if payload.get("bundle_content_hash") != _bundle_content_hash(records):
        raise PaidBundleRotationError("rotation manifest bundle hash mismatch")
    return records


def _completion_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": COMPLETION_PROTOCOL,
        "bundle_id": manifest["bundle_id"],
        "manifest_record_hash": manifest["record_hash"],
        "bundle_content_hash": manifest["bundle_content_hash"],
        "file_count": len(manifest["files"]),
        "status": "complete",
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _validate_completion(payload: Any, *, manifest: Mapping[str, Any]) -> None:
    expected = _completion_payload(manifest)
    if payload != expected:
        raise PaidBundleRotationError("rotation completion marker authentication failed")


def _pending_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.pending")


def _install_exact_file(destination: Path, content: bytes) -> None:
    """Install exact bytes without replacement and recover the bounded link step."""

    _require_owned_directory(destination.parent, label=f"parent of {destination}")
    pending = _pending_path(destination)
    destination_exists = os.path.lexists(destination)
    pending_exists = os.path.lexists(pending)

    if not destination_exists and not pending_exists:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(pending, flags, 0o600)
        except OSError as exc:
            raise PaidBundleRotationError(f"cannot exclusively stage archive file: {destination}") from exc
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS boundary
                    raise PaidBundleRotationError("short write while staging archive file")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(destination.parent)
        pending_exists = True

    if pending_exists:
        staged = _read_owned_regular(
            pending,
            label=f"pending archive file {pending}",
            allowed_link_counts=frozenset({1, 2}),
        )
        if staged != content:
            raise PaidBundleRotationError(f"pending archive file has different bytes: {pending}")
    if destination_exists:
        archived = _read_owned_regular(
            destination,
            label=f"archive file {destination}",
            allowed_link_counts=frozenset({1, 2}),
        )
        if archived != content:
            raise PaidBundleRotationError(f"archive destination already contains different bytes: {destination}")

    if pending_exists and destination_exists:
        destination_stat = destination.lstat()
        pending_stat = pending.lstat()
        if (
            destination_stat.st_dev != pending_stat.st_dev
            or destination_stat.st_ino != pending_stat.st_ino
            or destination_stat.st_nlink != 2
            or pending_stat.st_nlink != 2
        ):
            raise PaidBundleRotationError("archive link recovery state is unsafe")
    elif pending_exists and pending.lstat().st_nlink != 1:
        raise PaidBundleRotationError("unclaimed pending archive file has an unsafe link count")
    elif destination_exists and destination.lstat().st_nlink != 1:
        raise PaidBundleRotationError("archive destination has an unsafe link count")

    if not destination_exists:
        try:
            os.link(pending, destination, follow_symlinks=False)
        except FileExistsError:
            archived = _read_owned_regular(destination, label=f"archive file {destination}")
            if archived != content:
                raise PaidBundleRotationError(
                    f"archive destination was concurrently claimed: {destination}"
                ) from None
        except OSError as exc:
            raise PaidBundleRotationError(f"cannot claim archive destination: {destination}") from exc
        _fsync_directory(destination.parent)

    if os.path.lexists(pending):
        try:
            destination_stat = destination.lstat()
            pending_stat = pending.lstat()
        except OSError as exc:
            raise PaidBundleRotationError("archive link state changed during installation") from exc
        if (
            destination_stat.st_dev != pending_stat.st_dev
            or destination_stat.st_ino != pending_stat.st_ino
            or destination_stat.st_uid != os.getuid()
            or not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_nlink != 2
            or pending_stat.st_nlink != 2
        ):
            raise PaidBundleRotationError("archive link recovery state is unsafe")
        pending.unlink()
        _fsync_directory(destination.parent)

    archived = _read_owned_regular(destination, label=f"archive file {destination}")
    if archived != content:
        raise PaidBundleRotationError(f"archive file verification failed: {destination}")


def _ensure_archive_parent(root: Path) -> Path:
    runpod = _require_owned_directory(root / ".runpod", label="canonical .runpod directory")
    archive = runpod / "archive"
    if not os.path.lexists(archive):
        _mkdir_private(archive, parent_label="canonical .runpod directory")
    else:
        _require_owned_directory(archive, label="private archive directory")
    bundles = archive / "paid-bundles"
    if not os.path.lexists(bundles):
        _mkdir_private(bundles, parent_label="private archive directory")
    else:
        _require_owned_directory(bundles, label="paid-bundle archive root")
    return bundles


def _ensure_record_directories(bundle: Path, archive_relative: PurePosixPath) -> None:
    current = bundle
    for part in archive_relative.parts[:-1]:
        current = current / part
        if not os.path.lexists(current):
            _mkdir_private(current, parent_label=f"archive parent {current.parent}")
        else:
            _require_owned_directory(current, label=f"archive directory {current}")


def _validate_archive_inventory(
    *,
    bundle: Path,
    records: list[dict[str, Any]],
    allow_pending: bool,
    require_complete: bool,
) -> None:
    expected_files = {
        PurePosixPath(MANIFEST_FILENAME),
        *(PurePosixPath(str(item["archive_path"])) for item in records),
    }
    required_files = {PurePosixPath(MANIFEST_FILENAME)}
    if require_complete:
        expected_files.add(PurePosixPath(COMPLETION_FILENAME))
        required_files = set(expected_files)
    optional_completion = PurePosixPath(COMPLETION_FILENAME)
    expected_directories: set[PurePosixPath] = set()
    for path in [*expected_files, optional_completion]:
        parent = path.parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent)
            parent = parent.parent
    seen_files: set[PurePosixPath] = set()
    stack = [bundle]
    while stack:
        directory = stack.pop()
        _require_owned_directory(directory, label=f"archive directory {directory}")
        for entry in os.scandir(directory):
            path = Path(entry.path)
            try:
                relative = PurePosixPath(path.relative_to(bundle).as_posix())
                details = path.lstat()
            except (OSError, ValueError) as exc:
                raise PaidBundleRotationError("archive inventory changed during validation") from exc
            if stat.S_ISLNK(details.st_mode) or details.st_uid != os.getuid():
                raise PaidBundleRotationError("archive inventory contains a linked or foreign entry")
            if stat.S_ISDIR(details.st_mode):
                if relative not in expected_directories:
                    raise PaidBundleRotationError(
                        f"archive inventory contains an unexpected directory: {relative}"
                    )
                stack.append(path)
                continue
            if not stat.S_ISREG(details.st_mode):
                raise PaidBundleRotationError("archive inventory contains a non-regular entry")
            allowed_links = {1, 2} if allow_pending else {1}
            if details.st_nlink not in allowed_links:
                raise PaidBundleRotationError("archive inventory contains an unsafe hardlink")
            pending_target: PurePosixPath | None = None
            if path.name.startswith(".") and path.name.endswith(".pending"):
                pending_target = relative.with_name(path.name[1:-len(".pending")])
            allowed_final = relative in expected_files or relative == optional_completion
            allowed_pending = (
                allow_pending
                and pending_target is not None
                and (pending_target in expected_files or pending_target == optional_completion)
            )
            if not allowed_final and not allowed_pending:
                raise PaidBundleRotationError(
                    f"archive inventory contains an unexpected file: {relative}"
                )
            if relative in seen_files:
                raise PaidBundleRotationError("archive inventory contains a duplicate path")
            seen_files.add(relative)
    if not required_files.issubset(seen_files):
        missing = sorted(path.as_posix() for path in required_files - seen_files)
        raise PaidBundleRotationError(
            f"archive inventory is missing required completed files: {missing}"
        )
    if not require_complete and optional_completion in seen_files:
        # A completion file may legitimately be present during replay; its
        # authenticated content is checked immediately after this inventory.
        return


def _load_manifest(path: Path, *, bundle_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    content = _read_owned_regular(
        path,
        label="rotation manifest",
        maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
    )
    payload = _decode_json_object(content, label="rotation manifest")
    records = _validate_manifest(payload, expected_bundle_id=bundle_id)
    return payload, records


def _normalize_and_load_manifest(
    bundle: Path,
    *,
    bundle_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Finish a bounded pending/link state before trusting a manifest."""

    destination = bundle / MANIFEST_FILENAME
    pending = _pending_path(destination)
    candidates = [path for path in (destination, pending) if os.path.lexists(path)]
    if not candidates:
        raise PaidBundleRotationError("existing archive directory has no rotation manifest")
    content = _read_owned_regular(
        candidates[0],
        label="pending or installed rotation manifest",
        maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        allowed_link_counts=frozenset({1, 2}),
    )
    payload = _decode_json_object(content, label="rotation manifest")
    _validate_manifest(payload, expected_bundle_id=bundle_id)
    canonical = _canonical_json_bytes(payload)
    if content != canonical:
        raise PaidBundleRotationError("rotation manifest encoding is not canonical")
    _install_exact_file(destination, canonical)
    return _load_manifest(destination, bundle_id=bundle_id)


def _initialize_or_load_manifest(
    *,
    bundle: Path,
    bundle_id: str,
    expected: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recover empty/pending archive creation without accepting other content."""

    manifest_path = bundle / MANIFEST_FILENAME
    manifest_pending = _pending_path(manifest_path)
    allowed_unfinished = {MANIFEST_FILENAME, manifest_pending.name}
    unexpected = {item.name for item in bundle.iterdir()} - allowed_unfinished
    if not os.path.lexists(manifest_path) and not os.path.lexists(manifest_pending):
        if unexpected or expected is None:
            raise PaidBundleRotationError(
                "existing archive directory lacks a recoverable manifest"
            )
        _install_exact_file(manifest_path, _canonical_json_bytes(expected))
    elif unexpected and expected is not None:
        # Once a valid manifest exists, later transaction files are expected;
        # let the authenticated manifest and closed inventory validate them.
        pass
    manifest, records = _normalize_and_load_manifest(bundle, bundle_id=bundle_id)
    if expected is not None and manifest != dict(expected):
        raise PaidBundleRotationError(
            "archive bundle ID is already bound to different control bytes"
        )
    return manifest, records


def _validate_lifecycle(root: Path) -> dict[str, Any]:
    lifecycle_path = root / ".runpod" / "pod_lifecycle.json"
    try:
        lifecycle = load_lifecycle_state(lifecycle_path)
    except RunpodLifecycleStateError as exc:
        raise PaidBundleRotationError("canonical RunPod lifecycle is missing or invalid") from exc
    pod = lifecycle.get("pod")
    if (
        lifecycle.get("operation") != "stopped"
        or not isinstance(pod, Mapping)
        or pod.get("status") != "EXITED"
    ):
        raise PaidBundleRotationError(
            "paid-bundle rotation requires lifecycle operation stopped and Pod status EXITED"
        )
    return lifecycle


def _validate_safety_state(root: Path, ledger_document: Mapping[str, Any]) -> None:
    entries = ledger_document.get("entries")
    if not isinstance(entries, list):  # CostLedger validates this first
        raise PaidBundleRotationError("canonical cost ledger is malformed")
    outstanding = [
        item
        for item in entries
        if isinstance(item, Mapping)
        and item.get("kind") == "gpu"
        and item.get("status") == "estimated"
    ]
    if outstanding:
        raise PaidBundleRotationError(
            "refusing rotation while an estimated GPU reservation is outstanding"
        )
    lifecycle = _validate_lifecycle(root)
    sessions_root = root / ".runpod" / "sessions"
    try:
        summaries = validate_completed_runpod_sessions(
            sessions_root=sessions_root,
            ledger=_LedgerSnapshot(ledger_document),  # type: ignore[arg-type]
        )
    except RunpodSessionError as exc:
        raise PaidBundleRotationError(
            "every private RunPod session must have authenticated settlement evidence"
        ) from exc
    summary_pairs = {
        (str(item["reservation_id"]), str(item["session_hash"])) for item in summaries
    }
    if len(summary_pairs) != len(summaries):
        raise PaidBundleRotationError("completed RunPod session identities are not unique")
    gpu_entries = [
        item for item in entries if isinstance(item, Mapping) and item.get("kind") == "gpu"
    ]
    ledger_reservations = {
        str(item.get("entry_id"))
        for item in gpu_entries
        if item.get("status") == "incurred" and isinstance(item.get("entry_id"), str)
    }
    if (
        len(ledger_reservations) != len(gpu_entries)
        or ledger_reservations != {reservation for reservation, _session in summary_pairs}
    ):
        raise PaidBundleRotationError(
            "every incurred GPU ledger reservation must have one authenticated session closure"
        )
    history = lifecycle.get("authorization_history")
    current = lifecycle.get("current_authorization")
    if not isinstance(history, list) or not isinstance(current, Mapping):
        raise PaidBundleRotationError("RunPod lifecycle authorization inventory is malformed")
    authorizations = [*history, current]
    lifecycle_pairs: list[tuple[str, str]] = []
    for authorization in authorizations:
        if not isinstance(authorization, Mapping):
            raise PaidBundleRotationError("RunPod lifecycle authorization inventory is malformed")
        reservation_id = authorization.get("reservation_id")
        session_hash = authorization.get("session_hash")
        if (
            not isinstance(reservation_id, str)
            or _HASH_RE.fullmatch(reservation_id) is None
            or not isinstance(session_hash, str)
            or _HASH_RE.fullmatch(session_hash) is None
        ):
            raise PaidBundleRotationError("RunPod lifecycle authorization identity is malformed")
        lifecycle_pairs.append((reservation_id, session_hash))
    if len(set(lifecycle_pairs)) != len(lifecycle_pairs) or set(lifecycle_pairs) != summary_pairs:
        raise PaidBundleRotationError(
            "every lifecycle authorization must have one authenticated session closure"
        )
    if sessions_root.exists():
        for directory in sorted(sessions_root.iterdir(), key=lambda item: item.name):
            settlement_path = directory / "settlement.json"
            settlement = _decode_json_object(
                _read_owned_regular(
                    settlement_path,
                    label=f"schema-v2 settlement for session {directory.name}",
                ),
                label=f"schema-v2 settlement for session {directory.name}",
            )
            if (
                settlement.get("schema_version") != 2
                or settlement.get("protocol_version")
                != "cumulative-gpu-phase-settlement-v2"
                or settlement.get("status") != "settled"
            ):
                raise PaidBundleRotationError(
                    "every private RunPod session requires authenticated schema-v2 settlement"
                )


@contextmanager
def _canonical_ledger_lock(ledger_path: Path):  # type: ignore[no-untyped-def]
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    if os.path.lexists(lock_path):
        _read_owned_regular(
            lock_path,
            label="canonical cost-ledger lock",
            allow_empty=True,
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise PaidBundleRotationError("canonical cost-ledger lock is unsafe") from exc
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
        ):
            raise PaidBundleRotationError("canonical cost-ledger lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def paid_bundle_lock(
    *,
    project_root: str | Path,
    exclusive: bool,
):  # type: ignore[no-untyped-def]
    """Take the process-wide paid-control lock without waiting on an active peer.

    Rotation uses exclusive mode. Every paid consumer must hold shared mode from
    before loading its private context until its provider/model work ends.
    """

    try:
        root = Path(project_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PaidBundleRotationError("project root is missing or unsafe") from exc
    private = _require_owned_directory(root / ".runpod", label="canonical .runpod directory")
    lock_path = private / "paid_bundle.lock"
    if os.path.lexists(lock_path):
        _read_owned_regular(
            lock_path,
            label="shared paid-bundle lock",
            allow_empty=True,
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise PaidBundleRotationError("shared paid-bundle lock is unsafe") from exc
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
        ):
            raise PaidBundleRotationError("shared paid-bundle lock is unsafe")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PaidBundleRotationError(
                "paid bundle is already held by an active rotation or paid command"
            ) from exc
        yield
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _ledger_document_locked(root: Path) -> dict[str, Any]:
    ledger_path = root / "data" / "manifests" / "cost_ledger.yaml"
    _require_no_symlink_components(
        root=root,
        path=ledger_path,
        label="canonical cost ledger",
    )
    before = _read_owned_regular(ledger_path, label="canonical cost ledger")
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220.0, api=100.0, total=325.0))
    try:
        document = ledger._load_unlocked()
    except (OSError, RuntimeError, ValueError) as exc:
        raise PaidBundleRotationError("canonical cost ledger is invalid") from exc
    after = _read_owned_regular(ledger_path, label="canonical cost ledger")
    if before != after:
        raise PaidBundleRotationError("canonical cost ledger changed during authenticated read")
    return document


def _records_to_snapshots(root: Path, records: list[dict[str, Any]]) -> list[_ControlSnapshot]:
    specs = root / ".runpod" / "specs"
    if os.path.lexists(specs):
        _require_owned_directory(specs, label="private quote-spec directory")
    snapshots: list[_ControlSnapshot] = []
    for record in records:
        relative = PurePosixPath(str(record["source_path"]))
        content = _optional_owned_regular(
            root / relative.as_posix(),
            label=relative.as_posix(),
        )
        if content is None:
            continue
        _validate_control_payload(relative, content)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if digest != record["sha256"] or len(content) != record["size_bytes"]:
            raise PaidBundleRotationError(
                f"canonical paid control differs from the in-progress archive: {relative}"
            )
        snapshots.append(
            _ControlSnapshot(
                source_path=relative,
                archive_path=PurePosixPath(str(record["archive_path"])),
                sha256=digest,
                size_bytes=len(content),
                content=content,
            )
        )
    manifested = {str(item["source_path"]) for item in records}
    for relative in _CONTROL_PATHS:
        if relative.as_posix() not in manifested and os.path.lexists(root / relative.as_posix()):
            raise PaidBundleRotationError(
                f"unmanifested canonical paid control appeared during recovery: {relative}"
            )
    return snapshots


def _archive_content_for_record(bundle: Path, record: Mapping[str, Any]) -> bytes:
    archive_path = bundle / str(record["archive_path"])
    content = _read_owned_regular(archive_path, label=f"archived control {archive_path}")
    if (
        len(content) != record["size_bytes"]
        or "sha256:" + hashlib.sha256(content).hexdigest() != record["sha256"]
    ):
        raise PaidBundleRotationError(f"archived control authentication failed: {archive_path}")
    return content


def _unlink_manifested_source(root: Path, record: Mapping[str, Any]) -> None:
    source = root / str(record["source_path"])
    if not os.path.lexists(source):
        return
    content = _read_owned_regular(source, label=f"canonical paid control {source}")
    if (
        len(content) != record["size_bytes"]
        or "sha256:" + hashlib.sha256(content).hexdigest() != record["sha256"]
    ):
        raise PaidBundleRotationError(
            f"canonical paid control changed before removal: {record['source_path']}"
        )
    source.unlink()
    _fsync_directory(source.parent)


def _find_incomplete_bundle(archive_root: Path) -> tuple[str, Path] | None:
    candidates: list[tuple[str, Path]] = []
    for item in sorted(archive_root.iterdir(), key=lambda path: path.name):
        if item.is_symlink() or not item.is_dir() or _BUNDLE_ID_RE.fullmatch(item.name) is None:
            raise PaidBundleRotationError("paid-bundle archive root contains an unsafe entry")
        manifest = item / MANIFEST_FILENAME
        manifest_pending = _pending_path(manifest)
        completion = item / COMPLETION_FILENAME
        completion_pending = _pending_path(completion)
        if os.path.lexists(completion_pending):
            candidates.append((item.name, item))
        elif (
            (os.path.lexists(manifest) or os.path.lexists(manifest_pending))
            and not os.path.lexists(completion)
        ):
            candidates.append((item.name, item))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise PaidBundleRotationError(
            "multiple incomplete paid-bundle rotations require an explicit --bundle-id"
        )
    return candidates[0]


def rotate_paid_bundle(
    *,
    project_root: str | Path,
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Archive canonical paid controls after proving the environment is idle."""

    try:
        root = Path(project_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PaidBundleRotationError("project root is missing or unsafe") from exc
    _require_owned_directory(root, label="project root")
    _require_owned_directory(root / ".runpod", label="canonical .runpod directory")
    ledger_path = root / "data" / "manifests" / "cost_ledger.yaml"
    _require_no_symlink_components(
        root=root,
        path=ledger_path,
        label="canonical cost ledger",
    )

    with paid_bundle_lock(project_root=root, exclusive=True):
        with _canonical_ledger_lock(ledger_path):
            ledger_document = _ledger_document_locked(root)
            _validate_safety_state(root, ledger_document)
            archive_root = _ensure_archive_parent(root)

            selected_id: str
            bundle: Path
            manifest: dict[str, Any]
            records: list[dict[str, Any]]
            requested_id = _validate_bundle_id(bundle_id) if bundle_id is not None else None
            if requested_id is not None and os.path.lexists(archive_root / requested_id):
                selected_id = requested_id
                bundle = _require_owned_directory(
                    archive_root / selected_id,
                    label="existing paid-bundle archive",
                )
                completion_exists = os.path.lexists(bundle / COMPLETION_FILENAME) or os.path.lexists(
                    _pending_path(bundle / COMPLETION_FILENAME)
                )
                expected_manifest: Mapping[str, Any] | None = None
                if not completion_exists:
                    available = _snapshot_controls(root, require_required=False)
                    required_available = {
                        item.source_path
                        for item in available
                        if item.source_path in _REQUIRED_CONTROL_PATHS
                    }
                    if required_available == set(_REQUIRED_CONTROL_PATHS):
                        expected_manifest = _manifest_payload(
                            bundle_id=selected_id,
                            snapshots=available,
                        )
                manifest, records = _initialize_or_load_manifest(
                    bundle=bundle,
                    bundle_id=selected_id,
                    expected=expected_manifest,
                )
            else:
                controls = _snapshot_controls(root, require_required=False)
                required_present = {
                    item.source_path
                    for item in controls
                    if item.source_path in _REQUIRED_CONTROL_PATHS
                }
                if required_present != set(_REQUIRED_CONTROL_PATHS):
                    if requested_id is not None:
                        raise PaidBundleRotationError(
                            "required quote locks are missing and the requested archive does not exist"
                        )
                    incomplete = _find_incomplete_bundle(archive_root)
                    if incomplete is None:
                        raise PaidBundleRotationError(
                            "required quote locks are missing and no incomplete rotation is recoverable"
                        )
                    selected_id, bundle = incomplete
                    manifest, records = _initialize_or_load_manifest(
                        bundle=bundle,
                        bundle_id=selected_id,
                        expected=None,
                    )
                else:
                    records_for_hash = [item.manifest_record() for item in controls]
                    content_hash = _bundle_content_hash(records_for_hash)
                    selected_id = requested_id or f"sha256-{content_hash.removeprefix('sha256:')}"
                    selected_id = _validate_bundle_id(selected_id)
                    bundle = archive_root / selected_id
                    payload = _manifest_payload(bundle_id=selected_id, snapshots=controls)
                    if os.path.lexists(bundle):
                        bundle = _require_owned_directory(
                            bundle,
                            label="existing paid-bundle archive",
                        )
                        manifest, records = _initialize_or_load_manifest(
                            bundle=bundle,
                            bundle_id=selected_id,
                            expected=payload,
                        )
                    else:
                        bundle.mkdir(mode=0o700)
                        _fsync_directory(archive_root)
                        manifest = payload
                        records = list(payload["files"])
                        _install_exact_file(
                            bundle / MANIFEST_FILENAME,
                            _canonical_json_bytes(manifest),
                        )

            completion_path = bundle / COMPLETION_FILENAME
            completion_pending = _pending_path(completion_path)
            expected_completion = _completion_payload(manifest)
            _validate_archive_inventory(
                bundle=bundle,
                records=records,
                allow_pending=True,
                require_complete=False,
            )
            completed = os.path.lexists(completion_path) or os.path.lexists(completion_pending)
            if completed:
                _install_exact_file(
                    completion_path,
                    _canonical_json_bytes(expected_completion),
                )
                completion = _decode_json_object(
                    _read_owned_regular(completion_path, label="rotation completion marker"),
                    label="rotation completion marker",
                )
                _validate_completion(completion, manifest=manifest)
                if any(os.path.lexists(root / item.as_posix()) for item in _CONTROL_PATHS):
                    raise PaidBundleRotationError(
                        "completed archive bundle ID cannot be reused for new canonical controls"
                    )
            else:
                source_snapshots = _records_to_snapshots(root, records)
                sources_by_path = {
                    item.source_path.as_posix(): item for item in source_snapshots
                }
                for record in records:
                    archive_relative = PurePosixPath(str(record["archive_path"]))
                    _ensure_record_directories(bundle, archive_relative)
                    snapshot = sources_by_path.get(str(record["source_path"]))
                    if snapshot is not None:
                        _install_exact_file(bundle / archive_relative.as_posix(), snapshot.content)
                    else:
                        _archive_content_for_record(bundle, record)

                for record in records:
                    _archive_content_for_record(bundle, record)
                # Revalidate the lifecycle/session closure at the irreversible
                # boundary. The ledger lock still excludes reservations and
                # settlements, while the paid-bundle lock excludes paid peers.
                _validate_safety_state(root, ledger_document)
                for record in records:
                    _unlink_manifested_source(root, record)
                completion = expected_completion
                _install_exact_file(completion_path, _canonical_json_bytes(completion))
                _fsync_directory(bundle)

            _validate_archive_inventory(
                bundle=bundle,
                records=records,
                allow_pending=False,
                require_complete=True,
            )

            return {
                "schema_version": 1,
                "protocol_version": ROTATION_PROTOCOL,
                "bundle_id": selected_id,
                "bundle_content_hash": manifest["bundle_content_hash"],
                "manifest_record_hash": manifest["record_hash"],
                "file_count": len(records),
                "status": "complete",
                "provider_calls": 0,
            }


__all__ = [
    "ARCHIVE_RELATIVE_ROOT",
    "COMPLETION_FILENAME",
    "MANIFEST_FILENAME",
    "ROTATION_PROTOCOL",
    "PaidBundleRotationError",
    "paid_bundle_lock",
    "rotate_paid_bundle",
]
