"""One-shot, guarded transfer and recoverable remote install for RunPod sync.

The provider first clones the exact pushed runner commit into a new staging
checkout.  The host then adds only the selective private RunPod controls and
canonical cost ledger.  After staged verification, the provider archives the
old checkout and atomically promotes the complete authenticated stage, with
rollback on any failed installed-state verification.

This module's top-level imports are standard-library only so the remote
installer can run under ``python -I -S`` on a fresh provider image.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

REMOTE_WORKSPACE = Path("/workspace")
REMOTE_DESTINATION = Path("/workspace/model-forensics-value-leakage")
REMOTE_ARCHIVE_ROOT = Path("/workspace/.model-forensics-sync-archive")
PUBLIC_REPOSITORY_URL = "https://github.com/baeyongil/model-forensics-value-leakage.git"
_STAGE_PREFIX = ".model-forensics-sync-stage-"
_STAGE_NAME_RE = re.compile(r"\.model-forensics-sync-stage-[0-9a-f]{64}\Z")
_DIRECT_SSH_HOST_RE = re.compile(r"root@(?P<public_ip>[0-9]{1,3}(?:\.[0-9]{1,3}){3})\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SOURCE_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SESSION_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CURRENT_HOST_GUARD_KEYS = frozenset(
    {
        "acknowledgement_file_hash",
        "acknowledgement_record_hash",
        "watcher_process_identity_hash",
        "watchdog_invariant_hash",
        "direct_ssh_endpoint_hash",
    }
)
_CLAIM_PROTOCOL = "runpod-selective-sync-one-shot-claim-v1"
_MAXIMUM_MANIFEST_BYTES = 1024 * 1024
_RUNNER_SOURCE_PATHS = (
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


class RunpodSyncTransferError(RuntimeError):
    """A one-shot transfer or recoverable install could not be authenticated."""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any: ...


class EmergencyStopper(Protocol):
    def __call__(self) -> None: ...


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
        raise RunpodSyncTransferError("sync transfer record is not canonical JSON") from exc


def _stable_hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()}"


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RunpodSyncTransferError(f"{label} is missing or unsafe")
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid():
        raise RunpodSyncTransferError(f"{label} ownership is unsafe")
    return path


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RunpodSyncTransferError(
            "this RunPod session already has a one-shot transfer claim"
        ) from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RunpodSyncTransferError("one-shot transfer claim write failed")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_gid,
        details.st_nlink,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _read_claimed_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RunpodSyncTransferError("materialized manifest is missing or unsafe")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size > _MAXIMUM_MANIFEST_BYTES
    ):
        raise RunpodSyncTransferError("materialized manifest identity is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise RunpodSyncTransferError("materialized manifest changed before read")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAXIMUM_MANIFEST_BYTES:
                raise RunpodSyncTransferError("materialized manifest is oversized")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(after) != _file_identity(opened) or _file_identity(
        path.lstat()
    ) != _file_identity(opened):
        raise RunpodSyncTransferError("materialized manifest changed during read")
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodSyncTransferError("materialized manifest is not strict JSON") from exc
    if not isinstance(value, dict):
        raise RunpodSyncTransferError("materialized manifest is not an object")
    return value


def _remove_unclaimed_bundle(bundle: Path, *, root: Path) -> None:
    expected_parent = root / ".runpod" / "sync_bundles"
    if bundle.parent != expected_parent or re.fullmatch(r"[0-9a-f]{64}", bundle.name) is None:
        raise RunpodSyncTransferError("unclaimed sync-bundle cleanup target is unsafe")
    if not os.path.lexists(bundle):
        return
    if bundle.is_symlink():
        bundle.unlink()
    elif bundle.is_dir() and bundle.lstat().st_uid == os.getuid():
        shutil.rmtree(bundle)
    else:
        raise RunpodSyncTransferError("unclaimed sync bundle is unsafe")
    _fsync_directory(expected_parent)


def claim_one_shot_transfer(
    *,
    project_root: str | Path,
    plan: Mapping[str, Any],
    claimed_at: datetime | None = None,
) -> Path:
    """Atomically consume the sole transfer attempt for a session."""

    root = Path(project_root).resolve()
    private = _owned_directory(root / ".runpod", label="private RunPod root")
    session_hash = plan.get("session_hash")
    manifest_hash = plan.get("record_hash")
    source_commit = plan.get("source_commit")
    if (
        not isinstance(session_hash, str)
        or _SESSION_RE.fullmatch(session_hash) is None
        or not isinstance(manifest_hash, str)
        or _HASH_RE.fullmatch(manifest_hash) is None
        or not isinstance(source_commit, str)
        or _SOURCE_COMMIT_RE.fullmatch(source_commit) is None
    ):
        raise RunpodSyncTransferError("sync plan identity is malformed")
    unsigned = {key: value for key, value in plan.items() if key != "record_hash"}
    if _stable_hash(unsigned) != manifest_hash:
        raise RunpodSyncTransferError("sync plan record hash does not authenticate")

    claims = private / "sync_claims"
    if os.path.lexists(claims):
        _owned_directory(claims, label="one-shot sync-claim root")
    else:
        claims.mkdir(mode=0o700)
        _fsync_directory(private)
    claim_path = claims / f"{session_hash.removeprefix('sha256:')}.json"
    timestamp = (claimed_at or datetime.now(UTC)).astimezone(UTC)
    claim: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": _CLAIM_PROTOCOL,
        "claimed_at": timestamp.isoformat().replace("+00:00", "Z"),
        "session_hash": session_hash,
        "manifest_record_hash": manifest_hash,
        "source_commit": source_commit,
    }
    claim["record_hash"] = _stable_hash(claim)
    _write_exclusive_json(claim_path, claim)
    return claim_path


def _create_durable_stop_request(*, root: Path, session_hash: str) -> Path:
    if _SESSION_RE.fullmatch(session_hash) is None:
        raise RunpodSyncTransferError("stop-request session identity is malformed")
    session = _owned_directory(
        root / ".runpod" / "sessions" / session_hash.removeprefix("sha256:"),
        label="current host-watch session",
    )
    request = session / "runpod_stop.request"
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    if os.path.lexists(request):
        if request.is_symlink() or not request.is_file():
            raise RunpodSyncTransferError("host stop request path is unsafe")
        details = request.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
        ):
            raise RunpodSyncTransferError("host stop request identity is unsafe")
        descriptor = os.open(request, flags)
    else:
        descriptor = os.open(request, flags | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(session)
    return request


def _build_provider_emergency_stopper(*, root: Path, session_hash: str) -> EmergencyStopper:
    """Preflight an independent, exact-Pod stop path before the first SSH contact."""

    from model_forensics.runpod_lifecycle_state import (
        authorization_from_state,
        load_lifecycle_state,
    )
    from model_forensics.runpod_watchdog import RunpodStopClient, WatchdogError

    lifecycle_path = (root / ".runpod" / "pod_lifecycle.json").resolve()
    lifecycle = load_lifecycle_state(lifecycle_path)
    authorization = authorization_from_state(lifecycle)
    pod = lifecycle.get("pod")
    if (
        authorization.session_hash != session_hash
        or lifecycle.get("operation") != "rearmed"
        or not isinstance(pod, Mapping)
        or not isinstance(pod.get("id"), str)
        or pod.get("status") != "RUNNING"
    ):
        raise RunpodSyncTransferError(
            "sync emergency stop preflight disagrees with the active lifecycle"
        )
    pod_id = str(pod["id"])
    try:
        client = RunpodStopClient(
            pod_id=pod_id,
            expected_session_hash=session_hash,
            timeout_seconds=20,
        )
    except (OSError, ValueError, WatchdogError) as exc:
        raise RunpodSyncTransferError(
            "sync requires independent provider-stop credentials before remote contact"
        ) from exc

    def stop() -> None:
        last_error: BaseException | None = None
        for attempt in range(5):
            try:
                status = client.desired_status()
                if status in {"EXITED", "TERMINATED"}:
                    return
                client.stop()
                status = client.desired_status()
                if status in {"EXITED", "TERMINATED"}:
                    return
            except (OSError, ValueError, WatchdogError) as exc:
                last_error = exc
            if attempt < 4:
                time.sleep(1)
        raise RunpodSyncTransferError(
            "independent provider stop could not confirm EXITED after sync failure"
        ) from last_error

    return stop


def _remote_stage(plan: Mapping[str, Any]) -> Path:
    binding = {
        "manifest_record_hash": plan["record_hash"],
        "session_hash": plan["session_hash"],
        "source_commit": plan["source_commit"],
        "source_repository_url": plan["source_repository_url"],
    }
    digest = _stable_hash(binding).removeprefix("sha256:")
    return REMOTE_WORKSPACE / f"{_STAGE_PREFIX}{digest}"


def _rsync_inventory_filters(plan: Mapping[str, Any]) -> list[str]:
    inventory = plan.get("files")
    if not isinstance(inventory, list):
        raise RunpodSyncTransferError("sync plan inventory is malformed")
    files = {".runpod/selective_sync_manifest.json"}
    for item in inventory:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise RunpodSyncTransferError("sync plan inventory path is malformed")
        path = str(item["path"])
        parts = path.split("/")
        if (
            not path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or any(character in path for character in ("\n", "\r", "\0", "\\"))
        ):
            raise RunpodSyncTransferError("sync plan inventory path is unsafe")
        files.add(path)
    directories: set[str] = set()
    for path in files:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    filters = [
        f"--include=/{directory}/"
        for directory in sorted(
            directories,
            key=lambda value: (value.count("/"), value),
        )
    ]
    filters.extend(f"--include=/{path}" for path in sorted(files))
    filters.extend(("--exclude=*", "--prune-empty-dirs"))
    return filters


def _run_command(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    label: str,
    timeout_seconds: int,
) -> None:
    try:
        result = runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunpodSyncTransferError(f"{label} did not complete") from exc
    if getattr(result, "returncode", None) != 0:
        # Remote stdout/stderr is intentionally not reflected into the exception:
        # provider failures must not turn credentials into host logs.
        raise RunpodSyncTransferError(f"{label} failed closed")


def _ssh_command(
    *,
    remote_host: str,
    remote_port: int,
    arguments: Sequence[str],
) -> list[str]:
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-p",
        str(remote_port),
        remote_host,
        *arguments,
    ]


def _normalize_remote_endpoint(
    *,
    remote_host: str,
    remote_port: int,
) -> tuple[str, str, int]:
    if not isinstance(remote_host, str):
        raise RunpodSyncTransferError("remote SSH host must be root@canonical-IPv4")
    matched = _DIRECT_SSH_HOST_RE.fullmatch(remote_host)
    if matched is None:
        raise RunpodSyncTransferError("remote SSH host must be root@canonical-IPv4")
    try:
        parsed_ip = ipaddress.ip_address(matched.group("public_ip"))
    except ValueError as exc:
        raise RunpodSyncTransferError("remote SSH host must be root@canonical-IPv4") from exc
    canonical_ip = str(parsed_ip)
    if parsed_ip.version != 4 or matched.group("public_ip") != canonical_ip:
        raise RunpodSyncTransferError("remote SSH host must be root@canonical-IPv4")
    if (
        isinstance(remote_port, bool)
        or not isinstance(remote_port, int)
        or not 1 <= remote_port <= 65535
    ):
        raise RunpodSyncTransferError("remote SSH port is malformed")
    return f"root@{canonical_ip}", canonical_ip, remote_port


def _direct_ssh_endpoint_hash(*, public_ip: str, public_port: int) -> str:
    canonical = json.dumps(
        {"public_ip": public_ip, "public_port": public_port},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    raw = f"runpod-rest-v1-direct-ssh-v1:{canonical}".encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _authenticate_remote_endpoint(
    *,
    plan: Mapping[str, Any],
    public_ip: str,
    public_port: int,
) -> None:
    record_hash = plan.get("record_hash")
    unsigned = {key: value for key, value in plan.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _HASH_RE.fullmatch(record_hash) is None
        or not hmac.compare_digest(_stable_hash(unsigned), record_hash)
    ):
        raise RunpodSyncTransferError("sync plan record hash does not authenticate")
    guard = plan.get("current_host_guard")
    if (
        not isinstance(guard, Mapping)
        or set(guard) != _CURRENT_HOST_GUARD_KEYS
        or any(
            not isinstance(value, str) or _HASH_RE.fullmatch(value) is None
            for value in guard.values()
        )
    ):
        raise RunpodSyncTransferError("sync plan authenticated host guard binding is malformed")
    expected_hash = guard["direct_ssh_endpoint_hash"]
    observed_hash = _direct_ssh_endpoint_hash(
        public_ip=public_ip,
        public_port=public_port,
    )
    if not hmac.compare_digest(observed_hash, expected_hash):
        raise RunpodSyncTransferError(
            "remote SSH endpoint disagrees with the authenticated host guard"
        )


def _validate_remote_endpoint(
    *,
    remote_host: str,
    remote_port: int,
    remote_destination: str | Path,
) -> tuple[str, str, int]:
    normalized = _normalize_remote_endpoint(
        remote_host=remote_host,
        remote_port=remote_port,
    )
    if Path(remote_destination) != REMOTE_DESTINATION:
        raise RunpodSyncTransferError("remote destination is not the pinned project checkout")
    return normalized


def transfer_runpod_sync_bundle(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
    cost_ledger_path: str | Path,
    remote_host: str,
    remote_port: int = 22,
    remote_destination: str | Path = REMOTE_DESTINATION,
    limits: Any = None,
    command_runner: CommandRunner = subprocess.run,
    _plan_builder: Callable[..., dict[str, Any]] | None = None,
    _materializer: Callable[..., Path] | None = None,
    _revalidator: Callable[..., None] | None = None,
    _emergency_stopper: EmergencyStopper | None = None,
) -> dict[str, Any]:
    """Build, source-stage, claim, transfer, verify, and promote exactly once."""

    remote_host, public_ip, remote_port = _validate_remote_endpoint(
        remote_host=remote_host,
        remote_port=remote_port,
        remote_destination=remote_destination,
    )
    root = Path(project_root).resolve()
    if _plan_builder is None or _materializer is None or _revalidator is None:
        from model_forensics.runpod_sync import (
            build_selective_sync_plan,
            materialize_selective_sync_bundle,
            revalidate_selective_sync_plan,
        )

        _plan_builder = _plan_builder or build_selective_sync_plan
        _materializer = _materializer or materialize_selective_sync_bundle
        _revalidator = _revalidator or revalidate_selective_sync_plan

    plan = _plan_builder(
        project_root=root,
        phase=phase,
        reservation_path=reservation_path,
        cost_ledger_path=cost_ledger_path,
        limits=limits,
    )
    # Freeze the exact plan value before the durable claim and all callbacks.
    try:
        plan = json.loads(_canonical_json(plan))
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical JSON is ours
        raise RunpodSyncTransferError("sync plan cannot be frozen") from exc
    _authenticate_remote_endpoint(
        plan=plan,
        public_ip=public_ip,
        public_port=remote_port,
    )
    if plan.get("source_repository_url") != PUBLIC_REPOSITORY_URL:
        raise RunpodSyncTransferError("sync plan does not bind the canonical public repository")
    session_digest = str(plan["session_hash"]).removeprefix("sha256:")
    session_hash = str(plan["session_hash"])
    emergency_stopper = _emergency_stopper or _build_provider_emergency_stopper(
        root=root,
        session_hash=session_hash,
    )
    claim_path = root / ".runpod" / "sync_claims" / f"{session_digest}.json"
    if os.path.lexists(claim_path):
        _create_durable_stop_request(root=root, session_hash=session_hash)
        emergency_stopper()
        raise RunpodSyncTransferError("this RunPod session already has a one-shot transfer claim")
    bundle = root / ".runpod" / "sync_bundles" / session_digest
    materialized = _materializer(project_root=root, destination=bundle, plan=plan)
    if Path(materialized).absolute() != bundle.absolute():
        raise RunpodSyncTransferError("materializer returned a non-canonical bundle")
    manifest_path = bundle / ".runpod" / "selective_sync_manifest.json"
    try:
        manifest = _read_claimed_manifest(manifest_path)
    except OSError as exc:
        _remove_unclaimed_bundle(bundle, root=root)
        raise RunpodSyncTransferError("materialized manifest is unreadable") from exc
    except RunpodSyncTransferError:
        _remove_unclaimed_bundle(bundle, root=root)
        raise
    if manifest != plan:
        _remove_unclaimed_bundle(bundle, root=root)
        raise RunpodSyncTransferError("materialized manifest differs from the claimed plan")

    # From the first possible remote contact onward, every failure must own a
    # durable one-shot claim and an independent provider stop path. A dead host
    # watcher can therefore never turn an SSH/rsync failure into unbounded
    # billing.
    _revalidator(project_root=root, plan=plan)
    claim_one_shot_transfer(project_root=root, plan=plan)

    stage = _remote_stage(plan)
    destination = str(REMOTE_DESTINATION)
    source_commit = str(plan["source_commit"])
    manifest_hash = str(plan["record_hash"])
    verifier = f"{destination}/scripts/verify_runpod_sync_bundle.py"
    staged_installer = f"{stage}/scripts/install_runpod_sync_bundle.py"
    staged_verifier = f"{stage}/scripts/verify_runpod_sync_bundle.py"
    cleanup_command = _ssh_command(
        remote_host=remote_host,
        remote_port=remote_port,
        arguments=["rm", "-rf", "--", str(stage)],
    )
    stage_may_exist = False

    def guarded_remote_command(
        argv: Sequence[str],
        *,
        label: str,
        timeout_seconds: float,
    ) -> None:
        _revalidator(project_root=root, plan=plan)
        _run_command(
            command_runner,
            argv,
            label=label,
            timeout_seconds=timeout_seconds,
        )

    try:
        # Bootstrap an exact pushed runner checkout without mutating the old
        # provider checkout.  Git clone itself is the exclusive stage claim.
        stage_may_exist = True
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "git",
                    "clone",
                    "--no-checkout",
                    "--",
                    PUBLIC_REPOSITORY_URL,
                    str(stage),
                ],
            ),
            label="remote exact-source clone",
            timeout_seconds=180,
        )
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "git",
                    "-C",
                    str(stage),
                    "checkout",
                    "--detach",
                    source_commit,
                ],
            ),
            label="remote exact-source checkout",
            timeout_seconds=60,
        )
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "python3",
                    "-I",
                    "-S",
                    staged_installer,
                    "prepare",
                    "--stage",
                    str(stage),
                    "--source-checkout",
                    str(stage),
                    "--expected-source-commit",
                    source_commit,
                    "--expected-source-repository-url",
                    PUBLIC_REPOSITORY_URL,
                ],
            ),
            label="remote exact-source validation",
            timeout_seconds=60,
        )
        guarded_remote_command(
            [
                "rsync",
                "-a",
                "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                *_rsync_inventory_filters(plan),
                "--rsh",
                (
                    "ssh -F /dev/null -o BatchMode=yes "
                    f"-o StrictHostKeyChecking=yes -p {remote_port}"
                ),
                f"{bundle}{os.sep}",
                f"{remote_host}:{stage}{os.sep}",
            ],
            label="selective bundle transfer",
            timeout_seconds=180,
        )
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "python3",
                    "-I",
                    "-S",
                    staged_verifier,
                    "--project-root",
                    str(stage),
                    "--source-checkout",
                    str(stage),
                ],
            ),
            label="remote staged-bundle verification",
            timeout_seconds=60,
        )
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "python3",
                    "-I",
                    "-S",
                    staged_installer,
                    "install",
                    "--stage",
                    str(stage),
                    "--source-checkout",
                    str(stage),
                    "--expected-manifest-record-hash",
                    manifest_hash,
                    "--expected-session-hash",
                    session_hash,
                    "--expected-source-commit",
                    source_commit,
                    "--expected-source-repository-url",
                    PUBLIC_REPOSITORY_URL,
                ],
            ),
            label="recoverable selective-sync install",
            timeout_seconds=90,
        )
        guarded_remote_command(
            _ssh_command(
                remote_host=remote_host,
                remote_port=remote_port,
                arguments=[
                    "python3",
                    "-I",
                    "-S",
                    verifier,
                    "--project-root",
                    destination,
                    "--source-checkout",
                    destination,
                ],
            ),
            label="remote installed-bundle verification",
            timeout_seconds=60,
        )
        # Final host authorization must still be the exact plan just installed.
        _revalidator(project_root=root, plan=plan)
    except BaseException:
        stop_request_error: BaseException | None = None
        provider_stop_error: BaseException | None = None
        claimed = os.path.lexists(claim_path)
        local_cleanup_error: BaseException | None = None
        if claimed:
            try:
                _create_durable_stop_request(root=root, session_hash=session_hash)
            except BaseException as exc:
                stop_request_error = exc
            try:
                emergency_stopper()
            except BaseException as exc:
                provider_stop_error = exc
        else:
            try:
                _remove_unclaimed_bundle(bundle, root=root)
            except BaseException as exc:
                local_cleanup_error = exc
        if stage_may_exist:
            try:
                _run_command(
                    command_runner,
                    cleanup_command,
                    label="remote stage cleanup",
                    timeout_seconds=30,
                )
            except RunpodSyncTransferError:
                pass
        if provider_stop_error is not None:
            raise RunpodSyncTransferError(
                "post-claim failure could not confirm the independent provider stop"
            ) from provider_stop_error
        if stop_request_error is not None:
            raise RunpodSyncTransferError(
                "post-claim failure could not durably request the mandatory host stop"
            ) from stop_request_error
        if local_cleanup_error is not None:
            raise RunpodSyncTransferError(
                "pre-claim failure could not clean the retryable local bundle"
            ) from local_cleanup_error
        raise

    return {
        "schema_version": 1,
        "protocol_version": "runpod-selective-sync-one-shot-transfer-v1",
        "session_hash": session_hash,
        "manifest_record_hash": manifest_hash,
        "source_commit": source_commit,
        "source_repository_url": PUBLIC_REPOSITORY_URL,
        "remote_stage_hash": _stable_hash({"remote_stage": stage.name}),
        "passed": True,
    }


def _validate_stage_path(stage: Path, *, workspace: Path) -> None:
    if stage.parent != workspace or _STAGE_NAME_RE.fullmatch(stage.name) is None:
        raise RunpodSyncTransferError("remote stage path is not canonical")


def _assert_source_checkout(
    checkout: Path,
    *,
    expected_commit: str,
    expected_repository_url: str,
) -> None:
    if (
        checkout.is_symlink()
        or not checkout.is_dir()
        or _SOURCE_COMMIT_RE.fullmatch(expected_commit) is None
        or expected_repository_url != PUBLIC_REPOSITORY_URL
    ):
        raise RunpodSyncTransferError("remote source checkout binding is malformed")

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        top_level = git("rev-parse", "--show-toplevel")
        head = git("rev-parse", "--verify", "HEAD")
        origin = git("remote", "get-url", "origin")
        status = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_RUNNER_SOURCE_PATHS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunpodSyncTransferError("remote source checkout cannot be authenticated") from exc
    if (
        top_level.returncode != 0
        or Path(top_level.stdout.strip()).resolve() != checkout.resolve()
        or head.returncode != 0
        or head.stdout.strip() != expected_commit
        or origin.returncode != 0
        or origin.stdout.strip() != PUBLIC_REPOSITORY_URL
        or status.returncode != 0
        or bool(status.stdout)
    ):
        raise RunpodSyncTransferError("remote source checkout commit disagrees")


def prepare_remote_stage(
    *,
    stage: str | Path,
    source_checkout: str | Path,
    expected_source_commit: str,
    expected_source_repository_url: str,
) -> dict[str, Any]:
    """Authenticate the exclusively cloned canonical provider source stage."""

    stage_path = Path(stage)
    checkout = Path(source_checkout)
    _validate_stage_path(stage_path, workspace=REMOTE_WORKSPACE)
    if checkout != stage_path:
        raise RunpodSyncTransferError("remote source stage and checkout disagree")
    _assert_source_checkout(
        checkout,
        expected_commit=expected_source_commit,
        expected_repository_url=expected_source_repository_url,
    )
    if REMOTE_WORKSPACE.is_symlink() or not REMOTE_WORKSPACE.is_dir():
        raise RunpodSyncTransferError("remote workspace root is unsafe")
    if os.path.lexists(stage_path / ".runpod"):
        raise RunpodSyncTransferError("new source stage already contains private RunPod state")
    return {
        "schema_version": 1,
        "source_commit": expected_source_commit,
        "source_repository_url": PUBLIC_REPOSITORY_URL,
        "stage_hash": _stable_hash({"remote_stage": stage_path.name}),
        "prepared": True,
    }


def cleanup_remote_stage(
    *,
    stage: str | Path,
    source_checkout: str | Path,
) -> dict[str, Any]:
    stage_path = Path(stage)
    if Path(source_checkout) != stage_path:
        raise RunpodSyncTransferError("remote source checkout is not the canonical stage")
    _validate_stage_path(stage_path, workspace=REMOTE_WORKSPACE)
    if os.path.lexists(stage_path):
        if stage_path.is_symlink() or not stage_path.is_dir():
            raise RunpodSyncTransferError("remote stage is unsafe")
        shutil.rmtree(stage_path)
        _fsync_directory(REMOTE_WORKSPACE)
    return {"schema_version": 1, "cleaned": True}


def _ensure_plain_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RunpodSyncTransferError(f"{label} is unsafe")
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid():
        raise RunpodSyncTransferError(f"{label} ownership is unsafe")


def _ensure_plain_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunpodSyncTransferError(f"{label} is unsafe")
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid() or details.st_nlink != 1:
        raise RunpodSyncTransferError(f"{label} is unsafe")


def _verify_expected_summary(
    summary: Mapping[str, Any],
    *,
    expected_manifest_record_hash: str,
    expected_session_hash: str,
    expected_source_commit: str,
) -> None:
    if (
        summary.get("passed") is not True
        or summary.get("manifest_record_hash") != expected_manifest_record_hash
        or summary.get("session_hash") != expected_session_hash
        or summary.get("source_commit") != expected_source_commit
    ):
        raise RunpodSyncTransferError("selective sync verification identity disagrees")


def _install_staged_sync(
    *,
    stage: Path,
    destination: Path,
    archive_root: Path,
    source_checkout: Path,
    expected_manifest_record_hash: str,
    expected_session_hash: str,
    expected_source_commit: str,
    verifier: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically promote a verified exact-source checkout, with rollback."""

    workspace = destination.parent
    if stage.parent != workspace or archive_root.parent != workspace:
        raise RunpodSyncTransferError("remote promotion paths do not share the workspace")
    _ensure_plain_directory(workspace, label="remote workspace")
    _ensure_plain_directory(destination, label="existing remote destination checkout")
    _ensure_plain_directory(stage, label="verified remote source stage")
    _ensure_plain_directory(stage / ".runpod", label="staged private control tree")
    _ensure_plain_file(
        stage / "data" / "manifests" / "cost_ledger.yaml",
        label="staged canonical cost ledger",
    )
    if source_checkout != stage:
        raise RunpodSyncTransferError("installer source checkout is not the source stage")
    first_summary = verifier(project_root=stage, source_checkout=stage)
    _verify_expected_summary(
        first_summary,
        expected_manifest_record_hash=expected_manifest_record_hash,
        expected_session_hash=expected_session_hash,
        expected_source_commit=expected_source_commit,
    )

    if os.path.lexists(archive_root):
        _ensure_plain_directory(archive_root, label="remote sync archive root")
    else:
        archive_root.mkdir(mode=0o700)
        _fsync_directory(workspace)
    archive = archive_root / stage.name.removeprefix(_STAGE_PREFIX)
    try:
        archive.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RunpodSyncTransferError("remote sync archive already exists") from exc
    _fsync_directory(archive_root)
    archived_destination = archive / destination.name
    destination_archived = False
    stage_promoted = False

    try:
        os.replace(destination, archived_destination)
        destination_archived = True
        _fsync_directory(workspace)
        _fsync_directory(archive)
        os.replace(stage, destination)
        stage_promoted = True
        _fsync_directory(workspace)
        second_summary = verifier(project_root=destination, source_checkout=destination)
        _verify_expected_summary(
            second_summary,
            expected_manifest_record_hash=expected_manifest_record_hash,
            expected_session_hash=expected_session_hash,
            expected_source_commit=expected_source_commit,
        )
    except BaseException as install_error:
        rollback_error: BaseException | None = None
        try:
            if stage_promoted and os.path.lexists(destination):
                os.replace(destination, stage)
                _fsync_directory(workspace)
            if destination_archived and os.path.lexists(archived_destination):
                os.replace(archived_destination, destination)
                _fsync_directory(archive)
                _fsync_directory(workspace)
        except BaseException as exc:  # pragma: no cover - catastrophic filesystem loss
            rollback_error = exc
        if rollback_error is not None:
            raise RunpodSyncTransferError(
                "selective sync checkout promotion failed and rollback was incomplete"
            ) from rollback_error
        raise RunpodSyncTransferError(
            "selective sync checkout promotion failed; the prior checkout was restored"
        ) from install_error

    return dict(second_summary)


def install_remote_stage(
    *,
    stage: str | Path,
    source_checkout: str | Path,
    expected_manifest_record_hash: str,
    expected_session_hash: str,
    expected_source_commit: str,
    expected_source_repository_url: str,
) -> dict[str, Any]:
    """Verify and recoverably install an authenticated provider stage."""

    from model_forensics.runpod_sync_verify import verify_selective_sync

    stage_path = Path(stage)
    checkout = Path(source_checkout)
    _validate_stage_path(stage_path, workspace=REMOTE_WORKSPACE)
    if checkout != stage_path:
        raise RunpodSyncTransferError("remote source checkout is not the canonical stage")
    if (
        _HASH_RE.fullmatch(expected_manifest_record_hash) is None
        or _SESSION_RE.fullmatch(expected_session_hash) is None
        or _SOURCE_COMMIT_RE.fullmatch(expected_source_commit) is None
        or expected_source_repository_url != PUBLIC_REPOSITORY_URL
    ):
        raise RunpodSyncTransferError("expected selective sync identity is malformed")
    _assert_source_checkout(
        checkout,
        expected_commit=expected_source_commit,
        expected_repository_url=expected_source_repository_url,
    )
    return _install_staged_sync(
        stage=stage_path,
        destination=REMOTE_DESTINATION,
        archive_root=REMOTE_ARCHIVE_ROOT,
        source_checkout=checkout,
        expected_manifest_record_hash=expected_manifest_record_hash,
        expected_session_hash=expected_session_hash,
        expected_source_commit=expected_source_commit,
        verifier=verify_selective_sync,
    )


__all__ = [
    "PUBLIC_REPOSITORY_URL",
    "REMOTE_DESTINATION",
    "RunpodSyncTransferError",
    "claim_one_shot_transfer",
    "cleanup_remote_stage",
    "install_remote_stage",
    "prepare_remote_stage",
    "transfer_runpod_sync_bundle",
]
