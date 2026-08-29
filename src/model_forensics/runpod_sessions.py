"""Private lifecycle records for sequential RunPod GPU phases."""

from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_forensics.budget import CostLedger
from model_forensics.gpu_budget import (
    GPU_PHASE_BUDGET_PROTOCOL,
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    GpuPhaseBudgetReservation,
    validate_existing_gpu_phase_reservation,
)
from model_forensics.io import read_json, stable_hash, write_json
from model_forensics.runpod_recovery import (
    EXTERNAL_STOP_RECEIPT_FILENAME,
    RunpodRecoveryError,
    load_external_stop_receipt,
)

GPU_BUDGET_BOOTSTRAP_FILENAME = "gpu_budget_bootstrap.json"
WATCHDOG_STATE_FILENAME = "runpod_watchdog.json"
SETTLEMENT_FILENAME = "settlement.json"
GPU_PREFLIGHT_FILENAME = "gpu_preflight.json"
WATCHDOG_PID_FILENAME = "runpod_watchdog.pid"
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACED_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_IMAGE_DIGEST_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_WATCHDOG_STATE_SCHEMA_VERSION = 2
_WATCHDOG_STATE_PROTOCOL_VERSION = "runpod-gpu-cost-watchdog-v2"
_SETTLEMENT_V2_PROTOCOL_VERSION = "cumulative-gpu-phase-settlement-v2"
_WATCHDOG_PROCESS_PROTOCOL_VERSION = "runpod-watchdog-process-identity-v1"
_RUNPOD_V1_PROVIDER_EVIDENCE_GAPS = (
    "cuda_version",
    "global_networking_enabled",
    "interruptible",
    "locked",
    "runtime_gpu_count",
)


class RunpodSessionError(RuntimeError):
    """A private GPU session lifecycle is incomplete or inconsistent."""


def _read_linux_process_identity(*, pid: int, proc_root: Path) -> dict[str, Any]:
    """Read the non-reusable Linux identity of one live process.

    A PID alone is not an identity: the kernel can recycle it after a process
    exits.  Linux exposes both the boot identity and the process start tick,
    which together distinguish reincarnations.  The exact NUL-delimited
    command line additionally prevents an unrelated live process from
    satisfying the watchdog gate.
    """

    if pid <= 1:
        raise RunpodSessionError("watchdog PID must identify a non-system process")
    process_root = proc_root / str(pid)
    try:
        stat = (process_root / "stat").read_text(encoding="utf-8")
        raw_cmdline = (process_root / "cmdline").read_bytes()
        boot_id = (proc_root / "sys" / "kernel" / "random" / "boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, OSError) as exc:
        raise RunpodSessionError("watchdog process identity is not live") from exc

    # ``comm`` is parenthesized and may contain spaces or closing parentheses;
    # the last ')' therefore marks the stable boundary before field 3.
    close = stat.rfind(")")
    if close < 0:
        raise RunpodSessionError("watchdog process stat record is malformed")
    fields_after_comm = stat[close + 1 :].split()
    if len(fields_after_comm) < 20:
        raise RunpodSessionError("watchdog process stat record is incomplete")
    raw_start_ticks = fields_after_comm[19]  # proc(5) field 22; field 3 is index 0 here.
    if not raw_start_ticks.isdigit() or int(raw_start_ticks) <= 0:
        raise RunpodSessionError("watchdog process start time is invalid")
    if not raw_cmdline or raw_cmdline.strip(b"\0") == b"":
        raise RunpodSessionError("watchdog process command line is missing")
    argv_bytes = raw_cmdline.rstrip(b"\0").split(b"\0")
    if not argv_bytes or any(not token for token in argv_bytes):
        raise RunpodSessionError("watchdog process command line is malformed")
    if re.fullmatch(r"[0-9a-fA-F-]{16,64}", boot_id) is None:
        raise RunpodSessionError("Linux boot identity is malformed")
    return {
        "pid": pid,
        "linux_boot_id_hash": "sha256:"
        + hashlib.sha256(boot_id.encode("utf-8")).hexdigest(),
        "linux_proc_start_ticks": int(raw_start_ticks),
        "cmdline_hash": "sha256:" + hashlib.sha256(raw_cmdline).hexdigest(),
        "argv": [os.fsdecode(token) for token in argv_bytes],
    }


def record_watchdog_process_identity(
    path: str | Path,
    *,
    pid: int,
    required_cmdline_tokens: tuple[str, ...],
    proc_root: str | Path = "/proc",
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Atomically bind the watchdog PID to its start time and exact command line."""

    destination = Path(path)
    if destination.is_symlink() or destination.exists():
        raise RunpodSessionError("refusing to replace watchdog process identity record")
    if not required_cmdline_tokens or any(not token for token in required_cmdline_tokens):
        raise RunpodSessionError("watchdog process identity requires nonempty command tokens")
    snapshot = _read_linux_process_identity(pid=pid, proc_root=Path(proc_root))
    argv = snapshot.pop("argv")
    missing = [token for token in required_cmdline_tokens if token not in argv]
    if missing:
        raise RunpodSessionError("watchdog process command line does not match the armed command")
    payload = {
        "schema_version": 1,
        "protocol_version": _WATCHDOG_PROCESS_PROTOCOL_VERSION,
        **snapshot,
        "required_cmdline_token_hashes": [
            "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
            for token in required_cmdline_tokens
        ],
        "captured_at_utc": (captured_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
    }
    payload["record_hash"] = stable_hash(payload)
    write_json(destination, payload)
    try:
        destination.chmod(0o600)
    except OSError:  # pragma: no cover - best effort on unusual filesystems
        pass
    return payload


def validate_watchdog_process_identity(
    path: str | Path,
    *,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Require that the process recorded at arm time is still that same process."""

    identity_path = Path(path)
    _require_regular_private_record(identity_path)
    try:
        payload = read_json(identity_path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("watchdog process identity record is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("protocol_version") != _WATCHDOG_PROCESS_PROTOCOL_VERSION
    ):
        raise RunpodSessionError("watchdog process identity record is malformed")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(record_hash) is None
        or record_hash != stable_hash(unsigned)
    ):
        raise RunpodSessionError("watchdog process identity record hash mismatch")
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise RunpodSessionError("watchdog process identity PID is invalid")
    for field in ("linux_boot_id_hash", "cmdline_hash"):
        if (
            not isinstance(payload.get(field), str)
            or _NAMESPACED_HASH_RE.fullmatch(str(payload[field])) is None
        ):
            raise RunpodSessionError(f"watchdog process identity {field} is invalid")
    start_ticks = payload.get("linux_proc_start_ticks")
    if isinstance(start_ticks, bool) or not isinstance(start_ticks, int) or start_ticks <= 0:
        raise RunpodSessionError("watchdog process identity start time is invalid")
    token_hashes = payload.get("required_cmdline_token_hashes")
    if (
        not isinstance(token_hashes, list)
        or not token_hashes
        or not all(
            isinstance(value, str) and _NAMESPACED_HASH_RE.fullmatch(value) is not None
            for value in token_hashes
        )
    ):
        raise RunpodSessionError("watchdog process command token binding is invalid")

    live = _read_linux_process_identity(pid=pid, proc_root=Path(proc_root))
    for field in (
        "linux_boot_id_hash",
        "linux_proc_start_ticks",
        "cmdline_hash",
    ):
        if live[field] != payload[field]:
            raise RunpodSessionError(
                "watchdog PID was reused or its process identity changed"
            )
    return dict(payload)


def _require_regular_private_record(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunpodSessionError(f"private session record is missing or unsafe: {path}")


def _finite_number(value: Any, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RunpodSessionError(f"active session {field} must be finite numeric")
    parsed = float(value)
    if parsed < 0 or (not allow_zero and parsed == 0):
        raise RunpodSessionError(f"active session {field} must be nonnegative")
    return parsed


def _require_close(value: Any, expected: float, *, field: str) -> float:
    parsed = _finite_number(value, field=field, allow_zero=True)
    if abs(parsed - expected) > 1e-6:
        raise RunpodSessionError(f"active session {field} mismatch")
    return parsed


def _authenticated_record(path: Path, *, protocol: str) -> dict[str, Any]:
    _require_regular_private_record(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError(f"cannot read authenticated session record: {path}") from exc
    if not isinstance(payload, dict):
        raise RunpodSessionError(f"session record must be a JSON object: {path}")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != protocol:
        raise RunpodSessionError(f"session record protocol mismatch: {path}")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(record_hash) is None
        or record_hash != stable_hash(unsigned)
    ):
        raise RunpodSessionError(f"session record content hash mismatch: {path}")
    return payload


def _ledger_entry(ledger: CostLedger, *, reservation_id: str) -> dict[str, Any]:
    document = ledger.document()
    matching = [entry for entry in document["entries"] if entry.get("entry_id") == reservation_id]
    if len(matching) != 1:
        raise RunpodSessionError(
            "canonical ledger does not contain exactly one session reservation"
        )
    return dict(matching[0])


def _validated_bootstrap(path: Path) -> dict[str, Any]:
    payload = _authenticated_record(path, protocol=GPU_PHASE_BUDGET_PROTOCOL)
    if payload.get("passed") is not True:
        raise RunpodSessionError("GPU bootstrap budget gate did not pass")
    for field in ("session_hash", "reservation_id", "reservation_record_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or _NAMESPACED_HASH_RE.fullmatch(value) is None:
            raise RunpodSessionError(f"GPU bootstrap record has invalid {field}")
    return payload


def _validated_watchdog(path: Path) -> dict[str, Any]:
    _require_regular_private_record(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError(f"cannot read prior watchdog state: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _WATCHDOG_STATE_SCHEMA_VERSION
    ):
        raise RunpodSessionError("prior watchdog state is malformed")
    if payload.get("watchdog_version") != _WATCHDOG_STATE_PROTOCOL_VERSION:
        raise RunpodSessionError("prior watchdog state version is unsupported")
    if payload.get("status") != "stopped_confirmed":
        raise RunpodSessionError("prior GPU session is not stopped_confirmed")
    return payload


def _validated_active_v1_metadata(watchdog: dict[str, Any]) -> dict[str, Any]:
    """Require the explicit REST-v1 evidence boundary for active compute."""

    metadata = watchdog.get("live_metadata")
    if not isinstance(metadata, dict) or metadata.get("provider_api") != "rest-v1":
        raise RunpodSessionError("active session watchdog lacks RunPod rest-v1 evidence")
    unavailable = metadata.get("provider_evidence_unavailable")
    expected_gaps = set(_RUNPOD_V1_PROVIDER_EVIDENCE_GAPS)
    if (
        not isinstance(unavailable, list)
        or not all(isinstance(item, str) for item in unavailable)
        or len(unavailable) != len(set(unavailable))
        or set(unavailable) != expected_gaps
    ):
        raise RunpodSessionError("active session watchdog provider-evidence gaps mismatch")
    for field in _RUNPOD_V1_PROVIDER_EVIDENCE_GAPS:
        if field not in metadata or metadata[field] is not None:
            raise RunpodSessionError(
                f"active session watchdog v1-unavailable field {field} is not null"
            )
    for field in (
        "execution_identity_hash",
        "machine_id_hash",
        "direct_ssh_endpoint_hash",
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or _NAMESPACED_HASH_RE.fullmatch(value) is None:
            raise RunpodSessionError(f"active session watchdog {field} is invalid")
    for field in ("pod_id", "provider_gpu_id", "data_center_id", "container_image"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            raise RunpodSessionError(f"active session watchdog {field} is invalid")
    if _CONTAINER_IMAGE_DIGEST_RE.fullmatch(str(metadata["container_image"])) is None:
        raise RunpodSessionError("active session watchdog container_image is not digest-pinned")
    if (
        metadata.get("ssh_ready") is not True
        or metadata.get("direct_ssh_ready") is not True
        or metadata.get("environment_verified") is not True
        or metadata.get("network_volume_attached") is not False
    ):
        raise RunpodSessionError("active session watchdog v1 SSH/environment evidence is unsafe")
    return metadata


def _validate_local_gpu_evidence(
    preflight: dict[str, Any], *, provider_gpu_id: Any
) -> None:
    inventory = preflight.get("gpus")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 8
        or not all(isinstance(item, dict) for item in inventory)
    ):
        raise RunpodSessionError("active session GPU preflight lacks eight local GPU records")
    indices = [item.get("index") for item in inventory]
    uuids = [item.get("uuid") for item in inventory]
    names = [item.get("name") for item in inventory]
    drivers = [item.get("driver_version") for item in inventory]
    if indices != list(range(8)):
        raise RunpodSessionError("active session local GPU indices are not exactly 0 through 7")
    if (
        not all(isinstance(value, str) and value for value in uuids)
        or len(set(uuids)) != 8
    ):
        raise RunpodSessionError("active session local GPU UUID evidence is invalid")
    if not all(isinstance(value, str) and value for value in names) or len(set(names)) != 1:
        raise RunpodSessionError("active session local GPU family evidence is not homogeneous")
    if not isinstance(provider_gpu_id, str):
        raise RunpodSessionError("active session provider GPU identity is invalid")
    provider_upper = provider_gpu_id.upper()
    if "H100" in provider_upper:
        family = "H100"
    elif "A100" in provider_upper:
        family = "A100"
    else:
        raise RunpodSessionError("active session provider GPU family is unsupported")
    if family not in str(names[0]).upper():
        raise RunpodSessionError("active session local GPU family disagrees with provider")
    for item in inventory:
        memory = item.get("memory_gib")
        if (
            isinstance(memory, bool)
            or not isinstance(memory, (int, float))
            or not math.isfinite(float(memory))
            or float(memory) < 79
        ):
            raise RunpodSessionError("active session local GPU memory evidence is insufficient")
        if str(item.get("mig_mode", "")).lower() != "disabled":
            raise RunpodSessionError("active session local GPU MIG evidence is unsafe")
    if not all(isinstance(value, str) and value for value in drivers) or len(set(drivers)) != 1:
        raise RunpodSessionError("active session local GPU driver evidence is invalid")
    if preflight.get("allowed_cuda_versions") != ["12.8"]:
        raise RunpodSessionError("active session frozen CUDA placement evidence mismatch")
    compatibility = preflight.get("cuda_compatibility")
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("required_environment")
        != "VLLM_ENABLE_CUDA_COMPATIBILITY=1"
        or compatibility.get("compatibility_directory") != "/usr/local/cuda-13.0/compat"
        or compatibility.get("required_libraries")
        != ["libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"]
    ):
        raise RunpodSessionError("active session local CUDA compatibility evidence is invalid")


def _validated_settlement(
    path: Path,
    *,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    _require_regular_private_record(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError(f"cannot read authenticated session record: {path}") from exc
    if not isinstance(payload, dict):
        raise RunpodSessionError(f"session record must be a JSON object: {path}")
    version = payload.get("schema_version")
    protocol = payload.get("protocol_version")
    if (version, protocol) not in {
        (1, GPU_PHASE_SETTLEMENT_PROTOCOL),
        (2, _SETTLEMENT_V2_PROTOCOL_VERSION),
    }:
        raise RunpodSessionError(f"session record protocol mismatch: {path}")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        not isinstance(record_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(record_hash) is None
        or record_hash != stable_hash(unsigned)
    ):
        raise RunpodSessionError(f"session record content hash mismatch: {path}")
    if payload.get("status") != "settled":
        raise RunpodSessionError("prior GPU session settlement is incomplete")
    for field in ("session_hash", "reservation_id", "reservation_record_hash"):
        if payload.get(field) != bootstrap.get(field):
            raise RunpodSessionError(f"prior settlement {field} disagrees with bootstrap")
    incurred = (
        payload.get("accounted_gpu_usd")
        if version == 2
        else payload.get("provider_incurred_usd")
    )
    if (
        isinstance(incurred, bool)
        or not isinstance(incurred, (int, float))
        or not math.isfinite(float(incurred))
        or float(incurred) < 0
    ):
        raise RunpodSessionError("prior settlement incurred cost is invalid")
    if version == 2:
        for field in (
            "external_stop_receipt_hash",
            "stop_evidence_hash",
            "billing_evidence_hash",
        ):
            if (
                not isinstance(payload.get(field), str)
                or _NAMESPACED_HASH_RE.fullmatch(str(payload[field])) is None
            ):
                raise RunpodSessionError(f"prior settlement {field} is invalid")
        if payload.get("billing_status") not in {"final", "pending"}:
            raise RunpodSessionError("prior settlement billing status is invalid")
    return payload


def validate_completed_runpod_sessions(
    *,
    sessions_root: str | Path,
    ledger: CostLedger,
) -> list[dict[str, Any]]:
    """Require every prior private session to be stopped and exactly settled."""

    root = Path(sessions_root)
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise RunpodSessionError("RunPod sessions root must be a real directory")
    summaries: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.is_symlink():
            raise RunpodSessionError(f"unexpected private session entry: {directory}")
        if _RAW_HASH_RE.fullmatch(directory.name) is None:
            raise RunpodSessionError(f"private session directory name is invalid: {directory}")
        external_path = directory / EXTERNAL_STOP_RECEIPT_FILENAME
        if external_path.exists():
            try:
                external = load_external_stop_receipt(external_path)
            except RunpodRecoveryError as exc:
                raise RunpodSessionError("prior external-stop receipt is invalid") from exc
            bootstrap_path = directory / GPU_BUDGET_BOOTSTRAP_FILENAME
            if bootstrap_path.exists():
                bootstrap = _validated_bootstrap(bootstrap_path)
            else:
                # A bootstrap failure can leave its authenticated receipt only
                # on the now-stopped remote volume.  The external-stop receipt
                # independently binds these three reservation identities, so
                # it is sufficient for closed-session validation.
                bootstrap = {
                    field: external[field]
                    for field in (
                        "session_hash",
                        "reservation_id",
                        "reservation_record_hash",
                    )
                }
            if bootstrap["session_hash"] != f"sha256:{directory.name}":
                raise RunpodSessionError("private session directory disagrees with session hash")
            settlement = _validated_settlement(
                directory / SETTLEMENT_FILENAME,
                bootstrap=bootstrap,
            )
            for field in ("session_hash", "reservation_id", "reservation_record_hash"):
                if external.get(field) != bootstrap.get(field):
                    raise RunpodSessionError(
                        f"prior external-stop receipt {field} disagrees with bootstrap"
                    )
            if settlement.get("schema_version") != 2:
                raise RunpodSessionError("external-stop session requires settlement v2")
            if settlement.get("external_stop_receipt_hash") != external.get("record_hash"):
                raise RunpodSessionError("prior settlement external-stop receipt hash mismatch")
            if settlement.get("stop_evidence_hash") != external.get("stop_evidence_hash"):
                raise RunpodSessionError("prior settlement stop evidence hash mismatch")
            if settlement.get("billing_evidence_hash") != external.get(
                "billing_evidence_hash"
            ):
                raise RunpodSessionError("prior settlement billing evidence hash mismatch")
            incurred = settlement.get("accounted_gpu_usd")
        else:
            bootstrap = _validated_bootstrap(directory / GPU_BUDGET_BOOTSTRAP_FILENAME)
            if bootstrap["session_hash"] != f"sha256:{directory.name}":
                raise RunpodSessionError("private session directory disagrees with session hash")
            watchdog = _validated_watchdog(directory / WATCHDOG_STATE_FILENAME)
            settlement = _validated_settlement(
                directory / SETTLEMENT_FILENAME,
                bootstrap=bootstrap,
            )
            if settlement.get("watchdog_state_hash") != stable_hash(watchdog):
                raise RunpodSessionError("prior settlement watchdog state hash mismatch")
            incurred = settlement.get("provider_incurred_usd")
        entry = _ledger_entry(ledger, reservation_id=str(bootstrap["reservation_id"]))
        if entry.get("kind") != "gpu" or entry.get("status") != "incurred":
            raise RunpodSessionError("prior GPU reservation is not settled in canonical ledger")
        if abs(float(entry.get("amount_usd")) - float(incurred)) > 1e-6:
            raise RunpodSessionError("prior GPU settlement disagrees with canonical ledger")
        summaries.append(
            {
                "session_hash": bootstrap["session_hash"],
                "reservation_id": bootstrap["reservation_id"],
                "settlement_record_hash": settlement["record_hash"],
                "status": "stopped_confirmed_and_settled",
            }
        )
    return summaries


def prepare_runpod_session_directory(
    *,
    sessions_root: str | Path,
    pending_bootstrap_path: str | Path,
    ledger: CostLedger,
) -> Path:
    """Atomically claim a new private session after all prior phases completed."""

    root = Path(sessions_root)
    pending = Path(pending_bootstrap_path)
    bootstrap = _validated_bootstrap(pending)
    validate_completed_runpod_sessions(sessions_root=root, ledger=ledger)

    session_hash = str(bootstrap["session_hash"])
    session_digest = session_hash.removeprefix("sha256:")
    if _RAW_HASH_RE.fullmatch(session_digest) is None:
        raise RunpodSessionError("current session hash is invalid")
    entry = _ledger_entry(ledger, reservation_id=str(bootstrap["reservation_id"]))
    if entry.get("kind") != "gpu" or entry.get("status") != "estimated":
        raise RunpodSessionError("current GPU reservation is not active in canonical ledger")
    active_gpu_entries = [
        item
        for item in ledger.document()["entries"]
        if item.get("kind") == "gpu" and item.get("status") == "estimated"
    ]
    if (
        len(active_gpu_entries) != 1
        or active_gpu_entries[0].get("entry_id") != bootstrap["reservation_id"]
    ):
        raise RunpodSessionError("current reservation is not the sole active GPU commitment")

    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:  # pragma: no cover
        pass
    target = root / session_digest
    try:
        target.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise RunpodSessionError("GPU phase/session directory has already been claimed") from exc
    destination = target / GPU_BUDGET_BOOTSTRAP_FILENAME
    try:
        os.replace(pending, destination)
        destination.chmod(0o600)
    except BaseException:
        # Leave the claimed but incomplete directory in place. A future launch
        # then fails closed instead of silently reusing this session identity.
        raise
    return target


def validate_active_runpod_session(
    *,
    session_directory: str | Path,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    phase: str,
    session_id: str,
    now: datetime | None = None,
    maximum_watchdog_age_seconds: float = 90,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Authenticate the live private session immediately before GPU backend use."""

    directory = Path(session_directory)
    if not directory.is_dir() or directory.is_symlink():
        raise RunpodSessionError("active RunPod session directory is missing or unsafe")
    if directory.name != reservation.session_hash.removeprefix("sha256:"):
        raise RunpodSessionError("active session directory disagrees with reservation")
    bootstrap = _validated_bootstrap(directory / GPU_BUDGET_BOOTSTRAP_FILENAME)
    if bootstrap.get("phase") != phase:
        raise RunpodSessionError("active session phase disagrees with bootstrap")
    if bootstrap.get("reservation_id") != reservation.reservation_id:
        raise RunpodSessionError("active session reservation disagrees with bootstrap")
    if bootstrap.get("reservation_record_hash") != reservation.manifest()["record_hash"]:
        raise RunpodSessionError("active session receipt hash disagrees with bootstrap")
    validate_existing_gpu_phase_reservation(
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=session_id,
        require_active=True,
    )

    watchdog_path = directory / WATCHDOG_STATE_FILENAME
    _require_regular_private_record(watchdog_path)
    try:
        watchdog = read_json(watchdog_path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("active session watchdog state is unreadable") from exc
    if (
        not isinstance(watchdog, dict)
        or watchdog.get("schema_version") != _WATCHDOG_STATE_SCHEMA_VERSION
    ):
        raise RunpodSessionError("active session watchdog state is malformed")
    if watchdog.get("watchdog_version") != _WATCHDOG_STATE_PROTOCOL_VERSION:
        raise RunpodSessionError("active session watchdog version is unsupported")
    if watchdog.get("status") != "armed":
        raise RunpodSessionError("active session watchdog is not armed")
    if watchdog.get("action") != "stop_only_preserve_volume":
        raise RunpodSessionError("active session watchdog action is unsafe")
    live_metadata = _validated_active_v1_metadata(watchdog)
    raw_updated = watchdog.get("updated_at")
    if not isinstance(raw_updated, str):
        raise RunpodSessionError("active session watchdog timestamp is missing")
    try:
        updated = datetime.fromisoformat(raw_updated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError("active session watchdog timestamp is malformed") from exc
    if updated.tzinfo is None or updated.utcoffset() is None:
        raise RunpodSessionError("active session watchdog timestamp lacks timezone")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age = (current - updated.astimezone(UTC)).total_seconds()
    if age < 0:
        raise RunpodSessionError("active session watchdog state is in the future")
    if age > maximum_watchdog_age_seconds:
        raise RunpodSessionError("active session watchdog state is stale")

    limits = watchdog.get("limits")
    if not isinstance(limits, dict):
        raise RunpodSessionError("active session watchdog limits are missing")
    _require_close(
        limits.get("gpu_hard_stop_usd"),
        reservation.global_gpu_hard_stop_usd,
        field="watchdog GPU hard stop",
    )
    _require_close(
        limits.get("global_safe_budget_usd"),
        reservation.safety_adjusted_gpu_ceiling_usd,
        field="watchdog global safe budget",
    )
    _require_close(
        limits.get("safe_budget_usd"),
        reservation.remaining_safe_gpu_before_phase_usd,
        field="watchdog remaining safe budget",
    )
    _require_close(
        limits.get("safety_margin_fraction"),
        reservation.safety_margin_fraction,
        field="watchdog safety margin",
    )
    _require_close(
        limits.get("maximum_runtime_hours"),
        reservation.maximum_safe_runtime_hours,
        field="watchdog maximum runtime",
    )
    _require_close(
        limits.get("maximum_approved_hourly_total_usd"),
        reservation.live_hourly_total_usd,
        field="watchdog approved hourly total",
    )
    _require_close(
        limits.get("prior_committed_gpu_usd"),
        reservation.prior_committed_gpu_usd,
        field="watchdog prior committed GPU cost",
    )

    deadline = watchdog.get("deadline")
    if not isinstance(deadline, dict):
        raise RunpodSessionError("active session watchdog deadline is missing")
    raw_deadline = deadline.get("effective_deadline")
    if not isinstance(raw_deadline, str):
        raise RunpodSessionError("active session watchdog deadline timestamp is missing")
    try:
        effective_deadline = datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError("active session watchdog deadline timestamp is malformed") from exc
    if effective_deadline.tzinfo is None or effective_deadline.utcoffset() is None:
        raise RunpodSessionError("active session watchdog deadline lacks timezone")
    if effective_deadline.astimezone(UTC) <= current:
        raise RunpodSessionError("active session watchdog deadline has elapsed")
    _finite_number(deadline.get("calculation_hourly_usd"), field="watchdog calculation rate")
    _finite_number(
        deadline.get("incurred_cost_usd"),
        field="watchdog incurred GPU cost",
        allow_zero=True,
    )

    process_identity = validate_watchdog_process_identity(
        directory / WATCHDOG_PID_FILENAME,
        proc_root=proc_root,
    )
    pid = int(process_identity["pid"])

    preflight_path = directory / GPU_PREFLIGHT_FILENAME
    _require_regular_private_record(preflight_path)
    try:
        preflight = read_json(preflight_path)
    except (OSError, ValueError) as exc:
        raise RunpodSessionError("active session GPU preflight is unreadable") from exc
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != 3
        or preflight.get("passed") is not True
    ):
        raise RunpodSessionError("active session GPU preflight did not pass")
    evidence_boundary = preflight.get("evidence_boundary")
    if not isinstance(evidence_boundary, dict):
        raise RunpodSessionError("active session GPU preflight lacks its evidence boundary")
    if evidence_boundary.get("provider_api") != "rest-v1" or evidence_boundary.get(
        "provider_evidence_unavailable"
    ) != list(_RUNPOD_V1_PROVIDER_EVIDENCE_GAPS):
        raise RunpodSessionError("active session GPU preflight provider evidence mismatch")
    local_substitutes = evidence_boundary.get("locally_verified_substitutes")
    if (
        not isinstance(local_substitutes, dict)
        or local_substitutes.get("runtime_gpu_count") != 8
        or local_substitutes.get("runtime_gpu_source") != "nvidia-smi"
        or local_substitutes.get("cuda_forward_compatibility") is not True
        or local_substitutes.get("cuda_source")
        != "local-driver-and-compatibility-libraries"
    ):
        raise RunpodSessionError("active session GPU preflight local evidence is incomplete")
    if evidence_boundary.get("approval_bound_but_not_live_provider_verified") != [
        "global_networking_enabled",
        "interruptible",
        "locked",
    ]:
        raise RunpodSessionError("active session GPU preflight residual evidence gaps mismatch")
    _validate_local_gpu_evidence(
        preflight,
        provider_gpu_id=live_metadata.get("provider_gpu_id"),
    )
    for preflight_field, metadata_field in (
        ("pod_id", "pod_id"),
        ("execution_identity_hash", "execution_identity_hash"),
        ("machine_id_hash", "machine_id_hash"),
        ("direct_ssh_endpoint_hash", "direct_ssh_endpoint_hash"),
        ("provider_gpu_id", "provider_gpu_id"),
        ("data_center_id", "data_center_id"),
        ("container_image_digest", "container_image"),
    ):
        if preflight.get(preflight_field) != live_metadata.get(metadata_field):
            raise RunpodSessionError(
                f"active session GPU preflight {preflight_field} disagrees with watchdog"
            )
    gate = preflight.get("gpu_budget_reservation")
    if not isinstance(gate, dict):
        raise RunpodSessionError("active session GPU preflight lacks budget binding")
    for field in ("reservation_id", "reservation_record_hash", "session_hash", "phase"):
        if gate.get(field) != bootstrap.get(field):
            raise RunpodSessionError(f"active session GPU preflight {field} mismatch")
    _require_close(
        preflight.get("planned_hours"),
        reservation.maximum_safe_runtime_hours,
        field="GPU preflight planned runtime",
    )
    _require_close(
        preflight.get("prior_committed_gpu_cost_usd"),
        reservation.prior_committed_gpu_usd,
        field="GPU preflight prior committed GPU cost",
    )
    _require_close(
        preflight.get("gpu_budget_usd"),
        reservation.global_gpu_hard_stop_usd,
        field="GPU preflight hard stop",
    )
    price = preflight.get("price")
    if not isinstance(price, dict):
        raise RunpodSessionError("active session GPU preflight price binding is missing")
    _require_close(
        price.get("approved_hourly_total_usd"),
        reservation.live_hourly_total_usd,
        field="GPU preflight approved hourly total",
    )
    preflight_watchdog = preflight.get("watchdog")
    if not isinstance(preflight_watchdog, dict):
        raise RunpodSessionError("active session GPU preflight watchdog binding is missing")
    if preflight_watchdog.get("pid") != pid:
        raise RunpodSessionError("active session GPU preflight watchdog PID mismatch")
    if preflight_watchdog.get("process_identity_hash") != process_identity.get("record_hash"):
        raise RunpodSessionError(
            "active session GPU preflight watchdog process identity mismatch"
        )
    state_path = preflight_watchdog.get("state_path")
    if not isinstance(state_path, str) or Path(state_path).resolve() != watchdog_path.resolve():
        raise RunpodSessionError("active session GPU preflight watchdog path mismatch")
    bound_updated = preflight_watchdog.get("state_updated_at")
    if not isinstance(bound_updated, str):
        raise RunpodSessionError("active session GPU preflight watchdog timestamp is missing")
    try:
        parsed_bound_updated = datetime.fromisoformat(bound_updated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunpodSessionError(
            "active session GPU preflight watchdog timestamp is malformed"
        ) from exc
    if (
        parsed_bound_updated.tzinfo is None
        or parsed_bound_updated.utcoffset() is None
        or parsed_bound_updated.astimezone(UTC) > updated.astimezone(UTC)
    ):
        raise RunpodSessionError("active session GPU preflight watchdog timestamp mismatch")

    payload = {
        "schema_version": 1,
        "protocol_version": "active-runpod-session-v1",
        "phase": phase,
        "session_hash": reservation.session_hash,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "watchdog_updated_at": updated.astimezone(UTC).isoformat(),
        "watchdog_process_identity_hash": process_identity["record_hash"],
        "gpu_preflight_hash": stable_hash(preflight),
        "passed": True,
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "GPU_BUDGET_BOOTSTRAP_FILENAME",
    "GPU_PREFLIGHT_FILENAME",
    "SETTLEMENT_FILENAME",
    "WATCHDOG_PID_FILENAME",
    "WATCHDOG_STATE_FILENAME",
    "RunpodSessionError",
    "prepare_runpod_session_directory",
    "record_watchdog_process_identity",
    "validate_active_runpod_session",
    "validate_completed_runpod_sessions",
    "validate_watchdog_process_identity",
]
