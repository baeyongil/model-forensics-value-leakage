"""Stop only the exact research Pod when bootstrap cannot begin safely.

The normal path authenticates the synced lifecycle and reservation. Those are
the files most likely to be unavailable when the pre-bootstrap bundle verifier
fails, so a second path binds the provider-managed ambient Pod id and the
in-memory session nonce to exact live REST-v1 metadata. That fallback never
uses local lifecycle or reservation content and issues no provider write until
all independent evidence agrees.

This module intentionally has standard-library-only top-level imports. It is
invoked with ``python -I -S`` on a fresh provider image.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import stat
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from model_forensics.runpod_contract import (
    CANDIDATE_DATA_CENTER_IDS,
    EXACT_CONTAINER_DISK_GB,
    EXACT_GPU_COUNT,
    EXACT_PORTS,
    EXACT_PROVIDER_GPU_ID,
    EXACT_VOLUME_DISK_GB,
    EXACT_VOLUME_MOUNT_PATH,
    PROVIDER_MANAGED_ENV_KEYS,
    REQUESTED_POD_ENV_KEYS,
    SESSION_ENV_NAME,
    STATIC_POD_ENV,
)

_POD_ID_RE = re.compile(r"[A-Za-z0-9_-]{3,128}\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_DIGEST_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
_MAXIMUM_RECEIPT_BYTES = 2 * 1024 * 1024
_MAXIMUM_PROVIDER_BYTES = 2 * 1024 * 1024
_REST_BASE = "https://rest.runpod.io/v1/pods"
_REST_V1_EXACT_QUERY = "includeMachine=true&includeNetworkVolume=true&includeTemplate=true"
_RESEARCH_PHASES = frozenset(
    {
        "behavior_baseline_gpu",
        "behavior_treatment_gpu",
        "resample_gpu",
        "lens_gpu",
    }
)

# These values identify the already-created, stopped research Pod that the
# approved workflow re-arms. They deliberately do not accept a caller-defined
# alternative, which prevents this emergency path from becoming a general Pod
# stop client.
EXACT_RESEARCH_POD_NAME = "model-forensics-behavior-baseline"
EXACT_RESEARCH_CONTAINER_IMAGE = (
    "runpod/pytorch@sha256:"
    "e855789ff7e4b1ad76698171b1974a99a5c48c5b3e80a908976987938b090992"
)
BOOTSTRAP_PROJECT_ROOT = Path("/workspace/model-forensics-value-leakage")

Transport = Callable[[str, str, str, float], tuple[int, bytes]]


class BootstrapFailureStopError(RuntimeError):
    """The failed bootstrap could not safely stop its exact Pod."""


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapFailureStopError("JSON evidence contains duplicate keys")
        result[key] = value
    return result


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _read_authenticated_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BootstrapFailureStopError("reservation receipt is missing or unsafe")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or before.st_size > _MAXIMUM_RECEIPT_BYTES
    ):
        raise BootstrapFailureStopError("reservation receipt identity is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise BootstrapFailureStopError("reservation changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAXIMUM_RECEIPT_BYTES:
                raise BootstrapFailureStopError("reservation receipt is oversized")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _identity(after) != _identity(opened)
        or _identity(path.lstat()) != _identity(opened)
        or total != opened.st_size
    ):
        raise BootstrapFailureStopError("reservation changed during read")
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapFailureStopError("reservation receipt is unreadable") from exc
    if not isinstance(value, dict):
        raise BootstrapFailureStopError("reservation receipt is malformed")
    unsigned = {key: item for key, item in value.items() if key != "record_hash"}
    if value.get("record_hash") != _stable_hash(unsigned):
        raise BootstrapFailureStopError("reservation receipt hash mismatch")
    return value


def _default_transport(
    method: str,
    url: str,
    api_key: str,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(_MAXIMUM_PROVIDER_BYTES + 1)
            if len(body) > _MAXIMUM_PROVIDER_BYTES:
                raise BootstrapFailureStopError("RunPod emergency response is oversized")
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BootstrapFailureStopError("RunPod emergency stop transport failed") from exc


def _provider_payload(
    *,
    pod_id: str,
    api_key: str,
    transport: Transport,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    status, raw = transport(
        "GET",
        f"{_REST_BASE}/{pod_id}?{_REST_V1_EXACT_QUERY}",
        api_key,
        timeout_seconds,
    )
    if status < 200 or status >= 300:
        raise BootstrapFailureStopError("RunPod emergency status check failed")
    if len(raw) > _MAXIMUM_PROVIDER_BYTES:
        raise BootstrapFailureStopError("RunPod emergency status response is oversized")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapFailureStopError("RunPod emergency status response is malformed") from exc
    if not isinstance(payload, Mapping):
        raise BootstrapFailureStopError("RunPod emergency status response is malformed")
    return payload


def _basic_provider_status(payload: Mapping[str, Any], *, pod_id: str) -> str:
    if payload.get("id") != pod_id:
        raise BootstrapFailureStopError("RunPod emergency status identity disagrees")
    desired = payload.get("desiredStatus")
    if desired not in {"RUNNING", "EXITED"}:
        raise BootstrapFailureStopError("RunPod emergency status is unsupported")
    return str(desired)


def _exact_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise BootstrapFailureStopError(f"RunPod provider {label} is malformed")
    parsed = int(value)
    if parsed <= 0:
        raise BootstrapFailureStopError(f"RunPod provider {label} is malformed")
    return parsed


def _network_volume_absent(payload: Mapping[str, Any]) -> bool:
    aliases = (
        payload.get("networkVolume"),
        payload.get("networkVolumeId"),
        payload.get("networkVolumeID"),
        payload.get("network_volume_id"),
    )
    return all(value is None for value in aliases)


def _global_networking_disabled(value: Any) -> bool:
    if value is False or value is None:
        return True
    return isinstance(value, Mapping) and value.get("enabled") is False


def _validate_provider_environment(value: Any, *, session_nonce: str) -> None:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise BootstrapFailureStopError("RunPod provider environment is malformed")
    allowed = REQUESTED_POD_ENV_KEYS | PROVIDER_MANAGED_ENV_KEYS
    if set(value) - allowed or not REQUESTED_POD_ENV_KEYS.issubset(value):
        raise BootstrapFailureStopError("RunPod provider environment identity disagrees")
    if not value.get("HF_TOKEN"):
        raise BootstrapFailureStopError("RunPod provider environment identity disagrees")
    if not hmac.compare_digest(str(value.get(SESSION_ENV_NAME, "")), session_nonce):
        raise BootstrapFailureStopError("RunPod provider session identity disagrees")
    if any(value.get(key) != expected for key, expected in STATIC_POD_ENV.items()):
        raise BootstrapFailureStopError("RunPod provider environment identity disagrees")
    public_key = value.get("PUBLIC_KEY")
    if public_key is not None and (not public_key.strip() or len(public_key) > 65536):
        raise BootstrapFailureStopError("RunPod provider environment identity disagrees")


def _validate_direct_endpoint(payload: Mapping[str, Any], *, required: bool) -> None:
    public_ip = payload.get("publicIp")
    mappings = payload.get("portMappings")
    if not required and public_ip in (None, "") and mappings in (None, {}):
        return
    if not isinstance(public_ip, str):
        raise BootstrapFailureStopError("RunPod provider endpoint binding is malformed")
    try:
        address = ipaddress.ip_address(public_ip)
    except ValueError as exc:
        raise BootstrapFailureStopError("RunPod provider endpoint binding is malformed") from exc
    if address.version != 4 or not isinstance(mappings, Mapping) or set(mappings) != {"22"}:
        raise BootstrapFailureStopError("RunPod provider endpoint binding is malformed")
    port = mappings.get("22")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise BootstrapFailureStopError("RunPod provider endpoint binding is malformed")


def _independent_provider_status(
    payload: Mapping[str, Any],
    *,
    pod_id: str,
    session_nonce: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: frozenset[str],
    expected_container_image: str,
) -> str:
    """Authenticate exact-current-Pod evidence without reading private files."""

    desired = _basic_provider_status(payload, pod_id=pod_id)
    if payload.get("name") != EXACT_RESEARCH_POD_NAME:
        raise BootstrapFailureStopError("RunPod provider research Pod identity disagrees")
    observed_image = payload.get("imageName")
    if observed_image != expected_container_image:
        raise BootstrapFailureStopError("RunPod provider image binding disagrees")
    image_alias = payload.get("image")
    if image_alias is not None and image_alias != observed_image:
        raise BootstrapFailureStopError("RunPod provider image aliases disagree")

    machine = payload.get("machine")
    if not isinstance(machine, Mapping):
        raise BootstrapFailureStopError("RunPod provider GPU metadata is malformed")
    if (
        _exact_positive_integer(payload.get("gpuCount"), label="GPU count")
        != EXACT_GPU_COUNT
        or machine.get("gpuTypeId") != expected_provider_gpu_id
        or machine.get("secureCloud") is not True
        or machine.get("dataCenterId") not in allowed_data_center_ids
    ):
        raise BootstrapFailureStopError("RunPod provider GPU binding disagrees")
    gpu = payload.get("gpu")
    if gpu is not None:
        if not isinstance(gpu, Mapping):
            raise BootstrapFailureStopError("RunPod provider GPU metadata is malformed")
        if gpu.get("id") not in (None, expected_provider_gpu_id) or gpu.get("count") not in (
            None,
            EXACT_GPU_COUNT,
        ):
            raise BootstrapFailureStopError("RunPod provider GPU aliases disagree")
    machine_id = payload.get("machineId")
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise BootstrapFailureStopError("RunPod provider machine identity is malformed")

    if (
        _exact_positive_integer(payload.get("containerDiskInGb"), label="container disk")
        != EXACT_CONTAINER_DISK_GB
        or _exact_positive_integer(payload.get("volumeInGb"), label="persistent disk")
        != EXACT_VOLUME_DISK_GB
        or payload.get("volumeMountPath") != EXACT_VOLUME_MOUNT_PATH
        or payload.get("ports") != list(EXACT_PORTS)
        or not _network_volume_absent(payload)
        or not _global_networking_disabled(payload.get("globalNetworking"))
    ):
        raise BootstrapFailureStopError("RunPod provider storage or networking binding disagrees")
    for optional_false in ("locked", "interruptible"):
        if optional_false in payload and payload[optional_false] is not False:
            raise BootstrapFailureStopError("RunPod provider rental binding disagrees")

    _validate_provider_environment(payload.get("env"), session_nonce=session_nonce)
    _validate_direct_endpoint(payload, required=desired == "RUNNING")
    return desired


def _authenticate_local_binding(
    *,
    lifecycle_path: Path,
    receipt_path: Path,
    phase: str,
    pod_id: str,
    session_nonce: str,
) -> None:
    """Authenticate the preferred lifecycle+reservation path lazily."""

    # Keep this import off the independent fallback path. Syntax/import/data
    # failures in the synced lifecycle reader are treated exactly like corrupt
    # lifecycle evidence and cannot prevent provider-bound emergency stopping.
    from model_forensics.runpod_lifecycle_state import (
        authorization_from_state,
        load_lifecycle_state,
    )

    lifecycle = load_lifecycle_state(lifecycle_path)
    authorization = authorization_from_state(lifecycle)
    pod = lifecycle.get("pod")
    receipt = _read_authenticated_receipt(receipt_path)
    expected_session_hash = _stable_hash({"opaque_gpu_session_id": session_nonce})
    if (
        lifecycle.get("operation") not in {"created", "rearmed"}
        or not isinstance(pod, Mapping)
        or pod.get("id") != pod_id
        or pod.get("status") != "RUNNING"
        or authorization.phase != phase
        or not hmac.compare_digest(authorization.session_hash, expected_session_hash)
        or receipt.get("phase") != phase
        or not hmac.compare_digest(str(receipt.get("session_hash", "")), expected_session_hash)
        or receipt.get("reservation_id") != authorization.reservation_id
        or receipt.get("record_hash") != authorization.reservation_record_hash
        or _HASH_RE.fullmatch(str(receipt.get("record_hash"))) is None
    ):
        raise BootstrapFailureStopError("bootstrap failure Pod binding disagrees")


def _validated_contract(
    *,
    phase: str,
    pod_id: str,
    api_key: str,
    session_nonce: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: Sequence[str],
    expected_container_image: str,
    confirmation_attempts: int,
    timeout_seconds: float,
) -> frozenset[str]:
    if (
        phase not in _RESEARCH_PHASES
        or _POD_ID_RE.fullmatch(pod_id) is None
        or not api_key
        or api_key != api_key.strip()
        or len(api_key) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in api_key)
        or not session_nonce
        or session_nonce != session_nonce.strip()
        or len(session_nonce) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in session_nonce)
        or isinstance(confirmation_attempts, bool)
        or confirmation_attempts <= 0
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 60
        or expected_provider_gpu_id != EXACT_PROVIDER_GPU_ID
        or expected_container_image != EXACT_RESEARCH_CONTAINER_IMAGE
        or _CONTAINER_DIGEST_RE.fullmatch(expected_container_image) is None
    ):
        raise BootstrapFailureStopError("bootstrap failure stop inputs are invalid")
    centers = tuple(allowed_data_center_ids)
    if (
        not centers
        or len(centers) != len(set(centers))
        or not all(isinstance(value, str) and value in CANDIDATE_DATA_CENTER_IDS for value in centers)
    ):
        raise BootstrapFailureStopError("bootstrap failure data-center binding is invalid")
    return frozenset(centers)


def stop_after_bootstrap_failure(
    *,
    project_root: str | Path,
    phase: str,
    reservation_path: str | Path,
    pod_id: str,
    api_key: str,
    session_nonce: str,
    expected_provider_gpu_id: str,
    allowed_data_center_ids: Sequence[str],
    expected_container_image: str,
    transport: Transport = _default_transport,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 15,
    confirmation_attempts: int = 12,
) -> dict[str, Any]:
    """Stop the locally authenticated Pod or its exact provider-bound fallback."""

    root = Path(project_root).resolve(strict=True)
    if root != BOOTSTRAP_PROJECT_ROOT:
        raise BootstrapFailureStopError("bootstrap failure stop inputs are invalid")
    centers = _validated_contract(
        phase=phase,
        pod_id=pod_id,
        api_key=api_key,
        session_nonce=session_nonce,
        expected_provider_gpu_id=expected_provider_gpu_id,
        allowed_data_center_ids=allowed_data_center_ids,
        expected_container_image=expected_container_image,
        confirmation_attempts=confirmation_attempts,
        timeout_seconds=timeout_seconds,
    )
    lifecycle_path = root / ".runpod" / "pod_lifecycle.json"
    expected_receipt = root / ".runpod" / "reservations" / f"{phase}.json"
    try:
        receipt_path = Path(reservation_path).resolve(strict=False)
    except OSError as exc:
        raise BootstrapFailureStopError("bootstrap failure reservation path is unsafe") from exc
    if receipt_path != expected_receipt:
        raise BootstrapFailureStopError("bootstrap failure reservation path is noncanonical")

    authentication = "local_lifecycle_reservation"
    try:
        _authenticate_local_binding(
            lifecycle_path=lifecycle_path,
            receipt_path=receipt_path,
            phase=phase,
            pod_id=pod_id,
            session_nonce=session_nonce,
        )
    except Exception:
        # The fallback trusts none of the failed local evidence. A provider GET
        # remains read-only, and a POST is unreachable until every independent
        # research-Pod binding below agrees.
        authentication = "independent_provider_evidence"

    def current_status() -> str:
        payload = _provider_payload(
            pod_id=pod_id,
            api_key=api_key,
            transport=transport,
            timeout_seconds=timeout_seconds,
        )
        # Local authentication remains the preferred provenance label, but it
        # never substitutes for current provider evidence.  Every path reaches
        # the same full immutable/session/endpoint gate before a stop POST.
        return _independent_provider_status(
            payload,
            pod_id=pod_id,
            session_nonce=session_nonce,
            expected_provider_gpu_id=expected_provider_gpu_id,
            allowed_data_center_ids=centers,
            expected_container_image=expected_container_image,
        )

    desired = current_status()
    if desired != "EXITED":
        status, _raw = transport(
            "POST",
            f"{_REST_BASE}/{pod_id}/stop",
            api_key,
            timeout_seconds,
        )
        if status < 200 or status >= 300:
            raise BootstrapFailureStopError("RunPod emergency stop request failed")
    for attempt in range(confirmation_attempts):
        if current_status() == "EXITED":
            return {
                "schema_version": 2,
                "status": "stop_confirmed",
                "authentication": authentication,
                "pod_id_hash": _stable_hash({"runpod_pod_id": pod_id}),
                "session_hash": _stable_hash({"opaque_gpu_session_id": session_nonce}),
            }
        if attempt + 1 < confirmation_attempts:
            sleep(2)
    raise BootstrapFailureStopError("RunPod emergency stop was not confirmed EXITED")


__all__ = [
    "BOOTSTRAP_PROJECT_ROOT",
    "EXACT_RESEARCH_CONTAINER_IMAGE",
    "EXACT_RESEARCH_POD_NAME",
    "BootstrapFailureStopError",
    "stop_after_bootstrap_failure",
]
