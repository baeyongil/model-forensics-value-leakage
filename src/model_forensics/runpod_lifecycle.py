"""Fail-closed RunPod Pod creation and stopped-Pod re-arming.

This module is deliberately narrower than a general RunPod client.  It can
list/get Pods, create one exact approved Pod, replace the approved environment
on a stopped Pod, and start that same Pod.  It has no delete, terminate, stop,
restart, or arbitrary update primitive.

Provider responses can contain environment secrets.  They are validated in
memory and reduced to a small private lifecycle record before persistence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from model_forensics.approval import (
    ApprovalBindings,
    PaidRunApproval,
    validate_paid_run_approval,
)
from model_forensics.budget import CostLedger
from model_forensics.gpu_budget import (
    GpuBudgetGateError,
    GpuPhaseBudgetReservation,
    validate_gpu_phase_bootstrap,
)
from model_forensics.io import stable_hash

RUNPOD_V2_BASE = "https://api.runpod.io/v2"
RUNPOD_V1_BASE = "https://rest.runpod.io/v1"
RUNPOD_V2_PODS_URL = f"{RUNPOD_V2_BASE}/pods"
RUNPOD_V1_PODS_URL = f"{RUNPOD_V1_BASE}/pods"

LIFECYCLE_PROTOCOL = "runpod-pod-lifecycle-v1"
LIFECYCLE_STATE_FILENAME = "pod_lifecycle.json"
EXACT_PROVIDER_GPU_ID = "NVIDIA H100 80GB HBM3"
EXACT_GPU_FAMILY = "H100_80GB"
EXACT_GPU_COUNT = 8
EXACT_CLOUD = "SECURE"
EXACT_CUDA_VERSIONS = ("12.8",)
CANDIDATE_DATA_CENTER_IDS = frozenset({"CA-MTL-1", "EUR-IS-3"})
EXACT_CONTAINER_DISK_GB = 50
EXACT_VOLUME_DISK_GB = 650
EXACT_VOLUME_MOUNT_PATH = "/workspace"
EXACT_PORTS = ("22/tcp",)

SESSION_ENV_NAME = "GPU_BUDGET_SESSION_ID"
HF_TOKEN_ENV_NAME = "HF_TOKEN"
STATIC_POD_ENV = {
    "HF_HOME": "/workspace/.cache/huggingface",
    "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
    "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
    "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
}
REQUESTED_POD_ENV_KEYS = frozenset({HF_TOKEN_ENV_NAME, SESSION_ENV_NAME, *STATIC_POD_ENV})
PROVIDER_MANAGED_ENV_KEYS = frozenset({"PUBLIC_KEY"})
TERMINAL_POD_STATUSES = frozenset({"EXITED", "TERMINATED"})
_POD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{5,127}\Z")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EXISTING_POD_ID_HASH_RE = re.compile(r"runpod-pod-id-sha256:[0-9a-f]{64}\Z")
_MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024


class RunpodLifecycleError(RuntimeError):
    """A lifecycle operation cannot proceed without weakening a safety invariant."""


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: bytes


HttpTransport = Callable[..., HttpResult]
Sleep = Callable[[float], None]
Monotonic = Callable[[], float]


@dataclass(frozen=True, slots=True)
class LifecycleAuthorization:
    phase: str
    reservation_id: str
    reservation_record_hash: str
    session_hash: str
    approval_hash: str
    bindings_hash: str
    gpu_lock_hash: str
    quote_hash: str
    immutable_spec: Mapping[str, Any]
    immutable_spec_hash: str
    approved_runtime_hours: float
    approved_phase_maximum_usd: float
    live_hourly_total_usd: float

    def manifest(
        self,
        *,
        launch_spec_hash: str,
        acknowledged_existing_pod_id_hashes: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "reservation_id": self.reservation_id,
            "reservation_record_hash": self.reservation_record_hash,
            "session_hash": self.session_hash,
            "approval_hash": self.approval_hash,
            "bindings_hash": self.bindings_hash,
            "gpu_lock_hash": self.gpu_lock_hash,
            "quote_hash": self.quote_hash,
            "immutable_spec_hash": self.immutable_spec_hash,
            "launch_spec_hash": launch_spec_hash,
            "acknowledged_existing_pod_id_hashes": list(
                acknowledged_existing_pod_id_hashes
            ),
            "approved_runtime_hours": self.approved_runtime_hours,
            "approved_phase_maximum_usd": self.approved_phase_maximum_usd,
            "live_hourly_total_usd": self.live_hourly_total_usd,
        }


def _utc_timestamp(now: datetime | None = None) -> str:
    value = now or datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("lifecycle timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_opaque_secret(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RunpodLifecycleError(f"{label} is missing or malformed")
    return value


def _ceil_usd(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def _exact_runtime(bindings: ApprovalBindings, phase: str) -> float:
    matches = [
        allocation.maximum_runtime_hours
        for allocation in bindings.gpu.phase_runtime_allocations
        if allocation.command_phase == phase
    ]
    if len(matches) != 1:
        raise RunpodLifecycleError("GPU phase has no unique approved runtime allocation")
    return float(matches[0])


def _immutable_spec(bindings: ApprovalBindings) -> dict[str, Any]:
    gpu = bindings.gpu
    spec = {
        "image": gpu.container_image_digest,
        "gpu": {"id": gpu.provider_gpu_id, "count": gpu.count},
        "cloud": gpu.cloud_type,
        "allowed_cuda_versions": list(gpu.allowed_cuda_versions),
        "data_center_ids": list(gpu.data_center_ids),
        "disk": gpu.container_disk_gb,
        "mounts": {
            "persistent": {
                "size": gpu.volume_disk_gb,
                "path": EXACT_VOLUME_MOUNT_PATH,
            }
        },
        "ports": list(EXACT_PORTS),
        "global_networking": False,
        "start_jupyter": False,
        "start_ssh": True,
        "network_volume": None,
    }
    expected = {
        "family": EXACT_GPU_FAMILY,
        "provider_gpu_id": EXACT_PROVIDER_GPU_ID,
        "cloud_type": EXACT_CLOUD,
        "allowed_cuda_versions": EXACT_CUDA_VERSIONS,
        "count": EXACT_GPU_COUNT,
        "container_disk_gb": EXACT_CONTAINER_DISK_GB,
        "volume_disk_gb": EXACT_VOLUME_DISK_GB,
    }
    observed = {
        "family": gpu.family,
        "provider_gpu_id": gpu.provider_gpu_id,
        "cloud_type": gpu.cloud_type,
        "allowed_cuda_versions": tuple(gpu.allowed_cuda_versions),
        "count": gpu.count,
        "container_disk_gb": gpu.container_disk_gb,
        "volume_disk_gb": gpu.volume_disk_gb,
    }
    if observed != expected:
        raise RunpodLifecycleError("approval does not bind the exact 8x H100 SXM Secure launch")
    approved_data_centers = tuple(gpu.data_center_ids)
    if (
        not approved_data_centers
        or len(set(approved_data_centers)) != len(approved_data_centers)
        or not set(approved_data_centers).issubset(CANDIDATE_DATA_CENTER_IDS)
    ):
        raise RunpodLifecycleError(
            "approval-bound data centers must be a nonempty subset of the frozen candidates"
        )
    return spec


def authorize_gpu_lifecycle(
    *,
    approval: PaidRunApproval,
    expected_bindings: ApprovalBindings,
    reservation: GpuPhaseBudgetReservation,
    ledger: CostLedger,
    phase: str,
    session_nonce: str,
    now: datetime | None = None,
) -> LifecycleAuthorization:
    """Authenticate approval, frozen quote/spec, and one active reservation."""

    nonce = _validate_opaque_secret(session_nonce, label="GPU session nonce")
    validate_paid_run_approval(
        approval,
        expected=expected_bindings,
        command_phase=phase,
        now=now,
    )
    spec = _immutable_spec(expected_bindings)
    runtime = _exact_runtime(expected_bindings, phase)
    gpu = expected_bindings.gpu
    live_rate = (
        gpu.count * gpu.quote.usd_per_gpu_hour + gpu.quote.running_storage_usd_per_hour
    )
    expected_maximum = _ceil_usd(live_rate * runtime)
    validate_gpu_phase_bootstrap(
        ledger=ledger,
        reservation=reservation,
        phase=phase,
        session_id=nonce,
        expected_approved_runtime_hours=runtime,
        expected_live_hourly_total_usd=live_rate,
    )
    if abs(reservation.approved_phase_maximum_usd - expected_maximum) > 1e-6:
        raise GpuBudgetGateError(
            "GPU reservation maximum omits or disagrees with the all-in compute/storage quote"
        )
    reservation_manifest = reservation.manifest()
    return LifecycleAuthorization(
        phase=phase,
        reservation_id=reservation.reservation_id,
        reservation_record_hash=str(reservation_manifest["record_hash"]),
        session_hash=reservation.session_hash,
        approval_hash=approval.content_hash,
        bindings_hash=stable_hash(expected_bindings.model_dump(mode="json")),
        gpu_lock_hash=expected_bindings.gpu_lock_hash,
        quote_hash=expected_bindings.gpu.quote.content_hash,
        immutable_spec=spec,
        immutable_spec_hash=stable_hash(spec),
        approved_runtime_hours=runtime,
        approved_phase_maximum_usd=expected_maximum,
        live_hourly_total_usd=live_rate,
    )


def pod_environment(*, hf_token: str, session_nonce: str) -> dict[str, str]:
    """Return the only caller-supplied environment allowed in a Pod request."""

    token = _validate_opaque_secret(hf_token, label="Hugging Face token")
    nonce = _validate_opaque_secret(session_nonce, label="GPU session nonce")
    environment = {
        HF_TOKEN_ENV_NAME: token,
        SESSION_ENV_NAME: nonce,
        **STATIC_POD_ENV,
    }
    if set(environment) != REQUESTED_POD_ENV_KEYS:
        raise RunpodLifecycleError("internal Pod environment allow-list drifted")
    return environment


def build_create_payload(
    *,
    authorization: LifecycleAuthorization,
    name: str,
    hf_token: str,
    session_nonce: str,
    acknowledged_existing_pod_id_hashes: Sequence[str] = (),
) -> tuple[dict[str, Any], str]:
    """Build the exact v2 payload and its secret-safe launch-spec hash."""

    if _NAME_RE.fullmatch(name) is None:
        raise RunpodLifecycleError("Pod name is malformed")
    acknowledged_hashes = _canonical_existing_pod_id_hashes(
        acknowledged_existing_pod_id_hashes
    )
    environment = pod_environment(hf_token=hf_token, session_nonce=session_nonce)
    spec = authorization.immutable_spec
    payload = {
        "name": name,
        "image": spec["image"],
        "disk": spec["disk"],
        "ports": list(spec["ports"]),
        "env": environment,
        "cloud": spec["cloud"],
        "gpu": {
            **dict(spec["gpu"]),
            "allowedCudaVersions": list(spec["allowed_cuda_versions"]),
        },
        "dataCenterIds": list(spec["data_center_ids"]),
        "globalNetworking": False,
        "mounts": deepcopy(spec["mounts"]),
        "startJupyter": False,
        "startSsh": True,
    }
    secret_safe_payload = deepcopy(payload)
    secret_safe_payload["env"][HF_TOKEN_ENV_NAME] = "present-redacted"
    secret_safe_payload["env"][SESSION_ENV_NAME] = authorization.session_hash
    launch_intent = {
        "create_payload": secret_safe_payload,
        "acknowledged_existing_pod_id_hashes": list(acknowledged_hashes),
    }
    return payload, stable_hash(launch_intent)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunpodLifecycleError("RunPod response contains a duplicate JSON key")
        result[key] = value
    return result


def _decode_json_response(result: HttpResult, *, expected_status: int) -> Any:
    if result.status_code != expected_status:
        raise RunpodLifecycleError(f"RunPod request failed with HTTP {result.status_code}")
    if len(result.body) > _MAX_HTTP_BODY_BYTES:
        raise RunpodLifecycleError("RunPod response exceeds the safe size limit")
    try:
        return json.loads(result.body.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodLifecycleError("RunPod returned malformed JSON") from exc


def urllib_http_transport(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float,
) -> HttpResult:
    """Minimal default transport; HTTP error bodies are deliberately discarded."""

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=dict(headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read(_MAX_HTTP_BODY_BYTES + 1)
            return HttpResult(status_code=int(response.status), body=response_body)
    except urllib.error.HTTPError as exc:
        # Provider error bodies can echo request fields or environment values.
        raise RunpodLifecycleError(f"RunPod request failed with HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RunpodLifecycleError("RunPod request outcome is uncertain") from exc


class RunpodLifecycleClient:
    """A capability-limited client with no destructive provider operation."""

    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport = urllib_http_transport,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = _validate_opaque_secret(api_key, label="RunPod API key")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("HTTP timeout must be in (0, 60]")
        self._transport = transport
        self._timeout_seconds = float(timeout_seconds)

    def _request(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        expected_status: int,
    ) -> Any:
        if method not in {"GET", "POST", "PATCH"}:
            raise RunpodLifecycleError("destructive or unsupported HTTP method refused")
        encoded = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "model-forensics-runpod-lifecycle/1",
        }
        if payload is not None:
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        result = self._transport(
            method=method,
            url=url,
            headers=headers,
            body=encoded,
            timeout_seconds=self._timeout_seconds,
        )
        if not isinstance(result, HttpResult):
            raise RunpodLifecycleError("injected HTTP transport returned an invalid result")
        return _decode_json_response(result, expected_status=expected_status)

    def list_pods_v1(self) -> list[Mapping[str, Any]]:
        value = self._request(
            method="GET",
            url=RUNPOD_V1_PODS_URL,
            payload=None,
            expected_status=200,
        )
        if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
            raise RunpodLifecycleError("RunPod v1 Pod list has an unexpected schema")
        return list(value)

    def get_pod_v1(self, pod_id: str) -> Mapping[str, Any]:
        _require_pod_id(pod_id)
        value = self._request(
            method="GET",
            url=f"{RUNPOD_V1_PODS_URL}/{pod_id}",
            payload=None,
            expected_status=200,
        )
        if not isinstance(value, Mapping):
            raise RunpodLifecycleError("RunPod v1 Pod status has an unexpected schema")
        return value

    def create_pod_v2(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        value = self._request(
            method="POST",
            url=RUNPOD_V2_PODS_URL,
            payload=payload,
            expected_status=201,
        )
        if not isinstance(value, Mapping):
            raise RunpodLifecycleError("RunPod v2 create response has an unexpected schema")
        return value

    def patch_environment_v2(
        self,
        *,
        pod_id: str,
        environment: Mapping[str, str],
    ) -> Mapping[str, Any]:
        _require_pod_id(pod_id)
        allowed = REQUESTED_POD_ENV_KEYS | PROVIDER_MANAGED_ENV_KEYS
        if set(environment) - allowed or not REQUESTED_POD_ENV_KEYS.issubset(environment):
            raise RunpodLifecycleError("Pod PATCH environment violates the allow-list")
        value = self._request(
            method="PATCH",
            url=f"{RUNPOD_V2_PODS_URL}/{pod_id}",
            payload={"env": dict(environment)},
            expected_status=200,
        )
        if not isinstance(value, Mapping):
            raise RunpodLifecycleError("RunPod v2 PATCH response has an unexpected schema")
        return value

    def start_pod_v2(self, *, pod_id: str) -> Mapping[str, Any]:
        _require_pod_id(pod_id)
        value = self._request(
            method="POST",
            url=f"{RUNPOD_V2_PODS_URL}/{pod_id}/action",
            payload={"action": "start"},
            expected_status=200,
        )
        if not isinstance(value, Mapping):
            raise RunpodLifecycleError("RunPod v2 start response has an unexpected schema")
        return value


def _require_pod_id(value: Any) -> str:
    if not isinstance(value, str) or _POD_ID_RE.fullmatch(value) is None:
        raise RunpodLifecycleError("RunPod Pod ID is malformed")
    return value


def existing_pod_id_hash(pod_id: str) -> str:
    """Hash one exact provider Pod ID without exposing it in an allow-list.

    This intentionally hashes the raw UTF-8 Pod ID rather than its JSON
    representation.  The output namespace makes the scheme unambiguous and
    keeps it distinct from hashes of structured lifecycle records.
    """

    value = _require_pod_id(pod_id)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"runpod-pod-id-sha256:{digest}"


def _canonical_existing_pod_id_hashes(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise RunpodLifecycleError("existing Pod allow-list must be a sequence of hashes")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or _EXISTING_POD_ID_HASH_RE.fullmatch(value) is None:
            raise RunpodLifecycleError(
                "existing Pod allow-list hash must use runpod-pod-id-sha256:<64 lowercase hex>"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise RunpodLifecycleError("existing Pod allow-list contains a duplicate hash")
    return tuple(sorted(normalized))


def _require_exact_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunpodLifecycleError(f"RunPod {label} is missing or malformed")
    return value


def _require_exact_environment(
    value: Any,
    *,
    expected: Mapping[str, str],
    expected_session_hash: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RunpodLifecycleError("RunPod Pod environment is missing or malformed")
    observed = dict(value)
    allowed = REQUESTED_POD_ENV_KEYS | PROVIDER_MANAGED_ENV_KEYS
    if set(observed) - allowed or not REQUESTED_POD_ENV_KEYS.issubset(observed):
        raise RunpodLifecycleError("RunPod Pod environment violates the allow-list")
    if observed[HF_TOKEN_ENV_NAME] != expected[HF_TOKEN_ENV_NAME]:
        raise RunpodLifecycleError("RunPod Pod Hugging Face token drifted")
    for key, expected_value in STATIC_POD_ENV.items():
        if observed.get(key) != expected_value:
            raise RunpodLifecycleError("RunPod Pod cache/compatibility environment drifted")
    if stable_hash({"opaque_gpu_session_id": observed[SESSION_ENV_NAME]}) != expected_session_hash:
        raise RunpodLifecycleError("RunPod Pod GPU session nonce drifted")
    public_key = observed.get("PUBLIC_KEY")
    if public_key is not None and (not public_key.strip() or len(public_key) > 65536):
        raise RunpodLifecycleError("RunPod provider-managed SSH key environment is malformed")
    return observed


def _validate_cost(raw: Mapping[str, Any], authorization: LifecycleAuthorization) -> None:
    value = raw.get("costPerHr", raw.get("cost"))
    if value is None:
        raise RunpodLifecycleError("RunPod Pod hourly cost is unavailable")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, str))
    ):
        raise RunpodLifecycleError("RunPod Pod hourly cost is malformed")
    try:
        observed = float(value)
    except ValueError as exc:
        raise RunpodLifecycleError("RunPod Pod hourly cost is malformed") from exc
    if not math.isfinite(observed) or observed < 0:
        raise RunpodLifecycleError("RunPod Pod hourly cost is malformed")
    # Provider status may exclude the separately quoted running-storage charge,
    # but it must never exceed the all-in reservation rate.
    if observed > authorization.live_hourly_total_usd + 1e-6:
        raise RunpodLifecycleError("RunPod Pod hourly cost exceeds the approved quote")


def _sanitize_ssh(value: Any) -> dict[str, Any]:
    ssh = _require_exact_mapping(value, label="SSH details")
    result: dict[str, Any] = {}
    for kind in ("proxy", "direct"):
        endpoint = ssh.get(kind)
        if endpoint is None:
            result[kind] = None
            continue
        mapping = _require_exact_mapping(endpoint, label=f"SSH {kind} endpoint")
        required = {"host", "port", "username", "command"}
        if not required.issubset(mapping):
            raise RunpodLifecycleError("RunPod SSH endpoint is incomplete")
        if not all(isinstance(mapping[key], str) for key in ("host", "username", "command")):
            raise RunpodLifecycleError("RunPod SSH endpoint is malformed")
        port = mapping["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise RunpodLifecycleError("RunPod SSH endpoint is malformed")
        result[kind] = {key: mapping[key] for key in ("host", "port", "username", "command")}
    return result


def _validate_v2_pod(
    raw: Mapping[str, Any],
    *,
    authorization: LifecycleAuthorization,
    expected_name: str | None,
    expected_environment: Mapping[str, str],
    expected_session_hash: str,
    allowed_statuses: frozenset[str],
) -> dict[str, Any]:
    pod_id = _require_pod_id(raw.get("id"))
    status = raw.get("status")
    if status not in allowed_statuses:
        raise RunpodLifecycleError("RunPod v2 Pod status is unexpected")
    if expected_name is not None and raw.get("name") != expected_name:
        raise RunpodLifecycleError("RunPod Pod name drifted")
    spec = authorization.immutable_spec
    if raw.get("image") != spec["image"] or raw.get("disk") != spec["disk"]:
        raise RunpodLifecycleError("RunPod Pod image or container disk drifted")
    if raw.get("ports") != list(EXACT_PORTS):
        raise RunpodLifecycleError("RunPod Pod ports drifted")
    gpu = _require_exact_mapping(raw.get("gpu"), label="GPU metadata")
    if gpu.get("id") != EXACT_PROVIDER_GPU_ID or gpu.get("count") != EXACT_GPU_COUNT:
        raise RunpodLifecycleError("RunPod Pod GPU identity/count drifted")
    if raw.get("cloud") != EXACT_CLOUD:
        raise RunpodLifecycleError("RunPod Pod cloud tier drifted")
    data_center = raw.get("dataCenterId")
    cuda_version = raw.get("cudaVersion")
    provisioning = status == "PROVISIONING"
    if data_center is not None and data_center not in set(spec["data_center_ids"]):
        raise RunpodLifecycleError("RunPod Pod data center is not approved")
    if data_center is None and not provisioning:
        raise RunpodLifecycleError("RunPod Pod data center is unavailable")
    if cuda_version is not None and cuda_version not in EXACT_CUDA_VERSIONS:
        raise RunpodLifecycleError("RunPod Pod CUDA version is not approved")
    if cuda_version is None and not provisioning:
        raise RunpodLifecycleError("RunPod Pod CUDA version is unavailable")
    if raw.get("mounts") != spec["mounts"]:
        raise RunpodLifecycleError("RunPod Pod host-local storage drifted")
    global_networking = _require_exact_mapping(
        raw.get("globalNetworking"), label="global networking metadata"
    )
    if global_networking.get("enabled") is not False:
        raise RunpodLifecycleError("RunPod global networking must remain disabled")
    if raw.get("locked") is not False:
        raise RunpodLifecycleError("RunPod Pod lock state drifted")
    _validate_cost(raw, authorization)
    _require_exact_environment(
        raw.get("env"),
        expected=expected_environment,
        expected_session_hash=expected_session_hash,
    )
    return {
        "id": pod_id,
        "name": raw.get("name"),
        "status": status,
        "image": raw.get("image"),
        "gpu": {"id": gpu.get("id"), "count": gpu.get("count")},
        "cloud": raw.get("cloud"),
        "data_center_id": data_center,
        "cuda_version": cuda_version,
        "disk": raw.get("disk"),
        "mounts": deepcopy(raw.get("mounts")),
        "ports": list(raw.get("ports", [])),
        "global_networking": False,
        "ssh": _sanitize_ssh(raw.get("ssh")),
    }


def _v1_global_networking_disabled(raw: Mapping[str, Any]) -> bool:
    value = raw.get("globalNetworking")
    if value is False:
        return True
    return isinstance(value, Mapping) and value.get("enabled") is False


def _validate_v1_pod(
    raw: Mapping[str, Any],
    *,
    authorization: LifecycleAuthorization,
    expected_pod_id: str,
    expected_status: str,
    expected_environment: Mapping[str, str],
    expected_session_hash: str,
    expected_machine_hash: str | None,
) -> dict[str, Any]:
    if _require_pod_id(raw.get("id")) != expected_pod_id:
        raise RunpodLifecycleError("RunPod status returned a different Pod")
    if raw.get("desiredStatus") != expected_status:
        raise RunpodLifecycleError(f"RunPod Pod must be exactly {expected_status}")
    spec = authorization.immutable_spec
    if raw.get("image") != spec["image"]:
        raise RunpodLifecycleError("RunPod Pod image drifted")
    if raw.get("containerDiskInGb") != EXACT_CONTAINER_DISK_GB:
        raise RunpodLifecycleError("RunPod Pod container disk drifted")
    if raw.get("volumeInGb") != EXACT_VOLUME_DISK_GB:
        raise RunpodLifecycleError("RunPod Pod persistent disk drifted")
    if raw.get("volumeMountPath") != EXACT_VOLUME_MOUNT_PATH:
        raise RunpodLifecycleError("RunPod Pod persistent mount path drifted")
    if raw.get("networkVolume") is not None:
        raise RunpodLifecycleError("RunPod Pod unexpectedly has a network volume")
    if raw.get("ports") != list(EXACT_PORTS):
        raise RunpodLifecycleError("RunPod Pod ports drifted")
    if not _v1_global_networking_disabled(raw):
        raise RunpodLifecycleError("RunPod global networking cannot be verified as disabled")
    if raw.get("interruptible") is not False or raw.get("locked") is not False:
        raise RunpodLifecycleError("RunPod rental/lock mode drifted")
    gpu = _require_exact_mapping(raw.get("gpu"), label="GPU metadata")
    if gpu.get("id") != EXACT_PROVIDER_GPU_ID or gpu.get("count") != EXACT_GPU_COUNT:
        raise RunpodLifecycleError("RunPod Pod GPU identity/count drifted")
    machine = _require_exact_mapping(raw.get("machine"), label="machine metadata")
    if machine.get("secureCloud") is not True:
        raise RunpodLifecycleError("RunPod Pod is not on Secure Cloud")
    if machine.get("dataCenterId") not in set(spec["data_center_ids"]):
        raise RunpodLifecycleError("RunPod Pod data center is not approved")
    machine_id = raw.get("machineId")
    if not isinstance(machine_id, str) or not machine_id:
        raise RunpodLifecycleError("RunPod Pod machine identity is unavailable")
    machine_hash = stable_hash({"runpod_machine_id": machine_id})
    if expected_machine_hash is not None and machine_hash != expected_machine_hash:
        raise RunpodLifecycleError("stopped RunPod Pod moved to a different machine")
    _validate_cost(raw, authorization)
    environment = _require_exact_environment(
        raw.get("env"),
        expected=expected_environment,
        expected_session_hash=expected_session_hash,
    )
    port_mappings = raw.get("portMappings")
    public_ip = raw.get("publicIp")
    direct_ssh = None
    if expected_status == "RUNNING":
        if not isinstance(port_mappings, Mapping) or not isinstance(public_ip, str):
            raise RunpodLifecycleError("RunPod direct SSH endpoint is unavailable")
        public_port = port_mappings.get("22")
        if isinstance(public_port, bool) or not isinstance(public_port, int):
            raise RunpodLifecycleError("RunPod direct SSH port is unavailable")
        direct_ssh = {"host": public_ip, "port": public_port, "username": "root"}
    return {
        "status": expected_status,
        "machine_id_hash": machine_hash,
        "data_center_id": machine.get("dataCenterId"),
        "direct_ssh": direct_ssh,
        "environment": environment,
    }


def _private_root(project_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(project_root)))
    private = root / ".runpod"
    if os.path.lexists(private) and private.is_symlink():
        raise RunpodLifecycleError("private .runpod directory must not be a symlink")
    private.mkdir(mode=0o700, exist_ok=True)
    details = private.lstat()
    if not stat.S_ISDIR(details.st_mode):
        raise RunpodLifecycleError("private .runpod path is not a directory")
    if details.st_uid != os.getuid():
        raise RunpodLifecycleError("private .runpod directory has an unexpected owner")
    private.chmod(0o700)
    return private


def lifecycle_state_path(project_root: str | Path) -> Path:
    return _private_root(Path(project_root)) / LIFECYCLE_STATE_FILENAME


def _encode_state(payload: Mapping[str, Any]) -> bytes:
    unsigned = deepcopy(dict(payload))
    unsigned.pop("record_hash", None)
    value = {**unsigned, "record_hash": stable_hash(unsigned)}
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_state_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.parent != _private_root(path.parent.parent):
        raise RunpodLifecycleError("lifecycle state must remain directly under .runpod")
    encoded = _encode_state(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RunpodLifecycleError("a local RunPod lifecycle claim already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)
    except Exception:
        # Never remove a partially claimed intent: it intentionally prevents a
        # second paid create after an uncertain first attempt.
        raise


def _replace_state(path: Path, payload: Mapping[str, Any]) -> None:
    _secure_existing_state_file(path)
    encoded = _encode_state(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pod-lifecycle.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _secure_existing_state_file(path)
        os.replace(temporary_name, path)
        path.chmod(0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _secure_existing_state_file(path: Path) -> None:
    if path.parent != _private_root(path.parent.parent):
        raise RunpodLifecycleError("lifecycle state must remain directly under .runpod")
    if path.is_symlink() or not path.is_file():
        raise RunpodLifecycleError("lifecycle state must be a regular non-symlink file")
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RunpodLifecycleError("lifecycle state must not be hard-linked")
    if details.st_uid != os.getuid():
        raise RunpodLifecycleError("lifecycle state has an unexpected owner")
    path.chmod(0o600)


def _load_state(path: Path) -> dict[str, Any]:
    _secure_existing_state_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunpodLifecycleError("private lifecycle state is unreadable") from exc
    if not isinstance(value, dict):
        raise RunpodLifecycleError("private lifecycle state has an invalid schema")
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
    if set(value) != expected_keys:
        raise RunpodLifecycleError("private lifecycle state has an unexpected schema")
    if value.get("schema_version") != 1 or value.get("protocol_version") != LIFECYCLE_PROTOCOL:
        raise RunpodLifecycleError("private lifecycle state protocol is unsupported")
    unsigned = {key: item for key, item in value.items() if key != "record_hash"}
    if value.get("record_hash") != stable_hash(unsigned):
        raise RunpodLifecycleError("private lifecycle state content hash mismatch")
    if not isinstance(value.get("current_authorization"), Mapping):
        raise RunpodLifecycleError("private lifecycle authorization is malformed")
    if not isinstance(value.get("authorization_history"), list):
        raise RunpodLifecycleError("private lifecycle history is malformed")
    return value


def _base_state(
    *,
    operation: str,
    authorization: LifecycleAuthorization,
    launch_spec_hash: str,
    pod: Mapping[str, Any] | None,
    acknowledged_existing_pod_id_hashes: Sequence[str] = (),
    history: Sequence[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": LIFECYCLE_PROTOCOL,
        "operation": operation,
        "updated_at": _utc_timestamp(now),
        "immutable_spec": deepcopy(dict(authorization.immutable_spec)),
        "current_authorization": authorization.manifest(
            launch_spec_hash=launch_spec_hash,
            acknowledged_existing_pod_id_hashes=acknowledged_existing_pod_id_hashes,
        ),
        "authorization_history": [deepcopy(dict(item)) for item in history],
        "pod": deepcopy(dict(pod)) if pod is not None else None,
    }


def _validate_existing_pod_allowlist(
    pods: Sequence[Mapping[str, Any]],
    *,
    acknowledged_existing_pod_id_hashes: Sequence[str],
) -> tuple[str, ...]:
    acknowledged = _canonical_existing_pod_id_hashes(
        acknowledged_existing_pod_id_hashes
    )
    observed: list[str] = []
    for pod in pods:
        status = pod.get("desiredStatus")
        if status not in TERMINAL_POD_STATUSES:
            observed.append(existing_pod_id_hash(_require_pod_id(pod.get("id"))))
    if len(observed) != len(set(observed)):
        raise RunpodLifecycleError("RunPod returned duplicate nonterminal Pod IDs")
    if set(observed) != set(acknowledged):
        missing = set(observed) - set(acknowledged)
        if missing:
            raise RunpodLifecycleError(
                "refusing paid creation while a nonterminal Pod is not explicitly allowlisted"
            )
        raise RunpodLifecycleError(
            "existing Pod allow-list contains a hash not present in the nonterminal Pod set"
        )
    return acknowledged


def _v1_running_metadata_ready(raw: Mapping[str, Any]) -> bool:
    """Distinguish ordinary provisioning omissions from explicit drift."""

    gpu = raw.get("gpu")
    machine = raw.get("machine")
    environment = raw.get("env")
    mappings = raw.get("portMappings")
    return bool(
        isinstance(gpu, Mapping)
        and gpu.get("id") is not None
        and gpu.get("count") is not None
        and isinstance(machine, Mapping)
        and machine.get("secureCloud") is not None
        and machine.get("dataCenterId") is not None
        and isinstance(raw.get("machineId"), str)
        and raw.get("machineId")
        and isinstance(environment, Mapping)
        and REQUESTED_POD_ENV_KEYS.issubset(environment)
        and isinstance(mappings, Mapping)
        and isinstance(mappings.get("22"), int)
        and not isinstance(mappings.get("22"), bool)
        and isinstance(raw.get("publicIp"), str)
        and raw.get("publicIp")
    )


def _pending_state(
    state: Mapping[str, Any],
    *,
    operation: str,
    now: datetime | None,
) -> dict[str, Any]:
    return {
        **dict(state),
        "operation": operation,
        "updated_at": _utc_timestamp(now),
    }


def _wait_for_running_v1(
    *,
    client: RunpodLifecycleClient,
    state_path: Path,
    state: Mapping[str, Any],
    sanitized_pod: dict[str, Any],
    authorization: LifecycleAuthorization,
    expected_environment: Mapping[str, str],
    maximum_wait_seconds: float,
    poll_interval_seconds: float,
    sleep: Sleep,
    monotonic: Monotonic,
    now: datetime | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not math.isfinite(maximum_wait_seconds)
        or not 0 <= maximum_wait_seconds <= 600
        or not math.isfinite(poll_interval_seconds)
        or not 0 < poll_interval_seconds <= 30
    ):
        raise ValueError("RunPod readiness wait must use <=10 minutes and <=30-second polls")
    pod_id = str(sanitized_pod["id"])
    deadline = monotonic() + maximum_wait_seconds
    current_state = dict(state)
    while True:
        live = client.get_pod_v1(pod_id)
        observed_id = _require_pod_id(live.get("id"))
        if observed_id != pod_id:
            failed = _pending_state(
                current_state,
                operation="create_verification_failed",
                now=now,
            )
            _replace_state(state_path, failed)
            raise RunpodLifecycleError("RunPod status returned a different Pod; create is locked")
        status = live.get("desiredStatus")
        if status in TERMINAL_POD_STATUSES or status == "ERROR":
            failed = _pending_state(
                current_state,
                operation="create_failed_terminal",
                now=now,
            )
            _replace_state(state_path, failed)
            raise RunpodLifecycleError(
                "new RunPod Pod became terminal before readiness; inspect status and do not recreate"
            )
        if status != "RUNNING":
            failed = _pending_state(
                current_state,
                operation="create_verification_failed",
                now=now,
            )
            _replace_state(state_path, failed)
            raise RunpodLifecycleError("RunPod returned an unknown Pod status; create is locked")
        if _v1_running_metadata_ready(live):
            try:
                details = _validate_v1_pod(
                    live,
                    authorization=authorization,
                    expected_pod_id=pod_id,
                    expected_status="RUNNING",
                    expected_environment=expected_environment,
                    expected_session_hash=authorization.session_hash,
                    expected_machine_hash=None,
                )
            except RunpodLifecycleError:
                failed = _pending_state(
                    current_state,
                    operation="create_verification_failed",
                    now=now,
                )
                _replace_state(state_path, failed)
                raise
            return details, current_state
        current_state = _pending_state(
            current_state,
            operation="create_pending",
            now=now,
        )
        _replace_state(state_path, current_state)
        remaining = deadline - monotonic()
        if remaining <= 0:
            timeout_state = _pending_state(
                current_state,
                operation="create_timeout",
                now=now,
            )
            _replace_state(state_path, timeout_state)
            raise RunpodLifecycleError(
                "RunPod readiness timed out; use read-only status and never rerun create"
            )
        sleep(min(poll_interval_seconds, remaining, 30.0))


def _safe_summary(state: Mapping[str, Any], *, provider_status: str) -> dict[str, Any]:
    pod = _require_exact_mapping(state.get("pod"), label="private Pod record")
    pod_id = _require_pod_id(pod.get("id"))
    ssh = pod.get("ssh")
    return {
        "schema_version": 1,
        "operation": state.get("operation"),
        "provider_status": provider_status,
        "pod_id_hash": stable_hash({"runpod_pod_id": pod_id}),
        "ssh_private_record_available": isinstance(ssh, Mapping),
        "immutable_spec_verified": True,
        "passed": True,
    }


def create_approved_pod(
    *,
    project_root: str | Path,
    client: RunpodLifecycleClient,
    authorization: LifecycleAuthorization,
    name: str,
    hf_token: str,
    session_nonce: str,
    acknowledged_existing_pod_id_hashes: Sequence[str] = (),
    now: datetime | None = None,
    maximum_wait_seconds: float = 600.0,
    poll_interval_seconds: float = 10.0,
    sleep: Sleep = time.sleep,
    monotonic: Monotonic = time.monotonic,
) -> dict[str, Any]:
    """Create one exact Pod after duplicate-spend and local-intent gates."""

    state_path = lifecycle_state_path(project_root)
    if os.path.lexists(state_path):
        raise RunpodLifecycleError("refusing to create a second Pod from an existing lifecycle")
    acknowledged_hashes = _canonical_existing_pod_id_hashes(
        acknowledged_existing_pod_id_hashes
    )
    payload, launch_hash = build_create_payload(
        authorization=authorization,
        name=name,
        hf_token=hf_token,
        session_nonce=session_nonce,
        acknowledged_existing_pod_id_hashes=acknowledged_hashes,
    )
    _validate_existing_pod_allowlist(
        client.list_pods_v1(),
        acknowledged_existing_pod_id_hashes=acknowledged_hashes,
    )
    intent = _base_state(
        operation="create_intent",
        authorization=authorization,
        launch_spec_hash=launch_hash,
        pod=None,
        acknowledged_existing_pod_id_hashes=acknowledged_hashes,
        now=now,
    )
    _exclusive_state_write(state_path, intent)
    response = client.create_pod_v2(payload)
    try:
        sanitized = _validate_v2_pod(
            response,
            authorization=authorization,
            expected_name=name,
            expected_environment=payload["env"],
            expected_session_hash=authorization.session_hash,
            allowed_statuses=frozenset({"PROVISIONING", "STARTING", "RUNNING"}),
        )
    except RunpodLifecycleError:
        pod_id = response.get("id")
        if isinstance(pod_id, str) and _POD_ID_RE.fullmatch(pod_id):
            uncertain = {**intent, "operation": "create_response_unverified", "pod": {"id": pod_id}}
            _replace_state(state_path, uncertain)
        raise
    received = {
        **intent,
        "operation": "create_response_received",
        "updated_at": _utc_timestamp(now),
        "pod": sanitized,
    }
    _replace_state(state_path, received)
    live_details, received = _wait_for_running_v1(
        client=client,
        state_path=state_path,
        state=received,
        sanitized_pod=sanitized,
        authorization=authorization,
        expected_environment=payload["env"],
        maximum_wait_seconds=maximum_wait_seconds,
        poll_interval_seconds=poll_interval_seconds,
        sleep=sleep,
        monotonic=monotonic,
        now=now,
    )
    sanitized["status"] = "RUNNING"
    sanitized["machine_id_hash"] = live_details["machine_id_hash"]
    sanitized["data_center_id"] = live_details["data_center_id"]
    if live_details["direct_ssh"] is not None:
        sanitized["ssh"]["direct"] = live_details["direct_ssh"]
    completed = {
        **received,
        "operation": "created",
        "updated_at": _utc_timestamp(now),
        "pod": sanitized,
    }
    _replace_state(state_path, completed)
    return _safe_summary(completed, provider_status="RUNNING")


def read_lifecycle_status(
    *,
    project_root: str | Path,
    client: RunpodLifecycleClient,
) -> dict[str, Any]:
    """Read and verify provider status without modifying local or remote state."""

    state = _load_state(lifecycle_state_path(project_root))
    pod = state.get("pod")
    if not isinstance(pod, Mapping) or set(pod) == {"id"}:
        return {
            "schema_version": 1,
            "operation": state.get("operation"),
            "provider_status": "UNKNOWN",
            "immutable_spec_verified": False,
            "passed": False,
        }
    auth = _authorization_from_state(state)
    live = client.get_pod_v1(_require_pod_id(pod.get("id")))
    provider_status = live.get("desiredStatus")
    if provider_status in {"TERMINATED", "ERROR"}:
        return {
            "schema_version": 1,
            "operation": state.get("operation"),
            "provider_status": provider_status,
            "pod_id_hash": stable_hash({"runpod_pod_id": pod["id"]}),
            "immutable_spec_verified": False,
            "passed": False,
        }
    if provider_status not in {"RUNNING", "EXITED"}:
        raise RunpodLifecycleError("RunPod Pod is in an unsupported lifecycle status")
    current = _require_exact_mapping(state["current_authorization"], label="authorization")
    placeholder_hf = live.get("env")
    if not isinstance(placeholder_hf, Mapping) or not isinstance(
        placeholder_hf.get(HF_TOKEN_ENV_NAME), str
    ):
        raise RunpodLifecycleError("RunPod Pod environment is missing")
    expected_env = {
        HF_TOKEN_ENV_NAME: placeholder_hf[HF_TOKEN_ENV_NAME],
        SESSION_ENV_NAME: "validated-by-hash",
        **STATIC_POD_ENV,
    }
    stored_machine_hash = pod.get("machine_id_hash")
    if stored_machine_hash is not None and not isinstance(stored_machine_hash, str):
        raise RunpodLifecycleError("private Pod machine binding is malformed")
    _validate_v1_pod(
        live,
        authorization=auth,
        expected_pod_id=str(pod["id"]),
        expected_status=str(provider_status),
        expected_environment=expected_env,
        expected_session_hash=str(current["session_hash"]),
        expected_machine_hash=stored_machine_hash,
    )
    return _safe_summary(state, provider_status=str(provider_status))


def _authorization_from_state(state: Mapping[str, Any]) -> LifecycleAuthorization:
    spec = _require_exact_mapping(state.get("immutable_spec"), label="immutable spec")
    current = _require_exact_mapping(state.get("current_authorization"), label="authorization")
    required_hashes = (
        "reservation_record_hash",
        "session_hash",
        "approval_hash",
        "bindings_hash",
        "gpu_lock_hash",
        "quote_hash",
        "immutable_spec_hash",
    )
    if any(not isinstance(current.get(key), str) for key in required_hashes):
        raise RunpodLifecycleError("private lifecycle authorization is malformed")
    if current.get("immutable_spec_hash") != stable_hash(dict(spec)):
        raise RunpodLifecycleError("private immutable launch specification drifted")
    return LifecycleAuthorization(
        phase=str(current.get("phase")),
        reservation_id=str(current.get("reservation_id")),
        reservation_record_hash=str(current["reservation_record_hash"]),
        session_hash=str(current["session_hash"]),
        approval_hash=str(current["approval_hash"]),
        bindings_hash=str(current["bindings_hash"]),
        gpu_lock_hash=str(current["gpu_lock_hash"]),
        quote_hash=str(current["quote_hash"]),
        immutable_spec=dict(spec),
        immutable_spec_hash=str(current["immutable_spec_hash"]),
        approved_runtime_hours=float(current.get("approved_runtime_hours")),
        approved_phase_maximum_usd=float(current.get("approved_phase_maximum_usd")),
        live_hourly_total_usd=float(current.get("live_hourly_total_usd")),
    )


def _require_prior_reservation_settled(
    *, ledger: CostLedger, reservation_id: str
) -> None:
    matching = [
        item for item in ledger.document()["entries"] if item.get("entry_id") == reservation_id
    ]
    if len(matching) != 1 or matching[0].get("status") != "incurred":
        raise RunpodLifecycleError("prior GPU reservation must be settled before re-arm")


def rearm_approved_pod(
    *,
    project_root: str | Path,
    client: RunpodLifecycleClient,
    authorization: LifecycleAuthorization,
    ledger: CostLedger,
    hf_token: str,
    session_nonce: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Change only the approved env on the same stopped Pod, then start it."""

    state_path = lifecycle_state_path(project_root)
    state = _load_state(state_path)
    recoverable_operations = {
        "created",
        "rearmed",
        "create_pending",
        "create_timeout",
        "create_verification_failed",
    }
    if state.get("operation") not in recoverable_operations:
        raise RunpodLifecycleError("lifecycle has an unresolved operation; re-arm refused")
    pod = _require_exact_mapping(state.get("pod"), label="private Pod record")
    previous = _authorization_from_state(state)
    previous_manifest = _require_exact_mapping(
        state.get("current_authorization"), label="authorization"
    )
    if previous.immutable_spec_hash != authorization.immutable_spec_hash:
        raise RunpodLifecycleError("re-arm immutable launch specification disagrees")
    if (
        previous.session_hash == authorization.session_hash
        or previous.reservation_id == authorization.reservation_id
    ):
        raise RunpodLifecycleError("re-arm requires a fresh reservation and session nonce")
    history = list(state["authorization_history"])
    used_session_hashes = {
        str(item.get("session_hash")) for item in [*history, previous_manifest] if isinstance(item, Mapping)
    }
    if authorization.session_hash in used_session_hashes:
        raise RunpodLifecycleError("re-arm session nonce was already used")
    _require_prior_reservation_settled(
        ledger=ledger,
        reservation_id=previous.reservation_id,
    )
    pod_id = _require_pod_id(pod.get("id"))
    new_environment = pod_environment(hf_token=hf_token, session_nonce=session_nonce)
    live = client.get_pod_v1(pod_id)
    stored_machine_hash = pod.get("machine_id_hash")
    if stored_machine_hash is not None and not isinstance(stored_machine_hash, str):
        raise RunpodLifecycleError("private Pod machine binding is malformed")
    live_details = _validate_v1_pod(
        live,
        authorization=previous,
        expected_pod_id=pod_id,
        expected_status="EXITED",
        expected_environment=new_environment,
        expected_session_hash=previous.session_hash,
        expected_machine_hash=stored_machine_hash,
    )
    old_environment = dict(live_details["environment"])
    patch_environment = dict(old_environment)
    patch_environment[SESSION_ENV_NAME] = session_nonce
    changed_keys = {
        key
        for key in set(old_environment) | set(patch_environment)
        if old_environment.get(key) != patch_environment.get(key)
    }
    if changed_keys != {SESSION_ENV_NAME} or set(patch_environment) - (
        REQUESTED_POD_ENV_KEYS | PROVIDER_MANAGED_ENV_KEYS
    ):
        raise RunpodLifecycleError("re-arm PATCH must change only the GPU session nonce")
    _, launch_hash = build_create_payload(
        authorization=authorization,
        name=str(pod.get("name")),
        hf_token=hf_token,
        session_nonce=session_nonce,
    )
    intent = _base_state(
        operation="rearm_intent",
        authorization=authorization,
        launch_spec_hash=launch_hash,
        pod={
            **dict(pod),
            "machine_id_hash": live_details["machine_id_hash"],
            "data_center_id": live_details["data_center_id"],
        },
        history=[*history, dict(previous_manifest)],
        now=now,
    )
    _replace_state(state_path, intent)
    patched_raw = client.patch_environment_v2(
        pod_id=pod_id,
        environment=patch_environment,
    )
    _validate_v2_pod(
        patched_raw,
        authorization=authorization,
        expected_name=str(pod.get("name")),
        expected_environment=patch_environment,
        expected_session_hash=authorization.session_hash,
        allowed_statuses=frozenset({"EXITED"}),
    )
    patched_state = {
        **intent,
        "operation": "rearm_patched",
        "updated_at": _utc_timestamp(now),
    }
    _replace_state(state_path, patched_state)
    started_raw = client.start_pod_v2(pod_id=pod_id)
    sanitized_started = _validate_v2_pod(
        started_raw,
        authorization=authorization,
        expected_name=str(pod.get("name")),
        expected_environment=patch_environment,
        expected_session_hash=authorization.session_hash,
        allowed_statuses=frozenset({"PROVISIONING", "STARTING", "RUNNING"}),
    )
    start_requested = {
        **patched_state,
        "operation": "rearm_start_requested",
        "updated_at": _utc_timestamp(now),
        "pod": {
            **dict(pod),
            "status": sanitized_started["status"],
            "ssh": sanitized_started["ssh"],
        },
    }
    _replace_state(state_path, start_requested)
    live_started = client.get_pod_v1(pod_id)
    started_details = _validate_v1_pod(
        live_started,
        authorization=authorization,
        expected_pod_id=pod_id,
        expected_status="RUNNING",
        expected_environment=patch_environment,
        expected_session_hash=authorization.session_hash,
        expected_machine_hash=str(pod.get("machine_id_hash")),
    )
    completed_pod = dict(start_requested["pod"])
    completed_pod["status"] = "RUNNING"
    completed_pod["data_center_id"] = started_details["data_center_id"]
    completed_pod["machine_id_hash"] = started_details["machine_id_hash"]
    if started_details["direct_ssh"] is not None:
        completed_pod["ssh"]["direct"] = started_details["direct_ssh"]
    completed = {
        **start_requested,
        "operation": "rearmed",
        "updated_at": _utc_timestamp(now),
        "pod": completed_pod,
    }
    _replace_state(state_path, completed)
    return _safe_summary(completed, provider_status="RUNNING")


__all__ = [
    "EXACT_CLOUD",
    "EXACT_CONTAINER_DISK_GB",
    "EXACT_CUDA_VERSIONS",
    "EXACT_GPU_COUNT",
    "EXACT_PROVIDER_GPU_ID",
    "EXACT_VOLUME_DISK_GB",
    "RUNPOD_V1_BASE",
    "RUNPOD_V1_PODS_URL",
    "RUNPOD_V2_BASE",
    "RUNPOD_V2_PODS_URL",
    "HttpResult",
    "LifecycleAuthorization",
    "RunpodLifecycleClient",
    "RunpodLifecycleError",
    "authorize_gpu_lifecycle",
    "build_create_payload",
    "create_approved_pod",
    "existing_pod_id_hash",
    "lifecycle_state_path",
    "pod_environment",
    "read_lifecycle_status",
    "rearm_approved_pod",
    "urllib_http_transport",
]
