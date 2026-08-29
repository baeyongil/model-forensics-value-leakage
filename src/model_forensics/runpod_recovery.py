"""Read-only attestation of an externally stopped RunPod Pod.

The recovery path in this module deliberately has no provider mutation method.
It reads one Pod and its billing history from the official REST v1 API, binds
the result to the private lifecycle claim, creates an authenticated receipt,
and only then changes the *local* lifecycle operation to ``stopped``.

Provider billing can lag Pod status.  A caller may explicitly opt into a
conservative accounting receipt derived from provider start/exit timestamps.
Such a receipt is labelled ``pending`` and never claims to be an invoice.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from model_forensics.io import read_json, sha256_file, stable_hash
from model_forensics.runpod_lifecycle import (
    EXACT_CONTAINER_DISK_GB,
    EXACT_GPU_COUNT,
    EXACT_PORTS,
    EXACT_PROVIDER_GPU_ID,
    EXACT_VOLUME_DISK_GB,
    EXACT_VOLUME_MOUNT_PATH,
    HF_TOKEN_ENV_NAME,
    LIFECYCLE_PROTOCOL,
    PROVIDER_MANAGED_ENV_KEYS,
    REQUESTED_POD_ENV_KEYS,
    SESSION_ENV_NAME,
    STATIC_POD_ENV,
)

RUNPOD_REST_V1_BASE = "https://rest.runpod.io/v1"
RUNPOD_BILLING_PODS_URL = f"{RUNPOD_REST_V1_BASE}/billing/pods"
EXTERNAL_STOP_PROTOCOL = "runpod-external-stop-v1"
EXTERNAL_STOP_RECEIPT_FILENAME = "external_stop_receipt.json"
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{5,127}\Z")
_MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024
_USD_QUANTUM = Decimal("0.000001")
_EXIT_MESSAGE_RE = re.compile(
    r"^Exited by (?:user|system): (?P<timestamp>.+)$",
    flags=re.IGNORECASE,
)


class RunpodRecoveryError(RuntimeError):
    """External-stop evidence is incomplete, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class RecoveryHttpResult:
    status_code: int
    body: bytes


RecoveryHttpTransport = Callable[..., RecoveryHttpResult]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunpodRecoveryError("RunPod response contains a duplicate JSON key")
        result[key] = value
    return result


def _decode_json(result: RecoveryHttpResult) -> Any:
    if result.status_code != 200:
        raise RunpodRecoveryError(f"RunPod read-only request failed with HTTP {result.status_code}")
    if len(result.body) > _MAX_HTTP_BODY_BYTES:
        raise RunpodRecoveryError("RunPod response exceeds the safe size limit")
    try:
        return json.loads(
            result.body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_float=Decimal,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as exc:
        raise RunpodRecoveryError("RunPod returned malformed JSON") from exc


def urllib_recovery_transport(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float,
) -> RecoveryHttpResult:
    """GET-only transport whose error path never exposes a provider body."""

    if method != "GET" or body is not None:
        raise RunpodRecoveryError("external-stop recovery permits provider GET requests only")
    request = urllib.request.Request(url=url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return RecoveryHttpResult(
                status_code=int(response.status),
                body=response.read(_MAX_HTTP_BODY_BYTES + 1),
            )
    except urllib.error.HTTPError as exc:
        raise RunpodRecoveryError(
            f"RunPod read-only request failed with HTTP {exc.code}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RunpodRecoveryError("RunPod read-only request outcome is uncertain") from exc


class RunpodRecoveryClient:
    """Capability-limited REST v1 client with no provider write primitive."""

    def __init__(
        self,
        *,
        api_key: str,
        transport: RecoveryHttpTransport = urllib_recovery_transport,
        timeout_seconds: float = 20.0,
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or api_key != api_key.strip()
            or len(api_key) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in api_key)
        ):
            raise RunpodRecoveryError("RunPod API key is missing or malformed")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("HTTP timeout must be in (0, 60]")
        self._api_key = api_key
        self._transport = transport
        self._timeout_seconds = float(timeout_seconds)

    def _get(self, url: str) -> Any:
        if not url.startswith(f"{RUNPOD_REST_V1_BASE}/"):
            raise RunpodRecoveryError("refusing a non-REST-v1 RunPod endpoint")
        result = self._transport(
            method="GET",
            url=url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            body=None,
            timeout_seconds=self._timeout_seconds,
        )
        return _decode_json(result)

    def get_pod(self, pod_id: str) -> Mapping[str, Any]:
        exact_id = _require_pod_id(pod_id)
        query = urllib.parse.urlencode(
            {
                "includeMachine": "true",
                "includeNetworkVolume": "true",
                "includeTemplate": "true",
            }
        )
        payload = self._get(
            f"{RUNPOD_REST_V1_BASE}/pods/{urllib.parse.quote(exact_id, safe='')}?{query}"
        )
        if not isinstance(payload, Mapping):
            raise RunpodRecoveryError("RunPod Pod response must be a JSON object")
        return payload

    def get_billing(
        self,
        *,
        pod_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Any]:
        exact_id = _require_pod_id(pod_id)
        query = urllib.parse.urlencode(
            {
                "grouping": "podId",
                "podId": exact_id,
                "startTime": _iso_z(start_time),
                "endTime": _iso_z(end_time),
            }
        )
        payload = self._get(f"{RUNPOD_BILLING_PODS_URL}?{query}")
        if not isinstance(payload, list):
            raise RunpodRecoveryError("RunPod billing response must be a JSON array")
        return payload


def _require_pod_id(value: Any) -> str:
    if not isinstance(value, str) or _POD_ID_RE.fullmatch(value) is None:
        raise RunpodRecoveryError("private lifecycle Pod identity is malformed")
    return value


def _iso_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("RunPod timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_provider_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RunpodRecoveryError(f"RunPod {field} timestamp is missing")
    raw = value.strip()
    candidates = [raw]
    if raw.endswith(" UTC"):
        candidates.append(raw[:-4])
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(UTC)
        for pattern in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
            try:
                return datetime.strptime(candidate, pattern).astimezone(UTC)
            except ValueError:
                pass
    raise RunpodRecoveryError(f"RunPod {field} timestamp is malformed")


def _parse_exit_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RunpodRecoveryError("RunPod exit timestamp is missing")
    message = _EXIT_MESSAGE_RE.fullmatch(value.strip())
    if message is None:
        # Some provider revisions expose the last status change as a bare ISO
        # timestamp.  It is still bound only after desiredStatus=EXITED.
        return _parse_provider_timestamp(value, field="exit")
    raw = message.group("timestamp")
    for pattern in (
        "%a %b %d %Y %H:%M:%S GMT%z (Coordinated Universal Time)",
        "%a %b %d %Y %H:%M:%S %z",
    ):
        try:
            return datetime.strptime(raw, pattern).astimezone(UTC)
        except ValueError:
            pass
    raise RunpodRecoveryError("RunPod exit timestamp is malformed")


def _decimal(value: Any, *, field: str, allow_zero: bool = True) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise RunpodRecoveryError(f"RunPod {field} must be numeric")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RunpodRecoveryError(f"RunPod {field} must be numeric") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        raise RunpodRecoveryError(f"RunPod {field} is outside the allowed range")
    return parsed


def _ceil_usd(value: Decimal) -> Decimal:
    return value.quantize(_USD_QUANTUM, rounding=ROUND_CEILING)


def _json_usd(value: Decimal) -> float:
    return float(_ceil_usd(value))


def _hashable_json(value: Any) -> Any:
    """Convert exact decoder Decimals to deterministic JSON scalars for hashing."""

    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        return {str(key): _hashable_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hashable_json(item) for item in value]
    return value


def _secure_lifecycle_state(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunpodRecoveryError("private lifecycle state is missing or unsafe")
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
    ):
        raise RunpodRecoveryError("private lifecycle state is not exclusively owned")


def _load_lifecycle(path: Path) -> dict[str, Any]:
    _secure_lifecycle_state(path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise RunpodRecoveryError("private lifecycle state is unreadable") from exc
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
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RunpodRecoveryError("private lifecycle state schema is unsupported")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != LIFECYCLE_PROTOCOL:
        raise RunpodRecoveryError("private lifecycle state protocol is unsupported")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if payload.get("record_hash") != stable_hash(unsigned):
        raise RunpodRecoveryError("private lifecycle state content hash mismatch")
    if not isinstance(payload.get("pod"), Mapping):
        raise RunpodRecoveryError("private lifecycle Pod record is missing")
    if not isinstance(payload.get("immutable_spec"), Mapping) or not isinstance(
        payload.get("current_authorization"), Mapping
    ):
        raise RunpodRecoveryError("private lifecycle authorization is missing")
    return payload


def _encoded_authenticated(payload: Mapping[str, Any]) -> bytes:
    unsigned = deepcopy(dict(payload))
    unsigned.pop("record_hash", None)
    authenticated = {**unsigned, "record_hash": stable_hash(unsigned)}
    return (
        json.dumps(authenticated, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_replace_lifecycle(path: Path, payload: Mapping[str, Any]) -> None:
    _secure_lifecycle_state(path)
    encoded = _encoded_authenticated(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pod-lifecycle-stop.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _secure_lifecycle_state(path)
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


def _authenticated_stopped_state(
    state: Mapping[str, Any], *, observed_at: datetime
) -> dict[str, Any]:
    stopped = deepcopy(dict(state))
    stopped["operation"] = "stopped"
    stopped["updated_at"] = _iso_z(observed_at)
    pod = deepcopy(dict(stopped["pod"]))
    pod["status"] = "EXITED"
    if "ssh" in pod:
        pod["ssh"] = {"proxy": None, "direct": None}
    stopped["pod"] = pod
    stopped.pop("record_hash", None)
    stopped["record_hash"] = stable_hash(stopped)
    return stopped


def _validate_environment(value: Any, *, expected_session_hash: str) -> None:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RunpodRecoveryError("RunPod environment is missing or malformed")
    observed = dict(value)
    allowed = REQUESTED_POD_ENV_KEYS | PROVIDER_MANAGED_ENV_KEYS
    if set(observed) - allowed or not REQUESTED_POD_ENV_KEYS.issubset(observed):
        raise RunpodRecoveryError("RunPod environment violates the frozen allow-list")
    for key, expected in STATIC_POD_ENV.items():
        if observed.get(key) != expected:
            raise RunpodRecoveryError("RunPod cache/compatibility environment drifted")
    if stable_hash({"opaque_gpu_session_id": observed[SESSION_ENV_NAME]}) != expected_session_hash:
        raise RunpodRecoveryError("RunPod GPU session nonce drifted")
    token = observed.get(HF_TOKEN_ENV_NAME)
    if not isinstance(token, str) or not token or len(token) > 4096:
        raise RunpodRecoveryError("RunPod Hugging Face credential is malformed")
    public_key = observed.get("PUBLIC_KEY")
    if public_key is not None and (not public_key.strip() or len(public_key) > 65536):
        raise RunpodRecoveryError("RunPod provider-managed SSH key is malformed")


def _validate_pod(
    payload: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    observed_at: datetime,
) -> tuple[dict[str, Any], str, Decimal, datetime, datetime]:
    stored = dict(state["pod"])
    spec = dict(state["immutable_spec"])
    authorization = dict(state["current_authorization"])
    pod_id = _require_pod_id(stored.get("id"))
    if payload.get("id") != pod_id:
        raise RunpodRecoveryError("RunPod status returned a different Pod")
    if payload.get("desiredStatus") != "EXITED":
        raise RunpodRecoveryError("RunPod Pod is not exactly EXITED")
    expected_name = stored.get("name")
    if not isinstance(expected_name, str) or payload.get("name") != expected_name:
        raise RunpodRecoveryError("RunPod Pod name drifted")
    if payload.get("imageName") != spec.get("image") or payload.get("imageName") != stored.get(
        "image"
    ):
        raise RunpodRecoveryError("RunPod Pod image drifted")
    if payload.get("gpuCount") != EXACT_GPU_COUNT:
        raise RunpodRecoveryError("RunPod Pod GPU count drifted")
    machine = payload.get("machine")
    if not isinstance(machine, Mapping):
        raise RunpodRecoveryError("RunPod machine metadata is missing")
    if machine.get("gpuTypeId") != EXACT_PROVIDER_GPU_ID:
        raise RunpodRecoveryError("RunPod machine GPU type drifted")
    approved_centers = spec.get("data_center_ids")
    if not isinstance(approved_centers, list) or machine.get("dataCenterId") not in set(
        approved_centers
    ):
        raise RunpodRecoveryError("RunPod data center drifted")
    if stored.get("data_center_id") != machine.get("dataCenterId"):
        raise RunpodRecoveryError("RunPod stored data-center binding drifted")
    if machine.get("secureCloud") is not True:
        raise RunpodRecoveryError("RunPod Pod is not in Secure Cloud")
    if (
        payload.get("containerDiskInGb") != EXACT_CONTAINER_DISK_GB
        or payload.get("volumeInGb") != EXACT_VOLUME_DISK_GB
        or payload.get("volumeMountPath") != EXACT_VOLUME_MOUNT_PATH
    ):
        raise RunpodRecoveryError("RunPod disk or persistent mount drifted")
    if "networkVolume" in payload and payload.get("networkVolume") is not None:
        raise RunpodRecoveryError("RunPod Pod unexpectedly has a network volume")
    if payload.get("ports") != list(EXACT_PORTS):
        raise RunpodRecoveryError("RunPod Pod ports drifted")
    session_hash = authorization.get("session_hash")
    if not isinstance(session_hash, str) or _HASH_RE.fullmatch(session_hash) is None:
        raise RunpodRecoveryError("private lifecycle session hash is malformed")
    _validate_environment(payload.get("env"), expected_session_hash=session_hash)

    provider_hourly = _decimal(payload.get("costPerHr"), field="hourly cost", allow_zero=False)
    approved_hourly = _decimal(
        authorization.get("live_hourly_total_usd"),
        field="approved all-in hourly cost",
        allow_zero=False,
    )
    if provider_hourly > approved_hourly:
        raise RunpodRecoveryError("RunPod hourly cost exceeds the approved all-in quote")
    created_at = _parse_provider_timestamp(payload.get("createdAt"), field="creation")
    started_at = _parse_provider_timestamp(payload.get("lastStartedAt"), field="start")
    exited_at = _parse_exit_timestamp(payload.get("lastStatusChange"))
    now = observed_at.astimezone(UTC)
    if started_at < created_at - timedelta(minutes=5) or started_at > created_at + timedelta(
        minutes=5
    ):
        raise RunpodRecoveryError("RunPod creation and start timestamps disagree")
    if exited_at < started_at or exited_at > now + timedelta(minutes=5):
        raise RunpodRecoveryError("RunPod start/exit timestamps are implausible")

    evidence = {
        "desired_status": "EXITED",
        "name": expected_name,
        "image": payload.get("imageName"),
        "gpu": {"id": machine.get("gpuTypeId"), "count": payload.get("gpuCount")},
        "cloud": "SECURE",
        "data_center_id": machine.get("dataCenterId"),
        "container_disk_gb": payload.get("containerDiskInGb"),
        "persistent_disk_gb": payload.get("volumeInGb"),
        "persistent_mount_path": payload.get("volumeMountPath"),
        "ports": list(payload.get("ports", [])),
        "environment_verified": True,
        "provider_hourly_compute_usd": _json_usd(provider_hourly),
        "approved_hourly_all_in_usd": _json_usd(approved_hourly),
        "created_at": _iso_z(created_at),
        "started_at": _iso_z(started_at),
        "exited_at": _iso_z(exited_at),
        "runtime_ms": int((exited_at - started_at).total_seconds() * 1000),
    }
    return evidence, pod_id, approved_hourly, started_at, exited_at


def _billing_query_evidence(
    *, pod_id_hash: str, start_time: datetime, end_time: datetime
) -> dict[str, Any]:
    return {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": _iso_z(start_time),
        "end_time": _iso_z(end_time),
    }


def _conservative_ceiling(
    *, start_time: datetime, end_time: datetime, all_in_hourly_usd: Decimal
) -> tuple[int, Decimal]:
    runtime_seconds = Decimal(str((end_time - start_time).total_seconds()))
    if runtime_seconds <= 0:
        raise RunpodRecoveryError("RunPod runtime must be positive")
    billed_minutes = int((runtime_seconds / Decimal(60)).to_integral_value(rounding=ROUND_CEILING))
    return billed_minutes, _ceil_usd(
        Decimal(billed_minutes) * all_in_hourly_usd / Decimal(60)
    )


def _validate_billing_row(
    row: Any,
    *,
    pod_id: str,
    pod_id_hash: str,
    expected_gpu_id: str,
    conservative_ceiling: Decimal,
    approved_runtime_hours: Decimal,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise RunpodRecoveryError("RunPod billing row must be a JSON object")
    if row.get("podId") != pod_id:
        raise RunpodRecoveryError("RunPod billing row belongs to a different Pod")
    gpu_type = row.get("gpuTypeId")
    if gpu_type is not None and gpu_type != expected_gpu_id:
        raise RunpodRecoveryError("RunPod billing GPU type drifted")
    amount = _decimal(row.get("amount"), field="billing amount")
    if amount > conservative_ceiling:
        raise RunpodRecoveryError("RunPod billing amount exceeds the timestamp-derived ceiling")
    billed_ms = row.get("timeBilledMs")
    if isinstance(billed_ms, bool) or not isinstance(billed_ms, int) or billed_ms < 0:
        raise RunpodRecoveryError("RunPod billing timeBilledMs is malformed")
    maximum_ms = int(approved_runtime_hours * Decimal(3_600_000)) + 60_000
    if billed_ms > maximum_ms:
        raise RunpodRecoveryError("RunPod billing duration exceeds the approved runtime")
    billing_time = _parse_provider_timestamp(row.get("time"), field="billing bucket")
    raw_row_hash = stable_hash(_hashable_json(dict(row)))
    return {
        "billing_status": "final",
        "evidence_kind": "provider_billing_row",
        "pod_id_hash": pod_id_hash,
        "provider_amount_usd": _json_usd(amount),
        "settlement_amount_usd": _json_usd(amount),
        "time_billed_ms": billed_ms,
        "billing_bucket_time": _iso_z(billing_time),
        "provider_billing_row_hash": raw_row_hash,
    }


def _source_hash(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RunpodRecoveryError(f"{label} source artifact is missing or unsafe")
    details = source.stat()
    if not stat.S_ISREG(details.st_mode):
        raise RunpodRecoveryError(f"{label} source artifact is not a regular file")
    return {
        "label": label,
        "sha256": f"sha256:{sha256_file(source)}",
        "size_bytes": details.st_size,
    }


def _write_receipt_idempotently(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise RunpodRecoveryError("external-stop receipt path must not be a symlink")
    encoded = (
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise RunpodRecoveryError("existing external-stop receipt has different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise RunpodRecoveryError(
                    "external-stop receipt was concurrently claimed"
                ) from exc
        path.chmod(0o600)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_external_stop_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RunpodRecoveryError("external-stop receipt is missing or unsafe")
    try:
        payload = read_json(source)
    except (OSError, ValueError) as exc:
        raise RunpodRecoveryError("external-stop receipt is unreadable") from exc
    if not isinstance(payload, dict):
        raise RunpodRecoveryError("external-stop receipt must be a JSON object")
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "provider_api",
        "observed_at",
        "prior_lifecycle_operation",
        "lifecycle_before_hash",
        "lifecycle_stopped_hash",
        "session_hash",
        "reservation_id",
        "reservation_record_hash",
        "pod_id_hash",
        "stop_evidence",
        "stop_evidence_hash",
        "billing_query",
        "billing_query_hash",
        "billing_evidence",
        "billing_evidence_hash",
        "billing_status",
        "evidence_kind",
        "settlement_amount_usd",
        "source_artifact_hashes",
        "record_hash",
    }
    if set(payload) != expected_keys:
        raise RunpodRecoveryError("external-stop receipt has an unexpected schema")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != EXTERNAL_STOP_PROTOCOL:
        raise RunpodRecoveryError("external-stop receipt protocol is unsupported")
    if payload.get("status") != "stopped_verified":
        raise RunpodRecoveryError("external-stop receipt is incomplete")
    if payload.get("provider_api") != "rest-v1-read-only":
        raise RunpodRecoveryError("external-stop receipt provider boundary is invalid")
    _parse_provider_timestamp(payload.get("observed_at"), field="receipt observation")
    record_hash = payload.get("record_hash")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if not isinstance(record_hash, str) or record_hash != stable_hash(unsigned):
        raise RunpodRecoveryError("external-stop receipt content hash mismatch")
    for field in (
        "session_hash",
        "reservation_id",
        "reservation_record_hash",
        "lifecycle_before_hash",
        "lifecycle_stopped_hash",
        "pod_id_hash",
        "stop_evidence_hash",
        "billing_query_hash",
        "billing_evidence_hash",
    ):
        if not isinstance(payload.get(field), str) or _HASH_RE.fullmatch(str(payload[field])) is None:
            raise RunpodRecoveryError(f"external-stop receipt {field} is malformed")
    if payload.get("billing_status") not in {"final", "pending"}:
        raise RunpodRecoveryError("external-stop receipt billing status is invalid")
    if payload.get("evidence_kind") not in {
        "provider_billing_row",
        "provider_timestamps_conservative_ceiling",
    }:
        raise RunpodRecoveryError("external-stop receipt evidence kind is invalid")
    amount = _decimal(payload.get("settlement_amount_usd"), field="settlement amount")
    stop_evidence = payload.get("stop_evidence")
    billing_query = payload.get("billing_query")
    billing = payload.get("billing_evidence")
    if not isinstance(stop_evidence, dict) or stable_hash(stop_evidence) != payload.get(
        "stop_evidence_hash"
    ):
        raise RunpodRecoveryError("external-stop receipt stop evidence hash mismatch")
    if not isinstance(billing_query, dict) or stable_hash(billing_query) != payload.get(
        "billing_query_hash"
    ):
        raise RunpodRecoveryError("external-stop receipt billing query hash mismatch")
    if not isinstance(billing, dict) or stable_hash(billing) != payload.get(
        "billing_evidence_hash"
    ):
        raise RunpodRecoveryError("external-stop receipt billing evidence hash mismatch")
    if (
        stop_evidence.get("desired_status") != "EXITED"
        or stop_evidence.get("environment_verified") is not True
        or billing_query.get("provider_api") != "rest-v1"
        or billing_query.get("method") != "GET"
        or billing_query.get("path") != "/v1/billing/pods"
        or billing_query.get("grouping") != "podId"
        or billing_query.get("pod_id_hash") != payload.get("pod_id_hash")
    ):
        raise RunpodRecoveryError("external-stop receipt evidence contract is invalid")
    if (
        billing.get("billing_status") != payload.get("billing_status")
        or billing.get("evidence_kind") != payload.get("evidence_kind")
        or billing.get("pod_id_hash") != payload.get("pod_id_hash")
        or _decimal(
            billing.get("settlement_amount_usd"), field="billing settlement amount"
        )
        != amount
    ):
        raise RunpodRecoveryError("external-stop receipt billing summary mismatch")
    if payload["billing_status"] == "pending":
        if (
            payload["evidence_kind"] != "provider_timestamps_conservative_ceiling"
            or billing.get("provider_amount_usd") is not None
            or billing.get("time_billed_ms") is not None
            or _decimal(
                billing.get("conservative_ceiling_usd"),
                field="conservative billing ceiling",
            )
            != amount
        ):
            raise RunpodRecoveryError("pending billing receipt overstates its evidence")
    elif (
        payload["evidence_kind"] != "provider_billing_row"
        or _decimal(billing.get("provider_amount_usd"), field="provider billing amount")
        != amount
        or isinstance(billing.get("time_billed_ms"), bool)
        or not isinstance(billing.get("time_billed_ms"), int)
        or billing.get("time_billed_ms") < 0
    ):
        raise RunpodRecoveryError("final billing receipt is incomplete")
    source_hashes = payload.get("source_artifact_hashes")
    if not isinstance(source_hashes, list):
        raise RunpodRecoveryError("external-stop source artifact hashes are malformed")
    labels: set[str] = set()
    for source_hash in source_hashes:
        if (
            not isinstance(source_hash, dict)
            or set(source_hash) != {"label", "sha256", "size_bytes"}
            or source_hash.get("label") not in {"failed_watchdog", "failed_log"}
            or source_hash.get("label") in labels
            or not isinstance(source_hash.get("sha256"), str)
            or _HASH_RE.fullmatch(str(source_hash["sha256"])) is None
            or isinstance(source_hash.get("size_bytes"), bool)
            or not isinstance(source_hash.get("size_bytes"), int)
            or source_hash.get("size_bytes") < 0
        ):
            raise RunpodRecoveryError("external-stop source artifact hash is malformed")
        labels.add(str(source_hash["label"]))
    return payload


def attest_external_stop(
    *,
    project_root: str | Path,
    client: RunpodRecoveryClient,
    output_path: str | Path,
    allow_pending_billing_ceiling: bool = False,
    failed_watchdog_path: str | Path | None = None,
    failed_log_path: str | Path | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Attest one already-stopped Pod and advance only the local lifecycle.

    The function never calls a provider write endpoint.  Empty billing is a
    hard failure unless ``allow_pending_billing_ceiling`` is explicitly true.
    """

    root = Path(project_root).resolve()
    lifecycle_path = root / ".runpod" / "pod_lifecycle.json"
    state = _load_lifecycle(lifecycle_path)
    current = (observed_at or datetime.now(UTC)).astimezone(UTC)
    prior_operation = state.get("operation")
    if prior_operation == "stopped":
        existing = load_external_stop_receipt(output_path)
        if existing.get("lifecycle_stopped_hash") != state.get("record_hash"):
            raise RunpodRecoveryError("stopped lifecycle disagrees with external-stop receipt")
        return existing
    if prior_operation not in {
        "created",
        "rearmed",
        "create_timeout",
        "create_verification_failed",
        "create_failed_terminal",
        "rearm_start_unverified",
        "rearm_failed",
    }:
        raise RunpodRecoveryError("lifecycle operation is not eligible for external-stop recovery")

    pod_id = _require_pod_id(dict(state["pod"]).get("id"))
    pod_payload = client.get_pod(pod_id)
    stop_evidence, exact_pod_id, all_in_rate, started_at, exited_at = _validate_pod(
        pod_payload,
        state=state,
        observed_at=current,
    )
    pod_id_hash = stable_hash({"runpod_pod_id": exact_pod_id})
    query_evidence = _billing_query_evidence(
        pod_id_hash=pod_id_hash,
        start_time=started_at,
        end_time=exited_at,
    )
    billing_rows = client.get_billing(
        pod_id=exact_pod_id,
        start_time=started_at,
        end_time=exited_at,
    )
    ceiling_minutes, ceiling_usd = _conservative_ceiling(
        start_time=started_at,
        end_time=exited_at,
        all_in_hourly_usd=all_in_rate,
    )
    authorization = dict(state["current_authorization"])
    approved_runtime = _decimal(
        authorization.get("approved_runtime_hours"),
        field="approved runtime",
        allow_zero=False,
    )
    if len(billing_rows) == 1:
        billing = _validate_billing_row(
            billing_rows[0],
            pod_id=exact_pod_id,
            pod_id_hash=pod_id_hash,
            expected_gpu_id=EXACT_PROVIDER_GPU_ID,
            conservative_ceiling=ceiling_usd,
            approved_runtime_hours=approved_runtime,
        )
        billing["conservative_ceiling_usd"] = _json_usd(ceiling_usd)
        billing["runtime_ceiling_minutes"] = ceiling_minutes
    elif not billing_rows and allow_pending_billing_ceiling:
        billing = {
            "billing_status": "pending",
            "evidence_kind": "provider_timestamps_conservative_ceiling",
            "pod_id_hash": pod_id_hash,
            "provider_amount_usd": None,
            "settlement_amount_usd": _json_usd(ceiling_usd),
            "time_billed_ms": None,
            "billing_bucket_time": None,
            "provider_billing_row_hash": None,
            "conservative_ceiling_usd": _json_usd(ceiling_usd),
            "runtime_ceiling_minutes": ceiling_minutes,
        }
    elif not billing_rows:
        raise RunpodRecoveryError("RunPod billing evidence is pending")
    else:
        raise RunpodRecoveryError("RunPod billing response is ambiguous; expected one row")

    sources: list[dict[str, Any]] = []
    if failed_watchdog_path is not None:
        sources.append(_source_hash(failed_watchdog_path, label="failed_watchdog"))
    if failed_log_path is not None:
        sources.append(_source_hash(failed_log_path, label="failed_log"))

    stopped_state = _authenticated_stopped_state(state, observed_at=current)
    stop_hash = stable_hash(stop_evidence)
    query_hash = stable_hash(query_evidence)
    billing_hash = stable_hash(billing)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": EXTERNAL_STOP_PROTOCOL,
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": _iso_z(current),
        "prior_lifecycle_operation": prior_operation,
        "lifecycle_before_hash": state["record_hash"],
        "lifecycle_stopped_hash": stopped_state["record_hash"],
        "session_hash": authorization.get("session_hash"),
        "reservation_id": authorization.get("reservation_id"),
        "reservation_record_hash": authorization.get("reservation_record_hash"),
        "pod_id_hash": pod_id_hash,
        "stop_evidence": stop_evidence,
        "stop_evidence_hash": stop_hash,
        "billing_query": query_evidence,
        "billing_query_hash": query_hash,
        "billing_evidence": billing,
        "billing_evidence_hash": billing_hash,
        "billing_status": billing["billing_status"],
        "evidence_kind": billing["evidence_kind"],
        "settlement_amount_usd": billing["settlement_amount_usd"],
        "source_artifact_hashes": sources,
    }
    receipt["record_hash"] = stable_hash(receipt)

    output = Path(output_path).resolve()
    session_digest = str(authorization.get("session_hash", "")).removeprefix("sha256:")
    expected_session_dir = root / ".runpod" / "sessions" / session_digest
    if output.parent != expected_session_dir.resolve():
        raise RunpodRecoveryError("external-stop receipt must stay in its private session directory")
    _write_receipt_idempotently(output, receipt)
    _atomic_replace_lifecycle(lifecycle_path, stopped_state)
    return receipt


def safe_recovery_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a secret-safe CLI summary with no raw Pod identity."""

    return {
        "schema_version": 1,
        "status": receipt.get("status"),
        "provider_status": "EXITED",
        "billing_status": receipt.get("billing_status"),
        "evidence_kind": receipt.get("evidence_kind"),
        "settlement_amount_usd": receipt.get("settlement_amount_usd"),
        "pod_id_hash": receipt.get("pod_id_hash"),
        "record_hash": receipt.get("record_hash"),
        "lifecycle_operation": "stopped",
        "passed": True,
    }


__all__ = [
    "EXTERNAL_STOP_PROTOCOL",
    "EXTERNAL_STOP_RECEIPT_FILENAME",
    "RUNPOD_BILLING_PODS_URL",
    "RecoveryHttpResult",
    "RunpodRecoveryClient",
    "RunpodRecoveryError",
    "attest_external_stop",
    "load_external_stop_receipt",
    "safe_recovery_summary",
    "urllib_recovery_transport",
]
