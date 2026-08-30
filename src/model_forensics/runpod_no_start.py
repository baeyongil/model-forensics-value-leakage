"""Authenticated reconciliation for RunPod re-arms that provably never started.

This path is deliberately capability-limited to the same official REST-v1
GET client used by external-stop recovery.  It may close only a pre-start
re-arm claim whose provider ``lastStartedAt`` remains byte-for-byte equal to
the baseline persisted before PATCH, whose Pod is still exactly EXITED, and
whose bounded billing query is empty.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash
from model_forensics.runpod_contract import (
    EXACT_CONTAINER_DISK_GB,
    EXACT_GPU_COUNT,
    EXACT_PORTS,
    EXACT_PROVIDER_GPU_ID,
    EXACT_VOLUME_DISK_GB,
    EXACT_VOLUME_MOUNT_PATH,
)
from model_forensics.runpod_recovery import (
    RunpodRecoveryClient,
    RunpodRecoveryError,
    _atomic_replace_lifecycle,
    _authenticated_stopped_state,
    _decimal,
    _iso_z,
    _load_lifecycle_snapshot,
    _parse_provider_timestamp,
    _require_pod_id,
    _validate_authorization_manifest,
    _validate_environment,
    _write_receipt_idempotently,
)

NO_START_PROTOCOL = "runpod-no-start-v1"
NO_START_RECEIPT_FILENAME = "no_start_receipt.json"
_ELIGIBLE_OPERATIONS = frozenset(
    {"rearm_intent", "rearm_patched", "rearm_start_intent"}
)
_CLOCK_SKEW = timedelta(minutes=5)
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROVIDER_EVIDENCE_KEYS = {
    "desired_status",
    "pod_id_hash",
    "name_hash",
    "image_hash",
    "machine_id_hash",
    "provider_binding_hash",
    "immutable_spec_hash",
    "gpu",
    "cloud",
    "data_center_id",
    "container_disk_gb",
    "persistent_disk_gb",
    "persistent_mount_path",
    "ports",
    "environment_verified",
    "environment_session_context",
    "pre_start_last_started_at",
    "observed_last_started_at",
    "last_started_at_unchanged",
    "provider_hourly_compute_usd",
    "approved_hourly_all_in_usd",
    "observation_count",
    "quiet_window_seconds",
    "first_observation_hash",
    "second_observation_hash",
}


class NoStartReconciliationError(RunpodRecoveryError):
    """A re-arm cannot be proven to have remained entirely pre-start."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NoStartReconciliationError("no-start receipt contains a duplicate JSON key")
        result[key] = value
    return result


def _require_private_output(
    *, root: Path, output_path: str | Path, session_hash: str
) -> Path:
    private = root / ".runpod"
    if private.is_symlink() or not private.is_dir():
        raise NoStartReconciliationError("private .runpod directory is missing or unsafe")
    expected = (
        private
        / "sessions"
        / session_hash.removeprefix("sha256:")
        / NO_START_RECEIPT_FILENAME
    )
    output = Path(output_path)
    if output.absolute() != expected.absolute():
        raise NoStartReconciliationError(
            "no-start receipt must use its canonical private session path"
        )
    sessions = expected.parent.parent
    if sessions.exists() and (sessions.is_symlink() or not sessions.is_dir()):
        raise NoStartReconciliationError("private sessions directory is unsafe")
    if expected.parent.exists() and (
        expected.parent.is_symlink() or not expected.parent.is_dir()
    ):
        raise NoStartReconciliationError("private session directory is unsafe")
    return expected


def _authorization_context(state: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    spec = state.get("immutable_spec")
    if not isinstance(spec, Mapping):
        raise NoStartReconciliationError("private lifecycle immutable specification is missing")
    immutable_hash = stable_hash(dict(spec))
    current = state.get("current_authorization")
    history = state.get("authorization_history")
    if not isinstance(current, Mapping) or not isinstance(history, list) or not history:
        raise NoStartReconciliationError("no-start re-arm authorization history is missing")
    try:
        current_identity = _validate_authorization_manifest(
            current,
            immutable_spec_hash=immutable_hash,
            label="current",
        )
        historical_identities = [
            _validate_authorization_manifest(
                item,
                immutable_spec_hash=immutable_hash,
                label="historical",
            )
            for item in history
        ]
    except RunpodRecoveryError as exc:
        raise NoStartReconciliationError(str(exc)) from exc
    sessions = [identity[0] for identity in historical_identities]
    reservations = [identity[1] for identity in historical_identities]
    if (
        len(sessions) != len(set(sessions))
        or len(reservations) != len(set(reservations))
        or current_identity[0] in set(sessions)
        or current_identity[1] in set(reservations)
    ):
        raise NoStartReconciliationError(
            "private lifecycle reuses a session or reservation authorization"
        )
    return dict(current), sessions[-1], immutable_hash


def _validate_provider_no_start(
    payload: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    current_authorization: Mapping[str, Any],
    prior_session_hash: str,
    immutable_spec_hash: str,
    observed_at: datetime,
) -> tuple[dict[str, Any], str, datetime]:
    operation = state.get("operation")
    if operation not in _ELIGIBLE_OPERATIONS:
        raise NoStartReconciliationError(
            "lifecycle operation is not eligible for no-start reconciliation"
        )
    stored = state.get("pod")
    spec = state.get("immutable_spec")
    if not isinstance(stored, Mapping) or not isinstance(spec, Mapping):
        raise NoStartReconciliationError("private lifecycle Pod record is missing")
    pod_id = _require_pod_id(stored.get("id"))
    if payload.get("id") != pod_id:
        raise NoStartReconciliationError("RunPod status returned a different Pod")
    if payload.get("desiredStatus") != "EXITED":
        raise NoStartReconciliationError("no-start reconciliation requires exactly EXITED")
    if payload.get("name") != stored.get("name"):
        raise NoStartReconciliationError("RunPod Pod name drifted")
    if payload.get("imageName") != spec.get("image") or payload.get("imageName") != stored.get(
        "image"
    ):
        raise NoStartReconciliationError("RunPod Pod image drifted")
    if payload.get("gpuCount") != EXACT_GPU_COUNT:
        raise NoStartReconciliationError("RunPod Pod GPU count drifted")
    machine = payload.get("machine")
    if not isinstance(machine, Mapping):
        raise NoStartReconciliationError("RunPod machine metadata is missing")
    if machine.get("gpuTypeId") != EXACT_PROVIDER_GPU_ID:
        raise NoStartReconciliationError("RunPod machine GPU type drifted")
    if machine.get("secureCloud") is not True:
        raise NoStartReconciliationError("RunPod Pod is not in Secure Cloud")
    approved_centers = spec.get("data_center_ids")
    if (
        not isinstance(approved_centers, list)
        or machine.get("dataCenterId") not in set(approved_centers)
        or machine.get("dataCenterId") != stored.get("data_center_id")
    ):
        raise NoStartReconciliationError("RunPod Pod data center drifted")
    machine_id = payload.get("machineId")
    if (
        not isinstance(machine_id, str)
        or stable_hash({"runpod_machine_id": machine_id}) != stored.get("machine_id_hash")
    ):
        raise NoStartReconciliationError("RunPod Pod machine identity drifted")
    expected_binding_hash = stable_hash(
        {"runpod_pod_id": pod_id, "data_center_id": machine.get("dataCenterId")}
    )
    if stored.get("provider_binding_hash") != expected_binding_hash:
        raise NoStartReconciliationError("private RunPod provider binding drifted")
    if (
        payload.get("containerDiskInGb") != EXACT_CONTAINER_DISK_GB
        or payload.get("volumeInGb") != EXACT_VOLUME_DISK_GB
        or payload.get("volumeMountPath") != EXACT_VOLUME_MOUNT_PATH
        or payload.get("ports") != list(EXACT_PORTS)
        or payload.get("networkVolume") is not None
        or payload.get("networkVolumeId") is not None
    ):
        raise NoStartReconciliationError("RunPod storage, mount, or ports drifted")
    environment_context = "current"
    try:
        _validate_environment(
            payload.get("env"),
            expected_session_hash=str(current_authorization["session_hash"]),
        )
    except RunpodRecoveryError as current_error:
        if operation != "rearm_intent":
            raise NoStartReconciliationError("RunPod current re-arm environment drifted") from current_error
        try:
            _validate_environment(
                payload.get("env"),
                expected_session_hash=prior_session_hash,
            )
        except RunpodRecoveryError as prior_error:
            raise NoStartReconciliationError(
                "RunPod environment matches neither side of the bounded PATCH"
            ) from prior_error
        environment_context = "prior"

    provider_hourly = _decimal(payload.get("costPerHr"), field="hourly cost", allow_zero=False)
    approved_hourly = _decimal(
        current_authorization.get("live_hourly_total_usd"),
        field="approved all-in hourly cost",
        allow_zero=False,
    )
    if provider_hourly > approved_hourly:
        raise NoStartReconciliationError(
            "RunPod hourly cost exceeds the approved all-in quote"
        )
    baseline = stored.get("pre_start_last_started_at")
    if not isinstance(baseline, str) or payload.get("lastStartedAt") != baseline:
        raise NoStartReconciliationError(
            "RunPod lastStartedAt changed or is ambiguous after the durable baseline"
        )
    try:
        baseline_at = _parse_provider_timestamp(baseline, field="pre-start baseline")
    except RunpodRecoveryError as exc:
        raise NoStartReconciliationError(str(exc)) from exc
    if baseline_at > observed_at + _CLOCK_SKEW:
        raise NoStartReconciliationError("RunPod pre-start baseline is implausibly in the future")
    pod_id_hash = stable_hash({"runpod_pod_id": pod_id})
    provider_evidence = {
        "desired_status": "EXITED",
        "pod_id_hash": pod_id_hash,
        "name_hash": stable_hash({"runpod_pod_name": payload.get("name")}),
        "image_hash": stable_hash({"runpod_image": payload.get("imageName")}),
        "machine_id_hash": stable_hash({"runpod_machine_id": machine_id}),
        "provider_binding_hash": expected_binding_hash,
        "immutable_spec_hash": immutable_spec_hash,
        "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": EXACT_GPU_COUNT},
        "cloud": "SECURE",
        "data_center_id": machine.get("dataCenterId"),
        "container_disk_gb": payload.get("containerDiskInGb"),
        "persistent_disk_gb": payload.get("volumeInGb"),
        "persistent_mount_path": payload.get("volumeMountPath"),
        "ports": list(payload.get("ports", [])),
        "environment_verified": True,
        "environment_session_context": environment_context,
        "pre_start_last_started_at": baseline,
        "observed_last_started_at": payload.get("lastStartedAt"),
        "last_started_at_unchanged": True,
        "provider_hourly_compute_usd": float(provider_hourly),
        "approved_hourly_all_in_usd": float(approved_hourly),
    }
    return provider_evidence, pod_id, baseline_at


def _validate_receipt_payload(payload: Any) -> dict[str, Any]:
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
        "provider_evidence",
        "provider_evidence_hash",
        "billing_query",
        "billing_query_hash",
        "billing_evidence",
        "billing_evidence_hash",
        "accounted_gpu_usd",
        "record_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise NoStartReconciliationError("no-start receipt has an unexpected schema")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_version") != NO_START_PROTOCOL
        or payload.get("status") != "no_start_verified"
        or payload.get("provider_api") != "rest-v1-read-only"
        or payload.get("record_hash") != stable_hash(unsigned)
    ):
        raise NoStartReconciliationError("no-start receipt authentication failed")
    if payload.get("accounted_gpu_usd") != 0.0:
        raise NoStartReconciliationError("no-start receipt must account exactly zero GPU cost")
    if payload.get("prior_lifecycle_operation") not in _ELIGIBLE_OPERATIONS:
        raise NoStartReconciliationError("no-start receipt lifecycle operation is ineligible")
    provider = payload.get("provider_evidence")
    query = payload.get("billing_query")
    billing = payload.get("billing_evidence")
    if (
        not isinstance(provider, dict)
        or set(provider) != _PROVIDER_EVIDENCE_KEYS
        or stable_hash(provider) != payload.get("provider_evidence_hash")
        or provider.get("desired_status") != "EXITED"
        or provider.get("pod_id_hash") != payload.get("pod_id_hash")
        or provider.get("environment_verified") is not True
        or provider.get("last_started_at_unchanged") is not True
        or provider.get("pre_start_last_started_at")
        != provider.get("observed_last_started_at")
        or provider.get("gpu")
        != {"id": EXACT_PROVIDER_GPU_ID, "count": EXACT_GPU_COUNT}
        or provider.get("cloud") != "SECURE"
        or provider.get("container_disk_gb") != EXACT_CONTAINER_DISK_GB
        or provider.get("persistent_disk_gb") != EXACT_VOLUME_DISK_GB
        or provider.get("persistent_mount_path") != EXACT_VOLUME_MOUNT_PATH
        or provider.get("ports") != list(EXACT_PORTS)
        or provider.get("environment_session_context") not in {"prior", "current"}
    ):
        raise NoStartReconciliationError("no-start provider evidence is invalid")
    if payload.get("prior_lifecycle_operation") == "rearm_start_intent":
        if (
            provider.get("observation_count") != 2
            or not isinstance(provider.get("quiet_window_seconds"), (int, float))
            or float(provider["quiet_window_seconds"]) < 30
            or provider.get("first_observation_hash")
            != provider.get("second_observation_hash")
        ):
            raise NoStartReconciliationError(
                "no-start uncertain start intent lacks repeated quiet-window evidence"
            )
    elif (
        provider.get("observation_count") != 1
        or provider.get("quiet_window_seconds") != 0.0
        or provider.get("second_observation_hash") is not None
    ):
        raise NoStartReconciliationError("no-start provider observation count is invalid")
    if (
        not isinstance(query, dict)
        or set(query)
        != {
            "provider_api",
            "method",
            "path",
            "grouping",
            "pod_id_hash",
            "start_time",
            "end_time",
        }
        or stable_hash(query) != payload.get("billing_query_hash")
        or query.get("provider_api") != "rest-v1"
        or query.get("method") != "GET"
        or query.get("path") != "/v1/billing/pods"
        or query.get("grouping") != "podId"
        or query.get("pod_id_hash") != payload.get("pod_id_hash")
    ):
        raise NoStartReconciliationError("no-start billing query evidence is invalid")
    if (
        not isinstance(billing, dict)
        or set(billing) != {"row_count", "response_hash"}
        or stable_hash(billing) != payload.get("billing_evidence_hash")
        or billing.get("row_count") != 0
        or billing.get("response_hash") != stable_hash([])
    ):
        raise NoStartReconciliationError("no-start billing evidence is not empty")
    for field in (
        "lifecycle_before_hash",
        "lifecycle_stopped_hash",
        "session_hash",
        "reservation_id",
        "reservation_record_hash",
        "pod_id_hash",
        "provider_evidence_hash",
        "billing_query_hash",
        "billing_evidence_hash",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
            raise NoStartReconciliationError(f"no-start receipt {field} is malformed")
    try:
        receipt_observed_at = _parse_provider_timestamp(
            payload.get("observed_at"), field="receipt observation"
        )
        query_start_at = _parse_provider_timestamp(
            query.get("start_time"), field="billing query start"
        )
        query_end_at = _parse_provider_timestamp(
            query.get("end_time"), field="billing query end"
        )
    except RunpodRecoveryError as exc:
        raise NoStartReconciliationError(str(exc)) from exc
    if query_start_at >= query_end_at or query_end_at != receipt_observed_at:
        raise NoStartReconciliationError("no-start billing query window is invalid")
    return payload


def _cas_stopped_lifecycle(
    path: Path,
    stopped_state: Mapping[str, Any],
    *,
    expected_before: bytes,
    expected_record_hash: str,
) -> None:
    try:
        _atomic_replace_lifecycle(
            path,
            stopped_state,
            expected_before=expected_before,
            expected_record_hash=expected_record_hash,
        )
    except RunpodRecoveryError as exc:
        raise NoStartReconciliationError(str(exc)) from exc


def load_no_start_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise NoStartReconciliationError("no-start receipt is missing or unsafe")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise NoStartReconciliationError("no-start receipt is unreadable") from exc
    return _validate_receipt_payload(payload)


def _revalidate_receipt_before_lifecycle_cas(
    *,
    client: RunpodRecoveryClient,
    state: Mapping[str, Any],
    authorization: Mapping[str, Any],
    prior_session_hash: str,
    immutable_hash: str,
    receipt: Mapping[str, Any],
    observed_at: datetime,
    quiet_window_seconds: float,
    sleep: Callable[[float], None],
) -> None:
    """Repeat provider proof when recovering a receipt written before CAS."""

    receipt_time = _parse_provider_timestamp(
        receipt.get("observed_at"), field="receipt observation"
    )
    current = observed_at.astimezone(UTC)
    if current < receipt_time:
        raise NoStartReconciliationError(
            "no-start receipt recovery observation predates the durable receipt"
        )
    pod_id = _require_pod_id(dict(state["pod"]).get("id"))
    first_provider, exact_pod_id, _baseline_at = _validate_provider_no_start(
        client.get_pod(pod_id),
        state=state,
        current_authorization=authorization,
        prior_session_hash=prior_session_hash,
        immutable_spec_hash=immutable_hash,
        observed_at=current,
    )
    if state.get("operation") == "rearm_start_intent":
        sleep(float(quiet_window_seconds))
        current = current + timedelta(seconds=float(quiet_window_seconds))
        second_provider, second_pod_id, _second_baseline_at = _validate_provider_no_start(
            client.get_pod(pod_id),
            state=state,
            current_authorization=authorization,
            prior_session_hash=prior_session_hash,
            immutable_spec_hash=immutable_hash,
            observed_at=current,
        )
        if second_pod_id != exact_pod_id or second_provider != first_provider:
            raise NoStartReconciliationError(
                "RunPod no-start identity changed across the recovery quiet window"
            )
    query = receipt.get("billing_query")
    if not isinstance(query, Mapping):
        raise NoStartReconciliationError("no-start receipt billing query is missing")
    query_start = _parse_provider_timestamp(
        query.get("start_time"), field="billing query start"
    )
    if query_start >= current:
        raise NoStartReconciliationError(
            "no-start receipt recovery billing window is missing or implausible"
        )
    rows = client.get_billing(
        pod_id=exact_pod_id,
        start_time=query_start,
        end_time=current,
    )
    if not isinstance(rows, list) or rows:
        raise NoStartReconciliationError(
            "RunPod returned billing evidence during no-start receipt recovery"
        )


def attest_no_start(
    *,
    project_root: str | Path,
    client: RunpodRecoveryClient,
    output_path: str | Path,
    observed_at: datetime | None = None,
    quiet_window_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Prove one re-arm never started, then close only its local lifecycle claim."""

    root = Path(project_root).absolute()
    lifecycle_path = root / ".runpod" / "pod_lifecycle.json"
    state, lifecycle_before_raw = _load_lifecycle_snapshot(lifecycle_path)
    if (
        isinstance(quiet_window_seconds, bool)
        or not isinstance(quiet_window_seconds, (int, float))
        or not math.isfinite(float(quiet_window_seconds))
        or not 30 <= float(quiet_window_seconds) <= 300
    ):
        raise ValueError("no-start quiet window must be between 30 and 300 seconds")
    current = (observed_at or datetime.now(UTC)).astimezone(UTC)
    authorization, prior_session_hash, immutable_hash = _authorization_context(state)
    output = _require_private_output(
        root=root,
        output_path=output_path,
        session_hash=str(authorization["session_hash"]),
    )
    prior_operation = state.get("operation")
    if prior_operation == "stopped":
        existing = load_no_start_receipt(output)
        if existing.get("lifecycle_stopped_hash") != state.get("record_hash"):
            raise NoStartReconciliationError(
                "stopped lifecycle disagrees with no-start receipt"
            )
        return existing
    if output.exists():
        existing = load_no_start_receipt(output)
        for field in ("session_hash", "reservation_id", "reservation_record_hash"):
            if existing.get(field) != authorization.get(field):
                raise NoStartReconciliationError(
                    f"existing no-start receipt {field} disagrees with lifecycle"
                )
        if (
            existing.get("prior_lifecycle_operation") != prior_operation
            or existing.get("lifecycle_before_hash") != state.get("record_hash")
        ):
            raise NoStartReconciliationError(
                "existing no-start receipt does not continue this lifecycle"
            )
        _revalidate_receipt_before_lifecycle_cas(
            client=client,
            state=state,
            authorization=authorization,
            prior_session_hash=prior_session_hash,
            immutable_hash=immutable_hash,
            receipt=existing,
            observed_at=current,
            quiet_window_seconds=float(quiet_window_seconds),
            sleep=sleep,
        )
        receipt_time = _parse_provider_timestamp(
            existing.get("observed_at"), field="receipt observation"
        )
        stopped_state = _authenticated_stopped_state(state, observed_at=receipt_time)
        if stopped_state.get("record_hash") != existing.get("lifecycle_stopped_hash"):
            raise NoStartReconciliationError(
                "existing no-start receipt stopped-state binding is invalid"
            )
        _cas_stopped_lifecycle(
            lifecycle_path,
            stopped_state,
            expected_before=lifecycle_before_raw,
            expected_record_hash=str(existing["lifecycle_before_hash"]),
        )
        return existing
    if prior_operation not in _ELIGIBLE_OPERATIONS:
        raise NoStartReconciliationError(
            "lifecycle operation is not eligible for no-start reconciliation"
        )

    pod_id = _require_pod_id(dict(state["pod"]).get("id"))
    provider_payload = client.get_pod(pod_id)
    first_provider, exact_pod_id, _baseline_at = _validate_provider_no_start(
        provider_payload,
        state=state,
        current_authorization=authorization,
        prior_session_hash=prior_session_hash,
        immutable_spec_hash=immutable_hash,
        observed_at=current,
    )
    if prior_operation == "rearm_start_intent":
        sleep(float(quiet_window_seconds))
        second_observed_at = (
            current + timedelta(seconds=float(quiet_window_seconds))
            if observed_at is not None
            else datetime.now(UTC)
        )
        second_payload = client.get_pod(pod_id)
        second_provider, second_pod_id, _second_baseline_at = _validate_provider_no_start(
            second_payload,
            state=state,
            current_authorization=authorization,
            prior_session_hash=prior_session_hash,
            immutable_spec_hash=immutable_hash,
            observed_at=second_observed_at,
        )
        if second_pod_id != exact_pod_id or second_provider != first_provider:
            raise NoStartReconciliationError(
                "RunPod no-start identity changed across the quiet window"
            )
        provider = {
            **second_provider,
            "observation_count": 2,
            "quiet_window_seconds": float(quiet_window_seconds),
            "first_observation_hash": stable_hash(first_provider),
            "second_observation_hash": stable_hash(second_provider),
        }
        current = second_observed_at
    else:
        provider = {
            **first_provider,
            "observation_count": 1,
            "quiet_window_seconds": 0.0,
            "first_observation_hash": stable_hash(first_provider),
            "second_observation_hash": None,
        }
    query_start = _parse_provider_timestamp(
        state.get("updated_at"), field="lifecycle update"
    )
    if query_start > current + _CLOCK_SKEW or query_start >= current:
        raise NoStartReconciliationError(
            "no-start billing window is missing or implausible"
        )
    billing_rows = client.get_billing(
        pod_id=exact_pod_id,
        start_time=query_start,
        end_time=current,
    )
    if not isinstance(billing_rows, list) or billing_rows:
        raise NoStartReconciliationError(
            "RunPod returned billing evidence for the alleged no-start window"
        )
    pod_id_hash = stable_hash({"runpod_pod_id": exact_pod_id})
    query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": _iso_z(query_start),
        "end_time": _iso_z(current),
    }
    billing = {"row_count": 0, "response_hash": stable_hash([])}
    stopped_state = _authenticated_stopped_state(state, observed_at=current)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": NO_START_PROTOCOL,
        "status": "no_start_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": _iso_z(current),
        "prior_lifecycle_operation": prior_operation,
        "lifecycle_before_hash": state["record_hash"],
        "lifecycle_stopped_hash": stopped_state["record_hash"],
        "session_hash": authorization["session_hash"],
        "reservation_id": authorization["reservation_id"],
        "reservation_record_hash": authorization["reservation_record_hash"],
        "pod_id_hash": pod_id_hash,
        "provider_evidence": provider,
        "provider_evidence_hash": stable_hash(provider),
        "billing_query": query,
        "billing_query_hash": stable_hash(query),
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "accounted_gpu_usd": 0.0,
    }
    receipt["record_hash"] = stable_hash(receipt)
    _write_receipt_idempotently(output, receipt)
    _cas_stopped_lifecycle(
        lifecycle_path,
        stopped_state,
        expected_before=lifecycle_before_raw,
        expected_record_hash=str(receipt["lifecycle_before_hash"]),
    )
    return receipt


def safe_no_start_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": receipt.get("status"),
        "provider_status": "EXITED",
        "evidence_kind": "provider_no_start",
        "accounted_gpu_usd": receipt.get("accounted_gpu_usd"),
        "pod_id_hash": receipt.get("pod_id_hash"),
        "record_hash": receipt.get("record_hash"),
        "lifecycle_operation": "stopped",
        "passed": True,
    }


__all__ = [
    "NO_START_PROTOCOL",
    "NO_START_RECEIPT_FILENAME",
    "NoStartReconciliationError",
    "attest_no_start",
    "load_no_start_receipt",
    "safe_no_start_summary",
]
