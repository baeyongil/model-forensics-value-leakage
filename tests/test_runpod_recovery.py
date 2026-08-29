from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from model_forensics.io import stable_hash
from model_forensics.runpod_lifecycle import (
    EXACT_PROVIDER_GPU_ID,
    LIFECYCLE_PROTOCOL,
    pod_environment,
)
from model_forensics.runpod_recovery import (
    RecoveryHttpResult,
    RunpodRecoveryClient,
    RunpodRecoveryError,
    attest_external_stop,
    load_external_stop_receipt,
    safe_recovery_summary,
)

POD_ID = "pod-recovery-123"
NONCE = "opaque-recovery-session-fixture"
HF_FIXTURE_VALUE = "hf_fixture_value"
RUNPOD_CREDENTIAL_FIXTURE = "runpod_fixture_value"
IMAGE = "runpod/pytorch@sha256:" + "a" * 64
STARTED = "2026-08-29 19:24:57.637 +0000 UTC"
CREATED = "2026-08-29 19:24:57.642 +0000 UTC"
EXITED = "Exited by user: Sat Aug 29 2026 19:41:22 GMT+0000 (Coordinated Universal Time)"
OBSERVED = datetime(2026, 8, 29, 20, tzinfo=UTC)
ALL_IN_RATE = 26.41722222222222


class FakeClient:
    def __init__(self, pod: dict[str, Any], billing: list[Any]) -> None:
        self.pod = pod
        self.billing = billing
        self.calls: list[tuple[str, str]] = []

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        self.calls.append(("pod", pod_id))
        return deepcopy(self.pod)

    def get_billing(self, *, pod_id: str, start_time: datetime, end_time: datetime) -> list[Any]:
        self.calls.append(("billing", pod_id))
        assert start_time < end_time
        return deepcopy(self.billing)


def _spec() -> dict[str, Any]:
    return {
        "image": IMAGE,
        "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": 8},
        "cloud": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["CA-MTL-1"],
        "disk": 50,
        "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
        "ports": ["22/tcp"],
        "global_networking": False,
        "start_jupyter": False,
        "start_ssh": True,
        "network_volume": None,
    }


def _project(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    session_hash = stable_hash({"opaque_gpu_session_id": NONCE})
    spec = _spec()
    authorization = {
        "phase": "behavior_baseline_gpu",
        "reservation_id": stable_hash({"reservation": "baseline"}),
        "reservation_record_hash": stable_hash({"receipt": "baseline"}),
        "session_hash": session_hash,
        "approval_hash": stable_hash({"approval": "baseline"}),
        "bindings_hash": stable_hash({"bindings": "baseline"}),
        "gpu_lock_hash": stable_hash({"gpu_lock": "baseline"}),
        "quote_hash": stable_hash({"quote": "baseline"}),
        "immutable_spec_hash": stable_hash(spec),
        "launch_spec_hash": stable_hash({"launch": "baseline"}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 1.5,
        "approved_phase_maximum_usd": 39.625834,
        "live_hourly_total_usd": ALL_IN_RATE,
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": LIFECYCLE_PROTOCOL,
        "operation": "created",
        "updated_at": "2026-08-29T19:25:00Z",
        "immutable_spec": spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {
            "id": POD_ID,
            "name": "model-forensics-behavior-baseline",
            "status": "RUNNING",
            "image": IMAGE,
            "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": 8},
            "cloud": "SECURE",
            "data_center_id": "CA-MTL-1",
            "cuda_version": None,
            "disk": 50,
            "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
            "ports": ["22/tcp"],
            "global_networking": None,
            "provider_binding_hash": stable_hash(
                {"runpod_pod_id": POD_ID, "data_center_id": "CA-MTL-1"}
            ),
            "ssh": {"proxy": None, "direct": {"private": "must-not-persist-after-stop"}},
        },
    }
    state["record_hash"] = stable_hash(state)
    lifecycle = tmp_path / ".runpod" / "pod_lifecycle.json"
    lifecycle.parent.mkdir(parents=True)
    lifecycle.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lifecycle.chmod(0o600)
    output = (
        tmp_path
        / ".runpod"
        / "sessions"
        / session_hash.removeprefix("sha256:")
        / "external_stop_receipt.json"
    )
    return lifecycle, output, authorization


def _pod(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": POD_ID,
        "name": "model-forensics-behavior-baseline",
        "desiredStatus": "EXITED",
        "imageName": IMAGE,
        "gpuCount": 8,
        "machineId": "private-machine-id",
        "machine": {
            "gpuTypeId": EXACT_PROVIDER_GPU_ID,
            "dataCenterId": "CA-MTL-1",
            "secureCloud": True,
        },
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "env": {
            **pod_environment(hf_token=HF_FIXTURE_VALUE, session_nonce=NONCE),
            "PUBLIC_KEY": "ssh",
        },
        "costPerHr": 26.32,
        "createdAt": CREATED,
        "lastStartedAt": STARTED,
        "lastStatusChange": EXITED,
    }
    payload.update(changes)
    return payload


def _billing(**changes: Any) -> dict[str, Any]:
    row = {
        "amount": 7.21,
        "diskSpaceBilledGb": 50,
        "gpuTypeId": EXACT_PROVIDER_GPU_ID,
        "podId": POD_ID,
        "time": "2026-08-29T19:00:00Z",
        "timeBilledMs": 984363,
    }
    row.update(changes)
    return row


def test_pending_billing_ceiling_is_explicit_secret_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    lifecycle, output, _authorization = _project(tmp_path)
    failed_watchdog = tmp_path / "failed-watchdog.json"
    failed_log = tmp_path / "failed.log"
    failed_watchdog.write_text(f'{{"pod":"{POD_ID}","status":"armed"}}', encoding="utf-8")
    failed_log.write_text(f"bootstrap failed for {POD_ID} with {NONCE}", encoding="utf-8")
    client = FakeClient(_pod(), [])

    receipt = attest_external_stop(
        project_root=tmp_path,
        client=client,  # type: ignore[arg-type]
        output_path=output,
        allow_pending_billing_ceiling=True,
        failed_watchdog_path=failed_watchdog,
        failed_log_path=failed_log,
        observed_at=OBSERVED,
    )

    assert receipt["billing_status"] == "pending"
    assert receipt["evidence_kind"] == "provider_timestamps_conservative_ceiling"
    assert receipt["settlement_amount_usd"] == pytest.approx(7.484880)
    assert receipt["billing_evidence"]["runtime_ceiling_minutes"] == 17
    assert receipt["billing_evidence"]["provider_amount_usd"] is None
    assert receipt["billing_evidence"]["time_billed_ms"] is None
    assert [item["label"] for item in receipt["source_artifact_hashes"]] == [
        "failed_watchdog",
        "failed_log",
    ]
    persisted = output.read_text(encoding="utf-8")
    assert POD_ID not in persisted and NONCE not in persisted
    assert HF_FIXTURE_VALUE not in persisted
    stopped = json.loads(lifecycle.read_text(encoding="utf-8"))
    assert stopped["operation"] == "stopped"
    assert stopped["pod"]["status"] == "EXITED"
    assert stopped["pod"]["ssh"] == {"direct": None, "proxy": None}
    assert stopped["record_hash"] == receipt["lifecycle_stopped_hash"]
    assert load_external_stop_receipt(output) == receipt
    assert POD_ID not in json.dumps(safe_recovery_summary(receipt))

    before = output.read_bytes()
    repeated = attest_external_stop(
        project_root=tmp_path,
        client=FakeClient({}, []),  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )
    assert repeated == receipt
    assert output.read_bytes() == before


def test_unique_provider_billing_row_is_bound_as_final(tmp_path: Path) -> None:
    _lifecycle, output, _authorization = _project(tmp_path)
    receipt = attest_external_stop(
        project_root=tmp_path,
        client=FakeClient(_pod(), [_billing()]),  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )

    assert receipt["billing_status"] == "final"
    assert receipt["evidence_kind"] == "provider_billing_row"
    assert receipt["settlement_amount_usd"] == pytest.approx(7.21)
    assert receipt["billing_evidence"]["time_billed_ms"] == 984363
    assert receipt["billing_evidence"]["provider_billing_row_hash"].startswith("sha256:")


def test_billing_lag_fails_closed_without_explicit_ceiling_opt_in(tmp_path: Path) -> None:
    lifecycle, output, _authorization = _project(tmp_path)
    before = lifecycle.read_bytes()

    with pytest.raises(RunpodRecoveryError, match="billing evidence is pending"):
        attest_external_stop(
            project_root=tmp_path,
            client=FakeClient(_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )

    assert lifecycle.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize(
    "rows,match",
    [
        ([_billing(), _billing(amount=7.22)], "ambiguous"),
        ([_billing(podId="different-pod")], "different Pod"),
        ([_billing(amount=7.6)], "timestamp-derived ceiling"),
        ([_billing(timeBilledMs=99_999_999)], "approved runtime"),
    ],
)
def test_billing_duplicate_wrong_pod_and_bounds_are_rejected(
    tmp_path: Path, rows: list[dict[str, Any]], match: str
) -> None:
    lifecycle, output, _authorization = _project(tmp_path)
    before = lifecycle.read_bytes()
    with pytest.raises(RunpodRecoveryError, match=match):
        attest_external_stop(
            project_root=tmp_path,
            client=FakeClient(_pod(), rows),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert lifecycle.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda value: value.update(desiredStatus="RUNNING"), "not exactly EXITED"),
        (lambda value: value.update(id="wrong-pod"), "different Pod"),
        (lambda value: value.update(name="wrong-name"), "name drifted"),
        (lambda value: value.update(imageName="wrong-image"), "image drifted"),
        (lambda value: value.update(gpuCount=7), "GPU count"),
        (lambda value: value["machine"].update(gpuTypeId="NVIDIA A100"), "GPU type"),
        (lambda value: value["machine"].update(dataCenterId="US-TX-1"), "data center"),
        (lambda value: value["machine"].update(secureCloud=False), "Secure Cloud"),
        (lambda value: value.update(containerDiskInGb=51), "disk or persistent"),
        (lambda value: value.update(volumeInGb=649), "disk or persistent"),
        (lambda value: value.update(volumeMountPath="/wrong"), "disk or persistent"),
        (lambda value: value.update(ports=["22/tcp", "8888/http"]), "ports"),
        (
            lambda value: value["env"].update(GPU_BUDGET_SESSION_ID="wrong-nonce"),
            "session nonce drifted",
        ),
        (lambda value: value.update(costPerHr=30), "approved all-in quote"),
        (
            lambda value: value.update(lastStartedAt="2026-08-29T19:50:00Z"),
            "creation and start",
        ),
        (
            lambda value: value.update(lastStatusChange="2026-08-29T19:00:00Z"),
            "start/exit",
        ),
    ],
)
def test_pod_status_identity_spec_environment_cost_and_timestamps_are_exact(
    tmp_path: Path, mutator: Any, match: str
) -> None:
    lifecycle, output, _authorization = _project(tmp_path)
    payload = _pod()
    mutator(payload)
    before = lifecycle.read_bytes()
    with pytest.raises(RunpodRecoveryError, match=match):
        attest_external_stop(
            project_root=tmp_path,
            client=FakeClient(payload, [_billing()]),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert lifecycle.read_bytes() == before
    assert not output.exists()


def test_real_client_uses_only_exact_rest_v1_get_endpoints_and_redacts_output() -> None:
    calls: list[dict[str, Any]] = []
    responses = [
        RecoveryHttpResult(200, json.dumps(_pod()).encode()),
        RecoveryHttpResult(200, json.dumps([_billing()]).encode()),
    ]

    def transport(**kwargs: Any) -> RecoveryHttpResult:
        calls.append(kwargs)
        return responses.pop(0)

    client = RunpodRecoveryClient(api_key=RUNPOD_CREDENTIAL_FIXTURE, transport=transport)
    assert client.get_pod(POD_ID)["desiredStatus"] == "EXITED"
    rows = client.get_billing(
        pod_id=POD_ID,
        start_time=datetime(2026, 8, 29, 19, 24, tzinfo=UTC),
        end_time=datetime(2026, 8, 29, 19, 42, tzinfo=UTC),
    )
    assert len(rows) == 1
    assert [call["method"] for call in calls] == ["GET", "GET"]
    assert all(call["body"] is None for call in calls)
    assert "/v1/pods/" in calls[0]["url"]
    assert "/v1/billing/pods?" in calls[1]["url"]
    assert "grouping=podId" in calls[1]["url"]
    assert "startTime=" in calls[1]["url"] and "endTime=" in calls[1]["url"]
