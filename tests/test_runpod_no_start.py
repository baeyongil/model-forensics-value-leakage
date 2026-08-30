from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import model_forensics.runpod_no_start as no_start_module
from model_forensics.io import stable_hash
from model_forensics.runpod_contract import EXACT_PROVIDER_GPU_ID, LIFECYCLE_PROTOCOL
from model_forensics.runpod_lifecycle import pod_environment
from model_forensics.runpod_no_start import (
    NO_START_RECEIPT_FILENAME,
    NoStartReconciliationError,
    attest_no_start,
    load_no_start_receipt,
)

POD_ID = "pod-no-start-123"
MACHINE_ID = "machine-no-start-123"
IMAGE = "runpod/pytorch@sha256:" + "b" * 64
OLD_NONCE = "old-no-start-session"
NEW_NONCE = "new-no-start-session"
HF_PLACEHOLDER = "hf_no_start_fixture"
BASELINE = "2026-08-30T00:00:00Z"
OBSERVED = datetime(2026, 8, 30, 0, 10, tzinfo=UTC)


class FakeClient:
    def __init__(self, pod: dict[str, Any], billing: list[Any]) -> None:
        self.pod = pod
        self.billing = billing
        self.calls: list[tuple[str, str]] = []

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        self.calls.append(("GET /v1/pods", pod_id))
        return deepcopy(self.pod)

    def get_billing(
        self,
        *,
        pod_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Any]:
        self.calls.append(("GET /v1/billing/pods", pod_id))
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


def _authorization(*, nonce: str, suffix: str) -> dict[str, Any]:
    spec = _spec()
    return {
        "phase": f"behavior_{suffix}_gpu",
        "reservation_id": stable_hash({"reservation": suffix}),
        "reservation_record_hash": stable_hash({"receipt": suffix}),
        "session_hash": stable_hash({"opaque_gpu_session_id": nonce}),
        "approval_hash": stable_hash({"approval": suffix}),
        "bindings_hash": stable_hash({"bindings": suffix}),
        "gpu_lock_hash": stable_hash({"gpu_lock": "fixed"}),
        "quote_hash": stable_hash({"quote": "fixed"}),
        "immutable_spec_hash": stable_hash(spec),
        "launch_spec_hash": stable_hash({"launch": suffix}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 1.0,
        "approved_phase_maximum_usd": 24.5,
        "live_hourly_total_usd": 24.5,
    }


def _project(
    tmp_path: Path,
    *,
    operation: str = "rearm_patched",
) -> tuple[Path, Path, dict[str, Any]]:
    prior = _authorization(nonce=OLD_NONCE, suffix="prior")
    current = _authorization(nonce=NEW_NONCE, suffix="current")
    pod = {
        "id": POD_ID,
        "name": "model-forensics-behavior",
        "status": "EXITED",
        "image": IMAGE,
        "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": 8},
        "cloud": "SECURE",
        "machine_id_hash": stable_hash({"runpod_machine_id": MACHINE_ID}),
        "data_center_id": "CA-MTL-1",
        "cuda_version": None,
        "disk": 50,
        "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
        "ports": ["22/tcp"],
        "global_networking": None,
        "provider_api": "rest-v1",
        "provider_evidence_unavailable": [
            "cuda_version",
            "global_networking",
            "interruptible",
            "locked",
        ],
        "provider_binding_hash": stable_hash(
            {"runpod_pod_id": POD_ID, "data_center_id": "CA-MTL-1"}
        ),
        "ssh": {"proxy": None, "direct": None},
        "pre_start_last_started_at": BASELINE,
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": LIFECYCLE_PROTOCOL,
        "operation": operation,
        "updated_at": "2026-08-30T00:01:00Z",
        "immutable_spec": _spec(),
        "current_authorization": current,
        "authorization_history": [prior],
        "pod": pod,
    }
    state["record_hash"] = stable_hash(state)
    lifecycle = tmp_path / ".runpod" / "pod_lifecycle.json"
    lifecycle.parent.mkdir(parents=True)
    lifecycle.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    lifecycle.chmod(0o600)
    output = (
        tmp_path
        / ".runpod"
        / "sessions"
        / current["session_hash"].removeprefix("sha256:")
        / NO_START_RECEIPT_FILENAME
    )
    return lifecycle, output, current


def _provider_pod(*, nonce: str = NEW_NONCE, **changes: Any) -> dict[str, Any]:
    payload = {
        "id": POD_ID,
        "name": "model-forensics-behavior",
        "desiredStatus": "EXITED",
        "imageName": IMAGE,
        "gpuCount": 8,
        "machineId": MACHINE_ID,
        "machine": {
            "gpuTypeId": EXACT_PROVIDER_GPU_ID,
            "dataCenterId": "CA-MTL-1",
            "secureCloud": True,
        },
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "networkVolume": None,
        "networkVolumeId": None,
        "ports": ["22/tcp"],
        "env": {
            **pod_environment(hf_token=HF_PLACEHOLDER, session_nonce=nonce),
            "PUBLIC_KEY": "ssh",
        },
        "costPerHr": 24.0,
        "lastStartedAt": BASELINE,
    }
    payload.update(changes)
    return payload


def test_no_start_receipt_is_zero_authenticated_and_idempotent(tmp_path: Path) -> None:
    lifecycle, output, current = _project(tmp_path)
    client = FakeClient(_provider_pod(), [])

    receipt = attest_no_start(
        project_root=tmp_path,
        client=client,  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )

    assert client.calls == [
        ("GET /v1/pods", POD_ID),
        ("GET /v1/billing/pods", POD_ID),
    ]
    assert receipt["status"] == "no_start_verified"
    assert receipt["accounted_gpu_usd"] == 0.0
    assert receipt["session_hash"] == current["session_hash"]
    assert receipt["reservation_id"] == current["reservation_id"]
    assert receipt["provider_evidence"]["last_started_at_unchanged"] is True
    assert receipt["billing_evidence"]["row_count"] == 0
    assert load_no_start_receipt(output) == receipt
    persisted = output.read_text()
    assert (
        POD_ID not in persisted
        and NEW_NONCE not in persisted
        and HF_PLACEHOLDER not in persisted
    )
    stopped = json.loads(lifecycle.read_text())
    assert stopped["operation"] == "stopped"
    assert stopped["record_hash"] == receipt["lifecycle_stopped_hash"]

    before = output.read_bytes()
    replay = attest_no_start(
        project_root=tmp_path,
        client=FakeClient({}, []),  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )
    assert replay == receipt
    assert output.read_bytes() == before


@pytest.mark.parametrize("nonce", [OLD_NONCE, NEW_NONCE])
def test_rearm_intent_closes_crash_on_either_side_of_patch(
    tmp_path: Path,
    nonce: str,
) -> None:
    lifecycle, output, _current = _project(tmp_path, operation="rearm_intent")

    receipt = attest_no_start(
        project_root=tmp_path,
        client=FakeClient(_provider_pod(nonce=nonce), []),  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )

    assert receipt["provider_evidence"]["environment_session_context"] == (
        "prior" if nonce == OLD_NONCE else "current"
    )
    assert json.loads(lifecycle.read_text())["operation"] == "stopped"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"desiredStatus": "RUNNING"}, "exactly EXITED"),
        ({"desiredStatus": "TERMINATED"}, "exactly EXITED"),
        ({"lastStartedAt": "2026-08-30T00:02:00Z"}, "lastStartedAt changed"),
        ({"machineId": "different-machine"}, "machine identity drifted"),
        ({"costPerHr": 25.0}, "exceeds the approved"),
        ({"volumeInGb": 649}, "storage, mount, or ports drifted"),
        ({"env": {}}, "current re-arm environment drifted"),
    ],
)
def test_no_start_rejects_status_start_or_identity_drift(
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    lifecycle, output, _current = _project(tmp_path)
    before = lifecycle.read_bytes()

    with pytest.raises(NoStartReconciliationError, match=message):
        attest_no_start(
            project_root=tmp_path,
            client=FakeClient(_provider_pod(**changes), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )

    assert lifecycle.read_bytes() == before
    assert not output.exists()


def test_no_start_rejects_billing_evidence_and_post_start_operation(tmp_path: Path) -> None:
    lifecycle, output, _current = _project(tmp_path)
    client = FakeClient(_provider_pod(), [{"amount": 0.01}])

    with pytest.raises(NoStartReconciliationError, match="billing evidence"):
        attest_no_start(
            project_root=tmp_path,
            client=client,  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )

    state = json.loads(lifecycle.read_text())
    state["operation"] = "rearm_start_requested"
    state.pop("record_hash")
    state["record_hash"] = stable_hash(state)
    lifecycle.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    never_called = FakeClient(_provider_pod(), [])
    with pytest.raises(NoStartReconciliationError, match="not eligible"):
        attest_no_start(
            project_root=tmp_path,
            client=never_called,  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert never_called.calls == []


def test_no_start_get_uncertainty_fails_without_local_mutation(tmp_path: Path) -> None:
    lifecycle, output, _current = _project(tmp_path)
    before = lifecycle.read_bytes()

    class UncertainClient:
        def get_pod(self, _pod_id: str) -> dict[str, Any]:
            raise NoStartReconciliationError("RunPod read-only request outcome is uncertain")

    with pytest.raises(NoStartReconciliationError, match="outcome is uncertain"):
        attest_no_start(
            project_root=tmp_path,
            client=UncertainClient(),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert lifecycle.read_bytes() == before
    assert not output.exists()

    class BillingUncertainClient(FakeClient):
        def get_billing(
            self,
            *,
            pod_id: str,
            start_time: datetime,
            end_time: datetime,
        ) -> list[Any]:
            raise NoStartReconciliationError(
                "RunPod billing GET outcome is uncertain"
            )

    with pytest.raises(NoStartReconciliationError, match="billing GET outcome"):
        attest_no_start(
            project_root=tmp_path,
            client=BillingUncertainClient(_provider_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )


def test_start_intent_requires_quiet_window_second_snapshot(tmp_path: Path) -> None:
    lifecycle, output, _current = _project(tmp_path, operation="rearm_start_intent")

    class AdvancingClient(FakeClient):
        def __init__(self) -> None:
            super().__init__(_provider_pod(), [])
            self.snapshots = [
                _provider_pod(),
                _provider_pod(lastStartedAt="2026-08-30T00:02:00Z"),
            ]

        def get_pod(self, pod_id: str) -> dict[str, Any]:
            self.calls.append(("GET /v1/pods", pod_id))
            return self.snapshots.pop(0)

    slept: list[float] = []
    with pytest.raises(NoStartReconciliationError, match="lastStartedAt changed"):
        attest_no_start(
            project_root=tmp_path,
            client=AdvancingClient(),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
            quiet_window_seconds=30,
            sleep=slept.append,
        )
    assert slept == [30]
    assert json.loads(lifecycle.read_text())["operation"] == "rearm_start_intent"
    assert not output.exists()


def test_start_intent_accepts_two_exact_snapshots_after_quiet_window(
    tmp_path: Path,
) -> None:
    lifecycle, output, _current = _project(tmp_path, operation="rearm_start_intent")
    client = FakeClient(_provider_pod(), [])
    slept: list[float] = []

    receipt = attest_no_start(
        project_root=tmp_path,
        client=client,  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
        quiet_window_seconds=30,
        sleep=slept.append,
    )

    assert slept == [30]
    assert [call[0] for call in client.calls] == [
        "GET /v1/pods",
        "GET /v1/pods",
        "GET /v1/billing/pods",
    ]
    assert receipt["provider_evidence"]["observation_count"] == 2
    assert receipt["provider_evidence"]["quiet_window_seconds"] == 30.0
    assert json.loads(lifecycle.read_text())["operation"] == "stopped"


def test_no_start_rejects_symlinked_session_output(tmp_path: Path) -> None:
    _lifecycle, output, _current = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    output.parent.parent.mkdir(parents=True)
    output.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(NoStartReconciliationError, match="session directory is unsafe"):
        attest_no_start(
            project_root=tmp_path,
            client=FakeClient(_provider_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )


def test_receipt_before_lifecycle_crash_revalidates_provider_before_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, output, _current = _project(tmp_path)
    real_replace = no_start_module._atomic_replace_lifecycle

    def crash_after_receipt(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("crash after durable receipt")

    monkeypatch.setattr(no_start_module, "_atomic_replace_lifecycle", crash_after_receipt)
    with pytest.raises(KeyboardInterrupt, match="durable receipt"):
        attest_no_start(
            project_root=tmp_path,
            client=FakeClient(_provider_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert output.is_file()
    assert json.loads(lifecycle.read_text())["operation"] == "rearm_patched"

    monkeypatch.setattr(no_start_module, "_atomic_replace_lifecycle", real_replace)
    replay_client = FakeClient(_provider_pod(), [])
    receipt = attest_no_start(
        project_root=tmp_path,
        client=replay_client,  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )
    assert replay_client.calls == [
        ("GET /v1/pods", POD_ID),
        ("GET /v1/billing/pods", POD_ID),
    ]
    assert json.loads(lifecycle.read_text())["record_hash"] == receipt[
        "lifecycle_stopped_hash"
    ]


def test_receipt_crash_replay_rejects_a_delayed_provider_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, output, _current = _project(tmp_path)
    real_replace = no_start_module._atomic_replace_lifecycle

    def crash_after_receipt(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("crash after durable receipt")

    monkeypatch.setattr(no_start_module, "_atomic_replace_lifecycle", crash_after_receipt)
    with pytest.raises(KeyboardInterrupt, match="durable receipt"):
        attest_no_start(
            project_root=tmp_path,
            client=FakeClient(_provider_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    before = lifecycle.read_bytes()

    monkeypatch.setattr(no_start_module, "_atomic_replace_lifecycle", real_replace)
    delayed_start = FakeClient(
        _provider_pod(
            desiredStatus="RUNNING",
            lastStartedAt="2026-08-30T00:11:00Z",
        ),
        [{"amount": 1.0}],
    )
    with pytest.raises(NoStartReconciliationError, match="exactly EXITED"):
        attest_no_start(
            project_root=tmp_path,
            client=delayed_start,  # type: ignore[arg-type]
            output_path=output,
            observed_at=datetime(2026, 8, 30, 0, 12, tzinfo=UTC),
        )

    assert delayed_start.calls == [("GET /v1/pods", POD_ID)]
    assert lifecycle.read_bytes() == before
    assert json.loads(lifecycle.read_text())["operation"] == "rearm_patched"


def test_concurrent_lifecycle_drift_leaves_receipt_but_never_overwrites(
    tmp_path: Path,
) -> None:
    lifecycle, output, _current = _project(tmp_path)

    class DriftingClient(FakeClient):
        def get_billing(
            self,
            *,
            pod_id: str,
            start_time: datetime,
            end_time: datetime,
        ) -> list[Any]:
            rows = super().get_billing(
                pod_id=pod_id,
                start_time=start_time,
                end_time=end_time,
            )
            state = json.loads(lifecycle.read_text())
            state["updated_at"] = "2026-08-30T00:09:00Z"
            state.pop("record_hash")
            state["record_hash"] = stable_hash(state)
            lifecycle.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
            lifecycle.chmod(0o600)
            return rows

    with pytest.raises(NoStartReconciliationError, match="changed before"):
        attest_no_start(
            project_root=tmp_path,
            client=DriftingClient(_provider_pod(), []),  # type: ignore[arg-type]
            output_path=output,
            observed_at=OBSERVED,
        )
    assert output.is_file()
    assert json.loads(lifecycle.read_text())["operation"] == "rearm_patched"


def test_duplicate_or_tampered_no_start_receipt_is_rejected(tmp_path: Path) -> None:
    _lifecycle, output, _current = _project(tmp_path)
    receipt = attest_no_start(
        project_root=tmp_path,
        client=FakeClient(_provider_pod(), []),  # type: ignore[arg-type]
        output_path=output,
        observed_at=OBSERVED,
    )
    tampered = deepcopy(receipt)
    tampered["accounted_gpu_usd"] = 0.01
    output.write_text(json.dumps(tampered, sort_keys=True, indent=2) + "\n")
    with pytest.raises(NoStartReconciliationError, match="authentication failed"):
        load_no_start_receipt(output)

    encoded = json.dumps(receipt, sort_keys=True)
    duplicate = encoded[:-1] + ',"status":"no_start_verified"}'
    output.write_text(duplicate)
    with pytest.raises(NoStartReconciliationError, match="duplicate JSON key"):
        load_no_start_receipt(output)
