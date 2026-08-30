from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import model_forensics.runpod_lifecycle as lifecycle_module
from model_forensics.approval import PaidRunApprovalError
from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.io import stable_hash, write_json
from model_forensics.paid_phase_receipt import PaidPhaseReceiptStore
from model_forensics.runpod_lifecycle import (
    EXACT_PROVIDER_GPU_ID,
    RUNPOD_V1_EXACT_QUERY,
    RUNPOD_V1_PODS_URL,
    RUNPOD_V2_PODS_URL,
    HttpResult,
    LifecycleAuthorization,
    RunpodLifecycleClient,
    RunpodLifecycleError,
    build_create_payload,
    create_approved_pod,
    existing_pod_id_hash,
    lifecycle_state_path,
    pod_environment,
    read_lifecycle_status,
    rearm_approved_pod,
    recover_created_pod,
)
from model_forensics.runpod_recovery import attest_external_stop
from model_forensics.runpod_watchdog import (
    HOST_REARM_ACK_FILENAME,
    _host_ack_payload,
    _write_host_rearm_ack,
)

HF_FIXTURE_VALUE = "hf_test_token_must_never_be_persisted"
OLD_NONCE = "old-session-nonce-must-never-be-persisted"
NEW_NONCE = "new-session-nonce-must-never-be-persisted"
API_FIXTURE_VALUE = "runpod-account-key-must-never-be-persisted"
POD_ID = "podabc123"
UNRELATED_POD_ID = "claudeprojectpod123"
MACHINE_ID = "machine-secure-001"
IMAGE = "runpod/pytorch@sha256:" + "ab" * 32


class FakeTransport:
    def __init__(self, responses: list[HttpResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
    ) -> HttpResult:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def _json_result(status: int, value: Any) -> HttpResult:
    return HttpResult(status_code=status, body=json.dumps(value).encode())


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


def _authorization(*, phase: str, nonce: str, suffix: str) -> LifecycleAuthorization:
    spec = _spec()
    return LifecycleAuthorization(
        phase=phase,
        reservation_id=stable_hash({"reservation": suffix}),
        reservation_record_hash=stable_hash({"receipt": suffix}),
        session_hash=stable_hash({"opaque_gpu_session_id": nonce}),
        approval_hash=stable_hash({"approval": suffix}),
        bindings_hash=stable_hash({"bindings": suffix}),
        gpu_lock_hash=stable_hash({"gpu_lock": "fixed"}),
        quote_hash=stable_hash({"quote": "fixed"}),
        immutable_spec=spec,
        immutable_spec_hash=stable_hash(spec),
        approved_runtime_hours=1.0,
        approved_phase_maximum_usd=24.5,
        live_hourly_total_usd=24.5,
    )


def _environment(nonce: str, *, provider_key: bool = True) -> dict[str, str]:
    result = pod_environment(hf_token=HF_FIXTURE_VALUE, session_nonce=nonce)
    if provider_key:
        result["PUBLIC_KEY"] = "ssh-ed25519 AAAATEST"
    return result


def _v2_pod(*, nonce: str, status: str, provider_key: bool = True) -> dict[str, Any]:
    return {
        "id": POD_ID,
        "name": "model-forensics-behavior",
        "status": status,
        "image": IMAGE,
        "disk": 50,
        "ports": ["22/tcp"],
        "env": _environment(nonce, provider_key=provider_key),
        "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": 8},
        "cloud": "SECURE",
        "dataCenterId": "CA-MTL-1",
        "cudaVersion": "12.8",
        "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
        "cost": 24.0,
        "locked": False,
        "globalNetworking": {"enabled": False},
        "ssh": {
            "proxy": {
                "host": "ssh.runpod.io",
                "port": 22,
                "username": "opaque-route",
                "command": "ssh opaque-route@ssh.runpod.io",
            },
            "direct": None,
        },
    }


def _v1_pod(*, nonce: str, status: str) -> dict[str, Any]:
    running = status == "RUNNING"
    return {
        "id": POD_ID,
        "name": "model-forensics-behavior",
        "desiredStatus": status,
        "imageName": IMAGE,
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "networkVolume": None,
        "ports": ["22/tcp"],
        "gpuCount": 8,
        "machine": {
            "gpuTypeId": EXACT_PROVIDER_GPU_ID,
            "secureCloud": True,
            "dataCenterId": "CA-MTL-1",
        },
        "machineId": MACHINE_ID,
        # The live REST v1 response retains the hourly rate while EXITED.
        "costPerHr": 24.0,
        "lastStartedAt": (
            "2026-08-29T12:05:00Z" if running else "2026-08-29T12:00:00Z"
        ),
        "env": _environment(nonce),
        "portMappings": {"22": 32101} if running else {},
        "publicIp": "203.0.113.10" if running else None,
    }


def _v1_pod_url() -> str:
    return f"{RUNPOD_V1_PODS_URL}/{POD_ID}?{RUNPOD_V1_EXACT_QUERY}"


def _create_once(tmp_path: Path) -> tuple[LifecycleAuthorization, Path]:
    authorization = _authorization(
        phase="behavior_baseline_gpu",
        nonce=OLD_NONCE,
        suffix="old",
    )
    transport = FakeTransport(
        [
            _json_result(200, []),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, _v2_pod(nonce=OLD_NONCE, status="RUNNING")),
        ]
    )
    result = create_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
    )
    assert result["provider_status"] == "RUNNING"
    return authorization, lifecycle_state_path(tmp_path)


def _mark_lifecycle_stopped(state_path: Path) -> None:
    """Model the authenticated external-stop transition used before re-arm."""

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["operation"] = "stopped"
    state["pod"]["status"] = "EXITED"
    unsigned = {key: value for key, value in state.items() if key != "record_hash"}
    state["record_hash"] = stable_hash(unsigned)
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _settled_ledger(path: Path, *, prior: LifecycleAuthorization) -> CostLedger:
    document = {
        "schema_version": 1,
        "currency": "USD",
        "hard_stops": {"gpu": 220, "api": 100, "total": 325},
        "entries": [
            {
                "entry_id": prior.reservation_id,
                "kind": "gpu",
                "amount_usd": 12.0,
                "description": "settled old phase",
                "status": "incurred",
                "occurred_at": "2026-08-29T18:00:00+00:00",
            }
        ],
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return CostLedger(path, BudgetLimits(gpu=220, api=100, total=325))


def _host_rearm_ack(
    tmp_path: Path,
    *,
    state_path: Path,
    authorization: LifecycleAuthorization,
    acknowledged_at: datetime | None = None,
) -> Path:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    destination = (
        tmp_path
        / ".runpod"
        / "sessions"
        / authorization.session_hash.removeprefix("sha256:")
        / HOST_REARM_ACK_FILENAME
    )
    observed = acknowledged_at or datetime.now(UTC)
    _write_host_rearm_ack(
        destination,
        _host_ack_payload(
            expected_session_hash=authorization.session_hash,
            expected_phase=authorization.phase,
            lifecycle_before_hash=state["record_hash"],
            pod_id=POD_ID,
            watcher_pid=os.getpid(),
            acknowledged_at=observed,
        ),
    )
    write_json(
        destination.with_name("host_rearm_watchdog.json"),
        {
            "schema_version": 2,
            "watchdog_version": "runpod-gpu-cost-watchdog-v2",
            "pod_id": POD_ID,
            "status": "waiting_for_start",
            "armed_at": observed.isoformat(),
            "updated_at": observed.isoformat(),
            "live_metadata": None,
            "limits": {
                "gpu_hard_stop_usd": 220.0,
                "global_safe_budget_usd": 213.4,
                "safe_budget_usd": 213.4,
                "safety_margin_fraction": 0.03,
                "maximum_runtime_hours": authorization.approved_runtime_hours,
                "maximum_approved_hourly_total_usd": (
                    authorization.live_hourly_total_usd
                ),
                "maximum_approved_compute_hourly_usd": (
                    authorization.live_hourly_total_usd - 0.1
                ),
                "maximum_approved_storage_hourly_usd": 0.1,
                "prior_committed_gpu_usd": 0.0,
            },
            "deadline": None,
            "stop_reason": None,
            "action": "stop_only_preserve_volume",
            "deletion": "manual_after_verified_sync",
            "error": None,
        },
    )
    return destination


def test_lifecycle_authorization_binds_approval_to_clean_source_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_commit = "a" * 40
    ledger = type(
        "LedgerFixture",
        (),
        {"path": tmp_path / "data" / "manifests" / "cost_ledger.yaml"},
    )()
    observed: dict[str, object] = {}

    def clean_source(project_root: Path, *, mutable_paths: tuple[Path, ...]) -> str:
        observed["root"] = Path(project_root)
        observed["mutable_paths"] = mutable_paths
        return expected_commit

    def reject_after_source_binding(*_args: object, **kwargs: object) -> None:
        observed["expected_source_commit"] = kwargs.get("expected_source_commit")
        observed["expected_ledger_path"] = kwargs.get("expected_ledger_path")
        raise PaidRunApprovalError("source-bound validation sentinel")

    monkeypatch.setattr(lifecycle_module, "require_clean_source_commit", clean_source)
    monkeypatch.setattr(
        lifecycle_module,
        "validate_paid_run_approval",
        reject_after_source_binding,
    )

    with pytest.raises(PaidRunApprovalError, match="source-bound validation sentinel"):
        lifecycle_module.authorize_gpu_lifecycle(
            project_root=tmp_path,
            approval=None,  # type: ignore[arg-type]
            expected_bindings=None,  # type: ignore[arg-type]
            reservation=None,  # type: ignore[arg-type]
            ledger=ledger,  # type: ignore[arg-type]
            phase="behavior_treatment_gpu",
            session_nonce="source-binding-session-nonce",
        )

    assert observed == {
        "root": tmp_path,
        "mutable_paths": (ledger.path,),
        "expected_source_commit": expected_commit,
        "expected_ledger_path": "data/manifests/cost_ledger.yaml",
    }


def test_v2_create_payload_is_exact_and_launch_hash_is_secret_safe() -> None:
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")

    payload, launch_hash = build_create_payload(
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
    )

    assert payload == {
        "name": "model-forensics-behavior",
        "image": IMAGE,
        "disk": 50,
        "ports": ["22/tcp"],
        "env": pod_environment(hf_token=HF_FIXTURE_VALUE, session_nonce=OLD_NONCE),
        "cloud": "SECURE",
        "gpu": {
            "id": EXACT_PROVIDER_GPU_ID,
            "count": 8,
            "allowedCudaVersions": ["12.8"],
        },
        "dataCenterIds": ["CA-MTL-1"],
        "globalNetworking": False,
        "mounts": {"persistent": {"size": 650, "path": "/workspace"}},
        "startJupyter": False,
        "startSsh": True,
    }
    assert launch_hash.startswith("sha256:")
    assert HF_FIXTURE_VALUE not in launch_hash
    assert OLD_NONCE not in launch_hash
    assert "OPENROUTER_API_KEY" not in payload["env"]
    assert "RUNPOD_API_KEY" not in payload["env"]


def test_existing_pod_hash_uses_canonical_raw_id_scheme_and_binds_launch_hash() -> None:
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    expected = (
        "runpod-pod-id-sha256:" + hashlib.sha256(UNRELATED_POD_ID.encode("utf-8")).hexdigest()
    )

    assert existing_pod_id_hash(UNRELATED_POD_ID) == expected
    _, without_acknowledgement = build_create_payload(
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
    )
    _, with_acknowledgement = build_create_payload(
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
        acknowledged_existing_pod_id_hashes=(expected,),
    )

    assert with_acknowledgement != without_acknowledgement


def test_create_uses_v1_duplicate_gate_and_official_v2_post_without_secret_persistence(
    tmp_path: Path,
) -> None:
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    transport = FakeTransport(
        [
            _json_result(200, [{"id": "oldterminal", "desiredStatus": "TERMINATED"}]),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, _v2_pod(nonce=OLD_NONCE, status="RUNNING")),
        ]
    )

    summary = create_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
    )

    assert [(call["method"], call["url"]) for call in transport.calls] == [
        ("GET", RUNPOD_V1_PODS_URL),
        ("POST", RUNPOD_V2_PODS_URL),
        ("GET", f"{RUNPOD_V2_PODS_URL}/{POD_ID}"),
    ]
    create_body = json.loads(transport.calls[1]["body"])
    assert create_body["gpu"] == {
        "allowedCudaVersions": ["12.8"],
        "count": 8,
        "id": EXACT_PROVIDER_GPU_ID,
    }
    assert summary["pod_id_hash"] != POD_ID
    state_path = lifecycle_state_path(tmp_path)
    encoded = state_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert HF_FIXTURE_VALUE not in encoded
    assert OLD_NONCE not in encoded
    assert API_FIXTURE_VALUE not in encoded
    assert "PUBLIC_KEY" not in encoded
    assert json.loads(encoded)["operation"] == "created"


def test_create_refuses_any_nonterminal_pod_before_post_or_local_claim(tmp_path: Path) -> None:
    transport = FakeTransport([_json_result(200, [{"id": "somepod", "desiredStatus": "RUNNING"}])])

    with pytest.raises(RunpodLifecycleError, match="nonterminal"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=_authorization(
                phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
            ),
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
        )

    assert [call["method"] for call in transport.calls] == ["GET"]
    assert not lifecycle_state_path(tmp_path).exists()


def test_create_allows_only_exact_acknowledged_nonterminal_pod_set(tmp_path: Path) -> None:
    acknowledged = existing_pod_id_hash(UNRELATED_POD_ID)
    transport = FakeTransport(
        [
            _json_result(
                200,
                [{"id": UNRELATED_POD_ID, "desiredStatus": "RUNNING"}],
            ),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, _v2_pod(nonce=OLD_NONCE, status="RUNNING")),
        ]
    )

    summary = create_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=_authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"),
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
        acknowledged_existing_pod_id_hashes=(acknowledged,),
    )

    assert summary["operation"] == "created"
    state_text = lifecycle_state_path(tmp_path).read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["current_authorization"]["acknowledged_existing_pod_id_hashes"] == [acknowledged]
    assert UNRELATED_POD_ID not in state_text


def test_create_rejects_duplicate_or_extra_existing_pod_hashes(tmp_path: Path) -> None:
    acknowledged = existing_pod_id_hash(UNRELATED_POD_ID)
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    duplicate_transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="duplicate hash"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=duplicate_transport),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
            acknowledged_existing_pod_id_hashes=(acknowledged, acknowledged),
        )
    assert duplicate_transport.calls == []

    extra_transport = FakeTransport([_json_result(200, [])])
    with pytest.raises(RunpodLifecycleError, match="not present"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=extra_transport),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
            acknowledged_existing_pod_id_hashes=(acknowledged,),
        )
    assert [call["method"] for call in extra_transport.calls] == ["GET"]
    assert not lifecycle_state_path(tmp_path).exists()


def test_uncertain_create_claim_prevents_retry_and_keeps_secrets_out(tmp_path: Path) -> None:
    transport = FakeTransport([_json_result(200, []), HttpResult(503, b"secret echo")])
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")

    with pytest.raises(RunpodLifecycleError, match="HTTP 503"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
        )

    encoded = lifecycle_state_path(tmp_path).read_text(encoding="utf-8")
    assert json.loads(encoded)["operation"] == "create_intent"
    assert (
        HF_FIXTURE_VALUE not in encoded
        and OLD_NONCE not in encoded
        and API_FIXTURE_VALUE not in encoded
    )
    with pytest.raises(RunpodLifecycleError, match="second Pod"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=FakeTransport([])),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
        )


def test_create_polls_bounded_pending_state_until_running_and_ssh_ready(tmp_path: Path) -> None:
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    pending = _v2_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["ssh"] = {"proxy": None, "direct": None}
    transport = FakeTransport(
        [
            _json_result(200, []),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, pending),
            _json_result(200, pending),
            _json_result(200, _v2_pod(nonce=OLD_NONCE, status="RUNNING")),
        ]
    )
    clock = [0.0]
    observed_states: list[str] = []

    def sleep(seconds: float) -> None:
        observed_states.append(
            json.loads(lifecycle_state_path(tmp_path).read_text(encoding="utf-8"))["operation"]
        )
        assert 0 < seconds <= 30
        clock[0] += seconds

    create_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=authorization,
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
        maximum_wait_seconds=60,
        poll_interval_seconds=5,
        sleep=sleep,
        monotonic=lambda: clock[0],
    )

    assert observed_states == ["create_pending", "create_pending"]
    assert json.loads(lifecycle_state_path(tmp_path).read_text())["operation"] == "created"


def test_create_timeout_is_fail_closed_and_instructs_status_not_retry(tmp_path: Path) -> None:
    pending = _v2_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["ssh"] = {"proxy": None, "direct": None}
    transport = FakeTransport(
        [
            _json_result(200, []),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, pending),
        ]
    )

    with pytest.raises(RunpodLifecycleError, match="read-only status"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=_authorization(
                phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
            ),
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
            maximum_wait_seconds=0,
            sleep=lambda _seconds: pytest.fail("timeout must not sleep"),
            monotonic=lambda: 0.0,
        )

    state_path = lifecycle_state_path(tmp_path)
    encoded = state_path.read_text(encoding="utf-8")
    assert json.loads(encoded)["operation"] == "create_timeout"
    assert HF_FIXTURE_VALUE not in encoded and OLD_NONCE not in encoded


def test_recover_create_uses_only_v2_get_and_promotes_verified_state(tmp_path: Path) -> None:
    authorization = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    pending = _v2_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["ssh"] = {"proxy": None, "direct": None}
    with pytest.raises(RunpodLifecycleError, match="read-only status"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE,
                transport=FakeTransport(
                    [
                        _json_result(200, []),
                        _json_result(
                            201,
                            _v2_pod(nonce=OLD_NONCE, status="PROVISIONING"),
                        ),
                        _json_result(200, pending),
                    ]
                ),
            ),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
            maximum_wait_seconds=0,
            sleep=lambda _seconds: pytest.fail("timeout must not sleep"),
            monotonic=lambda: 0.0,
        )

    state_path = lifecycle_state_path(tmp_path)
    transport = FakeTransport([_json_result(200, _v2_pod(nonce=OLD_NONCE, status="RUNNING"))])
    summary = recover_created_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=authorization,
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
    )

    assert [(call["method"], call["url"]) for call in transport.calls] == [
        ("GET", f"{RUNPOD_V2_PODS_URL}/{POD_ID}")
    ]
    assert summary["operation"] == "created"
    encoded = state_path.read_text(encoding="utf-8")
    recovered = json.loads(encoded)
    assert recovered["operation"] == "created"
    assert recovered["pod"]["provider_binding_hash"].startswith("sha256:")
    assert HF_FIXTURE_VALUE not in encoded
    assert OLD_NONCE not in encoded
    assert API_FIXTURE_VALUE not in encoded


def test_recover_create_rejects_authorization_drift_before_provider_get(
    tmp_path: Path,
) -> None:
    old = _authorization(phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old")
    pending = _v2_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["ssh"] = {"proxy": None, "direct": None}
    with pytest.raises(RunpodLifecycleError, match="read-only status"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE,
                transport=FakeTransport(
                    [
                        _json_result(200, []),
                        _json_result(
                            201,
                            _v2_pod(nonce=OLD_NONCE, status="PROVISIONING"),
                        ),
                        _json_result(200, pending),
                    ]
                ),
            ),
            authorization=old,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
            maximum_wait_seconds=0,
            sleep=lambda _seconds: pytest.fail("timeout must not sleep"),
            monotonic=lambda: 0.0,
        )
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="authorization"):
        recover_created_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=_authorization(
                phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="different"
            ),
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
        )

    assert transport.calls == []
    assert json.loads(lifecycle_state_path(tmp_path).read_text())["operation"] == ("create_timeout")


def test_rearm_verifies_same_stopped_pod_patches_only_nonce_and_starts(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    host_ack = _host_rearm_ack(tmp_path, state_path=state_path, authorization=new)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport(
        [
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="EXITED")),
            _json_result(
                200,
                {
                    **_v1_pod(nonce=NEW_NONCE, status="RUNNING"),
                    "portMappings": {},
                    "publicIp": None,
                },
            ),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="RUNNING")),
        ]
    )
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    summary = rearm_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=new,
        ledger=ledger,
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=NEW_NONCE,
        host_watchdog_ack_path=host_ack,
        maximum_wait_seconds=60,
        poll_interval_seconds=5,
        sleep=sleep,
        monotonic=lambda: clock[0],
    )

    assert [(call["method"], call["url"]) for call in transport.calls] == [
        ("GET", _v1_pod_url()),
        ("PATCH", f"{RUNPOD_V1_PODS_URL}/{POD_ID}"),
        ("GET", _v1_pod_url()),
        ("POST", f"{RUNPOD_V1_PODS_URL}/{POD_ID}/start"),
        ("GET", _v1_pod_url()),
        ("GET", _v1_pod_url()),
        ("GET", _v1_pod_url()),
    ]
    patch_body = json.loads(transport.calls[1]["body"])
    assert set(patch_body) == {"env"}
    old_env = _environment(OLD_NONCE)
    new_env = patch_body["env"]
    assert {key for key in old_env if old_env[key] != new_env[key]} == {"GPU_BUDGET_SESSION_ID"}
    assert transport.calls[3]["body"] is None
    assert all(call["method"] != "DELETE" for call in transport.calls)
    assert all(RUNPOD_V2_PODS_URL not in call["url"] for call in transport.calls)
    assert sleeps == [5, 5]
    encoded = state_path.read_text(encoding="utf-8")
    assert json.loads(encoded)["operation"] == "rearmed"
    assert HF_FIXTURE_VALUE not in encoded and OLD_NONCE not in encoded and NEW_NONCE not in encoded
    assert summary["provider_status"] == "RUNNING"


def test_rearm_requires_authenticated_local_stopped_transition(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="authenticated stopped receipt"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "missing-host-ack.json",
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "created"


def test_rearm_rejects_reusing_one_approval_for_the_same_paid_phase(
    tmp_path: Path,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    fresh = _authorization(
        phase="behavior_baseline_gpu",
        nonce=NEW_NONCE,
        suffix="fresh-reservation",
    )
    replay = replace(fresh, approval_hash=old.approval_hash)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="already used for this paid command phase"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=replay,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "unused-host-ack.json",
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


@pytest.mark.parametrize("malformation", ["hash", "phase"])
def test_rearm_rejects_malformed_historical_authorization_before_provider_call(
    tmp_path: Path,
    malformation: str,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    historical = dict(state["current_authorization"])
    historical["session_hash"] = stable_hash({"session": "historical"})
    historical["reservation_id"] = stable_hash({"reservation": "historical"})
    historical["phase"] = "behavior_treatment_gpu"
    if malformation == "hash":
        historical["approval_hash"] = "sha256:not-a-valid-digest"
        error_pattern = r"historical\[0\].*hash"
    else:
        historical["phase"] = "arbitrary_paid_phase"
        error_pattern = r"historical\[0\].*phase"
    state["authorization_history"] = [historical]
    state["record_hash"] = stable_hash(
        {key: value for key, value in state.items() if key != "record_hash"}
    )
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    candidate = _authorization(
        phase="behavior_treatment_gpu",
        nonce=NEW_NONCE,
        suffix="fresh-candidate",
    )
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match=error_pattern):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=candidate,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "unused-host-ack.json",
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


def test_rearm_rejects_approval_phase_replay_from_older_history(
    tmp_path: Path,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    candidate = _authorization(
        phase="behavior_treatment_gpu",
        nonce=NEW_NONCE,
        suffix="fresh-candidate",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    historical = dict(state["current_authorization"])
    historical["phase"] = candidate.phase
    historical["approval_hash"] = candidate.approval_hash
    historical["session_hash"] = stable_hash({"session": "older-history"})
    historical["reservation_id"] = stable_hash({"reservation": "older-history"})
    historical["reservation_record_hash"] = stable_hash({"receipt": "older-history"})
    state["authorization_history"] = [historical]
    state["record_hash"] = stable_hash(
        {key: value for key, value in state.items() if key != "record_hash"}
    )
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="already used for this paid command phase"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=candidate,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "unused-host-ack.json",
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


def test_rearm_rejects_post_receipt_same_phase_retry_before_provider_call(
    tmp_path: Path,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    candidate = _authorization(
        phase="behavior_treatment_gpu",
        nonce=NEW_NONCE,
        suffix="fresh-candidate",
    )
    PaidPhaseReceiptStore(tmp_path / ".runpod/paid_phase_receipts").authorize(
        command_phase=candidate.phase,
        approval_content_hash=candidate.approval_hash,
        approval_id_hash=stable_hash({"approval_id": "prior-attempt"}),
        bindings_hash=candidate.bindings_hash,
        plan_hash=stable_hash({"plan": "prior-attempt"}),
    )
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match=r"unsupported after.*paid-plan receipt"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=candidate,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "unused-host-ack.json",
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


@pytest.mark.parametrize("authorization_case", ["same_approval_new_phase", "new_approval_same_phase"])
def test_rearm_replay_gate_allows_only_a_new_approval_phase_pair(
    tmp_path: Path,
    authorization_case: str,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    if authorization_case == "same_approval_new_phase":
        candidate = replace(
            _authorization(
                phase="behavior_treatment_gpu",
                nonce=NEW_NONCE,
                suffix="fresh-phase",
            ),
            approval_hash=old.approval_hash,
        )
    else:
        candidate = _authorization(
            phase="behavior_baseline_gpu",
            nonce=NEW_NONCE,
            suffix="fresh-approval",
        )
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    missing_ack = (
        tmp_path
        / ".runpod"
        / "sessions"
        / candidate.session_hash.removeprefix("sha256:")
        / HOST_REARM_ACK_FILENAME
    )
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="host re-arm watchdog"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=candidate,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=missing_ack,
        )

    assert transport.calls == []


def test_rearm_requires_live_host_watchdog_ack_before_provider_calls(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    missing_ack = (
        tmp_path
        / ".runpod"
        / "sessions"
        / new.session_hash.removeprefix("sha256:")
        / HOST_REARM_ACK_FILENAME
    )
    transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="host re-arm watchdog"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=missing_ack,
        )

    assert transport.calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


def test_rearm_fails_before_patch_if_machine_or_storage_drifted(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    host_ack = _host_rearm_ack(tmp_path, state_path=state_path, authorization=new)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    drifted = _v1_pod(nonce=OLD_NONCE, status="EXITED")
    drifted["volumeInGb"] = 649
    transport = FakeTransport([_json_result(200, drifted)])

    with pytest.raises(RunpodLifecycleError, match="persistent disk"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=host_ack,
        )

    assert [call["method"] for call in transport.calls] == ["GET"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


def test_rearm_persists_pre_start_baseline_before_patch_crash(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    host_ack = _host_rearm_ack(tmp_path, state_path=state_path, authorization=new)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport(
        [_json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED"))]
    )

    def crash_on_patch(**kwargs: Any) -> HttpResult:
        if kwargs["method"] == "PATCH":
            transport.calls.append(dict(kwargs))
            raise KeyboardInterrupt("crash at PATCH boundary")
        return transport(**kwargs)

    with pytest.raises(KeyboardInterrupt, match="PATCH boundary"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE,
                transport=crash_on_patch,
            ),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=host_ack,
        )

    intent = json.loads(state_path.read_text(encoding="utf-8"))
    assert [call["method"] for call in transport.calls] == ["GET", "PATCH"]
    assert intent["operation"] == "rearm_intent"
    assert intent["pod"]["pre_start_last_started_at"] == "2026-08-29T12:00:00Z"


def test_rearm_rejects_machine_drift_after_patch_before_start(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    host_ack = _host_rearm_ack(tmp_path, state_path=state_path, authorization=new)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    moved = _v1_pod(nonce=NEW_NONCE, status="EXITED")
    moved["machineId"] = "machine-secure-002"
    transport = FakeTransport(
        [
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, moved),
        ]
    )

    with pytest.raises(RunpodLifecycleError, match="different machine"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=host_ack,
        )

    assert [call["method"] for call in transport.calls] == ["GET", "PATCH", "GET"]
    assert all(RUNPOD_V2_PODS_URL not in call["url"] for call in transport.calls)
    encoded = state_path.read_text(encoding="utf-8")
    assert json.loads(encoded)["operation"] == "rearm_intent"
    assert MACHINE_ID not in encoded and "machine-secure-002" not in encoded


def test_rearm_running_poll_is_bounded_and_persists_timeout(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    host_ack = _host_rearm_ack(tmp_path, state_path=state_path, authorization=new)
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport(
        [
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="EXITED")),
        ]
    )

    with pytest.raises(RunpodLifecycleError, match="timed out"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=host_ack,
            maximum_wait_seconds=0,
            sleep=lambda _seconds: pytest.fail("zero timeout must not sleep"),
            monotonic=lambda: 0.0,
        )

    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == ("rearm_timeout")
    assert all(RUNPOD_V2_PODS_URL not in call["url"] for call in transport.calls)


def test_crash_after_accepted_start_leaves_recoverable_authenticated_intent(
    tmp_path: Path,
) -> None:
    old, state_path = _create_once(tmp_path)
    _mark_lifecycle_stopped(state_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    transition_at = datetime(2026, 8, 29, 12, 5, tzinfo=UTC)
    host_ack = _host_rearm_ack(
        tmp_path,
        state_path=state_path,
        authorization=new,
        acknowledged_at=transition_at,
    )
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    before_start = FakeTransport(
        [
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED")),
            HttpResult(200, b""),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="EXITED")),
        ]
    )

    def crash_transport(**kwargs: Any) -> HttpResult:
        if kwargs["method"] == "POST" and str(kwargs["url"]).endswith("/start"):
            before_start.calls.append(dict(kwargs))
            raise KeyboardInterrupt("provider accepted start before client crashed")
        return before_start(**kwargs)

    with pytest.raises(KeyboardInterrupt, match="provider accepted start"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE,
                transport=crash_transport,
            ),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=host_ack,
            now=transition_at,
        )

    intent = json.loads(state_path.read_text(encoding="utf-8"))
    assert intent["operation"] == "rearm_start_intent"
    assert intent["pod"]["pre_start_last_started_at"] == "2026-08-29T12:00:00Z"

    live_after_host_stop = {
        **_v1_pod(nonce=NEW_NONCE, status="EXITED"),
        "createdAt": "2026-08-29T10:00:00Z",
        "lastStartedAt": "2026-08-29T12:05:01Z",
        "lastStatusChange": "2026-08-29T12:10:00Z",
    }
    billing = [
        {
            "amount": 2.0,
            "diskSpaceBilledGb": 50,
            "gpuTypeId": EXACT_PROVIDER_GPU_ID,
            "podId": POD_ID,
            "time": "2026-08-29T12:05:00Z",
            "timeBilledMs": 299_000,
        }
    ]

    class RecoveryClient:
        def get_pod(self, pod_id: str) -> dict[str, Any]:
            assert pod_id == POD_ID
            return live_after_host_stop

        def get_billing(
            self,
            *,
            pod_id: str,
            start_time: datetime,
            end_time: datetime,
        ) -> list[Any]:
            assert pod_id == POD_ID and start_time < end_time
            return billing

    output = (
        tmp_path
        / ".runpod"
        / "sessions"
        / new.session_hash.removeprefix("sha256:")
        / "external_stop_receipt.json"
    )
    receipt = attest_external_stop(
        project_root=tmp_path,
        client=RecoveryClient(),  # type: ignore[arg-type]
        output_path=output,
        observed_at=datetime(2026, 8, 29, 12, 15, tzinfo=UTC),
    )
    assert receipt["prior_lifecycle_operation"] == "rearm_start_intent"
    assert receipt["stop_evidence"]["start_context"] == "rearm"
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "stopped"


@pytest.mark.parametrize(
    ("maximum_wait_seconds", "poll_interval_seconds"),
    ((-1, 5), (601, 5), (60, 0), (60, 31)),
)
def test_rearm_rejects_unbounded_poll_configuration_before_provider_calls(
    tmp_path: Path,
    maximum_wait_seconds: float,
    poll_interval_seconds: float,
) -> None:
    old, _ = _create_once(tmp_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport([])

    with pytest.raises(ValueError, match="readiness wait"):
        rearm_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
            authorization=new,
            ledger=ledger,
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=NEW_NONCE,
            host_watchdog_ack_path=tmp_path / "missing-host-ack.json",
            maximum_wait_seconds=maximum_wait_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    assert transport.calls == []


def test_status_is_read_only_and_redacts_private_identifiers(tmp_path: Path) -> None:
    _, state_path = _create_once(tmp_path)
    before = state_path.read_bytes()
    transport = FakeTransport([_json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING"))])

    status = read_lifecycle_status(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
    )

    assert state_path.read_bytes() == before
    assert status["provider_status"] == "RUNNING"
    assert status["pod_id_hash"] != POD_ID
    assert POD_ID not in json.dumps(status)
    assert [(call["method"], call["url"]) for call in transport.calls] == [("GET", _v1_pod_url())]
    assert RUNPOD_V2_PODS_URL not in transport.calls[0]["url"]


def test_symlinked_private_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / ".runpod").symlink_to(target, target_is_directory=True)

    with pytest.raises(RunpodLifecycleError, match="symlink"):
        lifecycle_state_path(tmp_path)
