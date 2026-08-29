from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from model_forensics.budget import BudgetLimits, CostLedger
from model_forensics.io import stable_hash
from model_forensics.runpod_lifecycle import (
    EXACT_PROVIDER_GPU_ID,
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
        "image": IMAGE,
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "networkVolume": None,
        "ports": ["22/tcp"],
        "globalNetworking": False,
        "interruptible": False,
        "locked": False,
        "gpu": {"id": EXACT_PROVIDER_GPU_ID, "count": 8},
        "machine": {"secureCloud": True, "dataCenterId": "CA-MTL-1"},
        "machineId": MACHINE_ID,
        "costPerHr": 24.0 if running else 0.0,
        "env": _environment(nonce),
        "portMappings": {"22": 32101} if running else {},
        "publicIp": "203.0.113.10" if running else None,
    }


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
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING")),
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


def test_v2_create_payload_is_exact_and_launch_hash_is_secret_safe() -> None:
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )

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
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )
    expected = "runpod-pod-id-sha256:" + hashlib.sha256(
        UNRELATED_POD_ID.encode("utf-8")
    ).hexdigest()

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
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )
    transport = FakeTransport(
        [
            _json_result(200, [{"id": "oldterminal", "desiredStatus": "TERMINATED"}]),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING")),
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
        ("GET", f"{RUNPOD_V1_PODS_URL}/{POD_ID}"),
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
    transport = FakeTransport(
        [_json_result(200, [{"id": "somepod", "desiredStatus": "RUNNING"}])]
    )

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
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING")),
        ]
    )

    summary = create_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=_authorization(
            phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
        ),
        name="model-forensics-behavior",
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=OLD_NONCE,
        acknowledged_existing_pod_id_hashes=(acknowledged,),
    )

    assert summary["operation"] == "created"
    state_text = lifecycle_state_path(tmp_path).read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["current_authorization"][
        "acknowledged_existing_pod_id_hashes"
    ] == [acknowledged]
    assert UNRELATED_POD_ID not in state_text


def test_create_rejects_duplicate_or_extra_existing_pod_hashes(tmp_path: Path) -> None:
    acknowledged = existing_pod_id_hash(UNRELATED_POD_ID)
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )
    duplicate_transport = FakeTransport([])

    with pytest.raises(RunpodLifecycleError, match="duplicate hash"):
        create_approved_pod(
            project_root=tmp_path,
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE, transport=duplicate_transport
            ),
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
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE, transport=extra_transport
            ),
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
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )

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
            client=RunpodLifecycleClient(
                api_key=API_FIXTURE_VALUE, transport=FakeTransport([])
            ),
            authorization=authorization,
            name="model-forensics-behavior",
            hf_token=HF_FIXTURE_VALUE,
            session_nonce=OLD_NONCE,
        )


def test_create_polls_bounded_pending_state_until_running_and_ssh_ready(tmp_path: Path) -> None:
    authorization = _authorization(
        phase="behavior_baseline_gpu", nonce=OLD_NONCE, suffix="old"
    )
    pending = _v1_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["portMappings"] = {}
    pending["publicIp"] = None
    transport = FakeTransport(
        [
            _json_result(200, []),
            _json_result(201, _v2_pod(nonce=OLD_NONCE, status="PROVISIONING")),
            _json_result(200, pending),
            _json_result(200, pending),
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING")),
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
    pending = _v1_pod(nonce=OLD_NONCE, status="RUNNING")
    pending["portMappings"] = {}
    pending["publicIp"] = None
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


def test_rearm_verifies_same_stopped_pod_patches_only_nonce_and_starts(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
    ledger = _settled_ledger(tmp_path / "cost_ledger.yaml", prior=old)
    transport = FakeTransport(
        [
            _json_result(200, _v1_pod(nonce=OLD_NONCE, status="EXITED")),
            _json_result(200, _v2_pod(nonce=NEW_NONCE, status="EXITED")),
            _json_result(200, _v2_pod(nonce=NEW_NONCE, status="STARTING")),
            _json_result(200, _v1_pod(nonce=NEW_NONCE, status="RUNNING")),
        ]
    )

    summary = rearm_approved_pod(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
        authorization=new,
        ledger=ledger,
        hf_token=HF_FIXTURE_VALUE,
        session_nonce=NEW_NONCE,
    )

    assert [(call["method"], call["url"]) for call in transport.calls] == [
        ("GET", f"{RUNPOD_V1_PODS_URL}/{POD_ID}"),
        ("PATCH", f"{RUNPOD_V2_PODS_URL}/{POD_ID}"),
        ("POST", f"{RUNPOD_V2_PODS_URL}/{POD_ID}/action"),
        ("GET", f"{RUNPOD_V1_PODS_URL}/{POD_ID}"),
    ]
    patch_body = json.loads(transport.calls[1]["body"])
    assert set(patch_body) == {"env"}
    old_env = _environment(OLD_NONCE)
    new_env = patch_body["env"]
    assert {key for key in old_env if old_env[key] != new_env[key]} == {
        "GPU_BUDGET_SESSION_ID"
    }
    assert json.loads(transport.calls[2]["body"]) == {"action": "start"}
    assert all(call["method"] != "DELETE" for call in transport.calls)
    encoded = state_path.read_text(encoding="utf-8")
    assert json.loads(encoded)["operation"] == "rearmed"
    assert (
        HF_FIXTURE_VALUE not in encoded
        and OLD_NONCE not in encoded
        and NEW_NONCE not in encoded
    )
    assert summary["provider_status"] == "RUNNING"


def test_rearm_fails_before_patch_if_machine_or_storage_drifted(tmp_path: Path) -> None:
    old, state_path = _create_once(tmp_path)
    new = _authorization(phase="behavior_treatment_gpu", nonce=NEW_NONCE, suffix="new")
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
        )

    assert [call["method"] for call in transport.calls] == ["GET"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["operation"] == "created"


def test_status_is_read_only_and_redacts_private_identifiers(tmp_path: Path) -> None:
    _, state_path = _create_once(tmp_path)
    before = state_path.read_bytes()
    transport = FakeTransport(
        [_json_result(200, _v1_pod(nonce=OLD_NONCE, status="RUNNING"))]
    )

    status = read_lifecycle_status(
        project_root=tmp_path,
        client=RunpodLifecycleClient(api_key=API_FIXTURE_VALUE, transport=transport),
    )

    assert state_path.read_bytes() == before
    assert status["provider_status"] == "RUNNING"
    assert status["pod_id_hash"] != POD_ID
    assert POD_ID not in json.dumps(status)
    assert [(call["method"], call["url"]) for call in transport.calls] == [
        ("GET", f"{RUNPOD_V1_PODS_URL}/{POD_ID}")
    ]


def test_symlinked_private_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / ".runpod").symlink_to(target, target_is_directory=True)

    with pytest.raises(RunpodLifecycleError, match="symlink"):
        lifecycle_state_path(tmp_path)
