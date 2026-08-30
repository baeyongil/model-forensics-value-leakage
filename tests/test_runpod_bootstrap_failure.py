from __future__ import annotations

import json
from pathlib import Path

import pytest

import model_forensics.runpod_bootstrap_failure as failure_module
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_bootstrap_failure import (
    EXACT_RESEARCH_CONTAINER_IMAGE,
    BootstrapFailureStopError,
    stop_after_bootstrap_failure,
)
from model_forensics.runpod_contract import LIFECYCLE_PROTOCOL


def _artifacts(root: Path, *, nonce: str) -> tuple[str, Path]:
    phase = "behavior_treatment_gpu"
    session_hash = stable_hash({"opaque_gpu_session_id": nonce})
    reservation_id = stable_hash({"reservation": session_hash})
    receipt = {
        "schema_version": 1,
        "protocol_version": "cumulative-gpu-phase-budget-v1",
        "phase": phase,
        "session_hash": session_hash,
        "reservation_id": reservation_id,
    }
    receipt["record_hash"] = stable_hash(receipt)
    receipt_path = root / ".runpod" / "reservations" / f"{phase}.json"
    write_json(receipt_path, receipt)
    spec = {"image": EXACT_RESEARCH_CONTAINER_IMAGE, "gpu": {"count": 8}}
    authorization = {
        "phase": phase,
        "reservation_id": reservation_id,
        "reservation_record_hash": receipt["record_hash"],
        "session_hash": session_hash,
        "approval_hash": stable_hash({"approval": "fixture"}),
        "bindings_hash": stable_hash({"bindings": "fixture"}),
        "gpu_lock_hash": stable_hash({"gpu_lock": "fixture"}),
        "quote_hash": stable_hash({"quote": "fixture"}),
        "immutable_spec_hash": stable_hash(spec),
        "launch_spec_hash": stable_hash({"launch": "fixture"}),
        "acknowledged_existing_pod_id_hashes": [],
        "approved_runtime_hours": 1.0,
        "approved_phase_maximum_usd": 10.0,
        "live_hourly_total_usd": 10.0,
    }
    lifecycle = {
        "schema_version": 1,
        "protocol_version": LIFECYCLE_PROTOCOL,
        "operation": "rearmed",
        "updated_at": "2026-08-30T00:00:00Z",
        "immutable_spec": spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": "pod-fixture", "status": "RUNNING"},
    }
    lifecycle["record_hash"] = stable_hash(lifecycle)
    write_json(root / ".runpod" / "pod_lifecycle.json", lifecycle)
    return phase, receipt_path


def _provider_payload(
    *,
    nonce: str,
    pod_id: str = "pod-fixture",
) -> dict[str, object]:
    return {
        "id": pod_id,
        "name": "model-forensics-behavior-baseline",
        "desiredStatus": "RUNNING",
        "imageName": EXACT_RESEARCH_CONTAINER_IMAGE,
        "gpuCount": 8,
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "networkVolume": None,
        "networkVolumeId": None,
        "ports": ["22/tcp"],
        "globalNetworking": False,
        "interruptible": False,
        "locked": False,
        "machineId": "machine-fixture",
        "machine": {
            "gpuTypeId": "NVIDIA H100 80GB HBM3",
            "dataCenterId": "CA-MTL-1",
            "secureCloud": True,
        },
        "publicIp": "192.0.2.10",
        "portMappings": {"22": 32101},
        "env": {
            "HF_TOKEN": "hf-fixture",
            "GPU_BUDGET_SESSION_ID": nonce,
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
            "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
            "PUBLIC_KEY": "ssh-ed25519 AAAATEST",
        },
    }


def _stop_kwargs(
    *,
    root: Path,
    phase: str,
    receipt: Path,
    nonce: str,
) -> dict[str, object]:
    return {
        "project_root": root,
        "phase": phase,
        "reservation_path": receipt,
        "pod_id": "pod-fixture",
        "api_key": "API_KEY",
        "session_nonce": nonce,
        "expected_provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "allowed_data_center_ids": ("CA-MTL-1", "EUR-IS-3"),
        "expected_container_image": EXACT_RESEARCH_CONTAINER_IMAGE,
    }


def test_failed_or_expired_prebootstrap_verification_stops_promptly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonce = "bootstrap-failure-stop-session"
    phase, receipt = _artifacts(tmp_path, nonce=nonce)
    monkeypatch.setattr(failure_module, "BOOTSTRAP_PROJECT_ROOT", tmp_path.resolve())
    calls: list[tuple[str, str]] = []
    statuses = iter(("RUNNING", "EXITED"))

    def transport(
        method: str,
        url: str,
        _api_key: str,
        _timeout: float,
    ) -> tuple[int, bytes]:
        calls.append((method, url.rsplit("/", 1)[-1]))
        if method == "POST":
            return 200, b"{}"
        payload = _provider_payload(nonce=nonce)
        payload["desiredStatus"] = next(statuses)
        if payload["desiredStatus"] == "EXITED":
            payload["publicIp"] = None
            payload["portMappings"] = {}
        return 200, json.dumps(payload).encode()

    summary = stop_after_bootstrap_failure(
        **_stop_kwargs(root=tmp_path, phase=phase, receipt=receipt, nonce=nonce),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    assert [method for method, _target in calls] == ["GET", "POST", "GET"]
    assert summary["status"] == "stop_confirmed"
    assert summary["authentication"] == "local_lifecycle_reservation"
    assert "pod-fixture" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        (lambda payload: payload.update(name="claude-project"), "research Pod"),
        (
            lambda payload: payload.update(
                imageName="runpod/pytorch@sha256:" + "b" * 64
            ),
            "image",
        ),
        (
            lambda payload: payload["env"].update(  # type: ignore[union-attr]
                GPU_BUDGET_SESSION_ID="other-session"
            ),
            "session",
        ),
        (
            lambda payload: payload["machine"].update(  # type: ignore[union-attr]
                gpuTypeId="NVIDIA A100 80GB PCIe"
            ),
            "GPU",
        ),
        (
            lambda payload: payload["machine"].update(  # type: ignore[union-attr]
                secureCloud=False
            ),
            "GPU",
        ),
        (
            lambda payload: payload["machine"].update(  # type: ignore[union-attr]
                dataCenterId="US-TX-1"
            ),
            "GPU",
        ),
        (lambda payload: payload.update(volumeInGb=700), "storage"),
        (lambda payload: payload.update(networkVolumeId="volume-other"), "storage"),
        (lambda payload: payload.update(publicIp="not-an-ip"), "endpoint"),
    ),
)
def test_valid_local_binding_still_requires_full_provider_evidence_before_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    expected_error: str,
) -> None:
    nonce = "valid-local-provider-gate"
    phase, receipt = _artifacts(tmp_path, nonce=nonce)
    monkeypatch.setattr(failure_module, "BOOTSTRAP_PROJECT_ROOT", tmp_path.resolve())
    calls: list[str] = []

    def transport(
        method: str,
        _url: str,
        _api_key: str,
        _timeout: float,
    ) -> tuple[int, bytes]:
        calls.append(method)
        payload = _provider_payload(nonce=nonce)
        mutation(payload)  # type: ignore[operator]
        return 200, json.dumps(payload).encode()

    with pytest.raises(BootstrapFailureStopError, match=expected_error):
        stop_after_bootstrap_failure(
            **_stop_kwargs(root=tmp_path, phase=phase, receipt=receipt, nonce=nonce),
            transport=transport,
        )
    assert calls == ["GET"]


def test_emergency_stop_rejects_wrong_session_before_provider_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, receipt = _artifacts(tmp_path, nonce="correct-session-nonce")
    monkeypatch.setattr(failure_module, "BOOTSTRAP_PROJECT_ROOT", tmp_path.resolve())
    calls: list[str] = []

    def transport(
        _method: str,
        _url: str,
        _api_key: str,
        _timeout: float,
    ) -> tuple[int, bytes]:
        calls.append(_method)
        return 200, json.dumps(
            _provider_payload(nonce="correct-session-nonce")
        ).encode()

    with pytest.raises(BootstrapFailureStopError, match="session identity"):
        stop_after_bootstrap_failure(
            **_stop_kwargs(
                root=tmp_path,
                phase=phase,
                receipt=receipt,
                nonce="wrong-session-nonce",
            ),
            transport=transport,
        )
    assert calls == ["GET"]


@pytest.mark.parametrize(
    "local_failure",
    ("missing_lifecycle", "missing_reservation", "corrupt_lifecycle", "corrupt_reservation"),
)
def test_provider_bound_fallback_stops_when_local_control_files_are_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_failure: str,
) -> None:
    nonce = "independent-provider-session"
    phase, receipt = _artifacts(tmp_path, nonce=nonce)
    lifecycle = tmp_path / ".runpod" / "pod_lifecycle.json"
    if local_failure == "missing_lifecycle":
        lifecycle.unlink()
    elif local_failure == "missing_reservation":
        receipt.unlink()
    elif local_failure == "corrupt_lifecycle":
        lifecycle.write_text("{corrupt", encoding="utf-8")
    else:
        receipt.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr(failure_module, "BOOTSTRAP_PROJECT_ROOT", tmp_path.resolve())
    calls: list[tuple[str, str]] = []
    statuses = iter(("RUNNING", "EXITED"))

    def transport(
        method: str,
        _url: str,
        _api_key: str,
        _timeout: float,
    ) -> tuple[int, bytes]:
        calls.append((method, _url))
        if method == "POST":
            return 200, b"{}"
        payload = _provider_payload(nonce=nonce)
        payload["desiredStatus"] = next(statuses)
        if payload["desiredStatus"] == "EXITED":
            payload["publicIp"] = None
            payload["portMappings"] = {}
        return 200, json.dumps(payload).encode()

    summary = stop_after_bootstrap_failure(
        **_stop_kwargs(root=tmp_path, phase=phase, receipt=receipt, nonce=nonce),
        transport=transport,
        sleep=lambda _seconds: None,
    )

    assert [method for method, _url in calls] == ["GET", "POST", "GET"]
    assert all(
        url.endswith("?includeMachine=true&includeNetworkVolume=true&includeTemplate=true")
        for method, url in calls
        if method == "GET"
    )
    assert summary["status"] == "stop_confirmed"
    assert summary["authentication"] == "independent_provider_evidence"
    assert "pod-fixture" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        (lambda payload: payload.update(id="other-pod"), "identity"),
        (
            lambda payload: payload["env"].update(  # type: ignore[union-attr]
                GPU_BUDGET_SESSION_ID="other-session"
            ),
            "session",
        ),
        (
            lambda payload: payload["machine"].update(  # type: ignore[union-attr]
                gpuTypeId="NVIDIA A100 80GB PCIe"
            ),
            "GPU",
        ),
        (
            lambda payload: payload.update(
                imageName="runpod/pytorch@sha256:" + "b" * 64
            ),
            "image",
        ),
        (lambda payload: payload.update(name="claude-project"), "research Pod"),
        (lambda payload: payload.update(volumeInGb=700), "storage"),
        (lambda payload: payload.update(publicIp="not-an-ip"), "endpoint"),
    ),
)
def test_provider_fallback_mismatch_never_posts_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    expected_error: str,
) -> None:
    nonce = "provider-mismatch-session"
    phase = "behavior_treatment_gpu"
    receipt = tmp_path / ".runpod" / "reservations" / f"{phase}.json"
    monkeypatch.setattr(failure_module, "BOOTSTRAP_PROJECT_ROOT", tmp_path.resolve())
    calls: list[str] = []

    def transport(
        method: str,
        _url: str,
        _api_key: str,
        _timeout: float,
    ) -> tuple[int, bytes]:
        calls.append(method)
        payload = _provider_payload(nonce=nonce)
        mutation(payload)  # type: ignore[operator]
        return 200, json.dumps(payload).encode()

    with pytest.raises(BootstrapFailureStopError, match=expected_error):
        stop_after_bootstrap_failure(
            **_stop_kwargs(root=tmp_path, phase=phase, receipt=receipt, nonce=nonce),
            transport=transport,
        )
    assert calls == ["GET"]


def test_bootstrap_arms_failure_stop_before_selective_verifier() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "bootstrap_gpu.sh"
    ).read_text(encoding="utf-8")

    trap_position = script.index("trap early_bootstrap_failure_exit EXIT")
    verifier_position = script.index("python3 -I -S scripts/verify_runpod_sync_bundle.py")
    mutation_position = script.index("mkdir -p .runpod")
    assert trap_position < verifier_position < mutation_position
    assert "bootstrap_failure_stop" in script[script.index("cleanup_on_exit()") :]
    early_stop = script[script.index("bootstrap_failure_stop()") : verifier_position]
    for argument in (
        "--expected-provider-gpu-id",
        "--allowed-data-center-ids-csv",
        "--expected-container-image",
    ):
        assert argument in early_stop
