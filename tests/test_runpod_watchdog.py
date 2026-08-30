from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import model_forensics.runpod_watchdog as watchdog_module
from model_forensics.io import read_json, stable_hash
from model_forensics.runpod_lifecycle import existing_pod_id_hash
from model_forensics.runpod_watchdog import (
    RunpodStopClient,
    WatchdogError,
    WatchdogLimits,
    _host_ack_payload,
    _write_host_rearm_ack,
    bind_lifecycle_pod,
    derive_deadline,
    parse_pod_metadata,
    run_watchdog,
    validate_host_rearm_ack,
    validate_live_metadata,
    wait_for_rearm_then_run_watchdog,
)

GPU_ID = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch@sha256:" + "a" * 64
DATA_CENTERS = ("US-IL-1",)
CUDA_VERSIONS = ("12.8",)
SESSION_FIXTURE_VALUE = "session-fixture-value"
HF_FIXTURE_VALUE = "hf-fixture-value"
SESSION_HASH = stable_hash({"opaque_gpu_session_id": SESSION_FIXTURE_VALUE})
HF_TOKEN_HASH = stable_hash({"hf_token": HF_FIXTURE_VALUE})
OLD_SESSION_FIXTURE_VALUE = "old-session-fixture-value"
OLD_SESSION_HASH = stable_hash({"opaque_gpu_session_id": OLD_SESSION_FIXTURE_VALUE})


def test_host_ack_rejects_reused_pid_with_changed_start_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_identity = stable_hash({"process_start": "first"})
    second_identity = stable_hash({"process_start": "second"})
    monkeypatch.setattr(
        watchdog_module,
        "_host_process_identity_hash",
        lambda _pid: first_identity,
    )
    now = datetime.now(UTC)
    payload = _host_ack_payload(
        expected_session_hash=SESSION_HASH,
        expected_phase="behavior_treatment_gpu",
        lifecycle_before_hash=stable_hash({"lifecycle": "stopped"}),
        pod_id="pod_123",
        watcher_pid=4242,
        acknowledged_at=now,
    )
    acknowledgement = tmp_path / "host_rearm_watchdog_ack.json"
    _write_host_rearm_ack(acknowledgement, payload)
    monkeypatch.setattr(
        watchdog_module,
        "_host_process_identity_hash",
        lambda _pid: second_identity,
    )

    with pytest.raises(WatchdogError, match="process identity changed"):
        validate_host_rearm_ack(
            acknowledgement,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_treatment_gpu",
            expected_lifecycle_hash=stable_hash({"lifecycle": "stopped"}),
            expected_pod_id="pod_123",
            observed_at=now,
        )


def _payload(
    *,
    pod_id: str = "pod_123",
    status: str = "RUNNING",
    display_name: str = "NVIDIA H100 80GB HBM3",
    count: int = 8,
    cost: float | str = "82.40",
    adjusted: float | None = None,
    started_at: str = "2026-08-29T12:00:00Z",
    locked: bool | None = None,
    image: str = IMAGE,
    secure_cloud: bool = True,
    data_center_id: str = "US-IL-1",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": pod_id,
        "name": "model-forensics-behavior-baseline",
        "imageName": image,
        "gpuCount": count,
        "costPerHr": cost,
        "desiredStatus": status,
        "lastStartedAt": started_at,
        "containerDiskInGb": 50,
        "volumeInGb": 650,
        "volumeMountPath": "/workspace",
        "networkVolume": None,
        "ports": ["22/tcp"],
        "machineId": "machine_123",
        "machine": {
            "gpuTypeId": display_name,
            "gpuDisplayName": display_name,
            "dataCenterId": data_center_id,
            "secureCloud": secure_cloud,
        },
        "publicIp": "192.0.2.10",
        "portMappings": {"22": 32101},
        # The real API can return environment variables. They must never be persisted.
        "env": {
            "HF_TOKEN": HF_FIXTURE_VALUE,
            "GPU_BUDGET_SESSION_ID": SESSION_FIXTURE_VALUE,
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
            "VLLM_CACHE_ROOT": "/workspace/.cache/vllm",
            "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
            "PUBLIC_KEY": "ssh-ed25519 AAAATEST",
        },
    }
    if adjusted is not None:
        payload["adjustedCostPerHr"] = adjusted
    if locked is not None:
        payload["locked"] = locked
    return payload


def _metadata(payload: dict[str, object], observed_at: datetime) -> object:
    return parse_pod_metadata(
        payload,
        expected_pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        expected_hf_token_hash=HF_TOKEN_HASH,
        observed_at=observed_at,
    )


@pytest.mark.parametrize(
    "provider_timestamp",
    (
        "2026-08-29 12:00:00.637 +0000 UTC",
        "2026-08-29 12:00:00 +0000 UTC",
    ),
)
def test_live_v1_go_timestamp_is_accepted(provider_timestamp: str) -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    metadata = _metadata(_payload(started_at=provider_timestamp), observed)

    assert metadata.last_started_at == datetime(
        2026,
        8,
        29,
        12,
        microsecond=637000 if ".637" in provider_timestamp else 0,
        tzinfo=UTC,
    )


def test_deadline_and_incurred_cost_come_from_live_metadata() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    metadata = _metadata(_payload(cost=20), observed)
    limits = WatchdogLimits(gpu_hard_stop_usd=220, maximum_runtime_hours=20)
    derived = derive_deadline(metadata, limits)

    assert derived.incurred_cost_usd == pytest.approx(20)
    assert derived.budget_deadline == datetime(2026, 8, 29, 22, 40, 12, tzinfo=UTC)
    assert derived.deadline == derived.budget_deadline
    assert derived.reason == "safe_budget"


def test_deadline_subtracts_prior_canonical_gpu_commitment() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    metadata = _metadata(_payload(cost=20), observed)
    limits = WatchdogLimits(
        gpu_hard_stop_usd=220,
        maximum_runtime_hours=20,
        prior_committed_gpu_usd=100,
    )
    derived = derive_deadline(metadata, limits)

    assert limits.global_safe_budget_usd == pytest.approx(213.4)
    assert limits.safe_budget_usd == pytest.approx(113.4)
    assert derived.budget_deadline == datetime(2026, 8, 29, 17, 40, 12, tzinfo=UTC)
    assert derived.deadline == derived.budget_deadline


def test_live_metadata_requires_exact_pod_hardware_status_quote_and_unlock() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    limits = WatchdogLimits(
        gpu_hard_stop_usd=220,
        maximum_runtime_hours=10,
        maximum_approved_hourly_total_usd=83,
    )
    metadata = _metadata(_payload(), observed)
    validate_live_metadata(
        metadata,
        expected_gpu_count=8,
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=DATA_CENTERS,
        allowed_cuda_versions=CUDA_VERSIONS,
        expected_container_image=IMAGE,
        limits=limits,
    )

    for payload, match in (
        (_payload(count=7), "exactly 8"),
        (_payload(display_name="NVIDIA A100-SXM4-80GB"), "does not match"),
        (_payload(status="EXITED"), "RUNNING"),
        (_payload(locked=True), "locked"),
        (_payload(cost=84), "approved compute quote"),
        (_payload(image="runpod/pytorch@sha256:" + "b" * 64), "live image"),
        (_payload(secure_cloud=False), "Secure Cloud"),
        (_payload(data_center_id="US-TX-1"), "data center"),
    ):
        with pytest.raises(WatchdogError, match=match):
            validate_live_metadata(
                _metadata(payload, observed),
                expected_gpu_count=8,
                expected_gpu_family="H100_80GB",
                expected_provider_gpu_id=GPU_ID,
                allowed_data_center_ids=DATA_CENTERS,
                allowed_cuda_versions=CUDA_VERSIONS,
                expected_container_image=IMAGE,
                limits=limits,
            )

    split_brain = _payload()
    split_brain["gpu"] = {"id": GPU_ID, "count": 7}
    with pytest.raises(WatchdogError, match="aliases for gpuCount disagree"):
        _metadata(split_brain, observed)

    with pytest.raises(WatchdogError, match="different Pod"):
        parse_pod_metadata(
            _payload(pod_id="wrong_pod"),
            expected_pod_id="pod_123",
            expected_session_hash=SESSION_HASH,
            expected_hf_token_hash=HF_TOKEN_HASH,
            observed_at=observed,
        )


def test_v1_metadata_rejects_launch_drift_and_never_exposes_secret_values() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    limits = WatchdogLimits(220, 10, maximum_approved_hourly_total_usd=83)

    variants: list[tuple[dict[str, object], str]] = []
    wrong_disk = _payload()
    wrong_disk["containerDiskInGb"] = 51
    variants.append((wrong_disk, "container disk"))
    wrong_mount = _payload()
    wrong_mount["volumeInGb"] = 649
    variants.append((wrong_mount, "persistent volume"))
    wrong_networking = _payload()
    wrong_networking["globalNetworking"] = {"enabled": True}
    variants.append((wrong_networking, "global networking"))
    wrong_ports = _payload()
    wrong_ports["ports"] = ["22/tcp", "8888/http"]
    variants.append((wrong_ports, "ports"))
    network_volume = _payload()
    network_volume["networkVolume"] = {"id": "volume_123", "size": 650}
    variants.append((network_volume, "network volume"))

    for payload, match in variants:
        with pytest.raises(WatchdogError, match=match):
            validate_live_metadata(
                _metadata(payload, observed),
                expected_gpu_count=8,
                expected_gpu_family="H100_80GB",
                expected_provider_gpu_id=GPU_ID,
                allowed_data_center_ids=DATA_CENTERS,
                allowed_cuda_versions=CUDA_VERSIONS,
                expected_container_image=IMAGE,
                limits=limits,
            )

    secret_drift = _payload()
    assert isinstance(secret_drift["env"], dict)
    secret_drift["env"]["UNAPPROVED_SECRET"] = "do-not-echo-this"
    with pytest.raises(WatchdogError, match="allow-list") as error:
        _metadata(secret_drift, observed)
    assert "do-not-echo-this" not in str(error.value)

    alias_drift = _payload()
    alias_drift["image"] = "runpod/pytorch@sha256:" + "b" * 64
    with pytest.raises(WatchdogError, match="aliases for imageName disagree"):
        _metadata(alias_drift, observed)

    bad_route = _payload()
    bad_route["portMappings"] = {"22": 32101, "8888": 32102}
    with pytest.raises(WatchdogError, match="SSH-only"):
        _metadata(bad_route, observed)

    public = _metadata(_payload(), observed).public_dict()
    assert public["provider_api"] == "rest-v1"
    assert public["runtime_gpu_count"] is None
    assert public["cuda_version"] is None
    assert public["global_networking_enabled"] is None
    assert public["locked"] is None
    assert public["direct_ssh_ready"] is True
    assert public["direct_ssh_endpoint_hash"].startswith("sha256:")
    assert "192.0.2.10" not in json.dumps(public)
    assert "must-not-leak" not in json.dumps(public)


def test_network_volume_aliases_are_all_checked_and_conflicts_are_rejected() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    alias_only = _payload()
    alias_only["networkVolumeId"] = "volume_alias"
    with pytest.raises(WatchdogError, match="network volume"):
        validate_live_metadata(
            _metadata(alias_only, observed),
            expected_gpu_count=8,
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 10),
        )

    conflicting = _payload()
    conflicting["networkVolume"] = {"id": "volume_a"}
    conflicting["networkVolumeId"] = "volume_b"
    with pytest.raises(WatchdogError, match="aliases disagree"):
        _metadata(conflicting, observed)


def test_compute_ceiling_excludes_separately_calculated_storage() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    limits = WatchdogLimits(
        gpu_hard_stop_usd=220,
        maximum_runtime_hours=10,
        maximum_approved_compute_hourly_usd=26.32,
        maximum_approved_storage_hourly_usd=0.097222,
        maximum_approved_hourly_total_usd=26.417222,
    )
    metadata = _metadata(_payload(cost=26.40), observed)

    with pytest.raises(WatchdogError, match="compute quote"):
        validate_live_metadata(
            metadata,
            expected_gpu_count=8,
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=limits,
        )
    derived = derive_deadline(
        _metadata(_payload(cost=26.32), observed),
        limits,
        calculation_hourly_usd=26.32 + 0.097222,
    )
    assert derived.calculation_hourly_usd == pytest.approx(26.417222)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("GPU_BUDGET_SESSION_ID", "wrong-session-never-echo", "session identity"),
        ("HF_TOKEN", "wrong-hf-token-never-echo", "credential identity"),
    ),
)
def test_secret_identity_drift_is_rejected_without_leaking_values(
    field: str, replacement: str, match: str
) -> None:
    payload = _payload()
    environment = payload["env"]
    assert isinstance(environment, dict)
    environment[field] = replacement

    with pytest.raises(WatchdogError, match=match) as error:
        _metadata(payload, datetime(2026, 8, 29, 13, tzinfo=UTC))
    rendered = str(error.value)
    assert replacement not in rendered
    assert SESSION_FIXTURE_VALUE not in rendered
    assert HF_FIXTURE_VALUE not in rendered
    assert SESSION_HASH not in rendered
    assert HF_TOKEN_HASH not in rendered


def test_watchdog_gets_live_metadata_stops_exact_pod_and_confirms_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "rpa_do-not-persist")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    metadata_responses = iter(
        (
            _payload(started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="EXITED", started_at=base.isoformat()),
        )
    )
    get_requests: list[tuple[str, str]] = []
    stop_requests: list[tuple[str, str, dict[str, str] | None]] = []

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del timeout
        get_requests.append((url, api_key))
        return 200, json.dumps(next(metadata_responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        del timeout
        stop_requests.append((url, api_key, payload))
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    times = iter((base, base + timedelta(seconds=1), base + timedelta(seconds=1)))
    state_path = tmp_path / "watchdog.json"
    result = run_watchdog(
        pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=DATA_CENTERS,
        allowed_cuda_versions=CUDA_VERSIONS,
        expected_container_image=IMAGE,
        limits=WatchdogLimits(gpu_hard_stop_usd=220, maximum_runtime_hours=1 / 3600),
        state_path=state_path,
        client=client,
        now=lambda: next(times, base + timedelta(seconds=1)),
        sleep=lambda _: None,
    )

    assert get_requests[0][0] == (
        "https://rest.runpod.io/v1/pods/pod_123"
        "?includeMachine=true&includeNetworkVolume=true&includeTemplate=true"
    )
    assert stop_requests == [
        (
            "https://rest.runpod.io/v1/pods/pod_123/stop",
            "rpa_do-not-persist",
            None,
        )
    ]
    assert result["status"] == "stopped_confirmed"
    state_text = state_path.read_text(encoding="utf-8")
    assert read_json(state_path)["action"] == "stop_only_preserve_volume"
    assert "rpa_do-not-persist" not in state_text
    assert "must-not-leak" not in state_text


def test_live_verification_failure_requests_stop_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    responses = iter(
        (
            _payload(display_name="NVIDIA A100-SXM4-80GB"),
            _payload(status="RUNNING"),
            _payload(status="EXITED"),
        )
    )
    methods: list[str] = []

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        methods.append("GET")
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        del url, api_key, timeout, payload
        methods.append("POST_STOP")
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    state_path = tmp_path / "state.json"
    with pytest.raises(WatchdogError, match="does not match"):
        run_watchdog(
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 10),
            state_path=state_path,
            client=client,
            now=lambda: base,
            sleep=lambda _: None,
        )
    assert methods == ["GET", "GET", "POST_STOP", "GET"]
    state = read_json(state_path)
    assert state["status"] == "stopped_confirmed"
    assert state["deletion"] == "manual_after_verified_sync"


def test_watchdog_fails_closed_without_key_or_valid_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_RUNPOD_KEY", raising=False)
    with pytest.raises(WatchdogError, match="MISSING_RUNPOD_KEY"):
        RunpodStopClient(
            pod_id="pod_123",
            expected_session_hash=SESSION_HASH,
            api_key_env="MISSING_RUNPOD_KEY",
        )
    with pytest.raises(ValueError, match="positive"):
        WatchdogLimits(gpu_hard_stop_usd=0, maximum_runtime_hours=1)
    with pytest.raises(ValueError, match="leaves no"):
        WatchdogLimits(
            gpu_hard_stop_usd=220,
            maximum_runtime_hours=1,
            prior_committed_gpu_usd=213.4,
        )


def test_monotonic_guard_stops_even_if_wall_clock_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    responses = iter(
        (
            _payload(started_at=base.isoformat()),
            _payload(started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="EXITED", started_at=base.isoformat()),
        )
    )
    stop_count = 0

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_count
        del url, api_key, timeout, payload
        stop_count += 1
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    monotonic_values = iter((0.0, 0.0, 1.0, 1.0))
    result = run_watchdog(
        pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=DATA_CENTERS,
        allowed_cuda_versions=CUDA_VERSIONS,
        expected_container_image=IMAGE,
        limits=WatchdogLimits(220, 1 / 3600),
        state_path=tmp_path / "state.json",
        client=client,
        now=lambda: base,
        monotonic=lambda: next(monotonic_values, 1.0),
        sleep=lambda _: None,
    )
    assert stop_count == 1
    assert result["status"] == "stopped_confirmed"


def _write_lifecycle_state(
    root: Path,
    *,
    pod_id: str = "pod_123",
    operation: str = "created",
    pod_status: str = "RUNNING",
    phase: str = "behavior_baseline_gpu",
    session_hash: str = SESSION_HASH,
    acknowledged_existing_pod_id_hashes: list[str] | None = None,
    history: list[dict[str, object]] | None = None,
) -> Path:
    private = root / ".runpod"
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    immutable_spec = {"gpu": {"count": 8, "id": GPU_ID}, "image": IMAGE}
    current: dict[str, object] = {
        "phase": phase,
        "reservation_id": stable_hash({"reservation": session_hash}),
        "reservation_record_hash": stable_hash({"receipt": session_hash}),
        "session_hash": session_hash,
        "approval_hash": stable_hash({"approval": 1}),
        "bindings_hash": stable_hash({"bindings": 1}),
        "gpu_lock_hash": stable_hash({"gpu_lock": 1}),
        "quote_hash": stable_hash({"quote": 1}),
        "immutable_spec_hash": stable_hash(immutable_spec),
        "launch_spec_hash": stable_hash({"launch": session_hash}),
        "acknowledged_existing_pod_id_hashes": (
            acknowledged_existing_pod_id_hashes or []
        ),
        "approved_runtime_hours": 1.5,
        "approved_phase_maximum_usd": 39.625834,
        "live_hourly_total_usd": 26.417222,
    }
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-pod-lifecycle-v1",
        "operation": operation,
        "updated_at": "2026-08-29T12:00:00Z",
        "immutable_spec": immutable_spec,
        "current_authorization": current,
        "authorization_history": history or [],
        "pod": {
            "id": pod_id,
            "name": "model-forensics-behavior-baseline",
            "status": pod_status,
        },
    }
    path = private / "pod_lifecycle.json"
    path.write_text(
        json.dumps({**unsigned, "record_hash": stable_hash(unsigned)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_lifecycle_binding_rejects_same_spec_wrong_id_and_allowlisted_unrelated_pod(
    tmp_path: Path,
) -> None:
    lifecycle = _write_lifecycle_state(tmp_path, pod_id="research_pod")
    with pytest.raises(WatchdogError, match="not the private lifecycle-bound"):
        bind_lifecycle_pod(
            lifecycle_state_path=lifecycle,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_baseline_gpu",
            ambient_pod_id="same_spec_wrong_pod",
        )

    unrelated_id = "claude_unrelated_pod"
    lifecycle = _write_lifecycle_state(
        tmp_path,
        pod_id="research_pod",
        acknowledged_existing_pod_id_hashes=[existing_pod_id_hash(unrelated_id)],
    )
    assert unrelated_id not in lifecycle.read_text(encoding="utf-8")
    with pytest.raises(WatchdogError, match="not the private lifecycle-bound"):
        bind_lifecycle_pod(
            lifecycle_state_path=lifecycle,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_baseline_gpu",
            ambient_pod_id=unrelated_id,
        )


def test_initial_verification_state_write_failure_cannot_suppress_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    responses = iter(
        (
            _payload(display_name="NVIDIA A100-SXM4-80GB"),
            _payload(status="RUNNING"),
            _payload(status="EXITED"),
        )
    )
    stop_calls = 0

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_calls
        del url, api_key, timeout, payload
        stop_calls += 1
        return 200, "{}"

    real_write_json = watchdog_module.write_json
    writes = 0

    def fail_first_write(path: str | Path, payload: object) -> Path:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("simulated private-state disk error")
        return real_write_json(path, payload)

    monkeypatch.setattr(watchdog_module, "write_json", fail_first_write)
    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    with pytest.raises(WatchdogError, match="does not match"):
        run_watchdog(
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 10),
            state_path=tmp_path / "watchdog.json",
            client=client,
            now=lambda: base,
            sleep=lambda _: None,
        )
    assert stop_calls == 1
    assert read_json(tmp_path / "watchdog.json")["status"] == "stopped_confirmed"


def test_stop_retries_continue_beyond_old_twelve_attempt_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    responses = iter(
        [
            _payload(display_name="NVIDIA A100-SXM4-80GB"),
            *[_payload(status="RUNNING") for _ in range(14)],
            _payload(status="EXITED"),
        ]
    )
    stop_calls = 0

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_calls
        del url, api_key, timeout, payload
        stop_calls += 1
        return 503, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    with pytest.raises(WatchdogError, match="does not match"):
        run_watchdog(
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 10),
            state_path=tmp_path / "watchdog.json",
            client=client,
            now=lambda: base,
            sleep=lambda _: None,
        )
    assert stop_calls == 14
    assert read_json(tmp_path / "watchdog.json")["status"] == "stopped_confirmed"


def test_host_watcher_acknowledges_before_rearm_without_remote_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    lifecycle = _write_lifecycle_state(
        tmp_path,
        operation="stopped",
        pod_status="EXITED",
        session_hash=OLD_SESSION_HASH,
    )
    state_path = tmp_path / ".runpod" / "host_watchdog.json"
    acknowledgement_path = state_path.with_name("host_rearm_watchdog_ack.json")

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(_payload(status="EXITED"))

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        metadata_transport=get_transport,
    )

    class WatcherPaused(Exception):
        pass

    def pause_after_ack(seconds: float) -> None:
        del seconds
        assert read_json(state_path)["status"] == "waiting_for_start"
        assert read_json(acknowledgement_path)["status"] == (
            "armed_and_provider_exited_verified"
        )
        raise WatcherPaused

    with pytest.raises(WatcherPaused):
        wait_for_rearm_then_run_watchdog(
            lifecycle_state_path=lifecycle,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_treatment_gpu",
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 1.5),
            state_path=state_path,
            acknowledgement_path=acknowledgement_path,
            client=client,
            sleep=pause_after_ack,
        )


def test_host_watcher_bounds_desired_status_errors_and_attempts_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    lifecycle = _write_lifecycle_state(
        tmp_path,
        operation="stopped",
        pod_status="EXITED",
        session_hash=OLD_SESSION_HASH,
    )
    clock = [0.0]
    stop_calls = 0

    def unavailable_get(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 503, "{}"

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_calls
        del url, api_key, timeout, payload
        stop_calls += 1
        return 200, "{}"

    def advance(seconds: float) -> None:
        clock[0] += seconds

    state_path = tmp_path / ".runpod" / "host_watchdog.json"
    acknowledgement_path = state_path.with_name("host_rearm_watchdog_ack.json")
    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=unavailable_get,
    )

    with pytest.raises(WatchdogError, match="failed to confirm Pod stop"):
        wait_for_rearm_then_run_watchdog(
            lifecycle_state_path=lifecycle,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_treatment_gpu",
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 1.5),
            state_path=state_path,
            acknowledgement_path=acknowledgement_path,
            client=client,
            running_readiness_timeout_seconds=10,
            stop_attempts=1,
            sleep=advance,
            monotonic=lambda: clock[0],
        )

    assert stop_calls == 1
    assert not acknowledgement_path.exists()
    stopped = read_json(state_path)
    assert stopped["status"] == "stop_unconfirmed"
    assert stopped["stop_reason"] == "rearm_start_or_readiness_timeout"


def test_host_watcher_stops_if_lifecycle_becomes_unreadable_after_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    lifecycle = _write_lifecycle_state(
        tmp_path,
        operation="stopped",
        pod_status="EXITED",
        session_hash=OLD_SESSION_HASH,
    )
    responses = iter(
        (
            _payload(status="EXITED"),
            _payload(status="RUNNING"),
            _payload(status="EXITED"),
        )
    )
    stop_calls = 0
    corrupted = False

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_calls
        del url, api_key, timeout, payload
        stop_calls += 1
        return 200, "{}"

    def corrupt_after_ack(_seconds: float) -> None:
        nonlocal corrupted
        if not corrupted:
            lifecycle.write_text("not-json\n", encoding="utf-8")
            corrupted = True

    state_path = tmp_path / ".runpod" / "host_watchdog.json"
    acknowledgement_path = state_path.with_name("host_rearm_watchdog_ack.json")
    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )

    with pytest.raises(WatchdogError, match="lifecycle binding is invalid"):
        wait_for_rearm_then_run_watchdog(
            lifecycle_state_path=lifecycle,
            expected_session_hash=SESSION_HASH,
            expected_phase="behavior_treatment_gpu",
            pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            allowed_cuda_versions=CUDA_VERSIONS,
            expected_container_image=IMAGE,
            limits=WatchdogLimits(220, 1.5),
            state_path=state_path,
            acknowledgement_path=acknowledgement_path,
            client=client,
            stop_attempts=1,
            sleep=corrupt_after_ack,
        )

    assert acknowledgement_path.is_file()
    assert stop_calls == 1
    stopped = read_json(state_path)
    assert stopped["status"] == "stopped_confirmed"
    assert stopped["stop_reason"] == "lifecycle_read_failed_during_rearm"


def test_host_watcher_survives_rearm_client_crash_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "secret")
    monkeypatch.setenv("TEST_HF_TOKEN", HF_FIXTURE_VALUE)
    lifecycle = _write_lifecycle_state(
        tmp_path,
        operation="stopped",
        pod_status="EXITED",
        session_hash=OLD_SESSION_HASH,
    )
    old_authorization = json.loads(lifecycle.read_text(encoding="utf-8"))[
        "current_authorization"
    ]
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    responses = iter(
        (
            _payload(status="EXITED", started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="EXITED", started_at=base.isoformat()),
        )
    )
    stop_calls = 0
    transitioned = False

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        return 200, json.dumps(next(responses))

    def stop_transport(
        url: str,
        api_key: str,
        timeout: float,
        payload: dict[str, str] | None,
    ) -> tuple[int, str]:
        nonlocal stop_calls
        del url, api_key, timeout, payload
        stop_calls += 1
        return 200, "{}"

    def simulate_rearm_process_crash(seconds: float) -> None:
        nonlocal transitioned
        del seconds
        if transitioned:
            return
        transitioned = True
        _write_lifecycle_state(
            tmp_path,
            operation="rearm_patched",
            pod_status="EXITED",
            phase="behavior_treatment_gpu",
            session_hash=SESSION_HASH,
            history=[old_authorization],
        )

    client = RunpodStopClient(
        pod_id="pod_123",
        expected_session_hash=SESSION_HASH,
        api_key_env="TEST_RUNPOD_KEY",
        hf_token_env="TEST_HF_TOKEN",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    result = wait_for_rearm_then_run_watchdog(
        lifecycle_state_path=lifecycle,
        expected_session_hash=SESSION_HASH,
        expected_phase="behavior_treatment_gpu",
        pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=DATA_CENTERS,
        allowed_cuda_versions=CUDA_VERSIONS,
        expected_container_image=IMAGE,
        limits=WatchdogLimits(220, 1 / 3600),
        state_path=tmp_path / ".runpod" / "host_watchdog.json",
        acknowledgement_path=(
            tmp_path / ".runpod" / "host_rearm_watchdog_ack.json"
        ),
        client=client,
        sleep=simulate_rearm_process_crash,
        now=lambda: base + timedelta(seconds=2),
    )
    assert transitioned is True
    assert stop_calls == 1
    assert result["status"] == "stopped_confirmed"
