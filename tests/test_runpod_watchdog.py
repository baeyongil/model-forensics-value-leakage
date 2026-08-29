from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_forensics.io import read_json
from model_forensics.runpod_watchdog import (
    RunpodStopClient,
    WatchdogError,
    WatchdogLimits,
    derive_deadline,
    parse_pod_metadata,
    run_watchdog,
    validate_live_metadata,
)

GPU_ID = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch@sha256:" + "a" * 64
DATA_CENTERS = ("US-IL-1",)


def _payload(
    *,
    pod_id: str = "pod_123",
    status: str = "RUNNING",
    display_name: str = "NVIDIA H100 80GB HBM3",
    count: int = 8,
    cost: float | str = "82.40",
    adjusted: float = 80.0,
    started_at: str = "2026-08-29T12:00:00Z",
    locked: bool = False,
    image: str = IMAGE,
    machine_id: str = "machine_123",
    secure_cloud: bool = True,
    data_center_id: str = "US-IL-1",
) -> dict[str, object]:
    return {
        "id": pod_id,
        "gpu": {"id": display_name, "count": count, "displayName": display_name},
        "machine": {
            "gpuTypeId": display_name,
            "gpuType": {"displayName": display_name},
            "gpuDisplayName": display_name,
            "secureCloud": secure_cloud,
            "dataCenterId": data_center_id,
        },
        "machineId": machine_id,
        "image": image,
        "costPerHr": cost,
        "adjustedCostPerHr": adjusted,
        "desiredStatus": status,
        "lastStartedAt": started_at,
        "locked": locked,
        "networkVolume": None,
        # The real API can return environment variables. They must never be persisted.
        "env": {"PRIVATE_TOKEN": "must-not-leak"},
    }


def _metadata(payload: dict[str, object], observed_at: datetime) -> object:
    return parse_pod_metadata(payload, expected_pod_id="pod_123", observed_at=observed_at)


def test_deadline_and_incurred_cost_come_from_live_metadata() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    metadata = _metadata(_payload(adjusted=20), observed)
    limits = WatchdogLimits(gpu_hard_stop_usd=220, maximum_runtime_hours=20)
    derived = derive_deadline(metadata, limits)

    assert derived.incurred_cost_usd == pytest.approx(20)
    assert derived.budget_deadline == datetime(2026, 8, 29, 22, 40, 12, tzinfo=UTC)
    assert derived.deadline == derived.budget_deadline
    assert derived.reason == "safe_budget"


def test_deadline_subtracts_prior_canonical_gpu_commitment() -> None:
    observed = datetime(2026, 8, 29, 13, tzinfo=UTC)
    metadata = _metadata(_payload(adjusted=20), observed)
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
        expected_container_image=IMAGE,
        limits=limits,
    )

    for payload, match in (
        (_payload(count=7), "exactly 8"),
        (_payload(display_name="NVIDIA A100-SXM4-80GB"), "does not match"),
        (_payload(status="EXITED"), "RUNNING"),
        (_payload(locked=True), "locked"),
        (_payload(cost=84), "approved quote"),
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
                expected_container_image=IMAGE,
                limits=limits,
            )

    split_brain = _payload()
    split_brain["machine"] = {
        "gpuTypeId": "NVIDIA A100-SXM4-80GB",
        "gpuType": {"displayName": "NVIDIA A100-SXM4-80GB"},
        "secureCloud": True,
        "dataCenterId": "US-IL-1",
    }
    with pytest.raises(WatchdogError, match="machine GPU identity"):
        validate_live_metadata(
            _metadata(split_brain, observed),
            expected_gpu_count=8,
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=DATA_CENTERS,
            expected_container_image=IMAGE,
            limits=limits,
        )

    with pytest.raises(WatchdogError, match="different Pod"):
        parse_pod_metadata(
            _payload(pod_id="wrong_pod"),
            expected_pod_id="pod_123",
            observed_at=observed,
        )


def test_watchdog_gets_live_metadata_stops_exact_pod_and_confirms_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_RUNPOD_KEY", "rpa_do-not-persist")
    base = datetime(2026, 8, 29, 12, tzinfo=UTC)
    metadata_responses = iter(
        (
            _payload(started_at=base.isoformat()),
            _payload(status="RUNNING", started_at=base.isoformat()),
            _payload(status="EXITED", started_at=base.isoformat()),
        )
    )
    get_requests: list[tuple[str, str]] = []
    stop_requests: list[tuple[str, str]] = []

    def get_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del timeout
        get_requests.append((url, api_key))
        return 200, json.dumps(next(metadata_responses))

    def stop_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del timeout
        stop_requests.append((url, api_key))
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        api_key_env="TEST_RUNPOD_KEY",
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
        expected_container_image=IMAGE,
        limits=WatchdogLimits(gpu_hard_stop_usd=220, maximum_runtime_hours=1 / 3600),
        state_path=state_path,
        client=client,
        now=lambda: next(times, base + timedelta(seconds=1)),
        sleep=lambda _: None,
    )

    assert get_requests[0][0].startswith("https://rest.runpod.io/v1/pods/pod_123?")
    assert "includeMachine=true" in get_requests[0][0]
    assert stop_requests == [
        ("https://rest.runpod.io/v1/pods/pod_123/stop", "rpa_do-not-persist")
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

    def stop_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        del url, api_key, timeout
        methods.append("POST_STOP")
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        api_key_env="TEST_RUNPOD_KEY",
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
        RunpodStopClient(pod_id="pod_123", api_key_env="MISSING_RUNPOD_KEY")
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

    def stop_transport(url: str, api_key: str, timeout: float) -> tuple[int, str]:
        nonlocal stop_count
        del url, api_key, timeout
        stop_count += 1
        return 200, "{}"

    client = RunpodStopClient(
        pod_id="pod_123",
        api_key_env="TEST_RUNPOD_KEY",
        transport=stop_transport,
        metadata_transport=get_transport,
    )
    monotonic_values = iter((0.0, 0.0, 1.0, 1.0))
    result = run_watchdog(
        pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=DATA_CENTERS,
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
