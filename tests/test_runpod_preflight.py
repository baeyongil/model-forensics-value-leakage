from __future__ import annotations

import importlib.util
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_preflight.py"
BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_gpu.sh"
SPEC = importlib.util.spec_from_file_location("runpod_preflight", SCRIPT)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)

GPU_ID = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch@sha256:" + "a" * 64


def _inventory(name: str = "NVIDIA H100 80GB HBM3") -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": name,
            "memory_gib": 79.1,
            "uuid": f"GPU-{index}",
            "mig_mode": "Disabled",
        }
        for index in range(8)
    ]


def test_inventory_requires_exactly_eight_homogeneous_approved_gpus() -> None:
    preflight.validate_inventory(
        _inventory(),
        required_gpus=8,
        minimum_memory_gib=79,
        expected_gpu_family="H100_80GB",
    )
    with pytest.raises(ValueError, match="exactly 8"):
        preflight.validate_inventory(
            _inventory()[:7],
            required_gpus=8,
            minimum_memory_gib=79,
            expected_gpu_family="H100_80GB",
        )
    mixed = _inventory()
    mixed[-1] = {**mixed[-1], "name": "NVIDIA A100-SXM4-80GB"}
    with pytest.raises(ValueError, match="homogeneous"):
        preflight.validate_inventory(
            mixed,
            required_gpus=8,
            minimum_memory_gib=79,
            expected_gpu_family="H100_80GB",
        )
    mig = _inventory()
    mig[0] = {**mig[0], "mig_mode": "Enabled"}
    with pytest.raises(ValueError, match="MIG"):
        preflight.validate_inventory(
            mig,
            required_gpus=8,
            minimum_memory_gib=79,
            expected_gpu_family="H100_80GB",
        )


def test_cuda_visibility_must_not_hide_or_duplicate_gpus() -> None:
    preflight.validate_cuda_visible_devices(None, required_gpus=8)
    preflight.validate_cuda_visible_devices("0,1,2,3,4,5,6,7", required_gpus=8)
    with pytest.raises(ValueError, match="exactly 8"):
        preflight.validate_cuda_visible_devices("0,1,2,3", required_gpus=8)


def test_price_check_must_be_timezone_aware_and_recent() -> None:
    now = datetime(2026, 8, 29, 18, tzinfo=UTC)
    assert preflight.parse_fresh_price_timestamp((now - timedelta(hours=1)).isoformat(), now=now)
    with pytest.raises(ValueError, match="six hours"):
        preflight.parse_fresh_price_timestamp((now - timedelta(hours=7)).isoformat(), now=now)
    with pytest.raises(ValueError, match="timezone"):
        preflight.parse_fresh_price_timestamp("2026-08-29T17:00:00", now=now)


def _watchdog_state(now: datetime) -> dict[str, object]:
    started = now - timedelta(hours=1)
    deadline = started + timedelta(hours=10)
    return {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "pod_id": "pod_123",
        "status": "armed",
        "updated_at": now.isoformat(),
        "action": "stop_only_preserve_volume",
        "live_metadata": {
            "pod_id": "pod_123",
            "gpu_count": 8,
            "provider_gpu_id": GPU_ID,
            "gpu_display_name": "NVIDIA H100 80GB HBM3",
            "machine_gpu_identity": ["NVIDIA H100 80GB HBM3"],
            "machine_id_hash": "sha256:" + "b" * 64,
            "data_center_id": "US-IL-1",
            "secure_cloud": True,
            "container_image": IMAGE,
            "desired_status": "RUNNING",
            "cost_per_hr": 20.0,
            "adjusted_cost_per_hr": 20.0,
            "last_started_at": started.isoformat(),
            "locked": False,
        },
        "limits": {
            "gpu_hard_stop_usd": 220,
            "global_safe_budget_usd": 213.4,
            "safe_budget_usd": 213.4,
            "safety_margin_fraction": 0.03,
            "prior_committed_gpu_usd": 0.0,
            "maximum_approved_hourly_total_usd": 24.0,
            "maximum_approved_storage_hourly_usd": 0.1,
        },
        "deadline": {
            "effective_deadline": deadline.isoformat(),
            "calculation_hourly_usd": 20.0,
            "incurred_cost_usd": 20.0,
        },
    }


def test_watchdog_state_must_be_live_exact_and_fit_planned_total_runtime() -> None:
    now = datetime(2026, 8, 29, 13, tzinfo=UTC)
    summary = preflight.validate_watchdog_state(
        _watchdog_state(now),
        expected_pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=("US-IL-1",),
        expected_container_image=IMAGE,
        expected_gpu_count=8,
        planned_hours=7,
        approved_hourly_total_usd=24,
        approved_storage_hourly_usd=0.1,
        gpu_budget_usd=220,
        now=now,
    )
    assert summary["live_effective_hourly_usd"] == 20
    assert summary["projected_cost_usd"] == 140

    wrong_pod = _watchdog_state(now)
    wrong_pod["pod_id"] = "pod_other"
    with pytest.raises(ValueError, match="different Pod"):
        preflight.validate_watchdog_state(
            wrong_pod,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )


def test_watchdog_state_binds_prior_canonical_gpu_spend_and_remaining_budget() -> None:
    now = datetime(2026, 8, 29, 13, tzinfo=UTC)
    state = _watchdog_state(now)
    limits = state["limits"]
    assert isinstance(limits, dict)
    limits["prior_committed_gpu_usd"] = 73.4
    limits["safe_budget_usd"] = 140.0
    deadline = state["deadline"]
    assert isinstance(deadline, dict)
    started = now - timedelta(hours=1)
    deadline["effective_deadline"] = (started + timedelta(hours=7)).isoformat()

    summary = preflight.validate_watchdog_state(
        state,
        expected_pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=("US-IL-1",),
        expected_container_image=IMAGE,
        expected_gpu_count=8,
        planned_hours=7,
        approved_hourly_total_usd=24,
        approved_storage_hourly_usd=0.1,
        gpu_budget_usd=220,
        expected_prior_committed_gpu_usd=73.4,
        now=now,
    )
    assert summary["prior_committed_gpu_usd"] == 73.4
    assert summary["safe_budget_usd"] == 140

    with pytest.raises(ValueError, match="canonical ledger"):
        preflight.validate_watchdog_state(
            state,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            expected_prior_committed_gpu_usd=70,
            now=now,
        )

    stale = _watchdog_state(now - timedelta(minutes=2))
    with pytest.raises(ValueError, match="90 seconds"):
        preflight.validate_watchdog_state(
            stale,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )


def test_watchdog_pid_file_must_point_to_live_process(tmp_path: Path) -> None:
    pid_path = tmp_path / "watchdog.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert preflight.validate_watchdog_pid(pid_path) == os.getpid()
    pid_path.write_text("not-a-pid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid"):
        preflight.validate_watchdog_pid(pid_path)


def test_bootstrap_arms_watchdog_before_any_download_and_never_deletes() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    budget_gate = script.index("python3 scripts/gpu_budget_preflight.py")
    arm = script.index("nohup python3 scripts/runpod_watchdog.py")
    readiness = script.index("if watchdog_is_armed")
    first_download = script.index("curl --fail")
    assert budget_gate < arm < readiness < first_download
    assert '--prior-committed-gpu-usd "$PRIOR_COMMITTED_GPU_USD"' in script
    assert '--maximum-runtime-hours "$MAXIMUM_SAFE_RUNTIME_HOURS"' in script
    assert "env -u GPU_BUDGET_SESSION_ID" in script
    assert '--session-id "$GPU_BUDGET_SESSION_ID"' not in script
    assert "scripts/runpod_session_prepare.py" in script
    assert "scripts/gpu_setup_lock.py validate" in script
    assert "existing GPU environment is incomplete and cannot be re-armed" in script
    assert "data/manifests/runpod_watchdog.json" not in script
    assert "data/manifests/gpu_preflight.json" not in script
    assert "data/manifests/gpu_environment.json" not in script
    assert "/stop" not in script  # all mutation logic stays in the tested Python client
    assert "DELETE" not in script
