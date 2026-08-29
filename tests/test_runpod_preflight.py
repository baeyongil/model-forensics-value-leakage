from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_forensics.runpod_sessions import record_watchdog_process_identity

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


def _fake_proc(
    tmp_path: Path,
    *,
    pid: int = 4242,
    start_ticks: int = 123456,
    argv: tuple[str, ...] = ("python3", "scripts/runpod_watchdog.py", "--state", "state.json"),
) -> Path:
    proc_root = tmp_path / "proc"
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True, exist_ok=True)
    # stat fields 3..21 precede field 22 (starttime).
    (process_root / "stat").write_text(
        f"{pid} (python3) S " + " ".join(["0"] * 18) + f" {start_ticks}\n",
        encoding="utf-8",
    )
    (process_root / "cmdline").write_bytes(
        b"\0".join(token.encode("utf-8") for token in argv) + b"\0"
    )
    boot_id = proc_root / "sys" / "kernel" / "random" / "boot_id"
    boot_id.parent.mkdir(parents=True, exist_ok=True)
    boot_id.write_text("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n", encoding="utf-8")
    return proc_root


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
    with pytest.raises(ValueError, match="future"):
        preflight.parse_fresh_price_timestamp(
            (now + timedelta(microseconds=1)).isoformat(),
            now=now,
        )


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
            "provider_api": "rest-v1",
            "provider_evidence_unavailable": [
                "cuda_version",
                "global_networking_enabled",
                "interruptible",
                "locked",
                "runtime_gpu_count",
            ],
            "pod_id": "pod_123",
            "gpu_count": 8,
            "provider_gpu_id": GPU_ID,
            "gpu_display_name": "NVIDIA H100 80GB HBM3",
            "runtime_gpu_count": None,
            "execution_identity_hash": "sha256:" + "b" * 64,
            "machine_id_hash": "sha256:" + "c" * 64,
            "data_center_id": "US-IL-1",
            "cuda_version": None,
            "secure_cloud": True,
            "container_image": IMAGE,
            "container_disk_gb": 50,
            "persistent_volume_disk_gb": 650,
            "persistent_volume_mount_path": "/workspace",
            "ports": ["22/tcp"],
            "global_networking_enabled": None,
            "interruptible": None,
            "network_volume_attached": False,
            "ssh_ready": True,
            "direct_ssh_ready": True,
            "direct_ssh_endpoint_hash": "sha256:" + "d" * 64,
            "environment_verified": True,
            "desired_status": "RUNNING",
            "cost_per_hr": 20.0,
            "adjusted_cost_per_hr": 20.0,
            "last_started_at": started.isoformat(),
            "locked": None,
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
            "calculation_hourly_usd": 20.1,
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
        allowed_cuda_versions=("12.8",),
        expected_container_image=IMAGE,
        expected_gpu_count=8,
        planned_hours=7,
        approved_hourly_total_usd=24,
        approved_storage_hourly_usd=0.1,
        gpu_budget_usd=220,
        now=now,
    )
    assert summary["live_effective_hourly_usd"] == 20
    assert summary["projected_cost_usd"] == pytest.approx(140.7)
    assert summary["provider_api"] == "rest-v1"
    assert summary["provider_evidence_unavailable"] == [
        "cuda_version",
        "global_networking_enabled",
        "interruptible",
        "locked",
        "runtime_gpu_count",
    ]

    wrong_pod = _watchdog_state(now)
    wrong_pod["pod_id"] = "pod_other"
    with pytest.raises(ValueError, match="different Pod"):
        preflight.validate_watchdog_state(
            wrong_pod,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"provider_api": "api-v2"}, "rest-v1"),
        ({"provider_evidence_unavailable": ["cuda_version"]}, "evidence gaps"),
        ({"runtime_gpu_count": 8}, "must be present and null"),
        ({"global_networking_enabled": False}, "must be present and null"),
        ({"interruptible": False}, "must be present and null"),
        ({"locked": False}, "must be present and null"),
    ),
)
def test_watchdog_v1_evidence_gaps_are_explicit_and_never_fabricated(
    mutation: dict[str, object], match: str
) -> None:
    now = datetime(2026, 8, 29, 13, tzinfo=UTC)
    state = _watchdog_state(now)
    metadata = state["live_metadata"]
    assert isinstance(metadata, dict)
    metadata.update(mutation)

    with pytest.raises(ValueError, match=match):
        preflight.validate_watchdog_state(
            state,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            allowed_cuda_versions=("12.8",),
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
    metadata = state["live_metadata"]
    assert isinstance(metadata, dict)
    metadata["cost_per_hr"] = 19.9
    metadata["adjusted_cost_per_hr"] = 19.9
    deadline["calculation_hourly_usd"] = 20.0
    started = now - timedelta(hours=1)
    deadline["effective_deadline"] = (started + timedelta(hours=7)).isoformat()

    summary = preflight.validate_watchdog_state(
        state,
        expected_pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=("US-IL-1",),
        allowed_cuda_versions=("12.8",),
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
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            expected_prior_committed_gpu_usd=70,
            now=now,
        )


def test_watchdog_rate_boundaries_separate_compute_from_storage() -> None:
    now = datetime(2026, 8, 29, 13, tzinfo=UTC)
    exact = _watchdog_state(now)
    metadata = exact["live_metadata"]
    deadline = exact["deadline"]
    assert isinstance(metadata, dict)
    assert isinstance(deadline, dict)
    metadata["cost_per_hr"] = 23.9
    metadata["adjusted_cost_per_hr"] = 23.9
    deadline["calculation_hourly_usd"] = 24.0
    summary = preflight.validate_watchdog_state(
        exact,
        expected_pod_id="pod_123",
        expected_gpu_family="H100_80GB",
        expected_provider_gpu_id=GPU_ID,
        allowed_data_center_ids=("US-IL-1",),
        allowed_cuda_versions=("12.8",),
        expected_container_image=IMAGE,
        expected_gpu_count=8,
        planned_hours=7,
        approved_hourly_total_usd=24,
        approved_storage_hourly_usd=0.1,
        gpu_budget_usd=220,
        now=now,
    )
    assert summary["approved_compute_hourly_usd"] == pytest.approx(23.9)

    over_compute = _watchdog_state(now)
    over_metadata = over_compute["live_metadata"]
    over_deadline = over_compute["deadline"]
    assert isinstance(over_metadata, dict)
    assert isinstance(over_deadline, dict)
    # This is below the old all-in comparison ($24) but above the actual
    # compute-only approval ($23.90).
    over_metadata["cost_per_hr"] = 23.900001
    over_metadata["adjusted_cost_per_hr"] = 23.900001
    over_deadline["calculation_hourly_usd"] = 24.000001
    with pytest.raises(ValueError, match="compute-only"):
        preflight.validate_watchdog_state(
            over_compute,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )

    understated = _watchdog_state(now)
    understated_metadata = understated["live_metadata"]
    understated_deadline = understated["deadline"]
    assert isinstance(understated_metadata, dict)
    assert isinstance(understated_deadline, dict)
    understated_metadata["cost_per_hr"] = 21.0
    understated_metadata["adjusted_cost_per_hr"] = 20.0
    understated_deadline["calculation_hourly_usd"] = 21.099
    with pytest.raises(ValueError, match="compute plus running storage"):
        preflight.validate_watchdog_state(
            understated,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )


def test_watchdog_state_rejects_any_future_timestamp() -> None:
    now = datetime(2026, 8, 29, 13, tzinfo=UTC)
    future = _watchdog_state(now)
    future["updated_at"] = (now + timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError, match="future"):
        preflight.validate_watchdog_state(
            future,
            expected_pod_id="pod_123",
            expected_gpu_family="H100_80GB",
            expected_provider_gpu_id=GPU_ID,
            allowed_data_center_ids=("US-IL-1",),
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
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
            allowed_cuda_versions=("12.8",),
            expected_container_image=IMAGE,
            expected_gpu_count=8,
            planned_hours=7,
            approved_hourly_total_usd=24,
            approved_storage_hourly_usd=0.1,
            gpu_budget_usd=220,
            now=now,
        )


def test_watchdog_pid_file_must_point_to_live_process(tmp_path: Path) -> None:
    proc_root = _fake_proc(tmp_path)
    pid_path = tmp_path / "watchdog.pid"
    recorded = record_watchdog_process_identity(
        pid_path,
        pid=4242,
        required_cmdline_tokens=("scripts/runpod_watchdog.py", "state.json"),
        proc_root=proc_root,
    )
    assert preflight.validate_watchdog_pid(pid_path, proc_root=proc_root) == recorded

    (proc_root / "4242" / "stat").write_text(
        "4242 (python3) S " + " ".join(["0"] * 18) + " 123457\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PID was reused"):
        preflight.validate_watchdog_pid(pid_path, proc_root=proc_root)

    (proc_root / "4242" / "stat").write_text(
        "4242 (python3) S " + " ".join(["0"] * 18) + " 123456\n",
        encoding="utf-8",
    )
    (proc_root / "4242" / "cmdline").write_bytes(b"python3\0unrelated.py\0")
    with pytest.raises(ValueError, match="identity changed"):
        preflight.validate_watchdog_pid(pid_path, proc_root=proc_root)


def test_bootstrap_arms_watchdog_before_any_download_and_never_deletes() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    budget_gate = script.index("python3 scripts/gpu_budget_preflight.py")
    arm = script.rindex('with_watchdog_credentials env PYTHONPATH="$PWD/src" nohup')
    identity = script.index("python3 scripts/runpod_process_identity.py record")
    readiness = script.index("if watchdog_is_armed")
    first_download = script.index("curl --fail")
    scrub = script.index("unset GPU_BUDGET_SESSION_ID RUNPOD_API_KEY HF_TOKEN")
    post_setup_gate = script.rindex("python3 scripts/runpod_active_session_verify.py")
    assert scrub < budget_gate < arm < identity < readiness < first_download < post_setup_gate
    assert "kill -0" not in script
    assert "printf -v GPU_BUDGET_SESSION_ID '%s' \"$GPU_BUDGET_SESSION_ID_PRIVATE\"" in script
    assert '--prior-committed-gpu-usd "$PRIOR_COMMITTED_GPU_USD"' in script
    assert '--maximum-runtime-hours "$MAXIMUM_SAFE_RUNTIME_HOURS"' in script
    assert "with_gpu_session_id env PYTHONPATH" in script
    assert '--session-id "$GPU_BUDGET_SESSION_ID"' not in script
    assert "scripts/runpod_session_prepare.py" in script
    assert "scripts/gpu_setup_lock.py validate" in script
    assert 'metadata.get("provider_api") == "rest-v1"' in script
    assert 'metadata.get("provider_evidence_unavailable")' in script
    assert "existing GPU environment is incomplete and cannot be re-armed" in script
    assert "data/manifests/runpod_watchdog.json" not in script
    assert "data/manifests/gpu_preflight.json" not in script
    assert "data/manifests/gpu_environment.json" not in script
    assert "/stop" not in script  # all mutation logic stays in the tested Python client
    assert "DELETE" not in script
    assert "pip install --upgrade" not in script
    assert "pip install uv" not in script
    assert "uv pip install" not in script
    assert "secret_free curl --fail" in script
    assert "secret_free .venv-gpu/bin/python -m pip install" in script
    assert "with_hf_token .venv-gpu/bin/python scripts/qwen4b_prefix_smoke.py" in script
    assert "-e . --no-deps" in script


def test_bootstrap_secret_wrappers_remove_inherited_private_exports() -> None:
    script = BOOTSTRAP.read_text(encoding="utf-8")
    wrapper_source = script[
        script.index('EXPECTED_SESSION_HASH=""') : script.index('case "$GPU_FAMILY" in')
    ]
    secret_names = (
        "GPU_BUDGET_SESSION_ID",
        "RUNPOD_API_KEY",
        "HF_TOKEN",
        "GPU_BUDGET_SESSION_ID_PRIVATE",
        "RUNPOD_API_KEY_PRIVATE",
        "HF_TOKEN_PRIVATE",
    )
    probe = (
        "import json, os; "
        f"names={secret_names!r}; "
        "print(json.dumps({name: os.environ.get(name) for name in names}, sort_keys=True))"
    )
    command = wrapper_source + "\n" + "\n".join(
        (
            'GPU_BUDGET_SESSION_ID_PRIVATE="actual-nonce"',
            'RUNPOD_API_KEY_PRIVATE="actual-runpod"',
            'HF_TOKEN_PRIVATE="actual-hf"',
            f"secret_free python3 -c {shlex.quote(probe)}",
            f"with_gpu_session_id python3 -c {shlex.quote(probe)}",
            f"with_hf_token python3 -c {shlex.quote(probe)}",
            f"with_watchdog_credentials python3 -c {shlex.quote(probe)}",
        )
    )
    inherited = {name: f"inherited-{name.lower()}" for name in secret_names}
    completed = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **inherited},
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]

    assert rows[0] == {name: None for name in secret_names}
    assert rows[1] == {
        **{name: None for name in secret_names},
        "GPU_BUDGET_SESSION_ID": "actual-nonce",
    }
    assert rows[2] == {
        **{name: None for name in secret_names},
        "HF_TOKEN": "actual-hf",
    }
    assert rows[3] == {
        **{name: None for name in secret_names},
        "RUNPOD_API_KEY": "actual-runpod",
        "HF_TOKEN": "actual-hf",
    }
