from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from model_forensics.approval import load_paid_run_approval
from model_forensics.execution_bindings import (
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.io import stable_hash

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_paid_bundle.py"


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _prepare_project(tmp_path: Path) -> tuple[list[str], str, str]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in ("run_122b.yaml", "preregistration.yaml", "gpu_lock.yaml"):
        shutil.copyfile(ROOT / "config" / name, config_dir / name)

    private = tmp_path / ".runpod"
    specs = private / "specs"
    specs.mkdir(parents=True, mode=0o700)
    private.chmod(0o700)
    specs.chmod(0o700)

    quote_time = datetime.now(UTC) - timedelta(minutes=2)
    approval_time = quote_time + timedelta(minutes=1)
    gpu_spec = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "runpod-h100-reviewed-20260829-001",
        "gpu_family": "H100_80GB",
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["US-IL-1"],
        "gpu_count": 8,
        "container_disk_gb": 50,
        "volume_disk_gb": 650,
        "usd_per_gpu_hour": 1.0,
        "running_storage_usd_per_hour": 700 * 0.10 / 720,
        "quoted_at": _iso(quote_time),
        "phase_runtime_allocations": [
            {
                "command_phase": "behavior_baseline_gpu",
                "maximum_runtime_hours": 1.0,
            },
            {
                "command_phase": "behavior_treatment_gpu",
                "maximum_runtime_hours": 1.0,
            },
            {"command_phase": "resample_gpu", "maximum_runtime_hours": 1.0},
            {"command_phase": "lens_gpu", "maximum_runtime_hours": 1.0},
        ],
        "source_url": "https://www.runpod.io/pricing",
    }

    preregistration = yaml.safe_load(
        (config_dir / "preregistration.yaml").read_text(encoding="utf-8")
    )
    external = preregistration["external_judging"]
    primary = external["high_volume_outcome_and_trajectory"]
    calibration = external["outcome_calibration"]
    semantic = {item["role"]: item for item in external["semantic_classification_routes"]}
    independent = next(
        item
        for item in external["semantic_classification_routes"]
        if item["model"] == calibration["independent_model"]
    )

    def route(role: str, source: dict[str, object]) -> dict[str, object]:
        return {
            "role": role,
            "model": source["model"],
            "input_usd_per_million_tokens": source["input_usd_per_million_tokens"],
            "output_usd_per_million_tokens": source["output_usd_per_million_tokens"],
        }

    api_spec = {
        "schema_version": 1,
        "provider": "openrouter",
        "source_url": "https://openrouter.ai/models",
        "checked_at": _iso(quote_time),
        "routes": [
            route("primary_final_and_trajectory", primary),
            route("independent_final", independent),
            route("classifier_anthropic", semantic["strongest_anthropic_route"]),
            route(
                "classifier_google",
                semantic["independent_frontier_google_route"],
            ),
        ],
    }
    for name, payload in (
        ("gpu_quote_spec.json", gpu_spec),
        ("api_route_quote_spec.json", api_spec),
    ):
        path = specs / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)

    common = [
        "--config",
        "config/run_122b.yaml",
        "--preregistration",
        "config/preregistration.yaml",
        "--gpu-lock",
        "config/gpu_lock.yaml",
        "--gpu-quote-lock",
        ".runpod/gpu_quote_lock.json",
        "--api-quote-lock",
        ".runpod/api_route_quote_lock.json",
    ]
    return common, _iso(quote_time), _iso(approval_time)


def _run(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _preview(project: Path, common: list[str]) -> subprocess.CompletedProcess[str]:
    return _run(
        project,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
    )


def test_preview_exclusively_freezes_authenticated_private_quote_locks(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)

    completed = _preview(tmp_path, common)

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["status"] == "preview"
    assert output["paid_execution_authorized"] is False
    assert output["ready_for_explicit_user_approval"] is True
    assert output["approval_schema_version"] == 2
    assert output["gpu"]["provider_gpu_id"] == "NVIDIA H100 80GB HBM3"
    assert output["gpu"]["cloud_type"] == "SECURE"
    assert output["gpu"]["allowed_cuda_versions"] == ["12.8"]
    assert output["gpu"]["data_center_ids"] == ["US-IL-1"]
    assert output["gpu"]["container_disk_gb"] == 50
    assert output["gpu"]["volume_disk_gb"] == 650
    assert output["gpu"]["projected_compute_usd"] == 32.0
    assert output["gpu"]["projected_running_storage_usd"] == 0.388889
    assert output["gpu"]["projected_maximum_usd"] == 32.388889
    assert output["hashes"]["gpu_lock"] == stable_hash(
        yaml.safe_load((tmp_path / "config/gpu_lock.yaml").read_text(encoding="utf-8"))
    )
    assert output["paths"] == {
        "config": "config/run_122b.yaml",
        "preregistration": "config/preregistration.yaml",
        "gpu_lock": "config/gpu_lock.yaml",
        "gpu_quote_lock": ".runpod/gpu_quote_lock.json",
        "api_quote_lock": ".runpod/api_route_quote_lock.json",
    }
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()
    for name in ("gpu_quote_lock.json", "api_route_quote_lock.json"):
        path = tmp_path / ".runpod" / name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / ".runpod").stat().st_mode) == 0o700
    load_gpu_quote_lock(tmp_path / ".runpod/gpu_quote_lock.json")
    load_api_route_quote_lock(tmp_path / ".runpod/api_route_quote_lock.json")

    authenticated_again = _run(tmp_path, "preview", *common)
    assert authenticated_again.returncode == 0, authenticated_again.stderr
    assert json.loads(authenticated_again.stdout)["hashes"] == output["hashes"]


def test_approve_requires_explicit_metadata_and_binds_full_gpu_lock(tmp_path: Path) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    assert _preview(tmp_path, common).returncode == 0
    approval_id = "yib-approval-20260829-001"

    missing_metadata = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
    )
    assert missing_metadata.returncode == 2
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()

    completed = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--approval-id",
        approval_id,
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
        "--allow-phase",
        "behavior_baseline_api",
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    approval_path = tmp_path / ".runpod/paid_run_approval.json"
    approval = load_paid_run_approval(approval_path)
    gpu_lock = yaml.safe_load((tmp_path / "config/gpu_lock.yaml").read_text(encoding="utf-8"))
    assert approval.bindings.gpu_lock_hash == stable_hash(gpu_lock)
    assert approval.allowed_command_phases == (
        "behavior_baseline_gpu",
        "behavior_baseline_api",
    )
    assert output["approval_id_hash"] == stable_hash(approval_id)
    assert approval_id not in completed.stdout
    assert stat.S_IMODE(approval_path.stat().st_mode) == 0o600

    original = approval_path.read_bytes()
    overwrite = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--approval-id",
        "yib-approval-20260829-002",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert overwrite.returncode == 2
    assert approval_path.read_bytes() == original


def test_approve_rejects_predated_approval_before_claiming_output(tmp_path: Path) -> None:
    common, quote_time, _ = _prepare_project(tmp_path)
    assert _preview(tmp_path, common).returncode == 0
    predates_quote = _iso(
        datetime.fromisoformat(quote_time.replace("Z", "+00:00")) - timedelta(seconds=1)
    )

    completed = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--approval-id",
        "yib-approval-20260829-003",
        "--approved-at",
        predates_quote,
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 2
    assert "predates the GPU quote" in completed.stderr
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()


def test_private_schema_failure_does_not_echo_mistaken_secret_value(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    spec_path = tmp_path / ".runpod/specs/gpu_quote_spec.json"
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    mistaken_secret = "sk-abcdefghijklmnop"
    raw["unexpected_private_value"] = mistaken_secret
    spec_path.write_text(json.dumps(raw), encoding="utf-8")
    spec_path.chmod(0o600)

    completed = _preview(tmp_path, common)

    assert completed.returncode == 2
    assert mistaken_secret not in completed.stdout
    assert mistaken_secret not in completed.stderr
    assert "private paid-bundle validation failed" in completed.stderr
    assert not (tmp_path / ".runpod/gpu_quote_lock.json").exists()
    assert not (tmp_path / ".runpod/api_route_quote_lock.json").exists()


def test_private_root_symlink_and_approval_outside_private_root_fail_closed(
    tmp_path: Path,
) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(tmp_path / ".runpod")
    (tmp_path / ".runpod").symlink_to(outside, target_is_directory=True)

    symlinked = _run(tmp_path, "preview", *common)
    assert symlinked.returncode == 2
    assert "must not be a symlink" in symlinked.stderr
    assert not any(outside.iterdir())

    second_project = tmp_path / "second"
    second_project.mkdir()
    common, _, approval_time = _prepare_project(second_project)
    assert _preview(second_project, common).returncode == 0
    escaped = _run(
        second_project,
        "approve",
        *common,
        "--output",
        "paid_run_approval.json",
        "--approval-id",
        "yib-approval-20260829-004",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert escaped.returncode == 2
    assert "must remain under" in escaped.stderr
    assert not (second_project / "paid_run_approval.json").exists()
