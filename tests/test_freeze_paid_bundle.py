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
from model_forensics.budget import BudgetLimits, CostEntry, CostLedger
from model_forensics.execution_bindings import (
    load_api_route_quote_lock,
    load_gpu_quote_lock,
)
from model_forensics.io import stable_hash

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_paid_bundle.py"


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _git(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


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

    ledger_path = tmp_path / "data" / "manifests" / "cost_ledger.yaml"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "currency": "USD",
                "hard_stops": {"gpu": 220.0, "api": 100.0, "total": 325.0},
                "entries": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text(
        ".runpod/\ndata/manifests/.*.lock\n",
        encoding="utf-8",
    )
    (tmp_path / "source_marker.py").write_text("SOURCE = 'reviewed'\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "paid-review@example.invalid")
    _git(tmp_path, "config", "user.name", "Paid Review Test")
    _git(tmp_path, "add", ".gitignore", "config", "data/manifests/cost_ledger.yaml", "source_marker.py")
    _git(tmp_path, "commit", "-qm", "reviewed source")

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


def _preview(
    project: Path,
    common: list[str],
    *phases: str,
) -> subprocess.CompletedProcess[str]:
    selected = phases or ("behavior_baseline_gpu",)
    phase_arguments = [item for phase in selected for item in ("--allow-phase", phase)]
    return _run(
        project,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        *phase_arguments,
    )


def test_preview_exclusively_freezes_authenticated_private_quote_locks(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)

    completed = _preview(tmp_path, common)

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["status"] == "preview"
    assert output["paid_execution_authorized"] is False
    assert output["ready_for_explicit_user_approval"] is True
    assert output["approval_schema_version"] == 4
    assert output["cumulative_cost_proven"] is True
    assert output["user_review_hash"] == output["approval_review"]["review_hash"]
    assert output["user_review_hash"] == stable_hash(output["approval_review"]["payload"])
    assert output["approval_review"]["payload"]["source_commit"] == _git(
        tmp_path,
        "rev-parse",
        "HEAD",
    ).stdout.strip()
    assert output["approval_review"]["payload"]["ledger"]["path"] == (
        "data/manifests/cost_ledger.yaml"
    )
    assert output["gpu"]["provider_gpu_id"] == "NVIDIA H100 80GB HBM3"
    assert output["gpu"]["cloud_type"] == "SECURE"
    assert output["gpu"]["allowed_cuda_versions"] == ["12.8"]
    assert output["gpu"]["data_center_ids"] == ["US-IL-1"]
    assert output["gpu"]["container_disk_gb"] == 50
    assert output["gpu"]["volume_disk_gb"] == 650
    assert output["planned_command_phases"] == ["behavior_baseline_gpu"]
    assert output["approval_review"]["payload"]["planned_command_phases"] == [
        "behavior_baseline_gpu"
    ]
    assert output["gpu"]["projected_compute_usd"] == 8.0
    assert output["gpu"]["projected_running_storage_usd"] == 0.097222
    assert output["gpu"]["phase_maxima_usd"] == [
        {"command_phase": "behavior_baseline_gpu", "maximum_usd": 8.097223},
        {"command_phase": "behavior_treatment_gpu", "maximum_usd": 8.097223},
        {"command_phase": "resample_gpu", "maximum_usd": 8.097223},
        {"command_phase": "lens_gpu", "maximum_usd": 8.097223},
    ]
    assert output["gpu"]["planned_phase_maxima_usd"] == [
        {"command_phase": "behavior_baseline_gpu", "maximum_usd": 8.097223}
    ]
    assert output["gpu"]["projected_maximum_usd"] == 8.097223
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

    authenticated_again = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert authenticated_again.returncode == 0, authenticated_again.stderr
    assert json.loads(authenticated_again.stdout)["hashes"] == output["hashes"]


def test_preview_proves_cumulative_safety_adjusted_gpu_headroom(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    ledger_path = tmp_path / "data" / "manifests" / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    ledger.append(
        CostEntry(
            kind="storage",
            amount_usd=5,
            description="bounded storage reserve",
            status="estimated",
        )
    )
    ledger.append(
        CostEntry(
            kind="gpu",
            amount_usd=9.246029,
            description="prior authenticated GPU incidents",
            status="incurred",
        )
    )

    completed = _run(
        tmp_path,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 0, completed.stderr
    cumulative = json.loads(completed.stdout)["cumulative_cost"]
    assert cumulative == {
        "ledger_incurred": {
            "gpu": 9.246029,
            "api": 0.0,
            "storage": 0.0,
            "other": 0.0,
            "total": 9.246029,
        },
        "ledger_committed": {
            "gpu": 9.246029,
            "api": 0.0,
            "storage": 5.0,
            "other": 0.0,
            "total": 14.246029,
        },
        "future_gpu_phase_maxima_usd": 8.097223,
        "gpu_worst_case_usd": 17.343252,
        "gpu_safety_margin_fraction": 0.03,
        "gpu_safety_adjusted_ceiling_usd": 213.4,
        "gpu_safety_headroom_usd": 196.056748,
        "gpu_hard_stop_headroom_usd": 202.656748,
        "api_hard_stop_usd": 100.0,
        "total_worst_case_usd": 122.343252,
        "total_hard_stop_headroom_usd": 202.656748,
    }


def test_phase_scoped_refresh_does_not_double_count_completed_gpu_work(
    tmp_path: Path,
) -> None:
    common, _, _ = _prepare_project(tmp_path)
    ledger = CostLedger(
        tmp_path / "data/manifests/cost_ledger.yaml",
        BudgetLimits(gpu=220, api=100, total=325),
    )
    ledger.append(
        CostEntry(
            kind="gpu",
            amount_usd=9.246029,
            description="settled behavior baseline GPU phase",
            status="incurred",
        )
    )

    completed = _preview(tmp_path, common, "behavior_treatment_gpu")

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["planned_command_phases"] == ["behavior_treatment_gpu"]
    assert output["cumulative_cost"]["ledger_incurred"]["gpu"] == 9.246029
    assert output["cumulative_cost"]["future_gpu_phase_maxima_usd"] == 8.097223
    assert output["cumulative_cost"]["gpu_worst_case_usd"] == 17.343252
    assert len(output["gpu"]["phase_maxima_usd"]) == 4
    assert output["gpu"]["planned_phase_maxima_usd"] == [
        {"command_phase": "behavior_treatment_gpu", "maximum_usd": 8.097223}
    ]


def test_phase_scope_is_review_hashed_canonical_and_cannot_be_changed_at_approval(
    tmp_path: Path,
) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    baseline = _preview(tmp_path, common, "behavior_baseline_gpu")
    assert baseline.returncode == 0, baseline.stderr
    baseline_hash = json.loads(baseline.stdout)["user_review_hash"]

    treatment = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_treatment_gpu",
    )
    assert treatment.returncode == 0, treatment.stderr
    treatment_hash = json.loads(treatment.stdout)["user_review_hash"]
    assert treatment_hash != baseline_hash

    changed_scope = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        baseline_hash,
        "--approval-id",
        "yib-approval-changed-phase-scope",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_treatment_gpu",
    )
    assert changed_scope.returncode == 2
    assert "does not match the user-reviewed hash" in changed_scope.stderr
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()


def test_preview_canonicalizes_phase_order_and_rejects_duplicates(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    canonical = _preview(
        tmp_path,
        common,
        "behavior_baseline_gpu",
        "behavior_baseline_api",
    )
    assert canonical.returncode == 0, canonical.stderr
    canonical_hash = json.loads(canonical.stdout)["user_review_hash"]

    reversed_scope = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_api",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert reversed_scope.returncode == 0, reversed_scope.stderr
    assert json.loads(reversed_scope.stdout)["user_review_hash"] == canonical_hash

    duplicate = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert duplicate.returncode == 2
    assert not duplicate.stdout


def test_cumulative_preview_rejects_outstanding_gpu_reservation(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    ledger_path = tmp_path / "data" / "manifests" / "cost_ledger.yaml"
    ledger = CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325))
    ledger.append(
        CostEntry(
            kind="gpu",
            amount_usd=5,
            description="unresolved prior GPU reservation",
            status="estimated",
        )
    )

    completed = _run(
        tmp_path,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 2
    assert "fresh cumulative preview requires no outstanding GPU reservation" in (
        completed.stderr
    )


def test_preview_requires_the_canonical_cost_ledger(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)

    completed = _run(
        tmp_path,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 2
    assert "--cost-ledger" in completed.stderr

    alternate = tmp_path / "data/manifests/alternate_cost_ledger.yaml"
    alternate.write_bytes((tmp_path / "data/manifests/cost_ledger.yaml").read_bytes())
    alternate_preview = _run(
        tmp_path,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--cost-ledger",
        "data/manifests/alternate_cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert alternate_preview.returncode == 2
    assert "canonical cumulative ledger" in alternate_preview.stderr


def test_preview_rejects_duplicate_ledger_keys_even_when_values_match(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    ledger = tmp_path / "data/manifests/cost_ledger.yaml"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "currency: USD\n",
        encoding="utf-8",
    )

    completed = _run(
        tmp_path,
        "preview",
        *common,
        "--gpu-quote-spec",
        ".runpod/specs/gpu_quote_spec.json",
        "--api-quote-spec",
        ".runpod/specs/api_route_quote_spec.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 2
    assert "duplicate key" in completed.stderr


def test_approve_rejects_changed_ledger_and_wrong_review_hash(tmp_path: Path) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    preview = _preview(tmp_path, common)
    assert preview.returncode == 0, preview.stderr
    review_hash = json.loads(preview.stdout)["user_review_hash"]
    ledger = CostLedger(
        tmp_path / "data/manifests/cost_ledger.yaml",
        BudgetLimits(gpu=220, api=100, total=325),
    )
    ledger.append(
        CostEntry(
            kind="storage",
            amount_usd=1,
            description="ledger changed after user review",
            status="estimated",
        )
    )

    changed = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
        "--approval-id",
        "yib-approval-ledger-changed",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert changed.returncode == 2
    assert "does not match the user-reviewed hash" in changed.stderr
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()

    fresh_preview = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert fresh_preview.returncode == 0, fresh_preview.stderr
    wrong = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        stable_hash({"wrong": "review"}),
        "--approval-id",
        "yib-approval-wrong-review",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert wrong.returncode == 2
    assert "does not match the user-reviewed hash" in wrong.stderr
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()


def test_stale_quote_preview_is_not_ready_and_cannot_be_approved(tmp_path: Path) -> None:
    common, _, _ = _prepare_project(tmp_path)
    stale_time = _iso(datetime.now(UTC) - timedelta(hours=7))
    for name, field in (
        ("gpu_quote_spec.json", "quoted_at"),
        ("api_route_quote_spec.json", "checked_at"),
    ):
        path = tmp_path / ".runpod/specs" / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = stale_time
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)

    preview = _preview(tmp_path, common)
    assert preview.returncode == 0, preview.stderr
    output = json.loads(preview.stdout)
    assert output["ready_for_explicit_user_approval"] is False

    approve = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        output["user_review_hash"],
        "--approval-id",
        "yib-approval-stale-quotes",
        "--approved-at",
        _iso(datetime.now(UTC)),
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert approve.returncode == 2
    assert "quotes must be fresh" in approve.stderr
    assert not (tmp_path / ".runpod/paid_run_approval.json").exists()


def test_review_rejects_source_commit_drift_dirty_source_and_ledger_symlink(
    tmp_path: Path,
) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    preview = _preview(tmp_path, common)
    assert preview.returncode == 0, preview.stderr
    review_hash = json.loads(preview.stdout)["user_review_hash"]

    marker = tmp_path / "source_marker.py"
    marker.write_text("SOURCE = 'new commit'\n", encoding="utf-8")
    _git(tmp_path, "add", "source_marker.py")
    _git(tmp_path, "commit", "-qm", "change reviewed source")
    commit_drift = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
        "--approval-id",
        "yib-approval-commit-drift",
        "--approved-at",
        approval_time,
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert commit_drift.returncode == 2
    assert "does not match the user-reviewed hash" in commit_drift.stderr

    marker.write_text("SOURCE = 'dirty'\n", encoding="utf-8")
    dirty = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert dirty.returncode == 2
    assert "tracked project source differs" in dirty.stderr
    _git(tmp_path, "restore", "source_marker.py")

    ledger_path = tmp_path / "data/manifests/cost_ledger.yaml"
    outside = tmp_path / ".runpod/outside-ledger.yaml"
    outside.write_bytes(ledger_path.read_bytes())
    ledger_path.unlink()
    ledger_path.symlink_to(outside)
    symlinked = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert symlinked.returncode == 2
    assert "canonical cost ledger is invalid" in symlinked.stderr


def test_approve_requires_explicit_metadata_and_binds_full_gpu_lock(tmp_path: Path) -> None:
    common, _, approval_time = _prepare_project(tmp_path)
    preview = _preview(
        tmp_path,
        common,
        "behavior_baseline_gpu",
        "behavior_baseline_api",
    )
    assert preview.returncode == 0
    review_hash = json.loads(preview.stdout)["user_review_hash"]
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
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
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
    assert output["user_review_hash"] == review_hash
    assert approval.review.review_hash == review_hash
    assert approval_id not in completed.stdout
    assert stat.S_IMODE(approval_path.stat().st_mode) == 0o600

    original = approval_path.read_bytes()
    overwrite = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
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
    preview = _preview(tmp_path, common)
    assert preview.returncode == 0
    review_hash = json.loads(preview.stdout)["user_review_hash"]
    predates_quote = _iso(
        datetime.fromisoformat(quote_time.replace("Z", "+00:00")) - timedelta(seconds=1)
    )

    completed = _run(
        tmp_path,
        "approve",
        *common,
        "--output",
        ".runpod/paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
        "--approval-id",
        "yib-approval-20260829-003",
        "--approved-at",
        predates_quote,
        "--allow-phase",
        "behavior_baseline_gpu",
    )

    assert completed.returncode == 2
    assert "predates the reviewed provider quotes" in completed.stderr
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

    symlinked = _run(
        tmp_path,
        "preview",
        *common,
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--allow-phase",
        "behavior_baseline_gpu",
    )
    assert symlinked.returncode == 2
    assert "owned real directory" in symlinked.stderr
    assert not any(outside.iterdir())

    second_project = tmp_path / "second"
    second_project.mkdir()
    common, _, approval_time = _prepare_project(second_project)
    preview = _preview(second_project, common)
    assert preview.returncode == 0
    review_hash = json.loads(preview.stdout)["user_review_hash"]
    escaped = _run(
        second_project,
        "approve",
        *common,
        "--output",
        "paid_run_approval.json",
        "--cost-ledger",
        "data/manifests/cost_ledger.yaml",
        "--review-hash",
        review_hash,
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
