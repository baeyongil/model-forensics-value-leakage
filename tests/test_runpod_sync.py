from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import model_forensics.runpod_sync as sync_module
from model_forensics.approval import (
    APPROVAL_SCHEMA_VERSION,
    PAID_RUN_REVIEW_PROTOCOL_VERSION,
    PHASE_CONTRACT_VERSION,
    ApiQuoteBinding,
    ApprovalBindings,
    GpuBinding,
    GpuPhaseRuntimeAllocation,
    GpuQuote,
    PaidRunApproval,
    PaidRunReview,
    PaidRunReviewPayload,
    RouteBinding,
    SpendingCaps,
    UserApproval,
    approval_content_hash,
    paid_run_review_hash,
)
from model_forensics.budget import BudgetLimits, CostEntry, CostLedger
from model_forensics.execution_bindings import (
    api_route_quote_lock_content_hash,
    gpu_quote_lock_content_hash,
)
from model_forensics.gpu_budget import (
    GPU_PHASE_SETTLEMENT_PROTOCOL,
    approved_gpu_phase_maximum_usd,
    reserve_gpu_phase_budget,
    settle_gpu_phase_budget,
)
from model_forensics.io import stable_hash, write_json
from model_forensics.runpod_contract import LIFECYCLE_PROTOCOL
from model_forensics.runpod_sessions import LEGACY_SETTLEMENT_V1_FILENAME
from model_forensics.runpod_sync import (
    SOURCE_REPOSITORY_URL,
    RunpodSyncError,
    build_selective_sync_plan,
    materialize_selective_sync_bundle,
    revalidate_selective_sync_plan,
)
from model_forensics.runpod_watchdog import (
    PodMetadata,
    WatchdogLimits,
    _host_ack_payload,
    _state,
    derive_deadline,
)
from model_forensics.settlement_upgrade import upgrade_legacy_gpu_settlement

_REAL_VALIDATED_SOURCE_COMMIT = sync_module._validated_source_commit


@pytest.fixture(autouse=True)
def _authenticated_source_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most sync fixtures exercise state transfer, not git plumbing."""

    monkeypatch.setattr(
        sync_module,
        "_validated_source_commit",
        lambda _root: "a" * 40,
    )


def _git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_commit_requires_exact_clean_runner_checkout(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "sync-test@example.invalid", cwd=tmp_path)
    _git("config", "user.name", "Sync Test", cwd=tmp_path)
    _git("remote", "add", "origin", SOURCE_REPOSITORY_URL, cwd=tmp_path)
    (tmp_path / "src").mkdir()
    runner = tmp_path / "src" / "runner.py"
    runner.write_text("print('pinned')\n", encoding="utf-8")
    _git("add", "src/runner.py", cwd=tmp_path)
    _git("commit", "-m", "fixture", cwd=tmp_path)

    expected = _git("rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    assert _REAL_VALIDATED_SOURCE_COMMIT(tmp_path.resolve()) == expected

    _git(
        "remote",
        "set-url",
        "origin",
        "https://example.invalid/untrusted.git",
        cwd=tmp_path,
    )
    with pytest.raises(RunpodSyncError, match="canonical public repository"):
        _REAL_VALIDATED_SOURCE_COMMIT(tmp_path.resolve())
    _git("remote", "set-url", "origin", SOURCE_REPOSITORY_URL, cwd=tmp_path)

    runner.write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(RunpodSyncError, match="runner source is dirty"):
        _REAL_VALIDATED_SOURCE_COMMIT(tmp_path.resolve())


def test_source_commit_rejects_untracked_runner_and_nested_root(tmp_path: Path) -> None:
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "sync-test@example.invalid", cwd=tmp_path)
    _git("config", "user.name", "Sync Test", cwd=tmp_path)
    _git("remote", "add", "origin", SOURCE_REPOSITORY_URL, cwd=tmp_path)
    (tmp_path / "src").mkdir()
    tracked = tmp_path / "src" / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "src/tracked.py", cwd=tmp_path)
    _git("commit", "-m", "fixture", cwd=tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "untracked.py").write_text(
        "print('untracked')\n",
        encoding="utf-8",
    )

    with pytest.raises(RunpodSyncError, match="runner source is dirty"):
        _REAL_VALIDATED_SOURCE_COMMIT(tmp_path.resolve())
    with pytest.raises(RunpodSyncError, match="git checkout root"):
        _REAL_VALIDATED_SOURCE_COMMIT((tmp_path / "src").resolve())


def _external_receipt(
    *,
    reservation: dict[str, object],
    amount: float,
) -> dict[str, object]:
    pod_id_hash = stable_hash({"runpod_pod_id": "prior-private-pod"})
    stop_evidence = {
        "desired_status": "EXITED",
        "environment_verified": True,
        "started_at": "2026-08-29T23:00:00Z",
        "exited_at": "2026-08-29T23:04:00Z",
        "runtime_ms": 240_000,
    }
    billing_query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": stop_evidence["started_at"],
        "end_time": stop_evidence["exited_at"],
    }
    billing_evidence = {
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "pod_id_hash": pod_id_hash,
        "provider_amount_usd": None,
        "settlement_amount_usd": amount,
        "time_billed_ms": None,
        "billing_bucket_time": None,
        "provider_billing_row_hash": None,
        "conservative_ceiling_usd": amount,
        "runtime_ceiling_minutes": 4,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-external-stop-v1",
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-29T23:10:00Z",
        "prior_lifecycle_operation": "rearmed",
        "lifecycle_before_hash": stable_hash({"lifecycle": "before"}),
        "lifecycle_stopped_hash": stable_hash({"lifecycle": "stopped"}),
        "session_hash": reservation["session_hash"],
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "pod_id_hash": pod_id_hash,
        "stop_evidence": stop_evidence,
        "stop_evidence_hash": stable_hash(stop_evidence),
        "billing_query": billing_query,
        "billing_query_hash": stable_hash(billing_query),
        "billing_evidence": billing_evidence,
        "billing_evidence_hash": stable_hash(billing_evidence),
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "settlement_amount_usd": amount,
        "source_artifact_hashes": [],
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _add_upgraded_prior_session(
    *,
    root: Path,
    ledger: CostLedger,
) -> str:
    amount = 1.25
    prior = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="prior_completed_gpu",
        session_id="selective-sync-prior-session",
        approved_phase_maximum_usd=2,
        approved_maximum_runtime_hours=1,
        live_hourly_total_usd=2,
    )
    reservation_path = root / ".runpod" / "reservations" / f"{prior.phase}.json"
    write_json(reservation_path, prior.manifest())
    settle_gpu_phase_budget(
        ledger=ledger,
        reservation=prior,
        incurred_usd=amount,
    )

    digest = prior.session_hash.removeprefix("sha256:")
    session = root / ".runpod" / "sessions" / digest
    session.mkdir(parents=True)
    watchdog = {
        "schema_version": 2,
        "watchdog_version": "runpod-gpu-cost-watchdog-v2",
        "status": "stopped_confirmed",
        "stop_reason": "provider_exit_verified",
    }
    watchdog_path = session / "runpod_watchdog.json"
    write_json(watchdog_path, watchdog)
    legacy: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": GPU_PHASE_SETTLEMENT_PROTOCOL,
        "phase": prior.phase,
        "reservation_id": prior.reservation_id,
        "reservation_record_hash": prior.manifest()["record_hash"],
        "session_hash": prior.session_hash,
        "provider_incurred_usd": amount,
        "watchdog_state_hash": stable_hash(watchdog),
        "status": "settled",
    }
    legacy["record_hash"] = stable_hash(legacy)
    settlement_path = session / "settlement.json"
    write_json(settlement_path, legacy)
    external_path = session / "external_stop_receipt.json"
    write_json(
        external_path,
        _external_receipt(reservation=prior.manifest(), amount=amount),
    )
    upgrade_legacy_gpu_settlement(
        project_root=root,
        reservation_receipt_path=reservation_path,
        cost_ledger_path=ledger.path,
        watchdog_state_path=watchdog_path,
        external_stop_receipt_path=external_path,
        settlement_path=settlement_path,
        limits=BudgetLimits(gpu=220, api=100, total=325),
    )
    return digest


def _add_no_start_prior_session(*, root: Path, ledger: CostLedger) -> str:
    prior = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="prior_no_start_gpu",
        session_id="selective-sync-no-start-session",
        approved_phase_maximum_usd=2,
        approved_maximum_runtime_hours=1,
        live_hourly_total_usd=2,
    )
    reservation_path = root / ".runpod" / "reservations" / f"{prior.phase}.json"
    write_json(reservation_path, prior.manifest())
    settle_gpu_phase_budget(ledger=ledger, reservation=prior, incurred_usd=0)
    digest = prior.session_hash.removeprefix("sha256:")
    session = root / ".runpod" / "sessions" / digest
    session.mkdir(parents=True)
    pod_id_hash = stable_hash({"runpod_pod_id": "no-start-private-pod"})
    baseline = "2026-08-29T21:00:00Z"
    provider = {
        "desired_status": "EXITED",
        "pod_id_hash": pod_id_hash,
        "name_hash": stable_hash({"runpod_pod_name": "prior-pod"}),
        "image_hash": stable_hash({"runpod_image": "prior-image"}),
        "machine_id_hash": stable_hash({"runpod_machine_id": "prior-machine"}),
        "provider_binding_hash": stable_hash({"provider": "binding"}),
        "immutable_spec_hash": stable_hash({"immutable": "spec"}),
        "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 8},
        "cloud": "SECURE",
        "data_center_id": "US-IL-1",
        "container_disk_gb": 50,
        "persistent_disk_gb": 650,
        "persistent_mount_path": "/workspace",
        "ports": ["22/tcp"],
        "environment_verified": True,
        "environment_session_context": "current",
        "pre_start_last_started_at": baseline,
        "observed_last_started_at": baseline,
        "last_started_at_unchanged": True,
        "provider_hourly_compute_usd": 2.0,
        "approved_hourly_all_in_usd": 2.0,
        "observation_count": 1,
        "quiet_window_seconds": 0.0,
        "first_observation_hash": stable_hash({"observation": "first"}),
        "second_observation_hash": None,
    }
    query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": "2026-08-29T22:00:00Z",
        "end_time": "2026-08-29T22:05:00Z",
    }
    billing = {"row_count": 0, "response_hash": stable_hash([])}
    receipt: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": "runpod-no-start-v1",
        "status": "no_start_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-29T22:05:00Z",
        "prior_lifecycle_operation": "rearm_patched",
        "lifecycle_before_hash": stable_hash({"lifecycle": "before-no-start"}),
        "lifecycle_stopped_hash": stable_hash({"lifecycle": "stopped-no-start"}),
        "session_hash": prior.session_hash,
        "reservation_id": prior.reservation_id,
        "reservation_record_hash": prior.manifest()["record_hash"],
        "pod_id_hash": pod_id_hash,
        "provider_evidence": provider,
        "provider_evidence_hash": stable_hash(provider),
        "billing_query": query,
        "billing_query_hash": stable_hash(query),
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "accounted_gpu_usd": 0.0,
    }
    receipt["record_hash"] = stable_hash(receipt)
    write_json(session / "no_start_receipt.json", receipt)
    settlement: dict[str, object] = {
        "schema_version": 2,
        "protocol_version": "cumulative-gpu-phase-settlement-v2",
        "phase": prior.phase,
        "reservation_id": prior.reservation_id,
        "reservation_record_hash": prior.manifest()["record_hash"],
        "session_hash": prior.session_hash,
        "provider_incurred_usd": 0.0,
        "accounted_gpu_usd": 0.0,
        "billing_status": "not_started",
        "evidence_kind": "provider_no_start",
        "no_start_receipt_hash": receipt["record_hash"],
        "provider_evidence_hash": receipt["provider_evidence_hash"],
        "billing_evidence_hash": receipt["billing_evidence_hash"],
        "status": "settled",
    }
    settlement["record_hash"] = stable_hash(settlement)
    write_json(session / "settlement.json", settlement)
    return digest


def _write_control_fixture(*, root: Path, observed_at: datetime) -> dict[str, str]:
    private = root / ".runpod"
    image = "runpod/pytorch@sha256:" + "ab" * 32
    gpu_quote: dict[str, object] = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "runpod-secure-h100-sync-fixture",
        "gpu_family": "H100_80GB",
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["US-IL-1"],
        "gpu_count": 8,
        "container_disk_gb": 50,
        "volume_disk_gb": 650,
        "usd_per_gpu_hour": 1.2375,
        "running_storage_usd_per_hour": 0.1,
        "quoted_at": (observed_at - timedelta(minutes=10)).isoformat(),
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
    gpu_quote["content_hash"] = gpu_quote_lock_content_hash(gpu_quote)
    write_json(private / "gpu_quote_lock.json", gpu_quote)

    route_values = (
        (
            "primary_final_and_trajectory",
            "anthropic/claude-opus-5",
            5.0,
            25.0,
        ),
        (
            "independent_final",
            "google/gemini-3.1-pro-preview",
            2.0,
            12.0,
        ),
        ("classifier_anthropic", "anthropic/claude-opus-5", 5.0, 25.0),
        (
            "classifier_google",
            "google/gemini-3.1-pro-preview",
            2.0,
            12.0,
        ),
    )
    api_quote: dict[str, object] = {
        "schema_version": 1,
        "provider": "openrouter",
        "source_url": "https://openrouter.ai/models",
        "checked_at": (observed_at - timedelta(minutes=10)).isoformat(),
        "routes": [
            {
                "role": role,
                "model": model,
                "input_usd_per_million_tokens": input_price,
                "output_usd_per_million_tokens": output_price,
            }
            for role, model, input_price, output_price in route_values
        ],
    }
    api_quote["content_hash"] = api_route_quote_lock_content_hash(api_quote)
    write_json(private / "api_route_quote_lock.json", api_quote)

    gpu_lock = {"fixture": "authenticated-gpu-lock"}
    (root / "config").mkdir()
    (root / "config" / "gpu_lock.yaml").write_text(
        yaml.safe_dump(gpu_lock, sort_keys=False),
        encoding="utf-8",
    )
    allocations = tuple(
        GpuPhaseRuntimeAllocation(
            command_phase=item["command_phase"],
            maximum_runtime_hours=item["maximum_runtime_hours"],
        )
        for item in gpu_quote["phase_runtime_allocations"]  # type: ignore[union-attr]
    )
    bindings = ApprovalBindings(
        phase_contract_version=PHASE_CONTRACT_VERSION,
        config_hash=stable_hash({"config": "fixture"}),
        preregistration_hash=stable_hash({"preregistration": "fixture"}),
        gpu_lock_hash=stable_hash(gpu_lock),
        gpu=GpuBinding(
            family="H100_80GB",
            provider_gpu_id="NVIDIA H100 80GB HBM3",
            cloud_type="SECURE",
            allowed_cuda_versions=("12.8",),
            data_center_ids=("US-IL-1",),
            count=8,
            container_disk_gb=50,
            volume_disk_gb=650,
            quote=GpuQuote(
                provider="runpod",
                quote_id="runpod-secure-h100-sync-fixture",
                usd_per_gpu_hour=1.2375,
                running_storage_usd_per_hour=0.1,
                quoted_at=observed_at - timedelta(minutes=10),
                source_url="https://www.runpod.io/pricing",
                content_hash=str(gpu_quote["content_hash"]),
            ),
            phase_runtime_allocations=allocations,
            container_image_digest=image,
            vllm_wheel_sha256=stable_hash({"wheel": "fixture"}).removeprefix(
                "sha256:"
            ),
        ),
        api_quote=ApiQuoteBinding(
            provider="openrouter",
            source_url="https://openrouter.ai/models",
            checked_at=observed_at - timedelta(minutes=10),
            content_hash=str(api_quote["content_hash"]),
        ),
        caps_usd=SpendingCaps(gpu=220.0, api=100.0, total=325.0),
        routes=tuple(
            RouteBinding(
                role=role,
                provider="openrouter",
                model=model,
                input_usd_per_million_tokens=input_price,
                output_usd_per_million_tokens=output_price,
            )
            for role, model, input_price, output_price in route_values
        ),
    )
    phase_maxima = [
        {
            "command_phase": allocation.command_phase,
            "maximum_usd": approved_gpu_phase_maximum_usd(
                gpu_count=bindings.gpu.count,
                quote_hourly_per_gpu_usd=bindings.gpu.quote.usd_per_gpu_hour,
                running_storage_hourly_usd=bindings.gpu.quote.running_storage_usd_per_hour,
                approved_runtime_hours=allocation.maximum_runtime_hours,
            ),
        }
        for allocation in bindings.gpu.phase_runtime_allocations
    ]
    future_gpu = round(sum(item["maximum_usd"] for item in phase_maxima), 6)
    review_payload = PaidRunReviewPayload.model_validate(
        {
            "protocol_version": PAID_RUN_REVIEW_PROTOCOL_VERSION,
            "source_commit": "a" * 40,
            "context_hashes": {
                "config": bindings.config_hash,
                "preregistration": bindings.preregistration_hash,
                "gpu_lock": bindings.gpu_lock_hash,
                "gpu_quote_lock": bindings.gpu.quote.content_hash,
                "api_quote_lock": bindings.api_quote.content_hash,
                "bindings": stable_hash(bindings.model_dump(mode="json")),
            },
            "ledger": {
                "path": "data/manifests/cost_ledger.yaml",
                "bytes_sha256": stable_hash({"ledger": "bytes"}),
                "document_hash": stable_hash({"ledger": "document"}),
                "byte_count": 123,
            },
            "planned_command_phases": [
                "behavior_baseline_gpu",
                "behavior_treatment_gpu",
                "resample_gpu",
                "lens_gpu",
            ],
            "phase_maxima_usd": phase_maxima,
            "caps_usd": bindings.caps_usd.model_dump(mode="json"),
            "cumulative_cost": {
                "ledger_incurred": {
                    "gpu": 0.0,
                    "api": 0.0,
                    "storage": 0.0,
                    "other": 0.0,
                    "total": 0.0,
                },
                "ledger_committed": {
                    "gpu": 0.0,
                    "api": 0.0,
                    "storage": 0.0,
                    "other": 0.0,
                    "total": 0.0,
                },
                "future_gpu_phase_maxima_usd": future_gpu,
                "gpu_worst_case_usd": future_gpu,
                "gpu_safety_margin_fraction": 0.03,
                "gpu_safety_adjusted_ceiling_usd": 213.4,
                "gpu_safety_headroom_usd": round(213.4 - future_gpu, 6),
                "gpu_hard_stop_headroom_usd": round(220.0 - future_gpu, 6),
                "api_hard_stop_usd": 100.0,
                "total_worst_case_usd": round(future_gpu + 100.0, 6),
                "total_hard_stop_headroom_usd": round(225.0 - future_gpu, 6),
            },
        }
    )
    review = PaidRunReview(
        payload=review_payload,
        review_hash=paid_run_review_hash(review_payload),
    )
    unsigned_approval = PaidRunApproval(
        schema_version=APPROVAL_SCHEMA_VERSION,
        bindings=bindings,
        review=review,
        allowed_command_phases=(
            "behavior_baseline_gpu",
            "behavior_treatment_gpu",
            "resample_gpu",
            "lens_gpu",
        ),
        user_approval=UserApproval(
            approval_id="approval-sync-fixture-20260829",
            approved_at=observed_at - timedelta(minutes=5),
        ),
        content_hash=stable_hash({"temporary": "approval"}),
    )
    approval = unsigned_approval.model_copy(
        update={"content_hash": approval_content_hash(unsigned_approval)},
    )
    write_json(private / "paid_run_approval.json", approval.model_dump(mode="json"))
    return {
        "approval_hash": approval.content_hash,
        "bindings_hash": stable_hash(bindings.model_dump(mode="json")),
        "gpu_lock_hash": stable_hash(gpu_lock),
        "quote_hash": str(gpu_quote["content_hash"]),
        "image": image,
    }


def _project(
    tmp_path: Path,
    *,
    upgraded_prior: bool = False,
    no_start_prior: bool = False,
    lifecycle_operation: str = "rearmed",
) -> tuple[str, Path, Path, str]:
    phase = "behavior_treatment_gpu"
    private = tmp_path / ".runpod"
    private.mkdir()
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    ledger_path = tmp_path / "data" / "manifests" / "cost_ledger.yaml"
    ledger = CostLedger(
        ledger_path,
        BudgetLimits(gpu=220, api=100, total=325),
    )
    if upgraded_prior:
        _add_upgraded_prior_session(root=tmp_path, ledger=ledger)
    if no_start_prior:
        _add_no_start_prior_session(root=tmp_path, ledger=ledger)
    observed_at = datetime.now(UTC).replace(microsecond=0)
    controls = _write_control_fixture(root=tmp_path, observed_at=observed_at)
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase=phase,
        session_id="selective-sync-current-session",
        approved_phase_maximum_usd=10,
        approved_maximum_runtime_hours=1,
        live_hourly_total_usd=10,
    )
    reservation_path = private / "reservations" / f"{phase}.json"
    write_json(reservation_path, reservation.manifest())

    spec = {
        "image": controls["image"],
        "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 8},
    }
    authorization = {
        "phase": phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
        "approval_hash": controls["approval_hash"],
        "bindings_hash": controls["bindings_hash"],
        "gpu_lock_hash": controls["gpu_lock_hash"],
        "quote_hash": controls["quote_hash"],
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
        "operation": lifecycle_operation,
        "updated_at": "2026-08-30T00:00:00Z",
        "immutable_spec": spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"id": "pod-fixture", "status": "RUNNING"},
    }
    lifecycle["record_hash"] = stable_hash(lifecycle)
    write_json(private / "pod_lifecycle.json", lifecycle)
    current_digest = reservation.session_hash.removeprefix("sha256:")
    current = private / "sessions" / current_digest
    current.mkdir(parents=True)
    acknowledged_at = observed_at
    armed_at = acknowledged_at + timedelta(seconds=1)
    acknowledgement = _host_ack_payload(
        expected_session_hash=reservation.session_hash,
        expected_phase=phase,
        lifecycle_before_hash=stable_hash({"stopped": "lifecycle"}),
        pod_id="pod-fixture",
        watcher_pid=os.getpid(),
        acknowledged_at=acknowledged_at,
    )
    write_json(current / "host_rearm_watchdog_ack.json", acknowledgement)
    limits = WatchdogLimits(
        **reservation.watchdog_budget_kwargs(),
        maximum_approved_hourly_total_usd=10.0,
        maximum_approved_compute_hourly_usd=9.9,
        maximum_approved_storage_hourly_usd=0.1,
    )
    metadata = PodMetadata(
        pod_id="pod-fixture",
        pod_name="model-forensics-sync-fixture",
        gpu_count=8,
        provider_gpu_id="NVIDIA H100 80GB HBM3",
        gpu_display_name="NVIDIA H100 80GB HBM3",
        runtime_gpu_count=None,
        machine_id_hash=stable_hash({"machine": "fixture"}),
        execution_identity_hash=stable_hash({"execution": "fixture"}),
        data_center_id="US-IL-1",
        cuda_version=None,
        secure_cloud=True,
        container_image=controls["image"],
        container_disk_gb=50,
        persistent_volume_disk_gb=650,
        persistent_volume_mount_path="/workspace",
        ports=("22/tcp",),
        global_networking_enabled=None,
        ssh_ready=True,
        direct_ssh_ready=True,
        direct_ssh_endpoint_hash=stable_hash({"ssh": "fixture"}),
        environment_verified=True,
        desired_status="RUNNING",
        cost_per_hr=9.9,
        adjusted_cost_per_hr=9.9,
        last_started_at=armed_at,
        observed_at=armed_at,
        locked=None,
        interruptible=None,
        network_volume_attached=False,
    )
    derived = derive_deadline(
        metadata,
        limits,
        calculation_hourly_usd=10.0,
    )
    write_json(
        current / "host_rearm_watchdog.json",
        _state(
            pod_id="pod-fixture",
            limits=limits,
            status="armed",
            armed_at=armed_at,
            metadata=metadata,
            derived=derived,
            now=armed_at,
        ),
    )
    return phase, reservation_path, ledger_path, current_digest


def test_selective_bundle_excludes_current_host_claim_and_keeps_required_controls(
    tmp_path: Path,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    paths = {item["path"] for item in plan["files"]}

    assert plan["current_host_session_excluded"] is True
    assert "watchdog_state_file_hash" not in plan["current_host_guard"]
    assert set(plan["current_host_guard"]) == {
        "acknowledgement_file_hash",
        "acknowledgement_record_hash",
        "watcher_process_identity_hash",
        "watchdog_invariant_hash",
        "direct_ssh_endpoint_hash",
    }
    assert f".runpod/reservations/{phase}.json" in paths
    assert ".runpod/pod_lifecycle.json" in paths
    assert "data/manifests/cost_ledger.yaml" in paths
    assert ".runpod/gpu_quote_lock.json" in paths
    assert not any(f".runpod/sessions/{current_digest}/" in path for path in paths)

    destination = tmp_path / ".runpod" / "sync_bundles" / current_digest
    materialize_selective_sync_bundle(
        project_root=tmp_path,
        destination=destination,
        plan=plan,
    )
    assert (destination / ".runpod" / "pod_lifecycle.json").is_file()
    assert (destination / "data" / "manifests" / "cost_ledger.yaml").is_file()
    assert not (destination / ".runpod" / "sessions" / current_digest).exists()
    manifest = json.loads(
        (destination / ".runpod" / "selective_sync_manifest.json").read_text()
    )
    assert manifest["record_hash"] == plan["record_hash"]


def test_first_created_pod_cannot_bypass_missing_guard_producer(
    tmp_path: Path,
) -> None:
    phase, reservation, ledger, _current_digest = _project(
        tmp_path,
        lifecycle_operation="created",
    )

    with pytest.raises(RunpodSyncError, match="authenticated re-armed Pod"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


def test_host_guard_revalidation_accepts_a_new_semantically_valid_heartbeat(
    tmp_path: Path,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    state_path = (
        tmp_path
        / ".runpod"
        / "sessions"
        / current_digest
        / "host_rearm_watchdog.json"
    )
    before_hash = sync_module._stable_file_record(
        state_path,
        label="test host heartbeat",
    )["sha256"]
    heartbeat = json.loads(state_path.read_text(encoding="utf-8"))
    armed_at = datetime.fromisoformat(str(heartbeat["armed_at"]))
    heartbeat_at = armed_at + timedelta(seconds=2)
    heartbeat["updated_at"] = heartbeat_at.isoformat()
    heartbeat["live_metadata"]["observed_at"] = heartbeat_at.isoformat()
    effective_deadline = datetime.fromisoformat(
        str(heartbeat["deadline"]["effective_deadline"])
    )
    heartbeat["deadline"]["remaining_seconds"] = round(
        (effective_deadline - heartbeat_at).total_seconds(),
        3,
    )
    heartbeat["deadline"]["incurred_cost_usd"] = round(10.0 / 60.0, 6)
    write_json(state_path, heartbeat)
    after_hash = sync_module._stable_file_record(
        state_path,
        label="test host heartbeat",
    )["sha256"]
    assert after_hash != before_hash

    revalidate_selective_sync_plan(project_root=tmp_path, plan=plan)


@pytest.mark.parametrize(
    ("artifact", "mutation", "match"),
    [
        (
            "host_rearm_watchdog_ack.json",
            lambda payload: payload.update(
                watcher_process_identity_hash="sha256:" + "f" * 64,
                record_hash=stable_hash(
                    {
                        **{
                            key: value
                            for key, value in payload.items()
                            if key not in {"record_hash", "watcher_process_identity_hash"}
                        },
                        "watcher_process_identity_hash": "sha256:" + "f" * 64,
                    }
                ),
            ),
            "not live and authenticated",
        ),
        (
            "host_rearm_watchdog.json",
            lambda payload: payload["live_metadata"].update(
                direct_ssh_endpoint_hash="sha256:" + "e" * 64
            ),
            "host guard changed after planning",
        ),
    ],
)
def test_host_guard_revalidation_rejects_process_or_invariant_drift(
    tmp_path: Path,
    artifact: str,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    path = tmp_path / ".runpod" / "sessions" / current_digest / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    write_json(path, payload)

    with pytest.raises(RunpodSyncError, match=match):
        revalidate_selective_sync_plan(project_root=tmp_path, plan=plan)


def test_host_guard_revalidation_rejects_running_pod_drift(tmp_path: Path) -> None:
    phase, reservation, ledger, _current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    lifecycle_path = tmp_path / ".runpod" / "pod_lifecycle.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["pod"]["id"] = "different-pod"
    lifecycle["record_hash"] = stable_hash(
        {key: value for key, value in lifecycle.items() if key != "record_hash"}
    )
    write_json(lifecycle_path, lifecycle)

    with pytest.raises(RunpodSyncError, match="lifecycle changed after planning"):
        revalidate_selective_sync_plan(project_root=tmp_path, plan=plan)


def test_current_host_session_remote_claim_artifact_blocks_sync(tmp_path: Path) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    current = tmp_path / ".runpod" / "sessions" / current_digest
    write_json(current / "gpu_budget_bootstrap.json", {"claimed": True})

    with pytest.raises(RunpodSyncError, match="exactly its acknowledgement and state"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


def test_pending_stop_request_blocks_selective_sync(tmp_path: Path) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    current = tmp_path / ".runpod" / "sessions" / current_digest
    (current / "runpod_stop.request").write_text("stop\n", encoding="utf-8")

    with pytest.raises(RunpodSyncError, match="pending stop request"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


@pytest.mark.parametrize(
    ("artifact", "mutation", "match"),
    [
        (
            "host_rearm_watchdog_ack.json",
            lambda payload: payload.update(expected_phase="wrong_phase"),
            "not live and authenticated",
        ),
        (
            "host_rearm_watchdog.json",
            lambda payload: payload.update(status="stopped_confirmed"),
            "not safely armed",
        ),
    ],
)
def test_tampered_or_terminal_host_guard_blocks_sync(
    tmp_path: Path,
    artifact: str,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    path = tmp_path / ".runpod" / "sessions" / current_digest / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    write_json(path, payload)

    with pytest.raises(RunpodSyncError, match=match):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload.update(
                updated_at=(datetime.now(UTC) - timedelta(seconds=30)).isoformat()
            ),
            "stale or future-dated",
        ),
        (
            lambda payload: payload["limits"].update(  # type: ignore[union-attr]
                gpu_hard_stop_usd=221.0
            ),
            "limits disagree with reservation",
        ),
        (
            lambda payload: payload["deadline"].update(  # type: ignore[union-attr]
                effective_deadline=(datetime.now(UTC) + timedelta(hours=3)).isoformat()
            ),
            "deadline is absent or elapsed",
        ),
        (
            lambda payload: payload["live_metadata"].update(  # type: ignore[union-attr]
                container_image="runpod/unapproved@sha256:" + "f" * 64
            ),
            "live Pod binding is incomplete",
        ),
    ],
)
def test_current_host_guard_rejects_stale_or_forged_state(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    state_path = (
        tmp_path
        / ".runpod"
        / "sessions"
        / current_digest
        / "host_rearm_watchdog.json"
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    mutation(payload)
    write_json(state_path, payload)

    with pytest.raises(RunpodSyncError, match=match):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


def test_incomplete_prior_session_evidence_blocks_selective_sync(tmp_path: Path) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    prior_digest = "a" * 64
    assert prior_digest != current_digest
    prior = tmp_path / ".runpod" / "sessions" / prior_digest
    prior.mkdir()
    write_json(prior / "settlement.json", {"incomplete": True})

    with pytest.raises(RunpodSyncError, match="prior session gpu_budget_bootstrap"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


def test_actual_upgraded_prior_session_syncs_legacy_provenance_exactly(
    tmp_path: Path,
) -> None:
    phase, reservation, ledger, current_digest = _project(
        tmp_path,
        upgraded_prior=True,
    )
    prior_digests = {
        item.name
        for item in (tmp_path / ".runpod" / "sessions").iterdir()
        if item.name != current_digest
    }
    assert len(prior_digests) == 1
    prior_digest = prior_digests.pop()

    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    paths = {item["path"] for item in plan["files"]}
    prior_prefix = f".runpod/sessions/{prior_digest}"
    assert f"{prior_prefix}/external_stop_receipt.json" in paths
    assert f"{prior_prefix}/settlement.json" in paths
    assert f"{prior_prefix}/{LEGACY_SETTLEMENT_V1_FILENAME}" in paths
    assert f"{prior_prefix}/runpod_watchdog.json" in paths
    assert not any(f".runpod/sessions/{current_digest}/" in path for path in paths)

    destination = tmp_path / ".runpod" / "sync_bundles" / current_digest
    materialize_selective_sync_bundle(
        project_root=tmp_path,
        destination=destination,
        plan=plan,
    )
    for name in (
        "external_stop_receipt.json",
        "settlement.json",
        LEGACY_SETTLEMENT_V1_FILENAME,
        "runpod_watchdog.json",
    ):
        assert (destination / prior_prefix / name).is_file()
    assert not (destination / ".runpod" / "sessions" / current_digest).exists()


def test_authenticated_no_start_prior_syncs_only_zero_settlement_evidence(
    tmp_path: Path,
) -> None:
    phase, reservation, ledger, current_digest = _project(
        tmp_path,
        no_start_prior=True,
    )
    prior = next(
        item
        for item in (tmp_path / ".runpod" / "sessions").iterdir()
        if item.name != current_digest
    )
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    paths = {item["path"] for item in plan["files"]}
    prefix = f".runpod/sessions/{prior.name}"
    assert {path for path in paths if path.startswith(prefix)} == {
        f"{prefix}/no_start_receipt.json",
        f"{prefix}/settlement.json",
    }
    assert not any(f".runpod/sessions/{current_digest}/" in path for path in paths)


def test_orphaned_incurred_gpu_ledger_entry_blocks_sync(tmp_path: Path) -> None:
    phase, reservation, ledger_path, _current_digest = _project(tmp_path)
    CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325)).append(
        CostEntry(
            kind="gpu",
            amount_usd=1,
            description="orphaned prior GPU cost with no completed session evidence",
            status="incurred",
        )
    )

    with pytest.raises(RunpodSyncError, match="exactly cover incurred GPU ledger"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger_path,
        )


def test_second_active_gpu_commitment_blocks_sync(tmp_path: Path) -> None:
    phase, reservation, ledger_path, _current_digest = _project(tmp_path)
    CostLedger(ledger_path, BudgetLimits(gpu=220, api=100, total=325)).append(
        CostEntry(
            kind="gpu",
            amount_usd=1,
            description="unapproved second active GPU commitment",
            status="estimated",
        )
    )

    with pytest.raises(RunpodSyncError, match="sole active GPU commitment"):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger_path,
        )


@pytest.mark.parametrize("source_kind", ["lifecycle", "reservation", "ledger"])
def test_semantic_source_mutation_during_plan_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    phase, reservation, ledger, _current_digest = _project(tmp_path)
    lifecycle = tmp_path / ".runpod" / "pod_lifecycle.json"
    if source_kind == "lifecycle":
        original = sync_module.load_lifecycle_state

        def load_then_mutate(path: str | Path) -> dict[str, object]:
            result = original(path)
            Path(path).write_text('{"tampered":true}\n', encoding="utf-8")
            return result

        monkeypatch.setattr(sync_module, "load_lifecycle_state", load_then_mutate)
        match = "lifecycle state changed across semantic validation"
    elif source_kind == "reservation":
        original = sync_module.load_gpu_phase_budget_reservation

        def load_then_mutate(path: str | Path) -> object:
            result = original(path)
            Path(path).write_text('{"tampered":true}\n', encoding="utf-8")
            return result

        monkeypatch.setattr(
            sync_module,
            "load_gpu_phase_budget_reservation",
            load_then_mutate,
        )
        match = "current reservation receipt changed across semantic validation"
    else:
        original = sync_module._active_ledger_entry

        def validate_then_mutate(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)  # type: ignore[arg-type]
            ledger.write_text("not: a canonical ledger\n", encoding="utf-8")

        monkeypatch.setattr(sync_module, "_active_ledger_entry", validate_then_mutate)
        match = "canonical cost ledger changed across semantic validation"

    with pytest.raises(RunpodSyncError, match=match):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )
    assert lifecycle.exists()


def test_prior_settlement_mutation_after_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, reservation, ledger, current_digest = _project(
        tmp_path,
        upgraded_prior=True,
    )
    prior = next(
        item
        for item in (tmp_path / ".runpod" / "sessions").iterdir()
        if item.name != current_digest
    )
    original = sync_module.validate_completed_runpod_sessions

    def validate_then_mutate(*args: object, **kwargs: object) -> list[dict[str, object]]:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        (prior / "settlement.json").write_text(
            '{"tampered":true}\n',
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        sync_module,
        "validate_completed_runpod_sessions",
        validate_then_mutate,
    )
    with pytest.raises(
        RunpodSyncError,
        match="prior completed session evidence changed across semantic validation",
    ):
        build_selective_sync_plan(
            project_root=tmp_path,
            phase=phase,
            reservation_path=reservation,
            cost_ledger_path=ledger,
        )


def test_source_path_swap_during_materialization_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    source = tmp_path / ".runpod" / "gpu_quote_lock.json"
    source_identity = source.stat()
    original_read = sync_module.os.read
    swapped = False

    def swap_path_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        opened = sync_module.os.fstat(descriptor)
        if not swapped and (opened.st_dev, opened.st_ino) == (
            source_identity.st_dev,
            source_identity.st_ino,
        ):
            raw = source.read_bytes()
            source.replace(source.with_name("gpu_quote_lock.original.json"))
            source.write_bytes(raw)
            source.chmod(0o600)
            swapped = True
        return original_read(descriptor, size)

    monkeypatch.setattr(sync_module.os, "read", swap_path_then_read)
    destination = tmp_path / ".runpod" / "sync_bundles" / current_digest
    with pytest.raises(
        RunpodSyncError,
        match=r"(?:source changed during copy|control changed while fingerprinting)",
    ):
        materialize_selective_sync_bundle(
            project_root=tmp_path,
            destination=destination,
            plan=plan,
        )
    assert swapped is True
    assert not destination.exists()


def test_destination_is_rehashed_after_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    destination = tmp_path / ".runpod" / "sync_bundles" / current_digest
    original_hash = sync_module.sha256_file
    tampered = False

    def tamper_before_destination_hash(path: str | Path) -> str:
        nonlocal tampered
        candidate = Path(path)
        if (
            not tampered
            and candidate.is_relative_to(destination)
            and candidate.name == "gpu_quote_lock.json"
        ):
            raw = candidate.read_bytes()
            candidate.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            tampered = True
        return original_hash(candidate)

    monkeypatch.setattr(sync_module, "sha256_file", tamper_before_destination_hash)
    with pytest.raises(RunpodSyncError, match="destination verification failed"):
        materialize_selective_sync_bundle(
            project_root=tmp_path,
            destination=destination,
            plan=plan,
        )
    assert tampered is True
    assert not destination.exists()


def test_materialization_rejects_symlinked_bundle_root(tmp_path: Path) -> None:
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle_root = tmp_path / ".runpod" / "sync_bundles"
    bundle_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunpodSyncError, match="sync-bundle root is unsafe"):
        materialize_selective_sync_bundle(
            project_root=tmp_path,
            destination=bundle_root / current_digest,
            plan=plan,
        )
    assert list(outside.iterdir()) == []


def test_materialization_rejects_noncanonical_destination(tmp_path: Path) -> None:
    phase, reservation, ledger, _current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    wrong = tmp_path / ".runpod" / "sync_bundles" / ("f" * 64)

    with pytest.raises(RunpodSyncError, match="session-bound sync-bundle path"):
        materialize_selective_sync_bundle(
            project_root=tmp_path,
            destination=wrong,
            plan=plan,
        )
    assert not wrong.exists()


def test_manifest_writer_fsyncs_file_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = sync_module.os.fsync

    def record(descriptor: int) -> None:
        calls.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(sync_module.os, "fsync", record)
    manifest = tmp_path / "bundle" / "manifest.json"
    sync_module._write_manifest_durable(manifest, {"record_hash": "sha256:test"})

    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "record_hash": "sha256:test"
    }
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert len(calls) == 2


def test_documented_rsync_shape_merges_bundle_at_remote_project_root(
    tmp_path: Path,
) -> None:
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("rsync is unavailable")
    phase, reservation, ledger, current_digest = _project(tmp_path)
    plan = build_selective_sync_plan(
        project_root=tmp_path,
        phase=phase,
        reservation_path=reservation,
        cost_ledger_path=ledger,
    )
    bundle = tmp_path / ".runpod" / "sync_bundles" / current_digest
    materialize_selective_sync_bundle(
        project_root=tmp_path,
        destination=bundle,
        plan=plan,
    )
    remote = tmp_path / "remote-project"
    remote.mkdir()
    subprocess.run(
        [rsync, "-a", f"{bundle}/", f"{remote}/"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (remote / ".runpod" / "pod_lifecycle.json").is_file()
    assert (remote / "data" / "manifests" / "cost_ledger.yaml").is_file()
    assert not (remote / ".runpod" / "sessions" / current_digest).exists()
    assert not (remote / ".runpod" / "sync_bundles").exists()
