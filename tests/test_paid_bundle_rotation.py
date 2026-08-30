from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import model_forensics.paid_bundle_rotation as rotation_module
from model_forensics.budget import CostEntry, CostLedger
from model_forensics.execution_bindings import (
    api_route_quote_lock_content_hash,
    gpu_quote_lock_content_hash,
)
from model_forensics.gpu_budget import (
    GpuPhaseBudgetReservation,
    reserve_gpu_phase_budget,
    settle_gpu_phase_budget,
)
from model_forensics.io import stable_hash
from model_forensics.paid_bundle_rotation import (
    PaidBundleRotationError,
    paid_bundle_lock,
    rotate_paid_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _quote_payloads() -> tuple[dict[str, object], dict[str, object]]:
    gpu: dict[str, object] = {
        "schema_version": 1,
        "provider": "runpod",
        "quote_id": "rotation-fixture-h100",
        "gpu_family": "H100_80GB",
        "provider_gpu_id": "NVIDIA H100 80GB HBM3",
        "cloud_type": "SECURE",
        "allowed_cuda_versions": ["12.8"],
        "data_center_ids": ["CA-MTL-1"],
        "gpu_count": 8,
        "container_disk_gb": 50,
        "volume_disk_gb": 650,
        "usd_per_gpu_hour": 2.0,
        "running_storage_usd_per_hour": 0.1,
        "quoted_at": "2026-08-29T20:00:00Z",
        "phase_runtime_allocations": [
            {"command_phase": "behavior_baseline_gpu", "maximum_runtime_hours": 1.0},
            {"command_phase": "behavior_treatment_gpu", "maximum_runtime_hours": 1.0},
            {"command_phase": "resample_gpu", "maximum_runtime_hours": 1.0},
            {"command_phase": "lens_gpu", "maximum_runtime_hours": 1.0},
        ],
        "source_url": "https://www.runpod.io/pricing",
    }
    api: dict[str, object] = {
        "schema_version": 1,
        "provider": "openrouter",
        "source_url": "https://openrouter.ai/models",
        "checked_at": "2026-08-29T20:00:00Z",
        "routes": [
            {
                "role": "primary_final_and_trajectory",
                "model": "anthropic/claude-opus-5",
                "input_usd_per_million_tokens": 15.0,
                "output_usd_per_million_tokens": 75.0,
            },
            {
                "role": "independent_final",
                "model": "google/gemini-3.1-pro-preview",
                "input_usd_per_million_tokens": 2.0,
                "output_usd_per_million_tokens": 12.0,
            },
            {
                "role": "classifier_anthropic",
                "model": "anthropic/claude-opus-5",
                "input_usd_per_million_tokens": 15.0,
                "output_usd_per_million_tokens": 75.0,
            },
            {
                "role": "classifier_google",
                "model": "google/gemini-3.1-pro-preview",
                "input_usd_per_million_tokens": 2.0,
                "output_usd_per_million_tokens": 12.0,
            },
        ],
    }
    return gpu, api


def _write_controls(root: Path, *, include_specs: bool = True) -> dict[str, bytes]:
    gpu, api = _quote_payloads()
    gpu_lock = {**gpu, "content_hash": gpu_quote_lock_content_hash(gpu)}
    api_lock = {**api, "content_hash": api_route_quote_lock_content_hash(api)}
    _write_json(root / ".runpod/gpu_quote_lock.json", gpu_lock)
    _write_json(root / ".runpod/api_route_quote_lock.json", api_lock)
    if include_specs:
        _write_json(root / ".runpod/specs/gpu_quote_spec.json", gpu)
        _write_json(root / ".runpod/specs/api_route_quote_spec.json", api)
    selected = [
        root / ".runpod/gpu_quote_lock.json",
        root / ".runpod/api_route_quote_lock.json",
    ]
    if include_specs:
        selected.extend(
            [
                root / ".runpod/specs/gpu_quote_spec.json",
                root / ".runpod/specs/api_route_quote_spec.json",
            ]
        )
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in selected}


def _lifecycle_payload(
    *,
    reservation: GpuPhaseBudgetReservation,
    operation: str = "stopped",
    pod_status: str = "EXITED",
) -> dict:
    immutable_spec = {"fixture": "rotation"}
    hash_value = stable_hash({"fixture": "binding"})
    authorization = {
        "acknowledged_existing_pod_id_hashes": [],
        "approval_hash": hash_value,
        "approved_phase_maximum_usd": 16.1,
        "approved_runtime_hours": 1.0,
        "bindings_hash": hash_value,
        "gpu_lock_hash": hash_value,
        "immutable_spec_hash": stable_hash(immutable_spec),
        "launch_spec_hash": hash_value,
        "live_hourly_total_usd": 16.1,
        "phase": "behavior_baseline_gpu",
        "quote_hash": hash_value,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
    }
    payload = {
        "schema_version": 1,
        "protocol_version": "runpod-pod-lifecycle-v1",
        "operation": operation,
        "updated_at": "2026-08-29T20:00:00Z",
        "immutable_spec": immutable_spec,
        "current_authorization": authorization,
        "authorization_history": [],
        "pod": {"status": pod_status},
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _complete_external_session(
    *,
    root: Path,
    ledger: CostLedger,
    reservation: GpuPhaseBudgetReservation,
    accounted_usd: float = 1.25,
) -> None:
    session = root / ".runpod/sessions" / reservation.session_hash.removeprefix("sha256:")
    session.mkdir(parents=True, mode=0o700)
    pod_id_hash = stable_hash({"runpod_pod_id": "rotation-fixture-pod"})
    started_at = "2026-08-29T19:00:00Z"
    exited_at = "2026-08-29T19:05:00Z"
    stop = {
        "desired_status": "EXITED",
        "environment_verified": True,
        "started_at": started_at,
        "exited_at": exited_at,
        "runtime_ms": 300_000,
    }
    query = {
        "provider_api": "rest-v1",
        "method": "GET",
        "path": "/v1/billing/pods",
        "grouping": "podId",
        "pod_id_hash": pod_id_hash,
        "start_time": started_at,
        "end_time": exited_at,
    }
    billing = {
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "pod_id_hash": pod_id_hash,
        "provider_amount_usd": None,
        "settlement_amount_usd": accounted_usd,
        "time_billed_ms": None,
        "conservative_ceiling_usd": accounted_usd,
    }
    external = {
        "schema_version": 1,
        "protocol_version": "runpod-external-stop-v1",
        "status": "stopped_verified",
        "provider_api": "rest-v1-read-only",
        "observed_at": "2026-08-29T20:00:00Z",
        "prior_lifecycle_operation": "created",
        "lifecycle_before_hash": stable_hash({"state": "before"}),
        "lifecycle_stopped_hash": stable_hash({"state": "stopped"}),
        "session_hash": reservation.session_hash,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "pod_id_hash": pod_id_hash,
        "stop_evidence": stop,
        "stop_evidence_hash": stable_hash(stop),
        "billing_query": query,
        "billing_query_hash": stable_hash(query),
        "billing_evidence": billing,
        "billing_evidence_hash": stable_hash(billing),
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "settlement_amount_usd": accounted_usd,
        "source_artifact_hashes": [],
    }
    external["record_hash"] = stable_hash(external)
    _write_json(session / "external_stop_receipt.json", external)
    settle_gpu_phase_budget(
        ledger=ledger,
        reservation=reservation,
        incurred_usd=accounted_usd,
    )
    settlement = {
        "schema_version": 2,
        "protocol_version": "cumulative-gpu-phase-settlement-v2",
        "phase": reservation.phase,
        "reservation_id": reservation.reservation_id,
        "reservation_record_hash": reservation.manifest()["record_hash"],
        "session_hash": reservation.session_hash,
        "provider_incurred_usd": None,
        "accounted_gpu_usd": accounted_usd,
        "billing_status": "pending",
        "evidence_kind": "provider_timestamps_conservative_ceiling",
        "external_stop_receipt_hash": external["record_hash"],
        "stop_evidence_hash": external["stop_evidence_hash"],
        "billing_evidence_hash": external["billing_evidence_hash"],
        "status": "settled",
    }
    settlement["record_hash"] = stable_hash(settlement)
    _write_json(session / "settlement.json", settlement)


def _prepare_project(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    private = tmp_path / ".runpod"
    private.mkdir(mode=0o700)
    ledger_path = tmp_path / "data/manifests/cost_ledger.yaml"
    ledger_path.parent.mkdir(parents=True)
    ledger = CostLedger(ledger_path)
    ledger.append(
        CostEntry(
            kind="storage",
            amount_usd=5.0,
            description="bounded RunPod storage reserve",
            status="estimated",
        )
    )
    reservation = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_baseline_gpu",
        session_id="rotation-session-fixture",
        approved_phase_maximum_usd=16.1,
        approved_maximum_runtime_hours=1.0,
        live_hourly_total_usd=16.1,
    )
    _complete_external_session(root=tmp_path, ledger=ledger, reservation=reservation)
    _write_json(
        private / "pod_lifecycle.json",
        _lifecycle_payload(reservation=reservation),
    )
    originals = _write_controls(tmp_path)
    return tmp_path, originals


def test_rotation_archives_exact_bytes_with_fsynced_hash_manifest(tmp_path: Path) -> None:
    root, originals = _prepare_project(tmp_path)

    result = rotate_paid_bundle(project_root=root)

    assert result["provider_calls"] == 0
    assert result["status"] == "complete"
    assert result["bundle_id"].startswith("sha256-")
    archive = root / ".runpod/archive/paid-bundles" / result["bundle_id"]
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_hash"] == stable_hash(
        {key: value for key, value in manifest.items() if key != "record_hash"}
    )
    assert manifest["bundle_content_hash"] == result["bundle_content_hash"]
    for record in manifest["files"]:
        expected = originals[record["source_path"]]
        archived = (archive / record["archive_path"]).read_bytes()
        assert archived == expected
        assert record["sha256"] == "sha256:" + __import__("hashlib").sha256(expected).hexdigest()
        assert record["size_bytes"] == len(expected)
        assert not (root / record["source_path"]).exists()
    completion = json.loads((archive / "rotation_complete.json").read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    assert completion["manifest_record_hash"] == manifest["record_hash"]


def test_rotation_refuses_outstanding_gpu_reservation(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    CostLedger(root / "data/manifests/cost_ledger.yaml").append(
        CostEntry(
            kind="gpu",
            amount_usd=10.0,
            description="active GPU reservation",
            status="estimated",
        )
    )

    with pytest.raises(PaidBundleRotationError, match="estimated GPU reservation"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-active")

    assert (root / ".runpod/gpu_quote_lock.json").exists()
    assert not (root / ".runpod/archive/paid-bundles/blocked-active").exists()


@pytest.mark.parametrize(
    ("operation", "pod_status"),
    [("rearmed", "RUNNING"), ("stopped", "RUNNING"), ("failed", "EXITED")],
)
def test_rotation_refuses_non_stopped_lifecycle(
    tmp_path: Path,
    operation: str,
    pod_status: str,
) -> None:
    root, _ = _prepare_project(tmp_path)
    lifecycle = json.loads(
        (root / ".runpod/pod_lifecycle.json").read_text(encoding="utf-8")
    )
    lifecycle["operation"] = operation
    lifecycle["pod"]["status"] = pod_status
    lifecycle.pop("record_hash")
    lifecycle["record_hash"] = stable_hash(lifecycle)
    _write_json(
        root / ".runpod/pod_lifecycle.json",
        lifecycle,
    )

    with pytest.raises(PaidBundleRotationError, match="stopped and Pod status EXITED"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-running")


def test_rotation_refuses_corrupt_control_before_creating_archive(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    (root / ".runpod/gpu_quote_lock.json").write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(PaidBundleRotationError, match="not valid UTF-8 JSON"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-corrupt")

    assert not (root / ".runpod/archive/paid-bundles/blocked-corrupt").exists()


def test_rotation_refuses_symlinked_or_hardlinked_control(
    tmp_path: Path,
) -> None:
    root, _ = _prepare_project(tmp_path)
    api = root / ".runpod/api_route_quote_lock.json"
    outside = root / "outside-api.json"
    outside.write_bytes(api.read_bytes())
    api.unlink()
    api.symlink_to(outside)

    with pytest.raises(PaidBundleRotationError, match="owned regular file"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-symlink")

    api.unlink()
    os.link(outside, api)
    with pytest.raises(PaidBundleRotationError, match="owned regular file"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-hardlink")


def test_rotation_refuses_archive_id_reuse_and_never_overwrites(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    first = rotate_paid_bundle(project_root=root, bundle_id="explicit-bundle-one")
    assert first["status"] == "complete"
    _write_controls(root, include_specs=False)

    with pytest.raises(PaidBundleRotationError, match="cannot be reused"):
        rotate_paid_bundle(project_root=root, bundle_id="explicit-bundle-one")

    assert (root / ".runpod/gpu_quote_lock.json").exists()


def test_rotation_resumes_after_crash_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, originals = _prepare_project(tmp_path)
    real_unlink = rotation_module._unlink_manifested_source
    calls = 0

    def crash_after_first(root_path: Path, record: dict[str, object]) -> None:
        nonlocal calls
        real_unlink(root_path, record)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated process termination")

    monkeypatch.setattr(rotation_module, "_unlink_manifested_source", crash_after_first)
    with pytest.raises(RuntimeError, match="simulated process termination"):
        rotate_paid_bundle(project_root=root, bundle_id="crash-resume-one")

    archive = root / ".runpod/archive/paid-bundles/crash-resume-one"
    assert (archive / "manifest.json").exists()
    assert not (archive / "rotation_complete.json").exists()
    monkeypatch.setattr(rotation_module, "_unlink_manifested_source", real_unlink)

    resumed = rotate_paid_bundle(project_root=root, bundle_id="crash-resume-one")

    assert resumed["status"] == "complete"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        assert (archive / record["archive_path"]).read_bytes() == originals[record["source_path"]]
        assert not (root / record["source_path"]).exists()


def test_rotation_rejects_tampered_incomplete_archive(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    real_unlink = rotation_module._unlink_manifested_source

    def crash(root_path: Path, record: dict[str, object]) -> None:
        del root_path, record
        raise RuntimeError("crash")

    rotation_module._unlink_manifested_source = crash
    try:
        with pytest.raises(RuntimeError):
            rotate_paid_bundle(project_root=root, bundle_id="tampered-resume")
    finally:
        rotation_module._unlink_manifested_source = real_unlink
    archive = root / ".runpod/archive/paid-bundles/tampered-resume"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    first = archive / manifest["files"][0]["archive_path"]
    first.write_bytes(first.read_bytes() + b"tamper")

    with pytest.raises(PaidBundleRotationError, match="archive"):
        rotate_paid_bundle(project_root=root, bundle_id="tampered-resume")


def test_rotation_rejects_unmanifested_archive_entry(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    real_unlink = rotation_module._unlink_manifested_source

    def crash(root_path: Path, record: dict[str, object]) -> None:
        del root_path, record
        raise RuntimeError("crash")

    rotation_module._unlink_manifested_source = crash
    try:
        with pytest.raises(RuntimeError):
            rotate_paid_bundle(project_root=root, bundle_id="extra-entry-resume")
    finally:
        rotation_module._unlink_manifested_source = real_unlink
    archive = root / ".runpod/archive/paid-bundles/extra-entry-resume"
    (archive / "unmanifested.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(PaidBundleRotationError, match="unexpected file"):
        rotate_paid_bundle(project_root=root, bundle_id="extra-entry-resume")


def test_rotation_requires_canonical_lifecycle_even_without_sessions(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    (root / ".runpod/pod_lifecycle.json").unlink()

    with pytest.raises(PaidBundleRotationError, match="lifecycle is missing or invalid"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-no-lifecycle")


def test_rotation_refuses_deleted_current_session_directory(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    shutil.rmtree(next((root / ".runpod/sessions").iterdir()))

    with pytest.raises(PaidBundleRotationError, match="incurred GPU ledger reservation"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-deleted-session")


def test_rotation_refuses_orphan_incurred_gpu_ledger_entry(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    ledger = CostLedger(root / "data/manifests/cost_ledger.yaml")
    orphan = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_treatment_gpu",
        session_id="orphan-rotation-session",
        approved_phase_maximum_usd=10.0,
        approved_maximum_runtime_hours=1.0,
        live_hourly_total_usd=10.0,
    )
    settle_gpu_phase_budget(ledger=ledger, reservation=orphan, incurred_usd=0.5)

    with pytest.raises(PaidBundleRotationError, match="incurred GPU ledger reservation"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-orphan-ledger")


def test_rotation_refuses_lifecycle_history_omission(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)
    lifecycle_path = root / ".runpod/pod_lifecycle.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    first_authorization = dict(lifecycle["current_authorization"])
    ledger = CostLedger(root / "data/manifests/cost_ledger.yaml")
    second = reserve_gpu_phase_budget(
        ledger=ledger,
        phase="behavior_treatment_gpu",
        session_id="second-rotation-session",
        approved_phase_maximum_usd=10.0,
        approved_maximum_runtime_hours=1.0,
        live_hourly_total_usd=10.0,
    )
    _complete_external_session(root=root, ledger=ledger, reservation=second, accounted_usd=0.5)
    second_authorization = {
        **first_authorization,
        "phase": second.phase,
        "reservation_id": second.reservation_id,
        "reservation_record_hash": second.manifest()["record_hash"],
        "session_hash": second.session_hash,
        "approved_phase_maximum_usd": second.approved_phase_maximum_usd,
        "approved_runtime_hours": second.approved_maximum_runtime_hours,
        "live_hourly_total_usd": second.live_hourly_total_usd,
    }
    lifecycle["current_authorization"] = second_authorization
    lifecycle["authorization_history"] = []  # first authorization was improperly omitted
    lifecycle.pop("record_hash")
    lifecycle["record_hash"] = stable_hash(lifecycle)
    _write_json(lifecycle_path, lifecycle)

    with pytest.raises(PaidBundleRotationError, match="lifecycle authorization"):
        rotate_paid_bundle(project_root=root, bundle_id="blocked-history-omission")


@pytest.mark.parametrize("stage", ["manifest", "archived-control", "completion"])
@pytest.mark.parametrize("boundary", ["pending-fsynced", "link-created"])
def test_rotation_recovers_every_install_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    boundary: str,
) -> None:
    root, _ = _prepare_project(tmp_path)
    real_install = rotation_module._install_exact_file
    injected = False

    def is_target(destination: Path) -> bool:
        if stage == "manifest":
            return destination.name == "manifest.json"
        if stage == "archived-control":
            return destination.name == "gpu_quote_lock.json" and "paid-bundles" in destination.parts
        return destination.name == "rotation_complete.json"

    def crash_install(destination: Path, content: bytes) -> None:
        nonlocal injected
        if injected or not is_target(destination):
            real_install(destination, content)
            return
        injected = True
        pending = destination.with_name(f".{destination.name}.pending")
        pending.write_bytes(content)
        pending.chmod(0o600)
        with pending.open("rb") as handle:
            os.fsync(handle.fileno())
        rotation_module._fsync_directory(destination.parent)
        if boundary == "link-created":
            os.link(pending, destination, follow_symlinks=False)
            rotation_module._fsync_directory(destination.parent)
        raise RuntimeError(f"crash at {stage}/{boundary}")

    monkeypatch.setattr(rotation_module, "_install_exact_file", crash_install)
    with pytest.raises(RuntimeError, match="crash at"):
        rotate_paid_bundle(project_root=root, bundle_id="fault-boundary-one")
    monkeypatch.setattr(rotation_module, "_install_exact_file", real_install)

    resumed = rotate_paid_bundle(project_root=root, bundle_id="fault-boundary-one")

    assert resumed["status"] == "complete"
    archive = root / ".runpod/archive/paid-bundles/fault-boundary-one"
    assert (archive / "manifest.json").stat().st_nlink == 1
    assert (archive / "rotation_complete.json").stat().st_nlink == 1
    assert not list(archive.rglob("*.pending"))


def test_rotation_refuses_while_paid_bundle_shared_lock_is_held(tmp_path: Path) -> None:
    root, _ = _prepare_project(tmp_path)

    with paid_bundle_lock(project_root=root, exclusive=False):
        with pytest.raises(PaidBundleRotationError, match="already held"):
            rotate_paid_bundle(project_root=root, bundle_id="blocked-concurrent")


def test_content_derived_rotation_discovers_linked_completion_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _prepare_project(tmp_path)
    real_install = rotation_module._install_exact_file

    def crash_completion(destination: Path, content: bytes) -> None:
        if destination.name != "rotation_complete.json":
            real_install(destination, content)
            return
        pending = destination.with_name(f".{destination.name}.pending")
        pending.write_bytes(content)
        pending.chmod(0o600)
        os.link(pending, destination, follow_symlinks=False)
        raise RuntimeError("completion link crash")

    monkeypatch.setattr(rotation_module, "_install_exact_file", crash_completion)
    with pytest.raises(RuntimeError, match="completion link crash"):
        rotate_paid_bundle(project_root=root)
    monkeypatch.setattr(rotation_module, "_install_exact_file", real_install)

    resumed = rotate_paid_bundle(project_root=root)

    assert resumed["status"] == "complete"
    assert resumed["bundle_id"].startswith("sha256-")


def test_explicit_rotation_recovers_empty_bundle_directory_after_mkdir_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _prepare_project(tmp_path)
    real_install = rotation_module._install_exact_file

    def crash_before_manifest(destination: Path, content: bytes) -> None:
        if destination.name == "manifest.json":
            raise RuntimeError("crash before manifest staging")
        real_install(destination, content)

    monkeypatch.setattr(rotation_module, "_install_exact_file", crash_before_manifest)
    with pytest.raises(RuntimeError, match="before manifest"):
        rotate_paid_bundle(project_root=root, bundle_id="empty-dir-recovery")
    bundle = root / ".runpod/archive/paid-bundles/empty-dir-recovery"
    assert bundle.is_dir() and not list(bundle.iterdir())
    monkeypatch.setattr(rotation_module, "_install_exact_file", real_install)

    resumed = rotate_paid_bundle(project_root=root, bundle_id="empty-dir-recovery")

    assert resumed["status"] == "complete"
