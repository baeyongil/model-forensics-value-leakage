from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import model_forensics.runpod_sync_transfer as transfer_module
from model_forensics.runpod_sync_transfer import (
    PUBLIC_REPOSITORY_URL,
    REMOTE_DESTINATION,
    RunpodSyncTransferError,
    _install_staged_sync,
    transfer_runpod_sync_bundle,
)

SESSION_HASH = "sha256:" + "1" * 64
SOURCE_COMMIT = "a" * 40
REMOTE_IPV4 = "198.51.100.23"
REMOTE_HOST = f"root@{REMOTE_IPV4}"
REMOTE_PORT = 2222


def _direct_ssh_endpoint_hash(*, public_ip: str, public_port: int) -> str:
    canonical = json.dumps(
        {"public_ip": public_ip, "public_port": public_port},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    raw = f"runpod-rest-v1-direct-ssh-v1:{canonical}".encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _plan() -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "runpod-selective-bootstrap-sync-v1",
        "phase": "gpu-rearm",
        "session_hash": SESSION_HASH,
        "created_at": "2026-08-29T12:00:00Z",
        "expires_at": "2026-08-29T12:05:00Z",
        "running_pod_id_hash": "sha256:" + "2" * 64,
        "source_commit": SOURCE_COMMIT,
        "source_repository_url": PUBLIC_REPOSITORY_URL,
        "lifecycle_record_hash": "sha256:" + "3" * 64,
        "reservation_record_hash": "sha256:" + "4" * 64,
        "current_host_session_excluded": True,
        "current_host_guard": {
            "acknowledgement_file_hash": "sha256:" + "5" * 64,
            "acknowledgement_record_hash": "sha256:" + "6" * 64,
            "direct_ssh_endpoint_hash": _direct_ssh_endpoint_hash(
                public_ip=REMOTE_IPV4,
                public_port=REMOTE_PORT,
            ),
            "watcher_process_identity_hash": "sha256:" + "7" * 64,
            "watchdog_invariant_hash": "sha256:" + "8" * 64,
        },
        "files": [],
    }
    plan["record_hash"] = transfer_module._stable_hash(plan)
    return plan


class FakeRunner:
    def __init__(
        self,
        *,
        fail_call: int | None = None,
        on_call: Any = None,
    ) -> None:
        self.commands: list[list[str]] = []
        self.fail_call = fail_call
        self.on_call = on_call

    def __call__(self, argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(argv))
        if self.on_call is not None:
            self.on_call(len(self.commands), list(argv))
        return SimpleNamespace(
            returncode=1 if len(self.commands) == self.fail_call else 0,
            stdout="not reflected",
            stderr="secret must not escape",
        )


def _dependencies(root: Path, *, revalidator: Any = None) -> tuple[Any, Any, Any]:
    plan = _plan()

    def builder(**_kwargs: Any) -> dict[str, Any]:
        return plan

    def materializer(*, destination: Path, plan: dict[str, Any], **_kwargs: Any) -> Path:
        manifest = destination / ".runpod" / "selective_sync_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(plan), encoding="utf-8")
        return destination

    def validate(**_kwargs: Any) -> None:
        return None

    return builder, materializer, revalidator or validate


def _invoke(
    root: Path,
    runner: FakeRunner,
    *,
    revalidator: Any = None,
    remote_destination: Path = REMOTE_DESTINATION,
    remote_host: str = REMOTE_HOST,
    remote_port: int = REMOTE_PORT,
    emergency_stopper: Any = None,
) -> dict[str, Any]:
    (root / ".runpod").mkdir(exist_ok=True)
    (root / ".runpod" / "sessions" / SESSION_HASH.removeprefix("sha256:")).mkdir(
        parents=True, exist_ok=True
    )
    builder, materializer, validate = _dependencies(root, revalidator=revalidator)
    return transfer_runpod_sync_bundle(
        project_root=root,
        phase="gpu-rearm",
        reservation_path=root / ".runpod" / "reservations" / "gpu-rearm.json",
        cost_ledger_path=root / "data" / "manifests" / "cost_ledger.yaml",
        remote_host=remote_host,
        remote_port=remote_port,
        remote_destination=remote_destination,
        command_runner=runner,
        _plan_builder=builder,
        _materializer=materializer,
        _revalidator=validate,
        _emergency_stopper=emergency_stopper or (lambda: None),
    )


def test_one_shot_transfer_uses_clean_stage_and_exact_destination(tmp_path: Path) -> None:
    runner = FakeRunner()
    validations = 0

    def revalidate(**_kwargs: Any) -> None:
        nonlocal validations
        validations += 1

    summary = _invoke(tmp_path, runner, revalidator=revalidate)

    assert summary["passed"] is True
    assert validations == 9
    assert len(runner.commands) == 7
    flattened = "\n".join(" ".join(command) for command in runner.commands)
    assert "/workspace/model-forensics-value-leakage" in flattened
    assert "/workspace/.model-forensics-sync-stage-" in flattened
    assert "--delete" not in flattened
    assert "pod_id" not in json.dumps(summary)
    assert "git clone" in " ".join(runner.commands[0])
    assert PUBLIC_REPOSITORY_URL in runner.commands[0]
    for command in runner.commands:
        if command[0] == "ssh":
            assert command[1:3] == ["-F", "/dev/null"]
    assert runner.commands[3][0] == "rsync"
    rsh_index = runner.commands[3].index("--rsh")
    assert runner.commands[3][rsh_index + 1].startswith("ssh -F /dev/null ")
    assert "--exclude=*" in runner.commands[3]
    assert "--include=/.runpod/selective_sync_manifest.json" in runner.commands[3]
    assert "verify_runpod_sync_bundle.py" in " ".join(runner.commands[4])
    assert "install_runpod_sync_bundle.py" in " ".join(runner.commands[5])
    assert "verify_runpod_sync_bundle.py" in " ".join(runner.commands[6])
    assert not (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()


def test_transfer_claim_cannot_be_replayed(tmp_path: Path) -> None:
    _invoke(tmp_path, FakeRunner())
    with pytest.raises(RunpodSyncTransferError, match="already has"):
        _invoke(tmp_path, FakeRunner())
    assert (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()


def test_wrong_remote_destination_fails_before_any_callback(tmp_path: Path) -> None:
    runner = FakeRunner()
    with pytest.raises(RunpodSyncTransferError, match="pinned project checkout"):
        _invoke(tmp_path, runner, remote_destination=Path("/workspace/wrong"))
    assert runner.commands == []


@pytest.mark.parametrize(
    "remote_host",
    (
        "pod.example.test",
        "root@pod.example.test",
        REMOTE_IPV4,
        f"ubuntu@{REMOTE_IPV4}",
        "root@198.51.100.023",
        "root@[198.51.100.23]",
        "root@2001:db8::1",
    ),
)
def test_ambiguous_or_non_direct_remote_host_fails_before_plan_callback(
    tmp_path: Path,
    remote_host: str,
) -> None:
    runner = FakeRunner()
    callback_called = False

    def builder(**_kwargs: Any) -> dict[str, Any]:
        nonlocal callback_called
        callback_called = True
        return _plan()

    with pytest.raises(RunpodSyncTransferError, match="root@canonical-IPv4"):
        transfer_runpod_sync_bundle(
            project_root=tmp_path,
            phase="gpu-rearm",
            reservation_path=tmp_path / "reservation.json",
            cost_ledger_path=tmp_path / "ledger.yaml",
            remote_host=remote_host,
            remote_port=REMOTE_PORT,
            _plan_builder=builder,
            _materializer=lambda **_kwargs: tmp_path,
            _revalidator=lambda **_kwargs: None,
            command_runner=runner,
        )

    assert callback_called is False
    assert runner.commands == []
    assert not (tmp_path / ".runpod").exists()


@pytest.mark.parametrize(
    ("remote_host", "remote_port"),
    (
        ("root@198.51.100.24", REMOTE_PORT),
        (REMOTE_HOST, 2223),
    ),
)
def test_endpoint_mismatch_fails_before_materialize_claim_stop_or_remote_command(
    tmp_path: Path,
    remote_host: str,
    remote_port: int,
) -> None:
    private = tmp_path / ".runpod"
    session = private / "sessions" / SESSION_HASH.removeprefix("sha256:")
    session.mkdir(parents=True)
    runner = FakeRunner()
    materialized = False
    revalidated = False

    def materializer(**_kwargs: Any) -> Path:
        nonlocal materialized
        materialized = True
        return tmp_path

    def revalidator(**_kwargs: Any) -> None:
        nonlocal revalidated
        revalidated = True

    with pytest.raises(RunpodSyncTransferError, match="authenticated host guard"):
        transfer_runpod_sync_bundle(
            project_root=tmp_path,
            phase="gpu-rearm",
            reservation_path=private / "reservations" / "gpu-rearm.json",
            cost_ledger_path=tmp_path / "data" / "manifests" / "cost_ledger.yaml",
            remote_host=remote_host,
            remote_port=remote_port,
            _plan_builder=lambda **_kwargs: _plan(),
            _materializer=materializer,
            _revalidator=revalidator,
            command_runner=runner,
        )

    assert materialized is False
    assert revalidated is False
    assert runner.commands == []
    assert not (private / "sync_bundles").exists()
    assert not (private / "sync_claims").exists()
    assert not (session / "runpod_stop.request").exists()


@pytest.mark.parametrize(
    "guard_mutation",
    (
        "missing-key",
        "extra-key",
        "malformed-value",
        "not-a-mapping",
    ),
)
def test_malformed_exact_host_guard_fails_before_any_transfer_side_effect(
    tmp_path: Path,
    guard_mutation: str,
) -> None:
    private = tmp_path / ".runpod"
    session = private / "sessions" / SESSION_HASH.removeprefix("sha256:")
    session.mkdir(parents=True)
    runner = FakeRunner()
    materialized = False
    revalidated = False
    plan = _plan()
    guard = plan["current_host_guard"]
    assert isinstance(guard, dict)
    if guard_mutation == "missing-key":
        guard.pop("watchdog_invariant_hash")
    elif guard_mutation == "extra-key":
        guard["unexpected_hash"] = "sha256:" + "9" * 64
    elif guard_mutation == "malformed-value":
        guard["watcher_process_identity_hash"] = "9" * 64
    else:
        plan["current_host_guard"] = [guard]
    plan["record_hash"] = transfer_module._stable_hash(
        {key: value for key, value in plan.items() if key != "record_hash"}
    )

    def materializer(**_kwargs: Any) -> Path:
        nonlocal materialized
        materialized = True
        return tmp_path

    def revalidator(**_kwargs: Any) -> None:
        nonlocal revalidated
        revalidated = True

    with pytest.raises(RunpodSyncTransferError, match="host guard binding"):
        transfer_runpod_sync_bundle(
            project_root=tmp_path,
            phase="gpu-rearm",
            reservation_path=private / "reservations" / "gpu-rearm.json",
            cost_ledger_path=tmp_path / "data" / "manifests" / "cost_ledger.yaml",
            remote_host=REMOTE_HOST,
            remote_port=REMOTE_PORT,
            _plan_builder=lambda **_kwargs: plan,
            _materializer=materializer,
            _revalidator=revalidator,
            command_runner=runner,
        )

    assert materialized is False
    assert revalidated is False
    assert runner.commands == []
    assert not (private / "sync_bundles").exists()
    assert not (private / "sync_claims").exists()
    assert not (session / "runpod_stop.request").exists()


def test_bad_plan_record_hash_fails_before_any_transfer_side_effect(
    tmp_path: Path,
) -> None:
    private = tmp_path / ".runpod"
    session = private / "sessions" / SESSION_HASH.removeprefix("sha256:")
    session.mkdir(parents=True)
    runner = FakeRunner()
    materialized = False
    revalidated = False
    plan = _plan()
    plan["record_hash"] = "sha256:" + "f" * 64

    def materializer(**_kwargs: Any) -> Path:
        nonlocal materialized
        materialized = True
        return tmp_path

    def revalidator(**_kwargs: Any) -> None:
        nonlocal revalidated
        revalidated = True

    with pytest.raises(RunpodSyncTransferError, match="record hash does not authenticate"):
        transfer_runpod_sync_bundle(
            project_root=tmp_path,
            phase="gpu-rearm",
            reservation_path=private / "reservations" / "gpu-rearm.json",
            cost_ledger_path=tmp_path / "data" / "manifests" / "cost_ledger.yaml",
            remote_host=REMOTE_HOST,
            remote_port=REMOTE_PORT,
            _plan_builder=lambda **_kwargs: plan,
            _materializer=materializer,
            _revalidator=revalidator,
            command_runner=runner,
        )

    assert materialized is False
    assert revalidated is False
    assert runner.commands == []
    assert not (private / "sync_bundles").exists()
    assert not (private / "sync_claims").exists()
    assert not (session / "runpod_stop.request").exists()


def test_noncanonical_manifest_repository_is_rejected_before_materialization(
    tmp_path: Path,
) -> None:
    (tmp_path / ".runpod").mkdir()
    plan = _plan()
    plan["source_repository_url"] = "https://example.invalid/untrusted.git"
    plan["record_hash"] = transfer_module._stable_hash(
        {key: value for key, value in plan.items() if key != "record_hash"}
    )
    materialized = False

    def materializer(**_kwargs: Any) -> Path:
        nonlocal materialized
        materialized = True
        return tmp_path

    with pytest.raises(RunpodSyncTransferError, match="canonical public repository"):
        transfer_runpod_sync_bundle(
            project_root=tmp_path,
            phase="gpu-rearm",
            reservation_path=tmp_path / "reservation.json",
            cost_ledger_path=tmp_path / "ledger.yaml",
            remote_host=REMOTE_HOST,
            remote_port=REMOTE_PORT,
            _plan_builder=lambda **_kwargs: plan,
            _materializer=materializer,
            _revalidator=lambda **_kwargs: None,
            command_runner=FakeRunner(),
        )
    assert materialized is False


def test_remote_prep_failure_burns_claim_and_uses_both_stop_paths(tmp_path: Path) -> None:
    runner = FakeRunner(fail_call=2)
    emergency_stops = 0

    def emergency_stop() -> None:
        nonlocal emergency_stops
        emergency_stops += 1

    with pytest.raises(RunpodSyncTransferError, match="exact-source checkout"):
        _invoke(tmp_path, runner, emergency_stopper=emergency_stop)
    assert (
        tmp_path
        / ".runpod"
        / "sync_claims"
        / f"{SESSION_HASH.removeprefix('sha256:')}.json"
    ).is_file()
    assert (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).is_file()
    assert emergency_stops == 1
    assert "rm -rf" in " ".join(runner.commands[-1])


def test_claim_is_created_before_the_first_remote_contact(tmp_path: Path) -> None:
    claim = tmp_path / ".runpod" / "sync_claims" / f"{SESSION_HASH.removeprefix('sha256:')}.json"

    def observe(call_number: int, command: list[str]) -> None:
        if call_number == 1:
            assert command[0] == "ssh"
            assert claim.is_file()

    _invoke(tmp_path, FakeRunner(on_call=observe))


def test_rsync_filters_copy_only_manifest_inventory(tmp_path: Path) -> None:
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("rsync is unavailable")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / ".runpod").mkdir(parents=True)
    (source / "data" / "manifests").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / ".runpod" / "selective_sync_manifest.json").write_text("{}\n")
    (source / "data" / "manifests" / "cost_ledger.yaml").write_text("ledger\n")
    (source / ".git" / "config").write_text("must not copy\n")
    (source / "unexpected.txt").write_text("must not copy\n")
    destination.mkdir()
    plan = _plan()
    plan["files"] = [
        {
            "path": "data/manifests/cost_ledger.yaml",
            "sha256": "sha256:" + "0" * 64,
            "size_bytes": 7,
        }
    ]

    subprocess.run(
        [
            rsync,
            "-a",
            *transfer_module._rsync_inventory_filters(plan),
            f"{source}/",
            f"{destination}/",
        ],
        check=True,
    )

    assert (destination / ".runpod" / "selective_sync_manifest.json").is_file()
    assert (destination / "data" / "manifests" / "cost_ledger.yaml").is_file()
    assert not (destination / ".git").exists()
    assert not (destination / "unexpected.txt").exists()


def test_stale_guard_fails_before_first_remote_command(tmp_path: Path) -> None:
    runner = FakeRunner()

    def stale(**_kwargs: Any) -> None:
        raise RunpodSyncTransferError("stale host guard")

    with pytest.raises(RunpodSyncTransferError, match="stale host guard"):
        _invoke(tmp_path, runner, revalidator=stale)
    assert runner.commands == []
    assert not (tmp_path / ".runpod" / "sync_claims").exists()
    assert not (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()


def test_watcher_death_after_rsync_stops_before_remote_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = FakeRunner()
    validations = 0
    emergency_stops = 0
    fsynced_directories: list[Path] = []
    original_fsync = transfer_module._fsync_directory

    def record_fsync(path: Path) -> None:
        fsynced_directories.append(path)
        original_fsync(path)

    monkeypatch.setattr(transfer_module, "_fsync_directory", record_fsync)

    def watcher(**_kwargs: Any) -> None:
        nonlocal validations
        validations += 1
        if validations == 6:
            raise RunpodSyncTransferError("watcher died")

    def emergency_stop() -> None:
        nonlocal emergency_stops
        emergency_stops += 1

    with pytest.raises(RunpodSyncTransferError, match="watcher died"):
        _invoke(
            tmp_path,
            runner,
            revalidator=watcher,
            emergency_stopper=emergency_stop,
        )
    assert runner.commands[0][0] == "ssh"
    assert runner.commands[3][0] == "rsync"
    assert "rm -rf" in " ".join(runner.commands[4])
    assert not any("verify_runpod_sync_bundle.py" in " ".join(c) for c in runner.commands)
    assert (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()
    assert emergency_stops == 1
    assert (
        tmp_path / ".runpod" / "sessions" / SESSION_HASH.removeprefix("sha256:")
    ) in fsynced_directories


def test_remote_stage_tamper_fails_before_install_and_is_cleaned(tmp_path: Path) -> None:
    # clone, checkout, source validation, rsync, then staged verification fails.
    runner = FakeRunner(fail_call=5)
    with pytest.raises(RunpodSyncTransferError, match="staged-bundle verification"):
        _invoke(tmp_path, runner)
    assert "verify_runpod_sync_bundle.py" in " ".join(runner.commands[4])
    assert "rm -rf" in " ".join(runner.commands[5])
    assert not any(" install " in f" {' '.join(c)} " for c in runner.commands)
    assert (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()


def test_lifecycle_change_after_final_remote_verify_fails_closed(tmp_path: Path) -> None:
    runner = FakeRunner()
    validations = 0

    def lifecycle(**_kwargs: Any) -> None:
        nonlocal validations
        validations += 1
        if validations == 9:
            raise RunpodSyncTransferError("lifecycle changed")

    with pytest.raises(RunpodSyncTransferError, match="lifecycle changed"):
        _invoke(tmp_path, runner, revalidator=lifecycle)
    assert any("verify_runpod_sync_bundle.py" in " ".join(c) for c in runner.commands)
    assert "rm -rf" in " ".join(runner.commands[-1])
    assert (
        tmp_path
        / ".runpod"
        / "sessions"
        / SESSION_HASH.removeprefix("sha256:")
        / "runpod_stop.request"
    ).exists()


def _write_install_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    destination = workspace / "model-forensics-value-leakage"
    (destination / ".runpod").mkdir(parents=True)
    (destination / "old-source.txt").write_text("old source", encoding="utf-8")
    (destination / ".runpod" / "old.txt").write_text("old private", encoding="utf-8")
    (destination / "data" / "manifests").mkdir(parents=True)
    (destination / "data" / "manifests" / "cost_ledger.yaml").write_text(
        "old ledger\n",
        encoding="utf-8",
    )
    stage = workspace / (".model-forensics-sync-stage-" + "8" * 64)
    (stage / ".runpod").mkdir(parents=True)
    (stage / "new-source.txt").write_text("new source", encoding="utf-8")
    (stage / ".runpod" / "new.txt").write_text("new private", encoding="utf-8")
    (stage / "data" / "manifests").mkdir(parents=True)
    (stage / "data" / "manifests" / "cost_ledger.yaml").write_text(
        "new ledger\n",
        encoding="utf-8",
    )
    return workspace, destination, stage


def _verification_summary() -> dict[str, Any]:
    return {
        "passed": True,
        "manifest_record_hash": "sha256:" + "9" * 64,
        "session_hash": SESSION_HASH,
        "source_commit": SOURCE_COMMIT,
    }


def test_installer_archives_old_checkout_and_promotes_verified_stage(tmp_path: Path) -> None:
    workspace, destination, stage = _write_install_fixture(tmp_path)
    calls: list[Path] = []

    def verify(**kwargs: Any) -> dict[str, Any]:
        calls.append(Path(kwargs["project_root"]))
        return _verification_summary()

    result = _install_staged_sync(
        stage=stage,
        destination=destination,
        archive_root=workspace / ".model-forensics-sync-archive",
        source_checkout=stage,
        expected_manifest_record_hash="sha256:" + "9" * 64,
        expected_session_hash=SESSION_HASH,
        expected_source_commit=SOURCE_COMMIT,
        verifier=verify,
    )

    archive = workspace / ".model-forensics-sync-archive" / ("8" * 64)
    assert result["passed"] is True
    assert calls == [stage, destination]
    assert not stage.exists()
    assert (destination / ".runpod" / "new.txt").read_text() == "new private"
    assert (destination / "data" / "manifests" / "cost_ledger.yaml").read_text() == "new ledger\n"
    archived_checkout = archive / destination.name
    assert (destination / "new-source.txt").read_text() == "new source"
    assert (archived_checkout / "old-source.txt").read_text() == "old source"
    assert (archived_checkout / ".runpod" / "old.txt").read_text() == "old private"
    assert (
        archived_checkout / "data" / "manifests" / "cost_ledger.yaml"
    ).read_text() == "old ledger\n"


def test_installer_rolls_back_checkout_when_installed_verify_fails(
    tmp_path: Path,
) -> None:
    workspace, destination, stage = _write_install_fixture(tmp_path)
    calls = 0

    def verify(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RunpodSyncTransferError("tampered after move")
        return _verification_summary()

    with pytest.raises(RunpodSyncTransferError, match="prior checkout was restored"):
        _install_staged_sync(
            stage=stage,
            destination=destination,
            archive_root=workspace / ".model-forensics-sync-archive",
            source_checkout=stage,
            expected_manifest_record_hash="sha256:" + "9" * 64,
            expected_session_hash=SESSION_HASH,
            expected_source_commit=SOURCE_COMMIT,
            verifier=verify,
        )

    assert (destination / ".runpod" / "old.txt").read_text() == "old private"
    assert (destination / "old-source.txt").read_text() == "old source"
    assert (destination / "data" / "manifests" / "cost_ledger.yaml").read_text() == "old ledger\n"
    assert (stage / ".runpod" / "new.txt").read_text() == "new private"
    assert (stage / "new-source.txt").read_text() == "new source"
    assert (stage / "data" / "manifests" / "cost_ledger.yaml").read_text() == "new ledger\n"


def test_installer_rejects_symlinked_archive_root(tmp_path: Path) -> None:
    workspace, destination, stage = _write_install_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".model-forensics-sync-archive").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunpodSyncTransferError, match="archive root"):
        _install_staged_sync(
            stage=stage,
            destination=destination,
            archive_root=workspace / ".model-forensics-sync-archive",
            source_checkout=stage,
            expected_manifest_record_hash="sha256:" + "9" * 64,
            expected_session_hash=SESSION_HASH,
            expected_source_commit=SOURCE_COMMIT,
            verifier=lambda **_kwargs: _verification_summary(),
        )
