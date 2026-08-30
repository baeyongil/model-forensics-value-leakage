from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from model_forensics.runpod_sync_verify import (
    MANIFEST_RELATIVE_PATH,
    SOURCE_REPOSITORY_URL,
    RunpodSyncVerificationError,
    verify_selective_sync,
)

NOW = datetime(2026, 8, 30, 12, 1, tzinfo=UTC)
PHASE = "behavior_treatment_gpu"
SESSION_HASH = "sha256:" + "1" * 64
POD_ID = "raw-provider-pod-must-never-appear-in-summary"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _authenticated(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "record_hash"}
    value["record_hash"] = _stable_hash(unsigned)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _reservation() -> dict[str, Any]:
    reservation_id = _stable_hash(
        {
            "protocol": "cumulative-gpu-phase-budget-v1",
            "phase": PHASE,
            "session_hash": SESSION_HASH,
        }
    )
    return _authenticated(
        {
            "schema_version": 1,
            "protocol_version": "cumulative-gpu-phase-budget-v1",
            "reservation_id": reservation_id,
            "phase": PHASE,
            "session_hash": SESSION_HASH,
            "approved_phase_maximum_usd": 10.0,
            "approved_maximum_runtime_hours": 1.0,
            "live_hourly_total_usd": 10.0,
            "safety_margin_fraction": 0.05,
            "global_gpu_hard_stop_usd": 220.0,
            "safety_adjusted_gpu_ceiling_usd": 209.0,
            "prior_incurred_gpu_usd": 0.0,
            "prior_reserved_gpu_usd": 0.0,
            "prior_committed_gpu_usd": 0.0,
            "prior_committed_total_usd": 0.0,
            "remaining_safe_gpu_before_phase_usd": 209.0,
            "remaining_total_before_phase_usd": 325.0,
            "maximum_safe_runtime_hours": 1.0,
            "committed_gpu_after_reservation_usd": 10.0,
            "committed_total_after_reservation_usd": 10.0,
        }
    )


def _lifecycle(reservation: dict[str, Any]) -> dict[str, Any]:
    immutable_spec = {"image": "fixture@sha256:" + "2" * 64, "gpu": {"count": 8}}
    authorization = {
        "acknowledged_existing_pod_id_hashes": [],
        "approval_hash": "sha256:" + "3" * 64,
        "approved_phase_maximum_usd": 10.0,
        "approved_runtime_hours": 1.0,
        "bindings_hash": "sha256:" + "4" * 64,
        "gpu_lock_hash": "sha256:" + "5" * 64,
        "immutable_spec_hash": _stable_hash(immutable_spec),
        "launch_spec_hash": "sha256:" + "6" * 64,
        "live_hourly_total_usd": 10.0,
        "phase": PHASE,
        "quote_hash": "sha256:" + "7" * 64,
        "reservation_id": reservation["reservation_id"],
        "reservation_record_hash": reservation["record_hash"],
        "session_hash": SESSION_HASH,
    }
    return _authenticated(
        {
            "schema_version": 1,
            "protocol_version": "runpod-pod-lifecycle-v1",
            "operation": "rearmed",
            "updated_at": "2026-08-30T12:00:00Z",
            "immutable_spec": immutable_spec,
            "current_authorization": authorization,
            "authorization_history": [],
            "pod": {"id": POD_ID, "status": "RUNNING"},
        }
    )


def _ledger(reservation: dict[str, Any], *, status: str = "estimated") -> str:
    return "\n".join(
        [
            "schema_version: 1",
            "currency: USD",
            "hard_stops:",
            "  gpu: 220.0",
            "  api: 100.0",
            "  total: 325.0",
            "entries:",
            "- kind: gpu",
            "  amount_usd: 10.0",
            f"  description: GPU phase {PHASE} session {SESSION_HASH}",
            f"  status: {status}",
            "  occurred_at: '2026-08-30T12:00:00+00:00'",
            f"  entry_id: {reservation['reservation_id']}",
            "",
        ]
    )


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_RELATIVE_PATH


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _ensure_source_checkout(root: Path) -> str:
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        _git(root, "config", "user.email", "verifier-fixture@example.invalid")
        _git(root, "config", "user.name", "Verifier Fixture")
        _git(root, "remote", "add", "origin", SOURCE_REPOSITORY_URL)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "config").mkdir(parents=True, exist_ok=True)
        (root / "src/runner.py").write_text("RUNNER = True\n", encoding="utf-8")
        (root / "scripts/runner.py").write_text("RUNNER = True\n", encoding="utf-8")
        (root / "config/runtime.yaml").write_text("runner: true\n", encoding="utf-8")
        (root / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            '[project]\nname = "verifier-fixture"\nversion = "0"\n',
            encoding="utf-8",
        )
        _git(root, "add", "src", "scripts", "config", "Makefile", "pyproject.toml")
        _git(root, "commit", "-q", "-m", "fixture source")
    return _git(root, "rev-parse", "HEAD")


def _rewrite_manifest(root: Path, mutation: Any | None = None) -> dict[str, Any]:
    path = _manifest_path(root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation is not None:
        mutation(manifest)
    _authenticated(manifest)
    _write_json(path, manifest)
    return manifest


def _refresh_inventory(root: Path) -> dict[str, Any]:
    path = _manifest_path(root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        raw = (root / item["path"]).read_bytes()
        item["sha256"] = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        item["size_bytes"] = len(raw)
    lifecycle = json.loads((root / ".runpod/pod_lifecycle.json").read_text())
    reservation = json.loads(
        (root / ".runpod/reservations" / f"{PHASE}.json").read_text()
    )
    manifest["lifecycle_record_hash"] = lifecycle["record_hash"]
    manifest["reservation_record_hash"] = reservation["record_hash"]
    _authenticated(manifest)
    _write_json(path, manifest)
    return manifest


def _bundle(
    root: Path,
    *,
    now: datetime = NOW,
    status: str = "estimated",
) -> dict[str, Any]:
    source_commit = _ensure_source_checkout(root)
    reservation = _reservation()
    lifecycle = _lifecycle(reservation)
    files: dict[str, bytes] = {
        ".runpod/pod_lifecycle.json": (
            json.dumps(lifecycle, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
        f".runpod/reservations/{PHASE}.json": (
            json.dumps(reservation, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
        ".runpod/gpu_quote_lock.json": b'{"fixture":"gpu-lock"}\n',
        ".runpod/api_route_quote_lock.json": b'{"fixture":"api-lock"}\n',
        ".runpod/paid_run_approval.json": b'{"fixture":"approval"}\n',
        "data/manifests/cost_ledger.yaml": _ledger(
            reservation,
            status=status,
        ).encode(),
    }
    inventory: list[dict[str, Any]] = []
    for relative, raw in sorted(files.items()):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        inventory.append(
            {
                "path": relative,
                "sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
                "size_bytes": len(raw),
            }
        )
    created = now - timedelta(minutes=1)
    manifest = _authenticated(
        {
            "schema_version": 1,
            "protocol_version": "runpod-selective-bootstrap-sync-v1",
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "expires_at": (created + timedelta(minutes=5))
            .isoformat()
            .replace("+00:00", "Z"),
            "phase": PHASE,
            "session_hash": SESSION_HASH,
            "lifecycle_record_hash": lifecycle["record_hash"],
            "reservation_record_hash": reservation["record_hash"],
            "running_pod_id_hash": _stable_hash({"runpod_pod_id": POD_ID}),
            "source_commit": source_commit,
            "source_repository_url": SOURCE_REPOSITORY_URL,
            "current_host_session_excluded": True,
            "current_host_guard": {
                "acknowledgement_file_hash": "sha256:" + "8" * 64,
                "acknowledgement_record_hash": "sha256:" + "9" * 64,
                "watcher_process_identity_hash": "sha256:" + "a" * 64,
                "watchdog_invariant_hash": "sha256:" + "b" * 64,
                "direct_ssh_endpoint_hash": "sha256:" + "c" * 64,
            },
            "files": inventory,
        }
    )
    _write_json(_manifest_path(root), manifest)
    return manifest


def _verify(root: Path, *, observed_at: datetime = NOW) -> dict[str, Any]:
    return verify_selective_sync(
        project_root=root,
        source_checkout=root,
        observed_at=observed_at,
    )


def test_valid_bundle_verifies_with_secret_safe_summary(tmp_path: Path) -> None:
    _bundle(tmp_path)

    summary = _verify(tmp_path)

    assert summary["passed"] is True
    assert summary["file_count"] == 6
    assert summary["current_host_session_absent"] is True
    assert summary["running_pod_id_hash"] == _stable_hash({"runpod_pod_id": POD_ID})
    assert POD_ID not in json.dumps(summary)


@pytest.mark.parametrize(
    ("created_delta", "expires_delta", "message"),
    [
        (timedelta(minutes=-10), timedelta(minutes=-5), "expired"),
        (timedelta(seconds=61), timedelta(minutes=5), "future"),
        (timedelta(minutes=-1), timedelta(minutes=6), "validity window"),
    ],
)
def test_manifest_time_window_fails_closed(
    tmp_path: Path,
    created_delta: timedelta,
    expires_delta: timedelta,
    message: str,
) -> None:
    _bundle(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        created = NOW + created_delta
        manifest["created_at"] = created.isoformat().replace("+00:00", "Z")
        manifest["expires_at"] = (NOW + expires_delta).isoformat().replace(
            "+00:00",
            "Z",
        )

    _rewrite_manifest(tmp_path, mutate)
    with pytest.raises(RunpodSyncVerificationError, match=message):
        _verify(tmp_path)


def test_manifest_record_hash_tamper_fails_closed(tmp_path: Path) -> None:
    _bundle(tmp_path)
    manifest = json.loads(_manifest_path(tmp_path).read_text())
    manifest["phase"] = "tampered_phase"
    _write_json(_manifest_path(tmp_path), manifest)

    with pytest.raises(RunpodSyncVerificationError, match="record hash"):
        _verify(tmp_path)


@pytest.mark.parametrize("mutation", ["content", "size"])
def test_inventory_hash_and_size_are_exact(tmp_path: Path, mutation: str) -> None:
    _bundle(tmp_path)
    target = tmp_path / ".runpod/gpu_quote_lock.json"
    if mutation == "content":
        target.write_bytes(b'{"fixture":"gpu-l0ck"}\n')
    else:
        target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(RunpodSyncVerificationError, match="hash or size"):
        _verify(tmp_path)


def test_unlisted_private_file_fails_exact_tree_check(tmp_path: Path) -> None:
    _bundle(tmp_path)
    (tmp_path / ".runpod/unlisted.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RunpodSyncVerificationError, match="exact inventory"):
        _verify(tmp_path)


def test_missing_listed_private_file_fails_exact_tree_check(tmp_path: Path) -> None:
    _bundle(tmp_path)
    (tmp_path / ".runpod/gpu_quote_lock.json").unlink()

    with pytest.raises(RunpodSyncVerificationError, match="exact inventory"):
        _verify(tmp_path)


def test_current_session_directory_must_be_absent(tmp_path: Path) -> None:
    _bundle(tmp_path)
    current = tmp_path / ".runpod/sessions" / SESSION_HASH.removeprefix("sha256:")
    current.mkdir(parents=True)

    with pytest.raises(RunpodSyncVerificationError, match="current host session"):
        _verify(tmp_path)


def test_running_pod_hash_must_bind_authenticated_lifecycle(tmp_path: Path) -> None:
    _bundle(tmp_path)
    lifecycle_path = tmp_path / ".runpod/pod_lifecycle.json"
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["pod"]["id"] = "different-private-provider-id"
    _authenticated(lifecycle)
    _write_json(lifecycle_path, lifecycle)
    _refresh_inventory(tmp_path)

    with pytest.raises(RunpodSyncVerificationError, match="running Pod hash"):
        _verify(tmp_path)


def test_ledger_must_keep_current_reservation_active(tmp_path: Path) -> None:
    manifest = _bundle(tmp_path)
    reservation = json.loads(
        (tmp_path / f".runpod/reservations/{PHASE}.json").read_text()
    )
    (tmp_path / "data/manifests/cost_ledger.yaml").write_text(
        _ledger(reservation, status="incurred"),
        encoding="utf-8",
    )
    _refresh_inventory(tmp_path)
    assert manifest["record_hash"] != json.loads(_manifest_path(tmp_path).read_text())[
        "record_hash"
    ]

    with pytest.raises(RunpodSyncVerificationError, match="not exact and active"):
        _verify(tmp_path)


def test_secret_field_is_rejected_even_when_inventory_is_reauthenticated(
    tmp_path: Path,
) -> None:
    _bundle(tmp_path)
    target = tmp_path / ".runpod/gpu_quote_lock.json"
    _write_json(target, {"api_key": "credential-value-was-never-printed"})
    _refresh_inventory(tmp_path)

    with pytest.raises(RunpodSyncVerificationError, match="secret field"):
        _verify(tmp_path)


def test_manifest_source_commit_must_equal_checkout_head(tmp_path: Path) -> None:
    _bundle(tmp_path)
    _rewrite_manifest(
        tmp_path,
        lambda manifest: manifest.update(source_commit="f" * 40),
    )

    with pytest.raises(RunpodSyncVerificationError, match="commit disagrees"):
        _verify(tmp_path)


def test_manifest_and_checkout_require_canonical_source_origin(tmp_path: Path) -> None:
    _bundle(tmp_path)
    _rewrite_manifest(
        tmp_path,
        lambda manifest: manifest.update(
            source_repository_url="https://example.invalid/untrusted.git"
        ),
    )
    with pytest.raises(RunpodSyncVerificationError, match="not canonical"):
        _verify(tmp_path)

    _rewrite_manifest(
        tmp_path,
        lambda manifest: manifest.update(
            source_repository_url=SOURCE_REPOSITORY_URL
        ),
    )
    _git(
        tmp_path,
        "remote",
        "set-url",
        "origin",
        "https://example.invalid/untrusted.git",
    )
    with pytest.raises(RunpodSyncVerificationError, match="origin disagrees"):
        _verify(tmp_path)


def test_dirty_tracked_runner_source_fails_closed(tmp_path: Path) -> None:
    _bundle(tmp_path)
    (tmp_path / "src/runner.py").write_text("RUNNER = False\n", encoding="utf-8")

    with pytest.raises(RunpodSyncVerificationError, match=r"runner source.*dirty"):
        _verify(tmp_path)


def test_cli_runs_without_site_packages_and_never_prints_raw_pod_id(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    repository = Path(__file__).parents[1]
    source = tmp_path / "source-checkout"
    (source / "src/model_forensics").mkdir(parents=True)
    (source / "scripts").mkdir(parents=True)
    shutil.copyfile(
        repository / "src/model_forensics/runpod_sync_verify.py",
        source / "src/model_forensics/runpod_sync_verify.py",
    )
    shutil.copyfile(
        repository / "scripts/verify_runpod_sync_bundle.py",
        source / "scripts/verify_runpod_sync_bundle.py",
    )
    _bundle(source, now=now)
    script = source / "scripts/verify_runpod_sync_bundle.py"

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(script),
            "--project-root",
            str(source),
            "--source-checkout",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["passed"] is True
    assert POD_ID not in completed.stdout
    assert POD_ID not in completed.stderr


def test_verifier_production_module_imports_only_standard_library() -> None:
    module = (
        Path(__file__).parents[1]
        / "src/model_forensics/runpod_sync_verify.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {
        "__future__",
        "collections",
        "datetime",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "subprocess",
        "typing",
    }
