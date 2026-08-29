from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import model_forensics.gpu_setup as gpu_setup
from model_forensics.bootstrap_environment import (
    BOOTSTRAP_CONSTRAINTS_PATH,
    BOOTSTRAP_CONSTRAINTS_SHA256,
    BOOTSTRAP_DISTRIBUTION_LOCK_HASH,
    BOOTSTRAP_DISTRIBUTION_VERSIONS,
    BOOTSTRAP_ENVIRONMENT_PROTOCOL,
    BootstrapEnvironmentError,
    capture_bootstrap_distribution_provenance,
)
from model_forensics.gpu_budget import GpuBudgetGateError
from model_forensics.gpu_setup import (
    GpuSetupSpec,
    create_gpu_setup_lock,
    validate_gpu_setup_lock,
)
from model_forensics.io import stable_hash, write_json
from model_forensics.semantic_backend import (
    SEMANTIC_DISTRIBUTION_VERSION,
    SEMANTIC_RUNTIME_PROTOCOL,
    SEMANTIC_STACK_LOCK_HASH,
    SEMANTIC_STACK_VERSIONS,
    SEMANTIC_WHEEL_SHA256,
    SEMANTIC_WHEEL_URL,
    TRANSFORMERS_COMMIT,
)


def _pip_freeze() -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in completed.stdout.splitlines() if line)


def _spec(*, wheel_hash: str = "a" * 64) -> GpuSetupSpec:
    return GpuSetupSpec(
        container_image_digest="vllm/vllm-openai@sha256:" + "b" * 64,
        vllm_wheel_url="https://files.pythonhosted.org/packages/vllm.whl",
        vllm_wheel_sha256=wheel_hash,
        transformers_commit=TRANSFORMERS_COMMIT,
        jlens_commit="d" * 40,
    )


def _smoke_manifest(path: Path) -> None:
    payload = {
        "schema_version": 2,
        "protocol_version": "qwen4b-bounded-integration-smoke-v2",
        "status": "passed",
        "scope": "bounded_nonprimary_qwen4b_integration_smoke",
        "experimental_sample": False,
        "primary_eligible": False,
        "synthetic_analysis_fixture": True,
        "paid_api_calls": 0,
        "model": {
            "id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        },
        "registered_prefixes": {
            "retain": {"exact_original_ids_reused": True},
            "resample": {"exact_original_ids_reused": True},
        },
        "forced_append_check": {"immutable_prefix_preserved": True},
        "raw_prefix_continuations": {
            "retain": {"prompt_ids_exact": True},
            "resample": {"prompt_ids_exact": True},
        },
        "deterministic_local_fixture": {
            "synthetic": True,
            "primary_eligible": False,
        },
        "lens_probe_grid": {
            "probe_cell_count": 15,
            "transport_boundary": {
                "fabricated_lens_record_count": 0,
                "activation_transport_executed": False,
            },
        },
        "analysis_evidence_handoff": {
            "analysis_ingest_allowed": False,
            "primary_eligible": False,
        },
    }
    payload["manifest_hash"] = stable_hash(payload)
    write_json(path, payload)


def _semantic_runtime() -> dict[str, object]:
    distributions: dict[str, object] = {}
    for index, (name, version) in enumerate(sorted(SEMANTIC_STACK_VERSIONS.items())):
        row: dict[str, object] = {
            "name": name,
            "version": version,
            "metadata_sha256": f"sha256:{index + 1:064x}",
            "record_sha256": f"sha256:{index + 11:064x}",
            "direct_url_sha256": None,
            "installer": "pip",
            "installed_files": {
                "algorithm": "sha256",
                "verified_file_count": 1,
                "verified_size_bytes": index + 1,
                "manifest_hash": f"sha256:{index + 31:064x}",
            },
        }
        if name == "sentence-transformers":
            row["direct_url_sha256"] = "sha256:" + "a" * 64
            row["source"] = {
                "wheel_filename": SEMANTIC_WHEEL_URL.rsplit("/", 1)[-1],
                "archive_sha256": SEMANTIC_WHEEL_SHA256,
            }
        elif name == "transformers":
            row["direct_url_sha256"] = "sha256:" + "b" * 64
            row["source"] = {
                "vcs": "git",
                "requested_revision": TRANSFORMERS_COMMIT,
                "commit_id": TRANSFORMERS_COMMIT,
            }
        distributions[name] = row
    payload: dict[str, object] = {
        "protocol_version": SEMANTIC_RUNTIME_PROTOCOL,
        "distribution_artifact": {
            "name": "sentence-transformers",
            "version": SEMANTIC_DISTRIBUTION_VERSION,
            "wheel_url": SEMANTIC_WHEEL_URL,
            "wheel_sha256": SEMANTIC_WHEEL_SHA256,
        },
        "stack_versions": dict(sorted(SEMANTIC_STACK_VERSIONS.items())),
        "stack_lock_hash": SEMANTIC_STACK_LOCK_HASH,
        "distributions": distributions,
        "verified_installed_files": {
            "algorithm": "sha256",
            "verified_file_count": len(distributions),
            "verified_size_bytes": sum(range(1, len(distributions) + 1)),
            "manifest_hash": "sha256:" + "c" * 64,
        },
    }
    payload["runtime_hash"] = stable_hash(payload)
    return payload


def _bootstrap_environment() -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": BOOTSTRAP_ENVIRONMENT_PROTOCOL,
        "constraints_path": BOOTSTRAP_CONSTRAINTS_PATH,
        "constraints_sha256": BOOTSTRAP_CONSTRAINTS_SHA256,
        "distribution_versions": dict(sorted(BOOTSTRAP_DISTRIBUTION_VERSIONS.items())),
        "distribution_lock_hash": BOOTSTRAP_DISTRIBUTION_LOCK_HASH,
        "artifact_scope": "exact_top_level_versions_post_install",
        "transitive_artifact_hashes_complete": False,
        "first_install_trust_boundary": "pinned_container_plus_pypi_tls",
    }
    payload["provenance_hash"] = stable_hash(payload)
    return payload


def test_bootstrap_distribution_lock_fails_closed_on_one_version_drift() -> None:
    def version(name: str) -> str:
        if name == "accelerate":
            return "0.0.0"
        return BOOTSTRAP_DISTRIBUTION_VERSIONS[name]

    with pytest.raises(BootstrapEnvironmentError, match="versions drifted"):
        capture_bootstrap_distribution_provenance(version_factory=version)


def test_gpu_setup_lock_allows_exact_rearm_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_runtime = _semantic_runtime()
    bootstrap_environment = _bootstrap_environment()
    monkeypatch.setattr(gpu_setup, "_live_semantic_runtime", lambda path: semantic_runtime)
    monkeypatch.setattr(
        gpu_setup, "_live_bootstrap_environment", lambda path: bootstrap_environment
    )
    environment_path = tmp_path / "gpu_environment.json"
    write_json(
        environment_path,
        {
            "schema_version": 2,
            "pip_freeze": _pip_freeze(),
            "vllm_wheel": {"sha256": "a" * 64},
            "semantic_wheel": {
                "sha256": _spec().semantic_wheel_sha256,
                "filename": SEMANTIC_WHEEL_URL.rsplit("/", 1)[-1],
            },
            "semantic_runtime": semantic_runtime,
            "bootstrap_environment": bootstrap_environment,
        },
    )
    lock_path = tmp_path / "setup_lock.json"
    smoke_path = tmp_path / "qwen4b_prefix_smoke.json"
    _smoke_manifest(smoke_path)
    created = create_gpu_setup_lock(
        path=lock_path,
        spec=_spec(),
        environment_manifest_path=environment_path,
        qwen4b_smoke_manifest_path=smoke_path,
        venv_python_path=sys.executable,
    )
    validated = validate_gpu_setup_lock(
        path=lock_path,
        expected_spec=_spec(),
        environment_manifest_path=environment_path,
        qwen4b_smoke_manifest_path=smoke_path,
        venv_python_path=sys.executable,
    )
    assert validated == created

    with pytest.raises(GpuBudgetGateError, match="pinned launch inputs"):
        validate_gpu_setup_lock(
            path=lock_path,
            expected_spec=_spec(wheel_hash="e" * 64),
            environment_manifest_path=environment_path,
            qwen4b_smoke_manifest_path=smoke_path,
            venv_python_path=sys.executable,
        )
    with pytest.raises(GpuBudgetGateError, match="overwrite"):
        create_gpu_setup_lock(
            path=lock_path,
            spec=_spec(),
            environment_manifest_path=environment_path,
            qwen4b_smoke_manifest_path=smoke_path,
            venv_python_path=sys.executable,
        )

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["raw_prefix_continuations"]["resample"]["prompt_ids_exact"] = False
    write_json(smoke_path, smoke)
    with pytest.raises(GpuBudgetGateError, match="raw-prefix evidence"):
        validate_gpu_setup_lock(
            path=lock_path,
            expected_spec=_spec(),
            environment_manifest_path=environment_path,
            qwen4b_smoke_manifest_path=smoke_path,
            venv_python_path=sys.executable,
        )


def test_gpu_setup_uses_sha256_for_wheel_and_git_sha1_for_commits() -> None:
    assert _spec().transformers_commit == TRANSFORMERS_COMMIT
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        _spec(wheel_hash="a" * 40)
    with pytest.raises(ValueError, match="40-character Git commit"):
        GpuSetupSpec(
            container_image_digest="vllm/vllm-openai@sha256:" + "b" * 64,
            vllm_wheel_url="https://files.pythonhosted.org/packages/vllm.whl",
            vllm_wheel_sha256="a" * 64,
            transformers_commit="c" * 64,
            jlens_commit="d" * 40,
        )
    with pytest.raises(ValueError, match="semantic source lock"):
        GpuSetupSpec(
            container_image_digest="vllm/vllm-openai@sha256:" + "b" * 64,
            vllm_wheel_url="https://files.pythonhosted.org/packages/vllm.whl",
            vllm_wheel_sha256="a" * 64,
            transformers_commit="c" * 40,
            jlens_commit="d" * 40,
        )
