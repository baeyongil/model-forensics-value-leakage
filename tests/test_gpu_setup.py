from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from model_forensics.gpu_budget import GpuBudgetGateError
from model_forensics.gpu_setup import (
    GpuSetupSpec,
    create_gpu_setup_lock,
    validate_gpu_setup_lock,
)
from model_forensics.io import write_json


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
        transformers_commit="c" * 40,
        jlens_commit="d" * 40,
    )


def test_gpu_setup_lock_allows_exact_rearm_and_rejects_drift(tmp_path: Path) -> None:
    environment_path = tmp_path / "gpu_environment.json"
    write_json(
        environment_path,
        {
            "schema_version": 1,
            "pip_freeze": _pip_freeze(),
            "vllm_wheel": {"sha256": "a" * 64},
        },
    )
    lock_path = tmp_path / "setup_lock.json"
    created = create_gpu_setup_lock(
        path=lock_path,
        spec=_spec(),
        environment_manifest_path=environment_path,
        venv_python_path=sys.executable,
    )
    validated = validate_gpu_setup_lock(
        path=lock_path,
        expected_spec=_spec(),
        environment_manifest_path=environment_path,
        venv_python_path=sys.executable,
    )
    assert validated == created

    with pytest.raises(GpuBudgetGateError, match="pinned launch inputs"):
        validate_gpu_setup_lock(
            path=lock_path,
            expected_spec=_spec(wheel_hash="e" * 64),
            environment_manifest_path=environment_path,
            venv_python_path=sys.executable,
        )
    with pytest.raises(GpuBudgetGateError, match="overwrite"):
        create_gpu_setup_lock(
            path=lock_path,
            spec=_spec(),
            environment_manifest_path=environment_path,
            venv_python_path=sys.executable,
        )


def test_gpu_setup_uses_sha256_for_wheel_and_git_sha1_for_commits() -> None:
    assert _spec().transformers_commit == "c" * 40
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
