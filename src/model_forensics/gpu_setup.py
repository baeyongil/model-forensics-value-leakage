"""Authenticated reusable GPU software setup lock for phase re-arming."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forensics.gpu_budget import GpuBudgetGateError, write_json_exclusive
from model_forensics.io import read_json, sha256_file, stable_hash

GPU_SETUP_PROTOCOL = "reusable-gpu-setup-v1"
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class GpuSetupSpec:
    container_image_digest: str
    vllm_wheel_url: str
    vllm_wheel_sha256: str
    transformers_commit: str
    jlens_commit: str

    def __post_init__(self) -> None:
        if _RAW_HASH_RE.fullmatch(self.vllm_wheel_sha256) is None:
            raise ValueError("vllm_wheel_sha256 must be 64 lowercase hexadecimal characters")
        for field in ("transformers_commit", "jlens_commit"):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, field)) is None:
                raise ValueError(f"{field} must be a 40-character Git commit")
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", self.container_image_digest):
            raise ValueError("container image must be pinned by SHA-256 digest")
        if not self.vllm_wheel_url.startswith("https://") or not self.vllm_wheel_url.endswith(
            ".whl"
        ):
            raise ValueError("vLLM wheel URL must be immutable credential-free HTTPS")


def _environment_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise GpuBudgetGateError("GPU environment manifest is malformed")
    pip_freeze = payload.get("pip_freeze")
    if (
        not isinstance(pip_freeze, list)
        or not all(isinstance(item, str) and item for item in pip_freeze)
        or pip_freeze != sorted(pip_freeze)
    ):
        raise GpuBudgetGateError("GPU environment pip freeze is malformed")
    wheel = payload.get("vllm_wheel")
    if not isinstance(wheel, dict) or _RAW_HASH_RE.fullmatch(str(wheel.get("sha256"))) is None:
        raise GpuBudgetGateError("GPU environment vLLM wheel identity is malformed")
    return payload


def _live_pip_freeze(venv_python: Path) -> list[str]:
    if not venv_python.is_file():
        raise GpuBudgetGateError("reusable GPU virtual environment Python is missing")
    completed = subprocess.run(
        [str(venv_python), "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise GpuBudgetGateError("cannot verify reusable GPU virtual environment")
    return sorted(line for line in completed.stdout.splitlines() if line)


def create_gpu_setup_lock(
    *,
    path: str | Path,
    spec: GpuSetupSpec,
    environment_manifest_path: str | Path,
    venv_python_path: str | Path,
) -> dict[str, Any]:
    """Create a non-overwritable lock after a successful first installation."""

    environment_path = Path(environment_manifest_path)
    environment = _environment_payload(environment_path)
    live_freeze = _live_pip_freeze(Path(venv_python_path))
    if live_freeze != environment["pip_freeze"]:
        raise GpuBudgetGateError("captured GPU environment disagrees with installed packages")
    if environment["vllm_wheel"]["sha256"] != spec.vllm_wheel_sha256:
        raise GpuBudgetGateError("captured GPU environment has the wrong vLLM wheel")
    payload = {
        "schema_version": 1,
        "protocol_version": GPU_SETUP_PROTOCOL,
        "spec": asdict(spec),
        "environment_manifest_sha256": sha256_file(environment_path),
        "pip_freeze_hash": stable_hash(live_freeze),
    }
    payload["record_hash"] = stable_hash(payload)
    write_json_exclusive(path, payload)
    return payload


def validate_gpu_setup_lock(
    *,
    path: str | Path,
    expected_spec: GpuSetupSpec,
    environment_manifest_path: str | Path,
    venv_python_path: str | Path,
) -> dict[str, Any]:
    """Fail closed if a reusable phase setup or installed packages drifted."""

    payload = read_json(path)
    expected_keys = {
        "schema_version",
        "protocol_version",
        "spec",
        "environment_manifest_sha256",
        "pip_freeze_hash",
        "record_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise GpuBudgetGateError("GPU setup lock has an unexpected schema")
    if payload.get("schema_version") != 1 or payload.get("protocol_version") != GPU_SETUP_PROTOCOL:
        raise GpuBudgetGateError("GPU setup lock protocol is unsupported")
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if payload.get("record_hash") != stable_hash(unsigned):
        raise GpuBudgetGateError("GPU setup lock content hash mismatch")
    if payload.get("spec") != asdict(expected_spec):
        raise GpuBudgetGateError("GPU setup lock disagrees with pinned launch inputs")
    environment_path = Path(environment_manifest_path)
    if payload.get("environment_manifest_sha256") != sha256_file(environment_path):
        raise GpuBudgetGateError("GPU setup environment manifest hash mismatch")
    environment = _environment_payload(environment_path)
    if environment["vllm_wheel"]["sha256"] != expected_spec.vllm_wheel_sha256:
        raise GpuBudgetGateError("GPU setup environment vLLM wheel drifted")
    live_freeze = _live_pip_freeze(Path(venv_python_path))
    if payload.get("pip_freeze_hash") != stable_hash(live_freeze):
        raise GpuBudgetGateError("reusable GPU virtual environment package set drifted")
    if live_freeze != environment["pip_freeze"]:
        raise GpuBudgetGateError("reusable GPU package set disagrees with its environment manifest")
    return payload


__all__ = [
    "GPU_SETUP_PROTOCOL",
    "GpuSetupSpec",
    "create_gpu_setup_lock",
    "validate_gpu_setup_lock",
]
