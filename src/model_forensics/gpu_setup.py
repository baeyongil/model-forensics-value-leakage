"""Authenticated reusable GPU software setup lock for phase re-arming."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forensics.bootstrap_environment import (
    BOOTSTRAP_CONSTRAINTS_SHA256,
    BOOTSTRAP_DISTRIBUTION_LOCK_HASH,
    BOOTSTRAP_DISTRIBUTION_VERSIONS,
    BOOTSTRAP_ENVIRONMENT_PROTOCOL,
)
from model_forensics.gpu_budget import GpuBudgetGateError, write_json_exclusive
from model_forensics.io import read_json, sha256_file, stable_hash
from model_forensics.semantic_backend import (
    SEMANTIC_DISTRIBUTION_VERSION,
    SEMANTIC_RUNTIME_PROTOCOL,
    SEMANTIC_STACK_LOCK_HASH,
    SEMANTIC_STACK_VERSIONS,
    SEMANTIC_WHEEL_SHA256,
    SEMANTIC_WHEEL_URL,
    TRANSFORMERS_COMMIT,
)

GPU_SETUP_PROTOCOL = "reusable-gpu-setup-v4"
_RAW_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACED_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
QWEN4B_SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN4B_SMOKE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@dataclass(frozen=True, slots=True)
class GpuSetupSpec:
    container_image_digest: str
    vllm_wheel_url: str
    vllm_wheel_sha256: str
    transformers_commit: str
    jlens_commit: str
    semantic_wheel_url: str = SEMANTIC_WHEEL_URL
    semantic_wheel_sha256: str = SEMANTIC_WHEEL_SHA256
    semantic_distribution_version: str = SEMANTIC_DISTRIBUTION_VERSION
    semantic_stack_lock_hash: str = SEMANTIC_STACK_LOCK_HASH
    bootstrap_constraints_sha256: str = BOOTSTRAP_CONSTRAINTS_SHA256
    bootstrap_distribution_lock_hash: str = BOOTSTRAP_DISTRIBUTION_LOCK_HASH

    def __post_init__(self) -> None:
        if _RAW_HASH_RE.fullmatch(self.vllm_wheel_sha256) is None:
            raise ValueError("vllm_wheel_sha256 must be 64 lowercase hexadecimal characters")
        if _RAW_HASH_RE.fullmatch(self.semantic_wheel_sha256) is None:
            raise ValueError("semantic_wheel_sha256 must be 64 lowercase hexadecimal characters")
        for field in ("transformers_commit", "jlens_commit"):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, field)) is None:
                raise ValueError(f"{field} must be a 40-character Git commit")
        if self.transformers_commit != TRANSFORMERS_COMMIT:
            raise ValueError("transformers_commit disagrees with the semantic source lock")
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", self.container_image_digest):
            raise ValueError("container image must be pinned by SHA-256 digest")
        if not self.vllm_wheel_url.startswith("https://") or not self.vllm_wheel_url.endswith(
            ".whl"
        ):
            raise ValueError("vLLM wheel URL must be immutable credential-free HTTPS")
        if (
            self.semantic_wheel_url != SEMANTIC_WHEEL_URL
            or self.semantic_wheel_sha256 != SEMANTIC_WHEEL_SHA256
            or self.semantic_distribution_version != SEMANTIC_DISTRIBUTION_VERSION
            or self.semantic_stack_lock_hash != SEMANTIC_STACK_LOCK_HASH
        ):
            raise ValueError("semantic runtime spec disagrees with the compiled artifact lock")
        if (
            self.bootstrap_constraints_sha256 != BOOTSTRAP_CONSTRAINTS_SHA256
            or self.bootstrap_distribution_lock_hash != BOOTSTRAP_DISTRIBUTION_LOCK_HASH
        ):
            raise ValueError("bootstrap environment spec disagrees with the compiled lock")


def _environment_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
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
    semantic_wheel = payload.get("semantic_wheel")
    if (
        not isinstance(semantic_wheel, dict)
        or _RAW_HASH_RE.fullmatch(str(semantic_wheel.get("sha256"))) is None
        or semantic_wheel.get("filename") != SEMANTIC_WHEEL_URL.rsplit("/", 1)[-1]
    ):
        raise GpuBudgetGateError("GPU environment semantic wheel identity is malformed")
    semantic_runtime = payload.get("semantic_runtime")
    if not isinstance(semantic_runtime, dict):
        raise GpuBudgetGateError("GPU environment semantic runtime is absent")
    runtime_hash = semantic_runtime.get("runtime_hash")
    unsigned_runtime = {
        key: value for key, value in semantic_runtime.items() if key != "runtime_hash"
    }
    if (
        not isinstance(runtime_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(runtime_hash) is None
        or runtime_hash != stable_hash(unsigned_runtime)
        or semantic_runtime.get("protocol_version") != SEMANTIC_RUNTIME_PROTOCOL
        or semantic_runtime.get("distribution_artifact")
        != {
            "name": "sentence-transformers",
            "version": SEMANTIC_DISTRIBUTION_VERSION,
            "wheel_url": SEMANTIC_WHEEL_URL,
            "wheel_sha256": SEMANTIC_WHEEL_SHA256,
        }
        or semantic_runtime.get("stack_versions")
        != dict(sorted(SEMANTIC_STACK_VERSIONS.items()))
        or semantic_runtime.get("stack_lock_hash") != SEMANTIC_STACK_LOCK_HASH
    ):
        raise GpuBudgetGateError("GPU environment semantic runtime is malformed or drifted")
    distributions = semantic_runtime.get("distributions")
    if not isinstance(distributions, dict) or set(distributions) != set(
        SEMANTIC_STACK_VERSIONS
    ):
        raise GpuBudgetGateError("GPU environment semantic distribution inventory drifted")
    for name, expected_version in SEMANTIC_STACK_VERSIONS.items():
        row = distributions.get(name)
        if (
            not isinstance(row, dict)
            or row.get("name") != name
            or row.get("version") != expected_version
            or _NAMESPACED_HASH_RE.fullmatch(str(row.get("metadata_sha256"))) is None
            or _NAMESPACED_HASH_RE.fullmatch(str(row.get("record_sha256"))) is None
        ):
            raise GpuBudgetGateError(
                f"GPU environment semantic distribution identity drifted: {name}"
            )
    semantic_distribution = distributions["sentence-transformers"]
    transformers_distribution = distributions["transformers"]
    if (
        _NAMESPACED_HASH_RE.fullmatch(
            str(semantic_distribution.get("direct_url_sha256"))
        )
        is None
        or semantic_distribution.get("source")
        != {
            "wheel_filename": SEMANTIC_WHEEL_URL.rsplit("/", 1)[-1],
            "archive_sha256": SEMANTIC_WHEEL_SHA256,
        }
        or _NAMESPACED_HASH_RE.fullmatch(
            str(transformers_distribution.get("direct_url_sha256"))
        )
        is None
        or transformers_distribution.get("source")
        != {
            "vcs": "git",
            "requested_revision": TRANSFORMERS_COMMIT,
            "commit_id": TRANSFORMERS_COMMIT,
        }
    ):
        raise GpuBudgetGateError("GPU environment semantic source artifact drifted")
    installed_aggregate = semantic_runtime.get("verified_installed_files")
    if (
        not isinstance(installed_aggregate, dict)
        or installed_aggregate.get("algorithm") != "sha256"
        or not isinstance(installed_aggregate.get("verified_file_count"), int)
        or installed_aggregate["verified_file_count"] <= 0
        or not isinstance(installed_aggregate.get("verified_size_bytes"), int)
        or installed_aggregate["verified_size_bytes"] < 0
        or _NAMESPACED_HASH_RE.fullmatch(
            str(installed_aggregate.get("manifest_hash"))
        )
        is None
    ):
        raise GpuBudgetGateError("GPU environment installed-file aggregate is malformed")
    for name, row in distributions.items():
        installed = row.get("installed_files") if isinstance(row, dict) else None
        if (
            not isinstance(installed, dict)
            or installed.get("algorithm") != "sha256"
            or not isinstance(installed.get("verified_file_count"), int)
            or installed["verified_file_count"] <= 0
            or not isinstance(installed.get("verified_size_bytes"), int)
            or installed["verified_size_bytes"] < 0
            or _NAMESPACED_HASH_RE.fullmatch(str(installed.get("manifest_hash"))) is None
        ):
            raise GpuBudgetGateError(
                f"GPU environment installed-file evidence drifted: {name}"
            )
    bootstrap = payload.get("bootstrap_environment")
    if not isinstance(bootstrap, dict):
        raise GpuBudgetGateError("GPU bootstrap environment provenance is absent")
    bootstrap_hash = bootstrap.get("provenance_hash")
    unsigned_bootstrap = {
        key: value for key, value in bootstrap.items() if key != "provenance_hash"
    }
    if (
        bootstrap.get("protocol_version") != BOOTSTRAP_ENVIRONMENT_PROTOCOL
        or bootstrap.get("constraints_sha256") != BOOTSTRAP_CONSTRAINTS_SHA256
        or bootstrap.get("distribution_versions")
        != dict(sorted(BOOTSTRAP_DISTRIBUTION_VERSIONS.items()))
        or bootstrap.get("distribution_lock_hash") != BOOTSTRAP_DISTRIBUTION_LOCK_HASH
        or bootstrap.get("transitive_artifact_hashes_complete") is not False
        or bootstrap.get("first_install_trust_boundary")
        != "pinned_container_plus_pypi_tls"
        or not isinstance(bootstrap_hash, str)
        or bootstrap_hash != stable_hash(unsigned_bootstrap)
    ):
        raise GpuBudgetGateError("GPU bootstrap environment provenance drifted")
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


def _live_semantic_runtime(venv_python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import json; "
                "from model_forensics.semantic_backend import "
                "capture_semantic_runtime_provenance; "
                "print(json.dumps(capture_semantic_runtime_provenance(), "
                "sort_keys=True, separators=(',', ':')))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise GpuBudgetGateError("cannot authenticate reusable semantic runtime")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        raise GpuBudgetGateError("live semantic runtime emitted malformed provenance") from exc
    if not isinstance(payload, dict):
        raise GpuBudgetGateError("live semantic runtime provenance is not a mapping")
    return payload


def _live_bootstrap_environment(venv_python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import json; "
                "from model_forensics.bootstrap_environment import "
                "capture_bootstrap_distribution_provenance; "
                "print(json.dumps(capture_bootstrap_distribution_provenance(), "
                "sort_keys=True, separators=(',', ':')))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise GpuBudgetGateError("cannot authenticate reusable bootstrap environment")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        raise GpuBudgetGateError("live bootstrap environment emitted malformed provenance") from exc
    if not isinstance(payload, dict):
        raise GpuBudgetGateError("live bootstrap environment provenance is not a mapping")
    return payload


def _qwen4b_smoke_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise GpuBudgetGateError("Qwen4B smoke manifest is malformed")
    if (
        payload.get("status") != "passed"
        or payload.get("protocol_version") != "qwen4b-bounded-integration-smoke-v2"
        or payload.get("scope") != "bounded_nonprimary_qwen4b_integration_smoke"
        or payload.get("experimental_sample") is not False
        or payload.get("primary_eligible") is not False
        or payload.get("synthetic_analysis_fixture") is not True
        or payload.get("paid_api_calls") != 0
    ):
        raise GpuBudgetGateError("Qwen4B smoke did not pass the bounded compatibility gate")
    model = payload.get("model")
    if not isinstance(model, dict) or model != {
        "id": QWEN4B_SMOKE_MODEL_ID,
        "revision": QWEN4B_SMOKE_MODEL_REVISION,
    }:
        raise GpuBudgetGateError("Qwen4B smoke used the wrong pinned model identity")
    registered_prefixes = payload.get("registered_prefixes")
    forced_append = payload.get("forced_append_check")
    continuations = payload.get("raw_prefix_continuations")
    if (
        not isinstance(registered_prefixes, dict)
        or any(
            not isinstance(registered_prefixes.get(arm), dict)
            or registered_prefixes[arm].get("exact_original_ids_reused") is not True
            for arm in ("retain", "resample")
        )
        or not isinstance(forced_append, dict)
        or forced_append.get("immutable_prefix_preserved") is not True
        or not isinstance(continuations, dict)
        or any(
            not isinstance(continuations.get(arm), dict)
            or continuations[arm].get("prompt_ids_exact") is not True
            for arm in ("retain", "resample")
        )
    ):
        raise GpuBudgetGateError("Qwen4B smoke lacks exact raw-prefix evidence")
    fixture = payload.get("deterministic_local_fixture")
    grid = payload.get("lens_probe_grid")
    handoff = payload.get("analysis_evidence_handoff")
    if (
        not isinstance(fixture, dict)
        or fixture.get("synthetic") is not True
        or fixture.get("primary_eligible") is not False
        or not isinstance(grid, dict)
        or grid.get("probe_cell_count") != 15
        or not isinstance(grid.get("transport_boundary"), dict)
        or grid["transport_boundary"].get("fabricated_lens_record_count") != 0
        or grid["transport_boundary"].get("activation_transport_executed") is not False
        or not isinstance(handoff, dict)
        or handoff.get("analysis_ingest_allowed") is not False
        or handoff.get("primary_eligible") is not False
    ):
        raise GpuBudgetGateError("Qwen4B smoke lacks non-primary integration evidence")
    claimed_hash = payload.get("manifest_hash")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if (
        not isinstance(claimed_hash, str)
        or _NAMESPACED_HASH_RE.fullmatch(claimed_hash) is None
        or claimed_hash != stable_hash(unsigned)
    ):
        raise GpuBudgetGateError("Qwen4B smoke manifest hash mismatch")
    return payload


def create_gpu_setup_lock(
    *,
    path: str | Path,
    spec: GpuSetupSpec,
    environment_manifest_path: str | Path,
    qwen4b_smoke_manifest_path: str | Path,
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
    if environment["semantic_wheel"]["sha256"] != spec.semantic_wheel_sha256:
        raise GpuBudgetGateError("captured GPU environment has the wrong semantic wheel")
    live_semantic_runtime = _live_semantic_runtime(Path(venv_python_path))
    if live_semantic_runtime != environment["semantic_runtime"]:
        raise GpuBudgetGateError("captured semantic runtime disagrees with installed packages")
    live_bootstrap_environment = _live_bootstrap_environment(Path(venv_python_path))
    if live_bootstrap_environment != environment["bootstrap_environment"]:
        raise GpuBudgetGateError("captured bootstrap environment disagrees with installed packages")
    smoke_path = Path(qwen4b_smoke_manifest_path)
    smoke = _qwen4b_smoke_payload(smoke_path)
    payload = {
        "schema_version": 4,
        "protocol_version": GPU_SETUP_PROTOCOL,
        "spec": asdict(spec),
        "environment_manifest_sha256": sha256_file(environment_path),
        "pip_freeze_hash": stable_hash(live_freeze),
        "semantic_runtime_hash": environment["semantic_runtime"]["runtime_hash"],
        "bootstrap_environment_hash": environment["bootstrap_environment"][
            "provenance_hash"
        ],
        "qwen4b_smoke_manifest_sha256": sha256_file(smoke_path),
        "qwen4b_smoke_manifest_hash": smoke["manifest_hash"],
    }
    payload["record_hash"] = stable_hash(payload)
    write_json_exclusive(path, payload)
    return payload


def validate_gpu_setup_lock(
    *,
    path: str | Path,
    expected_spec: GpuSetupSpec,
    environment_manifest_path: str | Path,
    qwen4b_smoke_manifest_path: str | Path,
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
        "semantic_runtime_hash",
        "bootstrap_environment_hash",
        "qwen4b_smoke_manifest_sha256",
        "qwen4b_smoke_manifest_hash",
        "record_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise GpuBudgetGateError("GPU setup lock has an unexpected schema")
    if payload.get("schema_version") != 4 or payload.get("protocol_version") != GPU_SETUP_PROTOCOL:
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
    if environment["semantic_wheel"]["sha256"] != expected_spec.semantic_wheel_sha256:
        raise GpuBudgetGateError("GPU setup environment semantic wheel drifted")
    live_semantic_runtime = _live_semantic_runtime(Path(venv_python_path))
    if live_semantic_runtime != environment["semantic_runtime"]:
        raise GpuBudgetGateError("reusable semantic package metadata drifted")
    if payload.get("semantic_runtime_hash") != live_semantic_runtime["runtime_hash"]:
        raise GpuBudgetGateError("GPU setup semantic runtime identity drifted")
    live_bootstrap_environment = _live_bootstrap_environment(Path(venv_python_path))
    if live_bootstrap_environment != environment["bootstrap_environment"]:
        raise GpuBudgetGateError("reusable bootstrap environment drifted")
    if (
        payload.get("bootstrap_environment_hash")
        != live_bootstrap_environment["provenance_hash"]
    ):
        raise GpuBudgetGateError("GPU setup bootstrap environment identity drifted")
    smoke_path = Path(qwen4b_smoke_manifest_path)
    smoke = _qwen4b_smoke_payload(smoke_path)
    if payload.get("qwen4b_smoke_manifest_sha256") != sha256_file(smoke_path):
        raise GpuBudgetGateError("Qwen4B smoke manifest file hash mismatch")
    if payload.get("qwen4b_smoke_manifest_hash") != smoke["manifest_hash"]:
        raise GpuBudgetGateError("Qwen4B smoke manifest identity drifted")
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
