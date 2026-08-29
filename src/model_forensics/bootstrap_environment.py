"""Exact top-level distribution lock for the GPU bootstrap environment."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from typing import Any

from model_forensics.io import stable_hash

BOOTSTRAP_ENVIRONMENT_PROTOCOL = "exact-direct-bootstrap-v1"
BOOTSTRAP_CONSTRAINTS_PATH = "config/gpu_bootstrap_constraints.txt"
BOOTSTRAP_CONSTRAINTS_SHA256 = (
    "eac31466ffe0668c030ef18bb2e88444ad2ff264733ae2526787db30148db662"
)
BOOTSTRAP_DISTRIBUTION_VERSIONS: Mapping[str, str] = {
    "accelerate": "1.12.0",
    "huggingface-hub": "1.29.0",
    "jlens": "0.1.0",
    "matplotlib": "3.10.8",
    "numpy": "2.5.2",
    "pandas": "3.0.3",
    "pydantic": "2.12.5",
    "PyYAML": "6.0.3",
    "safetensors": "0.7.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.1",
    "sentence-transformers": "5.7.0",
    "setuptools": "80.9.0",
    "tokenizers": "0.23.1",
    "torch": "2.13.0",
    "transformers": "5.16.0.dev0",
    "vllm": "0.28.0",
    "wheel": "0.46.3",
}
BOOTSTRAP_DISTRIBUTION_LOCK_HASH = stable_hash(
    dict(sorted(BOOTSTRAP_DISTRIBUTION_VERSIONS.items()))
)


class BootstrapEnvironmentError(RuntimeError):
    """The installed top-level bootstrap distributions drifted from the lock."""


def capture_bootstrap_distribution_provenance(
    *,
    version_factory: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Fail closed unless every explicitly installed distribution is exact.

    This validates top-level distribution versions after pip resolves and
    installs them. It intentionally does not claim a transitive wheel hash
    lock; the pinned container and PyPI TLS remain the first-install boundary.
    """

    observed: dict[str, str] = {}
    for name in sorted(BOOTSTRAP_DISTRIBUTION_VERSIONS):
        try:
            observed[name] = str(version_factory(name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise BootstrapEnvironmentError(
                f"required GPU bootstrap distribution is missing: {name}"
            ) from exc
    expected = dict(sorted(BOOTSTRAP_DISTRIBUTION_VERSIONS.items()))
    if observed != expected:
        raise BootstrapEnvironmentError(
            f"GPU bootstrap distribution versions drifted: expected={expected!r}, "
            f"observed={observed!r}"
        )
    payload: dict[str, Any] = {
        "protocol_version": BOOTSTRAP_ENVIRONMENT_PROTOCOL,
        "constraints_path": BOOTSTRAP_CONSTRAINTS_PATH,
        "constraints_sha256": BOOTSTRAP_CONSTRAINTS_SHA256,
        "distribution_versions": observed,
        "distribution_lock_hash": BOOTSTRAP_DISTRIBUTION_LOCK_HASH,
        "artifact_scope": "exact_top_level_versions_post_install",
        "transitive_artifact_hashes_complete": False,
        "first_install_trust_boundary": "pinned_container_plus_pypi_tls",
    }
    payload["provenance_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "BOOTSTRAP_CONSTRAINTS_PATH",
    "BOOTSTRAP_CONSTRAINTS_SHA256",
    "BOOTSTRAP_DISTRIBUTION_LOCK_HASH",
    "BOOTSTRAP_DISTRIBUTION_VERSIONS",
    "BOOTSTRAP_ENVIRONMENT_PROTOCOL",
    "BootstrapEnvironmentError",
    "capture_bootstrap_distribution_provenance",
]
