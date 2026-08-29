"""Resolve and freeze remote model/lens provenance before generation."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_forensics.io import stable_hash, write_json


class ProvenanceError(RuntimeError):
    pass


def resolve_hub_revision(
    repo_id: str,
    *,
    requested_revision: str | None = None,
    token_env: str = "HF_TOKEN",
    info_loader: Callable[..., Any] | None = None,
) -> str:
    """Resolve a Hugging Face model repository to an immutable commit SHA."""

    if info_loader is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ProvenanceError("install the gpu extra to resolve Hub revisions") from exc
        info_loader = HfApi().model_info
    token = os.environ.get(token_env)
    info = info_loader(repo_id=repo_id, revision=requested_revision, token=token)
    sha = getattr(info, "sha", None)
    if not sha or len(sha) != 40:
        raise ProvenanceError(f"Hub did not return an immutable SHA for {repo_id}")
    return str(sha)


def environment_manifest(*, package_versions: Mapping[str, str] | None = None) -> dict[str, Any]:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": dict(package_versions or {}),
    }
    payload["environment_hash"] = stable_hash(payload)
    return payload


def write_provenance_manifest(
    destination: str | Path,
    *,
    model_id: str,
    model_revision: str,
    lens_repository: str,
    lens_revision: str,
    lens_files: Mapping[str, Mapping[str, Any]],
    sampling: Mapping[str, Any],
    package_versions: Mapping[str, str] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {"id": model_id, "revision": model_revision},
        "lens": {
            "repository": lens_repository,
            "revision": lens_revision,
            "files": dict(lens_files),
        },
        "sampling": dict(sampling),
        "environment": environment_manifest(package_versions=package_versions),
    }
    payload["manifest_hash"] = stable_hash(payload)
    return write_json(destination, payload)
