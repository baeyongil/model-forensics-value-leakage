"""Content-addressed sentence-transformer adapter for the divergence gate.

The semantic cutoff changes which continuations enter the primary estimand, so
"whatever pip resolved" is not adequate provenance. This module authenticates
the inference-critical distribution set and model/tokenizer Hub revision before
the first primary embedding is produced.
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import hmac
import importlib.metadata
import io
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from model_forensics.io import stable_hash

SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEMANTIC_DISTRIBUTION_VERSION = "5.7.0"
SEMANTIC_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/e8/c8/"
    "f63d99e354532f5b83e735dd1e001bda92495fbfde934f65d924abf2b071/"
    "sentence_transformers-5.7.0-py3-none-any.whl"
)
SEMANTIC_WHEEL_SHA256 = "b78141da3d8137e70d965866e2ca43190b9266f3d4d8752e250ded75e7136730"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"

# Distributions that can change dense inference or the exact model/tokenizer
# loader. Versions were resolved against the pinned vLLM 0.28.0 / Torch 2.13.0
# environment. Installed METADATA and RECORD files are hashed as a second layer.
SEMANTIC_STACK_VERSIONS: Mapping[str, str] = {
    "huggingface-hub": "1.29.0",
    "numpy": "2.5.2",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.1",
    "sentence-transformers": SEMANTIC_DISTRIBUTION_VERSION,
    "tokenizers": "0.23.1",
    "torch": "2.13.0",
    "transformers": "5.16.0.dev0",
}
SEMANTIC_STACK_LOCK_HASH = stable_hash(dict(sorted(SEMANTIC_STACK_VERSIONS.items())))
SEMANTIC_RUNTIME_PROTOCOL = "pinned-sentence-transformer-runtime-v2"


class SemanticRuntimeError(RuntimeError):
    """The local embedding runtime cannot support primary eligibility."""


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verified_record_files(
    *,
    distribution: Any,
    distribution_name: str,
    record: str,
    installation_root: Path,
) -> dict[str, Any]:
    """Authenticate each hash-bearing RECORD file inside the install prefix."""

    try:
        root = Path(os.path.abspath(installation_root))
        root_stat = root.lstat()
    except OSError as exc:
        raise SemanticRuntimeError("semantic installation root is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise SemanticRuntimeError("semantic installation root must be a real directory")
    root_resolved = root.resolve(strict=True)
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    try:
        rows = csv.reader(io.StringIO(record), strict=True)
        for row in rows:
            if len(row) != 3:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD is malformed: {distribution_name}"
                )
            record_path, hash_spec, size_text = row
            if (
                not record_path
                or "\\" in record_path
                or "\x00" in record_path
                or PurePosixPath(record_path).is_absolute()
            ):
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD path is unsafe: {distribution_name}"
                )
            if record_path in seen:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD has a duplicate path: {distribution_name}"
                )
            seen.add(record_path)
            if not hash_spec:
                continue
            algorithm, separator, encoded_digest = hash_spec.partition("=")
            if algorithm != "sha256" or separator != "=" or not encoded_digest:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD hash is not SHA-256: {distribution_name}"
                )
            try:
                expected_digest = base64.b64decode(
                    encoded_digest + "=" * (-len(encoded_digest) % 4),
                    altchars=b"-_",
                    validate=True,
                )
                expected_size = int(size_text)
            except (binascii.Error, ValueError) as exc:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD hash/size is malformed: {distribution_name}"
                ) from exc
            if len(expected_digest) != hashlib.sha256().digest_size or expected_size < 0:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD hash/size is malformed: {distribution_name}"
                )
            try:
                located = Path(distribution.locate_file(record_path))
            except (AttributeError, OSError, TypeError) as exc:
                raise SemanticRuntimeError(
                    f"semantic distribution cannot locate a RECORD file: {distribution_name}"
                ) from exc
            candidate = Path(os.path.abspath(located))
            if not candidate.is_relative_to(root):
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD escapes the install root: {distribution_name}"
                )
            current = root
            try:
                for component in candidate.relative_to(root).parts:
                    current = current / component
                    current_stat = current.lstat()
                    if stat.S_ISLNK(current_stat.st_mode):
                        raise SemanticRuntimeError(
                            f"semantic distribution RECORD traverses a symlink: {distribution_name}"
                        )
                resolved = candidate.resolve(strict=True)
                candidate_stat = candidate.stat()
            except FileNotFoundError as exc:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD file is missing: {distribution_name}"
                ) from exc
            except OSError as exc:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD file cannot be authenticated: {distribution_name}"
                ) from exc
            if not resolved.is_relative_to(root_resolved) or not stat.S_ISREG(
                candidate_stat.st_mode
            ):
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD file is unsafe: {distribution_name}"
                )
            digest = hashlib.sha256()
            try:
                with candidate.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD file cannot be read: {distribution_name}"
                ) from exc
            if candidate_stat.st_size != expected_size or not hmac.compare_digest(
                digest.digest(), expected_digest
            ):
                raise SemanticRuntimeError(
                    f"semantic distribution RECORD file hash/size drifted: {distribution_name}"
                )
            verified.append(
                {
                    "path": record_path,
                    "sha256": "sha256:" + digest.hexdigest(),
                    "size_bytes": expected_size,
                }
            )
    except csv.Error as exc:
        raise SemanticRuntimeError(
            f"semantic distribution RECORD is malformed: {distribution_name}"
        ) from exc
    if not verified:
        raise SemanticRuntimeError(
            f"semantic distribution RECORD has no SHA-256-bearing files: {distribution_name}"
        )
    verified.sort(key=lambda row: str(row["path"]))
    return {
        "algorithm": "sha256",
        "verified_file_count": len(verified),
        "verified_size_bytes": sum(int(row["size_bytes"]) for row in verified),
        "manifest_hash": stable_hash({"files": verified}),
    }


def _distribution_record(
    name: str,
    *,
    distribution_factory: Callable[[str], Any] = importlib.metadata.distribution,
    installation_root: Path | None = None,
) -> dict[str, Any]:
    try:
        distribution = distribution_factory(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SemanticRuntimeError(f"semantic runtime distribution is missing: {name}") from exc
    version = str(distribution.version)
    metadata = distribution.read_text("METADATA")
    record = distribution.read_text("RECORD")
    if not metadata or not record:
        raise SemanticRuntimeError(f"semantic distribution lacks METADATA/RECORD: {name}")
    direct_url = distribution.read_text("direct_url.json")
    installer = distribution.read_text("INSTALLER")
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "metadata_sha256": _text_sha256(metadata),
        "record_sha256": _text_sha256(record),
        "direct_url_sha256": _text_sha256(direct_url) if direct_url else None,
        "installer": installer.strip() if installer else None,
        "installed_files": _verified_record_files(
            distribution=distribution,
            distribution_name=name,
            record=record,
            installation_root=installation_root or Path(sys.prefix),
        ),
    }
    if name == "transformers":
        if not direct_url:
            raise SemanticRuntimeError("transformers lacks its PEP-610 source identity")
        try:
            source = json.loads(direct_url)
        except json.JSONDecodeError as exc:
            raise SemanticRuntimeError("transformers direct_url.json is malformed") from exc
        vcs = source.get("vcs_info") if isinstance(source, dict) else None
        if not isinstance(vcs, dict):
            raise SemanticRuntimeError("transformers was not installed from the pinned VCS source")
        payload["source"] = {
            "vcs": vcs.get("vcs"),
            "requested_revision": vcs.get("requested_revision"),
            "commit_id": vcs.get("commit_id"),
        }
    elif name == "sentence-transformers":
        if not direct_url:
            raise SemanticRuntimeError(
                "sentence-transformers lacks its PEP-610 wheel identity"
            )
        try:
            source = json.loads(direct_url)
        except json.JSONDecodeError as exc:
            raise SemanticRuntimeError(
                "sentence-transformers direct_url.json is malformed"
            ) from exc
        archive = source.get("archive_info") if isinstance(source, dict) else None
        source_url = source.get("url") if isinstance(source, dict) else None
        hashes = archive.get("hashes") if isinstance(archive, dict) else None
        archive_sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        if not isinstance(source_url, str) or not source_url:
            raise SemanticRuntimeError("sentence-transformers wheel source URL is absent")
        if PurePath(source_url.removesuffix("/")).name != PurePath(SEMANTIC_WHEEL_URL).name:
            raise SemanticRuntimeError("sentence-transformers wheel filename drifted")
        if archive_sha256 != SEMANTIC_WHEEL_SHA256:
            raise SemanticRuntimeError("sentence-transformers wheel archive hash drifted")
        payload["source"] = {
            "wheel_filename": PurePath(source_url.removesuffix("/")).name,
            "archive_sha256": archive_sha256,
        }
    return payload


def capture_semantic_runtime_provenance(
    *,
    distribution_factory: Callable[[str], Any] = importlib.metadata.distribution,
    installation_root: Path | None = None,
) -> dict[str, Any]:
    """Capture and validate exact installed distribution identities.

    This imports no ML package. It fails closed on a missing version, METADATA,
    RECORD, or VCS commit and can therefore be used by setup-lock validation.
    """

    distributions = {
        name: _distribution_record(
            name,
            distribution_factory=distribution_factory,
            installation_root=installation_root,
        )
        for name in sorted(SEMANTIC_STACK_VERSIONS)
    }
    observed_versions = {name: row["version"] for name, row in distributions.items()}
    if observed_versions != dict(sorted(SEMANTIC_STACK_VERSIONS.items())):
        raise SemanticRuntimeError(
            "semantic runtime versions drifted from the frozen stack: "
            f"expected={dict(sorted(SEMANTIC_STACK_VERSIONS.items()))!r}, "
            f"observed={observed_versions!r}"
        )
    source = distributions["transformers"].get("source")
    if not isinstance(source, Mapping) or source != {
        "vcs": "git",
        "requested_revision": TRANSFORMERS_COMMIT,
        "commit_id": TRANSFORMERS_COMMIT,
    }:
        raise SemanticRuntimeError("transformers source commit drifted from the GPU lock")
    payload: dict[str, Any] = {
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
    }
    installed_file_manifests = {
        name: row["installed_files"] for name, row in sorted(distributions.items())
    }
    payload["verified_installed_files"] = {
        "algorithm": "sha256",
        "verified_file_count": sum(
            int(row["verified_file_count"]) for row in installed_file_manifests.values()
        ),
        "verified_size_bytes": sum(
            int(row["verified_size_bytes"]) for row in installed_file_manifests.values()
        ),
        "manifest_hash": stable_hash(installed_file_manifests),
    }
    payload["runtime_hash"] = stable_hash(payload)
    return payload


def _revision_candidates(value: Any) -> tuple[str, ...]:
    candidates: set[str] = set()
    for candidate in (
        getattr(value, "_commit_hash", None),
        getattr(value, "commit_hash", None),
    ):
        if isinstance(candidate, str):
            candidates.add(candidate)
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        for key in ("_commit_hash", "commit_hash", "revision", "name_or_path"):
            candidate = init_kwargs.get(key)
            if isinstance(candidate, str):
                candidates.update(_revision_strings(candidate))
    for candidate in (
        getattr(value, "name_or_path", None),
        getattr(value, "_name_or_path", None),
    ):
        if isinstance(candidate, str):
            candidates.update(_revision_strings(candidate))
    return tuple(sorted(candidates))


def _revision_strings(value: str) -> set[str]:
    """Extract an exact revision from a Hub value or snapshot cache path."""

    result = {value}
    result.update(
        match.group(1)
        for match in re.finditer(r"(?:^|[/\\])snapshots[/\\]([0-9a-f]{40})(?:[/\\]|$)", value)
    )
    return result


def _model_runtime_provenance(model: Any) -> dict[str, Any]:
    first_module_factory = getattr(model, "_first_module", None)
    first_module = first_module_factory() if callable(first_module_factory) else None
    auto_model = getattr(first_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    tokenizer = getattr(first_module, "tokenizer", None) or getattr(model, "tokenizer", None)
    if auto_model is None or config is None or tokenizer is None:
        raise SemanticRuntimeError(
            "sentence-transformer did not expose its model/config/tokenizer revision evidence"
        )
    model_revisions = tuple(sorted(set(_revision_candidates(config) + _revision_candidates(auto_model))))
    tokenizer_revisions = _revision_candidates(tokenizer)
    if SEMANTIC_MODEL_REVISION not in model_revisions:
        raise SemanticRuntimeError("semantic model config does not attest the frozen Hub revision")
    if SEMANTIC_MODEL_REVISION not in tokenizer_revisions:
        raise SemanticRuntimeError("semantic tokenizer does not attest the frozen Hub revision")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    tokenizer_payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "revision_candidates": list(tokenizer_revisions),
        "vocab_size": vocab_size if type(vocab_size) is int else None,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    payload: dict[str, Any] = {
        "model_id": SEMANTIC_MODEL_ID,
        "model_revision": SEMANTIC_MODEL_REVISION,
        "model_class": f"{type(auto_model).__module__}.{type(auto_model).__qualname__}",
        "model_revision_candidates": list(model_revisions),
        "tokenizer": tokenizer_payload,
        "max_seq_length": getattr(model, "max_seq_length", None),
        "truncate_dim": getattr(model, "truncate_dim", None),
    }
    payload["model_tokenizer_runtime_hash"] = stable_hash(payload)
    return payload


class PinnedSentenceTransformerEmbedder:
    """SentenceTransformer wrapper that proves eligibility before encoding."""

    def __init__(
        self,
        *,
        model_id: str = SEMANTIC_MODEL_ID,
        revision: str = SEMANTIC_MODEL_REVISION,
        device: str = "cpu",
        model_factory: Callable[..., Any] | None = None,
        distribution_factory: Callable[[str], Any] = importlib.metadata.distribution,
        installation_root: Path | None = None,
        require_primary_eligibility: bool | None = None,
    ) -> None:
        if model_id != SEMANTIC_MODEL_ID or revision != SEMANTIC_MODEL_REVISION:
            raise ValueError("semantic divergence model must use the frozen ID and revision")
        injected_factory = model_factory is not None
        if require_primary_eligibility is None:
            require_primary_eligibility = not injected_factory
        if model_factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for semantic divergence"
                ) from exc
            model_factory = SentenceTransformer
        self._model = model_factory(
            model_id,
            revision=revision,
            device=device,
            trust_remote_code=False,
        )
        if require_primary_eligibility:
            stack = capture_semantic_runtime_provenance(
                distribution_factory=distribution_factory,
                installation_root=installation_root,
            )
            model_runtime = _model_runtime_provenance(self._model)
            primary_eligible = True
            ineligibility_reason = None
        else:
            stack = None
            model_runtime = None
            primary_eligible = False
            ineligibility_reason = "injected_or_explicit_nonprimary_runtime"
        self._require_primary_eligibility = require_primary_eligibility
        self._provenance: dict[str, Any] = {
            "runtime_kind": "sentence_transformer",
            "adapter_protocol": SEMANTIC_RUNTIME_PROTOCOL,
            "model_id": model_id,
            "revision": revision,
            "device": device,
            "trust_remote_code": False,
            "encoding": {
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            },
            "distribution_runtime": stack,
            "model_tokenizer_runtime": model_runtime,
            "primary_eligible": primary_eligible,
            "primary_ineligibility_reason": ineligibility_reason,
        }
        self._provenance["provenance_hash"] = stable_hash(self._provenance)

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def assert_primary_eligible(self) -> None:
        if self._provenance.get("primary_eligible") is not True:
            raise SemanticRuntimeError("semantic runtime is explicitly non-primary")
        claimed = self._provenance.get("provenance_hash")
        unsigned = {
            key: value for key, value in self._provenance.items() if key != "provenance_hash"
        }
        if claimed != stable_hash(unsigned):
            raise SemanticRuntimeError("semantic runtime provenance hash drifted")

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("semantic inputs must be nonempty strings")
        if self._require_primary_eligibility:
            self.assert_primary_eligible()
        values = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("sentence-transformer returned an invalid embedding batch")
        return values


__all__ = [
    "SEMANTIC_DISTRIBUTION_VERSION",
    "SEMANTIC_MODEL_ID",
    "SEMANTIC_MODEL_REVISION",
    "SEMANTIC_RUNTIME_PROTOCOL",
    "SEMANTIC_STACK_LOCK_HASH",
    "SEMANTIC_STACK_VERSIONS",
    "SEMANTIC_WHEEL_SHA256",
    "SEMANTIC_WHEEL_URL",
    "PinnedSentenceTransformerEmbedder",
    "SemanticRuntimeError",
    "capture_semantic_runtime_provenance",
]
