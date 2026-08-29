from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from model_forensics.semantic_backend import (
    SEMANTIC_MODEL_REVISION,
    SEMANTIC_STACK_VERSIONS,
    SEMANTIC_WHEEL_SHA256,
    SEMANTIC_WHEEL_URL,
    TRANSFORMERS_COMMIT,
    PinnedSentenceTransformerEmbedder,
    SemanticRuntimeError,
    capture_semantic_runtime_provenance,
)


class Array:
    def tolist(self) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]


class Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> Array:
        self.calls.append({"texts": texts, **kwargs})
        return Array()


def test_pinned_embedder_uses_frozen_revision_and_stable_encoding_options() -> None:
    observed = {}
    model = Model()

    def factory(*args: Any, **kwargs: Any) -> Model:
        observed.update(args=args, kwargs=kwargs)
        return model

    embedder = PinnedSentenceTransformerEmbedder(model_factory=factory)
    assert embedder.encode(("one sentence", "another sentence")) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert len(observed["kwargs"]["revision"]) == 40
    assert observed["kwargs"]["device"] == "cpu"
    assert model.calls[0]["normalize_embeddings"] is False
    assert embedder.provenance["provenance_hash"].startswith("sha256:")
    assert embedder.provenance["primary_eligible"] is False


def test_embedder_refuses_revision_drift() -> None:
    with pytest.raises(ValueError, match="frozen"):
        PinnedSentenceTransformerEmbedder(revision="latest", model_factory=lambda *a, **k: Model())


class Distribution:
    def __init__(self, name: str, version: str, root: Path) -> None:
        self.name = name
        self.version = version
        self.base = root / "site-packages"
        self.file = self.base / name / "__init__.py"
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_bytes(b"x")

    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            return f"Name: {self.name}\nVersion: {self.version}\n"
        if filename == "RECORD":
            digest = base64.urlsafe_b64encode(hashlib.sha256(b"x").digest()).rstrip(b"=")
            return f"{self.name}/__init__.py,sha256={digest.decode()},1\n"
        if filename == "INSTALLER":
            return "pip\n"
        if filename == "direct_url.json" and self.name == "transformers":
            return (
                '{"url":"https://github.com/huggingface/transformers.git",'
                '"vcs_info":{"vcs":"git","requested_revision":"'
                + TRANSFORMERS_COMMIT
                + '","commit_id":"'
                + TRANSFORMERS_COMMIT
                + '"}}'
            )
        if filename == "direct_url.json" and self.name == "sentence-transformers":
            return (
                '{"url":"'
                + SEMANTIC_WHEEL_URL
                + '","archive_info":{"hashes":{"sha256":"'
                + SEMANTIC_WHEEL_SHA256
                + '"}}}'
            )
        return None

    def locate_file(self, filename: str) -> Path:
        return self.base / filename


def _distribution_factory(root: Path):
    def factory(name: str) -> Distribution:
        return Distribution(name, SEMANTIC_STACK_VERSIONS[name], root)

    return factory


def test_semantic_runtime_attests_exact_versions_metadata_records_and_source(
    tmp_path: Path,
) -> None:
    runtime = capture_semantic_runtime_provenance(
        distribution_factory=_distribution_factory(tmp_path),
        installation_root=tmp_path,
    )

    assert runtime["stack_versions"] == dict(sorted(SEMANTIC_STACK_VERSIONS.items()))
    assert runtime["distributions"]["sentence-transformers"]["record_sha256"].startswith(
        "sha256:"
    )
    assert runtime["distributions"]["transformers"]["source"]["commit_id"] == (
        TRANSFORMERS_COMMIT
    )
    assert runtime["distributions"]["sentence-transformers"]["source"] == {
        "wheel_filename": "sentence_transformers-5.7.0-py3-none-any.whl",
        "archive_sha256": SEMANTIC_WHEEL_SHA256,
    }
    assert runtime["runtime_hash"].startswith("sha256:")
    assert runtime["verified_installed_files"]["verified_file_count"] == len(
        SEMANTIC_STACK_VERSIONS
    )


def test_semantic_runtime_fails_closed_on_one_version_drift(tmp_path: Path) -> None:
    def drifted(name: str) -> Distribution:
        version = "0.0.0" if name == "scipy" else SEMANTIC_STACK_VERSIONS[name]
        return Distribution(name, version, tmp_path)

    with pytest.raises(SemanticRuntimeError, match="versions drifted"):
        capture_semantic_runtime_provenance(
            distribution_factory=drifted, installation_root=tmp_path
        )


def test_semantic_runtime_fails_closed_on_wheel_hash_drift(tmp_path: Path) -> None:
    class DriftedWheel(Distribution):
        def read_text(self, filename: str) -> str | None:
            value = super().read_text(filename)
            if filename == "direct_url.json":
                return value.replace(SEMANTIC_WHEEL_SHA256, "0" * 64) if value else value
            return value

    def drifted(name: str) -> Distribution:
        distribution = Distribution(name, SEMANTIC_STACK_VERSIONS[name], tmp_path)
        if name == "sentence-transformers":
            distribution = DriftedWheel(name, SEMANTIC_STACK_VERSIONS[name], tmp_path)
        return distribution

    with pytest.raises(SemanticRuntimeError, match="archive hash drifted"):
        capture_semantic_runtime_provenance(
            distribution_factory=drifted, installation_root=tmp_path
        )


def test_semantic_runtime_rejects_record_file_tamper(tmp_path: Path) -> None:
    distributions = {
        name: Distribution(name, version, tmp_path)
        for name, version in SEMANTIC_STACK_VERSIONS.items()
    }
    distributions["scipy"].file.write_bytes(b"tampered")

    with pytest.raises(SemanticRuntimeError, match="hash/size drifted"):
        capture_semantic_runtime_provenance(
            distribution_factory=distributions.__getitem__, installation_root=tmp_path
        )


def test_semantic_runtime_rejects_record_symlink(tmp_path: Path) -> None:
    distributions = {
        name: Distribution(name, version, tmp_path)
        for name, version in SEMANTIC_STACK_VERSIONS.items()
    }
    target = tmp_path / "target.py"
    target.write_bytes(b"x")
    distributions["scipy"].file.unlink()
    distributions["scipy"].file.symlink_to(target)

    with pytest.raises(SemanticRuntimeError, match="symlink"):
        capture_semantic_runtime_provenance(
            distribution_factory=distributions.__getitem__, installation_root=tmp_path
        )


class RevisionObject:
    _commit_hash = SEMANTIC_MODEL_REVISION


class Tokenizer(RevisionObject):
    vocab_size = 30_522
    model_max_length = 512
    padding_side = "right"
    truncation_side = "right"


class SnapshotTokenizer:
    vocab_size = 30_522
    model_max_length = 512
    padding_side = "right"
    truncation_side = "right"
    name_or_path = f"/cache/models--sentence-transformers/snapshots/{SEMANTIC_MODEL_REVISION}/"


class AutoModel(RevisionObject):
    config = RevisionObject()


class FirstModule:
    auto_model = AutoModel()
    tokenizer = Tokenizer()


class PrimaryModel(Model):
    max_seq_length = 256
    truncate_dim = None

    def _first_module(self) -> FirstModule:
        return FirstModule()


def test_primary_embedder_binds_model_and_tokenizer_revision_before_encode(
    tmp_path: Path,
) -> None:
    embedder = PinnedSentenceTransformerEmbedder(
        model_factory=lambda *args, **kwargs: PrimaryModel(),
        distribution_factory=_distribution_factory(tmp_path),
        installation_root=tmp_path,
        require_primary_eligibility=True,
    )

    embedder.assert_primary_eligible()
    provenance = embedder.provenance
    assert provenance["primary_eligible"] is True
    assert provenance["model_tokenizer_runtime"]["model_revision"] == (
        SEMANTIC_MODEL_REVISION
    )
    assert provenance["distribution_runtime"]["runtime_hash"].startswith("sha256:")


def test_primary_embedder_accepts_tokenizer_revision_from_hub_snapshot_path(
    tmp_path: Path,
) -> None:
    class SnapshotFirstModule:
        auto_model = AutoModel()
        tokenizer = SnapshotTokenizer()

    class SnapshotModel(PrimaryModel):
        def _first_module(self) -> SnapshotFirstModule:
            return SnapshotFirstModule()

    embedder = PinnedSentenceTransformerEmbedder(
        model_factory=lambda *args, **kwargs: SnapshotModel(),
        distribution_factory=_distribution_factory(tmp_path),
        installation_root=tmp_path,
        require_primary_eligibility=True,
    )

    assert SEMANTIC_MODEL_REVISION in embedder.provenance["model_tokenizer_runtime"][
        "tokenizer"
    ]["revision_candidates"]
