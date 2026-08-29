from __future__ import annotations

from typing import Any

import pytest

from model_forensics.semantic_backend import PinnedSentenceTransformerEmbedder


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


def test_embedder_refuses_revision_drift() -> None:
    with pytest.raises(ValueError, match="frozen"):
        PinnedSentenceTransformerEmbedder(revision="latest", model_factory=lambda *a, **k: Model())
