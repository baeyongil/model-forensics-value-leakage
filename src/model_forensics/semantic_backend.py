"""Pinned sentence-transformer adapter for the preregistered divergence gate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from model_forensics.io import stable_hash

SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class PinnedSentenceTransformerEmbedder:
    def __init__(
        self,
        *,
        model_id: str = SEMANTIC_MODEL_ID,
        revision: str = SEMANTIC_MODEL_REVISION,
        device: str = "cpu",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if model_id != SEMANTIC_MODEL_ID or revision != SEMANTIC_MODEL_REVISION:
            raise ValueError("semantic divergence model must use the frozen ID and revision")
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
        self._provenance = {
            "model_id": model_id,
            "revision": revision,
            "device": device,
            "trust_remote_code": False,
        }
        self._provenance["provenance_hash"] = stable_hash(self._provenance)

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("semantic inputs must be nonempty strings")
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
    "SEMANTIC_MODEL_ID",
    "SEMANTIC_MODEL_REVISION",
    "PinnedSentenceTransformerEmbedder",
]
