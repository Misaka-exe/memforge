"""Local sentence-transformers embedding provider.

Lazy-loads the model on first call so that importing the package
does not download anything.
"""
from __future__ import annotations

import os

from memforge.embeddings.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "MEMFORGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.device = device or os.environ.get("MEMFORGE_EMBEDDING_DEVICE", "cpu")
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]