"""Embedding provider abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]