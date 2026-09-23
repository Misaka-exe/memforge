"""Deterministic hash-based embedding provider.

Used for synthetic benchmarks and offline unit tests:
- no network, no model download
- fixed seed behaviour => reproducible experiments
- same tokens => higher cosine similarity (lexical signal)
"""
from __future__ import annotations

import hashlib
import math

from memforge.embeddings.base import EmbeddingProvider


def _h(token: str, salt: str) -> int:
    return int(hashlib.md5(f"{salt}:{token}".encode()).hexdigest()[:8], 16)


def hash_embedding(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    for tok in text.lower().split():
        vec[_h(tok, "t") % dim] += 1.0
    for ch in text.lower():
        vec[_h(ch, "c") % dim] += 0.3
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(t, self.dim) for t in texts]