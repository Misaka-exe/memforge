"""Utility-aware reranker.

score(m, q) = w1*relevance + w2*confidence + w3*importance + w4*freshness

- relevance: cosine similarity between query embedding and memory embedding
- freshness: exponential decay on updated_at
- Weights are configurable (RerankConfig) so ablation experiments can simply
  toggle each weight to zero.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from memforge.core.types import Memory
from memforge.embeddings.base import EmbeddingProvider
from memforge.retrieval.temporal import memory_active_at


class RerankConfig(BaseModel):
    relevance_weight: float = 0.4
    confidence_weight: float = 0.2
    importance_weight: float = 0.2
    freshness_weight: float = 0.2
    freshness_half_life_days: int = 90

    def as_weights(self) -> tuple[float, float, float, float]:
        return (
            self.relevance_weight,
            self.confidence_weight,
            self.importance_weight,
            self.freshness_weight,
        )


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def freshness_score(memory: Memory, now: datetime | None = None, half_life_days: int = 90) -> float:
    """Exponential decay: 1.0 at update time, ~0.5 after half-life."""
    now = now or datetime.now(memory.updated_at.tzinfo or __import__("datetime").timezone.utc)
    reference = memory.updated_at or memory.created_at
    age = (now - reference).total_seconds() / 86400.0
    if age <= 0:
        return 1.0
    return math.exp(-math.log(2) * age / half_life_days)


class UtilityReranker:
    def __init__(
        self,
        embedding: EmbeddingProvider | None = None,
        config: RerankConfig | None = None,
    ) -> None:
        self.embedding = embedding
        self.config = config or RerankConfig()

    def score(self, memory: Memory, query_embedding: list[float], now: datetime | None = None) -> float:
        w_rel, w_conf, w_imp, w_fresh = self.config.as_weights()
        relevance = cosine(query_embedding, memory.embedding or [])
        freshness = freshness_score(memory, now, self.config.freshness_half_life_days)
        return (
            w_rel * relevance
            + w_conf * memory.confidence
            + w_imp * memory.importance
            + w_fresh * freshness
        )

    def rerank(
        self,
        candidates: list[Memory],
        query: str,
        top_k: int = 5,
        as_of: datetime | None = None,
        now: datetime | None = None,
    ) -> list[Memory]:
        query_embedding = self.embedding.embed_one(query) if self.embedding else []
        if as_of is not None:
            candidates = [m for m in candidates if memory_active_at(m, as_of)]
        scored = sorted(
            candidates,
            key=lambda m: self.score(m, query_embedding, now),
            reverse=True,
        )
        return scored[:top_k]