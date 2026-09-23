"""Stage 3: temporal / hybrid / reranker / forgetting tests (10 tests, no DB)."""
import math
from datetime import datetime, timedelta, timezone

import pytest

from memforge.core.types import Memory, MemoryStatus, MemoryType
from memforge.embeddings.base import EmbeddingProvider
from memforge.retrieval.hybrid import HybridRetriever
from memforge.retrieval.reranker import RerankConfig, UtilityReranker
from memforge.retrieval.temporal import filter_temporal, memory_active_at


def dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


class FakeEmbedding(EmbeddingProvider):
    def __init__(self) -> None:
        self.vectors = {
            "phone": [0.9, 0.1, 0.0],
            "coffee": [0.0, 0.1, 0.9],
            "keyboard": [0.0, 0.9, 0.1],
            "battery": [0.8, 0.2, 0.0],
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.5, 0.5, 0.0]) for t in texts]


class FakeRepository:
    def __init__(self, memories: list[Memory]) -> None:
        self.memories = memories

    def _cos(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    async def search(self, query_embedding, user_id=None, memory_type=None, status=MemoryStatus.ACTIVE, top_k=20, as_of=None):
        scored = []
        for m in self.memories:
            if m.status != status:
                continue
            if user_id is not None and m.user_id != user_id:
                continue
            if memory_type is not None and m.memory_type.value != memory_type:
                continue
            if as_of is not None and not memory_active_at(m, as_of):
                continue
            scored.append((self._cos(query_embedding, m.embedding or []), m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    async def keyword_search(self, query, user_id=None, status=MemoryStatus.ACTIVE, top_k=20):
        hits = []
        for m in self.memories:
            if m.status != status:
                continue
            if user_id is not None and m.user_id != user_id:
                continue
            if query.lower() in m.content.lower():
                hits.append(m)
        return hits[:top_k]

    async def list_by_user(self, user_id, limit=10000):
        return [m for m in self.memories if m.user_id == user_id][:limit]


def make_memory(content: str, **kwargs) -> Memory:
    defaults = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "memory_type": MemoryType.FACT}
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def test_temporal_beijing_2024():
    beijing = make_memory(
        "User lives in Beijing.", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 6, 1)
    )
    shanghai = make_memory("User lives in Shanghai.", valid_from=dt(2025, 6, 1))
    memories = [beijing, shanghai]
    assert [m.content for m in filter_temporal(memories, dt(2024, 6, 1))] == ["User lives in Beijing."]
    assert [m.content for m in filter_temporal(memories, dt(2025, 12, 1))] == ["User lives in Shanghai."]


def test_temporal_half_open_boundary():
    m = make_memory("User lived in Wuhan.", valid_from=dt(2023, 1, 1), valid_until=dt(2024, 1, 1))
    assert memory_active_at(m, dt(2023, 12, 31)) is True
    assert memory_active_at(m, dt(2024, 1, 1)) is False  # half-open: T2 excluded


def test_temporal_inactive_not_returned():
    m = make_memory("User likes tea.", valid_from=dt(2020, 1, 1), status=MemoryStatus.DORMANT)
    assert memory_active_at(m, dt(2024, 1, 1)) is False


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

def test_reranker_high_confidence_ranks_above_low():
    emb = FakeEmbedding()
    reranker = UtilityReranker(embedding=emb)
    high = make_memory("User likes phones.", embedding=[0.9, 0.1, 0.0], confidence=0.95, importance=0.8)
    low = make_memory("User likes phones.", embedding=[0.85, 0.15, 0.0], confidence=0.3, importance=0.8)
    results = reranker.rerank([high, low], query="phone", top_k=2)
    assert results[0].id == high.id


def test_reranker_freshness_decay():
    emb = FakeEmbedding()
    now = dt(2026, 9, 21)
    old = make_memory(
        "User likes phones.",
        embedding=[0.9, 0.1, 0.0],
        confidence=0.9,
        importance=0.9,
        updated_at=now - timedelta(days=400),
    )
    new = make_memory(
        "User likes phones.",
        embedding=[0.85, 0.15, 0.0],
        confidence=0.9,
        importance=0.9,
        updated_at=now,
    )
    results = UtilityReranker(embedding=emb).rerank([old, new], query="phone", top_k=2, now=now)
    assert results[0].id == new.id


def test_reranker_importance_weight_effect():
    emb = FakeEmbedding()
    # With importance weight, high-importance memory wins despite lower relevance.
    full = UtilityReranker(embedding=emb, config=RerankConfig(importance_weight=0.8, relevance_weight=0.1, confidence_weight=0.05, freshness_weight=0.05))
    imp_high = make_memory("User likes phones.", embedding=[0.7, 0.3, 0.0], importance=0.95, confidence=0.5)
    imp_low = make_memory("User likes phones.", embedding=[0.9, 0.1, 0.0], importance=0.1, confidence=0.5)
    # With importance dominating, the high-importance (but less relevant) memory wins.
    results = full.rerank([imp_high, imp_low], query="phone", top_k=2)
    assert results[0].id == imp_high.id
    # Without importance weight, relevance decides (low-importance is more relevant).
    no_imp = UtilityReranker(embedding=emb, config=RerankConfig(importance_weight=0.0, relevance_weight=0.8, confidence_weight=0.1, freshness_weight=0.1))
    results2 = no_imp.rerank([imp_high, imp_low], query="phone", top_k=2)
    assert results2[0].id == imp_low.id


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------

def test_hybrid_returns_active_only():
    repo = FakeRepository(
        [
            make_memory("User likes phones.", embedding=[0.9, 0.1, 0.0]),
            make_memory("User liked old phones.", embedding=[0.9, 0.1, 0.0], status=MemoryStatus.DEPRECATED),
            make_memory("User likes coffee.", embedding=[0.0, 0.1, 0.9], status=MemoryStatus.DORMANT),
            make_memory("User liked fish.", embedding=[0.9, 0.1, 0.0], status=MemoryStatus.FORGOTTEN),
        ]
    )
    retriever = HybridRetriever(repo, FakeEmbedding())
    import asyncio
    results = asyncio.run(retriever.search("User likes phones"))
    assert len(results) == 1
    assert results[0].status == MemoryStatus.ACTIVE


def test_hybrid_keyword_and_semantic_union():
    repo = FakeRepository(
        [
            make_memory("User likes mechanical keyboards.", embedding=[0.0, 0.9, 0.1]),
            make_memory("User drinks coffee every morning.", embedding=[0.0, 0.1, 0.9]),
        ]
    )
    retriever = HybridRetriever(repo, FakeEmbedding())
    import asyncio
    # "keyboard" is a keyword hit but has low semantic similarity to "keyboard" query below.
    results = asyncio.run(retriever.search("keyboard", top_k=5))
    assert any("keyboard" in m.content for m in results)


def test_hybrid_status_filter_deprecated_excluded():
    repo = FakeRepository(
        [
            make_memory("User lives in Beijing.", embedding=[1.0, 0.0, 0.0]),
            make_memory("User lived in Beijing.", embedding=[1.0, 0.0, 0.0], status=MemoryStatus.DEPRECATED),
        ]
    )
    retriever = HybridRetriever(repo, FakeEmbedding())
    import asyncio
    results = asyncio.run(retriever.search("Beijing", top_k=5))
    assert len(results) == 1
    assert results[0].status == MemoryStatus.ACTIVE


def test_hybrid_top_k_respected():
    repo = FakeRepository(
        [
            make_memory(f"User likes topic {i}.", embedding=[0.9, 0.1, 0.0]) for i in range(10)
        ]
    )
    retriever = HybridRetriever(repo, FakeEmbedding())
    import asyncio
    results = asyncio.run(retriever.search("topic", top_k=3))
    assert len(results) <= 3