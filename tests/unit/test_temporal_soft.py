"""Phase 2: temporal soft decay tests.

v1 hard filtering dropped out-of-window memories entirely. v2 keeps them in the
candidate set and only down-weights them.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from memforge.core.types import Memory, MemoryStatus, MemoryType
from memforge.embeddings.base import EmbeddingProvider
from memforge.retrieval.reranker import RerankConfig, UtilityReranker
from memforge.retrieval.temporal import (
    ExponentialDecayScorer,
    LinearDecayScorer,
    NoDecayScorer,
    WindowScorer,
    filter_temporal,
    memory_active_at,
)


def dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def make_memory(content: str, **kwargs) -> Memory:
    defaults = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "memory_type": MemoryType.FACT}
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


class _ConstEmb(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [[0.5, 0.5, 0.0] for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        return [0.5, 0.5, 0.0]


# ---------------------------------------------------------------------------
# v1 hard filter preserved for backward compatibility
# ---------------------------------------------------------------------------

def test_v1_hard_filter_still_works():
    """memory_active_at / filter_temporal keep exact v1 semantics."""
    beijing = make_memory("User lives in Beijing.", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 6, 1))
    shanghai = make_memory("User lives in Shanghai.", valid_from=dt(2025, 6, 1))
    assert filter_temporal([beijing, shanghai], dt(2024, 6, 1)) == [beijing]
    assert filter_temporal([beijing, shanghai], dt(2025, 12, 1)) == [shanghai]
    assert memory_active_at(beijing, dt(2025, 6, 1)) is False  # half-open


# ---------------------------------------------------------------------------
# Soft scorer basics
# ---------------------------------------------------------------------------

def test_no_decay_scorer_always_one():
    s = NoDecayScorer()
    m = make_memory("x", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1))
    assert s.score(m, dt(2020, 1, 1)) == 1.0
    assert s.score(m, dt(2026, 1, 1)) == 1.0


def test_window_scorer_binary_soft():
    s = WindowScorer(out_of_window=0.1)
    inside = make_memory("inside", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1))
    future = make_memory("future", valid_from=dt(2030, 1, 1))
    expired = make_memory("expired", valid_until=dt(2020, 1, 1))
    assert s.score(inside, dt(2024, 6, 1)) == 1.0
    assert s.score(future, dt(2024, 6, 1)) == 0.1
    assert s.score(expired, dt(2024, 6, 1)) == 0.1


def test_future_memory_not_hard_deleted():
    """A future memory must remain in reranked candidates (soft, not drop)."""
    scorer = WindowScorer(out_of_window=0.05)
    reranker = UtilityReranker(
        embedding=_ConstEmb(),
        config=RerankConfig(temporal_weight=1.0, relevance_weight=0.0,
                            confidence_weight=0.0, importance_weight=0.0,
                            freshness_weight=0.0),
        temporal_scorer=scorer,
    )
    future = make_memory("future fact", valid_from=dt(2030, 1, 1))
    results = reranker.rerank([future], query="anything", top_k=5, as_of=dt(2024, 6, 1))
    # v1 would have dropped it; v2 keeps it.
    assert len(results) == 1
    assert results[0].id == future.id


def test_temporal_score_monotonic():
    """The closer as_of is to the valid window, the higher the score."""
    s = ExponentialDecayScorer(floor=0.1, half_life_days=30.0)
    m = make_memory("x", valid_until=dt(2024, 1, 1))  # expired
    near = s.score(m, dt(2024, 1, 15))   # 14 days after expiry
    far = s.score(m, dt(2025, 1, 1))     # ~1 year after expiry
    assert 0.0 < far < near <= 1.0
    inside = s.score(m, dt(2023, 12, 1))
    assert inside == 1.0


def test_boundary_conditions():
    s = WindowScorer(out_of_window=0.1)
    m = make_memory("x", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1))
    # Half-open: valid_from inclusive, valid_until exclusive.
    assert s.score(m, dt(2024, 1, 1)) == 1.0
    assert s.score(m, dt(2025, 1, 1)) == 0.1
    # No temporal constraints -> always active.
    free = make_memory("no window")
    assert s.score(free, dt(2000, 1, 1)) == 1.0
    assert s.score(free, dt(2030, 1, 1)) == 1.0


def test_all_strategies_runnable():
    memories = [
        make_memory("inside", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1)),
        make_memory("future", valid_from=dt(2030, 1, 1)),
    ]
    as_of = dt(2024, 6, 1)
    for scorer in (NoDecayScorer(), WindowScorer(),
                   ExponentialDecayScorer(), LinearDecayScorer()):
        scores = scorer.score_batch(memories, as_of)
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Reranker integration
# ---------------------------------------------------------------------------

def test_reranker_with_temporal_score():
    """With temporal_weight>0, an in-window memory ranks above an out-of-window
    one that is otherwise equally relevant/confident/important/fresh."""
    scorer = WindowScorer(out_of_window=0.1)
    reranker = UtilityReranker(
        embedding=_ConstEmb(),
        config=RerankConfig(temporal_weight=0.5, relevance_weight=0.3,
                            confidence_weight=0.1, importance_weight=0.1,
                            freshness_weight=0.0),
        temporal_scorer=scorer,
    )
    inside = make_memory("inside fact", confidence=0.5, importance=0.5,
                         valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1))
    future = make_memory("future fact", confidence=0.5, importance=0.5,
                         valid_from=dt(2030, 1, 1))
    results = reranker.rerank([future, inside], query="q", top_k=2, as_of=dt(2024, 6, 1))
    assert results[0].id == inside.id
    # future memory still present (not dropped), just ranked lower.
    assert results[-1].id == future.id


def test_reranker_v1_hard_filter_when_no_scorer():
    """Backward compat: without a temporal_scorer, as_of still hard-filters."""
    reranker = UtilityReranker(embedding=_ConstEmb())  # no scorer
    future = make_memory("future fact", valid_from=dt(2030, 1, 1))
    inside = make_memory("inside fact", valid_from=dt(2024, 1, 1), valid_until=dt(2025, 1, 1))
    results = reranker.rerank([future, inside], query="q", top_k=5, as_of=dt(2024, 6, 1))
    assert [m.id for m in results] == [inside.id]


def test_reranker_no_as_of_unchanged():
    """Without as_of the reranker ranking is identical to v1."""
    emb = _ConstEmb()
    a = make_memory("a", confidence=0.9, importance=0.9)
    b = make_memory("b", confidence=0.2, importance=0.2)
    r = UtilityReranker(embedding=emb, temporal_scorer=WindowScorer())
    results = r.rerank([a, b], query="q", top_k=2)
    assert results[0].id == a.id
