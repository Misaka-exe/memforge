"""Benchmark metrics: Recall@K / MRR / NDCG / Stale Rate / Timer."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    ranked = set(ranked_ids[:k])
    hits = sum(1 for g in gold_ids if g in ranked)
    return hits / len(gold_ids)


def recall_at_k_list(ranked_ids: list[str], gold_ids: list[str], ks: list[int]) -> dict[int, float]:
    return {k: recall_at_k(ranked_ids, gold_ids, k) for k in ks}


def mrr(ranked_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def dcg(relevance: list[float], k: int | None = None) -> float:
    k = k if k is not None else len(relevance)
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))


def ndcg(ranked_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    gold = set(gold_ids)
    rel = [1.0 if rid in gold else 0.0 for rid in ranked_ids[:k]]
    if not gold:
        return 0.0
    ideal = dcg([1.0] * min(k, len(gold)), k)
    if ideal == 0.0:
        return 0.0
    return dcg(rel, k) / ideal


def stale_rate(stale_flagged: list[str], actually_stale: list[str]) -> float:
    """Fraction of flagged-stale memories that are actually stale."""
    if not stale_flagged:
        return 0.0
    return sum(1 for s in stale_flagged if s in set(actually_stale)) / len(stale_flagged)


class Timer:
    def __init__(self) -> None:
        self.elapsed = 0.0

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed += time.perf_counter() - start