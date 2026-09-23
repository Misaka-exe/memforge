"""Baseline retrieval strategies (deterministic, no LLM).

- full_context: treat the whole memory set as context (created order).
- naive_vector: pure cosine top-k on hash embeddings.
- vector_rerank: vector top-20 then utility rerank.
"""
from __future__ import annotations

from benchmarks.metrics import recall_at_k_list, mrr, ndcg
from benchmarks.synthetic.generator import hash_embedding
from benchmarks.synthetic.ground_truth import BenchmarkQuery, BenchmarkScenario


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _active_memories(scenario: BenchmarkScenario) -> list:
    return [m for m in scenario.memories if m.status == "active"]


class FullContextBaseline:
    name = "full_context"

    def retrieve(self, scenario: BenchmarkScenario, query: BenchmarkQuery, top_k: int = 5) -> list[str]:
        # Everything is "in context"; ordering is creation order.
        return [m.id for m in scenario.memories]


class NaiveVectorBaseline:
    name = "naive_vector"

    def retrieve(self, scenario: BenchmarkScenario, query: BenchmarkQuery, top_k: int = 5) -> list[str]:
        qemb = hash_embedding(query.query)
        scored = [
            (_cos(qemb, hash_embedding(m.content)), m.id)
            for m in _active_memories(scenario)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:top_k]]


class VectorRerankBaseline:
    name = "vector_rerank"

    def retrieve(self, scenario: BenchmarkScenario, query: BenchmarkQuery, top_k: int = 5) -> list[str]:
        qemb = hash_embedding(query.query)
        candidates = _active_memories(scenario)
        scored = []
        for m in candidates:
            relevance = _cos(qemb, hash_embedding(m.content))
            freshness = 0.9  # deterministic stand-in
            utility = 0.4 * relevance + 0.2 * m.confidence + 0.2 * m.importance + 0.2 * freshness
            scored.append((utility, m.id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:top_k]]


def evaluate_strategy(strategy, scenario: BenchmarkScenario, top_k: int = 5) -> dict:
    r1, r5, r10, mrrs, ndcgs = [], [], [], [], []
    for q in scenario.queries:
        ranked = strategy.retrieve(scenario, q, top_k=top_k)
        gold = q.gold_memory_ids
        r1.append(recall_at_k_list(ranked, gold, [1, 5, 10]))
        mrrs.append(mrr(ranked, gold))
        ndcgs.append(ndcg(ranked, gold, k=top_k))
    return {
        "recall@1": sum(x[1] for x in r1) / len(r1) if r1 else 0,
        "recall@5": sum(x[5] for x in r1) / len(r1) if r1 else 0,
        "recall@10": sum(x[10] for x in r1) / len(r1) if r1 else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
        "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
    }