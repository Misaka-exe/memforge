"""Retrieval evaluation split by question category."""
from __future__ import annotations

from benchmarks.adapters.base import BenchmarkQuestion, QuestionCategory
from benchmarks.metrics import mrr, ndcg, recall_at_k

KS = (1, 5, 10)


def evaluate_question(ranked_ids: list[str], question: BenchmarkQuestion) -> dict:
    gold = question.gold_memory_ids or []
    if question.is_abstention:
        return {"category": question.category.value, "skipped": True}
    recall = {f"recall@{k}": recall_at_k(ranked_ids, gold, k) for k in KS}
    return {
        "category": question.category.value,
        **recall,
        "mrr": mrr(ranked_ids, gold),
        "ndcg": ndcg(ranked_ids, gold, k=5),
    }


def aggregate_by_category(results: list[dict]) -> dict[str, dict]:
    agg: dict[str, list[dict]] = {}
    for r in results:
        if r.get("skipped"):
            continue
        agg.setdefault(r["category"], []).append(r)
    out = {}
    for cat, rows in agg.items():
        n = len(rows)
        out[cat] = {
            "count": n,
            "recall@1": sum(r["recall@1"] for r in rows) / n,
            "recall@5": sum(r["recall@5"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "ndcg": sum(r["ndcg"] for r in rows) / n,
        }
    return out