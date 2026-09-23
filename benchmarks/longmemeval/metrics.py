"""Session-level metrics for LongMemEval retrieval.

An item maps to one session via its metadata["session_id"].
Given an ordered item list (highest score first), we compute:
- recall_any@K: at least one gold session appears in the top-K items
- recall_all@K: every gold session appears in the top-K items
- MRR: 1 / rank of the first gold session
- NDCG@10: graded over covered gold sessions
"""
from __future__ import annotations

import math


def _covered_sessions(item_ids: list[str], session_of: dict[str, str], k: int) -> set[str]:
    seen: set[str] = set()
    for i in item_ids[:k]:
        seen.add(session_of[i])
    return seen


def evaluate_question(
    ordered_item_ids: list[str],
    session_of: dict[str, str],
    gold_sessions: list[str],
    k_list: tuple[int, ...] = (1, 5, 10),
) -> dict:
    gold = set(gold_sessions)
    if not gold:
        return {"excluded": True}
    out: dict = {"excluded": False}
    for k in k_list:
        covered = _covered_sessions(ordered_item_ids, session_of, k)
        out[f"recall_any@{k}"] = 1.0 if covered & gold else 0.0
        out[f"recall_all@{k}"] = 1.0 if gold.issubset(covered) else 0.0

    # MRR: rank (1-based) of the first gold session, measured on the item list
    rank_of_session: dict[str, int] = {}
    for pos, iid in enumerate(ordered_item_ids, start=1):
        sid = session_of[iid]
        if sid not in rank_of_session:
            rank_of_session[sid] = pos
    first_rank = min(rank_of_session.get(g, 10**9) for g in gold)
    out["mrr"] = 1.0 / first_rank if first_rank < 10**9 else 0.0

    # NDCG@10 (session-graded, deduplicated: a session counts once at its first hit)
    k = 10
    dcg, idcg = 0.0, 0.0
    seen_gold: set[str] = set()
    for pos, iid in enumerate(ordered_item_ids[:k], start=1):
        sid = session_of[iid]
        if sid in gold and sid not in seen_gold:
            dcg += 1.0 / math.log2(pos + 1)
            seen_gold.add(sid)
    for i in range(1, min(len(gold), k) + 1):
        idcg += 1.0 / math.log2(i + 1)
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def aggregate(rows: list[dict]) -> dict:
    metrics: dict[str, float] = {}
    n = len(rows)
    for key in [k for k in rows[0] if k.startswith(("recall_", "mrr", "ndcg"))]:
        metrics[key] = round(sum(r[key] for r in rows) / n, 4)
    metrics["num_questions"] = n
    return metrics