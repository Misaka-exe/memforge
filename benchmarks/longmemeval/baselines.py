"""Retrieval baselines for LongMemEval-S (in-memory, session-level).

All baselines operate on per-turn MemoryItems and return an ordered list of
item indices.  Session-level metrics are computed afterwards by mapping
items back to their session_id.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "and",
    "or", "but", "of", "in", "on", "at", "to", "for", "with", "about",
    "what", "how", "when", "where", "which", "who", "why", "have", "has",
    "had", "be", "been", "not", "no", "yes", "me", "am", "this", "that",
    "there", "it's", "i'm", "you're", "can", "could", "would", "should",
    "will", "shall", "may", "might", "just", "so", "very", "really",
}


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP]


def keyword_score(query: str, content: str) -> float:
    q = set(_tokens(query))
    if not q:
        return 0.0
    c = set(_tokens(content))
    return len(q & c) / len(q)


def rank_by(
    scores: list[float],
    top_k: int,
) -> list[int]:
    """Return item indices sorted by score (desc), truncated to top_k."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:top_k]


def naive_vector(query_vec: list[float], item_vecs: list[list[float]], top_k: int) -> list[int]:
    scores = [cosine(query_vec, v) for v in item_vecs]
    return rank_by(scores, top_k)


def vector_rerank(
    query_vec: list[float],
    item_vecs: list[list[float]],
    freshness: list[float],
    top_k: int,
    w_rel: float = 0.8,
    w_fresh: float = 0.2,
) -> list[int]:
    """Semantic first; freshness (turn recency) as a second signal."""
    scores = [
        w_rel * cosine(query_vec, v) + w_fresh * f
        for v, f in zip(item_vecs, freshness)
    ]
    return rank_by(scores, top_k)


def hybrid(
    query: str,
    query_vec: list[float],
    item_vecs: list[list[float]],
    contents: list[str],
    top_k: int,
    candidate_k: int = 40,
    w_sem: float = 0.7,
    w_kw: float = 0.3,
) -> list[int]:
    """Vector + keyword union, then rerank by a blended score."""
    sem = [cosine(query_vec, v) for v in item_vecs]
    kw = [keyword_score(query, c) for c in contents]
    sem_top = set(rank_by(sem, candidate_k))
    kw_top = set(rank_by(kw, candidate_k))
    union = list(sem_top | kw_top)
    blended = {i: w_sem * sem[i] + w_kw * kw[i] for i in union}
    ordered = sorted(union, key=lambda i: blended[i], reverse=True)
    return ordered[:top_k]