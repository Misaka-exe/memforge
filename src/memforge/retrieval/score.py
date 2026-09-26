"""Retrieval score normalization for V2.1.

Provides unified retrieval features (top1, top5_mean, margin, n_retrieved)
computed from real embedding similarities, replacing V2's memory.importance
proxy which caused 100% SUFFICIENT gate rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RetrievalFeatures:
    """Aggregate retrieval quality features for one question."""
    question_id: str
    n_retrieved: int
    top1_score: float
    top5_mean: float
    score_margin: float
    mean_score: float
    std_score: float

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "n_retrieved": self.n_retrieved,
            "top1_score": round(self.top1_score, 6),
            "top5_mean": round(self.top5_mean, 6),
            "score_margin": round(self.score_margin, 6),
            "mean_score": round(self.mean_score, 6),
            "std_score": round(self.std_score, 6),
        }

    def feature_vector(self) -> list[float]:
        """Ordered feature vector for calibration."""
        return [self.top1_score, self.top5_mean, self.score_margin, self.n_retrieved]


def compute_retrieval_features(
    question_id: str,
    cache_dir: str | Path,
    top_k: int = 10,
) -> RetrievalFeatures:
    """Compute retrieval features from pre-computed embedding cache.

    Uses normalized embeddings (cosine similarity = dot product).
    Does NOT re-embed. Cache comes from V2's stage5_real/_cache/embeddings/.
    """
    cache_path = Path(cache_dir) / f"{question_id}.npz"
    if not cache_path.exists():
        return RetrievalFeatures(
            question_id=question_id, n_retrieved=0,
            top1_score=0.0, top5_mean=0.0, score_margin=0.0,
            mean_score=0.0, std_score=0.0,
        )

    data = np.load(cache_path)
    qvec = data["qvec_n"]  # (384,) normalized
    vecs = data["vecs_n"]  # (N, 384) normalized

    sims = vecs @ qvec  # (N,) cosine similarities
    sorted_sims = np.sort(sims)[::-1]

    top_k_sims = sorted_sims[:top_k]
    top5 = sorted_sims[:5] if len(sorted_sims) >= 5 else sorted_sims

    margin = float(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) >= 2 else 0.0

    return RetrievalFeatures(
        question_id=question_id,
        n_retrieved=min(len(sims), top_k),
        top1_score=float(sorted_sims[0]) if len(sims) > 0 else 0.0,
        top5_mean=float(np.mean(top5)),
        score_margin=margin,
        mean_score=float(np.mean(top_k_sims)),
        std_score=float(np.std(top_k_sims)),
    )
