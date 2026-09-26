"""Unit tests for retrieval score normalization (V2.1 Phase 1)."""
from __future__ import annotations

import numpy as np
import pytest

from memforge.retrieval.score import RetrievalFeatures, compute_retrieval_features


class TestRetrievalFeatures:
    def test_to_dict(self):
        f = RetrievalFeatures(
            question_id="q1", n_retrieved=10,
            top1_score=0.8, top5_mean=0.7, score_margin=0.1,
            mean_score=0.6, std_score=0.1,
        )
        d = f.to_dict()
        assert d["question_id"] == "q1"
        assert d["top1_score"] == 0.8
        assert d["n_retrieved"] == 10

    def test_feature_vector(self):
        f = RetrievalFeatures(
            question_id="q1", n_retrieved=10,
            top1_score=0.8, top5_mean=0.7, score_margin=0.1,
            mean_score=0.6, std_score=0.1,
        )
        vec = f.feature_vector()
        assert len(vec) == 4
        assert vec[0] == 0.8  # top1
        assert vec[1] == 0.7  # top5_mean
        assert vec[2] == 0.1  # margin
        assert vec[3] == 10   # n_retrieved


class TestComputeRetrievalFeatures:
    def test_missing_cache_returns_zeros(self, tmp_path):
        f = compute_retrieval_features("nonexistent", str(tmp_path))
        assert f.n_retrieved == 0
        assert f.top1_score == 0.0
        assert f.top5_mean == 0.0

    def test_with_cache(self, tmp_path):
        # Create a fake cache
        qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vecs = np.array([
            [1.0, 0.0, 0.0],   # sim=1.0
            [0.8, 0.6, 0.0],   # sim=0.8
            [0.5, 0.5, 0.7],   # sim=0.5
            [0.0, 1.0, 0.0],   # sim=0.0
        ], dtype=np.float32)
        np.savez(tmp_path / "q1.npz", qvec_n=qvec, vecs_n=vecs)

        f = compute_retrieval_features("q1", str(tmp_path))
        assert f.n_retrieved == 4
        assert abs(f.top1_score - 1.0) < 0.01
        assert abs(f.top5_mean - (1.0 + 0.8 + 0.5 + 0.0) / 4) < 0.01
        assert abs(f.score_margin - 0.2) < 0.01
