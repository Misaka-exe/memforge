"""Unit tests for calibrated sufficiency gate (V2.1 Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest

from memforge.retrieval.gate_v2 import (
    MultiFeatureGate,
    Sufficiency,
    ThresholdGate,
    evaluate_gate,
)
from memforge.retrieval.score import RetrievalFeatures


def make_features(top1=0.5, top5=0.4, margin=0.1, n=10, qid="q1"):
    return RetrievalFeatures(
        question_id=qid, n_retrieved=n,
        top1_score=top1, top5_mean=top5, score_margin=margin,
        mean_score=0.45, std_score=0.1,
    )


class TestThresholdGate:
    def test_sufficient(self):
        gate = ThresholdGate(threshold=0.5, band=0.05)
        d = gate.assess(make_features(top1=0.8))
        assert d.sufficiency == Sufficiency.SUFFICIENT

    def test_insufficient(self):
        gate = ThresholdGate(threshold=0.5, band=0.05)
        d = gate.assess(make_features(top1=0.2))
        assert d.sufficiency == Sufficiency.INSUFFICIENT

    def test_uncertain(self):
        gate = ThresholdGate(threshold=0.5, band=0.05)
        d = gate.assess(make_features(top1=0.5))
        assert d.sufficiency == Sufficiency.UNCERTAIN

    def test_calibration(self):
        gate = ThresholdGate()
        features = [make_features(top1=0.8), make_features(top1=0.2)]
        labels = [True, False]
        info = gate.calibrate(features, labels)
        assert "best_threshold" in info
        assert info["n_samples"] == 2


class TestMultiFeatureGate:
    def test_insufficient_low_n(self):
        gate = MultiFeatureGate(min_retrieved=5)
        d = gate.assess(make_features(n=2))
        assert d.sufficiency == Sufficiency.INSUFFICIENT

    def test_sufficient_high_scores(self):
        gate = MultiFeatureGate(top1_threshold=0.4, top5_threshold=0.3)
        d = gate.assess(make_features(top1=0.8, top5=0.7, n=10))
        assert d.sufficiency == Sufficiency.SUFFICIENT

    def test_calibration(self):
        gate = MultiFeatureGate()
        features = [make_features(top1=0.8, top5=0.7), make_features(top1=0.2, top5=0.1)]
        labels = [True, False]
        info = gate.calibrate(features, labels)
        assert "best_top1_threshold" in info


class TestEvaluateGate:
    def test_basic_metrics(self):
        gate = ThresholdGate(threshold=0.5)
        features = [make_features(top1=0.8), make_features(top1=0.2)]
        is_answerable = [True, False]
        is_correct = [True, False]
        metrics = evaluate_gate(gate, features, is_answerable, is_correct)
        assert 0 <= metrics.coverage <= 1
        assert metrics.n_sufficient + metrics.n_uncertain + metrics.n_insufficient == 2

    def test_perfect_gate(self):
        # Gate that allows answerable, blocks abstention
        class PerfectGate(ThresholdGate):
            def assess(self, f):
                from memforge.retrieval.gate_v2 import GateDecision
                if f.question_id == "ans":
                    return GateDecision(Sufficiency.SUFFICIENT, 1.0, "ok", {})
                return GateDecision(Sufficiency.INSUFFICIENT, 0.0, "block", {})

        gate = PerfectGate()
        features = [make_features(qid="ans"), make_features(qid="abs")]
        is_answerable = [True, False]
        is_correct = [True, False]
        metrics = evaluate_gate(gate, features, is_answerable, is_correct)
        assert metrics.answerable_recall == 1.0
        assert metrics.abstention_accuracy == 1.0
