"""Calibrated Sufficiency Gate for V2.1.

Replaces V2's gate which used memory.importance (default 0.8) as a proxy,
causing 100% SUFFICIENT rate. This gate uses real retrieval features
(top1_score, top5_mean, score_margin, n_retrieved) and supports calibration
on labeled data.

Decision strategies:
- ThresholdGate: single threshold on top1_score
- MultiFeatureGate: rule-based on multiple features
- LogisticGate: pre-trained logistic regression weights

Calibration is explicitly marked as exploratory when no held-out set exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from memforge.retrieval.score import RetrievalFeatures


class Sufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class GateDecision:
    sufficiency: Sufficiency
    score: float  # calibrated sufficiency score [0, 1]
    reason: str
    features: dict = field(default_factory=dict)


class BaseCalibratedGate:
    """Base class for calibrated sufficiency gates."""

    def assess(self, features: RetrievalFeatures) -> GateDecision:
        raise NotImplementedError

    def calibrate(self, features_list: list[RetrievalFeatures], labels: list[bool]) -> dict:
        """Calibrate on labeled data. labels[i] = True if answerable.

        Returns calibration info. Subclasses override.
        """
        raise NotImplementedError


class ThresholdGate(BaseCalibratedGate):
    """Single-threshold gate on top1_score.

    top1 >= threshold + band → SUFFICIENT
    top1 <= threshold - band → INSUFFICIENT
    else → UNCERTAIN
    """

    def __init__(self, threshold: float = 0.5, band: float = 0.05):
        self.threshold = threshold
        self.band = band

    def assess(self, features: RetrievalFeatures) -> GateDecision:
        score = features.top1_score
        if score >= self.threshold + self.band:
            return GateDecision(Sufficiency.SUFFICIENT, score,
                                f"top1={score:.3f} >= {self.threshold + self.band:.3f}",
                                features.to_dict())
        if score <= self.threshold - self.band:
            return GateDecision(Sufficiency.INSUFFICIENT, score,
                                f"top1={score:.3f} <= {self.threshold - self.band:.3f}",
                                features.to_dict())
        return GateDecision(Sufficiency.UNCERTAIN, score,
                            f"top1={score:.3f} within band [{self.threshold - self.band:.3f}, {self.threshold + self.band:.3f}]",
                            features.to_dict())

    def calibrate(self, features_list: list[RetrievalFeatures], labels: list[bool]) -> dict:
        """Find threshold that maximizes F1 on answerable prediction."""
        scores = np.array([f.top1_score for f in features_list])
        y = np.array(labels, dtype=bool)

        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, 81):
            pred = scores >= t
            tp = np.sum(pred & y)
            fp = np.sum(pred & ~y)
            fn = np.sum(~pred & y)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.threshold = best_t
        return {
            "method": "threshold_top1",
            "best_threshold": round(best_t, 3),
            "best_f1": round(best_f1, 3),
            "n_samples": len(labels),
            "note": "exploratory calibration on same dataset; not unbiased evaluation",
        }


class MultiFeatureGate(BaseCalibratedGate):
    """Rule-based gate using multiple features.

    SUFFICIENT if: top1 >= t1 AND top5_mean >= t2 AND n_retrieved >= min_n
    INSUFFICIENT if: top1 < t1 - band OR n_retrieved < min_n
    UNCERTAIN otherwise
    """

    def __init__(self, top1_threshold: float = 0.45, top5_threshold: float = 0.35,
                 min_retrieved: int = 3, band: float = 0.05):
        self.top1_threshold = top1_threshold
        self.top5_threshold = top5_threshold
        self.min_retrieved = min_retrieved
        self.band = band

    def assess(self, features: RetrievalFeatures) -> GateDecision:
        score = 0.4 * features.top1_score + 0.3 * features.top5_mean + 0.3 * min(features.n_retrieved / 10, 1.0)

        if features.n_retrieved < self.min_retrieved:
            return GateDecision(Sufficiency.INSUFFICIENT, score,
                                f"n_retrieved={features.n_retrieved} < {self.min_retrieved}",
                                features.to_dict())

        if (features.top1_score >= self.top1_threshold + self.band and
                features.top5_mean >= self.top5_threshold):
            return GateDecision(Sufficiency.SUFFICIENT, score,
                                f"top1={features.top1_score:.3f}, top5={features.top5_mean:.3f}, n={features.n_retrieved}",
                                features.to_dict())

        if (features.top1_score <= self.top1_threshold - self.band or
                features.top5_mean <= self.top5_threshold - self.band):
            return GateDecision(Sufficiency.INSUFFICIENT, score,
                                f"top1={features.top1_score:.3f}, top5={features.top5_mean:.3f} below threshold",
                                features.to_dict())

        return GateDecision(Sufficiency.UNCERTAIN, score,
                            f"top1={features.top1_score:.3f}, top5={features.top5_mean:.3f} within band",
                            features.to_dict())

    def calibrate(self, features_list: list[RetrievalFeatures], labels: list[bool]) -> dict:
        """Grid search over top1_threshold and top5_threshold."""
        best_f1 = 0
        best_params = (0.45, 0.35)

        for t1 in np.linspace(0.3, 0.7, 17):
            for t2 in np.linspace(0.2, 0.6, 17):
                self.top1_threshold = float(t1)
                self.top5_threshold = float(t2)
                preds = []
                for f in features_list:
                    d = self.assess(f)
                    preds.append(d.sufficiency == Sufficiency.SUFFICIENT)
                pred = np.array(preds, dtype=bool)
                y = np.array(labels, dtype=bool)
                tp = np.sum(pred & y)
                fp = np.sum(pred & ~y)
                fn = np.sum(~pred & y)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (float(t1), float(t2))

        self.top1_threshold, self.top5_threshold = best_params
        return {
            "method": "multifeature_grid",
            "best_top1_threshold": round(best_params[0], 3),
            "best_top5_threshold": round(best_params[1], 3),
            "best_f1": round(best_f1, 3),
            "n_samples": len(labels),
            "note": "exploratory calibration on same dataset; not unbiased evaluation",
        }


@dataclass
class GateMetrics:
    """Evaluation metrics for a gate."""
    coverage: float  # fraction of questions answered (SUFFICIENT or UNCERTAIN)
    answerable_recall: float  # answerable questions allowed to answer / total answerable
    abstention_precision: float  # correct abstentions / all abstentions
    abstention_accuracy: float
    selective_accuracy: float  # accuracy on answered questions only
    n_sufficient: int
    n_uncertain: int
    n_insufficient: int

    def to_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 4),
            "answerable_recall": round(self.answerable_recall, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "abstention_accuracy": round(self.abstention_accuracy, 4),
            "selective_accuracy": round(self.selective_accuracy, 4),
            "n_sufficient": self.n_sufficient,
            "n_uncertain": self.n_uncertain,
            "n_insufficient": self.n_insufficient,
        }


def evaluate_gate(
    gate: BaseCalibratedGate,
    features_list: list[RetrievalFeatures],
    is_answerable: list[bool],
    is_correct: list[bool],
) -> GateMetrics:
    """Evaluate gate on labeled data.

    is_answerable[i]: True if question should be answered (not abstention)
    is_correct[i]: True if QA answer was correct (for selective accuracy)
    """
    n = len(features_list)
    n_sufficient = n_uncertain = n_insufficient = 0
    answered_correct = 0
    answered_total = 0
    correct_abstain = 0
    total_abstain = 0
    answerable_allowed = 0
    answerable_total = sum(is_answerable)

    for i, f in enumerate(features_list):
        decision = gate.assess(f)
        if decision.sufficiency == Sufficiency.SUFFICIENT:
            n_sufficient += 1
        elif decision.sufficiency == Sufficiency.UNCERTAIN:
            n_uncertain += 1
        else:
            n_insufficient += 1

        # "Answered" = SUFFICIENT or UNCERTAIN (both proceed to LLM)
        answered = decision.sufficiency != Sufficiency.INSUFFICIENT
        if answered:
            answered_total += 1
            if is_correct[i]:
                answered_correct += 1
        else:
            total_abstain += 1
            if not is_answerable[i]:
                correct_abstain += 1

        if is_answerable[i] and answered:
            answerable_allowed += 1

    return GateMetrics(
        coverage=answered_total / n if n else 0,
        answerable_recall=answerable_allowed / answerable_total if answerable_total else 0,
        abstention_precision=correct_abstain / total_abstain if total_abstain else 0,
        abstention_accuracy=correct_abstain / sum(not a for a in is_answerable) if any(not a for a in is_answerable) else 0,
        selective_accuracy=answered_correct / answered_total if answered_total else 0,
        n_sufficient=n_sufficient,
        n_uncertain=n_uncertain,
        n_insufficient=n_insufficient,
    )
