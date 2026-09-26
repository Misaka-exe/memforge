"""Gate calibration experiment (V2.1 Phase 2).

Uses real retrieval features + V2's QA outcomes to calibrate and evaluate
sufficiency gates. This is EXPLORATORY calibration on the same dataset,
not an unbiased held-out evaluation.

Output: results/longmemeval/stage_v21/gate_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from memforge.retrieval.gate_v2 import (
    MultiFeatureGate,
    Sufficiency,
    ThresholdGate,
    evaluate_gate,
)
from memforge.retrieval.score import RetrievalFeatures

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "results" / "longmemeval" / "stage_v21" / "retrieval_features.jsonl"
V2_RAW_PATH = ROOT / "results" / "longmemeval" / "stage5_v2" / "raw_qa.jsonl"
OUT_PATH = ROOT / "results" / "longmemeval" / "stage_v21" / "gate_calibration.json"


def load_data():
    # Load features
    features = {}
    for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            features[d["question_id"]] = RetrievalFeatures(
                question_id=d["question_id"],
                n_retrieved=d["n_retrieved"],
                top1_score=d["top1_score"],
                top5_mean=d["top5_mean"],
                score_margin=d["score_margin"],
                mean_score=d["mean_score"],
                std_score=d["std_score"],
            )

    # Load V2 QA outcomes
    outcomes = {}
    for line in V2_RAW_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            outcomes[d["question_id"]] = d

    # Align
    qids = [qid for qid in features if qid in outcomes]
    features_list = [features[qid] for qid in qids]
    is_answerable = [not outcomes[qid]["is_abstention"] for qid in qids]
    is_correct = [outcomes[qid]["correct"] for qid in qids]

    return features_list, is_answerable, is_correct, qids


def risk_coverage_curve(gate_cls, features_list, is_answerable, is_correct, **kwargs):
    """Compute risk-coverage curve by sweeping threshold."""
    points = []
    for t in np.linspace(0.2, 0.8, 25):
        gate = gate_cls(threshold=float(t), **kwargs) if gate_cls == ThresholdGate else gate_cls(top1_threshold=float(t), **kwargs)
        metrics = evaluate_gate(gate, features_list, is_answerable, is_correct)
        # risk = 1 - selective_accuracy on answered
        risk = 1.0 - metrics.selective_accuracy
        points.append({
            "threshold": round(float(t), 3),
            "coverage": metrics.coverage,
            "risk": round(risk, 4),
            "selective_accuracy": metrics.selective_accuracy,
            "answerable_recall": metrics.answerable_recall,
        })
    return points


def main():
    features_list, is_answerable, is_correct, qids = load_data()
    print(f"Loaded {len(qids)} questions with features + outcomes")
    print(f"Answerable: {sum(is_answerable)}, Abstention: {sum(not a for a in is_answerable)}")

    results = {
        "n_questions": len(qids),
        "n_answerable": sum(is_answerable),
        "n_abstention": sum(not a for a in is_answerable),
        "calibration_note": "EXPLORATORY: calibrated and evaluated on same 500 questions. Not unbiased.",
        "gates": {},
    }

    # --- ThresholdGate ---
    print("\n=== ThresholdGate ===")
    gate = ThresholdGate(threshold=0.5)
    calib = gate.calibrate(features_list, is_answerable)
    print(f"Calibrated threshold: {calib['best_threshold']}, F1: {calib['best_f1']}")
    metrics = evaluate_gate(gate, features_list, is_answerable, is_correct)
    print(f"Metrics: {metrics.to_dict()}")
    results["gates"]["threshold"] = {
        "calibration": calib,
        "metrics": metrics.to_dict(),
    }

    # --- MultiFeatureGate ---
    print("\n=== MultiFeatureGate ===")
    gate2 = MultiFeatureGate()
    calib2 = gate2.calibrate(features_list, is_answerable)
    print(f"Calibrated: top1={calib2['best_top1_threshold']}, top5={calib2['best_top5_threshold']}, F1: {calib2['best_f1']}")
    metrics2 = evaluate_gate(gate2, features_list, is_answerable, is_correct)
    print(f"Metrics: {metrics2.to_dict()}")
    results["gates"]["multifeature"] = {
        "calibration": calib2,
        "metrics": metrics2.to_dict(),
    }

    # --- V2 Gate baseline (100% SUFFICIENT) ---
    print("\n=== V2 Gate Baseline (importance proxy, 100% SUFFICIENT) ===")
    v2_coverage = 1.0  # V2 gate always SUFFICIENT
    v2_selective_acc = sum(is_correct) / len(is_correct)
    results["gates"]["v2_baseline"] = {
        "description": "V2 gate using memory.importance proxy → 100% SUFFICIENT",
        "coverage": 1.0,
        "selective_accuracy": round(v2_selective_acc, 4),
        "note": "No abstention at all; coverage=1.0 by construction",
    }

    # --- Risk-Coverage Curve (ThresholdGate) ---
    print("\n=== Risk-Coverage Curve (ThresholdGate) ===")
    curve = risk_coverage_curve(ThresholdGate, features_list, is_answerable, is_correct)
    results["risk_coverage_curve"] = curve

    # --- Feature distribution by answerable/abstention ---
    ans_top1 = [f.top1_score for f, a in zip(features_list, is_answerable) if a]
    abs_top1 = [f.top1_score for f, a in zip(features_list, is_answerable) if not a]
    results["feature_distribution"] = {
        "answerable_top1_mean": round(float(np.mean(ans_top1)), 4),
        "answerable_top1_std": round(float(np.std(ans_top1)), 4),
        "abstention_top1_mean": round(float(np.mean(abs_top1)), 4),
        "abstention_top1_std": round(float(np.std(abs_top1)), 4),
        "diff": round(float(np.mean(ans_top1) - np.mean(abs_top1)), 4),
        "note": "Weak separability: diff small relative to std",
    }

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
