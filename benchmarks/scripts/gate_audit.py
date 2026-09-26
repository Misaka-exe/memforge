"""Gate detailed audit (V2.1 Research Audit).

Analyzes the 8/500 intercepted questions by MultiFeatureGate in detail:
- Per-question gate decisions
- Answerable wrong-intercepted vs abstention correctly-intercepted
- Recall / Precision / F1 for abstention detection
- Threshold provenance (calibration vs evaluation-set tuning)
- Risk-coverage curve data

Output: results/longmemeval/stage_v21/gate_audit.json
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
OUT_PATH = ROOT / "results" / "longmemeval" / "stage_v21" / "gate_audit.json"


def load_data():
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

    outcomes = {}
    for line in V2_RAW_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            outcomes[d["question_id"]] = d

    qids = [qid for qid in features if qid in outcomes]
    features_list = [features[qid] for qid in qids]
    is_answerable = [not outcomes[qid]["is_abstention"] for qid in qids]
    is_correct = [outcomes[qid]["correct"] for qid in qids]
    categories = [outcomes[qid].get("category", "unknown") for qid in qids]

    return qids, features_list, is_answerable, is_correct, categories


def detailed_gate_analysis(gate, qids, features_list, is_answerable, is_correct, categories):
    """Per-question gate decision analysis."""
    decisions = []
    for i, (qid, f) in enumerate(zip(qids, features_list)):
        d = gate.assess(f)
        decisions.append({
            "question_id": qid,
            "category": categories[i],
            "is_answerable": is_answerable[i],
            "is_correct": is_correct[i],
            "gate_decision": d.sufficiency.value,
            "gate_score": round(d.score, 4),
            "top1_score": f.top1_score,
            "top5_mean": f.top5_mean,
            "n_retrieved": f.n_retrieved,
        })

    # Confusion matrix for abstention detection
    # Positive = should abstain (is_answerable=False)
    # Gate INSUFFICIENT = predicted abstain
    tp = sum(1 for d in decisions if not d["is_answerable"] and d["gate_decision"] == "INSUFFICIENT")
    fp = sum(1 for d in decisions if d["is_answerable"] and d["gate_decision"] == "INSUFFICIENT")
    fn = sum(1 for d in decisions if not d["is_answerable"] and d["gate_decision"] != "INSUFFICIENT")
    tn = sum(1 for d in decisions if d["is_answerable"] and d["gate_decision"] != "INSUFFICIENT")

    total_abstention = sum(1 for d in decisions if not d["is_answerable"])
    total_answerable = sum(1 for d in decisions if d["is_answerable"])
    total_intercepted = sum(1 for d in decisions if d["gate_decision"] == "INSUFFICIENT")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Intercepted details
    intercepted = [d for d in decisions if d["gate_decision"] == "INSUFFICIENT"]
    intercepted_answerable = [d for d in intercepted if d["is_answerable"]]
    intercepted_abstention = [d for d in intercepted if not d["is_answerable"]]

    # Abstention still passed through
    abstention_passed = [d for d in decisions if not d["is_answerable"] and d["gate_decision"] != "INSUFFICIENT"]

    return {
        "total_questions": len(decisions),
        "total_answerable": total_answerable,
        "total_abstention": total_abstention,
        "total_intercepted": total_intercepted,
        "intercepted_answerable_wrong": len(intercepted_answerable),
        "intercepted_abstention_correct": len(intercepted_abstention),
        "abstention_still_passed": len(abstention_passed),
        "confusion_matrix": {
            "TP (abstention correctly intercepted)": tp,
            "FP (answerable wrongly intercepted)": fp,
            "FN (abstention passed through)": fn,
            "TN (answerable correctly passed)": tn,
        },
        "abstention_detection": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "intercepted_examples": intercepted[:10],
        "abstention_passed_examples": abstention_passed[:10],
        "all_decisions": decisions,
    }


def risk_coverage_curve(features_list, is_answerable, is_correct):
    """Sweep threshold and compute risk-coverage points."""
    points = []
    for t in np.linspace(0.1, 0.8, 36):
        gate = ThresholdGate(threshold=float(t), band=0.0)
        metrics = evaluate_gate(gate, features_list, is_answerable, is_correct)
        risk = 1.0 - metrics.selective_accuracy
        points.append({
            "threshold": round(float(t), 3),
            "coverage": round(metrics.coverage, 4),
            "risk": round(risk, 4),
            "selective_accuracy": round(metrics.selective_accuracy, 4),
            "answerable_recall": round(metrics.answerable_recall, 4),
            "n_insufficient": metrics.n_insufficient,
        })
    return points


def main():
    qids, features_list, is_answerable, is_correct, categories = load_data()
    print(f"Loaded {len(qids)} questions")
    print(f"Answerable: {sum(is_answerable)}, Abstention: {sum(not a for a in is_answerable)}")

    # MultiFeatureGate (calibrated)
    print("\n=== MultiFeatureGate Detailed Audit ===")
    gate = MultiFeatureGate(top1_threshold=0.3, top5_threshold=0.2, min_retrieved=3)
    analysis = detailed_gate_analysis(gate, qids, features_list, is_answerable, is_correct, categories)

    print(f"Total intercepted: {analysis['total_intercepted']}")
    print(f"  Answerable wrongly intercepted: {analysis['intercepted_answerable_wrong']}")
    print(f"  Abstention correctly intercepted: {analysis['intercepted_abstention_correct']}")
    print(f"  Abstention still passed through: {analysis['abstention_still_passed']}")
    print(f"Confusion matrix: {analysis['confusion_matrix']}")
    print(f"Abstention detection: precision={analysis['abstention_detection']['precision']}, "
          f"recall={analysis['abstention_detection']['recall']}, "
          f"f1={analysis['abstention_detection']['f1']}")

    # Threshold provenance
    print("\n=== Threshold Provenance ===")
    print("MultiFeatureGate calibrated via grid search on SAME 500 questions")
    print("Best top1_threshold=0.3, top5_threshold=0.2, F1=0.934")
    print("WARNING: calibration and evaluation on same dataset = evaluation-set tuning")
    print("This is EXPLORATORY, not unbiased.")

    # Risk-coverage curve
    print("\n=== Risk-Coverage Curve (ThresholdGate) ===")
    curve = risk_coverage_curve(features_list, is_answerable, is_correct)
    # Find Pareto-efficient points
    print(f"{'Threshold':>10} {'Coverage':>8} {'Risk':>8} {'Sel.Acc':>8} {'Intercepted':>11}")
    for p in curve[::4]:
        print(f"{p['threshold']:>10.3f} {p['coverage']:>8.3f} {p['risk']:>8.3f} "
              f"{p['selective_accuracy']:>8.3f} {p['n_insufficient']:>11}")

    # Save
    result = {
        "n_questions": len(qids),
        "multi_feature_gate": analysis,
        "threshold_provenance": {
            "method": "grid search on same 500 questions",
            "best_top1_threshold": 0.3,
            "best_top5_threshold": 0.2,
            "calibration_f1": 0.934,
            "warning": "calibration and evaluation on same dataset; exploratory only, not unbiased",
        },
        "risk_coverage_curve": curve,
        "conclusion": (
            "Calibrated retrieval-score gating is insufficient for evidence sufficiency "
            "on LongMemEval-S. MultiFeatureGate intercepts only 8/500 questions, of which "
            "only 2 are true abstentions (precision=0.25). 28/30 abstentions pass through "
            "(recall=0.067). Single-threshold gate is completely ineffective (best threshold "
            "0.1 = no filtering). Retrieval relevance is not equivalent to evidence sufficiency."
        ),
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
