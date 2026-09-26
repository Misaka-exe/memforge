"""V2.1 unified evaluation metrics.

Consolidates retrieval, QA, gate, evidence preservation, and efficiency
metrics into a single structured output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V21Metrics:
    """Unified metrics for a V2.1 experiment run."""

    # Config
    system_name: str = ""
    n_questions: int = 0
    mode: str = "static"  # static or adaptive
    enable_gate: bool = False
    enable_temporal: bool = False
    enable_reconsolidation: bool = False

    # Retrieval
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_all_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # QA
    overall_accuracy: float = 0.0
    answerable_accuracy: float = 0.0
    abstention_accuracy: float = 0.0
    abstention_rate: float = 0.0

    # Gate
    gate_coverage: float = 0.0
    gate_sufficient_rate: float = 0.0
    gate_uncertain_rate: float = 0.0
    gate_insufficient_rate: float = 0.0
    gate_answerable_recall: float = 0.0
    gate_abstention_precision: float = 0.0

    # Evidence / Grounding
    avg_citations: float = 0.0
    verification_pass_rate: float = 0.0
    verification_warn_rate: float = 0.0
    verification_fail_rate: float = 0.0
    unsupported_claim_rate: float = 0.0

    # Memory / Reconsolidation
    evidence_preservation_rate: float = 1.0
    evidence_loss_rate: float = 0.0
    memory_update_count: int = 0
    false_update_rate: float = 0.0
    reconsolidation_triggered: int = 0

    # Temporal
    future_leakage_rate: float = 0.0

    # Efficiency
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0

    # By category
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "n_questions": self.n_questions,
            "mode": self.mode,
            "config": {
                "enable_gate": self.enable_gate,
                "enable_temporal": self.enable_temporal,
                "enable_reconsolidation": self.enable_reconsolidation,
            },
            "retrieval": {
                "recall@1": round(self.recall_at_1, 4),
                "recall@5": round(self.recall_at_5, 4),
                "recall_all@5": round(self.recall_all_at_5, 4),
                "mrr": round(self.mrr, 4),
                "ndcg@10": round(self.ndcg_at_10, 4),
            },
            "qa": {
                "overall": round(self.overall_accuracy, 4),
                "answerable": round(self.answerable_accuracy, 4),
                "abstention": round(self.abstention_accuracy, 4),
                "abstention_rate": round(self.abstention_rate, 4),
            },
            "gate": {
                "coverage": round(self.gate_coverage, 4),
                "sufficient_rate": round(self.gate_sufficient_rate, 4),
                "uncertain_rate": round(self.gate_uncertain_rate, 4),
                "insufficient_rate": round(self.gate_insufficient_rate, 4),
                "answerable_recall": round(self.gate_answerable_recall, 4),
                "abstention_precision": round(self.gate_abstention_precision, 4),
            },
            "evidence": {
                "avg_citations": round(self.avg_citations, 3),
                "verification_pass": round(self.verification_pass_rate, 4),
                "verification_warn": round(self.verification_warn_rate, 4),
                "verification_fail": round(self.verification_fail_rate, 4),
                "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
            },
            "memory": {
                "evidence_preservation_rate": round(self.evidence_preservation_rate, 4),
                "evidence_loss_rate": round(self.evidence_loss_rate, 4),
                "memory_update_count": self.memory_update_count,
                "false_update_rate": round(self.false_update_rate, 4),
                "reconsolidation_triggered": self.reconsolidation_triggered,
            },
            "temporal": {
                "future_leakage_rate": round(self.future_leakage_rate, 4),
            },
            "efficiency": {
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "api_calls": self.api_calls,
            },
            "by_category": self.by_category,
        }
