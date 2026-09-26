"""v2 Retrieval Sufficiency Gate.

After retrieval we do NOT always force the LLM to answer. The gate inspects
the retrieved set (relevance distribution, coverage, confidence) and decides
whether the evidence is sufficient to ground an answer.

This module defines the decision object and the gate interface. The concrete
rule-based assessment lives in Phase 4; here we keep the contract so other
v2 modules (pipeline, verifier) can depend on it.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from memforge.core.types import Memory


class Sufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNCERTAIN = "UNCERTAIN"


class RetrievalDecision(BaseModel):
    query: str
    top_k_ids: list[str] = Field(default_factory=list)
    sufficiency: Sufficiency = Sufficiency.UNCERTAIN
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @property
    def should_abstain(self) -> bool:
        return self.sufficiency == Sufficiency.INSUFFICIENT


class SufficiencyGate:
    """Assess whether retrieved evidence is sufficient to answer a query.

    The gate NEVER mutates the retrieval result: it observes the top-k set and
    emits a decision. Thresholds are configurable so the v2 benchmark can
    ablate them.
    """

    def __init__(
        self,
        sufficiency_threshold: float = 0.5,
        min_relevant_count: int = 1,
        uncertain_band: float = 0.1,
    ) -> None:
        self.sufficiency_threshold = sufficiency_threshold
        self.min_relevant_count = min_relevant_count
        self.uncertain_band = uncertain_band

    def assess(
        self,
        query: str,
        memories: list[Memory],
        scores: list[float] | None = None,
    ) -> RetrievalDecision:
        """Assess whether the retrieved evidence is sufficient to answer.

        The gate is purely observational: it never mutates ``memories``.
        It computes a sufficiency level from the relevance distribution and
        the count of clearly-relevant items.

        Decision rules:
        - No memories at all -> INSUFFICIENT
        - Fewer than ``min_relevant_count`` items above the relevance floor
          -> INSUFFICIENT
        - Mean relevance >= threshold + band -> SUFFICIENT
        - Mean relevance <= threshold - band -> INSUFFICIENT
        - Otherwise -> UNCERTAIN
        """
        top_k_ids = [str(m.id) for m in memories]

        if not memories:
            return RetrievalDecision(
                query=query,
                top_k_ids=[],
                sufficiency=Sufficiency.INSUFFICIENT,
                confidence=0.0,
                reason="no retrieved memories",
            )

        # Default scores when caller does not provide them: use memory.importance
        # as a weak proxy, otherwise 0.5.
        if scores is None:
            scores = [getattr(m, "importance", 0.5) or 0.5 for m in memories]
        if len(scores) != len(memories):
            scores = list(scores) + [0.5] * (len(memories) - len(scores))

        relevance_floor = 0.3
        relevant_count = sum(1 for s in scores if s >= relevance_floor)
        mean_score = sum(scores) / len(scores)
        max_score = max(scores) if scores else 0.0

        if relevant_count < self.min_relevant_count:
            return RetrievalDecision(
                query=query,
                top_k_ids=top_k_ids,
                sufficiency=Sufficiency.INSUFFICIENT,
                confidence=max_score,
                evidence_ids=top_k_ids,
                reason=f"only {relevant_count} relevant item(s) (need {self.min_relevant_count})",
            )

        upper = self.sufficiency_threshold + self.uncertain_band
        lower = self.sufficiency_threshold - self.uncertain_band

        if mean_score >= upper:
            sufficiency = Sufficiency.SUFFICIENT
            reason = f"mean relevance {mean_score:.3f} >= {upper:.3f}"
        elif mean_score <= lower:
            sufficiency = Sufficiency.INSUFFICIENT
            reason = f"mean relevance {mean_score:.3f} <= {lower:.3f}"
        else:
            sufficiency = Sufficiency.UNCERTAIN
            reason = f"mean relevance {mean_score:.3f} in uncertain band [{lower:.3f}, {upper:.3f}]"

        return RetrievalDecision(
            query=query,
            top_k_ids=top_k_ids,
            sufficiency=sufficiency,
            confidence=mean_score,
            evidence_ids=top_k_ids,
            reason=reason,
        )
