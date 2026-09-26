"""ConflictDetector: similarity != conflict.

v1 pipeline:
1. coarse vector filter (same memory_type + ACTIVE, top-10 candidates)
2. only when cosine similarity > similarity_threshold do we call the LLM
3. LLM judges whether the pair really conflicts

v2 adds a typed ``ConflictDecision`` between the LLM judgement and the state
mutation. A similarity spike no longer means "overwrite the old memory":
- NO_CONFLICT       -> new memory becomes ACTIVE
- CONFIRMED_CONFLICT (conf >= threshold) -> old DEPRECATED, new ACTIVE
- POSSIBLE_CONFLICT -> event recorded, NO state change (v3 reconsolidation hook)
- UNCERTAIN         -> event recorded, NO destructive change (old stays ACTIVE,
                       new stays CANDIDATE). Better to miss an update than to
                       overwrite good memory.

All v2 decisions are mirrored into a TransitionLog for auditability.
"""
from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from memforge.core.lifecycle import LifecycleManager
from memforge.core.types import (
    ConflictResult,
    Memory,
    MemoryEdge,
    MemoryEventRecord,
    MemoryEventType,
    MemoryRelation,
    MemoryStatus,
)
from memforge.llm.base import LLMProvider, Message
from memforge.memory.transition import MemoryTransition, TransitionLog


CONFLICT_PROMPT = """You are a memory conflict judge for a personal assistant.

Two memory statements about the same user are given. Decide whether the NEW
statement contradicts the OLD one such that the old should be superseded.

IMPORTANT: semantic similarity is NOT conflict. For example:
- OLD: "User prefers lightweight phones."
- NEW: "User prefers phones with long battery life."
These are related but NOT contradictory; both can be true.

A real conflict looks like:
- OLD: "User lives in Beijing."
- NEW: "User moved to Shanghai."

Return JSON: {{"conflict": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}

OLD memory: {old}
NEW memory: {new}
"""


class ConflictResolution:
    """Outcome of a conflict resolution pass."""

    def __init__(
        self,
        action: str,
        new_memory: Memory,
        old_memory: Memory | None = None,
        reason: str = "",
        events: list[MemoryEventRecord] | None = None,
        edge: MemoryEdge | None = None,
        decision: "ConflictDecision | None" = None,
    ) -> None:
        self.action = action  # "activate" | "deprecate" | "noop"
        self.new_memory = new_memory
        self.old_memory = old_memory
        self.reason = reason
        self.events = events or []
        self.edge = edge
        self.decision = decision


# ---------------------------------------------------------------------------
# v2 typed conflict decision
# ---------------------------------------------------------------------------

class ConflictType(str, Enum):
    NO_CONFLICT = "NO_CONFLICT"
    POSSIBLE_CONFLICT = "POSSIBLE_CONFLICT"
    CONFIRMED_CONFLICT = "CONFIRMED_CONFLICT"
    UNCERTAIN = "UNCERTAIN"


class ConflictDecision(BaseModel):
    type: ConflictType
    confidence: float = 0.0
    affected_fields: list[str] = []
    evidence_ids: list[str] = []
    reason: str = ""


class ConflictClassifier:
    """Map (similarity, LLM judgement) -> a typed ConflictDecision.

    Pure rule-based on top of the LLM's ``ConflictResult``. No destructive
    default: ambiguous cases resolve to POSSIBLE_CONFLICT / UNCERTAIN.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        confirm_threshold: float = 0.7,
        possible_threshold: float = 0.5,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.confirm_threshold = confirm_threshold
        self.possible_threshold = possible_threshold

    def classify(
        self,
        best_similarity: float,
        llm_result: ConflictResult | None,
        has_candidates: bool,
    ) -> ConflictDecision:
        if not has_candidates:
            return ConflictDecision(
                type=ConflictType.NO_CONFLICT,
                confidence=1.0,
                reason="no_candidates",
            )
        if best_similarity < self.similarity_threshold:
            return ConflictDecision(
                type=ConflictType.NO_CONFLICT,
                confidence=1.0 - best_similarity,
                reason=f"below_similarity_threshold({best_similarity:.3f})",
            )
        if llm_result is None:
            # No LLM result available (e.g. call failed): never destructive.
            return ConflictDecision(
                type=ConflictType.UNCERTAIN,
                confidence=best_similarity,
                reason="no_llm_judgement",
            )
        if not llm_result.conflict:
            return ConflictDecision(
                type=ConflictType.NO_CONFLICT,
                confidence=llm_result.confidence,
                reason=llm_result.reason or "llm_says_no_conflict",
            )
        if llm_result.confidence >= self.confirm_threshold:
            return ConflictDecision(
                type=ConflictType.CONFIRMED_CONFLICT,
                confidence=llm_result.confidence,
                reason=llm_result.reason or "confirmed_conflict",
            )
        if llm_result.confidence >= self.possible_threshold:
            return ConflictDecision(
                type=ConflictType.POSSIBLE_CONFLICT,
                confidence=llm_result.confidence,
                reason=llm_result.reason or "possible_conflict",
            )
        return ConflictDecision(
            type=ConflictType.UNCERTAIN,
            confidence=llm_result.confidence,
            reason=llm_result.reason or "uncertain_conflict",
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ConflictDetector:
    def __init__(
        self,
        llm: LLMProvider,
        similarity_threshold: float = 0.75,
        confidence_threshold: float = 0.7,
        lifecycle: LifecycleManager | None = None,
        classifier: ConflictClassifier | None = None,
        transition_log: TransitionLog | None = None,
    ) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.lifecycle = lifecycle or LifecycleManager()
        self.classifier = classifier or ConflictClassifier(
            similarity_threshold=similarity_threshold,
            confirm_threshold=confidence_threshold,
        )
        # NOTE: use is-not-None, not `or` -- an empty TransitionLog has
        # __len__ == 0 and would be treated as falsy, discarding injected log.
        self.transition_log = transition_log if transition_log is not None else TransitionLog()

    async def _llm_judge(self, old: Memory, new: Memory) -> ConflictResult:
        messages = [
            Message(
                role="user",
                content=CONFLICT_PROMPT.format(old=old.content, new=new.content),
            )
        ]
        return await self.llm.generate(messages, ConflictResult)

    def _log(self, memory: Memory, from_state: MemoryStatus, to_state: MemoryStatus,
             reason: str, confidence: float, destructive: bool) -> None:
        self.transition_log.record(
            MemoryTransition(
                memory_id=memory.id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                confidence=confidence,
                decision_source="conflict_v2",
                destructive=destructive,
                timestamp=self.lifecycle.now,
            )
        )

    async def resolve(
        self,
        new_memory: Memory,
        candidates: list[Memory],
        now: datetime | None = None,
    ) -> ConflictResolution:
        """Resolve a new candidate memory against coarse-filtered candidates.

        State transitions are applied to the passed Memory objects in place;
        events and edges are returned so the caller can persist them in one
        transaction. Every v2 decision is also appended to ``transition_log``.
        """
        # Pick the most similar candidate (or None).
        best: Memory | None = None
        best_sim = 0.0
        if candidates:
            best = max(
                candidates,
                key=lambda c: cosine_similarity(new_memory.embedding or [], c.embedding or []),
            )
            best_sim = cosine_similarity(new_memory.embedding or [], best.embedding or [])

        llm_result: ConflictResult | None = None
        if best is not None and best_sim >= self.similarity_threshold:
            llm_result = await self._llm_judge(best, new_memory)

        decision = self.classifier.classify(
            best_similarity=best_sim,
            llm_result=llm_result,
            has_candidates=best is not None,
        )

        # ---- NO_CONFLICT: promote new memory to ACTIVE ---------------------
        if decision.type == ConflictType.NO_CONFLICT:
            new_from = new_memory.status
            event = self.lifecycle.transition(
                new_memory, MemoryStatus.ACTIVE, reason=decision.reason
            )
            self._log(new_memory, new_from, MemoryStatus.ACTIVE, decision.reason,
                      decision.confidence, destructive=False)
            return ConflictResolution(
                action="activate", new_memory=new_memory,
                old_memory=None, reason=decision.reason,
                events=[event], decision=decision,
            )

        # ---- CONFIRMED_CONFLICT: deprecate old, activate new --------------
        if decision.type == ConflictType.CONFIRMED_CONFLICT and best is not None:
            now = now or self.lifecycle.now
            old_from = best.status
            new_from = new_memory.status
            dep_event = self.lifecycle.transition(
                best, MemoryStatus.DEPRECATED, reason=decision.reason,
                source_memory_id=str(new_memory.id),
            )
            self._log(best, old_from, MemoryStatus.DEPRECATED, decision.reason,
                      decision.confidence, destructive=True)
            act_event = self.lifecycle.transition(
                new_memory, MemoryStatus.ACTIVE, reason=decision.reason,
                source_memory_id=str(best.id),
            )
            self._log(new_memory, new_from, MemoryStatus.ACTIVE, decision.reason,
                      decision.confidence, destructive=False)
            new_memory.supersedes = best.id
            edge = MemoryEdge(
                source_id=new_memory.id,
                target_id=best.id,
                relation=MemoryRelation.SUPERSEDES,
                confidence=decision.confidence,
                created_at=now,
            )
            return ConflictResolution(
                action="deprecate", new_memory=new_memory, old_memory=best,
                reason=decision.reason,
                events=[dep_event, act_event], edge=edge, decision=decision,
            )

        # ---- POSSIBLE_CONFLICT / UNCERTAIN: NO state change ----------------
        # Record the decision as a non-destructive NOOP event on the new memory.
        # Old memory is NEVER touched. This is the core v2 safety guarantee:
        # similarity spikes and weak LLM judgements cannot destroy gold memory.
        noop = MemoryEventRecord(
            memory_id=new_memory.id,
            event=MemoryEventType.NOOP,
            from_status=new_memory.status,
            to_status=new_memory.status,
            reason=f"{decision.type.value}: {decision.reason}",
            timestamp=self.lifecycle.now,
        )
        self._log(new_memory, new_memory.status, new_memory.status,
                  f"{decision.type.value}: {decision.reason}",
                  decision.confidence, destructive=False)
        return ConflictResolution(
            action="noop",
            new_memory=new_memory,
            old_memory=best,
            reason=decision.reason,
            events=[noop],
            decision=decision,
        )