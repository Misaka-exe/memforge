"""ConflictDetector: similarity != conflict.

Pipeline:
1. coarse vector filter (same memory_type + ACTIVE, top-10 candidates)
2. only when cosine similarity > similarity_threshold do we call the LLM
3. LLM judges whether the pair really conflicts

Outcomes:
- no conflict            -> new memory becomes ACTIVE
- conflict, conf>=thr    -> old DEPRECATED (valid_until=now), new ACTIVE,
                            SUPERSEDES edge, events recorded
- conflict, conf<thr     -> NOOP: old untouched, new stays CANDIDATE
                            (better to miss an update than to overwrite good memory)
"""
from __future__ import annotations

import math
from datetime import datetime

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
    ) -> None:
        self.action = action  # "activate" | "deprecate" | "noop"
        self.new_memory = new_memory
        self.old_memory = old_memory
        self.reason = reason
        self.events = events or []
        self.edge = edge


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
    ) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.lifecycle = lifecycle or LifecycleManager()

    async def _llm_judge(self, old: Memory, new: Memory) -> ConflictResult:
        messages = [
            Message(
                role="user",
                content=CONFLICT_PROMPT.format(old=old.content, new=new.content),
            )
        ]
        return await self.llm.generate(messages, ConflictResult)

    async def resolve(
        self,
        new_memory: Memory,
        candidates: list[Memory],
        now: datetime | None = None,
    ) -> ConflictResolution:
        """Resolve a new candidate memory against coarse-filtered candidates.

        State transitions are applied to the passed Memory objects in place;
        events and edges are returned so the caller can persist them in one
        transaction.
        """
        if not candidates:
            # No similar memory at all -> promote directly.
            event = self.lifecycle.transition(
                new_memory, MemoryStatus.ACTIVE, reason="no_candidates"
            )
            return ConflictResolution(
                action="activate", new_memory=new_memory, reason="no candidates", events=[event]
            )

        # Pick the most similar candidate.
        best = max(
            candidates,
            key=lambda c: cosine_similarity(new_memory.embedding or [], c.embedding or []),
        )
        best_sim = cosine_similarity(new_memory.embedding or [], best.embedding or [])

        if best_sim < self.similarity_threshold:
            # Not similar enough to warrant an LLM call.
            event = self.lifecycle.transition(
                new_memory, MemoryStatus.ACTIVE, reason=f"below_similarity_threshold({best_sim:.3f})"
            )
            return ConflictResolution(
                action="activate", new_memory=new_memory, reason="no similar candidate", events=[event]
            )

        result = await self._llm_judge(best, new_memory)

        if not result.conflict:
            event = self.lifecycle.transition(
                new_memory, MemoryStatus.ACTIVE, reason=result.reason or "no_conflict"
            )
            return ConflictResolution(
                action="activate", new_memory=new_memory, reason=result.reason, events=[event]
            )

        if result.confidence < self.confidence_threshold:
            # Not confident enough to overwrite the old memory: NOOP.
            noop = MemoryEventRecord(
                memory_id=new_memory.id,
                event=MemoryEventType.NOOP,
                from_status=new_memory.status,
                to_status=new_memory.status,
                reason=f"conflict_low_confidence({result.confidence:.2f}): {result.reason}",
                timestamp=self.lifecycle.now,
            )
            return ConflictResolution(
                action="noop",
                new_memory=new_memory,
                old_memory=best,
                reason=result.reason,
                events=[noop],
            )

        # Real conflict with sufficient confidence: deprecate old, activate new.
        now = now or self.lifecycle.now
        dep_event = self.lifecycle.transition(
            best, MemoryStatus.DEPRECATED, reason=result.reason or "superseded", source_memory_id=str(new_memory.id)
        )
        act_event = self.lifecycle.transition(
            new_memory, MemoryStatus.ACTIVE, reason=result.reason or "conflict", source_memory_id=str(best.id)
        )
        new_memory.supersedes = best.id
        edge = MemoryEdge(
            source_id=new_memory.id,
            target_id=best.id,
            relation=MemoryRelation.SUPERSEDES,
            confidence=result.confidence,
            created_at=now,
        )
        return ConflictResolution(
            action="deprecate",
            new_memory=new_memory,
            old_memory=best,
            reason=result.reason,
            events=[dep_event, act_event],
            edge=edge,
        )