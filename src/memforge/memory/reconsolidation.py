"""Evidence-Grounded Reconsolidation for V2.1 Phase 4.

Core principle: NEVER overwrite a memory. Instead, create a new version
with lineage tracking, and deprecate the old one.

Data model:
- lineage_id: groups all versions of the same conceptual memory
- version: integer, increments on each reconsolidation
- derived_from: UUID of the parent memory
- supersedes: UUID of the memory this version replaces (set on old memory)

Trigger: ONLY when post-hoc verification returns FAIL.
Decision: KEEP / UPDATE / ABSTAIN (first version only supports KEEP/UPDATE/ABSTAIN).

Slot-level updates: only modify the specific fields that new evidence supports,
preserving all other fields from the previous version.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from memforge.core.types import Memory, MemoryStatus


class ReconsolidationDecision(str, Enum):
    KEEP = "KEEP"          # No change; existing memory is correct
    UPDATE = "UPDATE"      # Create new version with updated content
    ABSTAIN = "ABSTAIN"    # Insufficient evidence; do nothing


@dataclass
class ReconsolidationResult:
    """Result of a reconsolidation attempt."""
    decision: ReconsolidationDecision
    original_memory_id: uuid.UUID
    new_memory_id: Optional[uuid.UUID] = None
    lineage_id: Optional[str] = None
    version: Optional[int] = None
    reason: str = ""
    updated_fields: list[str] = field(default_factory=list)
    evidence_text: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "original_memory_id": str(self.original_memory_id),
            "new_memory_id": str(self.new_memory_id) if self.new_memory_id else None,
            "lineage_id": self.lineage_id,
            "version": self.version,
            "reason": self.reason,
            "updated_fields": self.updated_fields,
            "evidence_text": self.evidence_text,
        }


class MemoryReconsolidator:
    """Evidence-grounded memory reconsolidation engine.

    Usage:
        reconsolidator = MemoryReconsolidator()
        result = reconsolidator.reconsolidate(
            memory=old_memory,
            new_evidence="Alice moved to Shanghai.",
            verification_fail_reason="answer contained unsupported claim about location",
        )
        if result.decision == UPDATE:
            # result contains new_memory with lineage tracking
            # old_memory should be set to DEPRECATED
            pass
    """

    def __init__(self, min_confidence: float = 0.6, slot_separator: str = "\n"):
        self.min_confidence = min_confidence
        self.slot_separator = slot_separator

    def reconsolidate(
        self,
        memory: Memory,
        new_evidence: str,
        verification_fail_reason: str = "",
        confidence: float = 0.8,
    ) -> ReconsolidationResult:
        """Attempt to reconsolidate a memory with new evidence.

        Args:
            memory: The existing memory to potentially update.
            new_evidence: Text of new evidence from the failed answer.
            verification_fail_reason: Why verification failed.
            confidence: Confidence in the new evidence [0, 1].

        Returns:
            ReconsolidationResult with decision and new memory if UPDATE.
        """
        # Guard: low confidence → ABSTAIN
        if confidence < self.min_confidence:
            return ReconsolidationResult(
                decision=ReconsolidationDecision.ABSTAIN,
                original_memory_id=memory.id,
                reason=f"confidence {confidence:.2f} < min {self.min_confidence:.2f}",
                evidence_text=new_evidence,
            )

        # Guard: empty evidence → KEEP
        if not new_evidence.strip():
            return ReconsolidationResult(
                decision=ReconsolidationDecision.KEEP,
                original_memory_id=memory.id,
                reason="empty new evidence",
                evidence_text="",
            )

        # Determine if evidence actually contradicts/extends the memory
        updated_fields = self._identify_updated_fields(memory, new_evidence)

        if not updated_fields:
            return ReconsolidationResult(
                decision=ReconsolidationDecision.KEEP,
                original_memory_id=memory.id,
                reason="no updatable fields identified in new evidence",
                evidence_text=new_evidence,
            )

        # Create new version
        new_memory = self._create_new_version(memory, new_evidence, updated_fields)

        return ReconsolidationResult(
            decision=ReconsolidationDecision.UPDATE,
            original_memory_id=memory.id,
            new_memory_id=new_memory.id,
            lineage_id=new_memory.metadata.get("lineage_id"),
            version=new_memory.metadata.get("version"),
            reason=f"updated fields: {', '.join(updated_fields)}",
            updated_fields=updated_fields,
            evidence_text=new_evidence,
        )

    def _identify_updated_fields(self, memory: Memory, new_evidence: str) -> list[str]:
        """Identify which slots/fields the new evidence updates.

        Simple keyword-based slot detection. In a full implementation,
        this would use an LLM or structured extraction.
        """
        evidence_lower = new_evidence.lower()
        content_lower = memory.content.lower()
        fields = []

        # Common slot patterns
        slot_keywords = {
            "location": ["live", "lives", "moved", "city", "location", "address", "reside"],
            "job": ["job", "work", "works", "employed", "occupation", "position", "role"],
            "preference": ["prefer", "likes", "favorite", "hate", "dislike", "want"],
            "relationship": ["married", "divorced", "partner", "spouse", "friend"],
            "name": ["name", "called", "known as"],
        }

        for slot, keywords in slot_keywords.items():
            evidence_has = any(kw in evidence_lower for kw in keywords)
            content_has = any(kw in content_lower for kw in keywords)
            if evidence_has and content_has:
                fields.append(slot)

        return fields

    def _create_new_version(
        self,
        old_memory: Memory,
        new_evidence: str,
        updated_fields: list[str],
    ) -> Memory:
        """Create a new version of the memory with lineage tracking."""
        # Lineage: use existing lineage_id or create new one
        lineage_id = old_memory.metadata.get("lineage_id", str(uuid.uuid4()))
        old_version = old_memory.metadata.get("version", 1)

        # Slot-level update: preserve old content, append new evidence
        # In a full implementation, this would merge at field level.
        new_content = self._slot_merge(old_memory.content, new_evidence, updated_fields)

        new_memory = Memory(
            content=new_content,
            memory_type=old_memory.memory_type,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            valid_from=datetime.utcnow(),
            importance=old_memory.importance,
            confidence=max(old_memory.confidence, 0.7),
            status=MemoryStatus.ACTIVE,
            source=old_memory.source,
            tags=list(old_memory.tags),
            metadata={
                **old_memory.metadata,
                "lineage_id": lineage_id,
                "version": old_version + 1,
                "derived_from": str(old_memory.id),
                "reconsolidation_reason": "verification_fail",
                "updated_fields": updated_fields,
            },
        )

        return new_memory

    def _slot_merge(self, old_content: str, new_evidence: str, updated_fields: list[str]) -> str:
        """Merge new evidence into old content at slot level.

        Simple implementation: append new evidence as an update note.
        Full implementation would parse and replace specific fields.
        """
        if not updated_fields:
            return old_content

        return f"{old_content}\n[Updated] {new_evidence}"

    def deprecate_old(self, old_memory: Memory) -> Memory:
        """Mark the old memory as DEPRECATED after reconsolidation."""
        old_memory.status = MemoryStatus.DEPRECATED
        old_memory.updated_at = datetime.utcnow()
        return old_memory


def compute_evidence_preservation_rate(
    gold_memory_ids: set[str],
    memories_after: list[Memory],
) -> float:
    """Compute Evidence Preservation Rate.

    EPR = gold evidence still retrievable after memory management
          / gold evidence before memory management
    """
    if not gold_memory_ids:
        return 1.0
    after_ids = {str(m.id) for m in memories_after if m.status != MemoryStatus.DEPRECATED}
    # Also include lineage: if a gold memory was reconsolidated, its version
    # should still be retrievable via lineage_id
    after_lineage = {m.metadata.get("lineage_id") for m in memories_after
                     if m.metadata.get("lineage_id")}
    preserved = 0
    for gid in gold_memory_ids:
        if gid in after_ids:
            preserved += 1
        elif any(gid in str(m.metadata.get("derived_from", "")) for m in memories_after):
            preserved += 1
    return preserved / len(gold_memory_ids)


def compute_evidence_loss_rate(
    gold_memory_ids: set[str],
    memories_after: list[Memory],
) -> float:
    """Evidence Loss Rate = 1 - EPR."""
    return 1.0 - compute_evidence_preservation_rate(gold_memory_ids, memories_after)
