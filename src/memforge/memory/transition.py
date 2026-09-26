"""v2 Evidence-driven transition records.

v1 changed memory state based on threshold rules (similarity > 0.75 -> deprecate).
v2 records *why* a transition happened: which evidence drove the decision, how
confident the system was, and which module produced the decision.

MemoryTransition is a complementary audit log. It does NOT replace
MemoryEventRecord (which remains the append-only lifecycle event stream emitted
by LifecycleManager); a MemoryTransition records the *decision rationale* that
led to one or more lifecycle events.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterable

from pydantic import BaseModel, Field

from memforge.core.types import MemoryStatus, utcnow


class MemoryTransition(BaseModel):
    """An auditable record of a state-changing (or state-preserving) decision.

    A transition may be *destructive* (to_state == DEPRECATED / FORGOTTEN) or
    *non-destructive* (to_state == from_state, e.g. UNCERTAIN conflict that
    decided to leave both memories untouched). Non-destructive transitions are
    still recorded so the decision trail is complete.
    """

    memory_id: uuid.UUID
    from_state: MemoryStatus
    to_state: MemoryStatus
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    decision_source: str = "unknown"  # module that produced the decision
    destructive: bool = False
    timestamp: datetime = Field(default_factory=utcnow)


class TransitionLog:
    """In-memory append-only audit log of v2 decisions.

    Not persisted to the database by design (no new DB schema in v2); callers
    that need durability can iterate ``records`` and write them wherever they
    like. This keeps v2 additive: v1 repository/lifecycle stay untouched.
    """

    def __init__(self) -> None:
        self._records: list[MemoryTransition] = []

    def record(self, transition: MemoryTransition) -> MemoryTransition:
        self._records.append(transition)
        return transition

    def records_for(self, memory_id: uuid.UUID | str) -> list[MemoryTransition]:
        key = str(memory_id)
        return [r for r in self._records if str(r.memory_id) == key]

    def all(self) -> list[MemoryTransition]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[MemoryTransition]:
        return iter(self._records)
