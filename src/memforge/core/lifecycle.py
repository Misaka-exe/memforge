"""LifecycleManager: the ONLY entry point for changing a memory's status.

Design rules:
- Status transitions must go through LifecycleManager.transition().
- ORM / API / Repository layers must never assign memory.status directly.
- Every transition appends an immutable MemoryEventRecord.
- The first DEPRECATE stamps valid_until automatically.
"""
from __future__ import annotations

from datetime import datetime

from memforge.core.types import (
    Memory,
    MemoryEventRecord,
    MemoryEventType,
    MemoryStatus,
)


class IllegalTransitionError(Exception):
    """Raised when a memory status transition is not allowed."""

    def __init__(self, memory: Memory, to_status: MemoryStatus) -> None:
        self.memory_id = memory.id
        self.from_status = memory.status
        self.to_status = to_status
        super().__init__(
            f"Illegal memory transition: {self.from_status.value} -> {self.to_status.value} "
            f"(memory {self.memory_id})"
        )


# The full legal transition graph.
ALLOWED_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.CANDIDATE: {MemoryStatus.ACTIVE},
    MemoryStatus.ACTIVE: {
        MemoryStatus.DORMANT,
        MemoryStatus.CONSOLIDATED,
        MemoryStatus.DEPRECATED,
    },
    MemoryStatus.CONSOLIDATED: {MemoryStatus.DORMANT},
    MemoryStatus.DORMANT: {MemoryStatus.ACTIVE, MemoryStatus.DEPRECATED},
    MemoryStatus.DEPRECATED: {MemoryStatus.FORGOTTEN},
    MemoryStatus.FORGOTTEN: set(),
}

# Event emitted for each legal transition.
TRANSITION_EVENTS: dict[tuple[MemoryStatus, MemoryStatus], MemoryEventType] = {
    (MemoryStatus.CANDIDATE, MemoryStatus.ACTIVE): MemoryEventType.ACTIVATE,
    (MemoryStatus.ACTIVE, MemoryStatus.DORMANT): MemoryEventType.SLEEP,
    (MemoryStatus.ACTIVE, MemoryStatus.CONSOLIDATED): MemoryEventType.CONSOLIDATE,
    (MemoryStatus.CONSOLIDATED, MemoryStatus.DORMANT): MemoryEventType.SLEEP,
    (MemoryStatus.ACTIVE, MemoryStatus.DEPRECATED): MemoryEventType.DEPRECATE,
    (MemoryStatus.DORMANT, MemoryStatus.ACTIVE): MemoryEventType.ACTIVATE,
    (MemoryStatus.DORMANT, MemoryStatus.DEPRECATED): MemoryEventType.DEPRECATE,
    (MemoryStatus.DEPRECATED, MemoryStatus.FORGOTTEN): MemoryEventType.FORGET,
}


class LifecycleManager:
    """State machine executor. Injects an optional clock for deterministic tests."""

    def __init__(self, now: datetime | None = None) -> None:
        self._now = now

    @property
    def now(self) -> datetime:
        if self._now is not None:
            return self._now
        from memforge.core.types import utcnow

        return utcnow()

    def transition(
        self,
        memory: Memory,
        to_status: MemoryStatus,
        reason: str = "",
        source_memory_id: str | None = None,
    ) -> MemoryEventRecord:
        """Validate and apply a transition, returning the event record.

        Raises IllegalTransitionError when the transition is not in
        ALLOWED_TRANSITIONS or is a no-op (same status).
        """
        if memory.status == to_status:
            raise IllegalTransitionError(memory, to_status)

        allowed = ALLOWED_TRANSITIONS.get(memory.status, set())
        if to_status not in allowed:
            raise IllegalTransitionError(memory, to_status)

        from_status = memory.status
        now = self.now

        memory.status = to_status
        memory.updated_at = now

        # First DEPRECATE stamps valid_until automatically.
        if to_status == MemoryStatus.DEPRECATED and memory.valid_until is None:
            memory.valid_until = now

        event = MemoryEventRecord(
            memory_id=memory.id,
            event=TRANSITION_EVENTS[(from_status, to_status)],
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            source_memory_id=source_memory_id,
            timestamp=now,
        )
        return event