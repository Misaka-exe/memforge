"""Core kernel: types + lifecycle state machine."""
from memforge.core.lifecycle import (
    ALLOWED_TRANSITIONS,
    IllegalTransitionError,
    LifecycleManager,
    TRANSITION_EVENTS,
)
from memforge.core.types import (
    CandidateMemory,
    ConflictResult,
    Memory,
    MemoryDecision,
    MemoryEdge,
    MemoryEvaluation,
    MemoryEventRecord,
    MemoryEventType,
    MemoryEvidence,
    MemoryRelation,
    MemorySource,
    MemoryStatus,
    MemoryType,
    utcnow,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "IllegalTransitionError",
    "LifecycleManager",
    "TRANSITION_EVENTS",
    "CandidateMemory",
    "ConflictResult",
    "Memory",
    "MemoryDecision",
    "MemoryEdge",
    "MemoryEvaluation",
    "MemoryEventRecord",
    "MemoryEventType",
    "MemoryEvidence",
    "MemoryRelation",
    "MemorySource",
    "MemoryStatus",
    "MemoryType",
    "utcnow",
]