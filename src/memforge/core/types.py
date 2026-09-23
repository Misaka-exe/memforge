"""Core memory entity and enums for MemForge.

This module is the pure-Python kernel of MemForge: it depends on nothing
external (no PostgreSQL, no LLM, no embedding model) and can be tested
in isolation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    PERSONAL = "personal"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    EXPERIENCE = "experience"
    SUMMARY = "summary"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CONSOLIDATED = "consolidated"
    DORMANT = "dormant"
    DEPRECATED = "deprecated"
    FORGOTTEN = "forgotten"


class MemoryDecision(str, Enum):
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    NOOP = "noop"


class MemoryRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    RELATED = "related"
    DERIVED_FROM = "derived_from"


class MemoryEventType(str, Enum):
    ADD = "ADD"
    ACTIVATE = "ACTIVATE"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    LINK = "LINK"
    RETRIEVE = "RETRIEVE"
    DEPRECATE = "DEPRECATE"
    SLEEP = "SLEEP"
    CONSOLIDATE = "CONSOLIDATE"
    RESTORE = "RESTORE"
    FORGET = "FORGET"
    NOOP = "NOOP"


class MemorySource(str, Enum):
    CONVERSATION = "conversation"
    USER = "user"
    SYSTEM = "system"
    DERIVED = "derived"


class Memory(BaseModel):
    """A single memory entity with a full lifecycle.

    Memory is the *current state* of a fact/preference/episode; its history
    lives in MemoryEventRecord (append-only) and its provenance lives in
    MemoryEvidence.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: str
    memory_type: MemoryType = MemoryType.FACT
    user_id: str = "default"
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    importance: float = 0.5
    confidence: float = 0.5
    utility: float = 0.0
    access_count: int = 0
    last_accessed_at: datetime | None = None
    status: MemoryStatus = MemoryStatus.CANDIDATE
    source: MemorySource = MemorySource.CONVERSATION
    embedding: list[float] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    supersedes: uuid.UUID | None = None

    def touch(self) -> None:
        """Record an access on this memory."""
        self.access_count += 1
        self.last_accessed_at = utcnow()
        self.updated_at = utcnow()


class MemoryEventRecord(BaseModel):
    """An immutable, append-only event in a memory's lifecycle."""

    memory_id: uuid.UUID
    event: MemoryEventType
    from_status: MemoryStatus | None = None
    to_status: MemoryStatus | None = None
    reason: str = ""
    source_memory_id: uuid.UUID | None = None
    timestamp: datetime = Field(default_factory=utcnow)


class MemoryEvidence(BaseModel):
    """A piece of evidence tying a memory back to its original source."""

    memory_id: uuid.UUID
    session_id: str | None = None
    message_id: str | None = None
    text: str = ""
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=utcnow)


class MemoryEdge(BaseModel):
    """A typed link between two memories."""

    source_id: uuid.UUID
    target_id: uuid.UUID
    relation: MemoryRelation
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=utcnow)


class MemoryEvaluation(BaseModel):
    """Structured scoring output from the Evaluator (LLM)."""

    importance: float
    confidence: float
    reason: str = ""


class ConflictResult(BaseModel):
    """Structured output from the ConflictDetector (LLM judgement)."""

    conflict: bool
    confidence: float
    reason: str = ""


class CandidateMemory(BaseModel):
    """A raw candidate extracted from a conversation before scoring."""

    content: str
    memory_type: MemoryType = MemoryType.FACT
    should_store: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    importance: float | None = None
    confidence: float | None = None