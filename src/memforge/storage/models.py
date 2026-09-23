"""SQLAlchemy ORM models for MemForge storage.

Four tables:
- memories:        the current state of each memory
- memory_events:   append-only lifecycle history
- memory_evidence: provenance links to original conversation
- memory_edges:    typed links between memories
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memforge.core.types import (
    MemoryEventType,
    MemoryRelation,
    MemorySource,
    MemoryStatus,
    MemoryType,
)


class Base(DeclarativeBase):
    pass


class MemoryORM(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, default=MemoryType.FACT.value)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    utility: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=MemoryStatus.CANDIDATE.value, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default=MemorySource.CONVERSATION.value)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=True)
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    supersedes: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)


class MemoryEventORM(Base):
    __tablename__ = "memory_events"
    __table_args__ = (UniqueConstraint("memory_id", "timestamp", "event", name="uq_event_once"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    event: Mapped[str] = mapped_column(String(32), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryEvidenceORM(Base):
    __tablename__ = "memory_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MemoryEdgeORM(Base):
    __tablename__ = "memory_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())