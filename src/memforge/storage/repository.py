"""Repository layer: pure SQL, no LLM decisions.

Design constraints:
- Repository NEVER decides status transitions.
- update() changes business fields only, never status.
- add_event() is append-only (insert only).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Sequence

from sqlalchemy import asc, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from memforge.core.types import (
    Memory,
    MemoryEdge,
    MemoryEventRecord,
    MemoryEvidence,
    MemoryStatus,
)
from memforge.storage.models import (
    MemoryEdgeORM,
    MemoryEventORM,
    MemoryEvidenceORM,
    MemoryORM,
)


def _orm_to_memory(orm: MemoryORM) -> Memory:
    return Memory(
        id=orm.id,
        content=orm.content,
        memory_type=orm.memory_type,  # type: ignore[arg-type]
        user_id=orm.user_id,
        session_id=orm.session_id,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        valid_from=orm.valid_from,
        valid_until=orm.valid_until,
        importance=orm.importance,
        confidence=orm.confidence,
        utility=orm.utility,
        access_count=orm.access_count,
        last_accessed_at=orm.last_accessed_at,
        status=orm.status,  # type: ignore[arg-type]
        source=orm.source,  # type: ignore[arg-type]
        embedding=list(orm.embedding or []),
        tags=list(orm.tags or []),
        metadata=dict(orm.metadata or {}),
        supersedes=orm.supersedes,
    )


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------ CRUD
    async def add(self, memory: Memory) -> Memory:
        orm = MemoryORM(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type.value,
            user_id=memory.user_id,
            session_id=memory.session_id,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            valid_from=memory.valid_from,
            valid_until=memory.valid_until,
            importance=memory.importance,
            confidence=memory.confidence,
            utility=memory.utility,
            access_count=memory.access_count,
            last_accessed_at=memory.last_accessed_at,
            status=memory.status.value,
            source=memory.source.value,
            embedding=memory.embedding or None,
            tags=memory.tags,
            metadata=memory.metadata,
            supersedes=memory.supersedes,
        )
        self.session.add(orm)
        await self.session.flush()
        return memory

    async def get(self, memory_id: uuid.UUID) -> Memory | None:
        orm = await self.session.get(MemoryORM, memory_id)
        return _orm_to_memory(orm) if orm is not None else None

    async def update(self, memory: Memory) -> Memory:
        """Update business fields only. Never touches status."""
        orm = await self.session.get(MemoryORM, memory.id)
        if orm is None:
            raise KeyError(f"Memory {memory.id} not found")
        orm.content = memory.content
        orm.memory_type = memory.memory_type.value
        orm.session_id = memory.session_id
        orm.updated_at = memory.updated_at
        orm.valid_from = memory.valid_from
        orm.valid_until = memory.valid_until
        orm.importance = memory.importance
        orm.confidence = memory.confidence
        orm.utility = memory.utility
        orm.access_count = memory.access_count
        orm.last_accessed_at = memory.last_accessed_at
        orm.source = memory.source.value
        orm.embedding = memory.embedding or None
        orm.tags = memory.tags
        orm.metadata = memory.metadata
        orm.supersedes = memory.supersedes
        await self.session.flush()
        return memory

    async def list_by_user(self, user_id: str, limit: int = 100) -> list[Memory]:
        stmt = (
            select(MemoryORM)
            .where(MemoryORM.user_id == user_id)
            .order_by(asc(MemoryORM.created_at))
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_orm_to_memory(r) for r in rows]

    # ---------------------------------------------------------- event / evidence
    async def add_event(self, event: MemoryEventRecord) -> MemoryEventRecord:
        orm = MemoryEventORM(
            memory_id=event.memory_id,
            event=event.event.value,
            from_status=event.from_status.value if event.from_status else None,
            to_status=event.to_status.value if event.to_status else None,
            reason=event.reason,
            source_memory_id=event.source_memory_id,
            timestamp=event.timestamp,
        )
        self.session.add(orm)
        await self.session.flush()
        return event

    async def get_events(self, memory_id: uuid.UUID) -> list[MemoryEventRecord]:
        stmt = (
            select(MemoryEventORM)
            .where(MemoryEventORM.memory_id == memory_id)
            .order_by(asc(MemoryEventORM.timestamp))
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            MemoryEventRecord(
                memory_id=r.memory_id,
                event=r.event,  # type: ignore[arg-type]
                from_status=r.from_status,  # type: ignore[arg-type]
                to_status=r.to_status,  # type: ignore[arg-type]
                reason=r.reason,
                source_memory_id=r.source_memory_id,
                timestamp=r.timestamp,
            )
            for r in rows
        ]

    async def add_evidence(self, evidence: MemoryEvidence) -> MemoryEvidence:
        orm = MemoryEvidenceORM(
            memory_id=evidence.memory_id,
            session_id=evidence.session_id,
            message_id=evidence.message_id,
            text=evidence.text,
            confidence=evidence.confidence,
            created_at=evidence.created_at,
        )
        self.session.add(orm)
        await self.session.flush()
        return evidence

    async def get_evidence(self, memory_id: uuid.UUID) -> list[MemoryEvidence]:
        stmt = select(MemoryEvidenceORM).where(MemoryEvidenceORM.memory_id == memory_id)
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            MemoryEvidence(
                memory_id=r.memory_id,
                session_id=r.session_id,
                message_id=r.message_id,
                text=r.text,
                confidence=r.confidence,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        orm = MemoryEdgeORM(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation.value,
            confidence=edge.confidence,
            created_at=edge.created_at,
        )
        self.session.add(orm)
        await self.session.flush()
        return edge

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query_embedding: list[float],
        user_id: str | None = None,
        memory_type: str | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        top_k: int = 20,
        as_of: datetime | None = None,
    ) -> list[Memory]:
        """pgvector similarity search with metadata + temporal filter.

        Temporal uses the half-open interval [valid_from, valid_until).
        """
        sql = """
            SELECT id, 1 - (embedding <=> :qemb) AS sim
            FROM memories
            WHERE embedding IS NOT NULL
              AND status = :status
              AND (:user_id IS NULL OR user_id = :user_id)
              AND (:mtype IS NULL OR memory_type = :mtype)
              AND (:as_of IS NULL
                   OR (valid_from IS NULL OR valid_from <= :as_of)
                   AND (valid_until IS NULL OR :as_of < valid_until))
            ORDER BY sim DESC
            LIMIT :top_k
        """
        params: dict = {
            "qemb": query_embedding,
            "status": status.value,
            "user_id": user_id,
            "mtype": memory_type,
            "as_of": as_of,
            "top_k": top_k,
        }
        rows = (await self.session.execute(text(sql), params)).all()
        memories: list[Memory] = []
        for row in rows:
            m = await self.get(row[0])
            if m is not None:
                memories.append(m)
        return memories

    async def search_same_type(
        self,
        query_embedding: list[float],
        memory_type: str,
        user_id: str | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        top_k: int = 10,
    ) -> list[Memory]:
        """Coarse vector filter used by the ConflictDetector."""
        return await self.search(
            query_embedding=query_embedding,
            user_id=user_id,
            memory_type=memory_type,
            status=status,
            top_k=top_k,
        )

    async def keyword_search(
        self,
        query: str,
        user_id: str | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        top_k: int = 20,
    ) -> list[Memory]:
        """Naive ILIKE keyword search used by hybrid retrieval."""
        pattern = f"%{query}%"
        stmt = (
            select(MemoryORM)
            .where(MemoryORM.content.ilike(pattern))
            .where(MemoryORM.status == status.value)
            .order_by(MemoryORM.updated_at.desc())
            .limit(top_k)
        )
        if user_id is not None:
            stmt = stmt.where(MemoryORM.user_id == user_id)
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_orm_to_memory(r) for r in rows]

    async def active_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        limit: int = 1000,
    ) -> list[Memory]:
        stmt = select(MemoryORM).where(MemoryORM.status == MemoryStatus.ACTIVE.value)
        if user_id is not None:
            stmt = stmt.where(MemoryORM.user_id == user_id)
        if memory_type is not None:
            stmt = stmt.where(MemoryORM.memory_type == memory_type)
        stmt = stmt.limit(limit)
        rows = (await self.session.execute(stmt)).scalars().all()
        return [_orm_to_memory(r) for r in rows]