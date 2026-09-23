"""FastAPI application for MemForge (Stage 1 CRUD + search)."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memforge.api.schemas import MemoryCreateRequest, MemoryResponse, SearchRequest
from memforge.core.lifecycle import LifecycleManager
from memforge.core.types import Memory, MemoryStatus
from memforge.embeddings.base import EmbeddingProvider
from memforge.storage.database import build_engine, build_sessionmaker
from memforge.storage.repository import MemoryRepository

app = FastAPI(title="MemForge", version="0.1.0")

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_embedding: EmbeddingProvider | None = None


def configure(database_url: str | None = None, embedding: EmbeddingProvider | None = None) -> None:
    global _engine, _session_factory, _embedding
    if _engine is None:
        _engine = build_engine(database_url)
        _session_factory = build_sessionmaker(_engine)
    _embedding = embedding


async def get_session() -> AsyncSession:
    assert _session_factory is not None, "call configure() first"
    async with _session_factory() as session:
        yield session


@app.post("/v1/memories", response_model=MemoryResponse)
async def create_memory(
    req: MemoryCreateRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MemoryResponse:
    memory = Memory(
        content=req.content,
        memory_type=req.memory_type,
        user_id=req.user_id,
        session_id=req.session_id,
        importance=req.importance,
        confidence=req.confidence,
        tags=req.tags,
        status=MemoryStatus.CANDIDATE,
    )
    if _embedding is not None:
        memory.embedding = _embedding.embed_one(memory.content)
    repo = MemoryRepository(session)
    await repo.add(memory)
    mgr = LifecycleManager()
    event = mgr.transition(memory, MemoryStatus.ACTIVE, reason="api_create")
    await repo.add_event(event)
    await session.commit()
    return MemoryResponse.model_validate(memory, from_attributes=True)


@app.get("/v1/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MemoryResponse:
    repo = MemoryRepository(session)
    memory = await repo.get(uuid.UUID(memory_id))
    if memory is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return MemoryResponse.model_validate(memory, from_attributes=True)


@app.post("/v1/memories/search", response_model=list[MemoryResponse])
async def search_memories(
    req: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[MemoryResponse]:
    assert _embedding is not None, "embedding provider required for search"
    from memforge.retrieval.engine import RetrievalEngine

    repo = MemoryRepository(session)
    engine = RetrievalEngine(repo, _embedding)
    results = await engine.search(
        query=req.query,
        user_id=req.user_id,
        memory_type=req.memory_type.value if req.memory_type else None,
        top_k=req.top_k,
        as_of=req.as_of,
    )
    return [MemoryResponse.model_validate(m, from_attributes=True) for m in results]


@app.get("/v1/memories/{memory_id}/history")
async def memory_history(
    memory_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    repo = MemoryRepository(session)
    events = await repo.get_events(uuid.UUID(memory_id))
    return [e.model_dump(mode="json") for e in events]


@app.get("/v1/memories/{memory_id}/evidence")
async def memory_evidence(
    memory_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[dict]:
    repo = MemoryRepository(session)
    evidence = await repo.get_evidence(uuid.UUID(memory_id))
    return [e.model_dump(mode="json") for e in evidence]