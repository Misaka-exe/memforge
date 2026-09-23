"""Semantic retrieval engine (Stage 1): embed query -> pgvector -> ACTIVE filter."""
from __future__ import annotations

from datetime import datetime

from memforge.core.types import Memory, MemoryStatus
from memforge.embeddings.base import EmbeddingProvider
from memforge.storage.repository import MemoryRepository


class RetrievalEngine:
    def __init__(
        self,
        repository: MemoryRepository,
        embedding: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.embedding = embedding

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        memory_type: str | None = None,
        top_k: int = 5,
        as_of: datetime | None = None,
    ) -> list[Memory]:
        qemb = self.embedding.embed_one(query)
        return await self.repository.search(
            query_embedding=qemb,
            user_id=user_id,
            memory_type=memory_type,
            status=MemoryStatus.ACTIVE,
            top_k=top_k,
            as_of=as_of,
        )