"""Hybrid retrieval: vector + keyword -> union -> temporal filter -> utility rerank."""
from __future__ import annotations

from datetime import datetime

from memforge.core.types import Memory, MemoryStatus
from memforge.embeddings.base import EmbeddingProvider
from memforge.retrieval.reranker import UtilityReranker
from memforge.storage.repository import MemoryRepository


class HybridRetriever:
    def __init__(
        self,
        repository: MemoryRepository,
        embedding: EmbeddingProvider,
        reranker: UtilityReranker | None = None,
        semantic_top_k: int = 20,
        keyword_top_k: int = 20,
    ) -> None:
        self.repository = repository
        self.embedding = embedding
        self.reranker = reranker or UtilityReranker(embedding=embedding)
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k

    async def _semantic(self, query: str, user_id: str | None) -> list[Memory]:
        qemb = self.embedding.embed_one(query)
        return await self.repository.search(
            query_embedding=qemb,
            user_id=user_id,
            status=MemoryStatus.ACTIVE,
            top_k=self.semantic_top_k,
        )

    async def _keyword(self, query: str, user_id: str | None) -> list[Memory]:
        return await self.repository.keyword_search(
            query=query, user_id=user_id, status=MemoryStatus.ACTIVE, top_k=self.keyword_top_k
        )

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
        as_of: datetime | None = None,
    ) -> list[Memory]:
        semantic = await self._semantic(query, user_id)
        keyword = await self._keyword(query, user_id)

        # Union with dedup by id.
        seen: dict[str, Memory] = {}
        for m in [*semantic, *keyword]:
            seen[str(m.id)] = m
        candidates = list(seen.values())

        # ACTIVE filter (search already restricts, but be explicit) + temporal + rerank.
        active = [m for m in candidates if m.status == MemoryStatus.ACTIVE]
        return self.reranker.rerank(active, query=query, top_k=top_k, as_of=as_of)