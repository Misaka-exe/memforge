"""MemForge end-to-end demo (Stage 1 pipeline).

Shows: add a memory -> search it back. Requires a running PostgreSQL
(docker compose up -d) and uses a deterministic local embedding.
"""
from __future__ import annotations

import asyncio

from memforge.core.lifecycle import LifecycleManager
from memforge.core.types import Memory, MemoryEventType, MemoryStatus
from memforge.embeddings.sentence_transformers import SentenceTransformersProvider
from memforge.storage.database import build_engine, build_sessionmaker, init_db, session_scope
from memforge.storage.repository import MemoryRepository


async def main() -> None:
    engine = build_engine()
    await init_db(engine)
    factory = build_sessionmaker(engine)
    embedding = SentenceTransformersProvider()

    async with session_scope(factory) as s:
        repo = MemoryRepository(s)
        memory = Memory(
            content="User prefers lightweight phones.",
            memory_type="preference",
            user_id="demo",
            importance=0.78,
            confidence=0.94,
            embedding=embedding.embed_one("User prefers lightweight phones."),
        )
        await repo.add(memory)
        mgr = LifecycleManager()
        event = mgr.transition(memory, MemoryStatus.ACTIVE, reason="demo_add")
        await repo.add_event(event)
        print("Stored:", memory.id)

        results = await repo.search(
            query_embedding=embedding.embed_one("What kind of phone does the user like?"),
            user_id="demo",
            top_k=3,
        )
        for m in results:
            print("Retrieved:", m.content, "| status:", m.status.value)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())