"""Stage 1 integration tests (10 tests, skipped without PostgreSQL)."""
import uuid
from datetime import datetime, timezone

import pytest

from memforge.core.lifecycle import LifecycleManager
from memforge.core.types import (
    Memory,
    MemoryEdge,
    MemoryEventRecord,
    MemoryEventType,
    MemoryEvidence,
    MemoryRelation,
    MemoryStatus,
)
from memforge.storage.repository import MemoryRepository
from memforge.storage.database import session_scope


def make_memory(content: str, **kwargs) -> Memory:
    defaults = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "importance": 0.7, "confidence": 0.9}
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


async def test_add_and_get(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        m = make_memory("User prefers lightweight phones.")
        await repo.add(m)
        got = await repo.get(m.id)
        assert got is not None
        assert got.content == "User prefers lightweight phones."
        assert got.status == MemoryStatus.ACTIVE


async def test_update_business_fields_not_status(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        m = make_memory("User likes coffee.")
        await repo.add(m)
        m2 = m.model_copy(update={"content": "User loves coffee.", "importance": 0.9})
        await repo.update(m2)
        got = await repo.get(m.id)
        assert got.content == "User loves coffee."
        assert got.importance == 0.9
        # status must never change through repository update
        assert got.status == MemoryStatus.ACTIVE


async def test_add_event_append_only(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        m = make_memory("x")
        await repo.add(m)
        mgr = LifecycleManager()
        e1 = mgr.transition(m, MemoryStatus.DORMANT, reason="sleep")
        e2 = mgr.transition(m, MemoryStatus.ACTIVE, reason="restore")
        await repo.add_event(e1)
        await repo.add_event(e2)
        events = await repo.get_events(m.id)
        assert len(events) == 2
        assert [e.event for e in events] == [MemoryEventType.SLEEP, MemoryEventType.ACTIVATE]
        assert events[0].reason == "sleep"


async def test_evidence_add_and_get(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        m = make_memory("User uses Python.")
        await repo.add(m)
        ev = MemoryEvidence(
            memory_id=m.id, session_id="s1", message_id="msg1", text="I code in Python", confidence=0.98
        )
        await repo.add_evidence(ev)
        rows = await repo.get_evidence(m.id)
        assert len(rows) == 1
        assert rows[0].text == "I code in Python"


async def test_vector_search_finds_similar(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        a = make_memory("User prefers lightweight phones.")
        a.embedding = [0.9, 0.1, 0.0]
        await repo.add(a)
        b = make_memory("User likes mechanical keyboards.")
        b.embedding = [0.0, 0.1, 0.9]
        await repo.add(b)
        results = await repo.search(query_embedding=[0.85, 0.1, 0.05], top_k=5)
        assert len(results) == 2
        assert results[0].id == a.id


async def test_search_excludes_non_active(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        active = make_memory("User likes cats.", status=MemoryStatus.ACTIVE)
        dormant = make_memory("User likes dogs.", status=MemoryStatus.DORMANT)
        deprecated = make_memory("User likes birds.", status=MemoryStatus.DEPRECATED)
        forgotten = make_memory("User likes fish.", status=MemoryStatus.FORGOTTEN)
        for m in [active, dormant, deprecated, forgotten]:
            m.embedding = [0.5, 0.5]
            await repo.add(m)
        results = await repo.search(query_embedding=[0.5, 0.5], top_k=10)
        assert len(results) == 1
        assert results[0].id == active.id


async def test_user_isolation(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        u1 = make_memory("User A likes tea.", user_id="userA")
        u2 = make_memory("User B likes tea.", user_id="userB")
        for m in [u1, u2]:
            m.embedding = [0.5, 0.5]
            await repo.add(m)
        results = await repo.search(query_embedding=[0.5, 0.5], user_id="userA", top_k=10)
        assert len(results) == 1
        assert results[0].user_id == "userA"


async def test_list_by_user(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        await repo.add(make_memory("m1", user_id="u1"))
        await repo.add(make_memory("m2", user_id="u1"))
        await repo.add(make_memory("m3", user_id="u2"))
        rows = await repo.list_by_user("u1")
        assert len(rows) == 2


async def test_keyword_search(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        await repo.add(make_memory("User prefers lightweight phones."))
        await repo.add(make_memory("User uses mechanical keyboards."))
        results = await repo.keyword_search("keyboard")
        assert len(results) == 1
        assert "keyboard" in results[0].content


async def test_search_same_type(session_factory):
    async with session_scope(session_factory) as s:
        repo = MemoryRepository(s)
        from memforge.core.types import MemoryType

        pref = make_memory("User likes thin phones.", memory_type=MemoryType.PREFERENCE)
        fact = make_memory("User lives in Wuhan.", memory_type=MemoryType.FACT)
        for m in [pref, fact]:
            m.embedding = [0.5, 0.5]
            await repo.add(m)
        results = await repo.search_same_type([0.5, 0.5], memory_type="preference", top_k=5)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.PREFERENCE