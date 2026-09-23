"""Stage 2: ConflictDetector unit tests (similar != conflict)."""
import pytest

from memforge.core.types import (
    Memory,
    MemoryEventType,
    MemoryRelation,
    MemoryStatus,
    MemoryType,
)
from memforge.llm.mock import MockLLMProvider
from memforge.memory.conflict import ConflictDetector


def make_memory(content: str, embedding: list[float], **kwargs) -> Memory:
    defaults = {"status": MemoryStatus.CANDIDATE, "memory_type": MemoryType.PREFERENCE}
    defaults.update(kwargs)
    return Memory(content=content, embedding=embedding, **defaults)


def make_active(content: str, embedding: list[float]) -> Memory:
    return make_memory(content, embedding, status=MemoryStatus.ACTIVE)


def make_llm() -> MockLLMProvider:
    llm = MockLLMProvider()
    llm.add_script(r"NEW memory: User might have moved to low_conf", {"conflict": True, "confidence": 0.3, "reason": "uncertain evidence"})
    llm.add_script(r"NEW memory: User prefers phones with long battery", {"conflict": False, "confidence": 0.9, "reason": "related, not contradictory"})
    llm.add_script(r"OLD memory: User lives in Beijing", {"conflict": True, "confidence": 0.95, "reason": "user moved to Shanghai"})
    return llm


@pytest.mark.asyncio
async def test_no_conflict_both_active():
    """Similar-but-not-contradictory memories must both stay ACTIVE."""
    detector = ConflictDetector(make_llm())
    old = make_active("User prefers lightweight phones.", [1.0, 0.0, 0.0])
    new = make_memory("User prefers phones with long battery life.", [0.9, 0.1, 0.0])
    resolution = await detector.resolve(new, candidates=[old])
    assert resolution.action == "activate"
    assert new.status == MemoryStatus.ACTIVE
    assert old.status == MemoryStatus.ACTIVE
    assert resolution.old_memory is None


@pytest.mark.asyncio
async def test_conflict_deprecates_old():
    detector = ConflictDetector(make_llm())
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User moved to Shanghai.", [0.8, 0.2, 0.0])
    resolution = await detector.resolve(new, candidates=[old])
    assert resolution.action == "deprecate"
    assert old.status == MemoryStatus.DEPRECATED
    assert old.valid_until is not None
    assert new.status == MemoryStatus.ACTIVE
    assert new.supersedes == old.id
    assert resolution.edge is not None
    assert resolution.edge.relation == MemoryRelation.SUPERSEDES


@pytest.mark.asyncio
async def test_low_conflict_confidence_noop():
    """Low-confidence conflict => NOOP: old untouched, new stays CANDIDATE."""
    detector = ConflictDetector(make_llm())
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User might have moved to low_conf city.", [0.8, 0.2, 0.0])
    resolution = await detector.resolve(new, candidates=[old])
    assert resolution.action == "noop"
    assert old.status == MemoryStatus.ACTIVE
    assert old.valid_until is None
    assert new.status == MemoryStatus.CANDIDATE
    assert resolution.events[0].event == MemoryEventType.NOOP


@pytest.mark.asyncio
async def test_no_candidates_activates_new():
    detector = ConflictDetector(make_llm())
    new = make_memory("User prefers mechanical keyboards.", [0.0, 1.0, 0.0])
    resolution = await detector.resolve(new, candidates=[])
    assert resolution.action == "activate"
    assert new.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_conflict_events_recorded():
    detector = ConflictDetector(make_llm())
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User moved to Shanghai.", [0.8, 0.2, 0.0])
    resolution = await detector.resolve(new, candidates=[old])
    events = resolution.events
    assert len(events) == 2
    assert events[0].event == MemoryEventType.DEPRECATE
    assert events[0].from_status == MemoryStatus.ACTIVE
    assert events[0].to_status == MemoryStatus.DEPRECATED
    assert events[1].event == MemoryEventType.ACTIVATE
    assert events[1].from_status == MemoryStatus.CANDIDATE
    assert events[1].to_status == MemoryStatus.ACTIVE