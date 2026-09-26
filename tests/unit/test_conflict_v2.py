"""Phase 3: Conflict v2 — typed decisions, no destructive default."""
from __future__ import annotations

import pytest

from memforge.core.types import Memory, MemoryEventType, MemoryStatus, MemoryType
from memforge.llm.mock import MockLLMProvider
from memforge.memory.conflict import (
    ConflictClassifier,
    ConflictDecision,
    ConflictDetector,
    ConflictType,
)
from memforge.memory.transition import TransitionLog


def make_memory(content: str, embedding: list[float], **kwargs) -> Memory:
    defaults = {"status": MemoryStatus.CANDIDATE, "memory_type": MemoryType.PREFERENCE}
    defaults.update(kwargs)
    return Memory(content=content, embedding=embedding, **defaults)


def make_active(content: str, embedding: list[float]) -> Memory:
    return make_memory(content, embedding, status=MemoryStatus.ACTIVE)


# ---------------------------------------------------------------------------
# Classifier unit tests
# ---------------------------------------------------------------------------

def test_conflict_decision_types():
    clf = ConflictClassifier()
    no = clf.classify(0.9, None, has_candidates=False)
    assert no.type == ConflictType.NO_CONFLICT

    below = clf.classify(0.5, None, has_candidates=True)
    assert below.type == ConflictType.NO_CONFLICT

    from memforge.core.types import ConflictResult
    llm_no = clf.classify(0.9, ConflictResult(conflict=False, confidence=0.9), has_candidates=True)
    assert llm_no.type == ConflictType.NO_CONFLICT

    confirmed = clf.classify(0.9, ConflictResult(conflict=True, confidence=0.95), has_candidates=True)
    assert confirmed.type == ConflictType.CONFIRMED_CONFLICT

    possible = clf.classify(0.9, ConflictResult(conflict=True, confidence=0.6), has_candidates=True)
    assert possible.type == ConflictType.POSSIBLE_CONFLICT

    uncertain = clf.classify(0.9, ConflictResult(conflict=True, confidence=0.3), has_candidates=True)
    assert uncertain.type == ConflictType.UNCERTAIN

    no_llm = clf.classify(0.95, None, has_candidates=True)
    assert no_llm.type == ConflictType.UNCERTAIN


# ---------------------------------------------------------------------------
# Detector integration tests
# ---------------------------------------------------------------------------

def _llm(script: dict) -> MockLLMProvider:
    llm = MockLLMProvider()
    llm.add_script(r"OLD memory: .*", script)
    return llm


@pytest.mark.asyncio
async def test_uncertain_no_destructive_transition():
    """UNCERTAIN: old memory stays ACTIVE, new stays CANDIDATE."""
    llm = _llm({"conflict": True, "confidence": 0.3, "reason": "weak evidence"})
    detector = ConflictDetector(llm)
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User might have moved.", [0.8, 0.2, 0.0])
    res = await detector.resolve(new, candidates=[old])
    assert res.decision.type == ConflictType.UNCERTAIN
    assert old.status == MemoryStatus.ACTIVE
    assert old.valid_until is None
    assert new.status == MemoryStatus.CANDIDATE
    assert res.action == "noop"


@pytest.mark.asyncio
async def test_possible_conflict_records_event():
    """POSSIBLE_CONFLICT: event recorded but no state change on either memory."""
    llm = _llm({"conflict": True, "confidence": 0.6, "reason": "maybe moved"})
    log = TransitionLog()
    detector = ConflictDetector(llm, transition_log=log)
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User possibly moved to Shanghai.", [0.8, 0.2, 0.0])
    res = await detector.resolve(new, candidates=[old])
    assert res.decision.type == ConflictType.POSSIBLE_CONFLICT
    assert res.action == "noop"
    assert old.status == MemoryStatus.ACTIVE
    assert old.valid_until is None
    assert new.status == MemoryStatus.CANDIDATE
    # event recorded
    assert res.events[0].event == MemoryEventType.NOOP
    # transition log has an entry for new memory, non-destructive
    recs = log.records_for(new.id)
    assert len(recs) == 1
    assert recs[0].destructive is False


@pytest.mark.asyncio
async def test_confirmed_conflict_deprecates_old():
    llm = _llm({"conflict": True, "confidence": 0.95, "reason": "user moved to Shanghai"})
    log = TransitionLog()
    detector = ConflictDetector(llm, transition_log=log)
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User moved to Shanghai.", [0.8, 0.2, 0.0])
    res = await detector.resolve(new, candidates=[old])
    assert res.decision.type == ConflictType.CONFIRMED_CONFLICT
    assert res.action == "deprecate"
    assert old.status == MemoryStatus.DEPRECATED
    assert new.status == MemoryStatus.ACTIVE
    # destructive transition logged for old
    old_recs = log.records_for(old.id)
    assert any(r.destructive for r in old_recs)


@pytest.mark.asyncio
async def test_no_conflict_activates_new():
    llm = _llm({"conflict": False, "confidence": 0.9, "reason": "related, not contradictory"})
    detector = ConflictDetector(llm)
    old = make_active("User prefers lightweight phones.", [1.0, 0.0, 0.0])
    new = make_memory("User prefers phones with long battery life.", [0.9, 0.1, 0.0])
    res = await detector.resolve(new, candidates=[old])
    assert res.decision.type == ConflictType.NO_CONFLICT
    assert res.action == "activate"
    assert new.status == MemoryStatus.ACTIVE
    assert old.status == MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_conflict_decision_auditable():
    log = TransitionLog()
    llm = _llm({"conflict": True, "confidence": 0.95, "reason": "moved"})
    detector = ConflictDetector(llm, transition_log=log)
    old = make_active("User lives in Beijing.", [1.0, 0.0, 0.0])
    new = make_memory("User moved to Shanghai.", [0.8, 0.2, 0.0])
    await detector.resolve(new, candidates=[old])
    all_records = log.all()
    # one destructive (old deprecated) + one non-destructive (new activated)
    assert len(all_records) == 2
    by_memory = {str(r.memory_id): r for r in all_records}
    assert by_memory[str(old.id)].to_state == MemoryStatus.DEPRECATED
    assert by_memory[str(old.id)].destructive is True
    assert by_memory[str(new.id)].to_state == MemoryStatus.ACTIVE
    assert by_memory[str(new.id)].decision_source == "conflict_v2"
