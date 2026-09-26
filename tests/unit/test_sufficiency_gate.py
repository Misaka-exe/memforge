"""Phase 4: Retrieval Sufficiency Gate tests."""
from __future__ import annotations

from memforge.core.types import Memory, MemoryStatus, MemoryType
from memforge.retrieval.gate import RetrievalDecision, Sufficiency, SufficiencyGate


def make_memory(content: str, **kwargs) -> Memory:
    defaults = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "memory_type": MemoryType.FACT}
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


def test_sufficient_when_high_relevance():
    gate = SufficiencyGate()
    mems = [make_memory("m1"), make_memory("m2")]
    d = gate.assess("where do I live", mems, scores=[0.9, 0.85])
    assert d.sufficiency == Sufficiency.SUFFICIENT
    assert not d.should_abstain


def test_insufficient_when_no_relevant():
    gate = SufficiencyGate()
    mems = [make_memory("m1"), make_memory("m2")]
    d = gate.assess("query", mems, scores=[0.1, 0.05])
    assert d.sufficiency == Sufficiency.INSUFFICIENT
    assert d.should_abstain


def test_insufficient_when_no_memories():
    gate = SufficiencyGate()
    d = gate.assess("query", [], scores=[])
    assert d.sufficiency == Sufficiency.INSUFFICIENT
    assert d.top_k_ids == []


def test_uncertain_boundary():
    gate = SufficiencyGate()  # threshold=0.5, band=0.1 -> band [0.4, 0.6]
    mems = [make_memory("m1")]
    # mean 0.5 sits inside the uncertain band.
    d = gate.assess("query", mems, scores=[0.5])
    assert d.sufficiency == Sufficiency.UNCERTAIN
    assert not d.should_abstain  # uncertain != abstain


def test_gate_does_not_modify_retrieval():
    gate = SufficiencyGate()
    mems = [make_memory("m1", confidence=0.9), make_memory("m2")]
    before_ids = [m.id for m in mems]
    gate.assess("query", mems, scores=[0.9, 0.8])
    after_ids = [m.id for m in mems]
    assert before_ids == after_ids
    assert all(m.status == MemoryStatus.ACTIVE for m in mems)


def test_abstain_on_insufficient():
    gate = SufficiencyGate()
    mems = [make_memory("m1")]
    d = gate.assess("query", mems, scores=[0.05])
    assert isinstance(d, RetrievalDecision)
    assert d.should_abstain is True


def test_gate_evidence_ids():
    gate = SufficiencyGate()
    m1, m2 = make_memory("a"), make_memory("b")
    d = gate.assess("query", [m1, m2], scores=[0.9, 0.8])
    assert set(d.evidence_ids) == {str(m1.id), str(m2.id)}
    assert d.top_k_ids == [str(m1.id), str(m2.id)]


def test_min_relevant_count_configurable():
    gate = SufficiencyGate(min_relevant_count=2)
    mems = [make_memory("a"), make_memory("b")]
    # one highly relevant, one irrelevant -> below min_relevant_count=2
    d = gate.assess("query", mems, scores=[0.95, 0.1])
    assert d.sufficiency == Sufficiency.INSUFFICIENT
