"""Phase 1: v2 data models import + behavior smoke tests."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from memforge.core.types import MemoryStatus
from memforge.memory.grounding import Citation, GroundedAnswer
from memforge.memory.transition import MemoryTransition, TransitionLog
from memforge.retrieval.gate import RetrievalDecision, Sufficiency, SufficiencyGate
from memforge.verification.verifier import VerificationResult, Verdict, Verifier


def test_memory_transition_construction():
    mid = uuid.uuid4()
    t = MemoryTransition(
        memory_id=mid,
        from_state=MemoryStatus.ACTIVE,
        to_state=MemoryStatus.DEPRECATED,
        reason="superseded by newer address",
        evidence_ids=["e1", "e2"],
        confidence=0.9,
        decision_source="conflict_v2",
        destructive=True,
    )
    assert t.memory_id == mid
    assert t.from_state == MemoryStatus.ACTIVE
    assert t.to_state == MemoryStatus.DEPRECATED
    assert t.evidence_ids == ["e1", "e2"]
    assert isinstance(t.timestamp, datetime)


def test_transition_log_append_only_and_queryable():
    log = TransitionLog()
    mid = uuid.uuid4()
    t1 = MemoryTransition(
        memory_id=mid,
        from_state=MemoryStatus.CANDIDATE,
        to_state=MemoryStatus.ACTIVE,
        reason="activate",
        confidence=1.0,
    )
    other = uuid.uuid4()
    t2 = MemoryTransition(
        memory_id=other,
        from_state=MemoryStatus.ACTIVE,
        to_state=MemoryStatus.ACTIVE,
        reason="uncertain_no_op",
        confidence=0.4,
    )
    log.record(t1)
    log.record(t2)
    assert len(log) == 2
    assert [r.memory_id for r in log.records_for(mid)] == [mid]
    assert len(log.records_for(other)) == 1
    # records are immutable-ish: log returns copies
    assert log.all() == [t1, t2]


def test_citation_and_grounded_answer():
    g = GroundedAnswer(
        answer="User lives in Shanghai.",
        citations=[
            Citation(memory_id="m1", claim_text="lives in Shanghai", supported=True),
            Citation(memory_id="m2", claim_text="likes tea", supported=False),
        ],
    )
    assert not g.abstain
    assert g.citation_count == 2
    assert g.supported_citations[0].memory_id == "m1"
    assert abs(g.citation_coverage() - 0.5) < 1e-9


def test_grounded_answer_abstain():
    g = GroundedAnswer(answer="", citations=[], abstain=True, reason="no evidence")
    assert g.abstain
    assert g.citation_coverage() == 0.0


def test_retrieval_decision_abstain_property():
    d = RetrievalDecision(
        query="where does user live",
        top_k_ids=[],
        sufficiency=Sufficiency.INSUFFICIENT,
        confidence=0.0,
        reason="no relevant memories",
    )
    assert d.should_abstain
    ok = RetrievalDecision(
        query="q", top_k_ids=["m1"], sufficiency=Sufficiency.SUFFICIENT, confidence=0.9
    )
    assert not ok.should_abstain




def test_verification_result_shape():
    r = VerificationResult(
        answer="User moved",
        verdict=Verdict.FAIL,
        unsupported_claims=["likes tea"],
        invalid_citations=["m9"],
        contradictions=["lived in Beijing"],
        should_abstain=True,
        reason="claim not supported by evidence",
    )
    assert r.verdict == Verdict.FAIL
    assert r.should_abstain
    assert "m9" in r.invalid_citations


# ---------------------------------------------------------------------------
# SufficiencyGate behavior (Phase 4)
# ---------------------------------------------------------------------------

def test_gate_empty_memories_insufficient():
    gate = SufficiencyGate()
    decision = gate.assess("where does user live", [])
    assert decision.sufficiency == Sufficiency.INSUFFICIENT
    assert decision.should_abstain


def test_gate_high_scores_sufficient():
    from memforge.core.types import Memory, MemoryStatus
    m = Memory(content="User lives in Shanghai.", status=MemoryStatus.ACTIVE)
    gate = SufficiencyGate(sufficiency_threshold=0.5, uncertain_band=0.1)
    decision = gate.assess("q", [m], scores=[0.9])
    assert decision.sufficiency == Sufficiency.SUFFICIENT


def test_gate_uncertain_band():
    from memforge.core.types import Memory, MemoryStatus
    m = Memory(content="fact", status=MemoryStatus.ACTIVE)
    gate = SufficiencyGate(sufficiency_threshold=0.5, uncertain_band=0.1)
    decision = gate.assess("q", [m], scores=[0.5])
    assert decision.sufficiency == Sufficiency.UNCERTAIN


def test_gate_low_relevant_count_insufficient():
    from memforge.core.types import Memory, MemoryStatus
    m = Memory(content="weak", status=MemoryStatus.ACTIVE)
    gate = SufficiencyGate(min_relevant_count=2)
    decision = gate.assess("q", [m], scores=[0.2])
    assert decision.sufficiency == Sufficiency.INSUFFICIENT


# ---------------------------------------------------------------------------
# Verifier behavior (Phase 6)
# ---------------------------------------------------------------------------

import asyncio


def test_verifier_supported_answer_pass():
    from memforge.core.types import Memory, MemoryStatus
    v = Verifier()
    ev = Memory(content="User lives in Shanghai and works at a tech company.", status=MemoryStatus.ACTIVE)
    result = asyncio.run(v.verify("where", "User lives in Shanghai.", [ev]))
    assert result.verdict == Verdict.PASS
    assert result.supported_claims >= 1


def test_verifier_unsupported_answer_fail():
    from memforge.core.types import Memory, MemoryStatus
    v = Verifier()
    ev = Memory(content="The weather is sunny today.", status=MemoryStatus.ACTIVE)
    result = asyncio.run(v.verify("where", "User lives in Antarctica and enjoys polar swimming.", [ev]))
    assert result.verdict == Verdict.FAIL
    assert result.unsupported_claim_rate > 0


def test_verifier_no_evidence_fail():
    v = Verifier()
    result = asyncio.run(v.verify("q", "User lives in Shanghai.", []))
    assert result.verdict == Verdict.FAIL
    assert result.should_abstain


def test_verifier_abstention_pass():
    v = Verifier()
    result = asyncio.run(v.verify("q", "INSUFFICIENT_INFORMATION", []))
    assert result.verdict == Verdict.PASS
    assert result.should_abstain


def test_verifier_invalid_citations_warn():
    from memforge.core.types import Memory, MemoryStatus
    v = Verifier()
    ev = Memory(content="User lives in Shanghai.", status=MemoryStatus.ACTIVE)
    result = asyncio.run(v.verify("q", "User lives in Shanghai.", [ev], citations=["nonexistent_id"]))
    assert "nonexistent_id" in result.invalid_citations


