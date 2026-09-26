"""Phase 6: Post-hoc verifier tests.

The verifier must judge an answer against evidence WITHOUT regenerating it.
"""
from __future__ import annotations

import pytest

from memforge.core.types import Memory, MemoryStatus, MemoryType
from memforge.verification.verifier import Verdict, Verifier, VerificationResult


def make_evidence(*texts: str) -> list[Memory]:
    return [
        Memory(content=t, status=MemoryStatus.ACTIVE, memory_type=MemoryType.FACT)
        for t in texts
    ]


@pytest.mark.asyncio
async def test_verifier_pass():
    v = Verifier()
    evidence = make_evidence("User lives in Shanghai.")
    result = await v.verify("where does the user live?", "The user lives in Shanghai.", evidence)
    assert result.verdict == Verdict.PASS
    assert result.unsupported_claims == []
    assert result.should_abstain is False


@pytest.mark.asyncio
async def test_verifier_fail_unsupported():
    v = Verifier()
    evidence = make_evidence("User lives in Shanghai.")
    answer = "The user lives in Shanghai. The user enjoys climbing mountains every weekend."
    result = await v.verify("q", answer, evidence)
    assert result.verdict == Verdict.FAIL
    assert len(result.unsupported_claims) >= 1


@pytest.mark.asyncio
async def test_verifier_warn_partial():
    v = Verifier()
    evidence = make_evidence(
        "User lives in Shanghai. User likes coffee. User runs every morning."
    )
    answer = (
        "The user lives in Shanghai. "
        "The user likes coffee. "
        "The user runs daily. "
        "The user plays chess professionally."
    )
    result = await v.verify("q", answer, evidence)
    # 3 supported, 1 unsupported -> low unsupported rate -> WARN
    assert result.verdict == Verdict.WARN


@pytest.mark.asyncio
async def test_verifier_detects_contradiction():
    v = Verifier()
    evidence = make_evidence("User lives in Beijing.")
    answer = "The user does not live in Beijing anymore."
    result = await v.verify("q", answer, evidence)
    assert len(result.contradictions) >= 1
    assert result.verdict == Verdict.FAIL


@pytest.mark.asyncio
async def test_verifier_should_abstain():
    v = Verifier()
    result = await v.verify("q", "The user lives in Shanghai.", evidence=[])
    assert result.should_abstain is True
    assert result.verdict == Verdict.FAIL


@pytest.mark.asyncio
async def test_verifier_invalid_citations():
    v = Verifier()
    evidence = make_evidence("User lives in Shanghai.")
    result = await v.verify(
        "q", "The user lives in Shanghai.", evidence, citations=["ghost-id"]
    )
    assert "ghost-id" in result.invalid_citations


@pytest.mark.asyncio
async def test_verifier_does_not_regenerate():
    v = Verifier()
    evidence = make_evidence("User lives in Shanghai.")
    original = "The user lives in Shanghai."
    result = await v.verify("q", original, evidence)
    # The answer returned must be the one we passed in, not a rewrite.
    assert isinstance(result, VerificationResult)
    assert result.answer == original
