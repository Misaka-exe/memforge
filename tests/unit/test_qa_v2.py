"""Phase 5+7: v2 QA pipeline tests (gate + grounding + verifier integration)."""
from __future__ import annotations

import asyncio

import pytest

from memforge.core.types import Memory, MemoryStatus
from memforge.llm.mock import MockLLMProvider
from memforge.memory.grounding import GroundedAnswer
from benchmarks.evaluation.qa_v2 import (
    CitationEntry,
    GroundedLLMResponse,
    V2QAPipeline,
    V2QAQuestion,
    build_grounded_prompt,
    llm_response_to_grounded,
)


def make_memory(content: str, **kwargs) -> Memory:
    defaults = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "importance": 0.8}
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_grounded_prompt_contains_numbered_memories():
    items = [make_memory("User lives in Shanghai."), make_memory("User likes tea.")]
    prompt = build_grounded_prompt("where", items)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Shanghai" in prompt


def test_grounded_prompt_empty_memories():
    prompt = build_grounded_prompt("q", [])
    assert "(none)" in prompt


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------

def test_llm_response_to_grounded_normal():
    items = [make_memory("User lives in Shanghai.")]
    resp = GroundedLLMResponse(
        answer="User lives in Shanghai.",
        citations=[CitationEntry(memory_index=1, claim="lives in Shanghai")],
        abstain=False,
    )
    g = llm_response_to_grounded(resp, items)
    assert not g.abstain
    assert g.answer == "User lives in Shanghai."
    assert len(g.citations) == 1
    assert g.citations[0].memory_id == str(items[0].id)


def test_llm_response_to_grounded_abstain():
    resp = GroundedLLMResponse(answer="", citations=[], abstain=True)
    g = llm_response_to_grounded(resp, [])
    assert g.abstain
    assert g.answer == ""


def test_llm_response_invalid_index_no_memory_id():
    items = [make_memory("fact")]
    resp = GroundedLLMResponse(
        answer="answer",
        citations=[CitationEntry(memory_index=99, claim="bad")],
    )
    g = llm_response_to_grounded(resp, items)
    assert g.citations[0].memory_id == ""


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_gate_insufficient_no_llm_call():
    llm = MockLLMProvider()
    pipeline = V2QAPipeline(llm=llm)
    q = V2QAQuestion(
        question_id="q1", question="where", gold_answer="x",
        is_abstention=False, category="test", retrieved_items=[],
    )
    result = asyncio.run(pipeline.answer(q))
    assert result.abstained
    assert result.gate_decision == "INSUFFICIENT"
    assert len(llm.calls) == 0  # gate blocked, no LLM call


def test_pipeline_sufficient_calls_llm():
    llm = MockLLMProvider()
    llm.add_script(
        "Retrieved memories",
        {"answer": "User lives in Shanghai.", "citations": [{"memory_index": 1, "claim": "lives in Shanghai"}], "abstain": False},
    )
    pipeline = V2QAPipeline(llm=llm)
    items = [make_memory("User lives in Shanghai.")]
    q = V2QAQuestion(
        question_id="q2", question="where", gold_answer="Shanghai",
        is_abstention=False, category="test", retrieved_items=items,
    )
    result = asyncio.run(pipeline.answer(q))
    assert not result.abstained
    assert result.gate_decision in ("SUFFICIENT", "UNCERTAIN")
    assert len(llm.calls) == 1
    assert result.verification is not None


def test_pipeline_llm_error_abstains():
    llm = MockLLMProvider()  # no scripts -> returns empty model
    pipeline = V2QAPipeline(llm=llm)
    items = [make_memory("User lives in Shanghai.")]
    q = V2QAQuestion(
        question_id="q3", question="where", gold_answer="x",
        is_abstention=False, category="test", retrieved_items=items,
    )
    result = asyncio.run(pipeline.answer(q))
    # Mock returns empty GroundedLLMResponse -> abstain=True
    assert result.abstained


def test_pipeline_abstention_question():
    llm = MockLLMProvider()
    llm.add_script(
        "Retrieved memories",
        {"answer": "", "citations": [], "abstain": True},
    )
    pipeline = V2QAPipeline(llm=llm)
    items = [make_memory("irrelevant fact.")]
    q = V2QAQuestion(
        question_id="q4", question="unknown", gold_answer="",
        is_abstention=True, category="test", retrieved_items=items,
    )
    result = asyncio.run(pipeline.answer(q))
    assert result.abstained
    assert result.verification is None  # no verification for abstention
