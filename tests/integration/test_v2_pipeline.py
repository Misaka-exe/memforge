"""Phase 7: v2 pipeline end-to-end tests (in-memory fakes, no PostgreSQL)."""
from __future__ import annotations

import asyncio
import math

import pytest

from memforge.core.types import Memory, MemoryStatus, MemoryType
from memforge.embeddings.base import EmbeddingProvider
from memforge.llm.mock import MockLLMProvider
from memforge.pipeline.v2 import MemForgeV2Pipeline, V2PipelineConfig
from memforge.retrieval.temporal import WindowScorer


class FakeEmbedding(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [[0.5, 0.5, 0.0] for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        return [0.5, 0.5, 0.0]


class FakeRepository:
    def __init__(self, memories: list[Memory]) -> None:
        self.memories = memories

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    async def search(self, query_embedding, user_id=None, memory_type=None,
                     status=MemoryStatus.ACTIVE, top_k=20, as_of=None,
                     hard_temporal=False):
        out = []
        for m in self.memories:
            if m.status != status:
                continue
            if user_id is not None and m.user_id != user_id:
                continue
            out.append((self._cos(query_embedding, m.embedding or []), m))
        out.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in out[:top_k]]

    async def keyword_search(self, query, user_id=None, status=MemoryStatus.ACTIVE, top_k=20):
        hits = [m for m in self.memories
                if m.status == status and query.lower() in m.content.lower()]
        return hits[:top_k]


def make_memory(content: str, **kw) -> Memory:
    d = {"user_id": "u1", "status": MemoryStatus.ACTIVE, "memory_type": MemoryType.FACT}
    d.update(kw)
    return Memory(content=content, embedding=[0.5, 0.5, 0.0], **d)


def make_llm(answer: str, abstain: bool = False) -> MockLLMProvider:
    llm = MockLLMProvider()
    llm.add_script(
        r"Question:",
        {
            "answer": answer,
            "citations": [{"memory_index": 1, "claim": answer}],
            "abstain": abstain,
        },
    )
    return llm


def test_v2_pipeline_end_to_end():
    mems = [make_memory("User lives in Shanghai."), make_memory("User likes coffee.")]
    repo = FakeRepository(mems)
    llm = make_llm("The user lives in Shanghai.")
    pipe = MemForgeV2Pipeline(repo, FakeEmbedding(), llm)
    res = asyncio.run(pipe.answer("where does the user live", top_k=5))
    assert res.abstained is False
    assert res.answer is not None
    assert "Shanghai" in res.answer.answer
    assert res.verification is not None
    assert res.verification.verdict in ("PASS", "WARN", "FAIL")


def test_v2_pipeline_abstain_path():
    repo = FakeRepository([])
    llm = make_llm("should not be used", abstain=False)
    pipe = MemForgeV2Pipeline(repo, FakeEmbedding(), llm)
    res = asyncio.run(pipe.answer("anything", top_k=5))
    assert res.abstained is True
    assert res.answer.abstain is True
    # No memories -> gate abstains before any LLM structured call.
    assert len(llm.calls) == 0


def test_v2_pipeline_citation_output():
    mems = [make_memory("User lives in Shanghai.")]
    repo = FakeRepository(mems)
    llm = make_llm("The user lives in Shanghai.")
    pipe = MemForgeV2Pipeline(repo, FakeEmbedding(), llm)
    res = asyncio.run(pipe.answer("where does the user live", top_k=5))
    assert res.answer is not None
    assert len(res.answer.citations) == 1
    assert res.answer.citations[0].memory_id == str(mems[0].id)


def test_v2_pipeline_verification_present():
    mems = [make_memory("User lives in Shanghai.")]
    repo = FakeRepository(mems)
    llm = make_llm("The user lives in Shanghai.")
    pipe = MemForgeV2Pipeline(repo, FakeEmbedding(), llm)
    res = asyncio.run(pipe.answer("q", top_k=5))
    assert res.verification is not None
    assert res.verification.answer == res.answer.answer


def test_v2_pipeline_temporal_soft():
    from datetime import datetime, timezone
    future = make_memory(
        "User will move to Berlin.",
        valid_from=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    repo = FakeRepository([future])
    llm = make_llm("x", abstain=True)
    pipe = MemForgeV2Pipeline(
        repo, FakeEmbedding(), llm,
        temporal_scorer=WindowScorer(out_of_window=0.05),
    )
    as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
    mems = asyncio.run(pipe.retrieve("move", top_k=5, as_of=as_of))
    # Soft temporal: future memory is still retrieved (not hard-filtered).
    assert any(m.id == future.id for m in mems)
