"""Stage 1: MemoryExtractor unit tests with Mock LLM + Mock Embedding."""
import pytest

from memforge.core.types import MemoryStatus, MemoryType
from memforge.embeddings.base import EmbeddingProvider
from memforge.llm.mock import MockLLMProvider
from memforge.memory.extractor import MemoryExtractor


class MockEmbedding(EmbeddingProvider):
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(t) % 100) / 100.0] * self.dim for t in texts]


def make_llm(memories: list[dict]) -> MockLLMProvider:
    llm = MockLLMProvider()
    llm.add_script("ExtractResponse|memory extractor", {"memories": memories})
    return llm


@pytest.mark.asyncio
async def test_extractor_extracts_memory():
    llm = make_llm(
        [{"content": "User prefers lightweight phones.", "memory_type": "preference", "should_store": True}]
    )
    emb = MockEmbedding()
    extractor = MemoryExtractor(llm, emb)
    results = await extractor.extract("I really like lightweight phones.", user_id="u1", session_id="s1")

    assert len(results) == 1
    memory, evidence = results[0]
    assert memory.content == "User prefers lightweight phones."
    assert memory.memory_type == MemoryType.PREFERENCE
    assert memory.user_id == "u1"
    assert memory.session_id == "s1"
    assert evidence.memory_id == memory.id
    assert evidence.session_id == "s1"


@pytest.mark.asyncio
async def test_extractor_skips_should_store_false():
    llm = make_llm(
        [
            {"content": "User prefers lightweight phones.", "memory_type": "preference", "should_store": True},
            {"content": "User said hi today.", "memory_type": "fact", "should_store": False},
        ]
    )
    extractor = MemoryExtractor(llm, MockEmbedding())
    results = await extractor.extract("hi")
    assert len(results) == 1
    assert results[0][0].content == "User prefers lightweight phones."


@pytest.mark.asyncio
async def test_extractor_keeps_candidate_status():
    llm = make_llm(
        [{"content": "User likes coffee.", "memory_type": "preference", "should_store": True}]
    )
    extractor = MemoryExtractor(llm, MockEmbedding())
    results = await extractor.extract("I like coffee")
    memory = results[0][0]
    assert memory.status == MemoryStatus.CANDIDATE
    assert memory.status != MemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_extractor_embeds_content():
    llm = make_llm(
        [{"content": "User lives in Shanghai.", "memory_type": "fact", "should_store": True}]
    )
    emb = MockEmbedding()
    extractor = MemoryExtractor(llm, emb)
    results = await extractor.extract("I live in Shanghai now")
    memory = results[0][0]
    assert len(memory.embedding) == 8
    assert len(emb.calls) == 1