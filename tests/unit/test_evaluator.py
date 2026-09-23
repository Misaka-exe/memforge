"""Stage 2: MemoryEvaluator unit tests."""
import pytest

from memforge.core.types import Memory
from memforge.llm.mock import MockLLMProvider
from memforge.memory.evaluator import MemoryEvaluator


def make_memory(content: str = "User likes coffee.") -> Memory:
    return Memory(content=content)


@pytest.mark.asyncio
async def test_evaluator_normal_scores():
    llm = MockLLMProvider()
    llm.add_script("importance", {"importance": 0.9, "confidence": 0.8, "reason": "explicit statement"})
    evaluator = MemoryEvaluator(llm)
    result = await evaluator.evaluate(make_memory())
    assert result.importance == 0.9
    assert result.confidence == 0.8
    assert result.reason == "explicit statement"


@pytest.mark.asyncio
async def test_evaluator_fallback_on_error():
    llm = MockLLMProvider()
    llm.add_script("importance", {"importance": 1.5, "confidence": 0.8, "reason": "bad"})
    evaluator = MemoryEvaluator(llm)
    result = await evaluator.evaluate(make_memory())
    assert result.importance == 0.5
    assert result.confidence == 0.5
    assert "fallback" in result.reason