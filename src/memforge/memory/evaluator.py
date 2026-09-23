"""Memory Evaluator: LLM scoring with Pydantic-validated, safe fallbacks.

Design rules:
- importance / confidence must be within [0.0, 1.0].
- When the LLM returns out-of-range values or the call fails, fall back to
  0.5/0.5 instead of letting bad floats into the database.
"""
from __future__ import annotations

from memforge.core.types import Memory, MemoryEvaluation
from memforge.llm.base import LLMProvider, Message


EVALUATE_PROMPT = """You are a memory quality evaluator for a personal assistant.

Given the memory statement below, score it on two dimensions:
- importance: how important is this memory for future interactions? (0.0 - 1.0)
- confidence: how confident are we that this memory is accurate? (0.0 - 1.0)

Return JSON: {{"importance": 0.0-1.0, "confidence": 0.0-1.0, "reason": "short reason"}}

Memory: {content}
"""


class MemoryEvaluator:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def evaluate(self, memory: Memory) -> MemoryEvaluation:
        messages = [
            Message(role="user", content=EVALUATE_PROMPT.format(content=memory.content))
        ]
        try:
            result = await self.llm.generate(messages, MemoryEvaluation)
        except Exception:
            return MemoryEvaluation(importance=0.5, confidence=0.5, reason="fallback: llm_error")

        if not (0.0 <= result.importance <= 1.0 and 0.0 <= result.confidence <= 1.0):
            return MemoryEvaluation(
                importance=0.5, confidence=0.5, reason="fallback: out_of_range"
            )
        return result