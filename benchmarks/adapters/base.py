"""Unified benchmark case interface.

Every benchmark (LongMemEval, LoCoMo, synthetic) is converted into a
BenchmarkCase so that runners and evaluators never care about the source.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"
    MULTI_SESSION = "multi-session"
    TEMPORAL_REASONING = "temporal-reasoning"
    KNOWLEDGE_UPDATE = "knowledge-update"
    ABSTENTION = "abstention"


class MemoryItem(BaseModel):
    id: str
    content: str
    timestamp: str | None = None
    metadata: dict = Field(default_factory=dict)


class BenchmarkQuestion(BaseModel):
    question_id: str
    question: str
    answer: str
    category: QuestionCategory
    gold_memory_ids: list[str] = Field(default_factory=list)
    is_abstention: bool = False
    question_date: str | None = None


class BenchmarkCase(BaseModel):
    case_id: str
    memories: list[MemoryItem] = Field(default_factory=list)
    questions: list[BenchmarkQuestion] = Field(default_factory=list)