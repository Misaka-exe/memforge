"""Synthetic benchmark ground truth data structures.

Ground truth is generated independently of MemForge internals so the
benchmark cannot leak implementation details.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkMemory(BaseModel):
    id: str
    content: str
    user_id: str
    memory_type: str = "fact"
    importance: float = 0.5
    confidence: float = 0.9
    created_at: str = ""
    valid_from: str | None = None
    valid_until: str | None = None
    status: str = "active"  # active | deprecated | dormant | forgotten
    is_noise: bool = False


class BenchmarkQuery(BaseModel):
    query: str
    gold_memory_ids: list[str] = Field(default_factory=list)
    category: str = "retrieval"  # retrieval | temporal | update | conflict | abstention
    user_id: str = "u0"


class BenchmarkScenario(BaseModel):
    scenario_id: str
    user_id: str
    memories: list[BenchmarkMemory] = Field(default_factory=list)
    queries: list[BenchmarkQuery] = Field(default_factory=list)