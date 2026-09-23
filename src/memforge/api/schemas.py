"""FastAPI request/response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from memforge.core.types import MemoryType


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.FACT
    user_id: str = "default"
    session_id: str | None = None
    importance: float = 0.5
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    memory_type: MemoryType | None = None
    top_k: int = 5
    as_of: datetime | None = None


class MemoryResponse(BaseModel):
    id: uuid.UUID
    content: str
    memory_type: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    valid_from: datetime | None
    valid_until: datetime | None
    importance: float
    confidence: float
    status: str
    tags: list[str]
    supersedes: uuid.UUID | None