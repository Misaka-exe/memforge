"""LLM provider abstraction.

All LLM calls in MemForge go through this interface. Outputs are
constrained by a Pydantic response model so invalid shapes fail fast.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: str = "user"
    content: str = ""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        response_model: type[T],
    ) -> T:
        """Generate a structured response matching response_model."""