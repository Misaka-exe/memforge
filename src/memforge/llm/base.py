"""LLM provider abstraction.

All LLM calls in MemForge go through this interface. Outputs are
constrained by a Pydantic response model so invalid shapes fail fast.

v2 adds ``generate_raw`` for free-text (JSON-instructed) generation used by the
grounded QA path; it is purely additive and never changes v1 ``generate``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: str = "user"
    content: str = ""


@dataclass
class RawCompletion:
    """Free-text LLM completion with token accounting."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        response_model: type[T],
    ) -> T:
        """Generate a structured response matching response_model."""

    async def generate_raw(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RawCompletion:
        """Generate free-text completion. Default: wrap generate() with a catch-all.

        Providers that can return raw text should override this. The default
        implementation asks the model for a JSON object with a single "text"
        field so it works with the structured-response contract.
        """

        class _Wrap(BaseModel):
            text: str

        wrapped = await self.generate(messages, _Wrap)
        return RawCompletion(text=wrapped.text)