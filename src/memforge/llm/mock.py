"""Mock LLM provider for tests and offline benchmarks.

Returns scripted responses based on a mapping keyed by prompt substrings.
No API key, no network.
"""
from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

from memforge.llm.base import LLMProvider, Message

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    def __init__(self) -> None:
        # mapping: regex pattern -> dict of field values
        self.scripts: list[tuple[str, dict]] = []
        self.calls: list[list[Message]] = []

    def add_script(self, pattern: str, response: dict) -> "MockLLMProvider":
        self.scripts.append((pattern, response))
        return self

    async def generate(
        self,
        messages: list[Message],
        response_model: type[T],
    ) -> T:
        self.calls.append(messages)
        prompt = "\n".join(m.content for m in messages)
        for pattern, values in self.scripts:
            if re.search(pattern, prompt, re.IGNORECASE):
                return response_model.model_validate(values)
        return response_model.model_validate({})