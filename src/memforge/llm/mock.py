"""Mock LLM provider for tests and offline benchmarks.

Returns scripted responses based on a mapping keyed by prompt substrings.
No API key, no network.
"""
from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

from memforge.llm.base import LLMProvider, Message, RawCompletion

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    def __init__(self) -> None:
        # mapping: regex pattern -> dict of field values
        self.scripts: list[tuple[str, dict]] = []
        self.raw_scripts: list[tuple[str, str]] = []
        self.calls: list[list[Message]] = []
        self.raw_calls: int = 0

    def add_script(self, pattern: str, response: dict) -> "MockLLMProvider":
        self.scripts.append((pattern, response))
        return self

    def add_raw_script(self, pattern: str, text: str) -> "MockLLMProvider":
        self.raw_scripts.append((pattern, text))
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

    async def generate_raw(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RawCompletion:
        self.raw_calls += 1
        prompt = "\n".join(m.content for m in messages)
        for pattern, text in self.raw_scripts:
            if re.search(pattern, prompt, re.IGNORECASE):
                return RawCompletion(text=text, prompt_tokens=len(prompt) // 4,
                                     completion_tokens=len(text) // 4)
        # Default raw response: empty JSON object.
        return RawCompletion(text="{}", prompt_tokens=len(prompt) // 4, completion_tokens=2)