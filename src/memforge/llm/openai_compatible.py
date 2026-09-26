"""OpenAI-compatible LLM provider.

Works with OpenAI, DeepSeek, Qwen, vLLM, Ollama, LM Studio, etc.
Uses httpx directly so no extra SDK dependency is required.
"""
from __future__ import annotations

import json
import os
from typing import TypeVar

from pydantic import BaseModel

from memforge.llm.base import LLMProvider, Message, RawCompletion

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("MEMFORGE_LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("MEMFORGE_LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("MEMFORGE_LLM_API_KEY", "")
        self.temperature = temperature

    async def generate(
        self,
        messages: list[Message],
        response_model: type[T],
    ) -> T:
        import httpx

        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return response_model.model_validate(json.loads(content))

    async def generate_raw(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RawCompletion:
        import httpx

        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        return RawCompletion(
            text=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )