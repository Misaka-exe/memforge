"""QA providers: Mock (zero-dependency) and OpenAI-compatible (real requests).

Protocol mirrors the project's LLMProvider abstraction (src/memforge/llm/) but
returns raw text + usage so benchmarks can do token accounting. No OpenAI SDK is
hard-coded; requests go over httpx to any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from abc import ABC, abstractmethod

from benchmarks.evaluation.qa import QACompletion

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class QAProviderError(RuntimeError):
    """Raised after retries are exhausted; contains the last status/body."""


class QAProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str) -> QACompletion:
        ...


class MockQAProvider(QAProvider):
    """Scripted responses keyed by prompt patterns. No API key, no network."""

    def __init__(self) -> None:
        self.scripts: list[tuple[str, str]] = []
        self.calls: list[str] = []

    def add_script(self, pattern: str, text: str) -> "MockQAProvider":
        self.scripts.append((pattern, text))
        return self

    async def complete(self, prompt: str) -> QACompletion:
        self.calls.append(prompt)
        for pattern, text in self.scripts:
            if re.search(pattern, prompt, re.IGNORECASE):
                return QACompletion(text=text, prompt_tokens=len(prompt.split()), completion_tokens=len(text.split()))
        return QACompletion(text="INSUFFICIENT_INFORMATION", prompt_tokens=len(prompt.split()), completion_tokens=4)


class OpenAICompatibleQAProvider(QAProvider):
    """Real LLM calls. Configure via args or env:
    MEMFORGE_LLM_MODEL / MEMFORGE_LLM_BASE_URL / MEMFORGE_LLM_API_KEY
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model or os.environ.get("MEMFORGE_LLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("MEMFORGE_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("MEMFORGE_LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        if not (self.model and self.base_url and self.api_key):
            raise ValueError(
                "Real LLM requires MEMFORGE_LLM_MODEL / MEMFORGE_LLM_BASE_URL / MEMFORGE_LLM_API_KEY "
                "(or constructor args)"
            )

    async def complete(self, prompt: str) -> QACompletion:
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.time()
        last_err: Exception | None = None
        last_status = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    latency = (time.time() - t0) * 1000.0
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return QACompletion(
                        text=content,
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        latency_ms=round(latency, 1),
                    )
                last_status = resp.status_code
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = QAProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    raise QAProviderError(
                        f"non-retryable HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except QAProviderError:
                raise
            except Exception as e:  # network / timeout
                last_err = e
            if last_err is not None:
                await asyncio.sleep(2 ** attempt)
        raise QAProviderError(f"exhausted retries (status={last_status}): {last_err}")