"""Memory Extractor: conversation -> LLM -> CandidateMemory -> Memory + Evidence.

The extractor never activates memories: candidates stay CANDIDATE until
a later pipeline stage decides to promote them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from memforge.core.types import (
    CandidateMemory,
    Memory,
    MemoryEvidence,
    MemoryStatus,
)
from memforge.embeddings.base import EmbeddingProvider
from memforge.llm.base import LLMProvider, Message


class ExtractResponse(BaseModel):
    memories: list[CandidateMemory] = Field(default_factory=list)


EXTRACT_PROMPT = """You are a memory extractor for a personal assistant agent.
From the conversation below, extract durable facts, preferences, personal details,
episodic events, procedural knowledge and experiences that are worth remembering
for future interactions.

Rules:
- Do NOT extract one-off small talk or information that will not be useful later.
- Each memory must be a concise, self-contained statement about the user.
- should_store must be false for trivia that is not worth remembering.

Return JSON: {{"memories": [{{"content": "...", "memory_type": "fact|preference|personal|episodic|procedural|experience|summary", "should_store": true, "valid_from": null, "valid_until": null}}]}}

Conversation:
{conversation}
"""


class MemoryExtractor:
    def __init__(
        self,
        llm: LLMProvider,
        embedding: EmbeddingProvider | None = None,
    ) -> None:
        self.llm = llm
        self.embedding = embedding

    async def extract(
        self,
        conversation: str,
        user_id: str = "default",
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> list[tuple[Memory, MemoryEvidence]]:
        messages = [Message(role="user", content=EXTRACT_PROMPT.format(conversation=conversation))]
        response = await self.llm.generate(messages, ExtractResponse)

        results: list[tuple[Memory, MemoryEvidence]] = []
        for cand in response.memories:
            if not cand.should_store:
                continue
            memory = Memory(
                content=cand.content,
                memory_type=cand.memory_type,
                user_id=user_id,
                session_id=session_id,
                valid_from=cand.valid_from,
                valid_until=cand.valid_until,
                importance=cand.importance if cand.importance is not None else 0.5,
                confidence=cand.confidence if cand.confidence is not None else 0.5,
                status=MemoryStatus.CANDIDATE,
            )
            if self.embedding is not None:
                memory.embedding = self.embedding.embed_one(cand.content)
            evidence = MemoryEvidence(
                memory_id=memory.id,
                session_id=session_id,
                message_id=message_id,
                text=conversation,
            )
            results.append((memory, evidence))
        return results