"""MemForge v2 QA pipeline: evidence-grounded answering with verification.

This module is the v2 counterpart to ``qa.py``. It adds three layers on top
of plain retrieval-to-answer:

1. **Sufficiency Gate** — before asking the LLM, check whether retrieved
   evidence is sufficient. If not, abstain immediately (no LLM call).

2. **Citation Grounding** — the prompt requires the LLM to return a
   structured JSON answer with citations pointing to specific retrieved
   memories. Unsupported claims are not allowed.

3. **Post-hoc Verification** — after the answer is produced, the Verifier
   checks each claim against the evidence and emits PASS/WARN/FAIL.

v1 ``qa.py`` is frozen and untouched; this module is purely additive.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from memforge.core.types import Memory
from memforge.llm.base import LLMProvider, Message
from memforge.memory.grounding import Citation, GroundedAnswer
from memforge.retrieval.gate import Sufficiency, SufficiencyGate
from memforge.verification.verifier import Verdict, Verifier, VerificationResult

PROMPT_VERSION = "v2.0-grounded"


# ---------------------------------------------------------------------------
# Structured LLM response models
# ---------------------------------------------------------------------------

class CitationEntry(BaseModel):
    memory_index: int = 0
    claim: str = ""


class GroundedLLMResponse(BaseModel):
    """Structured response expected from the LLM (enforced via JSON mode)."""
    answer: str = ""
    citations: list[CitationEntry] = Field(default_factory=list)
    abstain: bool = False


# ---------------------------------------------------------------------------
# Question / Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class V2QAQuestion:
    question_id: str
    question: str
    gold_answer: str
    is_abstention: bool
    category: str
    retrieved_items: list[Memory]


@dataclass
class V2QAResult:
    question_id: str
    category: str
    grounded_answer: GroundedAnswer
    verification: VerificationResult | None
    gate_decision: str
    abstained: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Grounded prompt builder
# ---------------------------------------------------------------------------

GROUNDED_PROMPT_TEMPLATE = """You are an AI assistant answering questions based ONLY on the provided retrieved memories.

You MUST return a JSON object with exactly these fields:
{{
  "answer": "your answer text, or empty string if insufficient",
  "citations": [
    {{"memory_index": 1, "claim": "the specific claim supported by memory [1]"}}
  ],
  "abstain": false
}}

Rules:
1. Answer using ONLY information from the retrieved memories below.
2. For every factual claim in your answer, include a citation with the memory index (1-based) that supports it.
3. If the retrieved memories do not contain enough information to answer, set "abstain": true, "answer": "", and "citations": [].
4. Do NOT include information not supported by the memories.
5. Return ONLY the JSON object, no markdown or code blocks.

Question:
{question}

Retrieved memories:
{retrieved_block}

Return the JSON object now:"""


def build_grounded_prompt(question: str, retrieved_items: list[Memory]) -> str:
    lines = []
    for i, mem in enumerate(retrieved_items, start=1):
        text = " ".join(str(mem.content).split())
        lines.append(f"[{i}] {text}")
    retrieved_block = "\n".join(lines) if lines else "(none)"
    return GROUNDED_PROMPT_TEMPLATE.format(question=question, retrieved_block=retrieved_block)


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------

def llm_response_to_grounded(resp: GroundedLLMResponse, retrieved_items: list[Memory]) -> GroundedAnswer:
    """Convert structured LLM response to GroundedAnswer."""
    if resp.abstain or not resp.answer.strip():
        return GroundedAnswer(answer="", citations=[], abstain=True, reason="model abstained")

    citations: list[Citation] = []
    for entry in resp.citations:
        memory_id = ""
        if 1 <= entry.memory_index <= len(retrieved_items):
            memory_id = str(retrieved_items[entry.memory_index - 1].id)
        citations.append(Citation(memory_id=memory_id, claim_text=entry.claim, supported=True))

    return GroundedAnswer(answer=resp.answer, citations=citations, abstain=False)


# ---------------------------------------------------------------------------
# V2 QA pipeline
# ---------------------------------------------------------------------------

class V2QAPipeline:
    """End-to-end v2 QA: gate -> grounded prompt -> LLM -> parse -> verify."""

    def __init__(
        self,
        llm: LLMProvider,
        gate: SufficiencyGate | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.llm = llm
        self.gate = gate or SufficiencyGate()
        self.verifier = verifier or Verifier()

    async def answer(self, q: V2QAQuestion) -> V2QAResult:
        start = time.monotonic()
        result = V2QAResult(
            question_id=q.question_id,
            category=q.category,
            grounded_answer=GroundedAnswer(answer="", abstain=True),
            verification=None,
            gate_decision="INSUFFICIENT",
            abstained=True,
        )

        try:
            # Step 1: Sufficiency Gate
            scores = [getattr(m, "importance", 0.5) or 0.5 for m in q.retrieved_items]
            decision = self.gate.assess(q.question, q.retrieved_items, scores=scores)
            result.gate_decision = decision.sufficiency.value

            if decision.sufficiency == Sufficiency.INSUFFICIENT:
                result.grounded_answer = GroundedAnswer(
                    answer="", citations=[], abstain=True, reason=f"gate: {decision.reason}"
                )
                result.abstained = True
                result.latency_ms = (time.monotonic() - start) * 1000
                return result

            # Step 2: Build grounded prompt and call LLM (structured JSON mode)
            prompt = build_grounded_prompt(q.question, q.retrieved_items)
            messages = [Message(role="user", content=prompt)]

            try:
                resp = await self.llm.generate(messages, GroundedLLMResponse)
                grounded = llm_response_to_grounded(resp, q.retrieved_items)
            except Exception:
                grounded = GroundedAnswer(answer="", citations=[], abstain=True, reason="llm_error")

            result.grounded_answer = grounded
            result.abstained = grounded.abstain

            # Step 3: Post-hoc verification (only for non-abstention answers)
            if not grounded.abstain:
                verification = await self.verifier.verify(
                    query=q.question,
                    answer=grounded.answer,
                    evidence=q.retrieved_items,
                    citations=[c.memory_id for c in grounded.citations if c.memory_id],
                )
                result.verification = verification

            result.latency_ms = (time.monotonic() - start) * 1000

        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.monotonic() - start) * 1000

        return result
