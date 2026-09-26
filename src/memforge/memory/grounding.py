"""v2 Citation / Evidence grounding data models.

A v2 answer is not free text: every factual claim must point at the memory
(and optionally the evidence span) that supports it. When retrieved evidence is
insufficient the model must abstain instead of guessing.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A pointer from one factual claim to the evidence that supports it."""

    memory_id: str
    evidence_id: str | None = None
    claim_text: str = ""
    supported: bool = True


class GroundedAnswer(BaseModel):
    """An answer produced by the v2 QA path, with mandatory citations."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstain: bool = False
    reason: str = ""

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def supported_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.supported]

    def citation_coverage(self) -> float:
        """supported claims / total cited claims. 1.0 when all citations hold."""
        if not self.citations:
            return 0.0
        return len(self.supported_citations) / len(self.citations)


_ABSTAIN_MARKERS = (
    "INSUFFICIENT_INFORMATION",
    "NOT ENOUGH INFORMATION",
    "DO NOT KNOW",
    "DON'T KNOW",
    "CANNOT ANSWER",
    "UNKNOWN",
)


def looks_like_abstention(text: str) -> bool:
    t = text.strip().upper()
    return any(marker in t for marker in _ABSTAIN_MARKERS)


def _extract_json(raw: str) -> dict | None:
    """Best-effort extraction of the first JSON object from LLM output."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None


def parse_grounded_answer(raw: str, item_ids: list[str]) -> GroundedAnswer:
    """Parse a v2 LLM response into a GroundedAnswer.

    Expected JSON shape:
        {"answer": "...", "citations": [1, 3], "abstain": false}

    Citations are 1-based indices into the retrieved item list; they are
    translated to concrete memory ids via ``item_ids``. Robust to missing /
    malformed output (falls back to a plain answer with no citations).
    """
    data = _extract_json(raw)
    if data is None:
        abstain = looks_like_abstention(raw)
        return GroundedAnswer(
            answer="" if abstain else raw.strip(),
            citations=[],
            abstain=abstain,
            reason="unparseable_response",
        )

    answer = str(data.get("answer", "")).strip()
    abstain = bool(data.get("abstain", False)) or looks_like_abstention(answer)

    citations: list[Citation] = []
    raw_cites = data.get("citations", []) or []
    for c in raw_cites:
        mem_id: str | None = None
        if isinstance(c, int):
            idx = c - 1  # 1-based prompt index
            if 0 <= idx < len(item_ids):
                mem_id = item_ids[idx]
        elif isinstance(c, str):
            mem_id = c if c in item_ids else None
        elif isinstance(c, dict):
            mid = c.get("memory_id") or c.get("id")
            if isinstance(mid, str) and mid in item_ids:
                mem_id = mid
        if mem_id is not None:
            citations.append(Citation(memory_id=mem_id, claim_text=answer))

    return GroundedAnswer(
        answer="" if abstain else answer,
        citations=citations,
        abstain=abstain,
        reason=str(data.get("reason", "")),
    )


def citation_validity(answer: GroundedAnswer, valid_memory_ids: set[str]) -> list[str]:
    """Return citation memory_ids that point at a memory not in the retrieved set.

    A citation is *invalid* when it references a memory id that was never
    retrieved (i.e. the model invented a pointer).
    """
    return [c.memory_id for c in answer.citations if c.memory_id not in valid_memory_ids]
