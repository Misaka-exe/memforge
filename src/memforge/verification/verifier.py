"""v2 post-hoc verifier.

Given retrieved evidence and a model answer, produce a VerificationResult:
which claims are unsupported, which citations are invalid, whether a
contradiction exists, and whether the system should have abstained.

The verifier is intentionally a *judge*, not a re-answerer. It uses
deterministic token-overlap heuristics (no LLM call) so that verification
is fast, reproducible, and cannot introduce a second model's subjectivity.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from memforge.core.types import Memory


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class VerificationResult(BaseModel):
    answer: str
    verdict: Verdict = Verdict.PASS
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reason: str = ""
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claim_rate: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will "
    "would could should may might must shall can need dare to of in for on "
    "with at by from as into through during before after above below between "
    "out off over under again further then once here there when where why how "
    "all each every both few more most other some such no nor not only own "
    "same so than too very just because but and or if while although though "
    "that this these those it its they them their we us our you your he him "
    "his she her i me my what which who whom whose about up also".split()
)


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"\w+", text.lower()))
    return tokens - _STOPWORDS


def _claim_support(claim: str, evidence_text: str) -> float:
    """Fraction of content tokens in ``claim`` that also appear in ``evidence_text``."""
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 0.0
    ev_tokens = _content_tokens(evidence_text)
    overlap = claim_tokens & ev_tokens
    return len(overlap) / len(claim_tokens)


def _is_abstention(answer: str) -> bool:
    a = answer.strip().lower()
    if not a:
        return True
    if "insufficient_information" in a:
        return True
    if any(p in a for p in ("not enough", "don't know", "do not know", "unknown",
                            "cannot answer", "can't answer", "no information")):
        return True
    return False


class Verifier:
    """Judge an answer against retrieved evidence. Does not regenerate.

    Uses deterministic token-overlap heuristics. Each sentence in the answer
    is treated as a claim; a claim is *supported* when at least
    ``support_threshold`` fraction of its content tokens appear in at least one
    evidence text. The overall verdict is derived from the unsupported-claim
    rate and whether the answer should have abstained.
    """

    def __init__(self, support_threshold: float = 0.3) -> None:
        self.support_threshold = support_threshold

    async def verify(
        self,
        query: str,
        answer: str,
        evidence: list[Memory],
        citations: list[str] | None = None,
    ) -> VerificationResult:
        evidence_texts = [m.content for m in evidence if m.content.strip()]
        reasons: list[str] = []

        # --- Abstention handling ---
        if _is_abstention(answer):
            if not evidence_texts:
                return VerificationResult(
                    answer=answer, verdict=Verdict.PASS,
                    should_abstain=True, reason="correct abstention with no evidence",
                )
            reasons.append("abstained despite having retrieved evidence")
            return VerificationResult(
                answer=answer, verdict=Verdict.WARN,
                should_abstain=False, reason="; ".join(reasons),
                total_claims=0, supported_claims=0,
            )

        # --- Non-abstention with no evidence -> FAIL ---
        if not evidence_texts:
            return VerificationResult(
                answer=answer, verdict=Verdict.FAIL,
                should_abstain=True,
                reason="answer provided with no supporting evidence",
                total_claims=0, supported_claims=0,
            )

        # --- Split answer into claims and check support ---
        claims = _split_sentences(answer)
        if not claims:
            claims = [answer.strip()]

        unsupported: list[str] = []
        supported_count = 0
        for claim in claims:
            best_overlap = max((_claim_support(claim, ev) for ev in evidence_texts), default=0.0)
            if best_overlap >= self.support_threshold:
                supported_count += 1
            else:
                unsupported.append(claim[:200])

        total = len(claims)
        rate = 1.0 - (supported_count / total) if total > 0 else 0.0

        # --- Citation validity (if provided) ---
        invalid_citations: list[str] = []
        if citations:
            evidence_ids = {str(m.id) for m in evidence}
            for c in citations:
                if c not in evidence_ids:
                    invalid_citations.append(c)

        # --- Contradiction check (simple negation detection) ---
        contradictions: list[str] = []
        answer_lower = answer.lower()
        for ev in evidence_texts:
            ev_lower = ev.lower()
            # Check if answer contains negation of a positive evidence statement
            for neg in ("not ", "no longer", "never ", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't"):
                if neg in answer_lower and neg not in ev_lower:
                    # Crude: extract key entity overlap
                    answer_keys = _content_tokens(answer)
                    ev_keys = _content_tokens(ev)
                    shared = answer_keys & ev_keys
                    if len(shared) >= 2:
                        contradictions.append(f"possible negation vs evidence: {ev[:100]}")
                        break

        # --- Verdict ---
        if rate == 0.0 and not invalid_citations and not contradictions:
            verdict = Verdict.PASS
            reasons.append("all claims supported")
        elif rate <= 0.3 and not contradictions:
            verdict = Verdict.WARN
            reasons.append(f"{supported_count}/{total} claims supported")
        else:
            verdict = Verdict.FAIL
            reasons.append(f"only {supported_count}/{total} claims supported")

        if invalid_citations:
            reasons.append(f"{len(invalid_citations)} invalid citation(s)")
        if contradictions:
            reasons.append(f"{len(contradictions)} possible contradiction(s)")

        return VerificationResult(
            answer=answer,
            verdict=verdict,
            unsupported_claims=unsupported,
            invalid_citations=invalid_citations,
            contradictions=contradictions,
            should_abstain=False,
            reason="; ".join(reasons),
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claim_rate=rate,
        )
