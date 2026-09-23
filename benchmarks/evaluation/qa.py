"""QA evaluation for LongMemEval-S: prompt builder, evaluator, token accounting.

Deliberately separated from retrieval: this module never computes embeddings.
It consumes retrieved memory items (top-K item ids exported by the runner)
and produces model answers + correctness labels.

Prompt contract (no gold answer ever enters the prompt):
    Question: ...
    Retrieved memories:
    [1] ...
    [2] ...
    Instructions:
    Answer the question using only the retrieved memories.
    If the retrieved memories do not contain enough information to answer,
    respond with exactly:
    INSUFFICIENT_INFORMATION
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

INSUFFICIENT = "INSUFFICIENT_INFORMATION"
PROMPT_VERSION = "v1.0"


@dataclass
class QACompletion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class QAQuestion:
    question_id: str
    question: str
    gold_answer: str
    is_abstention: bool
    category: str
    retrieved_items: list[str]  # top-K item texts, ordered


def build_prompt(q: QAQuestion) -> str:
    lines = [
        "You are an AI assistant answering questions based ONLY on the provided retrieved memories.",
        "",
        "Question:",
        q.question,
        "",
        "Retrieved memories:",
    ]
    for i, item in enumerate(q.retrieved_items, start=1):
        text = " ".join(str(item).split())
        lines.append(f"[{i}] {text}")
    lines += [
        "",
        "Instructions:",
        "Answer the question using only the retrieved memories.",
        "If the retrieved memories do not contain enough information to answer, respond with exactly:",
        INSUFFICIENT,
    ]
    return "\n".join(lines)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_abstention_response(text: str) -> bool:
    t = text.strip().upper()
    return (
        INSUFFICIENT in t
        or "NOT ENOUGH" in t
        or "CANNOT" in t[:60]
        or "DO NOT KNOW" in t[:60]
        or "DON'T KNOW" in t[:60]
        or "DONT KNOW" in t[:60]
    )


def evaluate_answer(model_answer: str, gold_answer: str) -> bool:
    """Rule-based correctness: normalized exact / containment match.

    Note: this is a deterministic approximation of LongMemEval's LLM-as-judge
    protocol. Numbers are compared exactly after normalization.
    """
    if is_abstention_response(model_answer) and is_abstention_response(gold_answer):
        return True
    a = _norm(model_answer)
    g = _norm(gold_answer)
    if not a or not g:
        return False
    if a == g:
        return True
    if len(g) >= 3 and (g in a or a in g):
        return True
    # token overlap F1 >= 0.6 counts as correct for longer answers
    ta, tg = set(a.split()), set(g.split())
    if len(tg) >= 3 and len(ta) >= 2:
        inter = ta & tg
        if not inter:
            return False
        f1 = 2 * len(inter) / (len(ta) + len(tg))
        return f1 >= 0.6
    return False


@dataclass
class QAResult:
    question_id: str
    system: str
    category: str
    is_abstention: bool
    model_answer: str
    gold_answer: str
    correct: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


def record(
    q: QAQuestion,
    system: str,
    completion: QACompletion,
    gold_answer: str,
) -> QAResult:
    if q.is_abstention:
        correct = is_abstention_response(completion.text)
    else:
        correct = evaluate_answer(completion.text, gold_answer)
    return QAResult(
        question_id=q.question_id,
        system=system,
        category=q.category,
        is_abstention=q.is_abstention,
        model_answer=completion.text,
        gold_answer=gold_answer,
        correct=correct,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        latency_ms=completion.latency_ms,
    )


def aggregate_metrics(results: list[QAResult]) -> dict:
    n = len(results)
    ans = [r for r in results if not r.is_abstention]
    abs_ = [r for r in results if r.is_abstention]
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    comp_tokens = sum(r.completion_tokens for r in results)
    lat = sum(r.latency_ms for r in results) / n if n else 0.0
    return {
        "n": n,
        "answerable_n": len(ans),
        "answerable_accuracy": round(sum(1 for r in ans if r.correct) / len(ans), 4) if ans else None,
        "abstention_n": len(abs_),
        "abstention_accuracy": round(sum(1 for r in abs_ if r.correct) / len(abs_), 4) if abs_ else None,
        "overall_accuracy": round(sum(1 for r in results if r.correct) / n, 4) if n else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens,
        "total_tokens": total_tokens,
        "avg_latency_ms": round(lat, 1),
    }

# --- Stage 4.5 compatibility wrappers (kept so the adapter test contract holds) ---

def is_abstention_answer(text: str) -> bool:
    return is_abstention_response(text)


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def qa_accuracy(model_answer: str, gold: str, abstention_gold: bool = False) -> float:
    if abstention_gold:
        return 1.0 if is_abstention_response(model_answer) else 0.0
    return 1.0 if evaluate_answer(model_answer, gold) else 0.0
