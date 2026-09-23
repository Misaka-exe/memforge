"""Stage 5 QA layer unit tests (zero external deps)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.evaluation.qa import (  # noqa: E402
    INSUFFICIENT,
    QACompletion,
    QAQuestion,
    aggregate_metrics,
    build_prompt,
    evaluate_answer,
    is_abstention_response,
    record,
)


def _q(text="What is the user's major?", gold="Business Administration", abstention=False, cat="single-session-user", items=("User studies business administration.",)):
    return QAQuestion(
        question_id="q1" + ("_abs" if abstention else ""),
        question=text,
        gold_answer=gold,
        is_abstention=abstention,
        category=cat,
        retrieved_items=list(items),
    )


def test_prompt_contains_no_gold_answer():
    q = _q(gold="Business Administration")
    p = build_prompt(q)
    assert "Business Administration" not in p
    assert "Question:" in p
    assert "[1] User studies business administration." in p
    assert INSUFFICIENT in p


def test_prompt_abstention_question_no_gold():
    q = _q(gold="INSUFFICIENT_INFORMATION", abstention=True, items=("User ordered a laptop.",))
    p = build_prompt(q)
    assert "INSUFFICIENT_INFORMATION" in p  # only as the instruction token
    assert "User ordered a laptop." in p


def test_abstention_judgement():
    assert is_abstention_response("INSUFFICIENT_INFORMATION")
    assert is_abstention_response("I cannot answer based on the memories.")
    assert is_abstention_response("not enough information provided")
    assert not is_abstention_response("Business Administration")


def test_evaluate_answer_exact():
    assert evaluate_answer("Business Administration", "Business Administration")
    assert evaluate_answer("45 minutes", "45 minutes")


def test_evaluate_answer_containment():
    assert evaluate_answer("The user studies Business Administration at the university", "Business Administration")
    assert evaluate_answer("Business Administration", "the user studies Business Administration")


def test_evaluate_answer_mismatch():
    assert not evaluate_answer("Computer Science", "Business Administration")
    assert not evaluate_answer("42", "45 minutes each way")


def test_abstention_gold_judgement():
    # _abs question: model must output INSUFFICIENT to be correct
    q = _q(gold="INSUFFICIENT_INFORMATION", abstention=True, items=("User mentioned nothing relevant.",))
    r = record(q, "hybrid", QACompletion(text="INSUFFICIENT_INFORMATION"), q.gold_answer)
    assert r.correct is True
    r2 = record(q, "hybrid", QACompletion(text="Business Administration"), q.gold_answer)
    assert r2.correct is False


def test_answerable_record():
    q = _q(items=("User studies business administration.",))
    r = record(q, "hybrid", QACompletion(text="Business Administration", prompt_tokens=50, completion_tokens=3, latency_ms=12.0), q.gold_answer)
    assert r.correct is True
    assert r.prompt_tokens == 50 and r.completion_tokens == 3 and r.latency_ms == 12.0


def test_aggregate_metrics():
    qa = _q(items=("x",))
    ab = _q(gold="INSUFFICIENT_INFORMATION", abstention=True, items=("x",))
    results = [
        record(qa, "hybrid", QACompletion(text="Business Administration", prompt_tokens=5, completion_tokens=1), qa.gold_answer),
        record(ab, "hybrid", QACompletion(text="INSUFFICIENT_INFORMATION", prompt_tokens=5, completion_tokens=1), ab.gold_answer),
        record(qa, "hybrid", QACompletion(text="Computer Science", prompt_tokens=5, completion_tokens=1), qa.gold_answer),
    ]
    m = aggregate_metrics(results)
    assert m["answerable_n"] == 2
    assert m["answerable_accuracy"] == 0.5
    assert m["abstention_n"] == 1
    assert m["abstention_accuracy"] == 1.0
    assert m["overall_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["total_tokens"] == 18


def test_provider_mock():
    from benchmarks.evaluation.providers import MockQAProvider

    p = MockQAProvider()
    p.add_script(r"Question:", "42")
    import asyncio

    prompt = "Question: what?\n" + INSUFFICIENT
    c = asyncio.run(p.complete(prompt))
    assert c.text == "42"
    assert c.prompt_tokens == len(prompt.split())
