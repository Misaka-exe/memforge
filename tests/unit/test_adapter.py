"""Stage 4.5: LongMemEval adapter + evaluation tests (4 tests, offline)."""
from pathlib import Path

from benchmarks.adapters.base import QuestionCategory
from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation.qa import is_abstention_answer, qa_accuracy, token_overlap

SAMPLE = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "data" / "sample_longmemeval.jsonl"


def test_load_sample_longmemeval():
    cases = load_longmemeval(SAMPLE)
    assert len(cases) == 3
    assert cases[0].case_id == "lme_ssu_001"
    assert len(cases[0].memories) == 2
    assert cases[0].questions[0].answer == "Alice"


def test_category_mapping():
    cases = load_longmemeval(SAMPLE)
    assert cases[0].questions[0].category == QuestionCategory.SINGLE_SESSION_USER
    assert cases[1].questions[0].category == QuestionCategory.MULTI_SESSION


def test_abstention_detected():
    cases = load_longmemeval(SAMPLE)
    q = cases[2].questions[0]
    assert q.question_id.endswith("_abs")
    assert q.is_abstention is True
    assert q.category == QuestionCategory.ABSTENTION


def test_qa_abstention_and_overlap():
    assert qa_accuracy("I don't know", "unknown", abstention_gold=True) == 1.0
    assert qa_accuracy("I think it might be Paris", "unknown", abstention_gold=True) == 0.0
    assert token_overlap("Alice works at Google", "Alice works at Google") > 0.9
    assert is_abstention_answer("Sorry, I cannot answer this from the conversation.") is True