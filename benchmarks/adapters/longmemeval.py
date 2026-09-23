"""LongMemEval JSON -> BenchmarkCase adapter.

Official dataset: longmemeval_s_cleaned.json (500 questions, JSON array).
Each instance has: question_id, question_type, question, answer,
question_date, haystack_session_ids, haystack_dates, haystack_sessions,
answer_session_ids.

Data conventions (verified against the actual downloaded file):
- The file is a single JSON array (NOT JSONL).
- Abstention is NOT a question_type; it is marked by the question_id
  suffix "_abs".  Its answer_session_ids contain sentinel ids like
  "answer_<hex>_abs" that do NOT mark real evidence.
- For normal questions, answer_session_ids are real haystack session ids
  and the evidence turns inside them carry "has_answer": true.

Memory granularity: one MemoryItem per turn, with metadata carrying
session_id / question_id / turn_index / has_answer so results can be
mapped back to official answer_session_ids.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.adapters.base import (
    BenchmarkCase,
    BenchmarkQuestion,
    MemoryItem,
    QuestionCategory,
)

CATEGORY_MAP = {
    "single-session-user": QuestionCategory.SINGLE_SESSION_USER,
    "single-session-assistant": QuestionCategory.SINGLE_SESSION_ASSISTANT,
    "single-session-preference": QuestionCategory.SINGLE_SESSION_PREFERENCE,
    "multi-session": QuestionCategory.MULTI_SESSION,
    "temporal-reasoning": QuestionCategory.TEMPORAL_REASONING,
    "knowledge-update": QuestionCategory.KNOWLEDGE_UPDATE,
}


def _load_rows(path: str | Path) -> list[dict]:
    """Detect format: single JSON array (official) or JSONL (test samples)."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
    if first == "[":
        return json.load(open(path, encoding="utf-8"))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_sessions(
    question_id: str,
    haystack_sessions: list,
    session_ids: list,
    session_dates: list | None = None,
) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    for sidx, session in enumerate(haystack_sessions):
        turns = session if isinstance(session, list) else session.get("messages", [])
        official_id = session_ids[sidx] if sidx < len(session_ids) else str(sidx)
        sdate = session_dates[sidx] if session_dates and sidx < len(session_dates) else None
        for tidx, turn in enumerate(turns):
            role = turn.get("role", "")
            text = turn.get("content", "")
            if role in ("user", "assistant") and text:
                items.append(
                    MemoryItem(
                        id=f"{question_id}:{sidx}:{tidx}",
                        content=text,
                        timestamp=turn.get("timestamp"),
                        metadata={
                            "session_id": official_id,
                            "session_index": sidx,
                            "session_date": sdate,
                            "question_id": question_id,
                            "turn_index": tidx,
                            "role": role,
                            "has_answer": turn.get("has_answer", False),
                        },
                    )
                )
    return items


def load_longmemeval(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for row in _load_rows(path):
        qid = row["question_id"]
        qtype = row.get("question_type", "multi-session")
        is_abs = qid.endswith("_abs")
        if is_abs:
            category = QuestionCategory.ABSTENTION
        else:
            category = CATEGORY_MAP.get(qtype, QuestionCategory.MULTI_SESSION)

        memories = _parse_sessions(
            qid,
            row.get("haystack_sessions", []),
            list(row.get("haystack_session_ids", [])),
            list(row.get("haystack_dates", [])),
        )
        # Abstention questions have no retrievable evidence location.
        gold = [] if is_abs else list(row.get("answer_session_ids", []))
        q = BenchmarkQuestion(
            question_id=qid,
            question=row.get("question", ""),
            answer=str(row.get("answer", "")),
            category=category,
            gold_memory_ids=gold,
            is_abstention=is_abs,
            question_date=row.get("question_date"),
        )
        cases.append(BenchmarkCase(case_id=qid, memories=memories, questions=[q]))
    return cases