"""Schema / distribution / abstention validation for LongMemEval-S.

Usage: python -m benchmarks.scripts.validate_longmemeval
Writes: results/longmemeval/validation.json + prints a readable summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "longmemeval_s_cleaned.json"
OUT = Path(__file__).resolve().parent.parent.parent / "results" / "longmemeval" / "validation.json"

REQUIRED_FIELDS = [
    "question_id", "question_type", "question", "answer", "question_date",
    "haystack_session_ids", "haystack_dates", "haystack_sessions", "answer_session_ids",
]


def main() -> None:
    rows = json.load(open(DATA, encoding="utf-8"))
    report = {
        "num_instances": len(rows),
        "unique_question_ids": len({r["question_id"] for r in rows}),
        "format": "json_array",
        "question_type_distribution": dict(Counter(r["question_type"] for r in rows)),
        "missing_fields": {},
        "abstention": {},
        "sessions_per_question": {},
        "answer_session_ids": {},
    }

    for field in REQUIRED_FIELDS:
        missing = [r["question_id"] for r in rows if field not in r]
        if missing:
            report["missing_fields"][field] = missing[:10]

    abs_rows = [r for r in rows if str(r["question_id"]).endswith("_abs")]
    report["abstention"] = {
        "count": len(abs_rows),
        "ids_end_with_abs": len(abs_rows),
        "sentinel_examples": [r["answer_session_ids"] for r in abs_rows[:3]],
        "sentinel_all": all(
            len(r["answer_session_ids"]) == 1
            and str(r["answer_session_ids"][0]).endswith("_abs")
            for r in abs_rows
        ),
    }

    sizes = [len(r["haystack_sessions"]) for r in rows]
    report["sessions_per_question"] = {
        "min": min(sizes), "max": max(sizes), "avg": round(sum(sizes) / len(sizes), 2),
    }

    gold_lens = [len(r["answer_session_ids"]) for r in rows if not r["question_id"].endswith("_abs")]
    report["answer_session_ids"] = {
        "non_abs_with_gold": len(gold_lens),
        "avg_gold_sessions": round(sum(gold_lens) / len(gold_lens), 2),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("num_instances:", report["num_instances"])
    print("unique_question_ids:", report["unique_question_ids"])
    print("distribution:", report["question_type_distribution"])
    print("missing_fields:", report["missing_fields"] or "none")
    print("abstention:", report["abstention"])
    print("sessions/question:", report["sessions_per_question"])
    print("answer_session_ids:", report["answer_session_ids"])
    print("saved:", OUT)


if __name__ == "__main__":
    main()