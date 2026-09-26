"""Stage 5 v2: LongMemEval-S end-to-end QA with evidence grounding.

Compares v1 hybrid (plain QA) vs v2 (gate + citation grounding + verification).
Reuses the frozen retrieval inputs from stage5_real/retrieved/hybrid/.

Usage:
    # smoke (10 questions)
    python -m benchmarks.scripts.run_longmemeval_qa_v2 --smoke

    # full 500 questions
    python -m benchmarks.scripts.run_longmemeval_qa_v2

Env: MEMFORGE_LLM_MODEL / MEMFORGE_LLM_BASE_URL / MEMFORGE_LLM_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation.qa import evaluate_answer, is_abstention_response
from benchmarks.evaluation.qa_v2 import V2QAPipeline, V2QAQuestion
from memforge.core.types import Memory, MemoryStatus
from memforge.llm.openai_compatible import OpenAICompatibleProvider

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
RETRIEVED_DIR = ROOT / "results" / "longmemeval" / "stage5_real" / "retrieved" / "hybrid"
OUT_DIR = ROOT / "results" / "longmemeval" / "stage5_v2"


def load_retrieved(question_id: str) -> list[Memory]:
    """Load frozen retrieval input for a question, convert to Memory objects."""
    path = RETRIEVED_DIR / f"{question_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    texts = data.get("texts", [])
    ids = data.get("ids", [])
    memories = []
    for i, text in enumerate(texts):
        mid = ids[i] if i < len(ids) else f"{question_id}:{i}"
        memories.append(Memory(
            content=text,
            status=MemoryStatus.ACTIVE,
            importance=0.8,
            metadata={"retrieval_id": mid, "rank": i},
        ))
    return memories


def build_question_cases(data_path: str, smoke: bool = False) -> list[V2QAQuestion]:
    """Build V2QAQuestion list from LongMemEval data + frozen retrieval."""
    cases = load_longmemeval(data_path)
    questions: list[V2QAQuestion] = []

    for case in cases:
        for q in case.questions:
            retrieved = load_retrieved(q.question_id)
            if not retrieved:
                continue
            questions.append(V2QAQuestion(
                question_id=q.question_id,
                question=q.question,
                gold_answer=q.answer,
                is_abstention=q.is_abstention,
                category=q.category.value if hasattr(q.category, "value") else str(q.category),
                retrieved_items=retrieved,
            ))

    if smoke:
        # Pick 5 answerable + 5 abstention, diverse categories
        ans = [q for q in questions if not q.is_abstention]
        abs_ = [q for q in questions if q.is_abstention]
        selected_ans = []
        seen_cats = set()
        for q in ans:
            if q.category not in seen_cats or len(selected_ans) < 3:
                selected_ans.append(q)
                seen_cats.add(q.category)
            if len(selected_ans) >= 5:
                break
        return selected_ans + abs_[:5]

    return questions


async def run_benchmark(questions: list[V2QAQuestion], smoke: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "raw_qa.jsonl"
    metrics_path = OUT_DIR / "metrics.json"

    # Resume: load already-done question ids
    done_ids: set[str] = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["question_id"])

    llm = OpenAICompatibleProvider(
        model=os.environ.get("MEMFORGE_LLM_MODEL", "deepseek-chat"),
        base_url=os.environ.get("MEMFORGE_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.environ.get("MEMFORGE_LLM_API_KEY", ""),
        temperature=0.0,
    )
    pipeline = V2QAPipeline(llm=llm)

    results: list[dict] = []
    total = len(questions)
    start = time.monotonic()

    for i, q in enumerate(questions, start=1):
        if q.question_id in done_ids:
            continue

        result = await pipeline.answer(q)

        # Evaluate using v1's deterministic evaluator (consistent comparison)
        model_answer = result.grounded_answer.answer if not result.abstained else "INSUFFICIENT_INFORMATION"
        if q.is_abstention:
            correct = is_abstention_response(model_answer)
        else:
            correct = evaluate_answer(model_answer, q.gold_answer)

        rec = {
            "question_id": q.question_id,
            "category": q.category,
            "is_abstention": q.is_abstention,
            "gold_answer": q.gold_answer,
            "model_answer": model_answer,
            "correct": correct,
            "abstained": result.abstained,
            "gate_decision": result.gate_decision,
            "citation_count": len(result.grounded_answer.citations),
            "citation_coverage": result.grounded_answer.citation_coverage(),
            "verification_verdict": result.verification.verdict.value if result.verification else None,
            "verification_unsupported_rate": result.verification.unsupported_claim_rate if result.verification else None,
            "latency_ms": round(result.latency_ms, 1),
            "error": result.error,
        }
        results.append(rec)

        # Append to checkpoint
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if i % 50 == 0 or i == total:
            elapsed = time.monotonic() - start
            print(f"[{i}/{total}] elapsed={elapsed:.0f}s correct_so_far={sum(r['correct'] for r in results)}", flush=True)

    # Aggregate metrics
    all_records = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_records.append(json.loads(line))

    n = len(all_records)
    answerable = [r for r in all_records if not r["is_abstention"]]
    abstention = [r for r in all_records if r["is_abstention"]]

    metrics = {
        "system": "v2_grounded",
        "n": n,
        "answerable_n": len(answerable),
        "abstention_n": len(abstention),
        "overall_accuracy": sum(r["correct"] for r in all_records) / n if n else 0,
        "answerable_accuracy": sum(r["correct"] for r in answerable) / len(answerable) if answerable else 0,
        "abstention_accuracy": sum(r["correct"] for r in abstention) / len(abstention) if abstention else 0,
        "abstention_rate": sum(r["abstained"] for r in all_records) / n if n else 0,
        "gate_sufficient_rate": sum(1 for r in all_records if r["gate_decision"] == "SUFFICIENT") / n if n else 0,
        "gate_insufficient_rate": sum(1 for r in all_records if r["gate_decision"] == "INSUFFICIENT") / n if n else 0,
        "avg_citations": sum(r["citation_count"] for r in all_records) / n if n else 0,
        "verification_pass_rate": sum(1 for r in all_records if r["verification_verdict"] == "PASS") / n if n else 0,
        "verification_fail_rate": sum(1 for r in all_records if r["verification_verdict"] == "FAIL") / n if n else 0,
        "avg_latency_ms": sum(r["latency_ms"] for r in all_records) / n if n else 0,
    }

    # By category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        by_cat[r["category"]].append(r)
    cat_metrics = {}
    for cat, recs in sorted(by_cat.items()):
        cat_metrics[cat] = {
            "n": len(recs),
            "accuracy": sum(r["correct"] for r in recs) / len(recs),
            "abstention_rate": sum(r["abstained"] for r in recs) / len(recs),
        }
    metrics["by_category"] = cat_metrics

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== v2 Benchmark Complete ===")
    print(f"n={n} overall={metrics['overall_accuracy']:.3f} answerable={metrics['answerable_accuracy']:.3f} abstention={metrics['abstention_accuracy']:.3f}")
    print(f"Results saved to {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run 10-question smoke test")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    args = parser.parse_args()

    questions = build_question_cases(args.data, smoke=args.smoke)
    print(f"Loaded {len(questions)} questions ({'smoke' if args.smoke else 'full'})")
    asyncio.run(run_benchmark(questions, smoke=args.smoke))


if __name__ == "__main__":
    main()
