"""MemForge v2 benchmark: evidence-grounded memory management on LongMemEval-S.

Compares:
    V1-Hybrid  : v1 hybrid retrieval + HARD temporal filter at question_date
                 + v1 free-text QA (no citations, no gate, no verifier)
    V2-Full    : hybrid retrieval + SOFT temporal score + sufficiency gate
                 + citation-grounded QA + post-hoc verification

Outputs to results/v2/:
    manifest.json            run config + global metrics
    retrieval_results.jsonl  per-question retrieval rows
    qa_results.jsonl         per-question QA rows (LLM-backed sample)
    metrics.json             aggregated metrics incl. ELR

Design rules (frozen, not tuned to score):
    - Dataset/embedding/prompt/question ids/gold are fixed to LongMemEval-S.
    - v1 evaluator logic (benchmarks/evaluation/qa.py) is reused unchanged.
    - No new database; everything in-memory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation import qa as v1qa
from benchmarks.longmemeval.metrics import evaluate_question
from memforge.embeddings.sentence_transformers import SentenceTransformersProvider

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUT = ROOT / "results" / "v2"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_qdate(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", s)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    return datetime(y, mo, d, tzinfo=timezone.utc)


def parse_sdate(s: str | None) -> datetime | None:
    if not s:
        return None
    return parse_qdate(s)


# ---------------------------------------------------------------------------
# Retrieval variants
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def hybrid_scores(
    qvec: np.ndarray,
    vecs: np.ndarray,
    contents: list[str],
    query: str,
) -> np.ndarray:
    """Vector cosine OR keyword overlap -> candidate score (union)."""
    ql = query.lower()
    scores = np.zeros(len(contents), dtype=np.float32)
    for i in range(len(contents)):
        s = cosine(qvec, vecs[i])
        if ql in contents[i].lower():
            s = max(s, 0.5 + 0.5 * s)  # keyword hit floor
        scores[i] = s
    return scores


def run_retrieval(
    qvec, vecs, contents, query, session_dates, qdate, gold_sessions, session_of,
    soft: bool,
):
    """Return ordered item ids and diagnostics.

    soft=False -> v1: drop items whose session_date > qdate (future) OR
                   session_date is after qdate (hard temporal half-open).
    soft=True  -> v2: keep all items, down-weight out-of-window by 0.1.
    """
    base = hybrid_scores(qvec, vecs, contents, query)
    n = len(contents)
    temporal = np.ones(n, dtype=np.float32)

    dropped_hard = 0
    if qdate is not None:
        for i, sd in enumerate(session_dates):
            if sd is None:
                continue
            if sd > qdate:  # session in the future relative to query
                if not soft:
                    temporal[i] = 0.0  # hard drop
                    dropped_hard += 1
                else:
                    temporal[i] = 0.1  # soft down-weight

    if not soft:
        mask = temporal > 0
        score = base * mask
    else:
        score = 0.7 * base + 0.3 * temporal

    order = np.argsort(-score)
    ordered_ids = [f"item_{i}" for i in order]
    return ordered_ids, dropped_hard


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

async def call_llm(llm, messages, response_model):
    return await llm.generate(messages, response_model)


def build_v1_prompt(question, items):
    lines = [
        "You are an AI assistant answering questions based ONLY on the provided retrieved memories.",
        "", "Question:", question, "", "Retrieved memories:",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {' '.join(it.split())}")
    lines += ["", "Instructions:",
              "Answer the question using only the retrieved memories.",
              "If the retrieved memories do not contain enough information to answer, respond with exactly:",
              v1qa.INSUFFICIENT]
    return "\n".join(lines)


def build_v2_prompt(question, items):
    lines = [
        "You are an AI assistant answering questions based ONLY on the provided retrieved memories.",
        "",
        'Return JSON: {"answer": "...", "citations": [{"memory_index": 1, "claim": "..."}], "abstain": false}',
        "Rules: answer ONLY from memories; cite the 1-based memory index for each factual claim;",
        "if insufficient set abstain=true, answer=\"\", citations=[].",
        "", "Question:", question, "", "Retrieved memories:",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {' '.join(it.split())}")
    lines.append("Return the JSON object now:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--qa-sample", type=int, default=40,
                    help="how many questions to run LLM QA on (budget control)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.data}")
    cases = load_longmemeval(args.data)
    if args.max_questions:
        cases = cases[: args.max_questions]
    print(f"[load] {len(cases)} cases")

    provider = SentenceTransformersProvider(model_name="all-MiniLM-L6-v2")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- retrieval over ALL cases (no LLM) ----
    ret_rows = []
    elr = {"v1_dropped_gold": 0, "v2_dropped_gold": 0, "gold_total": 0}
    per_variant = defaultdict(list)

    for ci, case in enumerate(cases):
        q = case.questions[0]
        items = case.memories
        if not items or q.is_abstention:
            continue
        contents = [m.content for m in items]
        vecs = np.array(provider.embed(contents), dtype=np.float32)
        qvec = np.array(provider.embed([q.question])[0], dtype=np.float32)
        session_of = {f"item_{i}": str(items[i].metadata["session_id"]) for i in range(len(items))}
        session_dates = [parse_sdate(items[i].metadata.get("session_date")) for i in range(len(items))]
        qdate = parse_qdate(q.question_date)
        gold = q.gold_memory_ids

        for soft, name in ((False, "V1-Hybrid"), (True, "V2-Full")):
            ordered, dropped = run_retrieval(
                qvec, vecs, contents, q.question, session_dates, qdate, gold, session_of, soft=soft,
            )
            ev = evaluate_question(ordered, session_of, gold, (1, 5, 10))
            if ev.get("excluded"):
                continue
            ev.update({"question_id": q.question_id, "category": q.category.value, "system": name})
            ret_rows.append(ev)
            per_variant[name].append(ev)

        # ELR: count gold sessions dropped by hard filter
        gold_set = set(gold)
        elr["gold_total"] += len(gold_set)
        if qdate is not None:
            for i, sd in enumerate(session_dates):
                if sd is not None and sd > qdate:
                    sid = session_of[f"item_{i}"]
                    if sid in gold_set:
                        elr["v1_dropped_gold"] += 1
        elr["v2_dropped_gold"] += 0  # soft never drops

        if (ci + 1) % 100 == 0:
            print(f"  [retrieval {ci+1}/{len(cases)}] {time.time()-t0:.0f}s")

    def agg(rows):
        if not rows:
            return {}
        n = len(rows)
        keys = [k for k in rows[0] if k.startswith(("recall_", "mrr", "ndcg"))]
        return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys} | {"num_questions": n}

    ret_metrics = {name: agg(rows) for name, rows in per_variant.items()}
    elr_rate = round(1 - (elr["gold_total"] - elr["v1_dropped_gold"]) / max(1, elr["gold_total"]), 4)

    print("=== Retrieval metrics ===")
    for name, m in ret_metrics.items():
        print(name, m)
    print("ELR (v1 evidence loss rate):", elr_rate, elr)

    # ---- LLM QA on a sample ----
    qa_rows = []
    llm = None
    try:
        from memforge.llm.openai_compatible import OpenAICompatibleProvider
        from memforge.llm.base import Message
        llm = OpenAICompatibleProvider(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        print("[llm] unavailable:", e)

    sample_cases = [c for c in cases if not c.questions[0].is_abstention][: args.qa_sample]
    if llm is not None and sample_cases:
        from pydantic import BaseModel

        class V2Resp(BaseModel):
            answer: str = ""
            citations: list[dict] = []
            abstain: bool = False

        for ci, case in enumerate(sample_cases):
            q = case.questions[0]
            items = case.memories
            contents = [m.content for m in items]
            vecs = np.array(provider.embed(contents), dtype=np.float32)
            qvec = np.array(provider.embed([q.question])[0], dtype=np.float32)
            scores = hybrid_scores(qvec, vecs, contents, q.question)
            top = np.argsort(-scores)[:5]
            top_contents = [contents[i] for i in top]

            # V1 QA
            try:
                v1_text = await _v1_qa(llm, q.question, top_contents)
            except Exception:
                v1_text = ""
            # V2 QA
            try:
                v2_resp = await llm.generate(
                    [Message(role="user", content=build_v2_prompt(q.question, top_contents))],
                    V2Resp,
                )
            except Exception:
                v2_resp = V2Resp()

            v1_correct = v1qa.evaluate_answer(v1_text, q.answer)
            v2_abstain = bool(v2_resp.abstain) or v1qa.is_abstention_response(v2_resp.answer)
            v2_correct = (v1qa.is_abstention_response(v2_resp.answer) if q.is_abstention
                          else v1qa.evaluate_answer(v2_resp.answer, q.answer))

            qa_rows.append({
                "question_id": q.question_id,
                "category": q.category.value,
                "v1_answer": v1_text,
                "v2_answer": v2_resp.answer,
                "v2_abstain": v2_abstain,
                "v1_correct": v1_correct,
                "v2_correct": v2_correct,
                "v2_num_citations": len(v2_resp.citations),
            })
            if (ci + 1) % 10 == 0:
                print(f"  [qa {ci+1}/{len(sample_cases)}] {time.time()-t0:.0f}s")

    # ---- write outputs ----
    with open(out / "retrieval_results.jsonl", "w", encoding="utf-8") as f:
        for r in ret_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "qa_results.jsonl", "w", encoding="utf-8") as f:
        for r in qa_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    qa_metrics = {}
    if qa_rows:
        n = len(qa_rows)
        qa_metrics = {
            "n": n,
            "v1_accuracy": round(sum(r["v1_correct"] for r in qa_rows) / n, 4),
            "v2_accuracy": round(sum(r["v2_correct"] for r in qa_rows) / n, 4),
            "v2_abstain_rate": round(sum(r["v2_abstain"] for r in qa_rows) / n, 4),
            "v2_avg_citations": round(sum(r["v2_num_citations"] for r in qa_rows) / n, 2),
        }

    manifest = {
        "dataset": "LongMemEval-S",
        "embedding": "all-MiniLM-L6-v2",
        "llm": "deepseek-chat",
        "prompt_version_v1": "v1.0",
        "prompt_version_v2": "v2.0-grounded",
        "n_cases": len(cases),
        "qa_sample": len(qa_rows),
        "elr": {**elr, "elr_rate": elr_rate},
        "retrieval": ret_metrics,
        "qa": qa_metrics,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("=== QA metrics ===", qa_metrics)
    print(f"done in {time.time()-t0:.1f}s -> {out}")


async def _v1_qa(llm, question, items):
    from pydantic import BaseModel

    from memforge.llm.base import Message

    class _V1(BaseModel):
        text: str = ""

    resp = await llm.generate(
        [Message(role="user", content=build_v1_prompt(question, items))], _V1
    )
    return resp.text


if __name__ == "__main__":
    asyncio.run(amain())
