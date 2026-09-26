"""Quick QA-only re-run: fix v1 answer extraction (raw text, not pydantic)."""
from __future__ import annotations

import asyncio
import json
import time

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation import qa as v1qa
from memforge.embeddings.sentence_transformers import SentenceTransformersProvider
from memforge.llm.base import Message
from memforge.llm.openai_compatible import OpenAICompatibleProvider


def build_v1_prompt(question, items):
    lines = [
        "You are an AI assistant answering questions based ONLY on the provided retrieved memories.",
        "", "Question:", question, "", "Retrieved memories:",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {' '.join(it.split())}")
    lines += ["", "Instructions:",
              "Answer the question using only the retrieved memories.",
              "If insufficient, respond with exactly:", v1qa.INSUFFICIENT]
    return "\n".join(lines)


def build_v2_prompt(question, items):
    lines = [
        "You are an AI assistant answering questions based ONLY on the provided retrieved memories.",
        'Return JSON: {"answer": "...", "citations": [{"memory_index": 1, "claim": "..."}], "abstain": false}',
        "Rules: answer ONLY from memories; cite the 1-based index; if insufficient abstain=true.",
        "", "Question:", question, "", "Retrieved memories:",
    ]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {' '.join(it.split())}")
    lines.append("Return the JSON object now:")
    return "\n".join(lines)


async def main():
    import numpy as np
    from pydantic import BaseModel

    cases = load_longmemeval("benchmarks/data/longmemeval_s_cleaned.json")
    provider = SentenceTransformersProvider(model_name="all-MiniLM-L6-v2")
    llm = OpenAICompatibleProvider(model="deepseek-chat",
                                   base_url="https://api.deepseek.com/v1",
                                   temperature=0.0)

    class V2Resp(BaseModel):
        answer: str = ""
        citations: list[dict] = []
        abstain: bool = False

    sample = [c for c in cases if not c.questions[0].is_abstention][:30]
    rows = []
    for case in sample:
        q = case.questions[0]
        contents = [m.content for m in case.memories]
        vecs = np.array(provider.embed(contents), dtype=np.float32)
        qv = np.array(provider.embed([q.question])[0], dtype=np.float32)
        # simple cosine rank
        sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-8)
        top = np.argsort(-sims)[:5]
        top_contents = [contents[i] for i in top]

        v1 = await llm.generate_raw([Message(role="user", content=build_v1_prompt(q.question, top_contents))])
        v2 = await llm.generate([Message(role="user", content=build_v2_prompt(q.question, top_contents))], V2Resp)

        v1_text = v1.text.strip()
        v1_correct = v1qa.evaluate_answer(v1_text, q.answer)
        v2_abstain = bool(v2.abstain) or v1qa.is_abstention_response(v2.answer)
        v2_correct = v1qa.evaluate_answer(v2.answer, q.answer)

        rows.append({
            "question_id": q.question_id, "category": q.category.value,
            "v1_answer": v1_text[:200], "v2_answer": v2.answer[:200],
            "v2_abstain": v2_abstain, "v1_correct": v1_correct, "v2_correct": v2_correct,
            "v2_num_citations": len(v2.citations),
        })
        print(".", end="", flush=True)

    n = len(rows)
    metrics = {
        "n": n,
        "v1_accuracy": round(sum(r["v1_correct"] for r in rows) / n, 4),
        "v2_accuracy": round(sum(r["v2_correct"] for r in rows) / n, 4),
        "v2_abstain_rate": round(sum(r["v2_abstain"] for r in rows) / n, 4),
        "v2_avg_citations": round(sum(r["v2_num_citations"] for r in rows) / n, 2),
    }
    print("\n", metrics)
    with open("results/v2/qa_results_fixed.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # update metrics.json
    m = json.load(open("results/v2/metrics.json", encoding="utf-8"))
    m["qa"] = metrics
    json.dump(m, open("results/v2/metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
