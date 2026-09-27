"""V2.1 Real LLM QA experiment on LongMemEval-S.

Two configurations:
  v21_recon: V2 grounded QA + verification + reconsolidation (no gate)
  v21_full:  V2.1 calibrated gate + V2 grounded QA + verification + reconsolidation

Reuses V2 frozen retrieval inputs (no re-embedding, no re-retrieval).
Uses V1 deterministic evaluator for consistent comparison with V2 baseline.

Checkpoint/resume: each question appended to raw_qa.jsonl immediately.
Already-completed questions are skipped on resume.

Usage:
  # dry-run (5 questions)
  python -m benchmarks.scripts.run_v21_real_qa --system v21_recon --dry-run
  python -m benchmarks.scripts.run_v21_real_qa --system v21_full --dry-run

  # full 500 questions
  python -m benchmarks.scripts.run_v21_real_qa --system v21_recon
  python -m benchmarks.scripts.run_v21_real_qa --system v21_full

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
from benchmarks.evaluation.qa_v2 import (
    GroundedLLMResponse,
    V2QAQuestion,
    build_grounded_prompt,
    llm_response_to_grounded,
)
from memforge.core.types import Memory, MemoryStatus
from memforge.llm.base import Message
from memforge.llm.openai_compatible import OpenAICompatibleProvider
from memforge.memory.grounding import GroundedAnswer
from memforge.memory.reconsolidation import MemoryReconsolidator, ReconsolidationDecision
from memforge.retrieval.gate_v2 import MultiFeatureGate, Sufficiency
from memforge.retrieval.score import RetrievalFeatures
from memforge.verification.verifier import Verifier, Verdict

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
RETRIEVED_DIR = ROOT / "results" / "longmemeval" / "stage5_real" / "retrieved" / "hybrid"
FEATURES_PATH = ROOT / "results" / "longmemeval" / "stage_v21" / "retrieval_features.jsonl"
OUT_BASE = ROOT / "results" / "longmemeval" / "stage_v21_real"


def load_retrieved(question_id: str) -> list[Memory]:
    """Load frozen V2 retrieval input for a question."""
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


def load_retrieval_features() -> dict[str, RetrievalFeatures]:
    """Load pre-computed retrieval features for all questions."""
    features = {}
    if not FEATURES_PATH.exists():
        return features
    for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            features[d["question_id"]] = RetrievalFeatures(
                question_id=d["question_id"],
                n_retrieved=d["n_retrieved"],
                top1_score=d["top1_score"],
                top5_mean=d["top5_mean"],
                score_margin=d["score_margin"],
                mean_score=d["mean_score"],
                std_score=d["std_score"],
            )
    return features


def build_questions(data_path: str, dry_run: bool = False) -> list[V2QAQuestion]:
    """Build question list from LongMemEval + frozen retrieval."""
    cases = load_longmemeval(data_path)
    questions = []
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

    if dry_run:
        # Pick 3 answerable + 2 abstention, diverse categories
        ans = [q for q in questions if not q.is_abstention]
        abs_ = [q for q in questions if q.is_abstention]
        selected = []
        seen_cats = set()
        for q in ans:
            if q.category not in seen_cats or len(selected) < 2:
                selected.append(q)
                seen_cats.add(q.category)
            if len(selected) >= 3:
                break
        return selected + abs_[:2]

    return questions


class V21Runner:
    """V2.1 real QA runner with gate + reconsolidation.

    Directly calls LLM (not via V2QAPipeline) so we can capture token usage
    without modifying frozen v2 code. Prompt/evaluator/verifier are identical
    to v2 for fair comparison.
    """

    def __init__(
        self,
        system: str,
        llm: OpenAICompatibleProvider,
        features: dict[str, RetrievalFeatures],
    ):
        self.system = system
        self.llm = llm
        self.features = features
        self.reconsolidator = MemoryReconsolidator(min_confidence=0.6)
        self.verifier = Verifier()

        # Gate: only for v21_full, using calibrated MultiFeatureGate
        self.use_gate = (system == "v21_full")
        if self.use_gate:
            self.gate = MultiFeatureGate(
                top1_threshold=0.3,
                top5_threshold=0.2,
                min_retrieved=3,
            )
        else:
            self.gate = None

    async def _call_llm_grounded(self, question: str, retrieved: list[Memory]) -> tuple[GroundedAnswer, int, int, str]:
        """Call LLM with grounded prompt, return (answer, prompt_tokens, completion_tokens, error)."""
        prompt = build_grounded_prompt(question, retrieved)
        messages = [Message(role="user", content=prompt)]

        try:
            raw = await self.llm.generate_raw(messages, max_tokens=512, temperature=0.0)
            text = raw.text.strip()
            # Parse JSON response
            try:
                # Strip potential markdown code fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1]
                    if text.endswith("```"):
                        text = text.rsplit("```", 1)[0]
                resp = GroundedLLMResponse.model_validate_json(text)
            except Exception:
                # Fallback: treat as plain answer
                resp = GroundedLLMResponse(answer=text, citations=[], abstain=False)
            grounded = llm_response_to_grounded(resp, retrieved)
            return grounded, raw.prompt_tokens, raw.completion_tokens, ""
        except Exception as e:
            return GroundedAnswer(answer="", citations=[], abstain=True, reason="llm_error"), 0, 0, str(e)

    async def run_question(self, q: V2QAQuestion) -> dict:
        """Run one question through the V2.1 pipeline."""
        start = time.monotonic()
        rec = {
            "question_id": q.question_id,
            "system": self.system,
            "category": q.category,
            "is_abstention": q.is_abstention,
            "gold_answer": q.gold_answer,
        }

        gate_decision = "DISABLED"
        gate_score = 0.0
        gate_reason = ""

        try:
            # Step 1: Gate (v21_full only)
            if self.use_gate and q.question_id in self.features:
                decision = self.gate.assess(self.features[q.question_id])
                gate_decision = decision.sufficiency.value
                gate_score = decision.score
                gate_reason = decision.reason

                if decision.sufficiency == Sufficiency.INSUFFICIENT:
                    rec.update({
                        "model_answer": "INSUFFICIENT_INFORMATION",
                        "abstained": True,
                        "correct": is_abstention_response("INSUFFICIENT_INFORMATION") if q.is_abstention else False,
                        "gate_decision": gate_decision,
                        "gate_score": round(gate_score, 4),
                        "gate_reason": gate_reason,
                        "verification_verdict": None,
                        "reconsolidation_decision": "NONE",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_ms": round((time.monotonic() - start) * 1000, 1),
                        "error": "",
                    })
                    return rec

            # Step 2: Grounded LLM call (direct, with token accounting)
            grounded, prompt_tokens, completion_tokens, llm_error = await self._call_llm_grounded(
                q.question, q.retrieved_items
            )

            model_answer = grounded.answer if not grounded.abstain else "INSUFFICIENT_INFORMATION"
            if q.is_abstention:
                correct = is_abstention_response(model_answer)
            else:
                correct = evaluate_answer(model_answer, q.gold_answer)

            # Step 3: Post-hoc verification (only for non-abstention)
            verification = None
            if not grounded.abstain:
                verification = await self.verifier.verify(
                    query=q.question,
                    answer=grounded.answer,
                    evidence=q.retrieved_items,
                    citations=[c.memory_id for c in grounded.citations if c.memory_id],
                )

            # Step 4: Reconsolidation (only if verification FAIL and not abstained)
            recon_decision = "NONE"
            recon_new_id = ""
            recon_reason = ""
            recon_updated_fields = []
            if (not grounded.abstain and
                    verification is not None and
                    verification.verdict == Verdict.FAIL):
                if q.retrieved_items:
                    recon_result = self.reconsolidator.reconsolidate(
                        memory=q.retrieved_items[0],
                        new_evidence=model_answer,
                        verification_fail_reason="verification FAIL",
                        confidence=0.8,
                    )
                    recon_decision = recon_result.decision.value
                    recon_new_id = str(recon_result.new_memory_id) if recon_result.new_memory_id else ""
                    recon_reason = recon_result.reason
                    recon_updated_fields = recon_result.updated_fields

            rec.update({
                "model_answer": model_answer,
                "abstained": grounded.abstain,
                "correct": correct,
                "gate_decision": gate_decision,
                "gate_score": round(gate_score, 4),
                "gate_reason": gate_reason,
                "citation_count": len(grounded.citations),
                "verification_verdict": verification.verdict.value if verification else None,
                "verification_unsupported_rate": (
                    verification.unsupported_claim_rate if verification else None
                ),
                "reconsolidation_decision": recon_decision,
                "reconsolidation_new_memory_id": recon_new_id,
                "reconsolidation_reason": recon_reason,
                "reconsolidation_updated_fields": recon_updated_fields,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "error": llm_error,
            })

        except Exception as e:
            rec.update({
                "model_answer": "INSUFFICIENT_INFORMATION",
                "abstained": True,
                "correct": False,
                "gate_decision": gate_decision,
                "verification_verdict": None,
                "reconsolidation_decision": "NONE",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": round((time.monotonic() - start) * 1000, 1),
                "error": str(e),
            })

        return rec


def aggregate_metrics(raw_path: Path, system: str) -> dict:
    """Aggregate metrics from raw JSONL."""
    records = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    n = len(records)
    answerable = [r for r in records if not r["is_abstention"]]
    abstention = [r for r in records if r["is_abstention"]]

    metrics = {
        "system": system,
        "n": n,
        "answerable_n": len(answerable),
        "abstention_n": len(abstention),
        "overall_accuracy": sum(r["correct"] for r in records) / n if n else 0,
        "answerable_accuracy": sum(r["correct"] for r in answerable) / len(answerable) if answerable else 0,
        "abstention_accuracy": sum(r["correct"] for r in abstention) / len(abstention) if abstention else 0,
        "abstention_rate": sum(r["abstained"] for r in records) / n if n else 0,
        "gate_sufficient_rate": sum(1 for r in records if r.get("gate_decision") == "SUFFICIENT") / n if n else 0,
        "gate_insufficient_rate": sum(1 for r in records if r.get("gate_decision") == "INSUFFICIENT") / n if n else 0,
        "verification_pass_rate": sum(1 for r in records if r.get("verification_verdict") == "PASS") / n if n else 0,
        "verification_fail_rate": sum(1 for r in records if r.get("verification_verdict") == "FAIL") / n if n else 0,
        "reconsolidation_triggered": sum(1 for r in records if r.get("reconsolidation_decision") == "UPDATE") / n if n else 0,
        "reconsolidation_update_count": sum(1 for r in records if r.get("reconsolidation_decision") == "UPDATE"),
        "avg_latency_ms": sum(r["latency_ms"] for r in records) / n if n else 0,
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in records),
        "error_count": sum(1 for r in records if r.get("error")),
    }

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    cat_metrics = {}
    for cat, recs in sorted(by_cat.items()):
        cat_metrics[cat] = {
            "n": len(recs),
            "accuracy": sum(r["correct"] for r in recs) / len(recs),
            "abstention_rate": sum(r["abstained"] for r in recs) / len(recs),
        }
    metrics["by_category"] = cat_metrics

    return metrics


async def run_experiment(system: str, questions: list[V2QAQuestion], dry_run: bool = False):
    out_dir = OUT_BASE / system
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_qa.jsonl"
    metrics_path = out_dir / "metrics.json"

    # Resume
    done_ids = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])

    print(f"System: {system}, Questions: {len(questions)}, Already done: {len(done_ids)}")

    llm = OpenAICompatibleProvider(
        model=os.environ.get("MEMFORGE_LLM_MODEL", "deepseek-chat"),
        base_url=os.environ.get("MEMFORGE_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.environ.get("MEMFORGE_LLM_API_KEY", ""),
        temperature=0.0,
    )
    features = load_retrieval_features()
    print(f"Loaded retrieval features for {len(features)} questions")

    runner = V21Runner(system=system, llm=llm, features=features)

    start = time.monotonic()
    new_results = 0

    for i, q in enumerate(questions, start=1):
        if q.question_id in done_ids:
            continue

        rec = await runner.run_question(q)
        new_results += 1

        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if i % 10 == 0 or i == len(questions) or dry_run:
            elapsed = time.monotonic() - start
            correct_so_far = sum(
                1 for line in raw_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["correct"]
            )
            total_done = sum(1 for _ in raw_path.read_text(encoding="utf-8").splitlines() if _.strip())
            print(f"[{i}/{len(questions)}] new={new_results} done={total_done} "
                  f"correct={correct_so_far}/{total_done} elapsed={elapsed:.0f}s", flush=True)

    # Aggregate
    metrics = aggregate_metrics(raw_path, system)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {system} Complete ===")
    print(f"n={metrics['n']} overall={metrics['overall_accuracy']:.3f} "
          f"answerable={metrics['answerable_accuracy']:.3f} "
          f"abstention={metrics['abstention_accuracy']:.3f}")
    print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["v21_recon", "v21_full"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="run 5 questions")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    args = parser.parse_args()

    questions = build_questions(args.data, dry_run=args.dry_run)
    print(f"Loaded {len(questions)} questions ({'dry-run' if args.dry_run else 'full'})")

    asyncio.run(run_experiment(args.system, questions, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
