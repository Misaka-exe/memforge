"""V2.1 experiment runner (mock LLM, no real API calls).

Runs 4 configurations on synthetic data to validate end-to-end pipeline:
1. V2-Grounded (baseline: no gate, no recon, static)
2. V2.1-Gate (gate only, static)
3. V2.1-Recon (recon only, adaptive)
4. V2.1-Full (gate + recon, adaptive)

This is INFRASTRUCTURE VALIDATION, not a research experiment.
Real LongMemEval-S 500-question experiment requires real LLM API.

Output: results/v21/experiments/
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from memforge.core.types import Memory, MemoryStatus
from memforge.evaluation.pipeline_v21 import V21Config, V21Pipeline
from memforge.retrieval.score import RetrievalFeatures

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "v21" / "experiments"


def make_synthetic_data(n: int = 50):
    """Generate synthetic questions + memories for pipeline validation."""
    data = []
    for i in range(n):
        is_abs = i % 10 == 0  # 10% abstention
        category = ["single-session-user", "multi-session", "temporal-reasoning",
                    "knowledge-update", "single-session-assistant"][i % 5]

        if is_abs:
            question = f"What is the secret code for user {i}?"
            gold = "INSUFFICIENT_INFORMATION"
            memories = [Memory(
                id=uuid.uuid4(),
                content=f"User {i} likes pizza.",
                status=MemoryStatus.ACTIVE,
            )]
            features = RetrievalFeatures(
                question_id=f"q{i:04d}", n_retrieved=1,
                top1_score=0.3, top5_mean=0.3, score_margin=0.0,
                mean_score=0.3, std_score=0.0,
            )
        else:
            question = f"What is user {i}'s favorite food?"
            gold = f"food_{i}"
            memories = [
                Memory(
                    id=uuid.uuid4(),
                    content=f"User {i}'s favorite food is food_{i}.",
                    status=MemoryStatus.ACTIVE,
                ),
                Memory(
                    id=uuid.uuid4(),
                    content=f"User {i} lives in city_{i}.",
                    status=MemoryStatus.ACTIVE,
                ),
            ]
            features = RetrievalFeatures(
                question_id=f"q{i:04d}", n_retrieved=2,
                top1_score=0.7 + (i % 3) * 0.05,
                top5_mean=0.6, score_margin=0.1,
                mean_score=0.55, std_score=0.1,
            )

        data.append({
            "question_id": f"q{i:04d}",
            "question": question,
            "gold_answer": gold,
            "memories": memories,
            "category": category,
            "is_abstention": is_abs,
            "features": features,
        })
    return data


def run_config(name: str, config: V21Config, data: list[dict]) -> dict:
    """Run one configuration and return metrics."""
    pipe = V21Pipeline(config)
    for item in data:
        pipe.run_question(
            question_id=item["question_id"],
            question=item["question"],
            gold_answer=item["gold_answer"],
            memories=item["memories"],
            category=item["category"],
            is_abstention=item["is_abstention"],
            retrieval_features=item["features"],
        )

    metrics = pipe.compute_metrics(name)
    out_path = OUT_DIR / f"{name}_raw.jsonl"
    pipe.save_results(str(out_path))

    result = metrics.to_dict()
    result["raw_results_path"] = str(out_path)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_data(n=50)
    print(f"Generated {len(data)} synthetic questions")

    configs = [
        ("v2_grounded", V21Config(mode="static")),
        ("v21_gate", V21Config(mode="static", enable_gate=True, gate_threshold=0.5)),
        ("v21_recon", V21Config(mode="adaptive", enable_reconsolidation=True)),
        ("v21_full", V21Config(mode="adaptive", enable_gate=True, enable_reconsolidation=True, gate_threshold=0.5)),
    ]

    all_results = {}
    for name, config in configs:
        print(f"\n=== {name} ===")
        result = run_config(name, config, data)
        all_results[name] = result
        print(f"  Overall: {result['qa']['overall']}")
        print(f"  Answerable: {result['qa']['answerable']}")
        print(f"  Abstention: {result['qa']['abstention']}")
        print(f"  Gate coverage: {result['gate']['coverage']}")
        print(f"  Recon triggered: {result['memory']['reconsolidation_triggered']}")

    # Save summary
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")

    # Print comparison table
    print("\n=== Comparison ===")
    print(f"{'System':<15} {'Overall':>8} {'Answerable':>10} {'Abstention':>10} {'Coverage':>8} {'Recon':>6}")
    for name, r in all_results.items():
        print(f"{name:<15} {r['qa']['overall']:>8.3f} {r['qa']['answerable']:>10.3f} "
              f"{r['qa']['abstention']:>10.3f} {r['gate']['coverage']:>8.3f} "
              f"{r['memory']['reconsolidation_triggered']:>6}")


if __name__ == "__main__":
    main()
