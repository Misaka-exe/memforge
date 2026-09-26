"""Export retrieval features for all 500 LongMemEval-S questions.

Computes real semantic scores from embedding cache (no re-embedding).
Output: results/longmemeval/stage_v21/retrieval_features.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.adapters.longmemeval import load_longmemeval
from memforge.retrieval.score import compute_retrieval_features

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "results" / "longmemeval" / "stage5_real" / "_cache" / "embeddings"
OUT_DIR = ROOT / "results" / "longmemeval" / "stage_v21"
DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "retrieval_features.jsonl"

    cases = load_longmemeval(str(DATA))
    question_ids = []
    for case in cases:
        for q in case.questions:
            question_ids.append(q.question_id)

    print(f"Computing retrieval features for {len(question_ids)} questions...")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, qid in enumerate(question_ids):
            features = compute_retrieval_features(qid, str(CACHE_DIR))
            f.write(json.dumps(features.to_dict(), ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(question_ids)}")

    # Summary stats
    all_features = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        all_features.append(json.loads(line))

    top1 = [f["top1_score"] for f in all_features]
    top5 = [f["top5_mean"] for f in all_features]
    print(f"\n=== Retrieval Features Summary ===")
    print(f"n={len(all_features)}")
    print(f"top1_score: mean={sum(top1)/len(top1):.4f}, std={__import__('numpy').std(top1):.4f}")
    print(f"top5_mean: mean={sum(top5)/len(top5):.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
