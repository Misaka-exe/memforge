"""Gate Calibration Check: analyze retrieval signal distributions.

Uses existing 500-question results + retrieval inputs + embedding cache.
No new LLM API calls.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_V2 = ROOT / "results" / "longmemeval" / "stage5_v2" / "raw_qa.jsonl"
RETRIEVED_DIR = ROOT / "results" / "longmemeval" / "stage5_real" / "retrieved" / "hybrid"
EMB_CACHE_DIR = ROOT / "results" / "longmemeval" / "stage5_real" / "_cache" / "embeddings"
OUT = ROOT / "results" / "longmemeval" / "stage5_v2" / "gate_analysis.json"


def load_raw_results():
    records = []
    for line in RAW_V2.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def get_retrieval_count(qid: str) -> int:
    path = RETRIEVED_DIR / f"{qid}.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("ids", []))


def compute_embedding_similarity(qid: str):
    """Compute top-1 cosine similarity from embedding cache if available."""
    # Try question embedding cache
    q_path = EMB_CACHE_DIR / f"{qid}.npz"
    if not q_path.exists():
        return None
    try:
        q_data = np.load(q_path)
        q_emb = q_data["query_embedding"] if "query_embedding" in q_data else None
        if q_emb is None:
            # Try other keys
            for key in q_data.files:
                arr = q_data[key]
                if arr.ndim == 1 and arr.shape[0] > 10:
                    q_emb = arr
                    break
        if q_emb is None:
            return None
        # Normalize
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        # Get candidate embeddings from same file
        cand_embs = None
        for key in q_data.files:
            arr = q_data[key]
            if arr.ndim == 2 and arr.shape[0] > 1:
                cand_embs = arr
                break
        if cand_embs is None:
            return None
        # Compute cosine similarities
        cand_norms = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
        sims = cand_norms @ q_norm
        return {
            "top1": float(np.max(sims)),
            "top5_mean": float(np.mean(np.sort(sims)[-5:])) if len(sims) >= 5 else float(np.mean(sims)),
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "n_candidates": int(len(sims)),
            "margin": float(np.sort(sims)[-1] - np.sort(sims)[-2]) if len(sims) >= 2 else 0.0,
        }
    except Exception:
        return None


def main():
    records = load_raw_results()
    print(f"Loaded {len(records)} records")

    answerable = [r for r in records if not r["is_abstention"]]
    abstention = [r for r in records if r["is_abstention"]]
    print(f"Answerable: {len(answerable)}, Abstention: {len(abstention)}")

    # Gate decision distribution
    gate_dist = defaultdict(int)
    for r in records:
        gate_dist[r["gate_decision"]] += 1
    print(f"\nGate decisions: {dict(gate_dist)}")

    # Retrieval count analysis
    ans_counts = [get_retrieval_count(r["question_id"]) for r in answerable]
    abs_counts = [get_retrieval_count(r["question_id"]) for r in abstention]
    print(f"\nRetrieval count (answerable): mean={np.mean(ans_counts):.1f}, std={np.std(ans_counts):.1f}, min={min(ans_counts)}, max={max(ans_counts)}")
    print(f"Retrieval count (abstention): mean={np.mean(abs_counts):.1f}, std={np.std(abs_counts):.1f}, min={min(abs_counts)}, max={max(abs_counts)}")

    # Embedding similarity analysis (sample)
    print("\nComputing embedding similarities (this may take a moment)...")
    ans_sims = []
    abs_sims = []
    for r in answerable[:200]:  # sample 200 for speed
        sim = compute_embedding_similarity(r["question_id"])
        if sim:
            ans_sims.append(sim)
    for r in abstention:
        sim = compute_embedding_similarity(r["question_id"])
        if sim:
            abs_sims.append(sim)

    print(f"\nEmbedding similarity (answerable, n={len(ans_sims)}):")
    if ans_sims:
        print(f"  top1: mean={np.mean([s['top1'] for s in ans_sims]):.4f}, std={np.std([s['top1'] for s in ans_sims]):.4f}")
        print(f"  top5_mean: mean={np.mean([s['top5_mean'] for s in ans_sims]):.4f}")
        print(f"  margin: mean={np.mean([s['margin'] for s in ans_sims]):.4f}")

    print(f"\nEmbedding similarity (abstention, n={len(abs_sims)}):")
    if abs_sims:
        print(f"  top1: mean={np.mean([s['top1'] for s in abs_sims]):.4f}, std={np.std([s['top1'] for s in abs_sims]):.4f}")
        print(f"  top5_mean: mean={np.mean([s['top5_mean'] for s in abs_sims]):.4f}")
        print(f"  margin: mean={np.mean([s['margin'] for s in abs_sims]):.4f}")

    # Verification results
    ans_verif = defaultdict(int)
    abs_verif = defaultdict(int)
    for r in answerable:
        ans_verif[r.get("verification_verdict", "NONE")] += 1
    for r in abstention:
        abs_verif[r.get("verification_verdict", "NONE")] += 1
    print(f"\nVerification (answerable): {dict(ans_verif)}")
    print(f"Verification (abstention): {dict(abs_verif)}")

    # Abstention behavior: for abstention questions, did model abstain?
    abs_model_abstained = sum(1 for r in abstention if r["abstained"])
    abs_model_answered = sum(1 for r in abstention if not r["abstained"])
    print(f"\nAbstention questions: model abstained={abs_model_abstained}, model answered={abs_model_answered}")

    # For answerable questions, did model incorrectly abstain?
    ans_model_abstained = sum(1 for r in answerable if r["abstained"])
    print(f"Answerable questions: model incorrectly abstained={ans_model_abstained}/{len(answerable)}")

    # Save analysis
    analysis = {
        "n_total": len(records),
        "n_answerable": len(answerable),
        "n_abstention": len(abstention),
        "gate_distribution": dict(gate_dist),
        "retrieval_count": {
            "answerable": {"mean": float(np.mean(ans_counts)), "std": float(np.std(ans_counts)), "min": int(min(ans_counts)), "max": int(max(ans_counts))},
            "abstention": {"mean": float(np.mean(abs_counts)), "std": float(np.std(abs_counts)), "min": int(min(abs_counts)), "max": int(max(abs_counts))},
        },
        "embedding_similarity": {
            "answerable_n": len(ans_sims),
            "answerable_top1_mean": float(np.mean([s["top1"] for s in ans_sims])) if ans_sims else None,
            "answerable_top1_std": float(np.std([s["top1"] for s in ans_sims])) if ans_sims else None,
            "abstention_n": len(abs_sims),
            "abstention_top1_mean": float(np.mean([s["top1"] for s in abs_sims])) if abs_sims else None,
            "abstention_top1_std": float(np.std([s["top1"] for s in abs_sims])) if abs_sims else None,
        },
        "verification": {
            "answerable": dict(ans_verif),
            "abstention": dict(abs_verif),
        },
        "abstention_behavior": {
            "abstention_questions_model_abstained": abs_model_abstained,
            "abstention_questions_model_answered": abs_model_answered,
            "answerable_questions_model_incorrectly_abstained": ans_model_abstained,
        },
        "conclusion": "",
    }

    # Generate conclusion
    if ans_sims and abs_sims:
        ans_top1 = np.mean([s["top1"] for s in ans_sims])
        abs_top1 = np.mean([s["top1"] for s in abs_sims])
        diff = ans_top1 - abs_top1
        if abs(diff) < 0.02:
            analysis["conclusion"] = (
                f"Retrieval top-1 similarity is nearly identical between answerable ({ans_top1:.4f}) "
                f"and abstention ({abs_top1:.4f}) questions (diff={diff:.4f}). "
                "This means the current retrieval signal CANNOT distinguish 'evidence sufficient' from 'evidence insufficient'. "
                "Gate signal is insufficient for calibration. The gate currently uses memory.importance (default 0.8) as a proxy, "
                "not real retrieval relevance, which explains 100% SUFFICIENT rate."
            )
        else:
            analysis["conclusion"] = (
                f"Retrieval top-1 similarity differs: answerable={ans_top1:.4f}, abstention={abs_top1:.4f} (diff={diff:.4f}). "
                "Signal may be usable for gate calibration, but requires a held-out validation set."
            )
    else:
        analysis["conclusion"] = "Embedding similarity computation failed; cannot assess gate signal separability."

    OUT.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Analysis saved to {OUT} ===")
    print(f"\nConclusion: {analysis['conclusion']}")


if __name__ == "__main__":
    main()
