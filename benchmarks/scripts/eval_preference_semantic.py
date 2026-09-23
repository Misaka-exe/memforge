"""
Stage 6-A: Preference semantic similarity evaluation.

Uses all-MiniLM-L6-v2 to compute cosine similarity between gold_answer and
model_answer for single-session-preference questions. This is a supplementary
metric only -- it does NOT replace the original exact-match QA accuracy.
"""
import json
import os
import sys
import numpy as np

os.environ.setdefault("HF_HOME", r"D:\agentmemory\memforge\.hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer

RESULTS_DIR = r"D:\agentmemory\memforge\results\longmemeval\stage5_real"
SYSTEMS = ["naive_vector", "hybrid", "full"]
CATEGORY = "single-session-preference"

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

all_results = {}

for sys_name in SYSTEMS:
    path = os.path.join(RESULTS_DIR, sys_name, "raw_qa.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pref = [r for r in records if r.get("category") == CATEGORY]
    print(f"=== {sys_name}: {len(pref)} preference records ===")

    if not pref:
        print("  No preference records found.\n")
        continue

    golds = [r.get("gold_answer", "") for r in pref]
    models = [r.get("model_answer", "") for r in pref]

    gold_emb = model.encode(golds, normalize_embeddings=True, show_progress_bar=False)
    model_emb = model.encode(models, normalize_embeddings=True, show_progress_bar=False)

    similarities = [float(np.dot(g, m)) for g, m in zip(gold_emb, model_emb)]

    avg_sim = np.mean(similarities)
    med_sim = np.median(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    # Count how many model answers are INSUFFICIENT_INFORMATION
    insufficient = sum(1 for m in models if "INSUFFICIENT" in m.upper())
    actual_answer = len(models) - insufficient

    print(f"  Avg cosine similarity: {avg_sim:.4f}")
    print(f"  Median: {med_sim:.4f}")
    print(f"  Std: {std_sim:.4f}")
    print(f"  Min: {min_sim:.4f}, Max: {max_sim:.4f}")
    print(f"  INSUFFICIENT_INFORMATION: {insufficient}/{len(models)}")
    print(f"  Actual answers: {actual_answer}/{len(models)}")

    # Per-question detail
    details = []
    for i, (r, sim) in enumerate(zip(pref, similarities)):
        details.append({
            "question_id": r.get("question_id"),
            "gold_answer": r.get("gold_answer", "")[:100],
            "model_answer": r.get("model_answer", "")[:100],
            "semantic_similarity": round(sim, 4),
            "is_insufficient": "INSUFFICIENT" in r.get("model_answer", "").upper(),
        })

    all_results[sys_name] = {
        "category": CATEGORY,
        "n": len(pref),
        "avg_semantic_similarity": round(float(avg_sim), 4),
        "median_semantic_similarity": round(float(med_sim), 4),
        "std_semantic_similarity": round(float(std_sim), 4),
        "min_semantic_similarity": round(float(min_sim), 4),
        "max_semantic_similarity": round(float(max_sim), 4),
        "insufficient_count": insufficient,
        "actual_answer_count": actual_answer,
        "details": details,
    }
    print()

# Save results
out_path = os.path.join(RESULTS_DIR, "preference_semantic_eval.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"Results saved to: {out_path}")

# Summary comparison
print("\n=== SUMMARY: Preference Semantic Similarity ===")
print(f"{'System':<20} {'Avg Sim':>10} {'Median':>10} {'Insufficient':>12}")
print("-" * 55)
for sys_name in SYSTEMS:
    if sys_name in all_results:
        r = all_results[sys_name]
        print(f"{sys_name:<20} {r['avg_semantic_similarity']:>10.4f} {r['median_semantic_similarity']:>10.4f} {r['insufficient_count']:>12}")
