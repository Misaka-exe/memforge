"""Stage 4.6-3: LongMemEval-S pure retrieval experiment.

Three baselines, per-turn MemoryItems, session-level metrics.
Abstention questions are excluded from the retrieval denominator and
reported separately.

Usage:
    python -m benchmarks.scripts.run_longmemeval_retrieval \
        [--data path] [--embedding sentence-transformers|hash] \
        [--model NAME] [--max-questions N] [--out dir]

Outputs (results/longmemeval/):
    manifest.json              run configuration
    retrieval_results.jsonl    per-question x per-system scores
    metrics.json               System x metrics + category breakdown
    by_category.csv            Category x System (recall_any@5 / recall_all@5)
    figures/recall_by_category.png
    figures/recall_at_k.png
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.longmemeval import baselines as BL
from benchmarks.longmemeval.metrics import aggregate, evaluate_question

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUT = ROOT / "results" / "longmemeval"

SYSTEMS = ["naive_vector", "vector_rerank", "hybrid"]
K_LIST = (1, 5, 10)
CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def get_provider(kind: str, model: str | None):
    if kind == "hash":
        from memforge.embeddings.hash import HashEmbeddingProvider

        return HashEmbeddingProvider(dim=64)
    from memforge.embeddings.sentence_transformers import SentenceTransformersProvider

    return SentenceTransformersProvider(model_name=model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--embedding", default="sentence-transformers", choices=["sentence-transformers", "hash"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.data}")
    cases = load_longmemeval(args.data)
    if args.max_questions:
        cases = cases[: args.max_questions]
    print(f"[load] {len(cases)} cases")

    provider = get_provider(args.embedding, args.model)
    print(f"[embed] provider={args.embedding} model={provider.model_name if hasattr(provider,'model_name') else 'hash'}")

    rows: list[dict] = []
    abs_count = 0
    per_system: dict[str, list[dict]] = defaultdict(list)

    for ci, case in enumerate(cases):
        q = case.questions[0]
        items = case.memories
        if not items:
            continue
        contents = [m.content for m in items]
        session_of = {m.id: str(m.metadata["session_id"]) for m in items}
        fresh = [m.metadata["turn_index"] / max(1, max(m.metadata["turn_index"] for m in items) + 1) for m in items]

        if q.is_abstention:
            abs_count += 1
            # abstention: record count + whether any real gold exists (none)
            rows.append({
                "question_id": q.question_id,
                "category": "abstention",
                "system": "naive_vector",
                "excluded": True,
                "is_abstention": True,
            })
            continue

        # embed this case's items + query
        vecs = np.array(provider.embed(contents), dtype=np.float32)
        qvec = provider.embed([q.question])[0]

        gold = q.gold_memory_ids
        runs = {
            "naive_vector": BL.naive_vector(qvec, vecs.tolist(), top_k=max(K_LIST)),
            "vector_rerank": BL.vector_rerank(qvec, vecs.tolist(), fresh, top_k=max(K_LIST)),
            "hybrid": BL.hybrid(q.question, qvec, vecs.tolist(), contents, top_k=max(K_LIST)),
        }
        for sys_name, order in runs.items():
            item_ids = [items[i].id for i in order]
            ev = evaluate_question(item_ids, session_of, gold, K_LIST)
            ev.update({"question_id": q.question_id, "category": q.category.value, "system": sys_name})
            rows.append(ev)
            per_system[sys_name].append(ev)

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s")

    # ---- aggregate ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    global_metrics: dict = {"abstention_count": abs_count}
    category_rows: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for sys_name in SYSTEMS:
        evs = per_system[sys_name]
        if not evs:
            continue
        agg = aggregate(evs)
        global_metrics[sys_name] = agg
        for cat in CATEGORY_ORDER:
            cat_evs = [e for e in evs if e["category"] == cat]
            if cat_evs:
                ca = aggregate(cat_evs)
                category_rows[cat][sys_name] = {
                    "recall_any@5": ca["recall_any@5"],
                    "recall_all@5": ca["recall_all@5"],
                    "recall_any@1": ca["recall_any@1"],
                    "mrr": ca["mrr"],
                    "n": ca["num_questions"],
                }

    # retrieval_results.jsonl
    with open(out / "retrieval_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # metrics.json
    manifest = {
        "dataset": "LongMemEval-S",
        "file": args.data,
        "embedding": args.embedding,
        "model": getattr(provider, "model_name", "hash"),
        "systems": SYSTEMS,
        "k_list": list(K_LIST),
        "abstention_excluded_from_retrieval": True,
        "abstention_count": abs_count,
        "elapsed_seconds": round(time.time() - t0, 1),
        "metrics": global_metrics,
        "by_category": {k: dict(v) for k, v in category_rows.items()},
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # by_category.csv
    import csv

    with open(out / "by_category.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "system", "n", "recall_any@1", "recall_any@5", "recall_all@5", "mrr"])
        for cat in CATEGORY_ORDER:
            for sys_name in SYSTEMS:
                row = category_rows[cat].get(sys_name)
                if row:
                    w.writerow([cat, sys_name, row["n"], row["recall_any@1"], row["recall_any@5"], row["recall_all@5"], row["mrr"]])

    # figures (regenerate from manifest, idempotent)
    try:
        _figs(out)
    except Exception as e:  # noqa: BLE001
        print("[figures] skipped:", e)

    print("\n=== Global metrics ===")
    for sys_name in SYSTEMS:
        m = global_metrics.get(sys_name, {})
        print(f"{sys_name:>14}: " + " | ".join(f"{k}={v}" for k, v in m.items() if k in ("recall_any@1", "recall_any@5", "recall_any@10", "recall_all@5", "recall_all@10", "mrr", "ndcg@10")))
    print("abstention_count:", abs_count)
    print(f"done in {time.time()-t0:.1f}s -> {out}")


def _figs(out: Path) -> None:
    """Regenerate figures from manifest.json (idempotent)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.load(open(out / "manifest.json", encoding="utf-8"))
    global_metrics = manifest["metrics"]
    category_rows = manifest["by_category"]

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # recall_at_k.png
    ks = list(K_LIST)
    x = np.arange(len(ks))
    for sys_name in SYSTEMS:
        m = global_metrics.get(sys_name, {})
        vals = [m.get(f"recall_any@{k}", 0) for k in ks]
        plt.plot(x, vals, marker="o", label=sys_name)
    plt.xticks(x, [f"R@any {k}" for k in ks])
    plt.ylim(0, 1.0)
    plt.title("LongMemEval-S retrieval: recall_any@K (session-level)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "recall_at_k.png", dpi=150)
    plt.close()

    # recall_by_category.png
    cats = [c for c in CATEGORY_ORDER if c in category_rows]
    x = np.arange(len(cats))
    width = 0.25
    for j, sys_name in enumerate(SYSTEMS):
        vals = [category_rows[c].get(sys_name, {}).get("recall_any@5", 0) for c in cats]
        plt.bar(x + j * width, vals, width, label=sys_name)
    plt.xticks(x + width, cats, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("recall_any@5")
    plt.title("LongMemEval-S retrieval by category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "recall_by_category.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()