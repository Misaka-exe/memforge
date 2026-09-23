"""Stage 4.7: LongMemEval-S lifecycle ablation.

One shared embedding pass per question, then evaluate 6 configurations
that differ in temporal filtering / conflict (supersede) filtering /
freshness reranking:

  naive_vector : pure cosine over all turns (baseline, no lifecycle)
  hybrid       : vector+keyword union, sem+kw blend (no lifecycle)
  full         : hybrid + temporal + conflict + freshness rerank
  no_temporal  : full - temporal
  no_conflict  : full - conflict
  no_freshness : full - freshness

Design notes
- Turn-level MemoryItems (same as Stage 4.6); session_id mapping preserved.
- Temporal: a turn is usable only if its session date <= question date.
- Conflict (deterministic approximation, no LLM): an earlier-session turn
  is superseded when a LATER-session turn is cosine-similar above a
  threshold; superseded turns are excluded from retrieval. This is an
  approximation of MemForge conflict resolution, not semantic slot-level
  conflict detection.
- Freshness: normalized recency of the session date within the usable set
  (0 = oldest .. 1 = newest).
- Streaming checkpoint: every question is appended to raw results
  immediately, so an interrupted run keeps already-computed questions.

Usage:
    python -m benchmarks.scripts.run_longmemeval_ablation \
        [--data PATH] [--model NAME] [--max-questions N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.longmemeval import baselines as BL
from benchmarks.longmemeval.metrics import aggregate, evaluate_question

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUT = ROOT / "results" / "longmemeval"

VARIANTS = ["naive_vector", "hybrid", "full", "no_temporal", "no_conflict", "no_freshness"]
K_LIST = (1, 5, 10)
CATEGORY_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

CANDIDATE_K = 40
CONFLICT_SIM_THRESHOLD = 0.80
W_SEM, W_KW, W_FRESH = 0.60, 0.25, 0.15
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\([^)]*\))?\s*(?:(\d{1,2}):(\d{2}))?")


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    m = DATE_RE.match(str(s).strip())
    if not m:
        return None
    return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4] or 0), int(m[5] or 0))


def get_provider(model: str | None):
    from memforge.embeddings.sentence_transformers import SentenceTransformersProvider

    return SentenceTransformersProvider(model_name=model)


def _rank_by_score(indices: list[int], scores: dict, top_k: int) -> list[int]:
    return sorted(indices, key=lambda i: scores.get(i, -1e9), reverse=True)[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t0 = time.time()
    cases = load_longmemeval(args.data)
    if args.max_questions:
        cases = cases[: args.max_questions]
    print(f"[load] {len(cases)} cases")

    provider = get_provider(args.model)
    print(f"[embed] model={provider.model_name}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "ablation_raw_results.jsonl"
    per_system: dict[str, list[dict]] = defaultdict(list)
    abs_count = 0
    done_before = 0
    diag_superseded_total = 0
    diag_superseded_gold_turns = 0
    diag_questions_with_supersede = 0

    # resume support: count questions already written
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            done_before = sum(1 for _ in f)

    with open(raw_path, "a", encoding="utf-8") as raw:
        for ci, case in enumerate(cases):
            q = case.questions[0]
            items = case.memories
            if not items:
                continue
            if q.is_abstention:
                abs_count += 1
                continue
            if ci < done_before:
                continue  # already computed in a previous run

            n = len(items)
            contents = [m.content for m in items]
            session_of = {m.id: str(m.metadata["session_id"]) for m in items}
            sdates = [parse_dt(m.metadata.get("session_date")) for m in items]
            qdate = parse_dt(q.question_date if hasattr(q, "question_date") else None)

            # embed once (shared across all variants)
            vecs = np.array(provider.embed(contents), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_n = vecs / np.clip(norms, 1e-9, None)
            qvec = np.array(provider.embed([q.question])[0], dtype=np.float32)
            qn = np.linalg.norm(qvec)
            qvec_n = qvec / max(qn, 1e-9)

            sim = (vecs_n @ qvec_n).astype(np.float64)
            kw = np.array([BL.keyword_score(q.question, c) for c in contents], dtype=np.float64)

            # temporal usable mask: session date <= question date
            usable_t = np.ones(n, dtype=bool)
            if qdate is not None:
                usable_t = np.array(
                    [False if d is None else d <= qdate for d in sdates], dtype=bool
                )
                if not usable_t.any():
                    usable_t = np.ones(n, dtype=bool)  # fallback: no usable turn

            # conflict (supersede) mask: earlier-session turn superseded by
            # later-session similar turn (different session, sim > threshold)
            superseded = np.zeros(n, dtype=bool)
            if n > 1 and n <= 4000:
                ssim = (vecs_n @ vecs_n.T).astype(np.float64)
                for i in range(n):
                    if superseded[i]:
                        continue
                    di = sdates[i]
                    for j in range(n):
                        if i == j or superseded[i]:
                            continue
                        if session_of[items[i].id] == session_of[items[j].id]:
                            continue
                        dj = sdates[j]
                        if dj is None or di is None:
                            continue
                        if dj <= di:
                            continue
                        if ssim[i, j] >= CONFLICT_SIM_THRESHOLD:
                            superseded[i] = True
                            break

            # freshness: normalized recency among usable turns
            usable_idx = [i for i in range(n) if usable_t[i]]
            sdate_vals = [(i, sdates[i]) for i in usable_idx if sdates[i] is not None]
            fresh = np.zeros(n, dtype=np.float64)
            if len(sdate_vals) >= 2:
                sdate_vals.sort(key=lambda x: x[1])
                for rank, (i, _) in enumerate(sdate_vals):
                    fresh[i] = rank / (len(sdate_vals) - 1)

            gold = q.gold_memory_ids
            gold_sess = set(gold)
            _sup_n = int(superseded.sum())
            diag_superseded_total += _sup_n
            if _sup_n:
                diag_questions_with_supersede += 1
            diag_superseded_gold_turns += sum(1 for i in range(n) if superseded[i] and session_of[items[i].id] in gold_sess)

            # ---- variant scoring ----
            def run_variant(usable: np.ndarray, exclude_superseded: bool, use_fresh: bool) -> list[int]:
                pool = [i for i in range(n) if usable[i] and (not exclude_superseded or not superseded[i])]
                if not pool:
                    pool = list(range(n))
                if len(pool) == 0:
                    return []
                if len(pool) <= max(K_LIST):
                    order = _rank_by_score(pool, {i: sim[i] for i in pool}, max(K_LIST))
                    return order
                # candidate union
                sem_top = _rank_by_score(pool, {i: sim[i] for i in pool}, CANDIDATE_K)
                kw_top = _rank_by_score(pool, {i: kw[i] for i in pool}, CANDIDATE_K)
                union = sorted(set(sem_top) | set(kw_top))
                scores = {
                    i: W_SEM * sim[i] + W_KW * kw[i] + (W_FRESH * fresh[i] if use_fresh else 0.0)
                    for i in union
                }
                return _rank_by_score(union, scores, max(K_LIST))

            all_idx = np.ones(n, dtype=bool)
            variants = {
                "naive_vector": _rank_by_score(list(range(n)), {i: sim[i] for i in range(n)}, max(K_LIST)),
                "hybrid": run_variant(all_idx, exclude_superseded=False, use_fresh=False),
                "full": run_variant(usable_t, exclude_superseded=True, use_fresh=True),
                "no_temporal": run_variant(all_idx, exclude_superseded=True, use_fresh=True),
                "no_conflict": run_variant(usable_t, exclude_superseded=False, use_fresh=True),
                "no_freshness": run_variant(usable_t, exclude_superseded=True, use_fresh=False),
            }

            for vname, order in variants.items():
                item_ids = [items[i].id for i in order]
                ev = evaluate_question(item_ids, session_of, gold, K_LIST)
                ev.update({"question_id": q.question_id, "category": q.category.value, "variant": vname})
                per_system[vname].append(ev)
                raw.write(json.dumps(ev, ensure_ascii=False) + "\n")
            raw.flush()

            if (ci + 1) % 50 == 0:
                print(f"  [{ci+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s", flush=True)

    # ---- aggregate ----
    global_metrics: dict = {"abstention_count": abs_count}
    category_rows: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for vname in VARIANTS:
        evs = per_system[vname]
        if not evs:
            continue
        agg = aggregate(evs)
        global_metrics[vname] = agg
        for cat in CATEGORY_ORDER:
            cat_evs = [e for e in evs if e["category"] == cat]
            if cat_evs:
                ca = aggregate(cat_evs)
                category_rows[cat][vname] = {
                    "recall_any@1": ca["recall_any@1"],
                    "recall_any@5": ca["recall_any@5"],
                    "recall_all@5": ca["recall_all@5"],
                    "mrr": ca["mrr"],
                    "ndcg@10": ca["ndcg@10"],
                    "n": ca["num_questions"],
                }

    import csv

    # ablation.csv: variant x global metrics
    with open(out / "ablation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "recall_any@1", "recall_any@5", "recall_all@5", "recall_any@10", "recall_all@10", "mrr", "ndcg@10"])
        for vname in VARIANTS:
            m = global_metrics.get(vname, {})
            if m:
                w.writerow([vname, m.get("num_questions", 0), m.get("recall_any@1", 0),
                            m.get("recall_any@5", 0), m.get("recall_all@5", 0),
                            m.get("recall_any@10", 0), m.get("recall_all@10", 0),
                            m.get("mrr", 0), m.get("ndcg@10", 0)])

    # by_category.csv (ablation)
    with open(out / "by_category_ablation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "variant", "n", "recall_any@1", "recall_any@5", "recall_all@5", "mrr", "ndcg@10"])
        for cat in CATEGORY_ORDER:
            for vname in VARIANTS:
                row = category_rows[cat].get(vname)
                if row:
                    w.writerow([cat, vname, row["n"], row["recall_any@1"], row["recall_any@5"],
                                row["recall_all@5"], row["mrr"], row["ndcg@10"]])

    manifest = {
        "stage": "4.7",
        "dataset": "LongMemEval-S",
        "model": provider.model_name,
        "variants": VARIANTS,
        "k_list": list(K_LIST),
        "config": {
            "candidate_k": CANDIDATE_K,
            "conflict_sim_threshold": CONFLICT_SIM_THRESHOLD,
            "weights": {"w_sem": W_SEM, "w_kw": W_KW, "w_fresh": W_FRESH},
            "temporal": "session_date <= question_date",
            "conflict": "earlier-session turn superseded if later-session turn cosine >= threshold",
        },
        "abstention_count": abs_count,
        "diagnostics": {
            "superseded_turns_total": diag_superseded_total,
            "superseded_gold_session_turns": diag_superseded_gold_turns,
            "questions_with_supersede": diag_questions_with_supersede,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "metrics": global_metrics,
        "by_category": {k: dict(v) for k, v in category_rows.items()},
    }
    with open(out / "ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # figures
    try:
        _figs(out, global_metrics, category_rows)
    except Exception as e:  # noqa: BLE001
        print("[figures] skipped:", e)

    print("\n=== Global metrics ===")
    for vname in VARIANTS:
        m = global_metrics.get(vname, {})
        if m:
            print(f"{vname:>13}: any@1={m.get('recall_any@1',0):.4f} any@5={m.get('recall_any@5',0):.4f} "
                  f"all@5={m.get('recall_all@5',0):.4f} mrr={m.get('mrr',0):.4f} ndcg@10={m.get('ndcg@10',0):.4f}")
    print("abstention_count:", abs_count)
    print("superseded_turns_total:", diag_superseded_total, "| questions_with_supersede:", diag_questions_with_supersede, "| superseded_gold_session_turns:", diag_superseded_gold_turns)
    print(f"done in {time.time()-t0:.1f}s -> {out}")


def _figs(out: Path, global_metrics: dict, category_rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    short = {"naive_vector": "naive", "hybrid": "hybrid", "full": "full",
             "no_temporal": "-temporal", "no_conflict": "-conflict", "no_freshness": "-freshness"}

    # ablation_any5.png
    xs = list(range(len(VARIANTS)))
    any5 = [global_metrics.get(v, {}).get("recall_any@5", 0) for v in VARIANTS]
    all5 = [global_metrics.get(v, {}).get("recall_all@5", 0) for v in VARIANTS]
    mrr = [global_metrics.get(v, {}).get("mrr", 0) for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, any5, marker="o", label="recall_any@5")
    ax.plot(xs, all5, marker="s", label="recall_all@5")
    ax.plot(xs, mrr, marker="^", label="MRR")
    ax.set_xticks(xs)
    ax.set_xticklabels([short[v] for v in VARIANTS], rotation=15)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("LongMemEval-S ablation (470 questions, session-level)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "ablation_any5.png", dpi=150)
    plt.close()

    # ablation_all5.png (recall_all@5 by variant, stacked bars per category group not needed)
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = [c for c in CATEGORY_ORDER if c in category_rows]
    width = 0.15
    x = np.arange(len(cats))
    for j, v in enumerate(VARIANTS):
        vals = [category_rows[c].get(v, {}).get("recall_all@5", 0) for c in cats]
        ax.bar(x + (j - 2.5) * width, vals, width, label=short[v])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall_all@5")
    ax.set_title("LongMemEval-S ablation: recall_all@5 by category")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(figdir / "category_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()