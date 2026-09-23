"""One-command benchmark runner.

Usage:
    python -m benchmarks.runner

Outputs CSVs and figures under results/. No PostgreSQL, no API key,
no network; fixed seed makes everything reproducible.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from benchmarks.baselines.strategies import (
    FullContextBaseline,
    NaiveVectorBaseline,
    VectorRerankBaseline,
    evaluate_strategy,
)
from benchmarks.synthetic.generator import ScenarioGenerator

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"


def _ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_baselines(generator: ScenarioGenerator, num_users: int = 8) -> list[dict]:
    scenarios = generator.generate(num_users=num_users)
    strategies = [FullContextBaseline(), NaiveVectorBaseline(), VectorRerankBaseline()]
    out = []
    for strat in strategies:
        agg = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg": 0.0}
        n = len(scenarios)
        for scn in scenarios:
            r = evaluate_strategy(strat, scn)
            for k in agg:
                agg[k] += r[k] / n
        out.append({"system": strat.name, **{k: round(v, 4) for k, v in agg.items()}})
    return out


def run_ablation(generator: ScenarioGenerator, num_users: int = 8) -> list[dict]:
    scenarios = generator.generate(num_users=num_users)
    base = VectorRerankBaseline()
    variants = [
        ("full", {"relevance": 0.4, "confidence": 0.2, "importance": 0.2, "freshness": 0.2}),
        ("-utility", {"relevance": 1.0, "confidence": 0.0, "importance": 0.0, "freshness": 0.0}),
        ("-confidence", {"relevance": 0.4, "confidence": 0.0, "importance": 0.3, "freshness": 0.3}),
        ("-importance", {"relevance": 0.4, "confidence": 0.3, "importance": 0.0, "freshness": 0.3}),
        ("-freshness", {"relevance": 0.4, "confidence": 0.2, "importance": 0.2, "freshness": 0.0}),
    ]
    out = []
    for name, w in variants:
        strat = _WeightedRerank(w)
        agg = {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}
        n = len(scenarios)
        for scn in scenarios:
            r = evaluate_strategy(strat, scn)
            agg["recall@1"] += r["recall@1"] / n
            agg["recall@5"] += r["recall@5"] / n
            agg["mrr"] += r["mrr"] / n
        out.append({"variant": name, **{k: round(v, 4) for k, v in agg.items()}})
    return out


class _WeightedRerank(VectorRerankBaseline):
    def __init__(self, weights: dict) -> None:
        self.weights = weights
        self.name = "weighted"

    def retrieve(self, scenario, query, top_k=5):
        from benchmarks.synthetic.generator import hash_embedding

        qemb = hash_embedding(query.query)
        candidates = [m for m in scenario.memories if m.status == "active"]
        scored = []
        for m in candidates:
            relevance = _cos(qemb, hash_embedding(m.content))
            freshness = 0.9
            utility = (
                self.weights["relevance"] * relevance
                + self.weights["confidence"] * m.confidence
                + self.weights["importance"] * m.importance
                + self.weights["freshness"] * freshness
            )
            scored.append((utility, m.id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mid for _, mid in scored[:top_k]]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def run_noise(generator: ScenarioGenerator, num_users: int = 8) -> list[dict]:
    rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    out = []
    for rate in rates:
        scenarios = generator.generate(num_users=num_users, noise_rate=rate)
        for strat in [NaiveVectorBaseline(), VectorRerankBaseline()]:
            r5, nd = [], []
            for scn in scenarios:
                r = evaluate_strategy(strat, scn)
                r5.append(r["recall@5"])
                nd.append(r["ndcg"])
            out.append({
                "noise_rate": rate,
                "system": strat.name,
                "recall@5": round(sum(r5) / len(r5), 4),
                "ndcg": round(sum(nd) / len(nd), 4),
            })
    return out


def run_memory_budget(generator: ScenarioGenerator, num_users: int = 8) -> list[dict]:
    budgets = [5, 10, 20, 40]
    scenarios = generator.generate(num_users=num_users)
    out = []
    for budget in budgets:
        r5s, mrrs = [], []
        for scn in scenarios:
            limited = scn.model_copy(deep=True)
            limited.memories = limited.memories[:budget]
            r = evaluate_strategy(VectorRerankBaseline(), limited)
            r5s.append(r["recall@5"])
            mrrs.append(r["mrr"])
        out.append({
            "memory_budget": budget,
            "recall@5": round(sum(r5s) / len(r5s), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
        })
    return out


def make_figures(baseline: list[dict], noise: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping figures")
        return

    # baseline comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["system"] for r in baseline]
    r1 = [r["recall@1"] for r in baseline]
    r5 = [r["recall@5"] for r in baseline]
    mrr = [r["mrr"] for r in baseline]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], r1, width=0.2, label="Recall@1")
    ax.bar([i for i in x], r5, width=0.2, label="Recall@5")
    ax.bar([i + 0.2 for i in x], mrr, width=0.2, label="MRR")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Baseline comparison (synthetic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_comparison.png", dpi=120)
    plt.close(fig)

    # noise curve
    fig, ax = plt.subplots(figsize=(7, 4))
    for system in {r["system"] for r in noise}:
        rows = [r for r in noise if r["system"] == system]
        rows.sort(key=lambda r: r["noise_rate"])
        ax.plot([r["noise_rate"] for r in rows], [r["recall@5"] for r in rows], marker="o", label=system)
    ax.set_xlabel("noise rate")
    ax.set_ylabel("Recall@5")
    ax.set_title("Accuracy vs memory noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_noise.png", dpi=120)
    plt.close(fig)


def main() -> None:
    _ensure_dirs()
    generator = ScenarioGenerator(seed=42)

    print("Running baselines...")
    baseline = run_baselines(generator)
    _write_csv(RESULTS / "baseline.csv", ["system", "recall@1", "recall@5", "recall@10", "mrr", "ndcg"],
               [[r["system"], r["recall@1"], r["recall@5"], r["recall@10"], r["mrr"], r["ndcg"]] for r in baseline])
    for r in baseline:
        print(f"  {r['system']:16s} R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f}")

    print("Running ablation...")
    ablation = run_ablation(generator)
    _write_csv(RESULTS / "ablation.csv", ["variant", "recall@1", "recall@5", "mrr"],
               [[r["variant"], r["recall@1"], r["recall@5"], r["mrr"]] for r in ablation])
    for r in ablation:
        print(f"  {r['variant']:12s} R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f}")

    print("Running noise injection...")
    noise = run_noise(generator)
    _write_csv(RESULTS / "noise.csv", ["noise_rate", "system", "recall@5", "ndcg"],
               [[r["noise_rate"], r["system"], r["recall@5"], r["ndcg"]] for r in noise])

    print("Running memory budget...")
    budget = run_memory_budget(generator)
    _write_csv(RESULTS / "memory_budget.csv", ["memory_budget", "recall@5", "mrr"],
               [[r["memory_budget"], r["recall@5"], r["mrr"]] for r in budget])

    make_figures(baseline, noise)
    print(f"Done. Results in {RESULTS}")


if __name__ == "__main__":
    main()