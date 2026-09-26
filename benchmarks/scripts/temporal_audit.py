"""Temporal Stress Benchmark audit (V2.1 Research Audit).

Fixes FLR definition: future memory that is the GOLD answer (T3 Future category)
should NOT count as leakage. FLR should only measure future contamination on
questions asking about current/historical state.

Also performs Pareto analysis: sweep temporal decay floor parameter to find
the evidence-loss / future-leakage trade-off frontier.

Output: results/v21/temporal/stress_audit.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from benchmarks.stress.temporal.benchmark import (
    TemporalCategory,
    TemporalMemory,
    evaluate_temporal_system,
    generate_temporal_cases,
    hard_temporal_filter,
    no_temporal_filter,
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "v21" / "temporal" / "stress_audit.json"


def make_soft_decay(floor: float = 0.1):
    """Create a soft temporal filter with configurable floor."""
    def soft_filter(memories, query_dt):
        def score(m):
            valid_from = datetime.fromisoformat(m.valid_from)
            valid_until = datetime.fromisoformat(m.valid_until) if m.valid_until else None
            if valid_from <= query_dt and (valid_until is None or query_dt < valid_until):
                return 1.0
            if valid_from > query_dt:
                return floor
            return floor * 0.5
        scored = [(score(m), m) for m in memories]
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]
    return soft_filter


def compute_corrected_flr(cases, filter_fn, top_k=5):
    """Compute FLR only on current-question categories (exclude T3 Future).

    T3 Future: future memory is the gold answer, not leakage.
    Current-question categories: current, historical, update, contradictory, future_contamination.
    """
    current_categories = {
        TemporalCategory.CURRENT,
        TemporalCategory.HISTORICAL,
        TemporalCategory.UPDATE,
        TemporalCategory.CONTRADICTORY,
        TemporalCategory.FUTURE_CONTAMINATION,
    }

    total_flr = 0
    n_current = 0
    by_cat = {}

    for case in cases:
        if case.category not in current_categories:
            continue
        query_dt = datetime.fromisoformat(case.query_time)
        filtered = filter_fn(case.memories, query_dt)
        top_k_items = filtered[:top_k]
        future_in_topk = sum(1 for m in top_k_items if m.is_future)
        flr = future_in_topk / max(len(top_k_items), 1)
        total_flr += flr
        n_current += 1

        cat = case.category.value
        if cat not in by_cat:
            by_cat[cat] = {"n": 0, "flr_sum": 0}
        by_cat[cat]["n"] += 1
        by_cat[cat]["flr_sum"] += flr

    for cat in by_cat:
        by_cat[cat]["flr"] = by_cat[cat]["flr_sum"] / by_cat[cat]["n"]

    return {
        "n_current_questions": n_current,
        "corrected_flr": total_flr / n_current if n_current else 0,
        "by_category": {k: {"n": v["n"], "flr": round(v["flr"], 4)} for k, v in by_cat.items()},
        "note": "FLR computed only on current-question categories (excludes T3 Future where future memory is gold)",
    }


def pareto_analysis(cases):
    """Sweep soft decay floor to find ELR vs FLR trade-off."""
    points = []
    for floor in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        filter_fn = make_soft_decay(floor)
        metrics = evaluate_temporal_system(cases, filter_fn)
        corrected = compute_corrected_flr(cases, filter_fn)
        points.append({
            "floor": floor,
            "recall@1": round(metrics["recall@1"], 4),
            "recall@5": round(metrics["recall@k"], 4),
            "elr": round(metrics["evidence_loss_rate"], 4),
            "flr_original": round(metrics["future_leakage_rate"], 4),
            "flr_corrected": round(corrected["corrected_flr"], 4),
        })
    return points


def main():
    cases = generate_temporal_cases(n_per_category=20)
    print(f"Generated {len(cases)} cases (6 categories x 20)")

    results = {
        "n_total": len(cases),
        "n_per_category": 20,
        "flr_definition_fix": {
            "original_flr_includes_t3_future_gold": True,
            "corrected_flr_excludes_t3_future": True,
            "t3_future_note": "In T3 Future category, the future memory IS the gold answer. Counting it as leakage is incorrect.",
        },
    }

    # Original vs corrected FLR for each system
    print("\n=== Original vs Corrected FLR ===")
    for name, fn in [
        ("hard_filter", hard_temporal_filter),
        ("no_temporal", no_temporal_filter),
        ("soft_decay_floor0.1", make_soft_decay(0.1)),
    ]:
        metrics = evaluate_temporal_system(cases, fn)
        corrected = compute_corrected_flr(cases, fn)
        print(f"\n{name}:")
        print(f"  Original FLR: {metrics['future_leakage_rate']:.3f}")
        print(f"  Corrected FLR: {corrected['corrected_flr']:.3f} (n={corrected['n_current_questions']} current questions)")
        print(f"  ELR: {metrics['evidence_loss_rate']:.3f}")
        print(f"  R@1: {metrics['recall@1']:.3f}, R@5: {metrics['recall@k']:.3f}")
        results[name] = {
            "original_flr": round(metrics["future_leakage_rate"], 4),
            "corrected_flr": corrected,
            "elr": round(metrics["evidence_loss_rate"], 4),
            "recall@1": round(metrics["recall@1"], 4),
            "recall@5": round(metrics["recall@k"], 4),
        }

    # Pareto analysis
    print("\n=== Pareto Analysis (soft decay floor sweep) ===")
    print(f"{'Floor':>6} {'R@1':>6} {'R@5':>6} {'ELR':>6} {'FLR_orig':>9} {'FLR_corr':>9}")
    pareto = pareto_analysis(cases)
    for p in pareto:
        print(f"{p['floor']:>6.2f} {p['recall@1']:>6.3f} {p['recall@5']:>6.3f} "
              f"{p['elr']:>6.3f} {p['flr_original']:>9.3f} {p['flr_corrected']:>9.3f}")
    results["pareto_sweep"] = pareto

    # Pareto-efficient points (minimize ELR and FLR_corrected, maximize R@1)
    # A point is Pareto-efficient if no other point has both lower ELR and lower FLR and higher R@1
    efficient = []
    for i, p in enumerate(pareto):
        dominated = False
        for j, q in enumerate(pareto):
            if i == j:
                continue
            if (q["elr"] <= p["elr"] and q["flr_corrected"] <= p["flr_corrected"]
                    and q["recall@1"] >= p["recall@1"]
                    and (q["elr"] < p["elr"] or q["flr_corrected"] < p["flr_corrected"] or q["recall@1"] > p["recall@1"])):
                dominated = True
                break
        if not dominated:
            efficient.append(p)
    results["pareto_efficient"] = efficient
    print(f"\nPareto-efficient points: {len(efficient)}")
    for p in efficient:
        print(f"  floor={p['floor']:.2f}: ELR={p['elr']:.3f}, FLR_corr={p['flr_corrected']:.3f}, R@1={p['recall@1']:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
