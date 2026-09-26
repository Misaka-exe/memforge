"""Temporal Stress Benchmark for V2.1 Phase 3.

Synthetic benchmark specifically designed to test temporal memory handling.
Addresses V2's limitation: LongMemEval-S has no future-dated gold evidence,
so temporal soft decay was never properly validated.

6 test categories:
- T1 Current: current fact at query time
- T2 Historical: what was true before a known update
- T3 Future: what will happen (future fact)
- T4 Update: A -> B transition, ask current state
- T5 Contradictory: A at t1, B at t2, ask about specific time
- T6 Future Contamination: future fact present, ask current question

Metrics:
- Recall@K, MRR
- Future Leakage Rate (FLR): future evidence in top-K / total retrieved
- Evidence Loss Rate (ELR): gold evidence removed by temporal filtering
- Temporal Accuracy: correct answer given temporal context
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


class TemporalCategory(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    FUTURE = "future"
    UPDATE = "update"
    CONTRADICTORY = "contradictory"
    FUTURE_CONTAMINATION = "future_contamination"


@dataclass
class TemporalMemory:
    memory_id: str
    content: str
    valid_from: str  # ISO date
    valid_until: str | None = None  # None = still valid
    is_gold: bool = False
    is_future: bool = False


@dataclass
class TemporalCase:
    case_id: str
    category: TemporalCategory
    question: str
    query_time: str  # ISO date
    gold_answer: str
    memories: list[TemporalMemory]
    gold_memory_ids: list[str] = field(default_factory=list)


def generate_temporal_cases(n_per_category: int = 20) -> list[TemporalCase]:
    """Generate synthetic temporal stress test cases."""
    cases = []
    base = datetime(2024, 1, 1)

    for i in range(n_per_category):
        idx = i + 1

        # T1: Current fact
        cases.append(TemporalCase(
            case_id=f"T1_{idx:03d}",
            category=TemporalCategory.CURRENT,
            question=f"What is user {idx}'s current city?",
            query_time=(base + timedelta(days=300)).isoformat(),
            gold_answer=f"City_{idx}_current",
            memories=[
                TemporalMemory(f"T1_{idx}_old", f"User {idx} lived in City_{idx}_old.",
                              (base + timedelta(days=0)).isoformat(),
                              (base + timedelta(days=100)).isoformat()),
                TemporalMemory(f"T1_{idx}_current", f"User {idx} lives in City_{idx}_current.",
                              (base + timedelta(days=100)).isoformat(), None, is_gold=True),
                TemporalMemory(f"T1_{idx}_future", f"User {idx} will move to City_{idx}_future.",
                              (base + timedelta(days=400)).isoformat(), None, is_future=True),
            ],
            gold_memory_ids=[f"T1_{idx}_current"],
        ))

        # T2: Historical fact
        cases.append(TemporalCase(
            case_id=f"T2_{idx:03d}",
            category=TemporalCategory.HISTORICAL,
            question=f"Where did user {idx} live before day 100?",
            query_time=(base + timedelta(days=50)).isoformat(),
            gold_answer=f"City_{idx}_old",
            memories=[
                TemporalMemory(f"T2_{idx}_old", f"User {idx} lived in City_{idx}_old.",
                              (base + timedelta(days=0)).isoformat(),
                              (base + timedelta(days=100)).isoformat(), is_gold=True),
                TemporalMemory(f"T2_{idx}_current", f"User {idx} lives in City_{idx}_current.",
                              (base + timedelta(days=100)).isoformat(), None),
            ],
            gold_memory_ids=[f"T2_{idx}_old"],
        ))

        # T3: Future fact
        cases.append(TemporalCase(
            case_id=f"T3_{idx:03d}",
            category=TemporalCategory.FUTURE,
            question=f"Where will user {idx} move to?",
            query_time=(base + timedelta(days=300)).isoformat(),
            gold_answer=f"City_{idx}_future",
            memories=[
                TemporalMemory(f"T3_{idx}_current", f"User {idx} lives in City_{idx}_current.",
                              (base + timedelta(days=100)).isoformat(), None),
                TemporalMemory(f"T3_{idx}_future", f"User {idx} will move to City_{idx}_future.",
                              (base + timedelta(days=400)).isoformat(), None, is_gold=True, is_future=True),
            ],
            gold_memory_ids=[f"T3_{idx}_future"],
        ))

        # T4: Update (A -> B)
        cases.append(TemporalCase(
            case_id=f"T4_{idx:03d}",
            category=TemporalCategory.UPDATE,
            question=f"After the update, where does user {idx} live?",
            query_time=(base + timedelta(days=200)).isoformat(),
            gold_answer=f"City_{idx}_new",
            memories=[
                TemporalMemory(f"T4_{idx}_old", f"User {idx} lived in City_{idx}_old.",
                              (base + timedelta(days=0)).isoformat(),
                              (base + timedelta(days=150)).isoformat()),
                TemporalMemory(f"T4_{idx}_new", f"User {idx} moved to City_{idx}_new.",
                              (base + timedelta(days=150)).isoformat(), None, is_gold=True),
            ],
            gold_memory_ids=[f"T4_{idx}_new"],
        ))

        # T5: Contradictory temporal
        cases.append(TemporalCase(
            case_id=f"T5_{idx:03d}",
            category=TemporalCategory.CONTRADICTORY,
            question=f"At day 80, where did user {idx} live?",
            query_time=(base + timedelta(days=80)).isoformat(),
            gold_answer=f"City_{idx}_A",
            memories=[
                TemporalMemory(f"T5_{idx}_A", f"User {idx} lived in City_{idx}_A.",
                              (base + timedelta(days=0)).isoformat(),
                              (base + timedelta(days=100)).isoformat(), is_gold=True),
                TemporalMemory(f"T5_{idx}_B", f"User {idx} lived in City_{idx}_B.",
                              (base + timedelta(days=100)).isoformat(),
                              (base + timedelta(days=200)).isoformat()),
            ],
            gold_memory_ids=[f"T5_{idx}_A"],
        ))

        # T6: Future contamination
        cases.append(TemporalCase(
            case_id=f"T6_{idx:03d}",
            category=TemporalCategory.FUTURE_CONTAMINATION,
            question=f"What is user {idx}'s current job?",
            query_time=(base + timedelta(days=300)).isoformat(),
            gold_answer=f"Job_{idx}_current",
            memories=[
                TemporalMemory(f"T6_{idx}_current", f"User {idx} works as Job_{idx}_current.",
                              (base + timedelta(days=100)).isoformat(), None, is_gold=True),
                TemporalMemory(f"T6_{idx}_future_job", f"User {idx} will become Job_{idx}_future.",
                              (base + timedelta(days=500)).isoformat(), None, is_future=True),
                TemporalMemory(f"T6_{idx}_future_city", f"User {idx} will move to City_{idx}_future.",
                              (base + timedelta(days=400)).isoformat(), None, is_future=True),
            ],
            gold_memory_ids=[f"T6_{idx}_current"],
        ))

    return cases


def evaluate_temporal_system(
    cases: list[TemporalCase],
    filter_fn,  # filter_fn(memories, query_time) -> filtered_memories
    top_k: int = 5,
) -> dict:
    """Evaluate a temporal filtering strategy.

    filter_fn: takes list[TemporalMemory] and query_time, returns filtered list
    """
    total = len(cases)
    recall_at_1 = 0
    recall_at_k = 0
    future_leakage = 0
    evidence_loss = 0
    gold_total = 0

    by_category: dict[str, dict] = {}

    for case in cases:
        query_dt = datetime.fromisoformat(case.query_time)
        filtered = filter_fn(case.memories, query_dt)

        # Top-K (simplified: use order as given, since synthetic)
        top_k_items = filtered[:top_k]
        top_k_ids = {m.memory_id for m in top_k_items}
        top_1_id = filtered[0].memory_id if filtered else None

        gold_ids = set(case.gold_memory_ids)
        gold_total += len(gold_ids)

        # Recall
        if top_1_id in gold_ids:
            recall_at_1 += 1
        if top_k_ids & gold_ids:
            recall_at_k += 1

        # Future Leakage: future memories in top-K
        future_in_topk = sum(1 for m in top_k_items if m.is_future)
        future_leakage += future_in_topk / max(len(top_k_items), 1)

        # Evidence Loss: gold memories removed by filtering
        gold_in_filtered = sum(1 for m in filtered if m.memory_id in gold_ids)
        evidence_loss += len(gold_ids) - gold_in_filtered

        # By category
        cat = case.category.value
        if cat not in by_category:
            by_category[cat] = {"n": 0, "recall@1": 0, "recall@k": 0, "flr": 0, "elr": 0}
        by_category[cat]["n"] += 1
        by_category[cat]["recall@1"] += 1 if top_1_id in gold_ids else 0
        by_category[cat]["recall@k"] += 1 if top_k_ids & gold_ids else 0
        by_category[cat]["flr"] += future_in_topk / max(len(top_k_items), 1)
        by_category[cat]["elr"] += len(gold_ids) - gold_in_filtered

    # Normalize by category
    for cat in by_category:
        n = by_category[cat]["n"]
        by_category[cat]["recall@1"] /= n
        by_category[cat]["recall@k"] /= n
        by_category[cat]["flr"] /= n
        by_category[cat]["elr"] /= n

    return {
        "n_cases": total,
        "recall@1": recall_at_1 / total,
        "recall@k": recall_at_k / total,
        "future_leakage_rate": future_leakage / total,
        "evidence_loss_rate": evidence_loss / gold_total if gold_total else 0,
        "by_category": by_category,
    }


def hard_temporal_filter(memories: list[TemporalMemory], query_dt: datetime) -> list[TemporalMemory]:
    """V1-style hard filter: only memories active at query time."""
    result = []
    for m in memories:
        valid_from = datetime.fromisoformat(m.valid_from)
        valid_until = datetime.fromisoformat(m.valid_until) if m.valid_until else None
        if valid_from <= query_dt and (valid_until is None or query_dt < valid_until):
            result.append(m)
    return result


def no_temporal_filter(memories: list[TemporalMemory], query_dt: datetime) -> list[TemporalMemory]:
    """No temporal filtering: all memories."""
    return list(memories)


def soft_temporal_filter(memories: list[TemporalMemory], query_dt: datetime, floor: float = 0.1) -> list[TemporalMemory]:
    """Soft temporal: all memories retained, but future/out-of-window sorted lower.

    For this benchmark, soft = retain all but reorder (future last).
    """
    def score(m: TemporalMemory) -> float:
        valid_from = datetime.fromisoformat(m.valid_from)
        valid_until = datetime.fromisoformat(m.valid_until) if m.valid_until else None
        if valid_from <= query_dt and (valid_until is None or query_dt < valid_until):
            return 1.0  # active
        if valid_from > query_dt:
            return floor  # future
        return floor * 0.5  # expired

    scored = [(score(m), m) for m in memories]
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def run_temporal_benchmark(n_per_category: int = 20, output_path: str | Path | None = None) -> dict:
    """Run full temporal stress benchmark."""
    cases = generate_temporal_cases(n_per_category)

    results = {
        "n_per_category": n_per_category,
        "n_total": len(cases),
        "systems": {},
    }

    for name, fn in [
        ("hard_filter", hard_temporal_filter),
        ("no_temporal", no_temporal_filter),
        ("soft_decay", soft_temporal_filter),
    ]:
        metrics = evaluate_temporal_system(cases, fn)
        results["systems"][name] = metrics
        print(f"\n=== {name} ===")
        print(f"  Recall@1: {metrics['recall@1']:.3f}")
        print(f"  Recall@5: {metrics['recall@k']:.3f}")
        print(f"  FLR: {metrics['future_leakage_rate']:.3f}")
        print(f"  ELR: {metrics['evidence_loss_rate']:.3f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {output_path}")

    return results


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent.parent.parent / "results" / "v21" / "temporal" / "stress_benchmark.json"
    run_temporal_benchmark(n_per_category=20, output_path=str(out))
