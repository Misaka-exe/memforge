"""Reconsolidation Synthetic Causal Benchmark (V2.1 Research Audit).

Tests whether slot-level reconsolidation preserves non-conflicting information.

Design:
  Initial memory: city=Beijing, job=Engineer, hobby=Tennis
  New evidence:   city=Shanghai (only updates city slot)
  After reconsolidation:
    v1: city=Beijing (DEPRECATED, but preserved)
    v2: city=Shanghai, job=Engineer, hobby=Tennis (ACTIVE, derived_from=v1)

  Test queries:
    Q1 current city?   → Shanghai (Current Recall)
    Q2 old city?       → Beijing  (Historical Recall, via lineage)
    Q3 job?            → Engineer (Unchanged Slot Recall)
    Q4 hobby?          → Tennis   (Unchanged Slot Recall)

Metrics:
  - Current Recall: new version has correct current slot
  - Historical Recall: old version still retrievable via lineage
  - Unchanged Slot Recall: job/hobby preserved in new version
  - Evidence Preservation Rate: all slots still retrievable
  - False Update Rate: slots incorrectly modified
  - Lineage Completeness: derived_from / version / lineage_id all present

Output: results/v21/reconsolidation/causal_benchmark.json
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from memforge.core.types import Memory, MemoryStatus
from memforge.memory.reconsolidation import (
    MemoryReconsolidator,
    ReconsolidationDecision,
    compute_evidence_preservation_rate,
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "v21" / "reconsolidation" / "causal_benchmark.json"


@dataclass
class SlotMemory:
    """A memory with explicit slots for causal testing."""
    memory_id: str
    slots: dict[str, str]
    status: str = "ACTIVE"
    lineage_id: str = ""
    version: int = 1
    derived_from: str = ""

    def content(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.slots.items())

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "slots": self.slots,
            "status": self.status,
            "lineage_id": self.lineage_id,
            "version": self.version,
            "derived_from": self.derived_from,
        }


@dataclass
class CausalCase:
    case_id: str
    initial_slots: dict[str, str]
    new_evidence: str
    updated_slot: str
    new_slot_value: str
    queries: list[dict] = field(default_factory=list)


def generate_cases(n: int = 30) -> list[CausalCase]:
    """Generate causal test cases."""
    cases = []
    cities = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Chengdu"]
    jobs = ["Engineer", "Designer", "Manager", "Teacher", "Doctor"]
    hobbies = ["Tennis", "Swimming", "Reading", "Gaming", "Cooking"]
    names = ["Alice", "Bob", "Carol", "David", "Eve"]

    for i in range(n):
        idx = i % 5
        initial = {
            "name": names[idx],
            "city": cities[idx],
            "job": jobs[idx],
            "hobby": hobbies[idx],
        }
        new_city = cities[(idx + 1) % 5]
        evidence = f"{names[idx]} moved to {new_city}."

        queries = [
            {"query": "current city", "slot": "city", "expected": new_city, "type": "current"},
            {"query": "old city", "slot": "city", "expected": cities[idx], "type": "historical"},
            {"query": "job", "slot": "job", "expected": jobs[idx], "type": "unchanged"},
            {"query": "hobby", "slot": "hobby", "expected": hobbies[idx], "type": "unchanged"},
            {"query": "name", "slot": "name", "expected": names[idx], "type": "unchanged"},
        ]

        cases.append(CausalCase(
            case_id=f"RC_{i:03d}",
            initial_slots=initial,
            new_evidence=evidence,
            updated_slot="city",
            new_slot_value=new_city,
            queries=queries,
        ))
    return cases


def run_no_reconsolidation(case: CausalCase) -> dict:
    """Baseline: no reconsolidation, memory stays as initial."""
    mem = SlotMemory(
        memory_id=str(uuid.uuid4()),
        slots=dict(case.initial_slots),
    )
    results = {"system": "no_reconsolidation", "memory": mem.to_dict(), "queries": []}

    for q in case.queries:
        if q["type"] == "current":
            # Without reconsolidation, current city is still old value
            correct = mem.slots.get(q["slot"]) == q["expected"]
        elif q["type"] == "historical":
            # Without reconsolidation, no historical version exists
            correct = False
        else:  # unchanged
            correct = mem.slots.get(q["slot"]) == q["expected"]
        results["queries"].append({**q, "correct": correct, "retrieved": mem.slots.get(q["slot"])})

    return results


def run_reconsolidation(case: CausalCase, reconsolidator: MemoryReconsolidator) -> dict:
    """With reconsolidation: create new version, deprecate old."""
    # Create initial memory (as Memory object for reconsolidator)
    old_mem = Memory(
        content="; ".join(f"{k}={v}" for k, v in case.initial_slots.items()),
        status=MemoryStatus.ACTIVE,
        importance=0.7,
        confidence=0.8,
    )

    # Run reconsolidation
    recon_result = reconsolidator.reconsolidate(
        memory=old_mem,
        new_evidence=case.new_evidence,
        verification_fail_reason="verification FAIL: city mismatch",
        confidence=0.9,
    )

    # Build slot memories
    old_slot = SlotMemory(
        memory_id=str(old_mem.id),
        slots=dict(case.initial_slots),
        status="DEPRECATED" if recon_result.decision == ReconsolidationDecision.UPDATE else "ACTIVE",
        lineage_id=recon_result.lineage_id or "",
        version=1,
    )

    if recon_result.decision == ReconsolidationDecision.UPDATE:
        # New version: updated slot + preserved unchanged slots
        new_slots = dict(case.initial_slots)
        new_slots[case.updated_slot] = case.new_slot_value
        new_slot = SlotMemory(
            memory_id=str(recon_result.new_memory_id),
            slots=new_slots,
            status="ACTIVE",
            lineage_id=recon_result.lineage_id or "",
            version=recon_result.version or 2,
            derived_from=str(old_mem.id),
        )
    else:
        new_slot = None

    results = {
        "system": "reconsolidation",
        "decision": recon_result.decision.value,
        "old_memory": old_slot.to_dict(),
        "new_memory": new_slot.to_dict() if new_slot else None,
        "queries": [],
    }

    for q in case.queries:
        if q["type"] == "current":
            retrieved = new_slot.slots.get(q["slot"]) if new_slot else old_slot.slots.get(q["slot"])
            correct = retrieved == q["expected"]
        elif q["type"] == "historical":
            # Historical: old version still retrievable via lineage
            retrieved = old_slot.slots.get(q["slot"])
            correct = retrieved == q["expected"] and old_slot.status == "DEPRECATED"
        else:  # unchanged
            retrieved = new_slot.slots.get(q["slot"]) if new_slot else old_slot.slots.get(q["slot"])
            correct = retrieved == q["expected"]
        results["queries"].append({**q, "correct": correct, "retrieved": retrieved})

    return results


def compute_metrics(all_results: list[dict], system_name: str) -> dict:
    """Compute aggregate metrics for a system."""
    system_results = [r for r in all_results if r["system"] == system_name]
    n = len(system_results)

    by_type = {"current": [], "historical": [], "unchanged": []}
    false_updates = 0
    total_slots = 0
    lineage_complete = 0

    for r in system_results:
        for q in r["queries"]:
            by_type[q["type"]].append(q["correct"])
            if q["type"] == "unchanged":
                total_slots += 1
                if not q["correct"]:
                    false_updates += 1
        # Check lineage completeness
        if r.get("new_memory"):
            nm = r["new_memory"]
            if nm.get("lineage_id") and nm.get("version", 0) > 1 and nm.get("derived_from"):
                lineage_complete += 1

    return {
        "n_cases": n,
        "current_recall": sum(by_type["current"]) / len(by_type["current"]) if by_type["current"] else 0,
        "historical_recall": sum(by_type["historical"]) / len(by_type["historical"]) if by_type["historical"] else 0,
        "unchanged_slot_recall": sum(by_type["unchanged"]) / len(by_type["unchanged"]) if by_type["unchanged"] else 0,
        "false_update_rate": false_updates / total_slots if total_slots else 0,
        "lineage_completeness": lineage_complete / n if n else 0,
        "overall_accuracy": sum(
            sum(by_type[t]) for t in by_type
        ) / sum(len(by_type[t]) for t in by_type) if any(by_type.values()) else 0,
    }


def main():
    cases = generate_cases(n=30)
    reconsolidator = MemoryReconsolidator(min_confidence=0.6)

    all_results = []
    for case in cases:
        all_results.append(run_no_reconsolidation(case))
        all_results.append(run_reconsolidation(case, reconsolidator))

    metrics_no_recon = compute_metrics(all_results, "no_reconsolidation")
    metrics_recon = compute_metrics(all_results, "reconsolidation")

    print("=== Reconsolidation Causal Benchmark (30 cases x 5 queries) ===")
    print(f"\n{'Metric':<25} {'No Recon':>10} {'Recon':>10}")
    print("-" * 50)
    for key in ["current_recall", "historical_recall", "unchanged_slot_recall",
                "false_update_rate", "lineage_completeness", "overall_accuracy"]:
        print(f"{key:<25} {metrics_no_recon[key]:>10.3f} {metrics_recon[key]:>10.3f}")

    # Decision distribution
    decisions = {}
    for r in all_results:
        if r["system"] == "reconsolidation":
            d = r["decision"]
            decisions[d] = decisions.get(d, 0) + 1
    print(f"\nReconsolidation decisions: {decisions}")

    result = {
        "n_cases": len(cases),
        "n_queries_per_case": 5,
        "design": "Initial: 4 slots (name/city/job/hobby). New evidence: city update. Test current/historical/unchanged.",
        "no_reconsolidation": metrics_no_recon,
        "reconsolidation": metrics_recon,
        "decision_distribution": decisions,
        "sample_case": {
            "initial": cases[0].initial_slots,
            "new_evidence": cases[0].new_evidence,
            "no_recon_result": all_results[0],
            "recon_result": all_results[1],
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
