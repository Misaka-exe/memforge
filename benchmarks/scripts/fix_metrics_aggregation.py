"""
Stage 6-B: Fix metrics aggregation bug.

Recomputes metrics.json from raw_qa.jsonl for each system.
The original metrics.json had checkpoint-counter overflow (n=1000 instead of 500,
tokens doubled). Accuracy numbers were correct (computed from raw_qa), but n and
token totals were wrong.

This script does NOT modify any experiment logic -- it only regenerates the
bookkeeping artifact (metrics.json) from the raw results.
"""
import json
import os

RESULTS_DIR = r"D:\agentmemory\memforge\results\longmemeval\stage5_real"
SYSTEMS = ["naive_vector", "hybrid", "full"]

for sys_name in SYSTEMS:
    raw_path = os.path.join(RESULTS_DIR, sys_name, "raw_qa.jsonl")
    metrics_path = os.path.join(RESULTS_DIR, sys_name, "metrics.json")

    # Load original metrics to preserve metadata
    with open(metrics_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    # Load raw records
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    answerable = [r for r in records if not r.get("is_abstention")]
    abstention = [r for r in records if r.get("is_abstention")]

    overall_acc = sum(1 for r in records if r.get("correct")) / n if n else 0
    ans_acc = sum(1 for r in answerable if r.get("correct")) / len(answerable) if answerable else 0
    abs_acc = sum(1 for r in abstention if r.get("correct")) / len(abstention) if abstention else 0

    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in records)
    total_tokens = prompt_tokens + completion_tokens

    latencies = [r.get("latency_ms", 0) for r in records if r.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # By category
    by_category = {}
    for r in records:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if r.get("correct"):
            by_category[cat]["correct"] += 1
    for cat in by_category:
        by_category[cat]["accuracy"] = round(
            by_category[cat]["correct"] / by_category[cat]["total"], 4
        )

    fixed = {
        "n": n,
        "answerable_n": len(answerable),
        "answerable_accuracy": round(ans_acc, 4),
        "abstention_n": len(abstention),
        "abstention_accuracy": round(abs_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_latency_ms": round(avg_latency, 1),
        "by_category": by_category,
        "system": orig.get("system", sys_name),
        "prompt_version": orig.get("prompt_version", "v1.0"),
        "model": orig.get("model", "deepseek-chat"),
        "temperature": orig.get("temperature", 0.0),
        "max_tokens": orig.get("max_tokens", 256),
        "total_run_seconds": orig.get("total_run_seconds", None),
        "note": "Regenerated from raw_qa.jsonl (Stage 6-B fix: corrected checkpoint counter overflow)",
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"=== {sys_name} ===")
    print(f"  n: {orig['n']} -> {n}")
    print(f"  answerable_n: {orig['answerable_n']} -> {len(answerable)}")
    print(f"  abstention_n: {orig['abstention_n']} -> {len(abstention)}")
    print(f"  overall_accuracy: {orig['overall_accuracy']} -> {fixed['overall_accuracy']}")
    print(f"  total_tokens: {orig['total_tokens']} -> {total_tokens}")
    print(f"  avg_latency_ms: {orig['avg_latency_ms']} -> {fixed['avg_latency_ms']}")
    print(f"  Saved to: {metrics_path}")
    print()

print("All metrics.json files regenerated.")
