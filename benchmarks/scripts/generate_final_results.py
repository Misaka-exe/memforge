"""
Stage 6-C: Generate final unified tables and figures.

Combines Stage 4.7 retrieval metrics with Stage 5B QA metrics (corrected).
Outputs:
  - qa_comparison.csv (with retrieval + QA + tokens)
  - qa_by_category.csv
  - figures/final_overview.png
  - figures/retrieval_vs_qa.png
  - figures/by_category_accuracy.png
  - figures/token_efficiency.png
"""
import json
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = r"D:\agentmemory\memforge\results\longmemeval\stage5_real"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SYSTEMS = ["naive_vector", "hybrid", "full"]
SYSTEM_LABELS = ["Naive Vector", "Hybrid", "Full Lifecycle"]

# Stage 4.7 retrieval metrics (from ablation.csv, n=470)
RETRIEVAL = {
    "naive_vector": {"r_any_5": 0.9468, "r_all_5": 0.6638, "mrr": 0.9009, "ndcg10": 0.8328},
    "hybrid":       {"r_any_5": 0.9638, "r_all_5": 0.7298, "mrr": 0.9345, "ndcg10": 0.8675},
    "full":         {"r_any_5": 0.9362, "r_all_5": 0.6234, "mrr": 0.8845, "ndcg10": 0.7880},
}

# Load corrected QA metrics
qa = {}
for sys_name in SYSTEMS:
    with open(os.path.join(RESULTS_DIR, sys_name, "metrics.json"), "r", encoding="utf-8") as f:
        qa[sys_name] = json.load(f)

# Load preference semantic eval
pref_path = os.path.join(RESULTS_DIR, "preference_semantic_eval.json")
pref_sem = {}
if os.path.exists(pref_path):
    with open(pref_path, "r", encoding="utf-8") as f:
        pref_sem = json.load(f)

# === 1. qa_comparison.csv ===
csv_path = os.path.join(RESULTS_DIR, "qa_comparison.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "system", "n", "r_any@5", "r_all@5", "mrr", "ndcg@10",
        "qa_overall", "qa_answerable", "qa_abstention",
        "answerable_n", "abstention_n",
        "prompt_tokens", "completion_tokens", "total_tokens", "avg_latency_ms"
    ])
    for sys_name in SYSTEMS:
        r = RETRIEVAL[sys_name]
        q = qa[sys_name]
        w.writerow([
            sys_name, q["n"],
            r["r_any_5"], r["r_all_5"], r["mrr"], r["ndcg10"],
            q["overall_accuracy"], q["answerable_accuracy"], q["abstention_accuracy"],
            q["answerable_n"], q["abstention_n"],
            q["prompt_tokens"], q["completion_tokens"], q["total_tokens"], q["avg_latency_ms"]
        ])
print(f"Written: {csv_path}")

# === 2. qa_by_category.csv ===
cat_csv = os.path.join(RESULTS_DIR, "qa_by_category.csv")
with open(cat_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "system", "n", "correct", "accuracy"])
    for sys_name in SYSTEMS:
        q = qa[sys_name]
        for cat, data in sorted(q.get("by_category", {}).items()):
            w.writerow([cat, sys_name, data["total"], data["correct"], data["accuracy"]])
print(f"Written: {cat_csv}")

# === 3. Figure: Final Overview (retrieval + QA + tokens) ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = np.arange(len(SYSTEMS))
width = 0.35

# Panel 1: Retrieval
ax = axes[0]
r_any = [RETRIEVAL[s]["r_any_5"] for s in SYSTEMS]
r_all = [RETRIEVAL[s]["r_all_5"] for s in SYSTEMS]
ax.bar(x - width/2, r_any, width, label="R@Any@5", color="#4C72B0")
ax.bar(x + width/2, r_all, width, label="R@All@5", color="#55A868")
ax.set_ylabel("Recall")
ax.set_title("Retrieval (Stage 4.7, n=470)")
ax.set_xticks(x)
ax.set_xticklabels(SYSTEM_LABELS, rotation=15)
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Panel 2: QA
ax = axes[1]
qa_overall = [qa[s]["overall_accuracy"] for s in SYSTEMS]
qa_ans = [qa[s]["answerable_accuracy"] for s in SYSTEMS]
qa_abs = [qa[s]["abstention_accuracy"] for s in SYSTEMS]
ax.bar(x - width, qa_overall, width, label="Overall QA", color="#C44E52")
ax.bar(x, qa_ans, width, label="Answerable", color="#8172B3")
ax.bar(x + width, qa_abs, width, label="Abstention", color="#CCB974")
ax.set_ylabel("Accuracy")
ax.set_title("End-to-End QA (Stage 5B, n=500)")
ax.set_xticks(x)
ax.set_xticklabels(SYSTEM_LABELS, rotation=15)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Panel 3: Tokens
ax = axes[2]
tokens = [qa[s]["total_tokens"] / 1e6 for s in SYSTEMS]
colors = ["#4C72B0", "#55A868", "#C44E52"]
bars = ax.bar(x, tokens, 0.5, color=colors)
ax.set_ylabel("Total Tokens (millions)")
ax.set_title("Token Cost")
ax.set_xticks(x)
ax.set_xticklabels(SYSTEM_LABELS, rotation=15)
for bar, val in zip(bars, tokens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}M",
            ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "final_overview.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Written: {fig_path}")

# === 4. Figure: Retrieval vs QA scatter ===
fig, ax = plt.subplots(figsize=(8, 6))
for i, sys_name in enumerate(SYSTEMS):
    r = RETRIEVAL[sys_name]["r_any_5"]
    q = qa[sys_name]["overall_accuracy"]
    ax.scatter(r, q, s=200, c=colors[i], label=SYSTEM_LABELS[i], zorder=5)
    ax.annotate(SYSTEM_LABELS[i], (r, q), textcoords="offset points",
                xytext=(10, 5), fontsize=10)
ax.set_xlabel("Retrieval R@Any@5 (Stage 4.7)")
ax.set_ylabel("QA Overall Accuracy (Stage 5B)")
ax.set_title("Retrieval Quality vs End-to-End QA")
ax.set_xlim(0.92, 0.98)
ax.set_ylim(0.32, 0.38)
ax.grid(alpha=0.3)
# Add annotation for the key finding
ax.annotate("Full has lower retrieval\nbut higher QA than Naive",
            xy=(0.94, 0.345), xytext=(0.955, 0.335),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray")
fig_path = os.path.join(FIG_DIR, "retrieval_vs_qa.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Written: {fig_path}")

# === 5. Figure: By-category accuracy ===
categories = ["knowledge-update", "single-session-user", "single-session-assistant",
              "multi-session", "temporal-reasoning", "single-session-preference"]
cat_labels = ["Knowledge\nUpdate", "Single-sess\nUser", "Single-sess\nAssistant",
              "Multi-\nsession", "Temporal\nReasoning", "Preference\n(eval mismatch)"]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(categories))
width = 0.25
for i, sys_name in enumerate(SYSTEMS):
    accs = []
    for cat in categories:
        bc = qa[sys_name].get("by_category", {}).get(cat, {})
        accs.append(bc.get("accuracy", 0))
    ax.bar(x + i*width - width, accs, width, label=SYSTEM_LABELS[i], color=colors[i])

ax.set_ylabel("QA Accuracy")
ax.set_title("QA Accuracy by Category (Stage 5B)")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 1.0)
# Mark preference as N/A
ax.axvspan(4.5, 5.5, alpha=0.1, color="gray")
ax.text(5, 0.95, "N/A:\nevaluator\nmismatch", ha="center", va="top", fontsize=8, color="gray")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "by_category_accuracy.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Written: {fig_path}")

# === 6. Figure: Token efficiency ===
fig, ax = plt.subplots(figsize=(8, 6))
for i, sys_name in enumerate(SYSTEMS):
    t = qa[sys_name]["total_tokens"] / 1e6
    q = qa[sys_name]["overall_accuracy"]
    ax.scatter(t, q, s=200, c=colors[i], label=SYSTEM_LABELS[i], zorder=5)
    ax.annotate(SYSTEM_LABELS[i], (t, q), textcoords="offset points",
                xytext=(10, 5), fontsize=10)
ax.set_xlabel("Total Tokens (millions)")
ax.set_ylabel("QA Overall Accuracy")
ax.set_title("Token Cost vs QA Accuracy")
ax.grid(alpha=0.3)
fig_path = os.path.join(FIG_DIR, "token_efficiency.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Written: {fig_path}")

# === Print final summary table ===
print("\n" + "=" * 90)
print("FINAL UNIFIED RESULTS (Stage 4.7 Retrieval + Stage 5B QA)")
print("=" * 90)
print(f"{'System':<18} {'R@Any5':>8} {'R@All5':>8} {'MRR':>8} {'QA':>8} {'AnsQA':>8} {'AbsQA':>8} {'Tokens':>10} {'Latency':>8}")
print("-" * 90)
for sys_name in SYSTEMS:
    r = RETRIEVAL[sys_name]
    q = qa[sys_name]
    print(f"{sys_name:<18} {r['r_any_5']:>8.4f} {r['r_all_5']:>8.4f} {r['mrr']:>8.4f} "
          f"{q['overall_accuracy']:>8.4f} {q['answerable_accuracy']:>8.4f} {q['abstention_accuracy']:>8.4f} "
          f"{q['total_tokens']:>10,} {q['avg_latency_ms']:>7.0f}ms")
print("=" * 90)

if pref_sem:
    print("\nPreference Semantic Similarity (supplementary, n=30):")
    for sys_name in SYSTEMS:
        if sys_name in pref_sem:
            p = pref_sem[sys_name]
            print(f"  {sys_name}: avg={p['avg_semantic_similarity']:.4f}, "
                  f"median={p['median_semantic_similarity']:.4f}, "
                  f"insufficient={p['insufficient_count']}/30")
