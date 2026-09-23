"""Stage 5 CLI: LongMemEval-S end-to-end QA.

Usage:
    # smoke (10 questions: 5 answerable + 5 abstention) with mock provider
    python -m benchmarks.scripts.run_longmemeval_qa --smoke --provider mock

    # real run (round 1: naive_vector / hybrid / full)
    python -m benchmarks.scripts.run_longmemeval_qa --provider openai

    # add systems later (reuses exported retrieval, no re-embedding)
    python -m benchmarks.scripts.run_longmemeval_qa --provider openai \
        --systems no_temporal,no_conflict,no_freshness

Real LLM env: MEMFORGE_LLM_MODEL / MEMFORGE_LLM_BASE_URL / MEMFORGE_LLM_API_KEY
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation import qa_runner as QR

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUT = ROOT / "results" / "longmemeval" / "stage5"


def pick_smoke_ids(data_path: str, n_ans: int = 5, n_abs: int = 5) -> list[str]:
    cases = load_longmemeval(data_path)
    ans_ids: list[str] = []
    seen_cats = set()
    abs_ids: list[str] = []
    for case in cases:
        q = case.questions[0]
        if q.is_abstention:
            if len(abs_ids) < n_abs:
                abs_ids.append(q.question_id)
            continue
        if len(ans_ids) >= n_ans:
            continue
        if q.category.value in seen_cats and len(ans_ids) >= 4:
            continue
        if q.category.value not in seen_cats or len(ans_ids) < n_ans - 1:
            ans_ids.append(q.question_id)
            seen_cats.add(q.category.value)
    return (ans_ids + abs_ids)[: n_ans + n_abs]


def build_figures(out: Path, systems: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in systems:
        p = out / s / "metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return
    xs = list(range(len(rows)))
    labels = [r.get("system", "") for r in rows]

    # qa_accuracy.png
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [r.get("answerable_accuracy", 0) for r in rows], marker="o", label="answerable acc")
    ax.plot(xs, [r.get("abstention_accuracy", 0) for r in rows], marker="s", label="abstention acc")
    ax.plot(xs, [r.get("overall_accuracy", 0) for r in rows], marker="^", label="overall acc")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title("LongMemEval-S end-to-end QA accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(figdir / "qa_accuracy.png", dpi=150); plt.close()

    # token_cost.png
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, [r.get("total_tokens", 0) for r in rows], color="#4c72b0")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("total tokens"); ax.set_title("LongMemEval-S QA token cost per system")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(figdir / "token_cost.png", dpi=150); plt.close()

    # abstention_accuracy.png
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, [r.get("abstention_accuracy", 0) for r in rows], color="#55a868")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("abstention accuracy"); ax.set_title("LongMemEval-S abstention accuracy")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(figdir / "abstention_accuracy.png", dpi=150); plt.close()
    print("[figures] qa_accuracy.png / abstention_accuracy.png / token_cost.png written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=None)
    ap.add_argument("--systems", default=",".join(QR.SYSTEMS_ROUND1))
    ap.add_argument("--provider", choices=["mock", "openai"], default="mock")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-questions", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only_ids = None
    qa_systems = systems
    if args.smoke:
        only_ids = set(pick_smoke_ids(args.data))
        qa_systems = systems[:1]  # QA smoke: one system is enough for pipeline check
        print(f"[smoke] {len(only_ids)} questions (5 answerable + 5 abstention); export all {len(systems)} systems, QA on {qa_systems[0]}")

    if not args.skip_export:
        QR.export_retrieval(args.data, args.model, systems, out, args.max_questions, only_ids)

    if args.provider == "mock":
        from benchmarks.evaluation.providers import MockQAProvider
        provider = MockQAProvider()
        # NOTE: every prompt contains the literal string INSUFFICIENT_INFORMATION
        # (in the instructions), so no pattern-based abstention routing is
        # possible. Mock returns a fixed pseudo-answer; the correctness of
        # abstention/answerable judgement is covered by unit tests instead.
        provider.add_script(r"Question:", "42")
    else:
        from benchmarks.evaluation.providers import OpenAICompatibleQAProvider
        provider = OpenAICompatibleQAProvider()

    import asyncio

    asyncio.run(QR.run_qa(args.data, qa_systems, provider, out, args.max_questions, only_ids))
    QR.write_artifacts(out, qa_systems)
    build_figures(out, qa_systems)


if __name__ == "__main__":
    main()