"""QA runner for Stage 5 (LongMemEval-S end-to-end QA).

Two phases, deliberately separated:

Phase A - retrieval export (runs ONCE, reuses Stage 4.7 definitions verbatim):
    embed each question's turns once -> per-system top-K item ids persisted
    under results/longmemeval/stage5/retrieved/<system>/<question_id>.json
    (also saves a per-question embedding cache .npz so re-running or adding
     systems never re-embeds).

Phase B - QA (runs per system, zero embedding):
    load item ids -> build prompt (no gold) -> LLM -> evaluate -> append to
    raw_qa.jsonl (checkpoint; interrupted runs resume by skipping done ids).

Definitions imported from Stage 4.7 (run_longmemeval_ablation) so the
retrieval configuration is verbatim identical:
    parse_dt, CANDIDATE_K, CONFLICT_SIM_THRESHOLD, W_SEM, W_KW, W_FRESH,
    _rank_by_score.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.adapters.longmemeval import load_longmemeval
from benchmarks.evaluation.qa import (
    PROMPT_VERSION,
    QAQuestion,
    QAResult,
    aggregate_metrics,
    build_prompt,
    record,
)
from benchmarks.evaluation.providers import MockQAProvider, OpenAICompatibleQAProvider
from benchmarks.longmemeval import baselines as BL
from benchmarks.scripts.run_longmemeval_ablation import (
    CANDIDATE_K,
    CONFLICT_SIM_THRESHOLD,
    W_FRESH,
    W_KW,
    W_SEM,
    _rank_by_score,
    parse_dt,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUT = ROOT / "results" / "longmemeval" / "stage5"
TOP_K = 10  # QA context size per system (matches ablation max K)

SYSTEMS_ALL = ["naive_vector", "hybrid", "full", "full_context", "no_temporal", "no_conflict", "no_freshness"]
SYSTEMS_ROUND1 = ["naive_vector", "hybrid", "full"]


def _variant_order(name: str, n: int, sim: np.ndarray, kw: np.ndarray,
                   fresh: np.ndarray, usable: np.ndarray, superseded: np.ndarray) -> list[int]:
    """Replicates Stage 4.7 variant scoring exactly (verbatim copy of run_variant)."""
    def run_variant(usable: np.ndarray, exclude_superseded: bool, use_fresh: bool) -> list[int]:
        pool = [i for i in range(n) if usable[i] and (not exclude_superseded or not superseded[i])]
        if not pool:
            pool = list(range(n))
        if not pool:
            return []
        if len(pool) <= 10:
            return _rank_by_score(pool, {i: float(sim[i]) for i in pool}, TOP_K)
        sem_top = _rank_by_score(pool, {i: float(sim[i]) for i in pool}, CANDIDATE_K)
        kw_top = _rank_by_score(pool, {i: float(kw[i]) for i in pool}, CANDIDATE_K)
        union = sorted(set(sem_top) | set(kw_top))
        scores = {
            i: W_SEM * float(sim[i]) + W_KW * float(kw[i]) + (W_FRESH * float(fresh[i]) if use_fresh else 0.0)
            for i in union
        }
        return _rank_by_score(union, scores, TOP_K)

    all_idx = np.ones(n, dtype=bool)
    if name == "naive_vector":
        return _rank_by_score(list(range(n)), {i: float(sim[i]) for i in range(n)}, TOP_K)
    if name == "hybrid":
        return run_variant(all_idx, exclude_superseded=False, use_fresh=False)
    if name == "full":
        return run_variant(usable, exclude_superseded=True, use_fresh=True)
    if name == "no_temporal":
        return run_variant(all_idx, exclude_superseded=True, use_fresh=True)
    if name == "no_conflict":
        return run_variant(usable, exclude_superseded=False, use_fresh=True)
    if name == "no_freshness":
        return run_variant(usable, exclude_superseded=True, use_fresh=False)
    if name == "full_context":
        order = list(range(n))
        order.sort(key=lambda i: (parse_dt(items_dates[i]) or datetime.min))
        return order
    raise ValueError(name)


items_dates: list = []  # filled per question by export_retrieval for full_context sort


def _get_provider(model: str | None):
    from memforge.embeddings.sentence_transformers import SentenceTransformersProvider
    return SentenceTransformersProvider(model_name=model)


def export_retrieval(data_path: str, model: str | None, systems: list[str],
                     out: Path, max_questions: int = 0,
                     only_ids: set[str] | None = None) -> None:
    """Phase A: embed once per question, persist top-K item ids per system."""
    from datetime import datetime as _dt
    global items_dates

    cases = load_longmemeval(data_path)
    if max_questions:
        cases = cases[:max_questions]
    provider = _get_provider(model)
    print(f"[export] {len(cases)} cases | model={provider.model_name} | systems={systems}")

    emb_dir = out / "_cache" / "embeddings"
    ret_dir = out / "retrieved"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for s in systems:
        (ret_dir / s).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    for ci, case in enumerate(cases):
        q = case.questions[0]
        if only_ids and q.question_id not in only_ids:
            continue
        items = case.memories
        if not items:
            continue
        # NOTE: abstention questions are exported too (no gold evidence for
        # retrieval metrics, but QA needs a retrieved-memory context)
        # resume: skip if all requested systems already exported
        cache_file = emb_dir / f"{q.question_id}.npz"
        all_done = all((ret_dir / s / f"{q.question_id}.json").exists() for s in systems)
        if all_done and cache_file.exists():
            done += 1
            continue

        n = len(items)
        contents = [m.content for m in items]
        session_of = {m.id: str(m.metadata["session_id"]) for m in items}
        sdates = [parse_dt(m.metadata.get("session_date")) for m in items]
        qdate = parse_dt(q.question_date if hasattr(q, "question_date") else None)

        if cache_file.exists():
            npz = np.load(cache_file)
            vecs_n = npz["vecs_n"]; qvec_n = npz["qvec_n"]
        else:
            vecs = np.array(provider.embed(contents), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs_n = vecs / np.clip(norms, 1e-9, None)
            qvec = np.array(provider.embed([q.question])[0], dtype=np.float32)
            qn = np.linalg.norm(qvec)
            qvec_n = (qvec / max(qn, 1e-9)).astype(np.float32)
            np.savez(cache_file, vecs_n=vecs_n, qvec_n=qvec_n)

        sim = (vecs_n @ qvec_n).astype(np.float64)
        kw = np.array([BL.keyword_score(q.question, c) for c in contents], dtype=np.float64)

        usable_t = np.ones(n, dtype=bool)
        if qdate is not None:
            usable_t = np.array([False if d is None else d <= qdate for d in sdates], dtype=bool)
            if not usable_t.any():
                usable_t = np.ones(n, dtype=bool)

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

        usable_idx = [i for i in range(n) if usable_t[i]]
        sdate_vals = [(i, sdates[i]) for i in usable_idx if sdates[i] is not None]
        fresh = np.zeros(n, dtype=np.float64)
        if len(sdate_vals) >= 2:
            sdate_vals.sort(key=lambda x: x[1])
            for rank, (i, _) in enumerate(sdate_vals):
                fresh[i] = rank / (len(sdate_vals) - 1)

        for s in systems:
            if s == "full_context":
                order = sorted(range(n), key=lambda i: (sdates[i] if sdates[i] is not None else _dt.min))
            else:
                order = _variant_order(s, n, sim, kw, fresh, usable_t, superseded)
            ids = [items[i].id for i in order]
            texts = [items[i].content for i in order]
            (ret_dir / s / f"{q.question_id}.json").write_text(
                json.dumps({"question_id": q.question_id, "ids": ids, "texts": texts}, ensure_ascii=False),
                encoding="utf-8",
            )
        done += 1
        if (ci + 1) % 50 == 0:
            print(f"  [export {ci+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[export] done {done} questions in {time.time()-t0:.0f}s")


def _load_retrieved(out: Path, system: str, qid: str) -> list[str]:
    p = out / "retrieved" / system / f"{qid}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("texts", [])


async def run_qa(
    data_path: str,
    systems: list[str],
    provider,
    out: Path,
    max_questions: int = 0,
    only_ids: set[str] | None = None,
) -> None:
    """Phase B: QA per system with checkpoint resume."""
    cases = load_longmemeval(data_path)
    if max_questions:
        cases = cases[:max_questions]

    meta = {
        "stage": "5.1",
        "prompt_version": PROMPT_VERSION,
        "model": getattr(provider, "model", "mock"),
        "temperature": getattr(provider, "temperature", 0.0),
        "max_tokens": getattr(provider, "max_tokens", None),
        "top_k": TOP_K,
        "systems": systems,
        "data_file": str(data_path),
        "provider": type(provider).__name__,
        "retrieval_config": {
            "temporal": "session_date <= question_date",
            "conflict_sim_threshold": CONFLICT_SIM_THRESHOLD,
            "weights": {"w_sem": W_SEM, "w_kw": W_KW, "w_fresh": W_FRESH},
        },
    }
    manifest_path = out / "manifest.json"
    existing = {}
    if manifest_path.exists():
        try:
            import json as _json
            existing = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(meta)
    manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    for system in systems:
        sdir = out / system
        sdir.mkdir(parents=True, exist_ok=True)
        raw_path = sdir / "raw_qa.jsonl"
        done_ids: set[str] = set()
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["question_id"])
                    except Exception:
                        pass
        print(f"[qa:{system}] resume: {len(done_ids)} already done", flush=True)

        results: list[QAResult] = []
        with open(raw_path, "a", encoding="utf-8") as raw:
            for ci, case in enumerate(cases):
                q = case.questions[0]
                if only_ids and q.question_id not in only_ids:
                    continue
                if q.question_id in done_ids:
                    continue
                items = _load_retrieved(out, system, q.question_id)
                if not items:
                    # full_context falls back to all memory contents
                    items = [m.content for m in case.memories]
                if not items:
                    continue
                question = QAQuestion(
                    question_id=q.question_id,
                    question=q.question,
                    gold_answer=q.answer,
                    is_abstention=q.is_abstention,
                    category=q.category.value,
                    retrieved_items=items,
                )
                prompt = build_prompt(question)
                # Structural no-leak guarantee (Stage 5.1): build_prompt only
                # receives question + retrieved items; the gold answer field is
                # never part of the prompt signature (enforced by unit tests).
                # NOTE: the gold TEXT may legitimately appear inside retrieved
                # turns (that is what correct retrieval finds), so no text-level
                # check is possible or desirable here.
                completion = await provider.complete(prompt)
                result = record(question, system, completion, q.answer)
                results.append(result)
                raw.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                raw.flush()
                if (ci + 1) % 25 == 0:
                    print(f"  [{system} {ci+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s", flush=True)

        # Rebuild metrics from the full raw file (resume-robust): if this run
        # was fully resumed, `results` is empty but the raw file is complete.
        full_results = results[:]
        try:
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    full_results.append(QAResult(**d))
        except FileNotFoundError:
            pass
        metrics = aggregate_metrics(full_results)
        metrics.update({
            "system": system,
            "prompt_version": PROMPT_VERSION,
            "model": meta["model"],
            "temperature": meta["temperature"],
            "max_tokens": meta["max_tokens"],
            "total_run_seconds": round(time.time() - t0, 1),
        })
        (sdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[qa:{system}] {metrics}")


def write_artifacts(out: Path, systems: list[str]) -> None:
    """qa_comparison.csv + qa_by_category.csv from per-system metrics.json."""
    import csv

    rows = []
    for s in systems:
        p = out / s / "metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return
    with open(out / "qa_comparison.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "n", "answerable_n", "answerable_accuracy", "abstention_n",
                    "abstention_accuracy", "overall_accuracy", "prompt_tokens",
                    "completion_tokens", "total_tokens", "avg_latency_ms"])
        for r in rows:
            w.writerow([r.get("system"), r.get("n"), r.get("answerable_n"), r.get("answerable_accuracy"),
                        r.get("abstention_n"), r.get("abstention_accuracy"), r.get("overall_accuracy"),
                        r.get("prompt_tokens"), r.get("completion_tokens"), r.get("total_tokens"),
                        r.get("avg_latency_ms")])
    print(f"[artifacts] qa_comparison.csv: {len(rows)} systems")

    # by category: recompute from raw_qa.jsonl
    from collections import defaultdict as dd
    cat_agg: dict[str, dict[str, dict]] = dd(lambda: dd(list))
    for s in systems:
        raw_path = out / s / "raw_qa.jsonl"
        if not raw_path.exists():
            continue
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cat_agg[r["category"]][s].append(r)
    with open(out / "qa_by_category.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "system", "n", "answerable_accuracy", "abstention_accuracy", "overall_accuracy"])
        for cat in sorted(cat_agg):
            for s in systems:
                rs = cat_agg[cat].get(s)
                if not rs:
                    continue
                ans = [r for r in rs if not r["is_abstention"]]
                abs_ = [r for r in rs if r["is_abstention"]]
                aa = round(sum(1 for r in ans if r["correct"]) / len(ans), 4) if ans else None
                ab = round(sum(1 for r in abs_ if r["correct"]) / len(abs_), 4) if abs_ else None
                oa = round(sum(1 for r in rs if r["correct"]) / len(rs), 4) if rs else None
                w.writerow([cat, s, len(rs), aa, ab, oa])
    print("[artifacts] qa_by_category.csv written")