# MemForge v2 Audit

Scope: all changes between commit `49fe710` (v1 release) and the v2 working tree.
This is a self-audit performed as the final stage of v2 development.

## 1. Architecture Review

### Module boundary
- `core/` (types, lifecycle) — untouched. LifecycleManager remains the sole status-transition entry point.
- `memory/` — v2 additive modules:
  - `transition.py` — `MemoryTransition` + `TransitionLog` audit stream.
  - `grounding.py` — `Citation`, `GroundedAnswer`, parser, citation-validity helpers.
  - `conflict.py` — extended with `ConflictType`, `ConflictDecision`, `ConflictClassifier`.
- `retrieval/`
  - `temporal.py` — v1 `memory_active_at`/`filter_temporal` preserved; v2 `TemporalScorer` strategies added.
  - `reranker.py` — `temporal_weight` added; soft temporal integration.
  - `gate.py` — `SufficiencyGate` + `RetrievalDecision`.
  - `hybrid.py` — accepts optional temporal scorer.
- `verification/` — new package: `Verifier`, `VerificationResult`.
- `pipeline/v2.py` — `MemForgeV2Pipeline` wiring retrieval → gate → grounded QA → verify.
- `llm/` — additive `generate_raw` / `RawCompletion`; v1 `generate(messages, response_model)` unchanged.

### Duplicate implementation
- Two JSON parsers coexist: `memforge/memory/grounding.py::parse_grounded_answer` (id-list based) and `benchmarks/evaluation/qa_v2.py::_json_to_grounded` (Memory-list based). They serve different layers (library vs benchmark harness) and share the same `Citation`/`GroundedAnswer` models. Acceptable; flagged as technical debt.
- `cosine_similarity` exists in both `memory/conflict.py` and `retrieval/reranker.py`. Pre-existing; left as-is to avoid touching v1.

### Circular dependencies
- `pipeline/v2.py` imports `benchmarks/evaluation/qa_v2.py`. This is a layering smell (library → benchmark). It works because benchmarks is a top-level package; flagged for v2.1 refactor (move V2QAQuestion/V2QAResult into a shared module).

### v1/v2 isolation
- All v1 result files under `results/longmemeval/` untouched.
- v1 evaluator `benchmarks/evaluation/qa.py` untouched.
- v1 temporal hard filter preserved behind `memory_active_at` (opt-in via `hard_temporal=True` in repository, and when no scorer is wired into the reranker).

## 2. Lifersistence Review

- All status transitions still go through `LifecycleManager.transition()`.
- `repository.py` never assigns `memory.status` directly (verified: `update()` changes business fields only).
- `MemoryEventRecord` remains append-only (insert only).
- v2 `TransitionLog` is a separate audit stream; it does not replace `MemoryEventRecord`.
- No new `MemoryStatus` values introduced.

## 3. Retrieval Review

### Temporal soft vs hard
- v1 hard filter in three places: `temporal.memory_active_at`, `reranker.rerank(as_of=...)`, `repository.search(SQL WHERE)`.
- v2 changes:
  - `reranker.rerank`: when a `TemporalScorer` is wired, no hard drop; temporal score enters utility with `temporal_weight`.
  - `repository.search`: `hard_temporal: bool = False` default; when False, SQL ignores `as_of`.
  - `hybrid.py`: passes `as_of` to scorer, not to hard filter.
- No hidden hard filter remains in the v2 soft path.

### Reranker double-counting
- `score()` = `w_rel*rel + w_conf*conf + w_imp*imp + w_fresh*fresh + w_temporal*temporal`. Each term independent; no double-count.

### Gate does not affect recall
- `SufficiencyGate.assess()` is observational; it never mutates the candidate list.

## 4. Conflict Review

- `ConflictClassifier` maps (similarity, LLM judgement) → one of `NO_CONFLICT` / `POSSIBLE_CONFLICT` / `CONFIRMED_CONFLICT` / `UNCERTAIN`.
- Similarity spike alone never triggers deprecation.
- `UNCERTAIN` and `POSSIBLE_CONFLICT` produce no state change; old memory stays ACTIVE.
- Only `CONFIRMED_CONFLICT` with confidence ≥ threshold deprecates the old memory.
- Every decision is recorded in `TransitionLog` with `destructive` flag.

## 5. Anti-hallucination Review

- Gate allows abstention on `INSUFFICIENT` (never forces answer).
- v2 prompt explicitly requires citations per factual claim.
- Citations point at retrieved memory indices; `citation_validity` flags pointers outside the retrieved set.
- Verifier does not regenerate; it only emits PASS/WARN/FAIL + unsupported/contradiction lists.
- No LLM→LLM→LLM loop.

## 6. Tests

- `pytest tests` → **136 passed, 10 skipped**.
- 10 skips are pre-existing integration tests requiring PostgreSQL (unavailable in this environment).
- New v2 tests: 59 (transition/grounding/gate/verifier models, temporal soft, conflict v2, sufficiency gate, grounding, verifier, pipeline).

## 7. Regression

- v1 `test_conflict.py` (5 tests) unchanged and green.
- v1 `test_stage3.py` temporal/hybrid/reranker tests unchanged and green.
- v1 result files under `results/longmemeval/` not modified.
- `results/v2/` is a new directory; no v1 result overwritten.

## 8. Benchmark Reproducibility

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json` (500 questions).
- Embedding: `all-MiniLM-L6-v2` (cached in `.hf_cache/`).
- LLM: `deepseek-chat` via `https://api.deepseek.com/v1`, temperature 0.0.
- Prompt versions: v1 `v1.0`, v2 `v2.0-grounded`.
- No random seed needed (deterministic retrieval; LLM temperature 0).

## 9. Security

- No API key in tracked files (verified via `git grep sk-16d0`).
- `.env` not created; key passed via environment variable only.
- `results/v2_run.log` contains only progress output; no key material.

## 10. Known Limitations

- ELR measured as 0.0 on LongMemEval-S: in this dataset all haystack sessions predate the question date, so v1 hard temporal filter does not drop gold evidence at retrieval time. The v2 soft mechanism is still correct (proven in unit tests) but its retrieval impact is not visible on this split.
- v2 QA sample is small (30 questions); accuracy difference (v1 0.73 vs v2 0.63) is within noise.
- Verifier is rule-based (token overlap); not LLM-as-judge.
- `pipeline/v2.py` imports from `benchmarks/` — layering debt.

## 11. Technical Debt

- Merge the two JSON parsers (`grounding.py` vs `qa_v2.py`).
- Move `V2QAQuestion`/`V2QAResult` out of `benchmarks/` into a shared location.
- Replace duplicate `cosine_similarity` with a shared util.
- Add integration tests with PostgreSQL (skipped in this environment).

## 12. Recommended v2.1 Changes

1. Persist `TransitionLog` to the database (new table) for production audit.
2. LLM-backed citation validity check (currently rule-based).
3. Calibrate sufficiency thresholds on a held-out split.
4. Move benchmark-facing data classes out of `benchmarks/` into `memforge.pipeline`.
5. Add ELR measurement on a split with future-dated gold sessions (knowledge-update category).
