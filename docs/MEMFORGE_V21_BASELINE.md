# MemForge V2.1 Baseline Audit

> 日期: 2026-09-27 | 基线 commit: 6550d57 | 状态: V1/V2 frozen, V2.1 starting

## Frozen Baselines

### V1 (commit 49fe710)
- Stage 4.6: Real LongMemEval-S Retrieval
- Stage 4.7: Lifecycle Ablation (negative result: deterministic lifecycle hurts)
- Stage 5B: Real LLM QA (1500 calls, deepseek-chat)
- Results: `results/longmemeval/stage5_real/`
- Tests: 67 passed + 10 skipped

### V2 (commits d143c9a → f996053 → 6550d57)
- 5 modules: Temporal Soft Decay, Conflict v2, Sufficiency Gate, Citation Grounding, Post-hoc Verification
- 500-question QA: v2 grounded overall=0.404 (v1 hybrid=0.364)
- Gate: 100% SUFFICIENT (implemented, not validated) — uses memory.importance proxy
- ELR=0 (benchmark limitation, not validation)
- Results: `results/longmemeval/stage5_v2/`
- Tests: 136 passed + 10 skipped

## V2.1 Starting Point

- HEAD: 6550d57
- Working tree: clean
- LongMemEval-S cleaned: 264.5MB, 500 questions
- Embedding cache: 500 .npz files (stage5_real/_cache/embeddings/)
- V2 retrieval inputs: frozen at `results/longmemeval/stage5_real/retrieved/hybrid/`

## V2 Exposed Problems (V2.1 Targets)

1. **Gate not working**: uses memory.importance (default 0.8) as proxy → 100% SUFFICIENT
2. **Temporal soft decay unvalidated**: LongMemEval-S has no future-dated gold evidence → ELR=0
3. **Verification FAIL is passive**: only detection, no memory update mechanism
4. **No retrieval score standardization**: Gate/Grounding/Verification don't share score structure

## V2.1 Goals

1. RetrievalScore normalization (real scores, not importance proxy)
2. Calibrated Sufficiency Gate (validation set, multi-feature)
3. Temporal Stress Benchmark (synthetic, future-dated gold, Future Leakage Rate)
4. Evidence-Grounded Reconsolidation (lineage/versioning, slot-level updates)
5. Evidence Preservation metrics
6. Unified adaptive-memory evaluation pipeline (static vs adaptive modes)
7. Full ablation (gate/temporal/reconsolidation independently toggleable)
8. Failure Taxonomy document

## Strict Constraints

- DO NOT modify V1 results, V2 results, evaluator, LongMemEval-S data
- DO NOT re-run V2's 1500 API calls
- DO NOT introduce V3 bio-inspired mechanisms (Hebbian, ACT-R, evolutionary)
- DO NOT overwrite memory (use lineage/versioning)
- DO NOT delete tests to pass
- Preserve negative results
