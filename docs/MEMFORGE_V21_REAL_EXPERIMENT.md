# MemForge V2.1 Real LongMemEval-S Experiment Report

**Date**: 2026-09-27
**Dataset**: LongMemEval-S cleaned (500 questions: 470 answerable + 30 abstention)
**Model**: deepseek-chat, temperature=0.0
**Retrieval**: V2 frozen hybrid retrieval inputs (no re-embedding)
**Evaluator**: V1 deterministic token-overlap (consistent with V2 baseline)
**API calls**: 1000 (V2.1-Recon 500 + V2.1-Full 500), V2 baseline reused

## 1. Main Results

| System | Overall | Answerable | Abstention | Gate Insufficient | Recon Triggered |
|--------|---------|------------|------------|-------------------|-----------------|
| V2 Baseline (frozen) | 0.404 | 0.381 | 0.767 | 0% (importance proxy) | N/A |
| **V2.1-Recon** | **0.422** | **0.396** | **0.833** | disabled | 2.8% (14/500) |
| V2.1-Full | 0.410 | 0.385 | 0.800 | 1.6% (8/500) | 2.6% (13/500) |

### Key Findings

1. **V2.1-Recon outperforms V2 baseline on all metrics**
   - Overall: +1.8pp (0.404 → 0.422)
   - Answerable: +1.5pp (0.381 → 0.396)
   - Abstention: +6.6pp (0.767 → 0.833)

2. **V2.1-Full underperforms V2.1-Recon**
   - The calibrated Gate intercepts 8/500 questions, of which 6 are answerable false positives (consistent with audit: precision=0.25)
   - This confirms the audit finding: retrieval relevance ≠ evidence sufficiency

3. **Reconsolidation trigger rate is low (2.6-2.8%)**
   - Only triggered on verification FAIL
   - 13-14 memory updates out of 500 questions
   - Per-question adaptive mode: updates do not affect other questions (LongMemEval protocol)

## 2. By Category

| Category | n | V2 Baseline | V2.1-Recon | V2.1-Full |
|----------|---|-------------|------------|-----------|
| single-session-assistant | 56 | — | 0.839 | 0.821 |
| single-session-user | 64 | — | 0.750 | 0.734 |
| knowledge-update | 72 | — | 0.542 | 0.514 |
| temporal-reasoning | 127 | — | 0.220 | 0.213 |
| multi-session | 121 | — | 0.198 | 0.198 |
| single-session-preference | 30 | 0.0* | 0.0* | 0.0* |
| abstention | 30 | 0.767 | 0.833 | 0.800 |

*single-session-preference = 0% is a known evaluator limitation (token-overlap不适用于开放式偏好描述), not a real system failure.

V2.1-Recon >= V2.1-Full in every category, confirming Gate has uniformly negative impact in this configuration.

## 3. Efficiency

| System | Prompt Tokens | Completion Tokens | Total | Avg Latency | Errors |
|--------|--------------|-------------------|-------|-------------|--------|
| V2.1-Recon | 1,372,976 | 48,095 | 1,421,071 | 859ms | 0 |
| V2.1-Full | 1,351,009 | 45,766 | 1,396,775 | 830ms | 0 |

Estimated cost: ~1-2 RMB (DeepSeek deepseek-chat pricing).

## 4. Gate Behavior Analysis (V2.1-Full)

- Gate sufficient rate: 91.0% (455/500)
- Gate insufficient rate: 1.6% (8/500)
- Gate uncertain/disabled: 7.4% (37/500) — these proceed to LLM
- Of 8 intercepted: 6 answerable (FP), 2 abstention (TP)
- Abstention precision: 0.25 (matches audit exactly)

## 5. Reconsolidation Analysis

- Trigger condition: verification verdict == FAIL
- V2.1-Recon: 14 updates (2.8%)
- V2.1-Full: 13 updates (2.6%)
- All updates use slot-level detection (location/job/preference/relationship/name)
- Lineage/versioning preserved for all updates

Note: In LongMemEval-S standard evaluation, each question is independent. Reconsolidation updates do not propagate to subsequent questions. This measures mechanism correctness, not adaptive memory benefit.

## 6. Comparison with V2

| Dimension | V2 | V2.1-Recon | V2.1-Full |
|-----------|-----|------------|-----------|
| Gate | importance proxy (100% pass) | disabled | calibrated retrieval features |
| Reconsolidation | no | yes (FAIL-triggered) | yes (FAIL-triggered) |
| Token accounting | no | yes | yes |
| Overall QA | 0.404 | **0.422** | 0.410 |

## 7. Conclusions

1. **Evidence-grounded reconsolidation infrastructure improves QA** even with low trigger rate, likely through better abstention behavior on verification FAIL cases.

2. **Calibrated retrieval-score gate hurts QA** on LongMemEval-S, confirming the audit negative result. The gate needs better features or a different signal.

3. **V2.1-Recon is the best configuration** among the three, achieving +1.8pp overall and +6.6pp abstention accuracy over V2.

4. **Reconsolidation real-world effectiveness remains unproven** in adaptive mode — LongMemEval-S's per-question independence means updates don't affect future questions. A dedicated adaptive benchmark is needed.

## 8. Artifacts

- `results/longmemeval/stage_v21_real/v21_recon/raw_qa.jsonl` — 500 raw results
- `results/longmemeval/stage_v21_real/v21_recon/metrics.json` — aggregated metrics
- `results/longmemeval/stage_v21_real/v21_full/raw_qa.jsonl` — 500 raw results
- `results/longmemeval/stage_v21_real/v21_full/metrics.json` — aggregated metrics
- `benchmarks/scripts/run_v21_real_qa.py` — experiment runner
