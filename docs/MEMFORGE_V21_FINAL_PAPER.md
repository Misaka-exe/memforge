# MemForge V2.1: Evidence-Grounded Adaptive Memory Management

## Final Research Report

**Version**: v2.1-final
**Date**: 2026-09-27
**Repository**: https://github.com/Misaka-exe/memforge
**Tag**: v2.1-final

---

## Abstract

Long-term memory systems for AI agents must balance evidence preservation against adaptive updating. MemForge V2.1 introduces three mechanisms—calibrated retrieval sufficiency gating, temporal soft-decay scoring, and evidence-grounded reconsolidation with lineage versioning—designed to address failures observed in V1's deterministic lifecycle. We evaluate on LongMemEval-S (500 questions) with a frozen retrieval backbone and the deepseek-chat LLM. Results show that V2.1-Recon (reconsolidation without gate) improves overall QA accuracy from 0.404 to 0.422 (+1.8pp) and abstention accuracy from 0.767 to 0.833 (+6.6pp) over the V2 baseline. However, the calibrated sufficiency gate hurts performance (V2.1-Full: 0.410), intercepting 8/500 questions with only 25% abstention precision. A synthetic causal benchmark confirms reconsolidation correctly updates target slots (current recall 1.0) while preserving unchanged slots (1.0) and historical versions (1.0 via lineage). We conclude that evidence-grounded memory updating is promising, but retrieval-relevance-based gating is insufficient for evidence sufficiency judgment.

---

## 1. Introduction

AI agents operating over long conversations require memory systems that can: (1) retrieve relevant past information, (2) update memories when new evidence contradicts old, and (3) avoid destroying useful history. MemForge V1 demonstrated that deterministic lifecycle heuristics—hard temporal filtering, cosine-similarity conflict detection, and freshness-based reranking—can actively harm retrieval by deleting gold evidence (Stage 4.7 ablation). V2 addressed this by introducing evidence-grounded answering (citation grounding + post-hoc verification), improving QA from 0.364 to 0.404, but its sufficiency gate used memory importance as a proxy, resulting in 100% pass-through.

V2.1 targets three specific failures:
1. **Gate failure**: V2's gate never actually blocks anything. We replace it with a calibrated gate using real retrieval features.
2. **Temporal validation gap**: V2's soft temporal decay was never stress-tested. We build a dedicated temporal stress benchmark with future-leakage metrics.
3. **No memory update mechanism**: V2's verification FAIL was a dead-end signal. We add evidence-grounded reconsolidation with lineage/versioning that produces new memory versions without overwriting.

The core research question: *Can calibrated evidence-aware memory management improve evidence preservation and downstream QA while reducing unsupported memory transitions?*

---

## 2. Related Work

**Long-term memory benchmarks.** LongMemEval (Wu et al., 2024) provides 500 evaluation instances across 6 question types with session-level gold evidence. LoCoMo (Sato et al., 2024) tests continual memory updates. MemForge uses LongMemEval-S cleaned as its primary benchmark.

**Memory lifecycle systems.** MemoryBank (Zhong et al., 2023) introduces Ebbinghaus-inspired forgetting. Generative Agents (Park et al., 2023) use reflection and importance scoring. MemForge differs by treating memory as a lifecycle entity with explicit state transitions, evidence tracking, and auditability.

**Conflict resolution.** RAG-based conflict detection typically uses semantic similarity. MemForge V2 introduced 4-level uncertainty (NO_CONFLICT / POSSIBLE / CONFIRMED / UNCERTAIN) with destructive transitions only on confirmed conflict.

**Reconsolidation.** Inspired by memory reconsolidation in neuroscience (Nader, 2003), where retrieved memories become labile and can be updated. MemForge V2.1 implements a computational analogue: verification FAIL triggers slot-level mutation with lineage preservation.

---

## 3. Method

### 3.1 System Overview

V2.1 maintains V2's frozen retrieval backbone (hybrid semantic + keyword retrieval with all-MiniLM-L6-v2 embeddings) and adds three modules:

```
Query → Hybrid Retrieval (frozen) → Retrieval Features
                                        ↓
                              Sufficiency Gate v2
                              (calibrated, real scores)
                                        ↓ sufficient
                              Grounded QA (V2 pipeline)
                                        ↓
                              Post-hoc Verification
                                        ↓ FAIL
                              Evidence-Grounded Reconsolidation
                              (lineage + slot-level update)
```

### 3.2 Retrieval Score Normalization

`RetrievalFeatures` captures per-question retrieval statistics: top1_score, top5_mean, score_margin, n_retrieved, mean_score, std_score. Computed from V2's frozen embedding cache (no re-embedding). 500 questions exported.

### 3.3 Calibrated Sufficiency Gate

Two gate implementations:
- **ThresholdGate**: single threshold on top1_score
- **MultiFeatureGate**: combined rules on top1 + top5_mean + n_retrieved

Calibration via grid search on the 500-question set (exploratory, not independent validation).

### 3.4 Temporal Soft Decay

Four strategies: no_decay, linear, exponential, window. Replaces V1's hard filtering with score-level modulation. Future-dated memories are downweighted but never removed.

### 3.5 Evidence-Grounded Reconsolidation

Triggered only on verification FAIL. Produces a new memory version with:
- `lineage_id`: shared across versions
- `version`: incremented
- `derived_from`: parent memory ID
- `supersedes`: set on parent (DEPRECATED but retrievable)
- Slot-level updates: only fields supported by new evidence are modified

Decisions: KEEP / UPDATE / ABSTAIN (low confidence → no destructive change).

---

## 4. Experiments

### 4.1 Gate Calibration and Audit

**Feature distribution**: answerable top1 mean=0.540 (std=0.139), abstention top1 mean=0.451 (std=0.136). Difference=0.089, but distributions overlap heavily.

**ThresholdGate**: optimal threshold=0.1 (F1=0.969), which equals no filtering since all questions have top1>0.1. Single-threshold gating is completely ineffective.

**MultiFeatureGate** (calibrated top1=0.3, top5=0.2):
- 8/500 intercepted
- 6 answerable false positives, 2 abstention true positives
- Abstention precision=0.25, recall=0.067, F1=0.105
- 28/30 abstention questions pass through

**Risk-coverage curve**: threshold=0.74 yields selective accuracy=0.756 but coverage=0.082 (must block 92% of questions).

**Conclusion**: Retrieval relevance is not equivalent to evidence sufficiency. Calibrated retrieval-score gating is insufficient on LongMemEval-S.

### 4.2 Temporal Stress Benchmark

120 synthetic cases across 6 categories: current, historical, future, update, contradictory_temporal, future_contamination.

| System | R@1 | R@5 | FLR (corrected) | ELR |
|--------|-----|-----|-----------------|-----|
| Hard Filter | 0.833 | 0.833 | 0.000 | 0.167 |
| No Temporal | 0.500 | 1.000 | 0.200 | 0.000 |
| Soft Decay | 0.833 | 1.000 | 0.200 | 0.000 |

FLR (Future Leakage Rate) corrected to exclude T3 Future category where future memory is the gold answer.

**Pareto analysis**: all soft-decay floor parameters produce identical results because the benchmark has only 2-3 memories per case (top-5 always contains all). The benchmark is too simple to distinguish decay parameters.

**Trade-off**: Hard Filter achieves zero leakage at the cost of 16.7% evidence loss. Soft Decay preserves all evidence and matches Hard Filter's R@1, but retains 20% future leakage.

### 4.3 Reconsolidation Causal Benchmark

30 cases × 5 queries. Initial memory has 4 slots (name/city/job/hobby). New evidence updates only city.

| Metric | No Reconsolidation | Reconsolidation |
|--------|-------------------|-----------------|
| Current Recall | 0.000 | **1.000** |
| Historical Recall | 0.000 | **1.000** |
| Unchanged Slot Recall | 1.000 | **1.000** |
| False Update Rate | 0.000 | **0.000** |
| Lineage Completeness | 0.000 | **1.000** |
| Overall Accuracy | 0.600 | **1.000** |

30/30 cases produce UPDATE decisions. The mechanism correctly updates the target slot, preserves all unchanged slots, and maintains historical accessibility via lineage.

### 4.4 Real LongMemEval-S Evaluation

**Setup**: 500 questions (470 answerable + 30 abstention), deepseek-chat, temperature=0.0, V2 frozen retrieval, V1 deterministic evaluator. V2 baseline reused (0 new API calls). V2.1-Recon and V2.1-Full each run 500 questions (1000 total API calls).

| System | Overall | Answerable | Abstention | Gate Insufficient | Recon Triggered |
|--------|---------|------------|------------|-------------------|-----------------|
| V2 Baseline | 0.404 | 0.381 | 0.767 | 0% (proxy) | N/A |
| **V2.1-Recon** | **0.422** | **0.396** | **0.833** | disabled | 2.8% (14) |
| V2.1-Full | 0.410 | 0.385 | 0.800 | 1.6% (8) | 2.6% (13) |

**By category** (V2.1-Recon / V2.1-Full):

| Category | n | Recon | Full |
|----------|---|-------|------|
| single-session-assistant | 56 | 0.839 | 0.821 |
| single-session-user | 64 | 0.750 | 0.734 |
| knowledge-update | 72 | 0.542 | 0.514 |
| temporal-reasoning | 127 | 0.220 | 0.213 |
| multi-session | 121 | 0.198 | 0.198 |
| single-session-preference | 30 | 0.0* | 0.0* |
| abstention | 30 | 0.833 | 0.800 |

*single-session-preference = 0% is a known evaluator limitation (token-overlap不适用于开放式偏好描述).

**Efficiency**: V2.1-Recon 1.42M tokens, V2.1-Full 1.40M tokens. 0 API errors. Estimated cost ~1-2 RMB.

---

## 5. Discussion

### 5.1 Why V2.1-Recon Outperforms V2

The +6.6pp abstention accuracy improvement is the most striking result. V2's grounded pipeline sometimes answers abstention questions incorrectly (abstention accuracy 0.767). V2.1-Recon's reconsolidation path, even with only 2.8% trigger rate, appears to improve the model's abstention behavior on verification-FAIL cases. This suggests that the act of attempting reconsolidation (rather than silently producing a potentially wrong answer) may calibrate the model's confidence.

### 5.2 Why the Gate Hurts

The gate's 8 interceptions include 6 answerable false positives. Each false positive directly converts a potentially correct answer into a guaranteed-wrong abstention. With only 2 true positives, the gate's net contribution is negative. This validates the audit conclusion: retrieval features do not carry enough signal for sufficiency judgment.

### 5.3 Reconsolidation's Low Trigger Rate

Only 13-14/500 questions trigger reconsolidation because: (1) only verification FAIL triggers it, (2) LongMemEval-S per-question independence means updates don't propagate. The mechanism is validated causally but its adaptive benefit remains untested. A dedicated adaptive benchmark is needed.

### 5.4 Retrieval ≠ QA

V2.1-Full has lower retrieval quality (gate removes candidates) but its QA is only 1.2pp below V2.1-Recon. This reinforces V2's finding that retrieval metrics do not fully determine downstream QA.

---

## 6. Limitations

1. **Gate calibration on evaluation set**: no independent validation set; MultiFeatureGate F1=0.105 for abstention detection
2. **Temporal benchmark simplicity**: 2-3 memories per case cannot distinguish decay parameters; FLR=0.200 may not generalize
3. **Reconsolidation slot detection is keyword-based**: synthetic causal test passes, but real LLM extraction is unvalidated
4. **No adaptive memory evaluation**: LongMemEval-S per-question independence means reconsolidation doesn't affect future questions
5. **Single evaluator**: token-overlap evaluator fails on preference questions (30/30 = 0%)
6. **Single embedding model**: all-MiniLM-L6-v2; results may vary
7. **Single LLM**: deepseek-chat; no cross-model validation
8. **No statistical significance testing**: all comparisons are observational
9. **Reconsolidation real-world effectiveness unproven**: mechanism correct ≠ beneficial in adaptive setting

---

## 7. Conclusion

MemForge V2.1 demonstrates that evidence-grounded memory management can improve downstream QA (+1.8pp overall, +6.6pp abstention) when reconsolidation is added, but that retrieval-relevance-based sufficiency gating is counterproductive. The three contributions—calibrated gate (negative result), temporal soft decay (trade-off characterized), and lineage-based reconsolidation (causally validated)—form a coherent foundation for V3's genome-based memory architecture, where genotype-phenotype separation and evidence-driven mutation may address the remaining limitations.

---

## 8. Failure Taxonomy

| ID | Failure Mode | Observed In |
|----|-------------|-------------|
| F1 | False Forgetting | V1 temporal hard filter (809 gold turns deleted) |
| F2 | False Conflict | V1 cosine conflict (semantic similarity ≠ contradiction) |
| F3 | Future Leakage | V2.1 temporal soft decay (FLR=0.200) |
| F4 | Over-Abstention | V2.1 Gate (6/8 interceptions are answerable FP) |
| F5 | Under-Abstention | V2 grounded (abstention accuracy 0.767) |
| F6 | Reconsolidation Drift | Not yet observed (keyword-based, synthetic only) |
| F7 | Evidence Collapse | Prevented by lineage/versioning |
| F8 | Rich-get-Richer | Not yet measured |

---

## 9. Reproducibility

- **Code**: commit 97e6536, tag v2.1-final
- **Data**: LongMemEval-S cleaned (264.5MB, SHA256 in manifest)
- **Embeddings**: all-MiniLM-L6-v2, cached in `.hf_cache`
- **LLM**: deepseek-chat, temperature=0.0, base_url=https://api.deepseek.com/v1
- **Retrieval inputs**: frozen at `results/longmemeval/stage5_real/retrieved/hybrid/`
- **Raw results**: `results/longmemeval/stage_v21_real/{v21_recon,v21_full}/raw_qa.jsonl`
- **Tests**: 179 passed + 10 skipped

---

## References

- Wu et al. (2024). LongMemEval: Benchmarking Long-Term Memory in Large Language Models.
- Sato et al. (2024). LoCoMo: A Benchmark for Long Context Memory.
- Zhong et al. (2023). MemoryBank: Enhancing Large Language Models with Long-Term Memory.
- Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior.
- Nader (2003). Memory reconsolidation. Current Biology.
