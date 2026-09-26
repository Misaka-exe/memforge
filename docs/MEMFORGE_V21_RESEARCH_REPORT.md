# MemForge V2.1 Research Report
## Evidence-Grounded Adaptive Memory

> 日期: 2026-09-27 | 版本: v2.1.0 | 状态: Infrastructure Complete, Real LLM Experiment Pending

---

## 1. Research Question

**Can calibrated evidence-aware memory management improve evidence preservation and downstream QA while reducing unsupported memory transitions?**

V2 完成了 evidence-grounded answering（retrieval → grounded answer → verification），但暴露了三个问题：
1. Gate 使用 memory.importance 代理 → 100% SUFFICIENT，未实际工作
2. Temporal soft decay 未被有效验证（LongMemEval-S 无 future-dated gold）
3. Verification FAIL 只是检测信号，没有对应的 memory update 机制

V2.1 解决这三个问题，从 "evidence-grounded answering" 升级为 "evidence-grounded memory management"。

---

## 2. V2.1 Contributions

### 2.1 Retrieval Score Normalization
- 统一的 `RetrievalFeatures` 数据结构（top1, top5_mean, margin, n_retrieved, mean, std）
- 从 V2 已固化的 embedding cache 计算真实 semantic scores（不重新 embedding）
- 500 题 retrieval features 已导出：`results/longmemeval/stage_v21/retrieval_features.jsonl`

### 2.2 Calibrated Sufficiency Gate
- `ThresholdGate`（单阈值 top1_score）
- `MultiFeatureGate`（top1 + top5_mean + n_retrieved 多维规则）
- 支持 calibration（grid search 最优 F1 阈值）
- 评估指标：coverage, answerable_recall, abstention_precision, selective_accuracy, risk-coverage curve

### 2.3 Temporal Stress Benchmark
- 6 类 synthetic temporal cases（Current/Historical/Future/Update/Contradictory/Future Contamination）
- 3 个 baseline：Hard Filter / No Temporal / Soft Decay
- 新指标：Future Leakage Rate (FLR), Evidence Loss Rate (ELR)

### 2.4 Evidence-Grounded Reconsolidation
- Lineage/versioning 数据模型（lineage_id, version, derived_from）
- NEVER overwrite：创建新版本，旧 memory 标记 DEPRECATED
- Slot-level update（仅修改新证据支持的字段）
- 仅在 Verification FAIL 时触发
- 决策：KEEP / UPDATE / ABSTAIN（低 confidence 时 ABSTAIN）

### 2.5 Unified Adaptive Pipeline
- 两种模式：Static（memory frozen，用于 V2 对比）/ Adaptive（question→retrieve→answer→verify→reconsolidate→next）
- `V21Config(enable_gate, enable_temporal, enable_reconsolidation)` 独立消融开关
- 统一指标 `V21Metrics`（retrieval + QA + gate + evidence + memory + temporal + efficiency）

---

## 3. Experimental Protocol

### 3.1 Gate Calibration (Exploratory)
- 数据：500 题 LongMemEval-S retrieval features + V2 QA outcomes
- 标签：is_answerable（470 answerable + 30 abstention）
- 方法：grid search 最优 F1 阈值
- **注意**：在同一数据集上校准和评估，属于 exploratory，不是无偏评估

### 3.2 Temporal Stress Benchmark
- 数据：120 synthetic cases（6 类 × 20/类）
- Baselines：Hard Filter, No Temporal, Soft Decay
- 指标：Recall@1, Recall@5, FLR, ELR

### 3.3 Pipeline Validation (Mock)
- 数据：50 synthetic questions
- 4 配置：V2-Grounded, V2.1-Gate, V2.1-Recon, V2.1-Full
- 使用 Mock LLM，不消耗真实 API
- 目的：验证端到端 pipeline 正确性

### 3.4 Real LLM Experiment (Pending)
- 需要真实 API 调用 500 × 4 = 2000 次
- 当前未执行，等待用户确认预算和时机

---

## 4. Gate Calibration Results (Detailed Audit)

### 4.1 Feature Distribution
| 分组 | n | top1 mean | top1 std | diff |
|------|---|-----------|----------|------|
| Answerable | 470 | 0.540 | 0.139 | — |
| Abstention | 30 | 0.451 | 0.136 | 0.089 |

**结论**：检索信号有弱区分度（diff=0.089），但 std≈0.14，分布重叠严重。

### 4.2 ThresholdGate
- 最优阈值：0.1（F1=0.969）
- 这意味着所有问题的 top1_score > 0.1，因此阈值 0.1 等于不过滤
- **单阈值 top1_score 完全无法区分 answerable 和 abstention**

### 4.3 MultiFeatureGate (Detailed Audit)
- 最优参数：top1=0.3, top5=0.2（F1=0.934）
- **逐题审计结果**：

| 指标 | 值 |
|------|-----|
| Total intercepted | 8/500 |
| Answerable wrongly intercepted (FP) | 6 |
| Abstention correctly intercepted (TP) | 2 |
| Abstention passed through (FN) | 28/30 |
| Answerable correctly passed (TN) | 464 |
| Abstention Precision | 0.250 |
| Abstention Recall | 0.067 |
| Abstention F1 | 0.105 |

### 4.4 Threshold Provenance
- MultiFeatureGate 通过 grid search 在**同一 500 题**上校准
- 最优 F1=0.934 是在 answerable prediction 上计算的，不是 abstention detection
- **警告**：calibration 和 evaluation 在同一数据集上 = evaluation-set tuning
- 这是 EXPLORATORY，不是无偏评估

### 4.5 Risk-Coverage Curve (ThresholdGate)
| Threshold | Coverage | Risk (1-Sel.Acc) | Selective Accuracy | Intercepted |
|-----------|----------|-------------------|--------------------|-------------|
| 0.10 | 1.000 | 0.596 | 0.404 | 0 |
| 0.34 | 0.930 | 0.589 | 0.411 | 35 |
| 0.50 | 0.562 | 0.566 | 0.434 | 219 |
| 0.66 | 0.202 | 0.406 | 0.594 | 399 |
| 0.74 | 0.082 | 0.244 | 0.756 | 459 |

Risk-coverage trade-off 存在：提高阈值可提升 selective accuracy（0.404→0.756），但 coverage 急剧下降（1.0→0.082）。当前信号质量下，需要拦截 90%+ 的问题才能显著降低错误率。

### 4.6 Gate Conclusion
**Calibrated retrieval-score gating is insufficient for evidence sufficiency on LongMemEval-S.**

这不是简单的"threshold 不好"，而是对 V2 Gate 假设的反证：**Retrieval relevance is not equivalent to evidence sufficiency.** 检索相关性高不代表证据充分，检索相关性低也不代表证据不足。当前的 retrieval features（top1, top5_mean, margin, n_retrieved）无法可靠区分。

---

## 5. Temporal Stress Benchmark Results

### 5.1 FLR Definition Fix
原始 FLR=0.250 包含了 T3 Future 类别的 gold future memory（在 T3 中，future memory 是正确答案，不应算 leakage）。修正后只在 current-question categories（current/historical/update/contradictory/future_contamination，共 100 题）上计算：

| System | R@1 | R@5 | FLR (original) | FLR (corrected) | ELR |
|--------|-----|-----|----------------|-----------------|-----|
| Hard Filter | 0.833 | 0.833 | 0.000 | **0.000** | 0.167 |
| No Temporal | 0.500 | 1.000 | 0.250 | 0.200 | **0.000** |
| Soft Decay | **0.833** | **1.000** | 0.250 | 0.200 | **0.000** |

### 5.2 Key Findings
1. **Hard Filter**: 零未来泄漏（FLR=0），但 16.7% gold evidence 被误删（ELR=0.167）——这正是 V1 的问题
2. **No Temporal**: 零证据损失，但 20% 未来泄漏，且 Recall@1 只有 0.5（排序差）
3. **Soft Decay**: 零证据损失 + Recall@5=1.0 + Recall@1=0.833（和 Hard Filter 一样好），但 FLR=0.200（仍有未来泄漏）

### 5.3 Pareto Analysis (Soft Decay Floor Sweep)
Sweep floor 参数 [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]：

**所有 floor 参数结果完全相同**（ELR=0, FLR_corr=0.200, R@1=0.833）。

原因：当前 synthetic benchmark 每个 case 只有 2-3 个 memory，top-5 总是包含所有 memory，因此 floor 不影响排序。**当前 benchmark 太简单，无法区分不同的 decay 参数。** 需要更复杂的 benchmark（更多 memory，top-K < total）才能验证 floor 的影响。

### 5.4 Trade-off
```
Evidence Loss ↔ Future Leakage
Hard Filter:    0.167 loss, 0.000 leakage (but loses gold evidence)
No Temporal:    0.000 loss, 0.200 leakage, bad ranking (R@1=0.5)
Soft Decay:     0.000 loss, 0.200 leakage, good ranking (R@1=0.833) ← best balance
```

**Soft Decay 在保留证据的同时达到了和 Hard Filter 一样的 Recall@1**，证明了 soft temporal scoring 的价值。但 FLR=0.200 说明未来泄漏问题仍未完全解决，且当前 benchmark 不足以找到 Pareto-efficient 参数。

---

## 6. Reconsolidation Design + Synthetic Causal Validation

### 6.1 Data Model
```
Memory A (v1)
  lineage_id = L1
  version = 1
  status = ACTIVE
       │ reconsolidation (Verification FAIL)
       ▼
Memory B (v2)
  lineage_id = L1
  version = 2
  derived_from = A.id
  status = ACTIVE
  supersedes = A.id (set on A)
```

### 6.2 Trigger Conditions
- 仅在 `verification_result == "FAIL"` 时触发
- confidence < min_confidence (0.6) → ABSTAIN（不更新）
- 无 updatable fields → KEEP
- 有新证据支持的字段 → UPDATE

### 6.3 Slot-Level Update
不重写整个 memory，仅修改新证据支持的字段。当前实现支持 location/job/preference/relationship/name 五类 slot（keyword-based detection）。

### 6.4 Synthetic Causal Benchmark
**设计**：30 cases × 5 queries。Initial memory 有 4 slots（name/city/job/hobby），新证据只更新 city slot。测试：
- Current Recall: 新版本中 current city 正确
- Historical Recall: 旧版本通过 lineage 仍可检索
- Unchanged Slot Recall: job/hobby/name 在新版本中保留
- False Update Rate: 错误更新的 slot 比例
- Lineage Completeness: lineage_id/version/derived_from 完整

| Metric | No Reconsolidation | Reconsolidation |
|--------|-------------------|-----------------|
| Current Recall | 0.000 | **1.000** |
| Historical Recall | 0.000 | **1.000** |
| Unchanged Slot Recall | 1.000 | **1.000** |
| False Update Rate | 0.000 | **0.000** |
| Lineage Completeness | 0.000 | **1.000** |
| Overall Accuracy | 0.600 | **1.000** |

**决策分布**：30/30 UPDATE（confidence=0.9 > min_confidence=0.6，且 location slot 匹配）。

### 6.5 Causal Result Interpretation
Reconsolidation infrastructure 在 synthetic causal test 上表现完美：
- ✅ 正确更新 target slot (city)
- ✅ 未破坏任何 unchanged slot (job/hobby/name)
- ✅ 保留 historical version (通过 lineage 可检索)
- ✅ 所有 30 cases 决策正确

**但这是 synthetic + keyword-based slot detection 的结果。真实 LLM 效果（slot extraction、confidence calibration、adaptive memory）尚未验证。** 当前状态：Implemented, Unit-tested, Synthetic-causal validated, Real-world effectiveness: unvalidated.

### 6.6 Evidence Preservation
- `compute_evidence_preservation_rate()`: gold evidence 仍可检索的比例
- `compute_evidence_loss_rate()`: 1 - EPR
- Lineage 确保即使旧 memory 被 DEPRECATED，仍可通过 lineage_id 追溯

---

## 7. Pipeline Validation Results (Mock)

4 配置 × 50 synthetic questions，Mock LLM：

| System | Overall | Answerable | Abstention | Coverage | Recon Triggered |
|--------|---------|------------|------------|----------|-----------------|
| V2-Grounded | 0.900 | 1.000 | 0.000 | 1.000 | 0 |
| V2.1-Gate | 0.900 | 1.000 | 0.000 | 0.900 | 0 |
| V2.1-Recon | 0.900 | 1.000 | 0.000 | 1.000 | 0 |
| V2.1-Full | 0.900 | 1.000 | 0.000 | 0.900 | 0 |

**注意**：这是 synthetic + mock 数据，仅验证 pipeline 端到端正确性，不代表真实性能。Mock verify 总是返回 PASS，因此 reconsolidation 未被触发。真实实验需要 LLM API。

---

## 8. Key Findings

### Finding 1: Retrieval relevance ≠ evidence sufficiency
Gate 详细审计证明：Calibrated retrieval-score gating is insufficient for evidence sufficiency on LongMemEval-S. MultiFeatureGate 只拦截 8/500，其中 6 个是 answerable 误拦（precision=0.25），28/30 abstention 被放行（recall=0.067, F1=0.105）。单阈值完全无效（最优=0.1 等于不过滤）。Risk-coverage curve 显示需要拦截 90%+ 问题才能显著降低错误率。**这是对 V2 Gate 假设的反证，不是简单的 threshold 问题。**

### Finding 2: Soft temporal decay preserves evidence while maintaining retrieval quality
Temporal Stress Benchmark（修正 FLR 定义后）证明 Soft Decay 在 ELR=0 的同时达到 R@1=0.833（与 Hard Filter 相同），R@5=1.0。但 FLR=0.200 说明未来泄漏仍存在。Pareto 分析发现当前 synthetic benchmark 太简单（每 case 2-3 memory，top-5 全包含），无法区分不同 decay 参数。

### Finding 3: Reconsolidation infrastructure validated on synthetic causal test
Synthetic Causal Benchmark（30 cases × 5 queries）证明 slot-level reconsolidation 正确更新 target slot（current recall=1.0）、保留 unchanged slots（unchanged recall=1.0）、保留 historical version（historical recall=1.0）、零错误更新（false update rate=0）。**但真实 LLM 效果尚未验证。**

### Finding 4: V1's negative result guided V2.1's design
V1 发现的 deterministic lifecycle 负贡献直接指导了 V2.1：
- Temporal 硬过滤 → Soft Decay（ELR=0 on stress test）
- Cosine conflict → 4-level uncertainty + ABSTAIN on low confidence
- 无 anti-hallucination → Calibrated Gate（发现 retrieval ≠ sufficiency）
- 无 memory update → Evidence-Grounded Reconsolidation（causal validated）

---

## 9. Limitations

1. **Gate calibration 是 exploratory**：在同一 500 题上校准和评估，无独立 validation set；MultiFeatureGate F1=0.105 for abstention detection
2. **Gate 信号不足**：当前 retrieval features 弱区分度，需要更好的特征（cross-encoder, query-doc interaction）或模型
3. **Temporal Stress Benchmark 是 synthetic**：每 case 仅 2-3 个 memory，top-5 全包含，无法区分 decay 参数；需要更复杂的 benchmark
4. **FLR 定义已修正**：原始 FLR=0.250 包含 T3 Future 的 gold memory，修正后 FLR=0.200（仅 current-question categories）
5. **Reconsolidation slot detection 是 keyword-based**：synthetic causal test 表现完美，但真实 LLM 效果未验证
6. **Mock pipeline 结果不是研究结果**：真实 LongMemEval-S 500 题实验待执行
7. **Reconsolidation 真实效果未验证**：synthetic causal validated ≠ real-world effectiveness
8. **单一 embedding model**：all-MiniLM-L6-v2，结果可能因模型不同而变化
9. **无统计显著性检验**：所有比较均为观察性
10. **Risk-coverage curve 基于 ThresholdGate**：MultiFeatureGate 的 risk-coverage 未单独分析

---

## 10. V2.2 / V3 Roadmap

### V2.2 (短期)
1. **Real LLM experiment**: 500 × 4 配置 on LongMemEval-S
2. **Gate feature engineering**: query-document interaction features, cross-encoder scores
3. **Independent validation set**: 拆分 calibration/evaluation
4. **Future leakage mitigation**: 更精细的 temporal penalty, as-of-aware reranking
5. **LLM-based slot detection**: 替代 keyword-based slot identification

### V3 (中期) — Bio-Inspired Adaptive Memory
1. **Synaptic Plasticity**: 基于 access pattern 动态调整 memory importance
2. **Homeostasis**: 记忆预算控制，自动 DORMANT 低价值记忆
3. **Spreading Activation**: 检索时激活关联记忆
4. **Reconsolidation v2**: 完整的 memory reconsolidation 机制（V2.1 是基础设施）
5. **Evolution**: 多代记忆策略进化实验

---

## 11. Reproducibility

### 代码
- `src/memforge/retrieval/score.py` — RetrievalFeatures
- `src/memforge/retrieval/gate_v2.py` — Calibrated Gate
- `src/memforge/memory/reconsolidation.py` — Reconsolidation
- `src/memforge/evaluation/metrics_v21.py` — Unified metrics
- `src/memforge/evaluation/pipeline_v21.py` — Unified pipeline
- `benchmarks/stress/temporal/benchmark.py` — Temporal Stress Benchmark

### 脚本
- `benchmarks/scripts/export_retrieval_features.py`
- `benchmarks/scripts/gate_calibration.py`
- `benchmarks/scripts/run_v21_experiments.py`

### 结果
- `results/longmemeval/stage_v21/retrieval_features.jsonl`
- `results/longmemeval/stage_v21/gate_calibration.json`
- `results/v21/temporal/stress_benchmark.json`
- `results/v21/experiments/summary.json`

### 文档
- `docs/MEMFORGE_V21_BASELINE.md`
- `docs/MEMFORGE_V21_RESEARCH_REPORT.md`（本文档）
- `docs/MEMFORGE_FAILURE_TAXONOMY.md`

---

## 12. Test Status

- V1: 67 passed + 10 skipped（冻结）
- V2: 136 passed + 10 skipped（冻结）
- **V2.1: 179 passed + 10 skipped**
  - 新增 43 个测试：retrieval_score (4), gate_v2 (9), temporal_stress (9), reconsolidation (11), pipeline_v21 (10)

---

## 13. Artifacts Summary

| 模块 | 状态 | 验证级别 |
|------|------|----------|
| Retrieval Score Normalization | ✅ Complete | Unit test + 500题导出 |
| Calibrated Sufficiency Gate | ✅ Complete | Unit test + detailed audit (8/500, F1=0.105) |
| Temporal Stress Benchmark | ✅ Complete | Unit test + 120 synthetic cases + FLR fix + Pareto |
| Evidence-Grounded Reconsolidation | ✅ Complete | Unit test + synthetic causal benchmark (30 cases) |
| Unified Adaptive Pipeline | ✅ Complete | Unit test + 50题 mock end-to-end |
| Research Audit | ✅ Complete | Gate audit + Temporal audit + Recon causal benchmark |
| Real LongMemEval-S Experiment | ⏳ Pending | 需要真实 API |

### 新增审计结果文件
- `results/longmemeval/stage_v21/gate_audit.json` — Gate 详细审计（逐题 decision, confusion matrix, risk-coverage curve）
- `results/v21/temporal/stress_audit.json` — Temporal FLR 修正 + Pareto 分析
- `results/v21/reconsolidation/causal_benchmark.json` — Reconsolidation causal benchmark

---

*报告结束。V2.1 基础设施完成，真实 LLM 实验待执行。*
