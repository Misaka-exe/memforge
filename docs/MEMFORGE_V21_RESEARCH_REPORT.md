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

## 4. Gate Calibration Results

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

### 4.3 MultiFeatureGate
- 最优参数：top1=0.3, top5=0.2（F1=0.934）
- Coverage: 0.984（8/500 被 INSUFFICIENT 拦截）
- Answerable Recall: 0.987
- Abstention Precision: 0.25（8 个被拦截中只有 2 个是真正 abstention）
- Abstention Accuracy: 0.067（30 个 abstention 中只有 2 个被正确拦截）

### 4.4 Gate Finding
**当前 retrieval features 不足以可靠地做 sufficiency gate。** 这是一个重要的 negative result：
- 单阈值完全无效
- 多特征 gate 只能微弱区分，且大部分拦截是错误的（abstention precision=0.25）
- 需要更好的 retrieval features（如 query-document interaction features）或独立 validation set

---

## 5. Temporal Stress Benchmark Results

| System | Recall@1 | Recall@5 | FLR | ELR |
|--------|----------|----------|-----|-----|
| Hard Filter | 0.833 | 0.833 | **0.000** | 0.167 |
| No Temporal | 0.500 | 1.000 | 0.250 | **0.000** |
| Soft Decay | **0.833** | **1.000** | 0.250 | **0.000** |

### 5.1 Key Findings
1. **Hard Filter**: 零未来泄漏（FLR=0），但 16.7% gold evidence 被误删（ELR=0.167）——这正是 V1 的问题
2. **No Temporal**: 零证据损失，但 25% 未来泄漏，且 Recall@1 只有 0.5（排序差）
3. **Soft Decay**: 零证据损失 + Recall@5=1.0 + Recall@1=0.833（和 Hard Filter 一样好），但 FLR=0.25（仍有未来泄漏）

### 5.2 Trade-off
```
Evidence Preservation ↔ Future Leakage
Hard Filter:    0 loss, 0 leakage (but loses gold)
No Temporal:    0 loss, high leakage, bad ranking
Soft Decay:     0 loss, high leakage, good ranking ← best balance
```

**Soft Decay 在保留证据的同时达到了和 Hard Filter 一样的 Recall@1**，证明了 soft temporal scoring 的价值。但 FLR=0.25 说明未来泄漏问题仍未完全解决。

### 5.3 By Category (Soft Decay)
- Current: Recall@1=1.0
- Historical: Recall@1=1.0
- Future: Recall@1=0.0（future fact 在 query time 之前，soft decay 降权到最低）
- Update: Recall@1=1.0
- Contradictory: Recall@1=1.0
- Future Contamination: Recall@1=1.0（current fact 排第一）

---

## 6. Reconsolidation Design

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
不重写整个 memory，仅修改新证据支持的字段。当前实现支持 location/job/preference/relationship/name 五类 slot。

### 6.4 Evidence Preservation
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

### Finding 1: Retrieval features have weak separability for sufficiency gate
Answerable vs abstention 的 top1_score 差异仅 0.089（std≈0.14），单阈值完全无效，多特征 gate 的 abstention precision 仅 0.25。**当前 retrieval signal 不足以可靠区分"证据充分"和"证据不足"。**

### Finding 2: Soft temporal decay preserves evidence while maintaining retrieval quality
Temporal Stress Benchmark 证明 Soft Decay 在 ELR=0 的同时达到 Recall@1=0.833（与 Hard Filter 相同），Recall@5=1.0。但 FLR=0.25 说明未来泄漏仍需进一步控制。

### Finding 3: Reconsolidation infrastructure is ready but needs real LLM validation
Lineage/versioning/slot-level update 框架已完成，单元测试覆盖。但真实效果需要在 LongMemEval-S 上用真实 LLM 验证。

### Finding 4: V1's negative result guided V2.1's design
V1 发现的 deterministic lifecycle 负贡献直接指导了 V2.1：
- Temporal 硬过滤 → Soft Decay
- Cosine conflict → 4-level uncertainty + ABSTAIN on low confidence
- 无 anti-hallucination → Calibrated Gate（虽然当前信号不足）
- 无 memory update → Evidence-Grounded Reconsolidation

---

## 9. Limitations

1. **Gate calibration 是 exploratory**：在同一 500 题上校准和评估，无独立 validation set
2. **Gate 信号不足**：当前 retrieval features 弱区分度，需要更好的特征或模型
3. **Temporal Stress Benchmark 是 synthetic**：与真实 LongMemEval-S 的关系需要进一步验证
4. **Reconsolidation 未在真实数据上验证**：slot detection 是 keyword-based，需要 LLM 提升
5. **Mock pipeline 结果不是研究结果**：真实 LongMemEval-S 500 题实验待执行
6. **FLR=0.25 未解决**：Soft Decay 保留了 future memory，仍有泄漏
7. **单一 embedding model**：all-MiniLM-L6-v2，结果可能因模型不同而变化
8. **无统计显著性检验**：所有比较均为观察性

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
| Calibrated Sufficiency Gate | ✅ Complete | Unit test + exploratory calibration |
| Temporal Stress Benchmark | ✅ Complete | Unit test + 120 synthetic cases |
| Evidence-Grounded Reconsolidation | ✅ Complete | Unit test (mock, no real LLM) |
| Unified Adaptive Pipeline | ✅ Complete | Unit test + 50题 mock end-to-end |
| Real LongMemEval-S Experiment | ⏳ Pending | 需要真实 API |

---

*报告结束。V2.1 基础设施完成，真实 LLM 实验待执行。*
