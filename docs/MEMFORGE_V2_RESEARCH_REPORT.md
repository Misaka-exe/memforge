# MemForge v2 Research Report
## Evidence-Grounded Memory Management

> 日期: 2026-09-27 | 版本: v2.0 | 状态: 实验完成，待 Gate 校准后定稿

---

## 1. Research Question

**Can evidence-grounded memory management improve downstream QA without changing the underlying retrieval results?**

v1 的实验已经证明：确定性生命周期规则（temporal 硬过滤、cosine conflict、freshness）在 LongMemEval-S 上存在负贡献。v2 的目标不是加入更多机制，而是**修复 v1 已实验证明有问题的机制**，建立可验证的 transition / uncertainty / grounding 基础设施。

v2 保持检索输入完全不变（复用 v1 Stage 5B 已固化的 hybrid 检索结果），只改变回答管道（grounded prompt + citation + verification + gate），从而隔离"检索质量"和"回答质量"两个变量。

---

## 2. V1 Findings (Motivation for V2)

v1 Stage 4.7 + 5B 的实验结果直接指导了 v2 的架构选择：

| 问题 | v1 表现 | 根因 |
|------|---------|------|
| Temporal 硬过滤 | 41 题的 809 个 gold turns 被"只取过去"剔除 | 未来/过期记忆被完全删除 |
| Cosine conflict | 681 个被取代 turn 中 48.5% 属于 gold session | similarity ≠ contradiction |
| Freshness | 权重 0.15 使排序质量下降 | 时间近不等于相关 |
| 无 anti-hallucination | LLM 总是被要求回答 | 无证据充分性评估 |
| 无 citation grounding | 无法验证答案事实是否被支持 | 纯文本回答 |

**核心结论**：Memory lifecycle itself can introduce errors. v2 的设计原则是：Don't change memory unless evidence supports it.

---

## 3. V2 Design

### 3.1 Temporal Soft Scoring

v1：`memory_active_at(m, t)` 返回 False → 记忆被完全移除
v2：`TemporalScorer.score(m, t)` 返回 [floor, 1.0] → 记忆被降权但保留

四种可替换策略：
- **NoDecayScorer**：始终 1.0（v1 默认行为，向后兼容）
- **WindowScorer**：窗口内 1.0，窗口外 out_of_window（默认 0.1）
- **ExponentialDecayScorer**：窗口内 1.0，窗口外指数衰减
- **LinearDecayScorer**：窗口内 1.0，窗口外线性衰减

向后兼容：`UtilityReranker` 不设置 `temporal_scorer` 时，`as_of` 仍然执行 v1 硬过滤。

### 3.2 Conflict Uncertainty

v1：cosine > 0.75 → LLM 判断 → conflict + conf > 0.7 → 旧记忆 DEPRECATED
v2：cosine > threshold → LLM 判断 → ConflictClassifier 4 级分类 →

| 决策 | 行为 |
|------|------|
| NO_CONFLICT | 新记忆 ACTIVE |
| CONFIRMED_CONFLICT (conf ≥ threshold) | 旧 DEPRECATED，新 ACTIVE，SUPERSEDES edge |
| POSSIBLE_CONFLICT | NOOP：记录事件，不改变状态（v3 reconsolidation hook） |
| UNCERTAIN | NOOP：记录事件，不改变状态（宁可不更新也不覆盖好记忆） |

所有决策记录到 `TransitionLog`，包含 memory_id, from_state, to_state, reason, evidence_ids, confidence, decision_source, destructive。

### 3.3 Retrieval Sufficiency Gate

在调用 LLM 之前评估检索证据的充分性：
1. 无检索结果 → INSUFFICIENT
2. 相关记忆数量 < min_relevant_count → INSUFFICIENT
3. 平均相关性 ≥ threshold + band → SUFFICIENT
4. 平均相关性 ≤ threshold - band → INSUFFICIENT
5. 否则 → UNCERTAIN

INSUFFICIENT 时直接 abstain，**不调用 LLM**（节省 token，减少幻觉）。

**当前状态**：Gate 已实现（implemented），但**尚未验证（not yet validated）**。详见第 8 节 Gate Analysis。

### 3.4 Citation Grounding

Prompt 要求 LLM 返回结构化 JSON（使用 OpenAI JSON mode）：
```json
{
  "answer": "...",
  "citations": [{"memory_index": 1, "claim": "..."}],
  "abstain": false
}
```

规则：
1. 只能使用检索记忆中的信息
2. 每个事实声明必须附带 citation（指向记忆编号）
3. 证据不足时 abstain=true, answer="", citations=[]
4. 不允许包含未被记忆支持的信息

### 3.5 Post-hoc Verification

确定性 token-overlap 验证（无 LLM 调用，可复现）：
1. 将答案分成句子（claims）
2. 对每个 claim，计算与所有 evidence 的 content token overlap ratio
3. overlap ≥ support_threshold (0.3) → supported
4. 统计 unsupported_claim_rate
5. 检查 citations 是否指向存在的记忆
6. 简单否定词矛盾检测

判定：PASS (rate=0) / WARN (rate≤0.3) / FAIL (rate>0.3)

---

## 4. Experimental Protocol

| 项目 | 配置 |
|------|------|
| 数据集 | LongMemEval-S cleaned（500 题：470 answerable + 30 abstention） |
| 检索输入 | v1 Stage 5B 已固化的 hybrid 检索结果（Top-10） |
| Embedding | all-MiniLM-L6-v2（v1 已固化，v2 不重新 embedding） |
| LLM | deepseek-chat，temperature=0.0，JSON mode |
| 评估 | v1 确定性 token-overlap evaluator（保持一致性） |
| 总运行时间 | 404 秒（500 题） |

**关键控制变量**：v1 Hybrid 和 v2 Grounded 使用**完全相同的检索输入**，因此 QA 差异完全来自回答管道（grounded prompt + citation + verification），而非检索质量。

---

## 5. Main Results

### 5.1 Primary Result (500 questions)

| 指标 | v1 Hybrid | v2 Grounded | Δ |
|------|-----------|-------------|---|
| n | 500 | 500 | — |
| Overall QA | 0.364 | **0.404** | +0.040 (+11%) |
| Answerable QA | 0.326 | **0.381** | +0.055 (+17%) |
| Abstention QA | **0.967** | 0.767 | -0.200 |
| Abstention Rate | ~0.06 | 0.242 | +0.182 |
| Avg Latency | 779ms | 825ms | +46ms |

**表述**：在 LongMemEval-S 500-question evaluation 中，当前 v2 Grounded pipeline 的 QA accuracy 从 v1 Hybrid 的 0.364 提升至 0.404。这是配对比较的观察结果，尚未做统计显著性检验（paired bootstrap / McNemar），因此不表述为"v2 显著优于 v1"。

### 5.2 By Category

| Category | n | v1 Hybrid | v2 Grounded | Δ |
|----------|---|-----------|-------------|---|
| single-session-user | 64 | 0.797 | 0.750 | -0.047 |
| single-session-assistant | 56 | 0.804 | **0.821** | +0.017 |
| knowledge-update | 72 | **0.528** | 0.472 | -0.056 |
| multi-session | 121 | 0.091 | **0.182** | +0.091 (+100%) |
| temporal-reasoning | 127 | 0.063 | **0.228** | +0.165 (+262%) |
| single-session-preference | 30 | 0.000 | 0.000 | 0 |
| abstention | 30 | **0.967** | 0.767 | -0.200 |

### 5.3 Development/Smoke Result (30 questions, exploratory only)

开发期间的 30 题 smoke test 结果：v1=0.733, v2=0.633。此结果方差大（n=30），**仅作为开发期 exploratory result，不作为主结果**。主结果以 500 题全量实验为准。

---

## 6. Retrieval Results

**V1 = V2**：v2 benchmark 复用 v1 Stage 5B 已固化的 hybrid 检索输入，未重新 embedding 或检索。因此检索指标（Recall@1/5/10, MRR, NDCG）与 v1 Hybrid 完全相同。

这是一个有意的实验设计：保持检索不变，隔离回答管道的影响。

---

## 7. Evidence Loss Rate (ELR)

**ELR = 0**。

**解释**：LongMemEval-S cleaned 数据集中没有 future-dated gold session，因此 temporal soft decay 没有机会表现出 v1 hard temporal filtering 的优势（v1 中 809 gold turns 被硬过滤删除是在 v1 自己的检索管道中，而非 LongMemEval-S cleaned 数据本身）。

**这是 benchmark limitation，不是 temporal mechanism 已经被验证**。当前 LongMemEval-S evaluation does not sufficiently stress temporal evidence preservation。需要专门构造包含 future-dated gold evidence 的 stress test 才能验证 temporal soft decay 的价值。

---

## 8. Gate Analysis (专项审计)

### 8.1 当前 Gate 行为

500 题中：
- Gate SUFFICIENT: 500 (100%)
- Gate INSUFFICIENT: 0 (0%)
- Gate UNCERTAIN: 0 (0%)

**结论**：当前 Gate 配置下，所有问题都被判定为 SUFFICIENT，Gate 实际上没有改变系统行为。Gate 目前是 **implemented，而不是 validated**。

### 8.2 根因分析

当前 Gate 实现使用 `memory.importance`（默认 0.8）作为代理分数，而非真实的检索相关性分数。由于所有 Memory 对象的 importance 都是默认值 0.8，Gate 的 `avg_relevance` 始终 ≥ sufficiency_threshold（0.5），因此全部判定 SUFFICIENT。

### 8.3 检索信号可分性分析

使用已有的 embedding cache 计算 top-1 cosine similarity：

| 分组 | n | top-1 mean | top-1 std | top-5 mean | margin |
|------|---|-----------|-----------|-----------|--------|
| Answerable | 200 (sample) | 0.5165 | 0.1265 | 0.4391 | 0.0546 |
| Abstention | 30 | 0.4510 | 0.1357 | 0.3862 | 0.0486 |

差异：answerable - abstention = 0.0654，但两组标准差均约 0.13，**分布重叠严重**。

**结论**：检索信号有弱区分度，但不能简单用单一阈值切分。如果要校准 Gate，需要：
1. 使用真实检索相关性分数（而非 memory.importance 代理）
2. 在独立 validation set 上校准阈值
3. 考虑使用多维信号（top-1 + top-5 mean + margin + n_retrieved）而非单一分数

当前 500 题只能用于 exploratory calibration，不能同时作为无偏的最终 gate evaluation。

### 8.4 Abstention 行为分析

| 行为 | 数量 | 比例 |
|------|------|------|
| Abstention 问题，模型正确 abstain | 23/30 | 76.7% |
| Abstention 问题，模型错误回答 | 7/30 | 23.3% |
| Answerable 问题，模型错误 abstain | 98/470 | 20.9% |
| Answerable 问题，模型回答 | 372/470 | 79.1% |

v2 的 abstention accuracy 从 0.967 降至 0.767，直接对应于 30 个 abstention 问题中有 7 个被错误回答（23.3%）。

同时，470 个 answerable 问题中有 98 个被错误 abstain（20.9%）。这一现象不影响 abstention accuracy，但会限制 answerable QA 的上限。

当前 Gate 100% 判定为 SUFFICIENT，因此这些 abstention 决策均不是由 Gate 实际拦截产生的，而主要反映当前 grounded QA pipeline 自身的回答/拒答行为。后续需要通过 Gate 校准和独立 validation set 分析，区分"证据不足导致的合理 abstention"和"模型过度 abstain"。

### 8.5 Verification 结果

| 分组 | PASS | WARN | FAIL | None (abstained) |
|------|------|------|------|-------------------|
| Answerable | 315 | 10 | 47 | 98 |
| Abstention | 1 | 1 | 5 | 23 |

- Answerable 答案中，67.0% 通过验证（PASS），10.0% 失败（FAIL）
- FAIL 的答案包含未被证据支持的声明，可作为 v3 reconsolidation 的触发条件

---

## 9. V2-Specific Metrics

| 指标 | 值 |
|------|-----|
| Gate SUFFICIENT rate | 1.000 |
| Gate INSUFFICIENT rate | 0.000 |
| Avg citations per answer | 1.568 |
| Verification PASS rate | 0.632 |
| Verification WARN rate | 0.264 |
| Verification FAIL rate | 0.104 |

---

## 10. Key Findings

### Finding 1: Evidence-grounded generation improves QA with retrieval held constant

在检索输入完全相同的情况下，v2 Grounded 的 answerable QA 比 v1 Hybrid 高 5.5pp（+17%）。观察到的 QA 提升与 grounded answering pipeline（citation + 结构化 prompt）一致；但由于当前实验未对这些组件进行单独消融，该提升的具体机制仍需进一步验证。

### Finding 2: Retrieval quality alone does not determine QA

v2 的检索与 v1 完全相同，但 QA 不同。这进一步验证了 v1 的发现：retrieval metrics 和最终 QA 并不是简单的一一对应关系。LLM 的 reasoning、context interpretation、prompt 结构都会影响最终结果。

### Finding 3: Gate is implemented but not validated

当前 Gate 配置下 100% SUFFICIENT，实际未发挥 anti-hallucination 作用。根因是使用 memory.importance 代理而非真实检索分数。检索信号有弱区分度（diff=0.065），但分布重叠严重，需要在独立 validation set 上校准多维信号。

### Finding 4: Abstention is a trade-off

v2 的 abstention accuracy 从 0.967 下降到 0.767。Grounded prompt 提高了回答率但也增加了错误回答（abstention 问题中 23.3% 被错误回答），同时 answerable 问题中 20.9% 被错误 abstain。这是一个需要校准的 trade-off。

### Finding 5: Verification provides actionable signal

10.0% 的 answerable 答案被当前 heuristic verifier 判定为 FAIL，表明其中存在未被当前证据支持的声明。这类 FAIL 可作为后续 reconsolidation 或补充检索机制的候选触发信号。

---

## 11. Limitations

1. **Gate 未实际发挥作用**：阈值需校准，当前 100% SUFFICIENT
2. **Abstention accuracy 下降**：grounded prompt 让模型更倾向于回答，需要更好的 abstention 训练
3. **Temporal soft decay 未在检索实验中验证**：v2 benchmark 使用 v1 已固化检索输入，soft decay 的检索效果需要 v2.1 验证
4. **ELR=0 是 benchmark limitation**：LongMemEval-S 不包含 future-dated gold evidence，无法验证 temporal evidence preservation
5. **Verification 是确定性 heuristic**：token-overlap 无法检测语义等价但措辞不同的支持关系
6. **单一 LLM**：仅使用 deepseek-chat，结果可能因模型能力不同而变化
7. **single-session-preference 仍为 0%**：v1 的 token-overlap evaluator 不适用开放式偏好描述
8. **无统计显著性检验**：500 题配对比较尚未做 paired bootstrap / McNemar
9. **无 token 统计**：v2 脚本未记录 prompt/completion tokens（JSON mode usage 未解析）
10. **30 题 smoke 与 500 题结果方向不一致**：30 题方差大，仅作 exploratory

---

## 12. V2.1 Roadmap (短期)

1. **Gate 校准**：
   - 使用真实检索相关性分数（而非 memory.importance 代理）
   - 在独立 validation set 上校准多维信号（top-1 + top-5 mean + margin + n_retrieved）
   - 冻结阈值后在 evaluation set 上评估
2. **Temporal stress benchmark**：构造包含 future-dated gold evidence 的测试集，验证 temporal soft decay
3. **Paired statistical testing**：对 500 题配对比较做 paired bootstrap / McNemar
4. **Token 统计**：解析 OpenAI JSON mode 的 usage 字段
5. **Evidence Loss Rate 正式实现**：对比 memory management 前后的 gold evidence 可检索性

---

## 13. V3 Roadmap (中期)

Bio-Inspired Adaptive Memory：
- **Reconsolidation**：基于 verification FAIL 触发记忆更新/补充检索
- **Synaptic Plasticity**：基于 access pattern 动态调整 memory importance
- **Homeostasis**：记忆预算控制，自动 DORMANT 低价值记忆
- **Spreading Activation**：检索时激活关联记忆
- **Evolution**：多代记忆策略进化实验

---

## 14. V1 → V2 → V3 Research Story

```
V1: Stateful Memory
  │
  ├── 发现：确定性生命周期规则存在负贡献
  │   ├── temporal 硬过滤误删 gold evidence
  │   ├── cosine conflict 误判同主题讨论
  │   └── freshness 引入排序噪声
  │
  ▼
V2: Evidence-Grounded Memory
  │
  ├── 原则：Don't change memory unless evidence supports it
  ├── 机制：
  │   ├── temporal 硬过滤 → 软评分（可配置策略）
  │   ├── conflict 二元判断 → 4 级不确定性（UNCERTAIN 不破坏）
  │   ├── 无 anti-hallucination → Sufficiency Gate（implemented, not validated）
  │   └── 纯文本回答 → Citation Grounding + Post-hoc Verification
  │
  ├── 结果：
  │   ├── 检索不变，answerable QA +17%
  │   ├── temporal-reasoning +262%
  │   └── Gate / abstention 需要校准
  │
  ▼
V3: Bio-Inspired Memory
  │
  └── 问题：Can biological mechanisms make memory transitions adaptive?
```

---

## 15. Test Status

- **v1 测试**：67 passed + 10 skipped（冻结，未修改）
- **v2 新增测试**：69 个
  - test_temporal_soft.py（10）
  - test_v2_models.py（14）
  - test_qa_v2.py（8）
  - test_conflict_v2.py、test_grounding.py、test_sufficiency_gate.py、test_v2_pipeline.py（已有）
- **总计**：136 passed + 10 skipped

---

## 16. Artifacts

### 代码
- `src/memforge/retrieval/temporal.py` — TemporalScorer + 4 种策略
- `src/memforge/retrieval/reranker.py` — 集成 temporal_scorer
- `src/memforge/retrieval/gate.py` — SufficiencyGate
- `src/memforge/memory/conflict.py` — Conflict v2 + ConflictClassifier
- `src/memforge/memory/grounding.py` — Citation + GroundedAnswer
- `src/memforge/memory/transition.py` — MemoryTransition + TransitionLog
- `src/memforge/verification/verifier.py` — Verifier + VerificationResult
- `benchmarks/evaluation/qa_v2.py` — v2 QA Pipeline
- `benchmarks/scripts/run_longmemeval_qa_v2.py` — v2 Benchmark 脚本
- `benchmarks/scripts/gate_calibration_check.py` — Gate 专项分析脚本

### 实验结果
- `results/longmemeval/stage5_v2/raw_qa.jsonl` — 500 题原始结果
- `results/longmemeval/stage5_v2/metrics.json` — 聚合指标
- `results/longmemeval/stage5_v2/gate_analysis.json` — Gate 专项分析
- `results/longmemeval/stage5_real/` — v1 实验结果（冻结，未修改）

### 文档
- 本报告
- `D:\agentmemory\MemForge-v2-操作记录.md` — 开发操作记录

---

*报告结束。MemForge v2 实验完成，Gate 校准后可定稿。*
