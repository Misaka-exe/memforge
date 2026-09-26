# MemForge Failure Taxonomy

> 版本: v2.1 | 日期: 2026-09-27

本文档系统化分类 MemForge 项目中观察到的 memory management failure modes。每个 failure mode 包含定义、检测方法、在 V1/V2/V2.1 中的表现，以及缓解方向。

---

## F1: False Forgetting (错误遗忘)

**定义**: 有价值的 memory 被删除、降低到不可检索，或被错误地标记为 DEPRECATED。

**根因**:
- 确定性 temporal 硬过滤将 future-dated 但正确的 memory 移除
- Conflict detection 将同主题讨论误判为 supersession
- Forgetting 策略过度激进

**V1 表现**: Stage 4.7 发现 41 题的 809 个 gold turns 被 temporal 硬过滤删除。

**V2 缓解**: Temporal Soft Decay（4 种策略），不再硬删除，改为降权。

**V2.1 验证**: Temporal Stress Benchmark 显示 Hard Filter ELR=0.167，Soft Decay ELR=0.0。

**检测指标**: Evidence Loss Rate (ELR), Evidence Preservation Rate (EPR)

**缓解方向**: 软评分替代硬过滤；uncertain conflict 不破坏旧 memory；reconsolidation 用 lineage 而非 overwrite。

---

## F2: False Conflict (错误冲突)

**定义**: 两个相关但不矛盾的 memory 被误认为冲突，导致旧 memory 被 DEPRECATED。

**根因**:
- Cosine similarity 高 ≠ 语义矛盾
- 同主题的不同方面被误判为 supersession

**V1 表现**: 681 个被取代 turn 中 48.5% 属于 gold session。

**V2 缓解**: Conflict v2 4 级分类（NO_CONFLICT/POSSIBLE/CONFIRMED/UNCERTAIN），UNCERTAIN 不破坏。

**检测指标**: False Conflict Rate, Memory Update Precision

**缓解方向**: 从 similarity-based 升级为 evidence-based contradiction detection；需要 LLM 或结构化推理。

---

## F3: Future Leakage (未来泄漏)

**定义**: 未来时间的 memory 被检索出来支持当前问题的答案。

**根因**:
- 无 temporal 过滤时，future evidence 与 current evidence 混合
- Soft decay 保留了 future memory（虽然降权），仍可能进入 top-K

**V2.1 验证**: Temporal Stress Benchmark:
- Hard Filter: FLR=0.000（零泄漏但有证据损失）
- No Temporal: FLR=0.250
- Soft Decay: FLR=0.250（仍有泄漏）

**检测指标**: Future Leakage Rate (FLR)

**缓解方向**: 更精细的 temporal scoring；as-of 时间感知的 reranking；future evidence 单独标记。

---

## F4: Over-Abstention (过度拒答)

**定义**: 有足够证据支持回答时，系统却选择 abstain。

**根因**:
- Gate 阈值过高
- Grounded prompt 让模型过于保守
- Retrieval 信号弱区分度导致 gate 误判

**V2 表现**: 470 个 answerable 问题中 98 个（20.9%）被模型错误 abstain。

**V2.1 发现**: Gate calibration 显示单阈值 top1_score 完全无法区分（最优阈值=0.1 等于不过滤）；多特征 gate 只能拦截 8/500 且 abstention precision=0.25。

**检测指标**: Answerable Recall, Over-Abstention Rate

**缓解方向**: 更好的 retrieval features；独立 validation set 校准；risk-coverage curve 分析。

---

## F5: Under-Abstention (拒答不足)

**定义**: 证据不足时，系统仍然回答（产生幻觉或 unsupported claims）。

**根因**:
- Gate 未实际拦截（V2: 100% SUFFICIENT）
- LLM 倾向于回答而非承认不知道
- Grounded prompt 没有强制 abstain

**V2 表现**: 30 个 abstention 问题中 7 个（23.3%）被错误回答。

**检测指标**: Abstention Accuracy, Unsupported Claim Rate, Verification FAIL Rate

**缓解方向**: Calibrated gate；prompt 中强化 abstain 指令；verification 后处理。

---

## F6: Reconsolidation Drift (重巩固漂移)

**定义**: Memory 更新后，新内容偏离原始 evidence，引入了未被支持的信息。

**根因**:
- LLM 在 reconsolidation 过程中添加了推断内容
- Slot-level merge 不准确
- 多轮更新累积误差

**V2.1 状态**: 基础设施已完成（lineage/versioning），但尚未在真实数据上验证。第一版仅在 Verification FAIL 时触发，且 confidence < min_confidence 时 ABSTAIN。

**检测指标**: Reconsolidation Accuracy, False Update Rate, Evidence Preservation Rate after update

**缓解方向**: 严格 slot-level update；保留原始 content；verification 对新 memory 再次检查。

---

## F7: Evidence Collapse (证据崩塌)

**定义**: 一次 memory 更新导致历史证据无法恢复，破坏了可追溯性。

**根因**:
- Overwrite memory 而非 versioning
- DEPRECATED memory 被物理删除
- Lineage 断裂

**V2.1 设计原则**: NEVER overwrite。使用 lineage_id/version/derived_from，旧 memory 标记 DEPRECATED 但保留。

**检测指标**: Lineage Completeness, Historical Recall

**缓解方向**: Append-only event log；lineage 索引；DEPRECATED 不删除。

---

## F8: Rich-get-Richer (富者愈富)

**定义**: 频繁访问的 memory 获得越来越高的权重/importance，导致检索偏差。

**根因**:
- access_count 直接影响 utility score
- 无衰减机制平衡访问频率

**V1/V2 状态**: access_count 存在但未直接用于 reranking（utility score 包含 importance/confidence/relevance，不含 access_count）。

**检测指标**: Retrieval Diversity, Top-K Frequency Distribution

**缓解方向**: V3 Homeostasis（记忆预算控制）；exploration-exploitation 平衡。

---

## Failure Mode 交叉关系

```
F1 False Forgetting ←── F2 False Conflict
       │                    │
       ▼                    ▼
  Evidence Loss      Wrong Memory Active
       │                    │
       └────────┬───────────┘
                ▼
         F4 Over-Abstention
         (no evidence → refuse)
                │
                ▼
         F5 Under-Abstention
         (weak evidence → answer)
                │
                ▼
         F6 Reconsolidation Drift
         (update with bad info)
                │
                ▼
         F7 Evidence Collapse
         (lose history)

F8 Rich-get-Richer: 独立的长期偏差
```

---

## V1 → V2 → V2.1 缓解进展

| Failure | V1 | V2 | V2.1 |
|---------|----|----|------|
| F1 False Forgetting | ❌ 809 gold turns lost | ✅ Soft decay | ✅ ELR=0 on stress test |
| F2 False Conflict | ❌ 48.5% false supersession | ✅ 4-level uncertainty | ⏳ Infrastructure ready |
| F3 Future Leakage | N/A | ⚠️ Soft decay retains future | ✅ FLR metric defined, 0.250 measured |
| F4 Over-Abstention | N/A | ❌ 20.9% over-abstain | ⚠️ Gate weak (exploratory) |
| F5 Under-Abstention | N/A | ❌ 23.3% under-abstain | ⚠️ Gate not validated |
| F6 Reconsolidation Drift | N/A | N/A | ✅ Infrastructure, not yet validated |
| F7 Evidence Collapse | N/A | N/A | ✅ Lineage/versioning design |
| F8 Rich-get-Richer | N/A | N/A | ⏳ V3 Homeostasis |

---

## 开放问题

1. F4/F5 的 trade-off 如何最优平衡？需要 risk-coverage curve 分析。
2. F3 Future Leakage 在 soft decay 下仍为 0.25，是否需要更激进的 future penalty？
3. F6 Reconsolidation Drift 需要真实 LLM 实验验证。
4. F2 False Conflict 的 evidence-based detection 需要 LLM 推理，成本如何控制？
5. 所有 failure mode 之间的交互效应尚未系统研究。
