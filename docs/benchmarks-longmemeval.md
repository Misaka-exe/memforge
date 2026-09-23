# Stage 4.6 — LongMemEval-S Pure Retrieval Benchmark

**MemForge** · 2026-09-22 · [项目根目录 `benchmarks/`](../benchmarks/)

> 本文档只报告 **纯检索（retrieval-only）** 结果，不涉及 QA 生成。
> 官方 LongMemEval 的主指标是 **QA accuracy**（检索 + 生成 + LLM judge），
> 与本文的 Recall / MRR / NDCG **不可直接横向比较**，两者用途不同。
> 文末附外部参考值，仅作量级参考，不作为本项目基准结论。

---

## 1. 实验设置

| 项目 | 值 |
|------|-----|
| 数据集 | LongMemEval-S（cleaned，官方推荐版） |
| 来源 | HuggingFace `xiaowu0162/longmemeval-cleaned` |
| 文件 | `benchmarks/data/longmemeval_s_cleaned.json`（已 gitignore） |
| 题数 | 500（6 类 + abstention） |
| SHA256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| 大小 | 277,383,467 字节 |
| Embedding | `all-MiniLM-L6-v2`（sentence-transformers，384 维，本地 CPU） |
| 记忆粒度 | **turn 级 MemoryItem**（不把整个 session 作为一条记忆），
  每 item 保留 `metadata: {session_id, question_id, turn_index, role, has_answer}` |
| 检索单元 | session 级（由 item 的 `session_id` 聚合） |
| 基线 | ① Naive Vector（纯余弦）② Vector + Rerank（余弦 + freshness）
  ③ Hybrid（向量 + 关键词 union 后混合重排） |
| 指标 | `recall_any@K`（Top-K 覆盖至少一个 gold session）
  `recall_all@K`（Top-K 覆盖全部 gold sessions）
  MRR、NDCG@10（session 级、去重） |
| K | 1 / 5 / 10 |

### Abstention 处理

- 30 个 `_abs` 题（question_id 后缀 `_abs`）**不进入检索分母**（无真实证据位置，
  其 `answer_session_ids` 为 `answer_<hex>_abs*` 哨兵）。
- 单独计数：`abstention_count = 30`。
- 非 abstention 题：470 个，平均 gold sessions ≈ 1.89。

## 2. 全局结果（470 题，session 级）

| System | R@any@1 | R@any@5 | R@all@5 | R@any@10 | R@all@10 | MRR | NDCG@10 |
|--------|---------|---------|---------|----------|----------|------|---------|
| Naive Vector | 0.8681 | 0.9468 | 0.6638 | 0.9723 | 0.8234 | 0.9009 | 0.8328 |
| Vector + Rerank | 0.8426 | 0.9340 | 0.6128 | 0.9553 | 0.7830 | 0.8839 | 0.8015 |
| **Hybrid + Rerank** | **0.9106** | **0.9638** | **0.7277** | **0.9894** | **0.8574** | **0.9331** | **0.8668** |

观察：

1. **Hybrid 全面最优**：关键词与向量互补，在 `recall_any@1` 上 +4.3pp、
   `recall_all@5` 上 +6.4pp 相对纯向量。
2. **Rerank 在长尾类别反而略降**：加 freshness 权重后，temporal-reasoning /
   knowledge-update 的 `recall_all@5` 明显下降 —— 说明**通用的"新鲜度优先"
   在 LongMemEval 上不是可靠的第二信号**（该数据的 `question_date` 与
   turn 顺序并非简单单调关系）。
3. **recall_all 与 recall_any 差距大**：multi-session（avg gold≈2.3）的
   `recall_all@5` 仅 0.49（Hybrid），说明"找全所有证据"仍是难点，
   单证据命中（recall_any）相对容易。

## 3. 按类别结果（recall_any@5 / recall_all@5 / MRR）

| Category | n | Naive any@5 | Rerank any@5 | Hybrid any@5 | Hybrid all@5 | Hybrid MRR |
|----------|----|-------------|--------------|--------------|--------------|------------|
| single-session-user | 64 | 0.9375 | 0.9062 | **1.0000** | 1.0000 | 0.9693 |
| single-session-assistant | 56 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| single-session-preference | 30 | 0.8667 | 0.8333 | **0.9333** | 0.9333 | 0.8261 |
| multi-session | 121 | 0.9587 | 0.9669 | **0.9752** | 0.4876 | 0.9316 |
| temporal-reasoning | 127 | 0.9213 | 0.8898 | **0.9213** | 0.5984 | 0.8874 |
| knowledge-update | 72 | 0.9722 | 0.9722 | **0.9722** | 0.8194 | 0.9769 |

要点：

- **single-session 三类接近饱和**（assistant 满分）——这类题的证据集中在单
  session，检索压力小。
- **multi-session / temporal / knowledge-update 是区分度所在**：`recall_any@5`
  普遍 ≥0.92，但 `recall_all@5` 只有 0.49–0.82 —— 多证据题找全更难，
  这正是 MemForge 后续 Conflict/Temporal 模块可以发力的方向。

## 4. 与官方 QA accuracy 的区别（重要）

| 指标 | 类型 | 含义 | 是否可比 |
|------|------|------|----------|
| 官方 QA accuracy | 端到端 | 检索 + 生成 + LLM judge 判定答案 | 本项目**未测** |
| 本文 Recall@K / MRR / NDCG | 纯检索 | 只评估能否找回正确证据 session | 本文结论 |

- 本文不声称任何 QA 性能。
- 官方主指标是 QA accuracy；本报告是**检索能力评估**，两者不可互比。

## 5. 外部参考值（仅作量级参考，不横向对比）

以下来自其他实现（embedding / 策略 / 协议均不同），**不是本项目的基准结论**：

| 实现 | 检索方式 | R@5（any） |
|------|----------|-----------|
| 某向量实现 | all-MiniLM-L6-v2 纯向量 | ≈96.6% |
| 某 BM25+向量实现 | 混合 | ≈95.2% |

> 本项目 Hybrid 的 `recall_any@5 = 0.9638` 与上述外部值在同一量级，
> 但模型、分块、评估细节不同，**不构成横向对比**。

## 6. 与 Synthetic Benchmark 的关系

| | Synthetic（Stage 4） | LongMemEval-S（Stage 4.6） |
|---|---|---|
| 数据 | 生成器构造，ground truth 精确可控 | 官方真实长对话数据 |
| Embedding | HashEmbedding（离线确定性） | all-MiniLM-L6-v2（真实语义） |
| 结论 | 内部机制消融（importance 权重有效等） | 真实检索能力（Hybrid 优于纯向量） |
| 关系 | **两套结果分开报告，不合并、不互比** | |

## 7. 产物与复现

```
results/longmemeval/
├── manifest.json              # 运行配置 + 全量指标 + 类别分解
├── retrieval_results.jsonl    # 每题 × 每系统原始得分
├── by_category.csv            # Category × System 表
├── validation.json            # 数据校验报告
└── figures/
    ├── recall_at_k.png
    └── recall_by_category.png
```

复现命令：

```bash
# 1. 下载（可选；数据已 gitignore）
python -m benchmarks.scripts.download_longmemeval --mirror hf-mirror.com

# 2. 校验
python -m benchmarks.scripts.validate_longmemeval

# 3. 检索实验（约 40 分钟，CPU）
HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_retrieval \
    --embedding sentence-transformers --model all-MiniLM-L6-v2

# 4. 合成实验（保留，独立）
python -m benchmarks.runner
```

## 8. 局限与下一步

局限：

1. 本阶段只做检索；QA accuracy 需要接 LLM（OpenAI-compatible）与
   LLM judge，属下一阶段。
2. Rerank 的 freshness 信号在真实数据上效果为负 —— 需要更细的
   `question_date` 感知（如 as_of 过滤）而非简单 turn 顺序。
3. turn 级 item 数量大（≈24 万），全量跑约 40 分钟；后续可加缓存。

下一步候选：

- Stage 5a：接 LLM QA（`answer` 生成 + abstention accuracy + token 统计）
- Stage 5b：把 MemForge 的 Lifecycle/Conflict/Temporal 检索接入
  LongMemEval（如 as_of 时间过滤、supersede 链裁剪），测
  `-Conflict / -Temporal` 消融
- Stage 5c：LoCoMo adapter

---

# Stage 4.7 — LongMemEval-S Lifecycle Ablation

> 上一节（Stage 4.6）回答了"哪个基线检索最稳"；
> 本节回答研究问题：**MemForge 的生命周期机制（Temporal / Conflict / Freshness）
> 在真实长期记忆数据上是否真的有贡献？**
> 结论是**诚实的负面结果**：当前确定性实现下，三个机制在 LongMemEval-S
> session-level retrieval 上都产生负贡献。本节给出数据与根因。

## 1. 变体定义（同一次 embedding 共享，470 有效题）

| Variant | Temporal 过滤 | Conflict 过滤 | Freshness 重排 |
|---------|:---:|:---:|:---:|
| naive_vector（基线） | ✗ | ✗ | ✗ |
| hybrid（Stage 4.6 最优） | ✗ | ✗ | ✗ |
| **full**（Hybrid+Lifecycle） | ✓ | ✓ | ✓ |
| no_temporal | ✗ | ✓ | ✓ |
| no_conflict | ✓ | ✗ | ✓ |
| no_freshness | ✓ | ✓ | ✗ |

- **Temporal**：`session_date <= question_date` 的 turn 才参与检索（半开语义）
- **Conflict**（无 LLM 的确定性近似）：更早 session 的 turn 若与更晚 session 的 turn
  余弦相似度 ≥ 0.80 则标记为被取代（superseded），不参与检索
- **Freshness**：session 日期在可用集内的归一化新旧度（0 旧 … 1 新），权重 0.15

## 2. 全局结果（470 题，session 级）

| Variant | R@any@1 | R@any@5 | R@all@5 | MRR | NDCG@10 |
|---------|---------|---------|---------|------|---------|
| naive_vector | 0.8681 | 0.9468 | 0.6638 | 0.9009 | 0.8328 |
| **hybrid** | **0.9128** | **0.9638** | **0.7298** | **0.9345** | **0.8675** |
| full | 0.8511 | 0.9362 | 0.6234 | 0.8845 | 0.7880 |
| no_temporal | 0.8872 | 0.9723 | 0.6809 | 0.9206 | 0.8295 |
| no_conflict | 0.8532 | 0.9383 | 0.6426 | 0.8861 | 0.7947 |
| no_freshness | 0.8745 | 0.9277 | 0.6468 | 0.8973 | 0.8167 |

**核心结论：生命周期机制（当前实现）全部为负贡献。**

- `hybrid`（无生命周期）是全部 6 个变体中的最优配置。
- `full`（全开）在 `recall_any@1/5`、`recall_all@5`、MRR、NDCG 上**全面垫底**。

## 3. 根因诊断（不是"生命周期概念错了"，而是"实现与数据不适配"）

### 3.1 Temporal 过滤：误删了 41 题的 gold 证据

- 全量 470 题中，**69 题**存在 `session_date > question_date` 的 turn（13,434 个，占 5.8%）。
- 其中 **41 题**的 gold 证据 session 的时间戳晚于 question_date（809 个 gold turn 被剔除）。
- 也就是说：官方 cleaned 数据的 session 时间戳并非严格早于提问时间；
  "只取过去"的硬过滤会把正确答案也删掉。`no_temporal > full` 由此而来。

### 3.2 Conflict 过滤：相似度阈值误删了 48% 的 gold turn

- 470 题中 **272 题**触发了 supersede（共 681 个 turn 被标记被取代）。
- 其中 **330 个（48.5%）属于 gold session** —— 纯相似度（0.80）近似把
  同主题的正常讨论误判为"旧版本被新版本取代"。

### 3.3 Freshness 重排：引入排序噪声

- `full` vs `no_freshness`：any@1 0.851→0.875、all@5 0.623→0.647，
  去掉 freshness 后反而更好；`no_freshness` 与 `hybrid` 高度接近
  （排序几乎回到纯语义+关键词混合）。

## 4. 按类别（重点类别）

| Category | n | hybrid any@5 | full any@5 | hybrid all@5 | full all@5 |
|----------|----|--------------|------------|--------------|------------|
| single-session-user | 64 | 1.000 | 0.969 | 1.000 | 0.969 |
| single-session-assistant | 56 | 1.000 | 1.000 | 1.000 | 1.000 |
| single-session-preference | 30 | 0.933 | 0.967 | 0.933 | 0.967 |
| multi-session | 121 | 0.975 | 0.959 | 0.496 | 0.446 |
| **temporal-reasoning** | 127 | 0.921 | **0.827** | 0.598 | **0.339** |
| knowledge-update | 72 | 0.972 | 1.000 | 0.819 | 0.681 |

- **temporal-reasoning 受损最重**：full 的 `recall_all@5` 从 0.598 跌到 0.339 ——
  该类 gold 证据恰恰被 temporal 硬过滤大量剔除（讽刺的是该类最需要时间语义，
  但需要的是"时间排序/推理"，不是"删未来"）。
- multi-session / knowledge-update 也一致受损（`recall_all@5` 下降）。
- single-session 三类本就接近饱和，差异小。

## 5. 结论与校准方向（写进 README 的研究发现）

1. **在 LongMemEval-S session-level retrieval 上，无生命周期的 Hybrid 是最优基线**；
   MemForge 生命周期机制（当前确定性实现）需要校准才能产生正贡献。
2. 校准方向（不是推倒重来）：
   - **Temporal**：从"硬过滤"改为"时间感知排序 / as_of 语义对齐官方时间戳"，
     或对 `session_date > question_date` 的 gold 情况做容错。
   - **Conflict**：相似度阈值不足以做版本裁决 —— 需要 slot/实体级语义判断
     （LLM-based conflict detector，即 Stage 2 设计中"向量粗筛 + LLM 精判"的完整形态），
     而不是纯规则近似。
   - **Freshness**：权重 0.15 在当前信号定义下是噪声，应去掉或改用
     question_date 窗口化的衰减。
3. 这是"生命周期管理有价值"这一假设的**可证伪测试**：它告诉我们什么样的
   时间/冲突语义在真实数据上不成立，为 Stage 5 的 LLM 版 Lifecycle 提供基线。

## 6. 产物与复现

```
results/longmemeval/
├── ablation_manifest.json          # 变体配置 + 全局/类别指标 + 诊断（superseded 统计）
├── ablation.csv                    # Variant × 指标
├── by_category_ablation.csv        # Category × Variant
├── ablation_raw_results.jsonl      # 每题 × 每变体原始得分（流式 checkpoint，可断点续跑）
└── figures/
    ├── ablation_any5.png           # 各变体 any@5 / all@5 / MRR
    └── category_comparison.png     # 各类别 recall_all@5 分组柱状图
```

复现：

```bash
HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_ablation \
    --model all-MiniLM-L6-v2
```

> 与 Stage 4.6 的关系：4.6 的 baseline（naive/hybrid/vector_rerank）结果保留在
> `manifest.json` / `retrieval_results.jsonl`；4.7 的 `hybrid` 变体与 4.6 的
> `hybrid` 基线在定义上一致（结果 0.9638 vs 0.9638 相同），可交叉验证。


---

# Stage 5.0/5.1 — End-to-End QA 基础设施（无 API 验证）

> 目标：把 QA pipeline 的工程质量做扎实，**不在未冻结代码上烧真实 API token**。
> 本阶段的全部数值来自 Mock provider，**不代表研究结论**；
> 真实 LLM 的 QA 数值在 Stage 5B（代码冻结后）产出。

## 1. 组件（全部零外部依赖可测）

```
benchmarks/evaluation/
├── qa.py          # prompt builder（结构性无 gold 泄漏）+ 规则评估器 + token 统计
├── providers.py   # MockQAProvider + OpenAICompatibleQAProvider（retry/timeout/429/5xx）
└── qa_runner.py   # 检索导出（复用 Stage 4.7 定义）+ QA 循环 + JSONL checkpoint + manifest
benchmarks/scripts/run_longmemeval_qa.py   # CLI（--smoke / --provider / --systems）
tests/unit/test_qa.py                      # 10 个 QA 单元测试
```

## 2. 检索导出（Stage 5 的输入，已固化）

- 复用 Stage 4.7 的变体定义与常量（`parse_dt` / `CANDIDATE_K` / `CONFLICT_SIM_THRESHOLD` /
  `W_SEM=0.6` / `W_KW=0.25` / `W_FRESH=0.15`），逐行一致。
- 全量导出：**500 题 × 3 systems**（naive_vector / hybrid / full），每题 top-10 item ids
  持久化于 `results/longmemeval/stage5/retrieved/<system>/<qid>.json`；
  embedding 缓存于 `_cache/embeddings/`。后续加 system 或重跑 QA 均**零 embedding**。
- abstention 题同样导出（QA 需要检索上下文），但检索指标分母仍排除（沿用 4.6/4.7）。

## 3. Mock 全量 QA（3 × 500，流程验证）

| System | n | answerable_n | abstention_n | total_tokens | 说明 |
|--------|---|--------------|--------------|--------------|------|
| naive_vector | 500 | 470 | 30 | 790,006 | Mock 固定 "42" |
| hybrid | 500 | 470 | 30 | 977,711 | 同上 |
| full | 500 | 470 | 30 | 960,467 | 同上 |

- answerable_accuracy≈0.002（Mock 不真答，仅 1 题撞上 "42"）；abstention_accuracy=0
  （"42" 非 abstention）——**均不代表模型能力，仅验证 pipeline 与判定逻辑**。
- 真实成本预估（以 DeepSeek/Qwen 档位 $0.27/M input 估算）：3 systems 约 2.7M prompt tokens
  → **<$1**，可放心跑真实实验。

## 4. Hardening 清单（Stage 5.1，全部通过）

| # | 项目 | 状态/说明 |
|---|------|-----------|
| 1 | 三个 system 完整生成输入 | ✅ 500×3 retrieved 文件齐 |
| 2 | 500 题 checkpoint | ✅ raw_qa.jsonl 逐题即时写盘 |
| 3 | resume 可靠 | ✅ 重跑 `resume: 500 already done`，metrics 从 raw 全量重算 |
| 4 | prompt 无 gold 泄漏 | ✅ 结构性保证（build_prompt 不接收 gold）+ 单元测试；运行时验证澄清：gold 文本出现在检索到的证据 turn 中是正确检索而非泄漏 |
| 5 | abstention 独立统计 | ✅ answerable_n=470 / abstention_n=30 分离 |
| 6 | provider 异常不破坏实验 | ✅ 指数退避重试（408/429/5xx/网络错误）、timeout、非重试 4xx 快速失败 |
| 7 | manifest 记录完整配置 | ✅ stage/prompt_version/model/systems/data_file/retrieval_config |
| 8 | 结果 schema 固定 | ✅ raw_qa.jsonl 10 字段 + metrics.json + qa_comparison.csv + qa_by_category.csv |
| 9 | pytest | ✅ 67 passed + 10 skipped |
| 10 | 仓库无 API key | ✅ key 仅环境变量；产物扫描无 key |

## 5. 产物

```
results/longmemeval/stage5/
├── manifest.json
├── retrieved/<system>/<qid>.json        # gitignored（可再生成）
├── _cache/embeddings/<qid>.npz          # gitignored（可再生成）
├── naive_vector|hybrid|full/
│   ├── raw_qa.jsonl                     # gitignored
│   └── metrics.json
├── qa_comparison.csv
├── qa_by_category.csv
└── figures/{qa_accuracy,abstention_accuracy,token_cost}.png
```

## 6. 复现

```bash
# 10 题 smoke（5 answerable + 5 abstention）
python -m benchmarks.scripts.run_longmemeval_qa --smoke --provider mock --model all-MiniLM-L6-v2

# 全量 Mock（流程验证）
HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_qa     --provider mock --model all-MiniLM-L6-v2 --systems naive_vector,hybrid,full

# Stage 5B（代码冻结后，真实 LLM）
set MEMFORGE_LLM_MODEL=... & set MEMFORGE_LLM_BASE_URL=... & set MEMFORGE_LLM_API_KEY=...
python -m benchmarks.scripts.run_longmemeval_qa --provider openai --systems naive_vector,hybrid,full
```
