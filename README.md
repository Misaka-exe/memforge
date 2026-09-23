# MemForge

**An Adaptive Memory Management Framework for LLM Agents**

MemForge 不是又一个"对话 → embedding → 向量库 → Top-K"的记忆库，而是把记忆当作**有生命周期的实体**来管理的框架：

```
Remember → Retrieve → Update → Consolidate → Forget
```

## Why?

大多数 Agent Memory 系统的范式是：

```
Store → Retrieve
```

MemForge 的范式是：

```
Storage → Retrieval → Reflection → Consolidation → Experience → Learning
```

记忆不只是被存储，而是被**管理**：创建、更新、冲突消解、废弃、遗忘，每一步都有审计事件和证据溯源。

## Features

- Long-term memory with lifecycle management (CANDIDATE → ACTIVE → DORMANT → DEPRECATED → FORGOTTEN)
- Hybrid retrieval (vector + keyword + temporal filter)
- Utility-aware reranking (`relevance + confidence + importance + freshness`)
- Temporal memory with half-open interval `[valid_from, valid_until)`
- Conflict detection & resolution (写入时检测，相似 ≠ 冲突)
- Evidence tracking (记忆 ↔ 原始对话双向溯源)
- Append-only event log (可审计、可回溯)
- Benchmark suite (synthetic + LongMemEval adapter)

## Quick Start

```bash
# 1. 启动 PostgreSQL + pgvector（仅 Stage 1 存储层需要；纯检索/QA benchmark 不需要）
docker compose up -d

# 2. 创建虚拟环境并安装（Python 3.12+）
python -m venv .venv
# Windows:
.venv\Scripts\python -m pip install -e ".[dev]"
# macOS / Linux:
# .venv/bin/python -m pip install -e ".[dev]"
# （若使用 uv：uv sync）

# 3. 配置环境变量
#    复制 .env.example 为 .env 并填入 LLM key（Stage 5B 需要）；
#    embedding 模型首次下载需联网，网络受限时设 HF_ENDPOINT=https://hf-mirror.com

# 4. 运行示例（纯内核，不依赖数据库/LLM）
.venv\Scripts\python examples/basic.py

# 5. 跑测试（当前 67 passed + 10 skipped）
.venv\Scripts\python -m pytest tests
```

## Architecture

```
USER → Agent → Memory Manager
                 ├── Extractor   (conversation → candidate memories)
                 ├── Evaluator   (importance / confidence scoring)
                 ├── ConflictDetector (vector coarse + LLM judgement)
                 ├── LifecycleManager (the ONLY status transition entry)
                 ├── Consolidator
                 └── Retrieval Engine (hybrid → utility rerank → top-k)
                          ↓
                 PostgreSQL + pgvector
                 (memories / memory_events / memory_evidence / memory_edges)
```

## Benchmark

两套独立的评测结果，**分开报告、不互比**（数据生成机制与指标口径不同）：

### A. Synthetic（Stage 4）— 内部机制消融，离线可复现（seed=42）

| System | Recall@1 | Recall@5 | MRR |
|--------|----------|----------|-----|
| Full Context | 0.17 | 0.50 | 0.38 |
| Naive Vector | 0.33 | 0.50 | 0.41 |
| Vector + Rerank | 0.33 | 0.63 | 0.44 |

> 消融结论：关闭 importance 权重后 Recall@5 0.63 → 0.48，utility 重排有效。
> 复现：`python -m benchmarks.runner`（产物在 `results/`）。

### B. LongMemEval-S（Stage 4.6）— 真实长对话纯检索

数据：`longmemeval_s_cleaned.json`（500 题，277MB，已 gitignore；SHA256 见
`benchmarks/data/manifests/longmemeval_s.json`）。Embedding：`all-MiniLM-L6-v2`。
记忆粒度：turn 级。指标：session 级 `recall_any@K` / `recall_all@K` / MRR / NDCG。
30 个 abstention 题排除出检索分母，单独计数。

| System | R@any@1 | R@any@5 | R@all@5 | MRR | NDCG@10 |
|--------|---------|---------|---------|------|---------|
| Naive Vector | 0.868 | 0.947 | 0.664 | 0.901 | 0.833 |
| Vector + Rerank | 0.843 | 0.934 | 0.613 | 0.884 | 0.802 |
| **Hybrid + Rerank** | **0.911** | **0.964** | **0.728** | **0.933** | **0.867** |

> **注意**：本表是**纯检索**指标，官方 LongMemEval 的主指标是 QA accuracy
> （检索+生成+LLM judge），两者不可横向比较。
> 在本次 LongMemEval-S session-level retrieval 实验中，Hybrid + Rerank configuration
> achieved the highest overall R@any@1/5, MRR and NDCG@10 among the evaluated
> configurations；但 R@all@5 与各类别表现见详细报告，不能把 rerank 与 hybrid
> 混成一个机制宣称无条件有效。
> 详细报告见 [docs/benchmarks-longmemeval.md](docs/benchmarks-longmemeval.md)。
> 复现：`HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_retrieval`。

### B2. LongMemEval-S 生命周期消融（Stage 4.7）— 诚实的负面结果

同一个 embedding 共享、6 个变体、470 有效题：

| Variant | R@any@1 | R@any@5 | R@all@5 | MRR | NDCG@10 |
|---------|---------|---------|---------|------|---------|
| naive_vector | 0.868 | 0.947 | 0.664 | 0.901 | 0.833 |
| **hybrid（无生命周期）** | **0.913** | **0.964** | **0.730** | **0.935** | **0.868** |
| full（Temporal+Conflict+Freshness） | 0.851 | 0.936 | 0.623 | 0.885 | 0.788 |
| no_temporal | 0.887 | 0.972 | 0.681 | 0.921 | 0.830 |
| no_conflict | 0.853 | 0.938 | 0.643 | 0.886 | 0.795 |
| no_freshness | 0.875 | 0.928 | 0.647 | 0.897 | 0.817 |

**结论（本阶段最重要的研究发现）**：在 LongMemEval-S 的 session-level retrieval 上，
MemForge 生命周期机制的**当前确定性实现**全部产生负贡献（`hybrid` 为最优配置）。
根因：① temporal 硬过滤误删 41 题的 gold 证据（官方 session 时间戳并非严格早于
提问时间）；② 相似度阈值（0.80）的 conflict 近似误删 48% 的 gold turn；
③ freshness 权重（0.15）引入排序噪声。这不是"生命周期概念失败"，而是指向校准方向：
时间感知排序替代硬过滤、LLM-based slot 级冲突检测替代相似度近似、窗口化 freshness。
详见 [docs/benchmarks-longmemeval.md](docs/benchmarks-longmemeval.md)。

复现：`HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_ablation`

### B3. LongMemEval-S 端到端 QA（Stage 5B）— 检索质量是否传导到最终回答

真实 LLM：`deepseek-chat`（temperature=0, max_tokens=256, thinking 关闭）。
每 system 500 题（470 answerable + 30 abstention），共 1500 次 API 调用。
检索输入已固化（Stage 4.7 导出），QA 阶段零 embedding、零检索计算。

| System | R@Any@5 | R@All@5 | QA Overall | QA Answerable | QA Abstention | Total Tokens |
|--------|---------|---------|-----------|--------------|---------------|-------------|
| Naive Vector | 0.947 | 0.664 | 0.338 | 0.296 | **1.000** | 1.06M |
| **Hybrid** | **0.964** | **0.730** | **0.364** | **0.326** | 0.967 | 1.32M |
| Full Lifecycle | 0.936 | 0.623 | 0.354 | 0.315 | 0.967 | 1.29M |

**三个核心发现：**

1. **Hybrid retrieval improves downstream QA**：Hybrid 在检索和 QA 上均最优（QA +2.6pp over Naive）。
2. **Retrieval quality alone does not determine QA**：Full 的 R@Any@5（0.936）比 Naive（0.947）低，但 QA 反而更高（0.354 vs 0.338）。LLM 对检索噪声有鲁棒性，记忆结构也很重要。
3. **Simple lifecycle heuristics are insufficient**：temporal 硬过滤导致 temporal-reasoning QA=0.024（Full 最差），但 single-session-user 上 Full 最优（0.828）——方向对，实现需校准。

> **注意**：`single-session-preference`（30题）三个 system 均为 0%，原因是该类别 gold_answer
> 是开放式偏好描述（50-100词），token-overlap 评估器不适用。语义相似度补充分析显示 Hybrid
> 最优（avg cosine=0.365）。详见最终研究报告。
> 费用：约 3.8 元（DeepSeek 谷时）。
> 详细报告：`D:\agentmemory\MemForge-Final-Research-Report.md` 和 `D:\agentmemory\MemForge-Stage5B-详细测试报告.md`。

## Reproduction

> 前提：下载 LongMemEval-S 数据（277MB，gitignored）并核对 SHA256——
> `benchmarks/scripts/download_longmemeval.py` + `benchmarks/data/manifests/longmemeval_s.json`。
> 真实数据与 embedding 缓存不提交 Git；以下命令均在项目根目录执行，
> embedding 模型为 `all-MiniLM-L6-v2`（本地 CPU，缓存于 `.hf_cache/`）。

### Stage 4.6 — LongMemEval-S 检索基线（3 systems）

```bash
HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_retrieval --model all-MiniLM-L6-v2
```
产物：`results/longmemeval/{manifest.json, retrieval_results.jsonl, by_category.csv, figures/}`

### Stage 4.7 — 生命周期消融（6 变体）

```bash
HF_HOME=.hf_cache python -m benchmarks.scripts.run_longmemeval_ablation --model all-MiniLM-L6-v2
```
产物：`results/longmemeval/{ablation_manifest.json, ablation.csv, by_category_ablation.csv, ablation_raw_results.jsonl, figures/}`

### Stage 5A — QA 基础设施验证（Mock，零 API key）

```bash
# 10 题 smoke（5 answerable + 5 abstention）
python -m benchmarks.scripts.run_longmemeval_qa --smoke --provider mock --model all-MiniLM-L6-v2
# 全量 3×500 流程验证（Mock 数值仅验证 pipeline，非研究结论）
python -m benchmarks.scripts.run_longmemeval_qa --provider mock --model all-MiniLM-L6-v2 --systems naive_vector,hybrid,full
```
产物：`results/longmemeval/stage5/{manifest.json, qa_comparison.csv, qa_by_category.csv, figures/}`

### Stage 5B — 真实 LLM QA（代码冻结后执行）

```bash
# Windows PowerShell:
# $env:MEMFORGE_LLM_MODEL="..."; $env:MEMFORGE_LLM_BASE_URL="..."; $env:MEMFORGE_LLM_API_KEY="..."
# 或复制 .env.example → .env 后按需导出
python -m benchmarks.scripts.run_longmemeval_qa --provider openai --systems naive_vector,hybrid,full
```
流程：先 `--smoke`（10 题，人工检查）→ 再全量 500×3。
检索输入已固化（`stage5/retrieved/`），正式 QA 阶段**零 embedding、零检索计算**。

---

## Limitations

- **Stage 4.7 的确定性 temporal 过滤会误删 gold evidence**：LongMemEval-S 的 session
  时间戳并非严格早于提问时间，41 题的 809 个 gold turns 被"只取过去"硬过滤剔除。
- **cosine-based conflict 检测会把正常同主题讨论误判为 supersession**：0.80 阈值下，
  681 个被取代 turn 中 48.5%（330 个）属于 gold session。
- **freshness 当前实现引入排序噪声**：权重 0.15 使 `any@1` 下降（0.851→0.875 在去掉后），
  应改为 question_date 窗口化衰减或移除。
- **当前生命周期消融不能证明 lifecycle 有益**：在 LongMemEval-S session-level retrieval 上，
  全部确定性 lifecycle 机制（当前实现）为负贡献。这是诚实的负面结果，指向校准方向
  （时间感知排序 / LLM-based slot 级冲突检测 / 窗口化 freshness），而非否定概念本身。
- **Mock QA 结果不是研究结果**：Stage 5A 数值仅验证 pipeline，不代表任何模型能力。
- **Stage 5B 已完成**：deepseek-chat 端到端 QA（1500 次调用），结果见 B3。单一 LLM 结果不代表所有模型。
- **single-session-preference 评估方法不适用**：该类别 gold_answer 为开放式描述，token-overlap 评估器全判 0%；语义相似度补充分析显示 Hybrid 最优。
- **LongMemEval retrieval 与 QA 指标必须分开解释**：官方主指标是 QA accuracy；
  本仓库 4.6/4.7 的 Recall/MRR/NDCG 为纯检索指标，两者不可横向比较。
- **当前结果不能代表所有 LLM / embedding model**：embedding 为 `all-MiniLM-L6-v2`（384 维），
  LLM 为 `deepseek-chat`；更换模型后数字会变化。
- **metrics.json 曾有 checkpoint 计数 bug**：Stage 5B 原始 metrics.json 显示 n=1000（实际 500），
  已在 Stage 6-B 从 raw_qa.jsonl 重新聚合修正。准确率数字未受影响。

## Roadmap

- [x] Stage 0: Pure-Python memory state machine kernel (22 tests)
- [x] Stage 1: Storage (PostgreSQL + pgvector) + Embedding + LLM + Extractor + API
- [x] Stage 2: Evaluator + Conflict detection & resolution
- [x] Stage 3: Temporal / Hybrid retrieval + Utility reranker + Forgetting
- [x] Stage 4: Benchmark framework (synthetic + ablation + noise)
- [x] Stage 4.5: LongMemEval adapter
- [x] Stage 4.6: Real LongMemEval-S retrieval experiments (3 baselines, 470 eval questions, abstention excluded)
- [x] Stage 4.7: Lifecycle ablation on LongMemEval-S (6 variants; honest negative result: current deterministic temporal/conflict/freshness all hurt retrieval)
- [x] Stage 5.0/5.1: QA infrastructure complete (prompt builder, evaluator, checkpoint, resume, error handling, manifest; Mock full 3×500 pipeline validated; 67 tests)
- [x] Stage 5B: End-to-end QA with real LLM (deepseek-chat, 1500 calls, 500题/system; Hybrid QA=0.364 optimal)
- [x] Stage 6: Final analysis (preference semantic eval, metrics fix, unified results, research report)
- [ ] Stage 7: GitHub release v0.1.0

## License

MIT