# my-agent：论文调研 Agent

一个自研的论文调研 Agent：LangGraph 状态图编排 + 自研多角色调研引擎 + 生产可靠性设施（幂等 / 熔断 / 审计 / 追踪）+ 合规全文获取 + 公开基准评测闭环。

**核心能力**：论文调研 + 知识库问答。你提问 → 自主判断意图 → 自主检索（PubMed/arXiv/本地 PDF）→ 证据不足自动升级自研多角色调研（问/答/写/审）→ 带引用与证据链回答；全文获取走合规链路（PMC/EuropePMC 文本接口 + 白名单下载 + 非白名单人工审批）。

**技术栈**：FastAPI + SSE、LangGraph（状态图 + SQLite 检查点）、litellm、SQLite（WAL）、sentence-transformers（可选）、rank-bm25、pypdf。

## 自研 / 改造 / 继承 边界

| 归属 | 模块 | 说明 |
|---|---|---|
| **自研（重写）** | `api.py` | FastAPI 接口 + SSE 推送 + 审批流 + 会话管理（建会话/压缩/还原/重新生成） |
| **自研（重写）** | `research_agent/research_production.py` | 生产控制面：幂等（SQLite 事务 + owner_token + 24h 保留窗口）、熔断（三态含 half-open 探测）、指数退避重试、审计、全链路 Span |
| **自研（重写）** | `research_agent/research_langgraph.py` + `research_graph_adapter.py` | 主编排状态图：10 节点 + 3 条件路由，RetryPolicy 只挂联网节点 |
| **自研（重写）** | `research_agent/research_loop.py` | 多角色调研引擎：问/答/写/审 4 Agent 子图 + 6 条硬边界（并发上限/汇合串行/成本护栏/无效计划回退） |
| **自研（重写）** | `research_agent/research_longterm_memory.py` | 长期记忆服务（写入策略/混合召回/合并） |
| **自研（重写）** | `research_agent/research_context.py` | 上下文引擎（token 计量/压缩/100% 还原） |
| **自研（重写）** | `research_agent/research_fulltext.py` | 合规全文获取四级链路（文本接口 → 白名单 → 人工审批 → 兜底链接） |
| **自研（重写）** | `research_agent/research_skill.py` | skill 领域知识注入机制（SKILL.md → 视角/检索词/术语） |
| **自研（重写）** | `research_agent/research_service.py` | 任务服务（队列/状态机/陈旧任务恢复/压测） |
| **自研（重写）** | `evaluation/` | 公开基准评测（SciFact/QASPER/LongMemEval）+ 自建评测 |
| **改造（重构自开源）** | `research_agent/research_intent_router.py`、`research_router_llm.py` | 意图路由（规则 + LLM，0.65 置信阈值）+ LLM 闭包工厂 |
| **改造（重构自开源）** | `research_agent/research_qa.py`、`research_kb_qa.py` | 调研问答（证据裁判三词判定）+ 知识库问答 |
| **改造（重构自开源）** | `research_agent/research_retrieval_common.py`、`research_retrieval_runtime.py`、`research_retrieval_index.py` | 混合检索（BM25 + 向量，向量缺失自动降级哈希） |
| **改造（重构自开源）** | `research_agent/research_pubmed.py`、`research_memory.py` | PubMed 检索、记忆压缩 |
| **继承（保留原实现）** | `research_agent/lm.py`、`rm.py` | LM 抽象层（litellm 统一入口）、Arxiv/LocalPDF 检索器（`utils.py` 792 行死代码已于 2026-08-16 删除） |

> 设计主线：**状态本地，能力外部**——记忆、审计、任务、审批、检查点全在本地 SQLite；LLM、检索、全文获取借用外部能力。本地管账，外部借力。

## 快速开始

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   venv/Scripts/pip install -r requirements.txt   # Windows
   venv/Scripts/pip install -r requirements-dev.txt  # 测试/CI 依赖（pytest/coverage/ruff）
   ```
   （可选装 sentence-transformers/transformers 启用真实向量；不装自动降级哈希向量）
2. 配置模型密钥（环境变量，litellm 风格）：`DEEPSEEK_API_KEY` 或 `MINIMAX_API_KEY`

   可选配置（均有安全/资源语义，见「参数设计依据」）：
   ```bash
   export MY_AGENT_API_KEY="xxx"               # 设置后所有 /api/* 需带 X-API-Key 头（OWASP API4 防滥用）
   export MY_AGENT_RATE_LIMIT="60"             # 每 IP 每分钟请求上限（默认 60）
   export MY_AGENT_MODEL_CONTEXT_TOKENS="128000"  # 主模型上下文窗口（默认 32768）
   ```

   > **安全提醒**：服务默认绑 127.0.0.1。除非配合 API_KEY 使用，否则**不要** `--host 0.0.0.0`——单用户放行模式下暴露到公网等于把 LLM 调用额度拱手让人（OWASP API4:2023 无限流风险）。
3. 启动服务：
   ```bash
   TIKTOKEN_CACHE_DIR=~/.tiktoken-cache venv/Scripts/python -m uvicorn api:app --port 8000
   ```
4. 接口测试：
   ```bash
   curl http://127.0.0.1:8000/api/health
   curl -X POST http://127.0.0.1:8000/api/research -H "Content-Type: application/json" -d '{"topic":"LangGraph 是什么","run_mode":"fake"}'
   curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d '{"message":"你好"}'
   ```

## 接口清单

- POST /api/research —— 提交调研任务（后台线程执行）
- GET /api/research/{task_id} —— 查任务状态 / 产物
- GET /api/research/{task_id}/stream —— SSE 事件流（task_status / heartbeat）
- POST /api/research/{task_id}/approve —— 审批回传（非白名单下载）
- POST /api/chat —— 问答（生产运行时 → 幂等/熔断/审计 → 主编排图）
- POST /api/kb/query —— 知识库查询
- POST /api/sessions —— 建会话（完整多轮会话：压缩/还原/重新生成）
- GET /api/approvals/pending —— 待审批列表
- GET /api/skills —— 已安装 skill 列表
- GET /api/admin/status | /api/admin/audit | /api/admin/spans —— 治理面板
- GET /api/health —— 健康检查

## 质量保障（CI 已验证）

- **测试**：71 个（检索栈 / 评测指标 / 熔断 / 幂等 / 意图路由 / 记忆 / skill / 会话 / 冒烟），离线可跑，不依赖真实 LLM
- **覆盖率门禁**：48.8%（`--cov-fail-under=45`，起点门槛，逐轮上调）
- **Lint**：ruff（F 类真实 bug + E4 导入位置），315 错误清零
- **CI**：`.github/workflows/ci.yml`——Python 3.11/3.12 矩阵、pytest + 覆盖率门禁、ruff；`requirements-dev.txt` 与 CI 严格对应，本地可完整复现

## 参数设计依据（2026-08-16 设计评审落地）

> 每项参数都带业界依据来源，不是拍脑袋。评审全文见 commit `70755e4` 信息与设计评审文档。

| 参数 | 值 | 依据 |
|---|---|---|
| RRF 融合常数 k | 60 | Cormack & Clarke, SIGIR'09（原论文默认值）；LangChain `EnsembleRetriever c=60`；Haystack `MultiRetriever` 默认 RRF |
| 分块大小 / 重叠 | 500 字符 / 50 | 2026 RAG chunking 基准（512 tokens + 50–100 overlap）；NVIDIA（句子边界优先）；arXiv 2601.14123（overlap 无显著收益取下限）；切分按句子边界对齐 |
| 候选池 candidate_k | max(top_k×5, 20) | LangChain `fetch_k=20 / k=4`（5 倍候选池）；RRF 需要两路交叉提名空间 |
| 重试退避 | 指数 + full jitter（base 0.5s） | AWS《Exponential Backoff And Jitter》：`sleep=uniform(0, base·2^n)`，防多客户端同步打点 |
| 熔断阈值 / 冷却 | 3 次 / 30s + half-open 探测 | AWS Prescriptive Guidance 三态模型（open→half-open→closed）；Hystrix 同款 |
| 幂等保留窗口 | 24h | Stripe《Idempotent Requests》默认 24h（窗口外允许复用键） |
| API 鉴权 / 限流 | 可选 X-API-Key + 每 IP 60 次/分 | OWASP API Top10（2023）API4 无限流→DoS+账单激增；REST Security Cheat Sheet |
| 评测最小来源数 | 2 | 至少两个独立证据（单点来源的系统性偏差无法暴露） |
| 上下文窗口 | 环境变量可配，默认 32k | 现代模型 128k/200k 常见，写死 32k 浪费大模型窗口、小模型会爆 |
| 成本护栏 | 真实成本累计（估算回退） | 护栏必须基于真实计费（litellm completion_cost），固定估算对贵模型差 100 倍 |
| MMR 多样性 | λ=0.7（冗余惩罚 0.3） | LangChain/Elastic 默认 0.5；MetricGate 实测 sweet spot [0.5, 0.7] |
| 评测指标 | recall@k / MRR / nDCG（TREC 口径） | NIST TREC 官方定义；Manning《Introduction to Information Retrieval》Ch.8 |

## 目录

- `research_agent/`——核心包（26 个模块）
- `evaluation/public_benchmarks/`——公开评测（SciFact/QASPER/LongMemEval，验收不退化）
- `skills/`——skill 领域知识包（每领域一个 SKILL.md：术语/视角/检索词/注意）
- `storage/`——运行数据（任务/审批/记忆/检查点，SQLite + JSON）
- `.github/workflows/`——CI（测试 + 覆盖率门禁 + lint）

## 关键设计（详见《my-agent-项目全解.md》）

- 主线 8 步：入口 → 两道闸（幂等/熔断，单用户模式）→ 意图路由（规则+LLM，0.65 置信）→ 记忆召回（五信号）→ 三路分支 → 证据裁判（规则+三词判定）→ 深度调研（自研问/答/写/审循环）→ 带引用回答
- 自研多角色循环：4 Agent（问/答/写/审）+ 模型自主并行调度 + 6 条硬边界
- 全文获取四级：文本接口 → 白名单下载（三重校验）→ 非白名单人工审批 → 兜底链接
- 评测验收：公开 Benchmark 实测（SciFact test 300 例，产物落 `storage/benchmark_runs/`）
  - **hybrid Recall@10=0.8114 / MRR@10=0.6298**（真实向量 all-MiniLM-L6-v2，2026-08-15 本机复现）
  - dense Recall@10=0.7857（真实向量）｜ bm25 Recall@10=0.7592
  - QASPER / LongMemEval 为早期基线记录（代码重构后待复跑）
- 评测复现：`python run_scifact_benchmark.py --real --model sentence-transformers/all-MiniLM-L6-v2 --output storage/benchmark_runs/scifact-real`（BEIR 官方数据集，产物含 git commit / 语料 sha256 / 时间戳，完全可复现）

## License

MIT License. 基于开源项目 PaperStorm（MIT, Stanford OVAL）fork 并重构；LM 抽象层、检索器保留原实现。
