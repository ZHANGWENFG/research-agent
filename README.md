# my-agent：论文调研 Agent

一个自研的论文调研 Agent：LangGraph 状态图编排 + 自研多角色调研引擎 + 生产可靠性设施（幂等 / 熔断 / 审计 / 追踪）+ 合规全文获取 + 公开基准评测闭环。

**核心能力**：论文调研 + 知识库问答。你提问 → 自主判断意图 → 自主检索（PubMed/arXiv/本地 PDF）→ 证据不足自动升级自研多角色调研（问/答/写/审）→ 带引用与证据链回答；全文获取走合规链路（PMC/EuropePMC 文本接口 + 白名单下载 + 非白名单人工审批）。

**技术栈**：FastAPI + SSE、LangGraph（状态图 + SQLite 检查点）、litellm、SQLite（WAL）、sentence-transformers（可选）、rank-bm25、pypdf。

## 自研 / 改造 / 继承 边界

| 归属 | 模块 | 说明 |
|---|---|---|
| **自研（重写）** | `api.py` | FastAPI 接口 + SSE 推送 + 审批流 + 会话管理（建会话/压缩/还原/重新生成） |
| **自研（重写）** | `research_agent/research_production.py` | 生产控制面：幂等（SQLite 事务 + owner_token）、熔断、审计、全链路 Span |
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
| **继承（保留原实现）** | `research_agent/lm.py`、`rm.py`、`utils.py` | LM 抽象层（litellm 统一入口）、Arxiv/LocalPDF 检索器、工具函数 |

> 设计主线：**状态本地，能力外部**——记忆、审计、任务、审批、检查点全在本地 SQLite；LLM、检索、全文获取借用外部能力。本地管账，外部借力。

## 快速开始

1. 创建虚拟环境并安装依赖：
   ```bash
   python -m venv venv
   venv/Scripts/pip install -r requirements.txt   # Windows
   ```
   （可选装 sentence-transformers/transformers 启用真实向量；不装自动降级哈希向量）
2. 配置模型密钥（环境变量，litellm 风格）：`DEEPSEEK_API_KEY` 或 `MINIMAX_API_KEY`
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

## 目录

- `research_agent/`——核心包（26 个模块）
- `evaluation/public_benchmarks/`——公开评测（SciFact/QASPER/LongMemEval，验收不退化）
- `skills/`——skill 领域知识包（每领域一个 SKILL.md：术语/视角/检索词/注意）
- `storage/`——运行数据（任务/审批/记忆/检查点，SQLite + JSON）

## 关键设计（详见《my-agent-项目全解.md》）

- 主线 8 步：入口 → 两道闸（幂等/熔断，单用户模式）→ 意图路由（规则+LLM，0.65 置信）→ 记忆召回（五信号）→ 三路分支 → 证据裁判（规则+三词判定）→ 深度调研（自研问/答/写/审循环）→ 带引用回答
- 自研多角色循环：4 Agent（问/答/写/审）+ 模型自主并行调度 + 6 条硬边界
- 全文获取四级：文本接口 → 白名单下载（三重校验）→ 非白名单人工审批 → 兜底链接
- 评测验收：跑公开 Benchmark 不退化（SciFact Recall@10=0.8379 / QASPER F1=0.5441 / LongMemEval-S Recall@5=0.8003）

## License

MIT License. 基于开源项目 PaperStorm（MIT, Stanford OVAL）fork 并重构；LM 抽象层、检索器、工具函数保留原实现。
