# my-agent：论文调研 Agent（独立新项目）

基于 PaperStorm v5.6 抽取主线 + 自研增强组建的独立项目（不含 STORM/Co-STORM/Zotero/企业知识库/fake/演示前端）。

**核心能力**：论文调研 + 知识库问答。你提问 → 自主判断意图 → 自主检索（PubMed/arXiv/本地 PDF）→ 证据不足自动升级自研多角色调研（问/答/写/审）→ 带引用与证据链回答；全文获取走合规链路（PMC/EuropePMC 文本接口 + 白名单下载 + 非白名单人工审批）。

**技术栈**：FastAPI + SSE、LangGraph（状态图 + SQLite 检查点）、litellm、SQLite（WAL）、sentence-transformers（可选）、rank-bm25、pypdf。

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
- POST /api/chat —— 问答（生产运行时 → 幂等/熔断/审计 → v45 主编排图）
- POST /api/kb/query —— 知识库查询
- GET /api/approvals/pending —— 待审批列表
- GET /api/skills —— 已安装 skill 列表
- GET /api/health —— 健康检查

## 目录

- `knowledge_storm/`——核心包（主线 18 文件 + 新增 5 文件 + 评测入口 5 文件）
- `evaluation/public_benchmarks/`——公开评测（SciFact/QASPER/LongMemEval，验收不退化）
- `skills/`——skill 领域知识包（每领域一个 SKILL.md：术语/视角/检索词/注意）

## 关键设计（详见《改后项目设计.md》）

- 主线 8 步：入口 → 两道闸（幂等/熔断，单用户模式）→ 意图路由（规则+LLM，0.65 置信）→ 记忆召回（五信号）→ 三路分支 → 证据裁判（规则+三词判定）→ 深度调研（自研问/答/写/审循环）→ 带引用回答
- 自研多角色循环：4 Agent（问/答/写/审）+ 模型自主并行调度 + 6 条硬边界
- 全文获取四级：文本接口 → 白名单下载（三重校验）→ 非白名单人工审批 → 兜底链接
- 评测验收：跑公开 Benchmark 不退化（SciFact Recall@10=0.8379 / QASPER F1=0.5441 / LongMemEval-S Recall@5=0.8003）
