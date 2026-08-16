# 缺口 1/2/4 改造说明（2026-08-16）

本文件记录三项 agent 能力缺口的改造：**LLM 自主工具调用循环**、**检索失败反馈闭环（查询改写）**、**真·多代理并行**。全部**默认关闭**（env 开关），零改动现有默认行为，天然可回滚。

## 文献依据

| 缺口 | 方法 | 来源 |
|---|---|---|
| 1. LLM 工具循环 | ReAct：思考→行动→观察循环 | Yao et al., 2022, arXiv:2210.03629 |
| 2. 查询改写反馈 | Corrective RAG（CRAG）：检索后分级纠正 | Yan et al., 2024, arXiv:2401.15884 |
| 4. 多代理并行 | STORM：多视角并行研究后聚合 | Shao et al., 2024, arXiv:2402.14207 |

## 缺口 2：查询改写（先落地，缺口 1 复用）

**模块** `research_agent/research_query_rewrite.py`

- `evaluate_retrieval(ranked, query, min_signal=0.3)` → 分级 `correct` / `ambiguous` / `incorrect`
  - 信号 = 最高分 + 有意义术语重叠（复用 `meaningful_terms`，与 relevance gate 同一口径）
- `rewrite_query_for_retrieval(query, failure_hint, llm_call)` → LLM 改写（扩充/拆解/术语化三变体）；**任何异常回退原查询**（改写是增强不是风险源）
- `adaptive_search(search_fn, query, llm_call, max_rounds=2)` → 检索→评估→不足则改写重搜；改写无进展/无 LLM 时单轮安全退化

**挂接** `research_agent/research_qa.py`：`ask()` 中 `RESEARCH_QUERY_REWRITE=1` 时，证据充分性不足 → 自适应重搜 → 改写后重新评估；达标则用新证据作答并记录 `query_rewrite_adaptive` 事件。

## 缺口 1：LLM 自主工具调用循环

**模块** `research_agent/research_agent_loop.py`

- `AgentLoop`：状态 `{query, evidence_pool, step, history, errors}`，每步 LLM 输出结构化决策 `{reasoning, tool, args}`
- 工具集：`search(query, top_k)` / `rewrite(hint)`（复用缺口 2）/ `fetch_fulltext(article_id)` / `answer(question)`
- **三重终止保险**：LLM 决定 answer ｜ `evaluate_evidence_sufficiency` 自动达标 ｜ step ≥ max(5)
- **容错**：非法决策记 error 继续（连续 3 次才终止防死循环）；工具抛异常隔离记录；证据池按 document_id 去重

**挂接** `research_agent/research_qa.py`：`ask(mode="agent")` 或 `RESEARCH_AGENT_LOOP=1` → 走 AgentLoop 替代固定流水线；无 LLM 配置时**安全回退固定流水线**。

## 缺口 4：真·多代理并行

**模块** `research_agent/research_parallel.py`

- `run_parallel_perspectives(topic, perspectives, llm_call, search, max_workers=3)`：
  - `ThreadPoolExecutor` 并行跑每个视角独立子任务（检索+合成段落）
  - 聚合：按视角原序拼装、证据按 document_id/url 去重
  - **失败隔离**：单视角失败记 error 继续，不拖垮整体

**挂接** `research_agent/research_service.py`：`_run_research_loop` 中 `RESEARCH_PARALLEL=1` → 复用 `generate_perspectives` 产出视角（STORM），执行阶段升级为真线程并行；聚合文章写 `myagent_article_polished.txt`（与串行同一产物契约）。

## 开关用法

| 开关 | 位置 | 效果 |
|---|---|---|
| `RESEARCH_QUERY_REWRITE=1` | 环境变量 | ask() 证据不足时查询改写重搜 |
| `RESEARCH_AGENT_LOOP=1` 或 `ask(mode="agent")` | 环境变量/参数 | LLM 自主工具循环替代固定流水线 |
| `RESEARCH_PARALLEL=1` | 环境变量 | research 任务走多视角线程并行 |

## 回滚方式

三个缺口互不依赖、各自独立开关：

1. **模块级回滚**：删除 `research_query_rewrite.py` / `research_agent_loop.py` / `research_parallel.py` 三个文件 + 对应挂接分支（均有 `os.getenv` 守卫，守卫前行为 = 改造前行为）
2. **行为级回滚**：不设任何 env 开关即回到旧行为（已验证：`test_gap_hooks.py` 中 `test_ask_default_mode_zero_change` / `test_service_parallel_off_default` 断言开关关闭零变化）

## 测试与验收

| 项 | 结果 |
|---|---|
| 新模块测试 | 缺口 2（16）+ 缺口 1（13）+ 缺口 4（9）+ 挂接（10）= 48 个，全离线 |
| 新增代码覆盖率 | 缺口 2 / 缺口 1 / 缺口 4 模块均 ≥70% |
| 回归 | 全量 pytest 全绿（含原有 242 测试） |
| 多租户 | 本次改造不含多租户设计（用户指令 2026-08-16），production 层保持 SINGLE_USER_MODE=True |
