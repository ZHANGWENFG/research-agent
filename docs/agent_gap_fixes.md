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

## 缺口①：事实性验证（Attribution，2026-08-16）

**依据**：Attributed QA（Rashkin et al., 2021, arXiv:2112.11961）——答案逐句回链证据；Self-RAG（Asai et al., 2023）——"断言是否被证据支撑"显式判定。

**实现** `research_agent/research_attribution.py`：
- `split_sentences(text)`：中英文句号/感叹/问号拆句，引用编号 [n] 不拆断（英文句号后须跟非数字空白，小数 3.14 不拆）
- `verify_claim(claim, evidence_pool)`：有意义术语重叠（复用 `meaningful_terms`，与检索层口径一致）≥2，或（数字命中 + ≥1 术语重叠）→ supported；返回 {supported, evidence_ids, score}；纯数字 token 不算术语重叠（数字由单独信号加权）
- `verify_article(article, evidence_pool)`：全文逐句验证 → {total_sentences, supported_count, unsupported: [...], coverage_ratio}；空证据池 → 全部 unsupported + reason="no evidence"
- **不删文只标注**——保持"证据驱动的诚实"；同义改写句子可能误标 unsupported，误标成本低

**挂接** `research_loop.py`：`write_article` 与 `qc_review` 之间插入 `attribution_check` 节点（图结构固定，开关在节点内判定）；结果写入 `state["qc_attribution"]`（`ResearchLoopState` TypedDict 已声明该键）；不阻断流程，qc 报告只读使用；revise 重写后自动重验证。

**开关**：`RESEARCH_ATTRIBUTION=1`
**回滚**：不设开关即节点透传零计算；删除节点函数 + `add_node`/两条 `add_edge` + schema 键即完全移除。

## 缺口②：代码执行沙箱（2026-08-16）

**依据**：Code Interpreter 模式——LLM 生成 Python → 受限环境执行 → 结果回传 agent。实现为 subprocess + 超时强杀 + 输出截断 + 白名单拒绝，零第三方依赖、离线可测。

**实现** `research_agent/research_code_sandbox.py`：
- `FORBIDDEN_PATTERNS` 白名单拒绝表：`import os`/`from os`、`import subprocess`、`open(`、`__import__(`、网络库（socket/requests/urllib/http/ftplib/smtplib/aiohttp/httpx）、文件系统库（shutil/glob/pathlib/tempfile）、`eval(`/`exec(`、`compile(`、`input(`、`os.system/popen/remove/...`
- `_is_blocked(code)` → {blocked, reason}；`run_python_sandbox(code, timeout=10, max_output_bytes=65536)` → {ok, stdout, stderr, exit_code, duration_ms, blocked, reason}：timeout 强杀（`subprocess.run` timeout → TimeoutExpired → 终止子进程）、输出按 UTF-8 字节截断并标记 `...[truncated]`
- **安全边界如实声明**：规则层尽力而为防护，非隔离容器——防御恶意代码/提权/资源耗尽不在范围内，面向研究验算场景

**挂接** `research_agent_loop.py`：`RESEARCH_CODE_SANDBOX=1` 时 TOOLS 追加 `run_code`（第五工具）；`AgentLoop(run_code=...)` 注入执行器（如 `run_python_sandbox`）；未注入 → ok=False "run_code not configured"；未启用时工具不存在（决策层拒绝，更诚实）。

**开关**：`RESEARCH_CODE_SANDBOX=1`
**回滚**：不设开关 → TOOLS 保持 4 工具零变化；删除模块文件 + TOOLS 条件 + `_execute_tool` 分支即完全移除。

## 缺口④：记忆自动沉淀（2026-08-16）

**依据**：MEMGPT（Packer et al., 2023, arXiv:2310.08560）——零散记忆在"睡眠"期被蒸馏成更抽象的语义记忆。

**实现** `research_agent/research_memory_consolidation.py`：
- `_cluster_records(records, min_episodes=3)`：术语重叠贪心聚类（复用 `meaningful_terms`），簇 < 阈值不沉淀
- `consolidate_memories(store, llm_call=None, min_episodes=3)`：每簇 LLM 总结（prompt 要求一句话 + #tags）→ `remember_semantic(content, metadata={consolidated_at, source_episode_ids}, tags)`；无 LLM → 规则提取（最高频术语 top3 + 首条摘录前 100 字）；源记录打标防重复
- 只依赖 `ResearchMemoryStore` 内存接口，持久化由调用方（挂接）负责

**挂接** `research_service.py`：`_run_research_loop` 成功后 `MEMORY_CONSOLIDATE=1` → `_consolidate_memories_after_run`：加载/新建 `root_dir/memory.json` store → 写入本次任务 2 条 episodic（主题/QC 结果）→ `consolidate_memories(store, llm_call=chat_llm, min_episodes=3)` → 无条件落盘（原料跨任务积累）；异常静默 warning，不阻断主线。

**开关**：`MEMORY_CONSOLIDATE=1`
**回滚**：不设开关 → 方法首行 return 零行为；删除模块文件 + 方法 + 调用行即完全移除（memory.json 为运行时数据，可一并删除）。

## 开关用法

| 开关 | 位置 | 效果 |
|---|---|---|
| `RESEARCH_QUERY_REWRITE=1` | 环境变量 | ask() 证据不足时查询改写重搜 |
| `RESEARCH_AGENT_LOOP=1` 或 `ask(mode="agent")` | 环境变量/参数 | LLM 自主工具循环替代固定流水线 |
| `RESEARCH_PARALLEL=1` | 环境变量 | research 任务走多视角线程并行 |
| `RESEARCH_ATTRIBUTION=1` | 环境变量 | write_article 后逐句回链验证 → state.qc_attribution（只读报告） |
| `RESEARCH_CODE_SANDBOX=1` | 环境变量 | AgentLoop TOOLS 追加 run_code 沙箱工具 |
| `MEMORY_CONSOLIDATE=1` | 环境变量 | 任务成功后记忆聚类沉淀（root_dir/memory.json） |

## 回滚方式

三个缺口互不依赖、各自独立开关：

1. **模块级回滚**：删除 `research_query_rewrite.py` / `research_agent_loop.py` / `research_parallel.py` / `research_attribution.py` / `research_code_sandbox.py` / `research_memory_consolidation.py` 六个文件 + 对应挂接分支（均有 `os.getenv` 守卫，守卫前行为 = 改造前行为）
2. **行为级回滚**：不设任何 env 开关即回到旧行为（已验证：`test_gap_hooks.py` 中 `test_ask_default_mode_zero_change` / `test_service_parallel_off_default` 断言开关关闭零变化；`test_loop_cost.py::test_attribution_disabled_default_no_state`、`test_agent_loop.py::test_run_code_disabled_default_tools`、`test_memory_consolidation.py::test_consolidate_hook_disabled_no_side_effect` 断言本轮三开关关闭零行为）
3. **本轮三开关独立回滚**：attribution（节点 + schema 键）、代码沙箱（TOOLS 条件 + 分支）、记忆沉淀（方法 + 调用行）各自删除/关闭互不影响

## 测试与验收

| 项 | 结果 |
|---|---|
| 新模块测试 | 缺口 2（16）+ 缺口 1（13）+ 缺口 4（9）+ 挂接（10）= 48 个，全离线 |
| 新增代码覆盖率 | 缺口 2 / 缺口 1 / 缺口 4 模块均 ≥70% |
| 回归 | 全量 pytest 全绿（含原有 242 测试） |
| 多租户 | 本次改造不含多租户设计（用户指令 2026-08-16），production 层保持 SINGLE_USER_MODE=True |
| 缺口① 事实性验证 | `tests/test_attribution.py` 18 用例 + 挂接 2（test_loop_cost.py） |
| 缺口② 代码沙箱 | `tests/test_code_sandbox.py` 12 用例 + 挂接 3（test_agent_loop.py） |
| 缺口④ 记忆沉淀 | `tests/test_memory_consolidation.py` 16 用例（含挂接 4） |
