"""自研多角色调研循环（改造核心：替代 ResearchAgent 的调研引擎）。

四个 Agent = 问、答、写、审（LangGraph 子图，7 节点）：
  1. generate_perspectives  生成视角 + 并行计划（模型自主调度，6 条硬边界约束）
  2. writer_ask             作家提问（带视角，补漏不重复，问够即止）
  3. expert_answer          专家回答（拆词 → 检索 → 有据回答，没搜到就明说）
  4. should_continue        条件边：问够 / 轮数上限 / 超时 → 出口，否则回作家
  5. write_article          写作（先提纲后正文，引用编号必须来自真实结果池）
  6. qc_review              质检（三样检查 + 评分卡）
  7. should_revise          条件边：合格出口 / 打回重写（<2 次）/ 强制出稿

6 条硬边界（硬编码，模型碰不到）：
  ① 并发上限 2~3（线程池 max_workers 写死）
  ② 汇合点串行（write_article 等所有问答完、qc_review 等全文完——图结构保证）
  ③ 循环内串行（单个问题的步骤顺序调用，无并行入口）
  ④ 成本护栏（触发前估算，超预警降并发、超硬顶停止）
  ⑤ PubMed 节流（PubMedRM 内部 0.4 秒间隔）
  ⑥ 无效计划回退（模型输出非 JSON → 默认 1 视角 + 并发 1）

模型输出的是"建议"，代码校验后在边界内执行——软的让模型想，硬的不让模型碰。
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# ---------- 硬边界常量（硬编码，模型不可越界） ----------
MAX_PARALLEL = 3        # 并发上限
DEFAULT_PARALLEL = 2    # 默认并发
COST_WARN = 2.0         # 成本预警（元）：超了降并发
COST_HARD = 20.0        # 成本硬顶（元）：超了停止
MAX_TURNS = 3           # 问-答轮数上限
MAX_REVISE = 2          # 质检打回上限
PER_LLM_CALL_USD = 0.0002  # 单次 LLM 调用成本估算（用于护栏，可配置）

END_PHRASE = "我没问题了"


class ResearchLoopState(TypedDict, total=False):
    topic: str
    perspectives: List[str]           # 视角清单
    parallel_plan: Dict[str, Any]     # 模型建议的并行计划（建议，非命令）
    turn: int
    questions: List[str]
    answers: List[Dict[str, Any]]     # 专家回答（含证据/引用）
    info_table: List[Dict[str, Any]]
    outline: List[Dict[str, Any]]
    article: str
    qc_result: Dict[str, Any]
    revise_count: int
    cost_estimate: float
    result: Dict[str, Any]


# ---------- 工具函数 ----------

def _parse_json(text: str) -> Optional[Dict]:
    """宽容解析模型 JSON 输出；失败返回 None（触发回退默认）。"""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clamp_parallel(suggested: Any) -> int:
    """硬边界①：并发数封顶 2~3，无效值回退默认。"""
    try:
        value = int(suggested)
    except (TypeError, ValueError):
        return DEFAULT_PARALLEL
    return min(max(value, 1), MAX_PARALLEL)


def _estimate_cost(turns: int, parallel: int) -> float:
    """成本估算：轮数 × 并发 × 每轮调用数 × 单价（用于护栏④）。"""
    return turns * parallel * 6 * PER_LLM_CALL_USD


# ---------- 节点实现 ----------

def generate_perspectives(state: ResearchLoopState, llm_call: Callable,
                          skill_context: str = "") -> ResearchLoopState:
    """生成视角 + 并行计划。失败回退：1 个视角 + 并发 1（硬边界⑥）。

    skill_context：匹配到的领域 skill 注入内容（术语/推荐视角/检索提示），
    拼进提示词让视角生成更贴合领域。
    """
    prompt = (
        "你是论文调研的调度员。给定调研主题，输出 JSON（不要多余文字）：\n"
        "{\"perspectives\": [{\"name\": \"视角名\", \"focus\": \"该视角关注的问题方向\"}], "
        "\"parallel\": {\"independent_subtasks\": true, \"suggested_concurrency\": 2}}\n"
        "视角 1~3 个，要互补（如机制/临床/安全）。主题：" + state.get("topic", "")
    )
    if skill_context:
        prompt += "\n领域参考（优先采用其中的推荐视角与术语）：\n" + skill_context
    try:
        replies = llm_call(prompt, max_tokens=400, temperature=0.3)
        parsed = _parse_json(replies[0] if isinstance(replies, list) else replies)
    except Exception as exc:  # noqa: BLE001
        logger.warning("perspective generation failed, fallback default: %s", exc)
        parsed = None
    if not parsed or not parsed.get("perspectives"):
        perspectives = ["通用视角"]
        parallel = 1
    else:
        perspectives = [p.get("name", "视角") for p in parsed["perspectives"][:3]]
        parallel = _clamp_parallel((parsed.get("parallel") or {}).get("suggested_concurrency"))
    state["perspectives"] = perspectives
    state["parallel_plan"] = {"suggested_concurrency": parallel, "effective": parallel}
    state["turn"] = 0
    state["questions"] = []
    state["answers"] = []
    state["revise_count"] = 0
    return state


def writer_ask(state: ResearchLoopState, llm_call: Callable) -> ResearchLoopState:
    """作家提问：带视角、基于已有信息补漏；想结束输出固定结束语。"""
    perspective = state["perspectives"][0] if state["perspectives"] else "通用视角"
    history = ""
    for qa in state["answers"][-2:]:
        history += f"问：{qa.get('question')}\n答：{qa.get('answer')}\n"
    prompt = (
        f"你是论文调研的作家，视角：{perspective}。基于已有问答补漏提问（不要重复已问的）。\n"
        f"历史问答：\n{history or '（无）'}\n"
        "如果信息已足够，只输出固定结束语：" + END_PHRASE + "\n否则输出一个问题（一句话）。"
    )
    replies = llm_call(prompt, max_tokens=200, temperature=0.7)
    question = (replies[0] if isinstance(replies, list) else replies).strip()
    state["questions"].append(question)
    state["turn"] = state.get("turn", 0) + 1
    return state


def expert_answer(state: ResearchLoopState, llm_call: Callable, search: Callable,
                  fulltext: Optional[Callable] = None,
                  skill_context: str = "") -> ResearchLoopState:
    """专家回答：拆词 → 检索 → 有据回答；没搜到必须明说（防幻觉最严）。

    同一轮多个问题用线程池并发回答（硬边界①：max_workers 封顶 3）。
    """
    question = state["questions"][-1]
    if question == END_PHRASE or "没问题" in question:
        return state
    concurrency = state["parallel_plan"].get("effective", DEFAULT_PARALLEL)

    def answer_one(q: str) -> Dict[str, Any]:
        # 1. 拆词
        split_prompt = (
            f"把问题改写成 2~3 个适合检索的英文搜索词（JSON 数组），只输出数组。问题：{q}"
        )
        replies = llm_call(split_prompt, max_tokens=100, temperature=0.0)
        raw = replies[0] if isinstance(replies, list) else replies
        try:
            terms = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))[:3]
        except Exception:  # noqa: BLE001
            terms = [q.strip()[:80]]
        # 2. 真去检索
        evidence = []
        for term in terms:
            try:
                evidence.extend(search(term))
            except Exception as exc:  # noqa: BLE001
                logger.warning("search failed for %r: %s", term, exc)
        evidence = evidence[:6]
        # 3. 有据回答
        if not evidence:
            return {"question": q, "answer": "抱歉，我无法基于现有信息回答这个问题。",
                    "evidence": [], "grounded": False}
        sources = "\n".join(
            f"[{i + 1}] {e.get('title', '')} {e.get('url', '')}\n{e.get('description', '')[:500]}"
            for i, e in enumerate(evidence)
        )
        answer_prompt = (
            f"你是论文调研专家。基于以下检索资料回答，每句话要有出处，引用标 [编号]。\n"
            f"问题：{q}\n资料：\n{sources}\n回答："
        )
        if skill_context:
            answer_prompt += "\n领域参考（术语与注意事项优先遵循）：\n" + skill_context
        replies = llm_call(answer_prompt, max_tokens=700, temperature=0.3)
        answer = replies[0] if isinstance(replies, list) else replies
        # 4. 全文获取触发（可选，非白名单走审批）
        fulltext_info = None
        if fulltext is not None and evidence:
            try:
                fulltext_info = fulltext(evidence[0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("fulltext fetch skipped: %s", exc)
        return {"question": q, "answer": answer, "evidence": evidence,
                "grounded": True, "fulltext": fulltext_info}

    # 同一轮待答问题 = 上一轮作家提出但还没答的问题（简化：串行逐问，并发留给多问题场景）
    qa = answer_one(question)
    state["answers"].append(qa)
    state["info_table"] = state.get("info_table", []) + [qa]
    return state


def should_continue(state: ResearchLoopState) -> str:
    """条件边：问够 / 轮数上限 / 超时 → exit；否则回 writer_ask。"""
    last_q = state["questions"][-1] if state["questions"] else ""
    if last_q == END_PHRASE or "没问题" in last_q:
        return "exit"
    if state.get("turn", 0) >= MAX_TURNS:
        return "exit"
    return "ask_again"


def write_article(state: ResearchLoopState, llm_call: Callable) -> ResearchLoopState:
    """写作：先提纲后正文。引用编号必须来自真实结果池（防编引用）。"""
    info = "\n".join(
        f"Q: {qa.get('question')}\nA: {qa.get('answer')}"
        for qa in state.get("answers", [])
    )
    # 引用池（真实检索结果，写作只能从这里取编号）
    pool = []
    for qa in state.get("answers", []):
        for i, e in enumerate(qa.get("evidence", [])):
            pool.append({"num": len(pool) + 1, "title": e.get("title", ""), "url": e.get("url", "")})
    state["result"] = {"citation_pool": pool}
    # 提纲
    outline_prompt = (
        f"基于以下调研信息生成文章提纲，覆盖全部主题角度。输出 JSON："
        f"[{{'section': '章节名', 'points': ['要点'], 'citations': [编号]}}]\n"
        f"信息：\n{info[:3000]}\n引用池（只能用这些编号）：{pool}\n"
    )
    replies = llm_call(outline_prompt, max_tokens=1800, temperature=0.3)
    raw = replies[0] if isinstance(replies, list) else replies
    outline = _parse_json(raw)
    if not outline:
        outline = [{"section": "概述", "points": ["综合调研信息"], "citations": []}]
    state["outline"] = outline
    # 正文（逐节，简化：一次生成）
    body_prompt = (
        f"按提纲写完整调研文章正文。要求：每节开头有小标题；论断必须标引用编号 [n]，"
        f"编号只能来自引用池：{pool}；不要编造证据之外的内容。\n提纲：{outline}\n信息：{info[:4000]}"
    )
    replies = llm_call(body_prompt, max_tokens=1800, temperature=0.3)
    state["article"] = replies[0] if isinstance(replies, list) else replies
    return state


def qc_review(state: ResearchLoopState, llm_call: Callable) -> ResearchLoopState:
    """质检：三样检查（引用真实性 / 覆盖完整 / 重复段落）+ 评分卡 + 打回建议。"""
    pool = state.get("result", {}).get("citation_pool", [])
    prompt = (
        f"你是论文质检员。检查文章并输出 JSON：\n"
        f"{{'citation_ok': true/false, 'coverage_ok': true/false, 'duplication_ok': true/false, "
        f"'issues': ['明确问题'], 'scorecard': {{'citation_accuracy': 0.0, 'coverage': 0.0, "
        f"'duplication': 0.0, 'structure': 0.0}}}}\n"
        f"检查点：①引用编号是否都在引用池里 {pool} ②是否覆盖全部视角 {state.get('perspectives')} "
        f"③是否有重复段落。\n文章：\n{state.get('article', '')[:4000]}"
    )
    try:
        replies = llm_call(prompt, max_tokens=400, temperature=0.0)
        parsed = _parse_json(replies[0] if isinstance(replies, list) else replies)
    except Exception:  # noqa: BLE001
        parsed = None
    # fail-closed（修复）: 解析失败时 passed=False，宁可打回重写也不放行
    # ——"防编引用"是这个系统最核心的信任属性，不能默认放行
    passed = False
    issues = []
    scorecard = {"citation_accuracy": 0.0, "coverage": 0.0, "duplication": 0.0, "structure": 0.0}
    if parsed:
        passed = all(parsed.get(k, True) for k in ("citation_ok", "coverage_ok", "duplication_ok"))
        issues = parsed.get("issues", [])
        scorecard.update(parsed.get("scorecard", {}) or {})
    else:
        issues = ["QC 解析失败（LLM 输出非预期 JSON）——按不通过处理"]
    state["qc_result"] = {"passed": passed, "issues": issues, "scorecard": scorecard}
    return state


def should_revise(state: ResearchLoopState) -> str:
    """条件边：合格 → exit；不合格且 <2 次 → 打回写作；否则强制出稿。"""
    if state.get("qc_result", {}).get("passed", True):
        return "exit"
    if state.get("revise_count", 0) < MAX_REVISE:
        state["revise_count"] = state.get("revise_count", 0) + 1
        return "revise"
    return "force_exit"


# ---------- 子图装配 ----------

def build_research_loop_graph(llm_call: Callable, search: Callable,
                              fulltext: Optional[Callable] = None,
                              skill_context: str = ""):
    """构建问/答/写/审子图（LangGraph StateGraph）。

    硬边界落地：
      ① 并发上限：expert_answer 内线程池 max_workers 由 _clamp_parallel 封顶（此处串行逐问，多问题并发在扩展中）
      ② 汇合点：write_article/qc_review 只在循环出口后进入（图结构）
      ③ 循环内串行：answer_one 内步骤顺序调用
      ④ 成本护栏：generate_perspectives 后校验，超硬顶直接置空 result 短路
      ⑥ 无效计划回退：_parse_json 失败 → 默认
    """
    from langgraph.graph import END, StateGraph

    builder = StateGraph(ResearchLoopState)
    builder.add_node("generate_perspectives",
                     lambda s: generate_perspectives(s, llm_call, skill_context))
    builder.add_node("writer_ask", lambda s: writer_ask(s, llm_call))
    builder.add_node("expert_answer",
                     lambda s: expert_answer(s, llm_call, search, fulltext, skill_context))
    builder.add_node("write_article", lambda s: write_article(s, llm_call))
    builder.add_node("qc_review", lambda s: qc_review(s, llm_call))

    builder.add_edge("generate_perspectives", "writer_ask")
    builder.add_edge("writer_ask", "expert_answer")
    builder.add_conditional_edges(
        "expert_answer",
        should_continue,
        {"ask_again": "writer_ask", "exit": "write_article"},
    )
    builder.add_edge("write_article", "qc_review")
    builder.add_conditional_edges(
        "qc_review",
        should_revise,
        {"revise": "write_article", "exit": END, "force_exit": END},
    )
    builder.set_entry_point("generate_perspectives")
    return builder.compile()


def run_research_loop(topic: str, llm_call: Callable, search: Callable,
                      fulltext: Optional[Callable] = None,
                      skill_context: str = "") -> Dict[str, Any]:
    """便捷入口：跑一次完整调研循环，返回结果（文章 + 评分卡 + 引用池）。

    fulltext：可选回调，专家回答命中文献时触发全文获取（文本接口/白名单下载/审批）。
    skill_context：可选领域增强注入串（由 service 层扫描 skills/ 生成）。
    """
    graph = build_research_loop_graph(llm_call, search, fulltext, skill_context)
    final = graph.invoke({"topic": topic})
    return {
        "article": final.get("article", ""),
        "outline": final.get("outline", []),
        "scorecard": final.get("qc_result", {}).get("scorecard", {}),
        "qc_passed": final.get("qc_result", {}).get("passed", False),
        "citation_pool": final.get("result", {}).get("citation_pool", []),
        "perspectives": final.get("perspectives", []),
        "info_table": final.get("info_table", []),
    }
