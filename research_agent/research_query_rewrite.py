"""检索失败反馈闭环（缺口 2，2026-08-16 新增）。

来源: Corrective RAG（Yan et al., 2024, arXiv:2401.15884）——检索后评估、
按分级（correct/ambiguous/incorrect）决定是否纠正；纠正手段为查询改写重搜。

三个层次：
1. `evaluate_retrieval` — 检索结果分级（纯函数，离线可测）
   - correct: 最高分达标 + 与查询有有意义术语重叠
   - incorrect: 分数低且无术语重叠 → 需要改写重搜
   - ambiguous: 介于两者之间 → 可尝试改写但不强制
2. `rewrite_query_for_retrieval` — LLM 改写（扩充/拆解/术语化三变体，失败回退原查询）
3. `adaptive_search` — 检索→评估→不足则改写重搜，max_rounds 收敛

设计约束: 全部默认关闭（由调用方传参启用），不改变任何现有调用路径。
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 分级默认阈值: 最高分低于此且无术语重叠 → incorrect
DEFAULT_MIN_SIGNAL = 0.3


def evaluate_retrieval(
    ranked: List[Dict],
    query: str,
    min_signal: float = DEFAULT_MIN_SIGNAL,
) -> Dict:
    """对一轮检索结果分级。

    信号 = 最高分数 + 有意义术语重叠（复用检索层的 meaningful_terms，
    单字中文与功能词不算信号——与 relevance gate 同一口径）。
    返回 {grade: correct|ambiguous|incorrect, top_score, overlap, hits}。
    """
    from .research_retrieval_runtime import meaningful_terms

    if not ranked:
        return {
            "grade": "incorrect",
            "top_score": 0.0,
            "overlap": [],
            "hits": 0,
            "reason": "no results",
        }
    top_score = float(ranked[0].get("score") or ranked[0].get("rrf_score") or 0.0)
    top_content = "\n".join(
        [
            str(ranked[0].get("title") or ""),
            str(ranked[0].get("content") or ""),
        ]
    )
    overlap = sorted(meaningful_terms(query) & meaningful_terms(top_content))
    hits = int(bool(overlap))

    if top_score >= min_signal and hits:
        grade = "correct"
    elif top_score < min_signal and not hits:
        grade = "incorrect"
    else:
        grade = "ambiguous"
    return {
        "grade": grade,
        "top_score": round(top_score, 6),
        "overlap": overlap,
        "hits": hits,
        "reason": grade,
    }


def rewrite_query_for_retrieval(
    query: str,
    failure_hint: str,
    llm_call: Callable[[str], str],
) -> str:
    """LLM 改写查询（扩充/拆解/术语化三变体之一）。

    llm_call: prompt -> 文本（与项目其余 LLM 调用同一签名）。
    输出非法/异常时回退原查询——改写是增强不是风险源。
    """
    prompt = (
        "你是检索查询改写器。上一轮检索质量不足，请改写查询以提升召回。\n"
        "原查询：{0}\n失败信号：{1}\n"
        "输出 JSON（不要多余文字）：{{\"variant\": \"expansion|decomposition|terminology\", "
        "\"rewritten_query\": \"改写后的查询\"}}\n"
        "变体含义：expansion=扩充同义术语/概念；decomposition=拆成子问题之一（选信息最密的）；"
        "terminology=换成更专业的领域术语。保持与原查询同语言。"
    ).format(query, failure_hint)
    try:
        reply = llm_call(prompt)
        if isinstance(reply, list):
            reply = reply[0] if reply else ""
        parsed = json.loads(str(reply).strip())
        rewritten = str(parsed.get("rewritten_query") or "").strip()
        if not rewritten:
            return query
        return rewritten
    except Exception as exc:  # noqa: BLE001 —— 改写是增强不是风险源：
        # 任何 LLM 异常（网络/限流/格式）都回退原查询，绝不让改写环节拖垮主流程
        logger.warning("query rewrite failed, fallback to original: %s", exc)
        return query


def adaptive_search(
    search_fn: Callable[[str, int], List[Dict]],
    query: str,
    llm_call: Optional[Callable[[str], str]] = None,
    max_rounds: int = 2,
    top_k: int = 5,
    min_signal: float = DEFAULT_MIN_SIGNAL,
) -> Dict:
    """检索→评估→不足则改写重搜，max_rounds 收敛。

    返回 {query, ranked, rounds, evaluations, grade, rewritten: bool}。
    无 llm_call 时退化为单轮（仅评估，不改写）——安全默认。
    """
    current_query = query
    evaluations = []
    ranked: List[Dict] = []
    rewritten = False
    for round_index in range(max_rounds):
        ranked = search_fn(current_query, top_k)
        evaluation = evaluate_retrieval(ranked, current_query, min_signal=min_signal)
        evaluations.append({"round": round_index + 1, "query": current_query, **evaluation})
        if evaluation["grade"] == "correct" or llm_call is None:
            break
        rewritten_query = rewrite_query_for_retrieval(
            current_query, evaluation["reason"], llm_call
        )
        if rewritten_query == current_query or not rewritten_query.strip():
            break  # 改写无效/回退 → 不再空转
        rewritten = True
        current_query = rewritten_query
    return {
        "query": current_query,
        "ranked": ranked,
        "rounds": len(evaluations),
        "evaluations": evaluations,
        "grade": evaluations[-1]["grade"],
        "rewritten": rewritten,
    }
