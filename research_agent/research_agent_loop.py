"""LLM 自主工具调用循环（缺口 1，2026-08-16 新增）。

来源: ReAct（Yao et al., 2022, arXiv:2210.03629）——LLM 在思考-行动-观察
循环中自主选择工具、观察结果、决定下一步。

`AgentLoop` 状态机:
  {query, evidence_pool, step, history}
  每步 LLM 输出结构化决策 {reasoning, tool, args}，工具集:
    search(query, top_k)      → 检索并入证据池
    rewrite(hint)             → 复用缺口 2 改写模块
    fetch_fulltext(id)        → 复用 research_fulltext 获取全文
    answer(question)          → 终止并合成
  终止三重保险:
    1. LLM 决定 answer
    2. evaluate_evidence_sufficiency 自动达标
    3. step >= max_steps（默认 5）

设计约束:
- 全部离线可测: llm_call / search_fn 均由调用方注入，测试用 mock
- 非法工具输出容错: 未知 tool / 缺 args / 抛异常 → 记 error 继续，不崩
- 默认关闭: 只有显式传入决策函数才启用，现有固定流水线零改动
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 5

# run_code 工具（缺口②挂接，2026-08-16）：RESEARCH_CODE_SANDBOX=1 才出现。
# 工具不存在比存在但报错更诚实——LLM 决策层就拒绝未知工具。
_SANDBOX_ENABLED = os.getenv("RESEARCH_CODE_SANDBOX") == "1"
TOOLS = ("search", "rewrite", "fetch_fulltext", "answer") + (
    ("run_code",) if _SANDBOX_ENABLED else ()
)


class AgentLoop:
    """ReAct 式工具调用循环（纯协调器，工具全部依赖注入）。"""

    def __init__(
        self,
        *,
        decide: Callable[[Dict], str],
        search: Callable[[str, int], List[Dict]],
        llm_call: Optional[Callable[[str], str]] = None,
        rewrite: Optional[Callable[[str, str, Callable], str]] = None,
        fetch_fulltext: Optional[Callable[[str], Optional[str]]] = None,
        run_code: Optional[Callable[[str], Dict]] = None,
        evidence_sufficient: Optional[Callable[[Dict], bool]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.decide = decide  # state -> 原始决策文本（含 JSON）
        self.search = search
        self.llm_call = llm_call
        self.rewrite = rewrite
        self.fetch_fulltext = fetch_fulltext
        self.run_code = run_code
        self.evidence_sufficient = evidence_sufficient
        self.max_steps = max_steps

    def run(self, query: str, top_k: int = 5) -> Dict:
        """执行循环直到终止条件。返回最终状态。"""
        state: Dict = {
            "query": query,
            "evidence_pool": [],
            "step": 0,
            "history": [],
            "errors": [],
            "termination": None,
        }
        for step in range(1, self.max_steps + 1):
            state["step"] = step
            decision = self._safe_decide(state)
            if decision is None:
                # 非法决策容错: 连续非法达上限才终止（防死循环），否则继续下一轮
                state["bad_decisions"] = state.get("bad_decisions", 0) + 1
                if state["bad_decisions"] >= 3:
                    state["termination"] = {"reason": "repeated_invalid_decisions"}
                    break
                continue
            tool = decision.get("tool")
            args = decision.get("args") or {}
            if tool == "answer":
                state["termination"] = {
                    "reason": "agent_decided_answer",
                    "question": args.get("question") or query,
                }
                break
            outcome = self._execute_tool(state, tool, args, query, top_k)
            state["history"].append(
                {
                    "step": step,
                    "reasoning": decision.get("reasoning", ""),
                    "tool": tool,
                    "args": args,
                    "outcome": outcome,
                }
            )
            if outcome.get("ok") is False:
                state["errors"].append(
                    {"step": step, "tool": tool, "error": outcome.get("error", "")}
                )
            if self.evidence_sufficient is not None and self.evidence_sufficient(state):
                state["termination"] = {"reason": "evidence_sufficient_auto"}
                break
        if state["termination"] is None:
            state["termination"] = {
                "reason": "max_steps",
                "max_steps": self.max_steps,
            }
        return state

    # ---------- 内部 ----------

    def _safe_decide(self, state: Dict) -> Optional[Dict]:
        try:
            raw = self.decide(state)
            parsed = json.loads(str(raw).strip())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("agent loop: invalid decision at step %s: %s", state["step"], exc)
            state["errors"].append({"step": state["step"], "error": repr(exc)})
            return None
        tool = str(parsed.get("tool") or "")
        if tool not in TOOLS:
            state["errors"].append(
                {"step": state["step"], "tool": tool, "error": "unknown tool"}
            )
            return None
        return parsed

    def _execute_tool(self, state: Dict, tool: str, args: Dict, query: str, top_k: int) -> Dict:
        try:
            if tool == "search":
                q = str(args.get("query") or query)
                k = max(1, int(args.get("top_k") or top_k))
                results = self.search(q, k)
                self._add_evidence(state, results)
                return {"ok": True, "tool": "search", "results": len(results), "query": q}
            if tool == "rewrite":
                if self.rewrite is None or self.llm_call is None:
                    return {"ok": False, "error": "rewrite not configured"}
                hint = str(args.get("hint") or "ambiguous")
                rewritten = self.rewrite(query, hint, self.llm_call)
                results = self.search(rewritten, top_k)
                self._add_evidence(state, results)
                return {"ok": True, "tool": "rewrite", "rewritten": rewritten, "results": len(results)}
            if tool == "fetch_fulltext":
                if self.fetch_fulltext is None:
                    return {"ok": False, "error": "fetch_fulltext not configured"}
                article_id = str(args.get("article_id") or "")
                text = self.fetch_fulltext(article_id)
                if not text:
                    return {"ok": False, "error": "fulltext unavailable", "article_id": article_id}
                return {"ok": True, "tool": "fetch_fulltext", "chars": len(text)}
            if tool == "run_code":
                if self.run_code is None:
                    return {"ok": False, "error": "run_code not configured"}
                code = str(args.get("code") or "")
                return self.run_code(code)
            return {"ok": False, "error": "unhandled tool: {0}".format(tool)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent loop tool %s failed: %s", tool, exc)
            return {"ok": False, "error": repr(exc)}

    def _add_evidence(self, state: Dict, results: List[Dict]):
        """并入证据池（按 document_id 去重，保留首次出现顺序）。"""
        seen = {
            item.get("document_id") or item.get("chunk_id")
            for item in state["evidence_pool"]
        }
        for item in results:
            key = item.get("document_id") or item.get("chunk_id")
            if key is not None and key not in seen:
                state["evidence_pool"].append(item)
                seen.add(key)
