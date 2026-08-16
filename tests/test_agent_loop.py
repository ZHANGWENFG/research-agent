"""缺口 1: LLM 工具调用循环测试（research_agent_loop.py）。

ReAct 式循环: mock 决策序列验证 search→rewrite→search→answer、
三重终止（agent 决定/证据达标/超步数）、非法输出容错、证据池去重。
全离线。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_agent_loop import AgentLoop, TOOLS


def _decider(*decisions):
    """按调用次序依次返回决策 JSON。"""
    queue = list(decisions)
    calls = {"n": 0}

    def decide(state):
        if calls["n"] >= len(queue):
            return json.dumps({"tool": "answer", "args": {"question": "q"}})
        item = queue[calls["n"]]
        calls["n"] += 1
        return item

    return decide


def _search_mock(records=None):
    """按查询返回固定证据列表。"""
    pool = {"calls": []}

    def search_fn(query, top_k):
        pool["calls"].append(query)
        return [
            {"document_id": "d1", "title": "T1", "content": "pim antenna analysis", "score": 0.9},
            {"document_id": "d2", "title": "T2", "content": "rf interference", "score": 0.7},
        ]

    return search_fn, pool


def test_agent_loop_full_sequence():
    """search→rewrite→search→answer 完整序列，证据池聚合。"""
    search_fn, pool = _search_mock()
    rewrite = lambda q, hint, llm: "rewritten: {0}".format(q)
    loop = AgentLoop(
        decide=_decider(
            json.dumps({"reasoning": "先检索", "tool": "search", "args": {"query": "pim"}}),
            json.dumps({"reasoning": "改写", "tool": "rewrite", "args": {"hint": "ambiguous"}}),
            json.dumps({"reasoning": "再搜", "tool": "search", "args": {"query": "pim rf"}}),
            json.dumps({"reasoning": "够了", "tool": "answer", "args": {"question": "pim 是什么"}}),
        ),
        search=search_fn,
        llm_call=lambda p: "ok",
        rewrite=rewrite,
        max_steps=10,
    )
    state = loop.run("pim")
    assert state["termination"]["reason"] == "agent_decided_answer"
    assert state["step"] == 4
    assert len(state["history"]) == 3  # answer 不入 history
    assert len(state["evidence_pool"]) == 2  # 证据池去重后 2 条
    assert state["errors"] == []


def test_agent_loop_max_steps_termination():
    """LLM 一直不 answer → 超步数强制终止。"""
    search_decision = json.dumps({"tool": "search", "args": {}})
    loop = AgentLoop(
        # 3 个 search 决策占满 max_steps=3，之后耗尽也不会 answer
        decide=_decider(search_decision, search_decision, search_decision),
        search=lambda q, k: [{"document_id": "d1", "content": "x", "score": 0.1}],
        max_steps=3,
    )
    state = loop.run("q")
    assert state["termination"]["reason"] == "max_steps"
    assert state["step"] == 3
    assert len(state["history"]) == 3


def test_agent_loop_evidence_sufficient_auto():
    """证据达标自动终止（不依赖 LLM 决定）。"""
    def sufficient(state):
        return len(state["evidence_pool"]) >= 1

    search_fn, _ = _search_mock()
    loop = AgentLoop(
        decide=_decider(json.dumps({"tool": "search", "args": {"query": "pim"}})),
        search=search_fn,
        evidence_sufficient=sufficient,
        max_steps=10,
    )
    state = loop.run("pim")
    assert state["termination"]["reason"] == "evidence_sufficient_auto"
    assert state["step"] == 1


def test_agent_loop_unknown_tool_tolerated():
    """未知 tool → 记 error 继续，不崩。"""
    loop = AgentLoop(
        decide=_decider(
            json.dumps({"tool": "hack_the_planet", "args": {}}),
            json.dumps({"tool": "search", "args": {}}),
            json.dumps({"tool": "answer", "args": {}}),
        ),
        search=lambda q, k: [{"document_id": "d1", "content": "x", "score": 0.1}],
        max_steps=5,
    )
    state = loop.run("q")
    assert state["termination"]["reason"] == "agent_decided_answer"
    assert state["errors"][0]["error"] == "unknown tool"


def test_agent_loop_invalid_json_decision():
    """非法 JSON 决策 → 记 error；连续非法才终止（防死循环）。"""
    loop = AgentLoop(
        decide=lambda state: "not json at all",
        search=lambda q, k: [],
        max_steps=5,
    )
    state = loop.run("q")
    assert len(state["errors"]) == 3  # 连续 3 次非法 → 终止
    assert state["termination"]["reason"] == "repeated_invalid_decisions"


def test_agent_loop_search_tool_failure_isolated():
    """search 抛异常 → ok=False 记 error，循环继续。"""
    def bad_search(q, k):
        raise RuntimeError("index down")

    loop = AgentLoop(
        decide=_decider(
            json.dumps({"tool": "search", "args": {}}),
            json.dumps({"tool": "search", "args": {}}),
        ),
        search=bad_search,
        max_steps=2,
    )
    state = loop.run("q")
    assert state["errors"][0]["error"] != ""
    assert state["termination"]["reason"] == "max_steps"


def test_agent_loop_rewrite_without_config():
    """rewrite 工具未注入 → ok=False（防御式配置检查）。"""
    loop = AgentLoop(
        decide=_decider(
            json.dumps({"tool": "rewrite", "args": {}}),
            json.dumps({"tool": "answer", "args": {}}),
        ),
        search=lambda q, k: [],
        max_steps=5,
    )
    state = loop.run("q")
    assert state["history"][0]["outcome"]["ok"] is False
    assert "not configured" in state["history"][0]["outcome"]["error"]


def test_agent_loop_fetch_fulltext():
    """fetch_fulltext 命中 → 记录 chars；未命中 → 记 error。"""
    def decide(state):
        if state["step"] == 1:
            return json.dumps({"tool": "fetch_fulltext", "args": {"article_id": "PMC1"}})
        return json.dumps({"tool": "answer", "args": {}})

    loop = AgentLoop(
        decide=decide,
        search=lambda q, k: [],
        fetch_fulltext=lambda aid: "x" * 300 if aid == "PMC1" else None,
        max_steps=5,
    )
    state = loop.run("q")
    assert state["history"][0]["outcome"]["ok"] is True
    assert state["history"][0]["outcome"]["chars"] == 300


def test_agent_loop_fetch_fulltext_missing():
    loop = AgentLoop(
        decide=_decider(
            json.dumps({"tool": "fetch_fulltext", "args": {"article_id": "missing"}}),
            json.dumps({"tool": "answer", "args": {}}),
        ),
        search=lambda q, k: [],
        fetch_fulltext=lambda aid: None,
        max_steps=5,
    )
    state = loop.run("q")
    assert state["history"][0]["outcome"]["ok"] is False
    assert "unavailable" in state["history"][0]["outcome"]["error"]


def test_agent_loop_evidence_dedup():
    """同一 document 多次搜索只入池一次。"""
    def search_fn(query, top_k):
        return [{"document_id": "d1", "content": "same doc", "score": 0.5}]

    loop = AgentLoop(
        decide=_decider(
            json.dumps({"tool": "search", "args": {}}),
            json.dumps({"tool": "search", "args": {}}),
            json.dumps({"tool": "answer", "args": {}}),
        ),
        search=search_fn,
        max_steps=5,
    )
    state = loop.run("q")
    assert len(state["evidence_pool"]) == 1


def test_tools_contract():
    """工具集常量与实现一致（文档契约）。"""
    assert set(TOOLS) == {"search", "rewrite", "fetch_fulltext", "answer"}


def test_run_code_disabled_default_tools():
    """挂接②：RESEARCH_CODE_SANDBOX 未设时 run_code 不在 TOOLS——决策层拒绝。"""
    assert "run_code" not in TOOLS  # 默认环境（本测试进程未设开关）
    run_code_decisions = [
        json.dumps({"tool": "run_code", "args": {"code": "print({0})".format(i)}})
        for i in range(5)
    ]
    loop = AgentLoop(
        decide=_decider(*run_code_decisions),
        search=_search_mock()[0],
        max_steps=5,
    )
    state = loop.run("q")
    assert state["termination"]["reason"] in ("repeated_invalid_decisions", "max_steps")
    assert any("unknown tool" in e.get("error", "") for e in state["errors"])


def test_run_code_not_configured():
    """挂接②：run_code 分支存在但执行器未注入 → ok=False 语义。"""
    loop = AgentLoop(decide=_decider(json.dumps({"tool": "answer"})), search=_search_mock()[0])
    outcome = loop._execute_tool(
        {"evidence_pool": []}, "run_code", {"code": "print(1)"}, "q", 5
    )
    assert outcome["ok"] is False
    assert outcome["error"] == "run_code not configured"


def test_run_code_injected_passthrough():
    """挂接②：注入执行器后结果透传（沙箱启用语义由调用方注入）。"""
    def fake_sandbox(code):
        assert code == "print(2**8)"
        return {"ok": True, "stdout": "256", "blocked": False}

    loop = AgentLoop(
        decide=_decider(json.dumps({"tool": "answer"})),
        search=_search_mock()[0],
        run_code=fake_sandbox,
    )
    outcome = loop._execute_tool(
        {"evidence_pool": []}, "run_code", {"code": "print(2**8)"}, "q", 5
    )
    assert outcome == {"ok": True, "stdout": "256", "blocked": False}
