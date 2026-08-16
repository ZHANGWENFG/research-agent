"""成本护栏接线测试（R4, 2026-08-16）。

此前 COST_WARN/COST_HARD/_estimate_cost 全是死代码——docstring 宣称"成本护栏"
但零调用点。本轮接线后: tracked llm 累计真实成本（dict 契约）/估算回退（str
契约），should_continue 硬顶退出，generate_perspectives 预警降并发。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_loop import (
    COST_HARD,
    COST_WARN,
    MAX_TURNS,
    build_research_loop_graph,
    generate_perspectives,
    should_continue,
)


class _FakeSearch:
    """假检索: 返回固定命中（不需要真网络）。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, query, top_k=5):
        self.calls += 1
        return [
            {
                "title": "Fake Paper {0}".format(i),
                "url": "https://example.org/{0}".format(i),
                "snippet": "相关描述 {0}".format(query),
            }
            for i in range(3)
        ]


# ---------- R4: 成本累计 ----------

def test_tracked_cost_real_usage_wins():
    """llm_call 返回增强 dict（含 cost_usd）时累计真实成本。"""
    calls = {"n": 0}

    def rich_llm(prompt, **kwargs):
        calls["n"] += 1
        if "作家" in prompt:
            return {"text": "我没问题了", "cost_usd": 0.01}  # 结束循环
        return {"text": "ok", "cost_usd": 0.42}

    graph = build_research_loop_graph(rich_llm, _FakeSearch())
    final = graph.invoke({"topic": "测试"})
    assert calls["n"] >= 1
    assert final.get("cost_used", 0.0) >= 0.42, (
        "真实成本应累计, got {0}".format(final.get("cost_used"))
    )


def test_tracked_cost_estimate_fallback():
    """旧契约（str 返回）按估算单价累计（兼容不破坏）。"""
    def plain_llm(prompt, **kwargs):
        if "作家" in prompt:
            return "我没问题了"  # 结束循环
        return "ok"

    graph = build_research_loop_graph(plain_llm, _FakeSearch())
    final = graph.invoke({"topic": "测试"})
    assert final.get("cost_used", 0.0) > 0.0, "估算回退也应累计"


# ---------- R4: 硬顶 ----------

def test_should_continue_exits_on_cost_hard():
    """累计成本超硬顶 → exit（停止继续烧钱）。"""
    state = {
        "questions": ["继续问吗"],
        "turn": 0,
        "cost_used": COST_HARD + 1.0,
    }
    assert should_continue(state) == "exit"


def test_should_continue_within_budget_keeps_asking():
    """成本未超限时按原有规则（未问完/未达轮数 → 继续）。"""
    state = {
        "questions": ["继续问吗"],
        "turn": 0,
        "cost_used": 0.01,
    }
    assert should_continue(state) == "ask_again"


# ---------- R4: 预警降并发 ----------

def test_generate_perspectives_warn_reduces_parallel(monkeypatch):
    """估算成本超预警 → 并发强制降 1（此前 docstring 宣称的语义现在生效）。"""
    from research_agent import research_loop

    # 人为把单价抬到必超预警: PER_LLM_CALL_USD * MAX_TURNS * 2 * 6 >= COST_WARN
    monkeypatch.setattr(research_loop, "PER_LLM_CALL_USD", COST_WARN / 6.0 / MAX_TURNS / 2 * 1.01)
    # 用廉价 llm 返回平行计划并发 3（会被预警压下）
    def llm(prompt, **kwargs):
        return (
            '{"perspectives": [{"name": "A", "focus": "x"}], '
            '"parallel": {"suggested_concurrency": 3}}'
        )

    state = generate_perspectives({"topic": "t"}, llm, skill_context="")
    assert state["parallel_plan"]["effective"] == 1, (
        "超预警应降并发到 1, got {0}".format(state["parallel_plan"]["effective"])
    )
    monkeypatch.undo()


def test_generate_perspectives_normal_parallel_kept():
    """成本正常时并发按模型建议（封顶 MAX_PARALLEL）。"""
    def llm(prompt, **kwargs):
        return (
            '{"perspectives": [{"name": "A", "focus": "x"}], '
            '"parallel": {"suggested_concurrency": 2}}'
        )

    state = generate_perspectives({"topic": "t"}, llm, skill_context="")
    assert state["parallel_plan"]["effective"] == 2


# ---------- 图 smoke: 签名统一没破坏构建 ----------

def test_loop_graph_builds_and_runs_fake():
    """完整图跑通（fake 模式）：验证 5 个节点签名统一 + tracked 层不破坏契约。"""
    def fake_llm(prompt, **kwargs):
        # 不同节点返回不同形态，模拟真实 prompt 分布
        text = prompt
        if "调度员" in text:  # generate_perspectives
            return '{"perspectives": [{"name": "A", "focus": "x"}], "parallel": {"suggested_concurrency": 1}}'
        if "作家" in text:  # writer_ask → 立即结束循环
            return "我没问题了"
        if "改写" in text:  # expert_answer 拆词
            return '["keyword"]'
        if "引用池" in text:  # write_article
            return '[{"section": "S", "points": ["p"], "citations": []}]'
        return "普通回答"  # expert_answer 回答 / qc_review 等

    graph = build_research_loop_graph(fake_llm, _FakeSearch())
    final = graph.invoke({"topic": "LangGraph 是什么"})
    # 关键断言: 图正常跑完（不抛异常），成本已累计
    assert final.get("cost_used", 0.0) >= 0.0
    assert "cost_used" in final


# ---------- revise 循环修复（条件边改 state 的隐藏 bug） ----------

def test_qc_reject_revises_bounded_then_force_exit():
    """QC 打回最多 MAX_REVISE 次后强制出稿（修复前会无限递归到 langgraph 上限）。"""
    from research_agent.research_loop import MAX_REVISE, qc_review, should_revise

    # 第一次打回: revise_count 0→1（节点内递增），still < 上限 → revise
    state1 = qc_review(
        {"topic": "t", "result": {"citation_pool": []}, "article": "内容",
         "perspectives": ["A"]},
        lambda prompt, **kw: '{"citation_ok": false, "coverage_ok": false, "duplication_ok": false, "issues": ["x"], "scorecard": {}}',
    )
    assert state1["revise_count"] == 1
    assert should_revise(state1) == "revise"

    # 打回到上限后 → force_exit（不再 revise）
    state2 = dict(state1)
    state2["revise_count"] = MAX_REVISE + 1
    state2["qc_result"] = {"passed": False}
    assert should_revise(state2) == "force_exit"

    # 合格 → exit
    state3 = dict(state1)
    state3["qc_result"] = {"passed": True}
    assert should_revise(state3) == "exit"


def test_qc_reject_loop_terminates_in_graph():
    """完整图 + QC 始终打回 → 在 MAX_REVISE 次后正常结束（不抛递归错误）。"""
    def rejecting_llm(prompt, **kwargs):
        if "调度员" in prompt:
            return '{"perspectives": [{"name": "A", "focus": "x"}], "parallel": {"suggested_concurrency": 1}}'
        if "质检员" in prompt:  # QC 永远打回（必须最先判断：prompt 也含"引用池"字样）
            return '{"citation_ok": false, "coverage_ok": false, "duplication_ok": false, "issues": ["假引用"], "scorecard": {}}'
        if "作家" in prompt:
            return "我没问题了"
        if "改写" in prompt:
            return '["keyword"]'
        if "引用池" in prompt:
            return '[{"section": "S", "points": ["p"], "citations": []}]'
        return "普通回答"

    graph = build_research_loop_graph(rejecting_llm, _FakeSearch())
    final = graph.invoke({"topic": "t"}, config={"recursion_limit": 50})
    # 图正常结束，QC 最终未通过
    assert final.get("qc_result", {}).get("passed") is False
    assert final.get("revise_count", 0) >= 1
