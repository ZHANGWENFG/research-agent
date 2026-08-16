"""缺口 2: 查询改写反馈闭环测试（research_query_rewrite.py）。

CRAG 分级（correct/ambiguous/incorrect）+ LLM 改写 + adaptive 收敛，
全离线：mock LLM 调用、纯函数黄金样例。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_query_rewrite import (
    adaptive_search,
    evaluate_retrieval,
    rewrite_query_for_retrieval,
)


def _ranked(*scores, content=None):
    return [
        {"title": "T{0}".format(i), "content": content or "正文", "score": score}
        for i, score in enumerate(scores)
    ]


# ================= 分级黄金样例 =================

def test_evaluate_empty_incorrect():
    result = evaluate_retrieval([], "pim antenna")
    assert result["grade"] == "incorrect"
    assert result["top_score"] == 0.0
    assert result["hits"] == 0


def test_evaluate_correct_high_score_and_overlap():
    ranked = _ranked(0.8, content="passive intermodulation antenna analysis")
    result = evaluate_retrieval(ranked, "pim antenna", min_signal=0.5)
    assert result["grade"] == "correct"
    assert result["top_score"] == 0.8
    assert result["hits"] == 1


def test_evaluate_incorrect_low_score_no_overlap():
    ranked = _ranked(0.1, content="muon collider optimization physics")
    result = evaluate_retrieval(ranked, "pim antenna", min_signal=0.5)
    assert result["grade"] == "incorrect"
    assert result["overlap"] == []


def test_evaluate_ambiguous_high_score_no_overlap():
    """分数高但无术语重叠 → ambiguous（可能是语义相关，需谨慎）。"""
    ranked = _ranked(0.7, content="passive intermodulation antenna")
    result = evaluate_retrieval(ranked, "射频无源互调分析", min_signal=0.5)
    assert result["grade"] == "ambiguous"


def test_evaluate_ambiguous_low_score_with_overlap():
    ranked = _ranked(0.2, content="pim antenna basics")
    result = evaluate_retrieval(ranked, "pim antenna", min_signal=0.5)
    assert result["grade"] == "ambiguous"


def test_evaluate_default_min_signal():
    """默认 min_signal=0.3: 0.3 分 + 重叠 → correct。"""
    ranked = _ranked(0.31, content="pim antenna analysis")
    assert evaluate_retrieval(ranked, "pim antenna")["grade"] == "correct"


# ================= LLM 改写 =================

def test_rewrite_parses_json():
    llm = lambda prompt: '{"variant": "expansion", "rewritten_query": "pim antenna rf interference analysis"}'
    assert rewrite_query_for_retrieval(
        "pim antenna", "ambiguous", llm
    ) == "pim antenna rf interference analysis"


def test_rewrite_falls_back_on_invalid_json():
    llm = lambda prompt: "这不是 JSON"
    assert rewrite_query_for_retrieval("pim antenna", "incorrect", llm) == "pim antenna"


def test_rewrite_falls_back_on_empty_rewrite():
    llm = lambda prompt: '{"variant": "terminology", "rewritten_query": ""}'
    assert rewrite_query_for_retrieval("pim", "ambiguous", llm) == "pim"


def test_rewrite_handles_list_reply():
    """llm_call 返回 list（项目内 LLM 调用惯例）→ 取首个元素。"""
    llm = lambda prompt: ['{"variant": "decomposition", "rewritten_query": "pim 成因"}']
    assert rewrite_query_for_retrieval("pim 是什么", "incorrect", llm) == "pim 成因"


def test_rewrite_exception_falls_back():
    def boom(prompt):
        raise RuntimeError("llm down")

    assert rewrite_query_for_retrieval("pim", "ambiguous", boom) == "pim"


# ================= adaptive_search =================

def test_adaptive_converges_in_two_rounds():
    """第一轮 incorrect → 改写 → 第二轮 correct → 停止。"""
    calls = []

    def search_fn(query, top_k):
        calls.append(query)
        if query == "pim antenna":
            return _ranked(0.05, content="unrelated physics")
        return _ranked(0.9, content="passive intermodulation antenna analysis")

    llm = lambda prompt: '{"variant": "expansion", "rewritten_query": "pim antenna rf"}'
    result = adaptive_search(search_fn, "pim antenna", llm_call=llm, max_rounds=3, top_k=5)
    assert result["grade"] == "correct"
    assert result["rounds"] == 2
    assert result["rewritten"] is True
    assert result["query"] == "pim antenna rf"
    assert len(calls) == 2  # 没有空转到第三轮


def test_adaptive_no_llm_single_round():
    """无 llm_call → 单轮评估，不改写（安全默认）。"""
    result = adaptive_search(
        lambda q, k: _ranked(0.1, content="x"), "pim antenna", llm_call=None
    )
    assert result["rounds"] == 1
    assert result["rewritten"] is False
    assert result["grade"] == "incorrect"


def test_adaptive_stops_when_rewrite_noop():
    """改写返回原查询 → 不空转，立即停止。"""
    def search_fn(query, top_k):
        return _ranked(0.1, content="x")

    llm = lambda prompt: '{"variant": "expansion", "rewritten_query": "pim antenna"}'
    result = adaptive_search(search_fn, "pim antenna", llm_call=llm, max_rounds=5)
    assert result["rounds"] == 1
    assert result["rewritten"] is False


def test_adaptive_max_rounds_cap():
    """一直 incorrect 且改写持续有新值 → 跑满 max_rounds 停止（防无限循环）。"""
    searches = {"n": 0}

    def search_fn(query, top_k):
        searches["n"] += 1
        return _ranked(0.0, content="nothing relevant")

    # 每次改写都返回新值，确保不会因"改写无进展"提前停
    # 注意: JSON 字面花括号须 {{}} 转义，否则 format 当占位符解析
    llm = lambda prompt: '{{"variant": "terminology", "rewritten_query": "variant-v{0}"}}'.format(
        searches["n"]
    )
    result = adaptive_search(search_fn, "q", llm_call=llm, max_rounds=3)
    assert searches["n"] == 3  # 精确三轮，不超
    assert result["rounds"] == 3
    assert result["grade"] == "incorrect"


def test_adaptive_tracks_evaluations():
    """evaluations 记录每一轮的 query/grade。"""
    def search_fn(query, top_k):
        return _ranked(0.9, content="pim antenna analysis")

    result = adaptive_search(search_fn, "pim antenna", llm_call=None, max_rounds=2)
    assert len(result["evaluations"]) == 1
    assert result["evaluations"][0]["query"] == "pim antenna"
    assert result["evaluations"][0]["grade"] == "correct"
