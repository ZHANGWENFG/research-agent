"""B2: research_qa.py（0%）与 research_retrieval_runtime.py（25%）纯函数覆盖。

证据充分性评估是"能否用现有知识库回答"的守门员——黄金样例锁定打分与放行逻辑；
runtime 的 stack/mode/embedding 决策 + LRU + 检索指标决定生产配置。
全部离线，不触碰 LLM/网络/真实索引。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_qa import (
    _append_qa_history,
    _event,
    _has_domain_anchor,
    _keyword_hits,
    _keywords_from_state,
    _meaningful_overlap,
    _terms,
    evaluate_evidence_sufficiency,
)
from research_agent.research_retrieval_runtime import (
    _IndexLRU,
    _index_cache_maxsize,
    _percentile,
    _retrieval_metrics,
    _run_dir_signature,
    _to_markdown,
    meaningful_terms,
    runtime_embedding,
    runtime_mode,
    runtime_stack,
)


# ================= research_qa.py =================

def _evidence(pairs):
    return [{"title": t, "content": c} for t, c in pairs]


def test_terms_extracts_latin_and_cjk():
    terms = _terms("pim in antenna 系统 研究")
    assert "pim" in terms
    assert "antenna" in terms
    # CJK 按单字切分; "系""研" 是停用词被过滤, "统""究" 保留
    assert "统" in terms and "究" in terms
    assert "系" not in terms and "研" not in terms
    assert "的" not in terms  # 单字停用词


def test_terms_empty():
    assert _terms("") == set()


def test_keyword_hits_substring():
    assert _keyword_hits("pim 天线 分析", ["pim", "天线"]) == ["pim", "天线"]
    assert _keyword_hits("其他内容", ["pim"]) == []
    assert _keyword_hits("x", [None, ""]) == []


def test_domain_anchor_pim():
    """PIM 问题 + expected 命中 → 域锚点成立（消歧场景放行）。"""
    assert _has_domain_anchor("pim 是什么", "topic", ["pim"], ["pim"]) is True
    assert _has_domain_anchor("pim 是什么", "", ["pim"], ["pim"]) is True  # 问题内含关键词
    assert _has_domain_anchor("其他问题", "topic", ["pim"], ["pim"]) is False  # 问题与期望词无交集


def test_meaningful_overlap_ignores_single_cjk():
    """单中文字符是噪声：只共享"系"不应算相关。"""
    assert _meaningful_overlap("pim 系统", "pim 天线") == ["pim"]
    assert _meaningful_overlap("系", "系统") == []  # 单字 vs 双字无重叠


def test_evidence_sufficiency_sufficient():
    """充足证据: 证据+引用+有意义重叠+分数达标 → 放行。"""
    result = evaluate_evidence_sufficiency(
        "pim antenna design",
        _evidence([("PIM 论文", "passive intermodulation antenna analysis")]),
        [{"id": 1}],
        topic="pim",
    )
    assert result["sufficient"] is True
    assert result["evidence_count"] == 1
    assert result["citation_count"] == 1
    assert "sufficient" in result["reason"]


def test_evidence_sufficiency_insufficient_no_evidence():
    """无证据 → 不足，不因 topic 误判。"""
    result = evaluate_evidence_sufficiency("pim antenna", [], [], topic="pim")
    assert result["sufficient"] is False
    assert result["score"] == 0


def test_evidence_sufficiency_insufficient_off_topic():
    """问题与证据无有意义重叠 → 拒绝用旧知识库回答（需重新检索）。"""
    result = evaluate_evidence_sufficiency(
        "muon collider optimization",
        _evidence([("PIM 论文", "passive intermodulation antenna analysis")]),
        [{"id": 1}],
        topic="physics",
    )
    assert result["sufficient"] is False


def test_evidence_sufficiency_forbidden_hits():
    """forbidden 词命中 → 判为消歧场景：sufficient 但标注风险。"""
    result = evaluate_evidence_sufficiency(
        "pim 和 processing in memory 的区别",
        _evidence([("PIM 论文", "passive intermodulation antenna dram analysis")]),
        [{"id": 1}],
        topic="pim",
        expected_keywords=["pim", "intermodulation"],
        forbidden_keywords=["processing in memory", "dram"],
    )
    assert result["forbidden_keyword_hits"] == ["processing in memory", "dram"]
    assert "forbidden" in result["reason"]


def test_event_structure():
    ev = _event("qa.answered", question="q")
    assert ev["event"] == "qa.answered"
    assert ev["payload"]["question"] == "q"
    assert "timestamp" in ev


def test_keywords_from_state():
    state = {"expected_keywords": ["a"], "forbidden_keywords": ["b"]}
    assert _keywords_from_state(state) == {"expected": ["a"], "forbidden": ["b"]}
    assert _keywords_from_state({}) == {"expected": [], "forbidden": []}


def test_append_qa_history_creates_and_appends(tmp_path):
    """qa_history.json 首次创建 + 再次追加。"""
    state = {"output_dir": str(tmp_path)}
    result = {"question": "q1", "answer": "a1", "citations": []}
    _append_qa_history(state, result)
    _append_qa_history(state, {"question": "q2", "answer": "a2", "citations": []})
    history = json.loads((tmp_path / "qa_history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[0]["question"] == "q1"
    assert history[1]["question"] == "q2"


def test_append_qa_history_no_output_dir():
    assert _append_qa_history({}, {"question": "q"}) == []


# ============ research_retrieval_runtime.py ============

def test_runtime_stack_override_valid():
    assert runtime_stack(override="hybrid") == "hybrid"
    assert runtime_stack(override="legacy") == "legacy"


def test_runtime_stack_override_invalid_falls_back_auto():
    value = runtime_stack(override="bogus")
    assert value in {"auto", "hybrid", "legacy"}


def test_runtime_embedding_override():
    assert runtime_embedding(override="real") == "real"
    assert runtime_embedding(override="hash") == "hash"
    assert runtime_embedding(override="nonsense") in {"auto", "real", "hash"}


def test_runtime_mode_override():
    assert runtime_mode(override="bm25") == "bm25"
    assert runtime_mode(override="hybrid_rerank") == "hybrid_rerank"
    assert runtime_mode(override="bogus") == "hybrid"  # 非法回退 hybrid


def test_index_cache_maxsize_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_RETRIEVAL_INDEX_CACHE_SIZE", "4")
    assert _index_cache_maxsize() == 4
    monkeypatch.setenv("RESEARCH_RETRIEVAL_INDEX_CACHE_SIZE", "abc")
    assert _index_cache_maxsize() == 16  # ValueError 回退


def test_index_lru_eviction():
    lru = _IndexLRU(maxsize=2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)  # 淘汰 a
    assert lru.get("a") is None
    assert lru.get("b") == 2
    lru.get("b")  # 刷新 b
    lru.put("d", 4)  # 淘汰 c（b 最近使用）
    assert lru.get("c") is None
    assert lru.get("b") == 2
    assert lru.get("d") == 4


def test_index_lru_zero_maxsize_no_eviction():
    lru = _IndexLRU(maxsize=0)
    lru.put("a", 1)
    lru.put("b", 2)
    assert lru.get("a") == 1  # maxsize=0 → 不淘汰
    assert lru.get("b") == 2


def test_meaningful_terms_filters_stop_and_single_cjk():
    terms = meaningful_terms("pim 天线 效果 如何 为什么")
    assert "pim" in terms
    assert "天线" in terms
    assert "效果" not in terms  # STOP_BIGRAMS
    assert "如" not in terms  # 单字中文字符过滤


def test_run_dir_signature_changes_with_files(tmp_path):
    """仅对喂入索引的四个命名文件签名；同名文件内容变化 → 签名变化。"""
    assert _run_dir_signature(str(tmp_path)) == ()  # 无目标文件 → 空签名
    target = tmp_path / "raw_search_results.json"
    target.write_text("{\"a\": 1}", encoding="utf-8")
    sig1 = _run_dir_signature(str(tmp_path))
    assert sig1 != ()
    target.write_text("{\"a\": 1234567890}", encoding="utf-8")  # 长度不同，签名必变
    sig2 = _run_dir_signature(str(tmp_path))
    assert sig1 != sig2  # 内容变了签名必变
    # 无关文件不影响签名
    (tmp_path / "other.txt").write_text("x" * 999, encoding="utf-8")
    assert _run_dir_signature(str(tmp_path)) == sig2


def test_retrieval_metrics_golden():
    recall, hit, mrr, ndcg = _retrieval_metrics(["a", "b", "c"], {"b"}, top_k=5)
    assert recall == pytest.approx(1.0)
    assert hit == 1
    assert mrr == pytest.approx(0.5)
    assert ndcg == pytest.approx(1.0 / math_log2(3), abs=1e-6)


def test_retrieval_metrics_partial_recall():
    recall, hit, mrr, ndcg = _retrieval_metrics(["a", "b", "c"], {"b", "d"}, top_k=5)
    assert recall == pytest.approx(0.5)
    assert hit == 1


def test_retrieval_metrics_no_hits():
    recall, hit, mrr, ndcg = _retrieval_metrics(["a", "b"], {"z"}, top_k=5)
    assert recall == 0.0 and hit == 0 and mrr == 0.0 and ndcg == 0.0


def test_retrieval_metrics_topk_caps():
    recall, hit, mrr, ndcg = _retrieval_metrics(["a", "b", "c"], {"c"}, top_k=1)
    assert hit == 0  # 相关在第 3 位，top1 没命中
    assert recall == 0.0


def test_percentile_edges():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([5, 1, 3], 0.5) == 3  # 中位数
    assert _percentile([7], 0.9) == 7  # 单值
    assert _percentile([1, 2, 3, 4], 0.0) == 1  # min
    assert _percentile([1, 2, 3, 4], 1.0) == 4  # max


def test_to_markdown_renders_table():
    report = {
        "dataset": "seed", "legacy": {"case_count": 10},
        "legacy": {"case_count": 10, "recall_at_k": 0.5, "mrr": 0.4,
                   "ndcg_at_k": 0.45, "p95_latency_ms": 12.0},
        "hybrid": {"case_count": 10, "recall_at_k": 0.7, "mrr": 0.6,
                   "ndcg_at_k": 0.65, "p95_latency_ms": 20.0},
        "deltas": {"recall_at_k": 0.2, "mrr": 0.2, "ndcg_at_k": 0.2,
                   "p95_latency_ms": 8.0, "relative_recall_gain_pct": 40.0},
    }
    md = _to_markdown(report)
    assert md.startswith("# Research Runtime Retrieval Benchmark")
    assert "| Recall@K | 0.5 | 0.7 | 0.2 |" in md
    assert "relative recall gain: 40.0%" in md


def math_log2(n):
    import math
    return math.log2(n)
