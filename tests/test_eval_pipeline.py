"""research_eval_pipeline 单元测试（2026-08-16 新增）。

此前该模块 372 行 0% 覆盖——评测流水线（选型/门禁/置信区间/人工审核存储）
是产出对外数字的最后一环，必须有黄金样例锁定。全部离线，不依赖 LLM/网络。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_eval_pipeline import (
    AnnotationStore,
    bootstrap_mean_ci,
    enrich_context_cases,
    normalize_corpus,
    paired_score_delta,
    ranked_document_ids,
    reranker_gate,
    retrieval_metrics,
    sanitize_report,
    select_deployable_configuration,
    select_dev_configuration,
    summarize_retrieval_cases,
    validate_review,
)


# ---------- corpus 归一化 ----------

def test_normalize_corpus_contract():
    """旧版 corpus 记录还原为 chunk 契约（chunk_id/document_id/content）。"""
    dataset = {
        "corpus": [
            {
                "document_id": "doc1",
                "title": "Paper A",
                "text": "正文",
                "metadata": {"context": "检索用文本", "source_document_id": "doc1"},
            }
        ]
    }
    chunks = normalize_corpus(dataset)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "doc1"
    assert chunks[0]["document_id"] == "doc1"
    assert chunks[0]["retrieval_content"] == "检索用文本"  # metadata.context 优先


def test_normalize_corpus_empty():
    assert normalize_corpus({}) == []


# ---------- 上下文证据富化 ----------

def test_enrich_context_cases_attaches_sections():
    """同一篇论文的多段证据按 chunk 附加到用例。"""
    chunks = [
        {"document_id": "d1", "chunk_id": "c1", "content": "第一段"},
        {"document_id": "d1", "chunk_id": "c2", "content": "第二段"},
        {"document_id": "d2", "chunk_id": "c3", "content": "别的论文"},
    ]
    cases = [{"query": "q", "metadata": {"source_document_id": "d1"}}]
    enriched = enrich_context_cases(cases, chunks, max_chunks=2)
    assert "[page ? | c1]" in enriched[0]["context_evidence"]
    assert "第一段" in enriched[0]["context_evidence"]
    assert "别的论文" not in enriched[0]["context_evidence"]  # 只富化同源


# ---------- 排名指标黄金值 ----------

def test_retrieval_metrics_golden_values():
    """TREC 口径黄金值: 排名 [d1,d2,d3] 相关 {d1} → recall@5=1/1, mrr=1, ndcg=1。"""
    metrics = retrieval_metrics(["d1", "d2", "d3"], ["d1"], top_k=5)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_5"] == 1.0
    assert metrics["first_relevant_rank"] == 1


def test_retrieval_metrics_second_rank_ndcg():
    """相关在第 2 位: nDCG@5 = (1/log2(3)) / 1 ≈ 0.6309。"""
    metrics = retrieval_metrics(["a", "b", "c"], ["b"], top_k=5)
    assert metrics["first_relevant_rank"] == 2
    assert abs(metrics["mrr"] - 0.5) < 1e-6
    assert abs(metrics["ndcg_at_5"] - (1.0 / 1.58496)) < 0.01


def test_retrieval_metrics_partial_recall():
    """2 个相关命中 1 个 → recall_at_5=0.5。"""
    metrics = retrieval_metrics(["d1", "d2", "d3"], ["d1", "d4"], top_k=5)
    assert metrics["recall_at_5"] == 0.5
    assert metrics["precision_at_5"] == 0.2


def test_retrieval_metrics_no_hits():
    metrics = retrieval_metrics(["a", "b"], ["c"], top_k=5)
    assert metrics["mrr"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0
    assert metrics["recall_at_5"] == 0.0


# ---------- 聚合与置信区间 ----------

def test_summarize_retrieval_cases_mean_and_ci():
    """聚合: 均值正确、置信区间有界（均值在 low/high 之间）。"""
    per_case = [
        {"recall_at_5": 1.0, "recall_at_10": 1.0, "precision_at_5": 0.2,
         "mrr": 1.0, "ndcg_at_5": 1.0, "latency_ms": 10},
        {"recall_at_5": 0.0, "recall_at_10": 0.5, "precision_at_5": 0.0,
         "mrr": 0.0, "ndcg_at_5": 0.0, "latency_ms": 20},
    ]
    summary = summarize_retrieval_cases(per_case, bootstrap_samples=200, seed=7)
    assert summary["case_count"] == 2
    assert summary["recall_at_5"] == 0.5
    assert summary["p50_latency_ms"] == 15.0
    ci = summary["confidence_intervals"]["mrr"]
    assert ci["low"] <= ci["mean"] <= ci["high"]


def test_bootstrap_mean_ci_single_value():
    """单值抽样: 均值=该值，区间=[该值,该值]。"""
    ci = bootstrap_mean_ci([0.5], samples=100)
    assert ci["mean"] == 0.5
    assert ci["low"] == 0.5 and ci["high"] == 0.5


def test_bootstrap_mean_ci_empty():
    ci = bootstrap_mean_ci([])
    assert ci["mean"] == 0.0 and ci["n"] == 0


def test_paired_score_delta():
    """成对差异: 赢/平/输计数与均值。"""
    delta = paired_score_delta([0.5, 0.5, 0.6], [0.7, 0.5, 0.4])
    assert delta["wins"] == 1
    assert delta["ties"] == 1
    assert delta["losses"] == 1
    assert abs(delta["mean_delta"] - (0.2 + 0.0 - 0.2) / 3) < 1e-9


# ---------- 配置选择与 reranker 门禁 ----------

def _report(ndcg, mrr, recall, p95=50):
    return {"ndcg_at_5": ndcg, "mrr": mrr, "recall_at_5": recall, "p95_latency_ms": p95}


def test_select_dev_configuration_prefers_quality():
    """dev 选型: nDCG 最高的配置胜出（质量优先）。"""
    reports = {
        "bm25": _report(0.6, 0.5, 0.7),
        "hybrid": _report(0.7, 0.6, 0.8),
    }
    assert select_dev_configuration(reports) == "hybrid"


def test_select_dev_configuration_empty_raises():
    with pytest.raises(ValueError):
        select_dev_configuration({})


def test_reranker_gate_allows_when_better():
    """重排质量提升且延迟达标 → 放行。"""
    gate = reranker_gate(
        _report(0.6, 0.5, 0.7, p95=100),
        _report(0.7, 0.6, 0.69, p95=200),
    )
    assert gate["enabled"] is True


def test_reranker_gate_rejects_latency():
    """重排超延迟预算 → 拒绝（即使质量提升）。"""
    gate = reranker_gate(
        _report(0.6, 0.5, 0.7, p95=100),
        _report(0.7, 0.6, 0.69, p95=2000),
        latency_budget_ms=500,
    )
    assert gate["enabled"] is False
    assert "延迟" in gate["reason"]


def test_reranker_gate_rejects_recall_drop():
    """重排导致 Recall@5 下降超容忍 → 拒绝。"""
    gate = reranker_gate(
        _report(0.6, 0.5, 0.8),
        _report(0.7, 0.6, 0.7),  # recall 掉 0.1 > 0.02
        max_recall_drop=0.02,
    )
    assert gate["enabled"] is False


def test_select_deployable_configuration_no_rerank():
    """重排不是质量最优时直接选质量最优（不触发 gate）。"""
    reports = {"bm25": _report(0.6, 0.5, 0.7), "hybrid": _report(0.7, 0.6, 0.8)}
    selection = select_deployable_configuration(reports)
    assert selection["selected"] == "hybrid"
    assert selection["reranker_gate"]["enabled"] is False


# ---------- 人工审核 ----------

def test_validate_review_roundtrip():
    """合法审核记录规范化: 状态/时间戳补齐。"""
    review = validate_review(
        {
            "case_id": "c1",
            "query_validity": "needs_edit",
            "edited_query": "改后问题",
            "relevant_document_ids": ["d1"],
            "evidence_sufficiency": "sufficient",
            "reviewer_notes": "note",
        }
    )
    assert review["review_status"] == "reviewed"
    assert review["query_validity"] == "needs_edit"
    assert review["relevant_document_ids"] == ["d1"]


@pytest.mark.parametrize(
    "review,error",
    [
        ({"query_validity": "valid"}, "case_id"),
        ({"case_id": "c", "query_validity": "bad"}, "问题有效性"),
        ({"case_id": "c", "query_validity": "needs_edit"}, "修改后的问题"),
        ({"case_id": "c", "query_validity": "valid"}, "相关论文"),
        ({"case_id": "c", "query_validity": "valid", "relevant_document_ids": ["d1"],
          "evidence_sufficiency": "bad"}, "证据充分性"),
    ],
)
def test_validate_review_rejects(review, error):
    with pytest.raises(ValueError, match=error):
        validate_review(review)


def _tiny_dataset(case_id="c1", domain="ai", split="test"):
    return {
        "metadata": {"dataset_sha256": "abc123"},
        "cases": [
            {
                "case_id": case_id,
                "split": split,
                "query": "问题",
                "relevant_document_ids": ["d1"],
                "metadata": {"domain": domain},
            }
        ],
    }


def test_annotation_store_save_and_progress(tmp_path):
    """保存审核 → progress 信任级别推进（stale 计数/冻结门禁）。"""
    store = AnnotationStore(
        str(tmp_path), _tiny_dataset(),
        min_reviewed_frozen=1, min_cases_per_domain=1,
    )
    store.save_review(
        {
            "case_id": "c1",
            "query_validity": "valid",
            "relevant_document_ids": ["d1"],
            "evidence_sufficiency": "sufficient",
        }
    )
    progress = store.progress()
    assert progress["reviewed_count"] == 1
    assert progress["valid_reviewed_test_count"] == 1
    assert progress["frozen_test_allowed"] is True
    assert progress["trust_level"] == "release_ready"


def test_annotation_store_rejects_unknown_case(tmp_path):
    store = AnnotationStore(str(tmp_path), _tiny_dataset())
    with pytest.raises(ValueError, match="找不到待审核用例"):
        store.save_review(
            {
                "case_id": "nope",
                "query_validity": "valid",
                "relevant_document_ids": ["d1"],
                "evidence_sufficiency": "sufficient",
            }
        )


def test_annotation_store_stale_review_detected(tmp_path):
    """换数据集后旧审核标记 stale，不污染新数据集的信任级别。"""
    store = AnnotationStore(str(tmp_path), _tiny_dataset())
    store.save_review(
        {
            "case_id": "c1",
            "query_validity": "valid",
            "relevant_document_ids": ["d1"],
            "evidence_sufficiency": "sufficient",
        }
    )
    # 数据集变了（sha256 不同）→ 审核变 stale
    dataset2 = _tiny_dataset(case_id="c2")
    dataset2["metadata"]["dataset_sha256"] = "different-hash"
    store2 = AnnotationStore(str(tmp_path), dataset2)
    progress = store2.progress()
    assert progress["stale_review_count"] == 1
    assert progress["reviewed_count"] == 0  # 当前数据集下无有效审核


def test_annotation_store_export_reviewed(tmp_path):
    """导出审核后数据集: 只含 valid/needs_edit，且注入审核标签。"""
    store = AnnotationStore(str(tmp_path), _tiny_dataset())
    store.save_review(
        {
            "case_id": "c1",
            "query_validity": "valid",
            "relevant_document_ids": ["d1"],
            "evidence_sufficiency": "sufficient",
        }
    )
    exported = store.export_reviewed_dataset()
    assert exported["metadata"]["annotation_status"] == "human_reviewed"
    assert len(exported["cases"]) == 1
    assert exported["cases"][0]["review"]["query_validity"] == "valid"


def test_annotation_store_jsonl_roundtrip_durable(tmp_path):
    """审核落盘为 JSONL 且原子写（tmp+replace）：重开 store 仍可读。"""
    store = AnnotationStore(str(tmp_path), _tiny_dataset())
    store.save_review(
        {
            "case_id": "c1",
            "query_validity": "valid",
            "relevant_document_ids": ["d1"],
            "evidence_sufficiency": "sufficient",
        }
    )
    # 重新打开（模拟重启）
    store2 = AnnotationStore(str(tmp_path), _tiny_dataset())
    assert store2.progress()["reviewed_count"] == 1
    lines = (Path(tmp_path) / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert not (Path(tmp_path) / "reviews.jsonl.tmp").exists()  # 无残留 tmp


# ---------- 报告净化 ----------

def test_sanitize_report_removes_private_fields():
    report = {
        "dataset_path": "/private/path",
        "zotero_root": "zotero://x",
        "evidence": {"excerpt": "秘密"},
        "keep": {"nested": {"path": "也要删"}},
        "fine": "保留",
    }
    clean = sanitize_report(report)
    assert "dataset_path" not in clean
    assert "evidence" not in clean
    assert "path" not in clean["keep"]["nested"]
    assert clean["fine"] == "保留"


# ---------- 工具函数 ----------

def test_term_retention_ratio():
    """术语保留率: 部分命中按比例。"""
    from research_agent.research_eval_pipeline import _term_retention

    assert _term_retention("包含医疗和影像的文本", ["医疗", "影像"]) == 1.0
    assert _term_retention("只包含医疗", ["医疗", "影像"]) == 0.5
    assert _term_retention("无命中", ["医疗", "影像"]) == 0.0
    assert _term_retention("任意", []) == 1.0  # 空术语恒 1
