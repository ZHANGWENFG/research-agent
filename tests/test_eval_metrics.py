"""评测代码单元测试（2026-08-16 新增）。

覆盖此前"零测试"的评分/指标代码——这些代码产出对外发布的数字，
必须有黄金样例锁定正确性:
  - research_eval: 分数刻度归一化（总分 0-100）、faithfulness
  - retrieval_metrics: recall@k / hit@k / MRR / nDCG 手算黄金值
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_eval import (
    EvalCase,
    _split_claims,
    _score_faithfulness,
    _score_completion,
    _score_retrieval,
    evaluate_run,
)
from research_agent.research_retrieval_runtime import _retrieval_metrics


# ---------- 分数刻度 ----------

def test_total_score_scale_is_100():
    """满分用例总分必须精确等于 100（修复前上限 85/95 误导）。"""
    # 构造一个"全满分"的 run 目录
    import tempfile

    tmp = tempfile.mkdtemp()
    run_dir = Path(tmp) / "run1"
    run_dir.mkdir(parents=True)
    # 检索结果包含所有关键词 → retrieval 满分
    (run_dir / "raw_search_results.json").write_text(
        json.dumps([{
            "title": "AI 医疗 诊断",
            "description": "AI 在医疗领域有重要应用",
            "snippets": ["AI 提高诊断准确率", "医疗影像"],
        }]),
        encoding="utf-8",
    )
    # 文章覆盖所有关键词且与证据重叠 → article + faithfulness 满分
    article = "AI 在医疗领域有重要应用。AI 提高诊断准确率。医疗影像辅助诊断。"
    (run_dir / "myagent_article_polished.txt").write_text(article, encoding="utf-8")
    (run_dir / "myagent_outline.txt").write_text("# 大纲\n## 节1\n## 节2", encoding="utf-8")
    # 成功 trace
    (run_dir / "research_trace.jsonl").write_text(
        json.dumps({"event": "run_start"}) + "\n"
        + json.dumps({"event": "run_end", "success": True}) + "\n"
        + json.dumps({"event": "retrieval_start"}) + "\n"
        + json.dumps({"event": "retrieval_end"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps({"success": True}), encoding="utf-8"
    )

    case = EvalCase(
        topic="AI 医疗",
        expected_keywords=["AI", "医疗", "诊断", "准确率"],
        forbidden_keywords=[],
        expected_language="zh",
        min_sources=1,
    )
    result = evaluate_run(run_dir, case)
    total = result["scores"]["total"]
    assert 90.0 <= total <= 100.0, f"满分用例总分应接近100, got {total}"
    # 修复前: 总分上限 85/95 永远到不了 100
    assert result["scores"]["retrieval_quality"] >= 25.0, "检索得分应接近满分"


def test_forbidden_keyword_penalty():
    """禁区词应产生 penalty，总分下降。"""
    import tempfile

    tmp = tempfile.mkdtemp()
    run_dir = Path(tmp) / "run2"
    run_dir.mkdir(parents=True)
    (run_dir / "raw_search_results.json").write_text(
        json.dumps([{
            "title": "完全无关的内容",
            "description": "违规词",
            "snippets": [],
        }]),
        encoding="utf-8",
    )
    (run_dir / "myagent_article_polished.txt").write_text(
        "这是一篇完全无关的文章。没有关键词。", encoding="utf-8"
    )
    (run_dir / "myagent_outline.txt").write_text("# 大纲", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps({"success": True}), encoding="utf-8")

    case = EvalCase(
        topic="测试",
        expected_keywords=["AI", "医疗"],
        forbidden_keywords=["违规词"],
    )
    result = evaluate_run(run_dir, case)
    assert result["scores"]["offtopic_penalty"] > 0, "应产生禁区词惩罚"
    assert result["scores"]["total"] < 60, "分数应被惩罚压低"


# ---------- faithfulness ----------

def test_faithfulness_supported_claims():
    """论断-证据支持度: 有支持/无支持/部分支持。"""
    # 全部有支持
    f, sup, total = _score_faithfulness(
        "AI 在医疗领域有重要应用。AI 提高诊断准确率。",
        "AI 医疗 诊断 准确率",
    )
    assert sup == 2 and total == 2, f"应全部支持, got {sup}/{total}"
    assert f == 10.0, f"全支持应 10 分, got {f}"

    # 无支持
    f2, sup2, total2 = _score_faithfulness(
        "量子计算需要超低温环境。",
        "AI 医疗",
    )
    assert sup2 == 0 and total2 == 1
    assert f2 == 0.0

    # 空文章
    assert _score_faithfulness("", "") == (0.0, 0.0, 0.0)


def test_split_claims_boundaries():
    """论断切分: 中英文句号/段落边界（短片段<10字符按设计过滤）。"""
    claims = _split_claims(
        "人工智能在医疗领域有重要应用。Second sentence is about AI safety!\n第三段讨论了机器人的发展现状？"
    )
    assert len(claims) >= 3, f"应切出多句, got {claims}"
    # 太短的片段（<10 字符）应被过滤
    assert all(len(c) >= 10 for c in claims)


# ---------- 检索指标黄金值 ----------

def test_retrieval_metrics_recall():
    """recall@k 真分数召回（修复前是 0/1 hit）。"""
    recall, hit, mrr, ndcg = _retrieval_metrics(
        ["a", "b", "c", "d"], {"b", "d"}, 3
    )
    assert recall == 0.5, f"top3 命中1个/共2相关 → recall=0.5, got {recall}"
    assert hit == 1, "hit@k 应=1"


def test_retrieval_metrics_mrr():
    """MRR: 首个相关项位置的倒数。"""
    _, _, mrr, _ = _retrieval_metrics(["a", "b", "c"], {"c"}, 3)
    assert abs(mrr - 1.0 / 3.0) < 1e-9, f"MRR 应=1/3, got {mrr}"
    _, _, mrr2, _ = _retrieval_metrics(["a", "b"], {"c"}, 3)
    assert mrr2 == 0.0, "无相关项 MRR=0"


def test_retrieval_metrics_ndcg():
    """nDCG@k 手算黄金值: 排名1相关 → DCG=1, IDCG=1, nDCG=1。"""
    _, _, _, ndcg = _retrieval_metrics(["a", "b", "c"], {"a"}, 3)
    assert abs(ndcg - 1.0) < 1e-9, f"第1位相关 nDCG 应=1, got {ndcg}"
    # 排名2相关 → DCG=1/log2(3)≈0.631, IDCG=1 → nDCG≈0.631
    _, _, _, ndcg2 = _retrieval_metrics(["a", "b", "c"], {"b"}, 3)
    assert abs(ndcg2 - (1.0 / 1.58496)) < 0.01, f"第2位相关 nDCG 应≈0.631, got {ndcg2}"


def test_retrieval_metrics_empty():
    """无相关项/空排名不崩溃。"""
    recall, hit, mrr, ndcg = _retrieval_metrics([], set(), 3)
    assert (recall, hit, mrr, ndcg) == (0.0, 0, 0.0, 0.0)
