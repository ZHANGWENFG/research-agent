"""缺口①：事实性验证（attribution）单测。

依据 Attributed QA（arXiv:2112.11961）句子级回链 + Self-RAG 证据支撑判定。
锁定语义：拆句保留引用编号 [n]；术语重叠 >=2 或（数字命中+至少 1 重叠）→ supported；
空证据池 → 全 unsupported + reason="no evidence"；覆盖比率 = supported/total。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_attribution import (  # noqa: E402
    split_sentences,
    verify_article,
    verify_claim,
)

POOL = [
    {"id": 1, "content": "PIM 方法在推荐系统数据集上效果显著。"},
    {"id": 2, "content": "MMR 用于多样性重排序。"},
]


class TestSplitSentences:
    def test_chinese_golden(self):
        assert split_sentences("A 方法有效。B 方法无效。") == [
            "A 方法有效。",
            "B 方法无效。",
        ]

    def test_english_golden(self):
        assert split_sentences("First works. Second fails.") == [
            "First works.",
            "Second fails.",
        ]

    def test_mixed_punctuation(self):
        assert split_sentences("真的吗？当然！试试看。") == [
            "真的吗？",
            "当然！",
            "试试看。",
        ]

    def test_citation_not_broken(self):
        # 引用编号 [1] 位于句内，拆句不得拆断
        sentences = split_sentences("A 方法有效[1]。B 方法无效。")
        assert sentences == ["A 方法有效[1]。", "B 方法无效。"]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestVerifyClaim:
    def test_supported_full_overlap(self):
        r = verify_claim("PIM 方法效果显著[1]。", POOL)
        assert r["supported"] is True
        assert r["evidence_ids"] == [1]
        assert r["score"] > 0

    def test_unsupported_no_overlap(self):
        r = verify_claim("量子计算在气象预报中应用广泛。", POOL)
        assert r["supported"] is False
        assert r["evidence_ids"] == []
        assert "no meaningful overlap" in r["reason"]

    def test_partial_overlap_insufficient(self):
        # 只共享 1 个术语（pim）→ 低于阈值，unsupported + partial reason
        r = verify_claim("PIM 在另一领域也有应用。", POOL)
        assert r["supported"] is False
        assert "partial overlap" in r["reason"]
        assert r["overlap"] == ["pim"]

    def test_number_boost_supported(self):
        # 1 个术语重叠 + 数字 95% 命中 → 数字加分支撑（number_ok 分支）
        r = verify_claim(
            "PIM 命中率达到 95%。",
            [{"id": 7, "content": "PIM 在召回率上达到 95% 的成绩。"}],
        )
        assert r["supported"] is True
        assert r["number_hits"] == ["95%"]
        assert r["evidence_ids"] == [7]

    def test_number_without_overlap_still_unsupported(self):
        # 数字命中但没有术语重叠 → 不支撑（数字加分要求至少部分重叠）
        r = verify_claim(
            "准确率达到 95%。",
            [{"id": 8, "content": "MMR 召回率 95%。"}],
        )
        assert r["supported"] is False

    def test_no_evidence(self):
        r = verify_claim("任何句子。", [])
        assert r["supported"] is False
        assert r["reason"] == "no evidence"
        assert r["score"] == 0.0

    def test_empty_claim(self):
        r = verify_claim("", POOL)
        assert r["supported"] is False
        assert r["reason"] == "empty claim"

    def test_proper_noun_weighted(self):
        r = verify_claim("MMR 重排序效果显著。", POOL)
        assert r["proper_noun_hits"] == ["MMR"]
        assert r["supported"] is True

    def test_min_term_overlap_parameter(self):
        r = verify_claim("PIM 在另一领域也有应用。", POOL, min_term_overlap=1)
        assert r["supported"] is True


class TestVerifyArticle:
    def test_coverage_ratio(self):
        article = "PIM 方法效果显著[1]。MMR 用于重排序。这句话没有证据支撑。"
        result = verify_article(article, POOL)
        assert result["total_sentences"] == 3
        assert result["supported_count"] == 2
        assert result["coverage_ratio"] == pytest.approx(2 / 3, abs=1e-4)
        assert len(result["unsupported"]) == 1
        assert result["unsupported"][0]["reason"] == (
            "no meaningful overlap with any evidence"
        )

    def test_no_evidence_all_unsupported(self):
        result = verify_article("句子一。句子二。", [])
        assert result["total_sentences"] == 2
        assert result["supported_count"] == 0
        assert result["coverage_ratio"] == 0.0
        assert all(item["reason"] == "no evidence" for item in result["unsupported"])

    def test_citation_maps_to_evidence_id(self):
        # [1] 引用所在句子回链到 id=1 的证据（evidence_ids 映射）
        pool = [{"id": 1, "content": "PIM 方法在推荐系统数据集上效果显著。"}]
        result = verify_article("PIM 方法效果显著[1]。", pool)
        assert result["supported_count"] == 1
        assert result["results"][0]["evidence_ids"] == [1]

    def test_empty_article(self):
        result = verify_article("", POOL)
        assert result["total_sentences"] == 0
        assert result["coverage_ratio"] == 0.0
        assert result["unsupported"] == []
