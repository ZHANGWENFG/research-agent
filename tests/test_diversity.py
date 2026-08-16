"""P2-A2: MMR 多样性重排单测（research_diversity.py + search 挂接）。

MMR 出自 Carbonell & Goldstein SIGIR'98——测试锁定 λ 语义：
λ=1.0 纯相关度排序、λ=0.7 多样性与相关度权衡、λ=0.0 纯多样性。
挂接测试用离线 HashEmbedder 构造迷你索引，验证 search(diversity_lambda=...)
行为变化且缺省时零回归。
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_diversity import (
    DEFAULT_MMR_LAMBDA,
    jaccard_similarity,
    mmr_rerank,
    token_set,
)
from research_agent.research_retrieval_index import HybridPaperIndex


# ================= 基础: token_set / jaccard =================

def test_token_set_mixed():
    terms = token_set("MMR reranking 中文连续串 多样性 中文连续串")
    assert "mmr" in terms
    assert "reranking" in terms
    assert "中文连续串" in terms
    assert "多样性" in terms


def test_jaccard_identical_ones_and_disjoint_zero():
    assert jaccard_similarity("pim antenna analysis", "pim antenna analysis") == 1.0
    assert jaccard_similarity("alpha beta", "gamma delta") == 0.0


def test_jaccard_partial_overlap():
    sim = jaccard_similarity("pim antenna", "pim filter")
    assert 0.0 < sim < 1.0
    assert sim == pytest.approx(1.0 / 3.0)  # {pim} / {pim, antenna, filter}


def test_jaccard_empty_returns_zero():
    assert jaccard_similarity("", "anything") == 0.0
    assert jaccard_similarity("", "") == 0.0


# ================= MMR 核心 =================

def test_mmr_empty_and_single():
    assert mmr_rerank([]) == []
    one = [{"content": "a", "score": 0.5}]
    assert mmr_rerank(one) == one


def test_mmr_lambda_one_pure_relevance():
    """λ=1.0: 无多样性惩罚 → 完全按相关性排序。"""
    ranked = [
        {"content": "x1", "score": 0.3},
        {"content": "x2", "score": 0.9},
        {"content": "x3", "score": 0.6},
    ]
    result = mmr_rerank(ranked, lambda_=1.0)
    assert [d["score"] for d in result] == [0.9, 0.6, 0.3]


def test_mmr_lambda_zero_pure_diversity():
    """λ=0.0: 只惩罚冗余 → 最不相似者先出。"""
    ranked = [
        {"content": "pim antenna analysis", "score": 0.9},
        {"content": "pim antenna filter design", "score": 0.8},
        {"content": "muon collider optimization", "score": 0.7},
    ]
    result = mmr_rerank(ranked, lambda_=0.0)
    # 与首个选中的最不相似者（muon 篇）排第 2
    assert result[1]["content"] == "muon collider optimization"


def test_mmr_lambda_070_diversifies_redundant_top():
    """λ=0.7: 两个高度相似文档，第二篇被多样性惩罚——不同主题者提前。"""
    ranked = [
        {"content": "pim antenna analysis rf", "score": 0.95},
        {"content": "pim antenna analysis rf", "score": 0.90},   # 与第一篇近同
        {"content": "muon collider optimization", "score": 0.60},  # 主题不同
    ]
    result = mmr_rerank(ranked, lambda_=0.7, top_k=3)
    # 多样性把 muon 篇提到第 2 位（其冗余惩罚 ≈ 0）
    assert result[1]["content"] == "muon collider optimization"
    assert result[2]["content"].startswith("pim antenna")


def test_mmr_top_k_truncates():
    ranked = [
        {"content": "a1 b1", "score": i} for i in range(6, 0, -1)
    ]
    result = mmr_rerank(ranked, lambda_=1.0, top_k=3)
    assert len(result) == 3


def test_mmr_invalid_lambda_raises():
    with pytest.raises(ValueError, match="lambda"):
        mmr_rerank([{"content": "x", "score": 1.0}], lambda_=1.5)


def test_mmr_custom_similarity_injectable():
    """注入相似度函数: 用 title 字段比较而非默认 content。"""
    ranked = [
        {"title": "same title", "content": "zzz", "score": 1.0},
        {"title": "same title", "content": "zzz", "score": 0.99},
        {"title": "different", "content": "zzz", "score": 0.5},
    ]
    result = mmr_rerank(
        ranked,
        lambda_=0.5,
        text_key="title",
        similarity=lambda d1, d2: jaccard_similarity(d1["title"], d2["title"]),
    )
    assert result[1]["title"] == "different"  # 与首篇 title 相似者被降权


# ================= search 挂接 =================

class _HashEmbedder:
    """离线确定性 embedder（测试用，与 smoke 基准同思路）。"""
    name = "test-hash-embedder"
    dim = 8
    normalize = False

    def _digest(self, text):
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed(self, texts):
        return [self._digest(t) for t in texts]

    def embed_query(self, text):
        return self._digest(text)


def _mini_index():
    chunks = [
        {"chunk_id": "c1", "content": "passive intermodulation antenna analysis"},
        {"chunk_id": "c2", "content": "passive intermodulation antenna filter"},
        {"chunk_id": "c3", "content": "muon collider optimization physics"},
        {"chunk_id": "c4", "content": "deep learning medical imaging"},
    ]
    return HybridPaperIndex(chunks, _HashEmbedder())


def test_search_hybrid_returns_ranked_without_mmr():
    index = _mini_index()
    results = index.search("passive intermodulation", mode="hybrid", top_k=3)
    assert len(results) == 3
    assert all(item["retrieval_mode"] == "hybrid" for item in results)
    assert results[0]["final_rank"] == 1


def test_search_diversity_lambda_default_none_unchanged():
    """缺省 diversity_lambda → 与旧行为一致（可回滚保证）。"""
    index = _mini_index()
    plain = index.search("passive intermodulation", mode="hybrid", top_k=4)
    default = index.search(
        "passive intermodulation", mode="hybrid", top_k=4, diversity_lambda=None
    )
    assert [d["chunk_id"] for d in plain] == [d["chunk_id"] for d in default]


def test_search_diversity_reorders_redundant_top():
    """c1/c2 高度相似: λ=0.7 时 c3（不同主题）提前进入 top-2。"""
    index = _mini_index()
    plain = index.search("passive intermodulation antenna", mode="hybrid", top_k=3)
    diverse = index.search(
        "passive intermodulation antenna", mode="hybrid", top_k=3, diversity_lambda=0.7
    )
    assert plain[0]["chunk_id"] == "c1"
    # 多样性版本 top-2 里必须出现不同主题（c3 或 c4）
    diverse_ids = [d["chunk_id"] for d in diverse[:2]]
    assert "c3" in diverse_ids or "c4" in diverse_ids


def test_search_marks_mode_still_hybrid():
    """MMR 只重排不换 mode 标记——下游引用链无感。"""
    index = _mini_index()
    results = index.search(
        "passive intermodulation", mode="hybrid", top_k=3, diversity_lambda=0.7
    )
    assert all(item["retrieval_mode"] == "hybrid" for item in results)


def test_mmr_default_lambda_is_070():
    assert DEFAULT_MMR_LAMBDA == 0.7
