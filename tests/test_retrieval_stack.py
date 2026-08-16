"""检索栈 v4.1 单元测试（2026-08-16 新增）。

覆盖此前"只靠 benchmark 间接验证"的高风险路径:
  - RRF 融合（权重/缺 chunk_id/tie-break）
  - multilingual_tokenize（中英混合/空串）
  - HybridPaperIndex 持久化往返（manifest 校验）
  - 空查询防护
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_retrieval_index import (
    HybridPaperIndex,
    reciprocal_rank_fusion,
    multilingual_tokenize,
)
from research_agent.research_retrieval_common import (
    HashEmbeddingProvider,
    tokenize,
)


# ---------- RRF 融合 ----------

def test_rrf_basic_fusion_order():
    """标准 RRF: 在两路排名中都靠前的 chunk 应排最前。"""
    bm25 = [
        {"chunk_id": "a", "bm25_score": 8.0},
        {"chunk_id": "b", "bm25_score": 7.0},
    ]
    dense = [
        {"chunk_id": "b", "dense_score": 0.9},
        {"chunk_id": "a", "dense_score": 0.8},
    ]
    fused = reciprocal_rank_fusion([bm25, dense])
    # a: 1/61 + 1/62 ≈ 0.0325;  b: 1/62 + 1/61 相同 → tie-break 按 chunk_id
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "a"
    assert fused[1]["chunk_id"] == "b"
    # rrf_score 范围 (0, 0.1]
    for item in fused:
        assert 0 < item["rrf_score"] <= 0.1


def test_rrf_weights_affect_order():
    """带权重: 权重高的一路主导融合结果。"""
    rankings = [
        [{"chunk_id": "a", "bm25_score": 1.0}],
        [{"chunk_id": "b", "dense_score": 0.5}],
    ]
    # 权重给第一路 -> a 优先
    fused_a = reciprocal_rank_fusion(rankings, weights=[10.0, 0.0])
    assert fused_a[0]["chunk_id"] == "a"
    # 权重给第二路 -> b 优先
    fused_b = reciprocal_rank_fusion(rankings, weights=[0.0, 10.0])
    assert fused_b[0]["chunk_id"] == "b"


def test_rrf_skips_missing_chunk_id():
    """缺 chunk_id 的条目应被安全跳过（不崩溃）。"""
    rankings = [
        [{"chunk_id": "a"}, {"bm25_score": 1.0}],  # 第二项无 chunk_id
    ]
    fused = reciprocal_rank_fusion(rankings)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "a"


def test_rrf_preserves_source_scores():
    """融合时保留双源分数（bm25_score/dense_score）供下游 gate 使用。"""
    fused = reciprocal_rank_fusion([
        [{"chunk_id": "a", "bm25_score": 5.0}],
        [{"chunk_id": "a", "dense_score": 0.7}],
    ])
    assert fused[0]["bm25_score"] == 5.0
    assert fused[0]["dense_score"] == 0.7


def test_rrf_weight_mismatch_raises():
    """权重数与排名数不匹配应抛 ValueError。"""
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[{"chunk_id": "a"}]], weights=[1.0, 2.0])


# ---------- 分词 ----------

def test_multilingual_tokenize_mixed():
    """中英混合: 英文词 + CJK unigram/bigram 都应出现。"""
    tokens = multilingual_tokenize("AI in 医疗领域")
    assert "ai" in tokens
    assert "in" in tokens
    assert "医疗" in tokens
    assert "领域" in tokens
    assert "疗领" in tokens  # bigram 滑动


def test_multilingual_tokenize_empty():
    """空串/空白返回空列表。"""
    assert multilingual_tokenize("") == []
    assert multilingual_tokenize("   ") == []


def test_tokenize_consistency_with_multilingual():
    """统一后的 tokenize 应与 multilingual_tokenize 语义一致（CJK bigram）。"""
    t = tokenize("医疗领域的AI应用")
    mt = set(multilingual_tokenize("医疗领域的AI应用"))
    assert t <= mt, f"common.tokenize 应覆盖 multilingual 的词项: {t} vs {mt}"
    assert "医疗" in t and "领域" in t


# ---------- 持久化往返 ----------

def _build_small_index(tmp_path):
    documents = [
        {"document_id": "d1", "text": "深度学习在医疗影像诊断中的应用。"},
        {"document_id": "d2", "text": "Transformer 架构是 NLP 的基础。"},
        {"document_id": "d3", "text": "强化学习在机器人控制中的进展。"},
    ]
    return HybridPaperIndex.from_documents(
        documents,
        embedding_provider=HashEmbeddingProvider(dim=32),
        chunk_size=200,
        chunk_overlap=40,
    )


def _load_with_provider(path):
    """用构建时的同款 provider 加载（load 需要 provider 做一致性校验）。"""
    return HybridPaperIndex.load(path, embedding_provider=HashEmbeddingProvider(dim=32))


def test_index_save_load_roundtrip(tmp_path):
    """save→load 往返: 检索结果一致（可复现性）。"""
    index = _build_small_index(tmp_path)
    save_path = tmp_path / "index.json"
    index.save(str(save_path))

    loaded = _load_with_provider(str(save_path))
    assert loaded is not None
    assert loaded.schema_version == index.schema_version

    q = "医疗影像诊断"
    before = [str(i.get("chunk_id")) for i in index.search(q, top_k=3)]
    after = [str(i.get("chunk_id")) for i in loaded.search(q, top_k=3)]
    assert before == after, f"往返检索结果应一致: {before} vs {after}"


def test_index_load_rejects_wrong_model(tmp_path):
    """换嵌入模型后 load 应拒绝（防止误用旧索引）。"""
    index = _build_small_index(tmp_path)
    save_path = tmp_path / "index.json"
    index.save(str(save_path))

    # 篡改 manifest 里的模型名，模拟换了模型
    import pytest

    payload = json.loads(save_path.read_text(encoding="utf-8"))
    payload["manifest"]["embedding_model"] = "different-model"
    save_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding model mismatch"):
        _load_with_provider(str(save_path))


def test_index_load_rejects_wrong_schema(tmp_path):
    """schema_version 不匹配应拒绝加载（持久化格式演进安全）。"""
    index = _build_small_index(tmp_path)
    save_path = tmp_path / "index.json"
    index.save(str(save_path))

    import pytest

    payload = json.loads(save_path.read_text(encoding="utf-8"))
    payload["manifest"]["schema_version"] = "research-hybrid-index-v9.9"
    save_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        _load_with_provider(str(save_path))


# ---------- 空查询防护 ----------

def test_empty_query_returns_empty(tmp_path):
    """空查询/纯停用词应返回空结果，而不是任意 chunk（2026-08-16 修复）。"""
    index = _build_small_index(tmp_path)
    # 空查询
    assert index.search("", top_k=3) == []
    # 纯停用词（全部在 _QUERY_STOP_TOKENS 中）
    assert index.search("的 与 和 是", top_k=3) == []
