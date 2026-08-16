"""低覆盖模块离线测试（B1, 2026-08-16 新增）。

rm.py（16%）: ArxivRM 的 Atom XML 解析 + 查询归一化 + 相关性过滤（纯函数，无网络）
research_kb_qa.py（19%）: 答案组合/引用构造/提示词/chunk 转换/切分（纯函数）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.rm import ArxivRM
from research_agent.research_kb_qa import (
    _citation_from_doc,
    _compose_answer,
    _first_sentence,
    _kb_answer_prompt,
    _memory_hint,
    _rag_chunk_to_doc,
    _split_paragraphs,
    write_qa_artifact,
)

# ---------- ArxivRM: Atom XML 解析（离线） ----------

_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345</id>
    <title>  Deep Learning  for  Medical Imaging  </title>
    <summary>  This paper  studies AI  diagnostics.  </summary>
    <published>2024-01-01T00:00:00Z</published>
    <updated>2024-01-02T00:00:00Z</updated>
    <author><name>  Alice  Zhang  </name></author>
    <author><name>Bob Li</name></author>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CV"/>
    <category term="eess.IV"/>
    <link rel="alternate" href="http://arxiv.org/abs/2401.12345"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.99999</id>
    <title>Second Paper</title>
    <summary>Another abstract.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2403.11111</id>
  </entry>
</feed>
"""


def test_arxiv_parse_response_full_entry():
    """完整条目: 标题/摘要/作者/分类/pdf 链接全部解析。"""
    results = ArxivRM()._parse_response(_ARXIV_XML)
    first = results[0]
    assert first["title"] == "Deep Learning for Medical Imaging"  # 空白归一化
    assert first["description"] == "This paper studies AI diagnostics."
    assert first["meta"]["authors"] == ["Alice Zhang", "Bob Li"]
    assert first["meta"]["primary_category"] == "cs.CV"
    assert first["meta"]["categories"] == ["cs.CV", "eess.IV"]
    assert first["meta"]["pdf_url"] == "http://arxiv.org/pdf/2401.12345"
    assert first["meta"]["source_type"] == "arxiv"
    assert first["snippets"] == ["This paper studies AI diagnostics."]


def test_arxiv_parse_skips_incomplete_entry():
    """缺 url/title/abstract 的条目被跳过（feed 内残缺 entry 不产出脏数据）。"""
    results = ArxivRM()._parse_response(_ARXIV_XML)
    assert len(results) == 2  # 第三个 entry 只有 id → 跳过


def test_arxiv_parse_second_entry():
    """第二个条目（只有 id/title/summary）也能解析。"""
    results = ArxivRM()._parse_response(_ARXIV_XML)
    second = results[1]
    assert second["title"] == "Second Paper"
    assert second["meta"]["authors"] == []


def test_arxiv_normalize_query():
    """查询归一化: 空白压缩、大小写处理、去括号。"""
    assert ArxivRM._normalize_query_for_arxiv("  deep   learning  ") == \
        "deep learning"
    assert ArxivRM._normalize_query_for_arxiv("") == ""
    assert ArxivRM._normalize_query_for_arxiv("   ") == ""


def test_arxiv_relevance_filter_pim():
    """PIM 专有过滤: 无 PIM 关键词时恒放行；PIM 查询时过滤 off-topic。"""
    # 非 PIM 查询 → 恒 True
    assert ArxivRM._is_result_relevant_to_query("AI 医疗", {"title": "x", "description": "y"})
    # PIM 查询 + 匹配被动互调术语 → True
    hit = {
        "title": "Passive intermodulation in antenna",
        "description": "radio frequency analysis",
        "snippets": [],
    }
    assert ArxivRM._is_result_relevant_to_query("passive intermodulation", hit)
    # PIM 查询 + 命中 off-topic（processing-in-memory）→ False
    off_topic = {
        "title": "Passive intermodulation in DRAM",
        "description": "processing-in-memory architecture",
        "snippets": [],
    }
    assert not ArxivRM._is_result_relevant_to_query("passive intermodulation", off_topic)


# ---------- kb_qa: 答案组合 ----------

def test_compose_answer_with_evidence():
    """带证据: 首句 + 引用编号 [1][2]，记忆提示前置。"""
    answer = _compose_answer(
        "问题",
        [
            {"content": "第一句证据内容。第二句不要。", "title": "A"},
            {"content": "另一句。", "title": "B"},
        ],
        {"semantic": [{"content": "记忆提示语"}]},
    )
    assert answer.startswith("记忆提示语")
    assert "第一句证据内容。[1]" in answer
    assert "另一句。[2]" in answer
    assert "第二句不要" not in answer  # 只取首句


def test_compose_answer_no_evidence():
    """无证据: 返回明说的"找不到证据"。"""
    assert _compose_answer("q", [], {}) == "没有找到足够证据回答该问题。"


def test_memory_hint_layer_priority():
    """记忆提示按 semantic → episodic → working 优先级取第一条。"""
    assert _memory_hint({"working": [{"content": "w"}]}) == "w"
    assert _memory_hint({"episodic": [{"content": "e"}], "working": [{"content": "w"}]}) == "e"
    assert _memory_hint({}) == ""


def test_citation_from_doc():
    """引用对象: 字段映射完整，score 默认 0。"""
    citation = _citation_from_doc(3, {"title": "T", "url": "http://x", "source": "arxiv"})
    assert citation["id"] == 3
    assert citation["title"] == "T"
    assert citation["source_type"] == "arxiv"
    assert citation["score"] == 0


def test_kb_answer_prompt_limits_evidence():
    """提示词最多带 6 条证据，且截断超长文本。"""
    evidence = [{"title": "T{0}".format(i), "content": "内容" * 100} for i in range(10)]
    prompt = _kb_answer_prompt("问题", evidence)
    assert "[6]" in prompt
    assert "[7]" not in prompt  # 第 7 条不进提示词


def test_rag_chunk_to_doc_contract():
    """RAG chunk → 文档: 检索模式/分数全字段保留。"""
    doc = _rag_chunk_to_doc(
        {"chunk_id": "c1", "title": "T", "content": "C", "url": "u",
         "source_type": "retrieval", "rrf_score": 0.5, "retrieval_mode": "hybrid"}
    )
    assert doc["id"] == "c1"
    assert doc["rrf_score"] == 0.5
    assert doc["retrieval_mode"] == "hybrid"
    assert doc["source"] == "retrieval"


def test_write_qa_artifact(tmp_path):
    """问答产物落盘: JSON 可读回。"""
    path = write_qa_artifact(str(tmp_path), {"answer": "答案", "citations": []})
    assert path.name == "qa_answer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["answer"] == "答案"


def test_split_paragraphs_skips_headings():
    """段落切分: 空行分割，跳过 # 标题行。"""
    text = "# 标题\n\n第一段内容。\n\n第二段内容。"
    paragraphs = _split_paragraphs(text)
    assert paragraphs == ["第一段内容。", "第二段内容。"]


def test_first_sentence_extracts_chinese_period():
    """首句提取: 中英文句号都认，截断兜底 240 字符。"""
    assert _first_sentence("第一句。第二句") == "第一句。"
    assert _first_sentence("First sentence. Second") == "First sentence."
    assert _first_sentence("") == ""
    long_text = "长" * 300
    assert len(_first_sentence(long_text)) <= 240
