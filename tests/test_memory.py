"""长期记忆服务测试：写入 / 检索召回 / 命名空间隔离。

注意：MemoryWritePolicy 提取的记忆若置信度 < 0.85 会进候选队列，
需 consolidate_pending() 落库后才能被 search 召回；高置信事实直接持久化。
"""
import tempfile
from pathlib import Path

import pytest

from research_agent.research_longterm_memory import LongTermMemoryService


@pytest.fixture()
def memory():
    with tempfile.TemporaryDirectory() as tmp:
        yield LongTermMemoryService(str(Path(tmp)))


def test_ingest_and_search_recall(memory):
    """写入事实后，语义相关查询能召回（stable_fact 触发词：项目…使用）。"""
    outcome = memory.ingest_message(
        namespace="local-user",
        message="我的项目使用 LangGraph 做 Agent 编排，我是后端工程师。",
        subject="user",
    )
    assert outcome["status"] in ("persisted", "queued"), outcome
    if outcome["status"] == "queued":
        memory.consolidate_pending()

    hits = memory.search(namespace="local-user", query="用户做什么工作", top_k=5)
    results = hits["results"]
    assert hits["status"] == "ok"
    assert len(results) >= 1
    assert "LangGraph" in results[0]["content"]


def test_namespaces_isolated(memory):
    memory.ingest_message(namespace="ns-a", message="我的项目使用 LangGraph 做编排。", subject="user")
    memory.ingest_message(namespace="ns-b", message="我的项目使用 PyTorch 做训练。", subject="user")
    memory.consolidate_pending()

    hits_a = memory.search(namespace="ns-a", query="LangGraph", top_k=5)
    results_a = hits_a["results"]
    assert results_a and "LangGraph" in results_a[0]["content"]

    hits_b = memory.search(namespace="ns-b", query="LangGraph", top_k=5)
    for item in hits_b["results"]:
        assert "LangGraph" not in item["content"]


def test_direct_upsert_immediately_searchable(memory):
    """upsert 直接落库，无需 consolidate。"""
    memory.upsert(
        namespace="n3",
        memory_type="semantic",
        subject="user",
        content="用户坐标北京，关注大模型应用。",
        canonical_key="user-location",
        confidence=0.95,
    )
    hits = memory.search(namespace="n3", query="用户在哪里", top_k=5)
    assert hits["results"], "upsert 后应立即可检索"


def test_storage_info_reports_counts(memory):
    info = memory.storage_info()
    # 至少包含基础键（表行数 / 目录结构）
    assert isinstance(info, dict)
    assert len(info) > 0
