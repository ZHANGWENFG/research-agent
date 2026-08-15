"""上下文引擎测试：token 计量 / 压缩 / 100% 还原 / 会话集成。

ContextEngine（兼容壳）→ ContextEngineCore 为核心实现。
compact 与 restore 必须共用同一个 ContextEventStore（ledger 持有原始消息）。
"""
import tempfile
from pathlib import Path

import pytest

from research_agent.research_context import (
    ContextEngine,
    ContextEngineConfig,
    ContextEventStore,
    estimate_tokens,
)


def _message(role, content, message_id=None, metadata=None):
    return {
        "id": message_id or f"m-{role}-{len(content)}",
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }


def _long_history(limit=24):
    messages = [
        _message("system", "你是论文调研助手。", message_id="sys-1", metadata={"pinned": True}),
        _message("user", "帮我调研 LangGraph 是什么。", message_id="m-1"),
        _message("assistant", "LangGraph 是一个基于图的状态编排框架……", message_id="m-2"),
    ]
    for i in range(3, limit):
        messages.append(_message(
            "user" if i % 2 else "assistant",
            f"第 {i} 条消息：关于论文调研的详细讨论内容，包含检索、证据和引用。",
            message_id=f"m-{i}",
        ))
    return messages


def _engine_with_store(tmp):
    """带 SQLite ledger 的引擎：compact 产物可 restore。"""
    store = ContextEventStore(str(Path(tmp) / "events.jsonl"))
    config = ContextEngineConfig(total_tokens=4096, output_reserve_tokens=768)
    return ContextEngine(config=config, store=store), config


def test_estimate_tokens_counts_content():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("") == 0


def test_compact_creates_summary_and_restore_recovers_original():
    """压缩后消息减少、含摘要；按 compaction_id 100% 还原原文。"""
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = _engine_with_store(tmp)
        messages = _long_history(limit=30)
        before_count = len(messages)
        # 原始消息先进 ledger
        for m in messages:
            engine.store.append_message(m)

        result = engine.compact(messages, force=True)
        assert result["status"] == "compacted"
        assert result["compaction_id"]
        assert len(result["messages"]) < before_count
        assert "summary_text" in result and result["summary_text"]
        assert result["validation"]["passed"] is True

        # 还原：所有原始消息完整恢复
        restored = engine.restore(result["compaction_id"])
        restored_messages = restored["messages"]
        assert len(restored_messages) == before_count
        original_contents = {m["id"]: m["content"] for m in messages}
        for m in restored_messages:
            if m["id"] in original_contents:
                assert m["content"] == original_contents[m["id"]], "原文必须逐条一致"


def test_compact_respects_constraints():
    """expected_constraints 进入压缩校验：满足则 compacted，缺失则 warning+missing。"""
    with tempfile.TemporaryDirectory() as tmp:
        engine, _ = _engine_with_store(tmp)
        messages = _long_history(limit=20)
        for m in messages:
            engine.store.append_message(m)

        # 约束出现在消息内容中 → compacted
        ok = engine.compact(
            messages, expected_constraints=["LangGraph"], force=True,
        )
        assert ok["status"] == "compacted"
        assert ok["validation"]["expected_constraints"] == ["LangGraph"]
        assert ok["validation"]["missing_constraints"] == []

        # 约束不在任何消息中 → warning + missing_constraints 非空
        missing = engine.compact(
            messages, expected_constraints=["完全不存在的话题"], force=True,
        )
        assert missing["status"] == "warning"
        assert missing["validation"]["missing_constraints"] == ["完全不存在的话题"]


def test_engine_with_event_store_persists_events():
    """ContextEventStore + 引擎：消息写入 SQLite，可查询事件。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = ContextEventStore(str(Path(tmp) / "events.jsonl"))
        config = ContextEngineConfig(total_tokens=4096)
        engine = ContextEngine(config=config, store=store)

        messages = _long_history(limit=12)
        for m in messages:
            store.append_message(m)

        assembled = engine.assemble(messages)
        assert "messages" in assembled and assembled["messages"]
        assert "meter" in assembled

        events = store.read_events()
        assert len(events) >= len(messages)
