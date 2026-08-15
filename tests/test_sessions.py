"""会话 API 测试：建会话 / 发消息 / 压缩 / 还原 / 重新生成。

fake 模式全程离线：LLM 降级为规则回复，但仍覆盖完整 API 链路。
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_session_create_and_list(client):
    r = client.post("/api/sessions", json={"title": "测试会话", "run_mode": "fake"})
    assert r.status_code == 200
    chat_id = r.json()["chat_id"]
    assert chat_id

    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = [s["chat_id"] for s in r.json()["sessions"]]
    assert chat_id in ids


def test_session_send_message_returns_answer(client):
    r = client.post("/api/sessions", json={"title": "会话B", "run_mode": "fake"})
    chat_id = r.json()["chat_id"]
    r = client.post(
        f"/api/sessions/{chat_id}/message",
        json={"message": "你好，介绍一下你自己"},
    )
    assert r.status_code == 200
    body = r.json()
    # fake 模式返回 assistant_message（LLM 降级为规则回复）
    assert "assistant_message" in body
    assert "content" in body["assistant_message"]


def test_session_compact_and_restore(client):
    """压缩后产生 compaction_id；还原后原文完整。"""
    r = client.post("/api/sessions", json={"title": "会话C", "run_mode": "fake"})
    chat_id = r.json()["chat_id"]
    client.post(f"/api/sessions/{chat_id}/message", json={"message": "第一条消息"})
    client.post(f"/api/sessions/{chat_id}/message", json={"message": "第二条消息"})

    r = client.post(f"/api/sessions/{chat_id}/compact")
    assert r.status_code == 200
    compaction_id = r.json().get("compaction_id")
    assert compaction_id, "压缩应产生 compaction_id: " + r.text[:200]

    r = client.post(f"/api/sessions/{chat_id}/restore", json={"compaction_id": compaction_id})
    assert r.status_code == 200
    assert r.json().get("raw_messages_unchanged") is True


def test_session_context_query(client):
    r = client.post("/api/sessions", json={"title": "会话D", "run_mode": "fake"})
    chat_id = r.json()["chat_id"]
    client.post(f"/api/sessions/{chat_id}/message", json={"message": "测试上下文"})
    r = client.get(f"/api/sessions/{chat_id}/context")
    assert r.status_code == 200
    meter = r.json().get("context_meter", {})
    assert meter.get("message_count", 0) >= 1


def test_session_regenerate(client):
    r = client.post("/api/sessions", json={"title": "会话E", "run_mode": "fake"})
    chat_id = r.json()["chat_id"]
    client.post(f"/api/sessions/{chat_id}/message", json={"message": "请重新生成这句"})
    r = client.post(f"/api/sessions/{chat_id}/regenerate")
    assert r.status_code == 200
    assert r.json().get("regenerated") is True or "answer" in r.json()


def test_unknown_session_404(client):
    r = client.post("/api/sessions/not-exist/message", json={"message": "hi"})
    assert r.status_code == 404
