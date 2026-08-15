"""冒烟测试：服务可导入、健康检查、fake 调研任务全链路。

fake 模式不调 LLM / 不联网，可在无 API key 环境稳定运行。
"""
import time

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "my-agent"


def test_research_fake_task_full_cycle(client):
    """fake 调研任务：queued -> running -> succeeded，产生产物文件。"""
    r = client.post(
        "/api/research",
        json={"topic": "LangGraph 和 LangChain 的区别", "run_mode": "fake"},
    )
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    # 轮询任务直到终态
    status = "queued"
    for _ in range(50):
        r = client.get(f"/api/research/{task_id}")
        assert r.status_code == 200
        status = r.json()["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert status == "succeeded", f"任务未成功: {status}"

    # SSE 流可读且正常结束
    with client.stream("GET", f"/api/research/{task_id}/stream") as stream:
        chunks = [line for line in stream.iter_lines() if line]
    assert any("succeeded" in line for line in chunks)


def test_research_rejects_empty_topic(client):
    r = client.post("/api/research", json={"topic": "  ", "run_mode": "fake"})
    assert r.status_code == 400


def test_unknown_task_returns_404(client):
    r = client.get("/api/research/nonexistent-task-id")
    assert r.status_code == 404
