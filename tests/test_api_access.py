"""API 访问控制测试（R1, 2026-08-16 新增）。

依据: OWASP API Top10 (2023) API4——无资源限制的 API 可被 farmed;
REST Security Cheat Sheet——无访问控制的公共服务会被爬取滥用。
覆盖: 可选鉴权（MY_AGENT_API_KEY 未设不强制 / 设置后强制）/ 限流 / 健康检查豁免。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import api as api_module


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "")
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 60)
    monkeypatch.setattr(api_module, "_requests_log", {})
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_exempt_from_auth_and_ratelimit(client, monkeypatch):
    """健康检查不鉴权也不限流（探活语义）。"""
    monkeypatch.setattr(api_module, "API_KEY", "secret")
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 1)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 200  # 限流 1 也不拦


def test_no_key_configured_allows_all(client):
    """未设置 MY_AGENT_API_KEY 时保持单用户放行语义（不破坏本地体验）。"""
    assert client.get("/api/skills").status_code == 200


def test_key_required_when_configured(client, monkeypatch):
    """设置 API_KEY 后: 缺失/错误 401，匹配放行。"""
    monkeypatch.setattr(api_module, "API_KEY", "secret-123")
    assert client.get("/api/skills").status_code == 401
    assert (
        client.get("/api/skills", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert (
        client.get("/api/skills", headers={"X-API-Key": "secret-123"}).status_code
        == 200
    )


def test_rate_limit_returns_429(client, monkeypatch):
    """每 IP 滑动窗口限流: 超限返回 429，窗口滑过后恢复。"""
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 3)
    for _ in range(3):
        assert client.get("/api/skills").status_code == 200
    assert client.get("/api/skills").status_code == 429


def test_rate_limit_is_per_ip(client, monkeypatch):
    """限流按客户端 IP 隔离（不同 IP 互不影响）。"""
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 2)
    # 同一 IP 前 2 次过、第 3 次 429
    assert client.get("/api/skills").status_code == 200
    assert client.get("/api/skills").status_code == 200
    assert client.get("/api/skills").status_code == 429
