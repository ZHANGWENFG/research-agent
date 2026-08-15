"""生产控制面测试：幂等 / 熔断 / 审计。

直接在 ProductionControlPlane 上测试，不依赖 LLM，全部离线。
"""
import tempfile
from pathlib import Path

import pytest

from research_agent.research_production import ProductionControlPlane


@pytest.fixture()
def control():
    with tempfile.TemporaryDirectory() as tmp:
        yield ProductionControlPlane(str(Path(tmp) / "control.sqlite"))


def test_idempotent_replay_returns_cached_result(control):
    """同一个 request_id 调两次，第二次直接返回上次结果，操作只执行一次。"""
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        return {"answer": "result"}

    first = control.execute_idempotent(
        scope="local/thread-1", key="req-1", payload={"message": "hi"}, operation=operation
    )
    second = control.execute_idempotent(
        scope="local/thread-1", key="req-1", payload={"message": "hi"}, operation=operation
    )
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert second["result"] == {"answer": "result"}
    assert calls["n"] == 1, "同一请求号不应重复执行操作"


def test_idempotent_key_reused_with_different_payload_raises(control):
    """同一 request_id 换 payload 必须报错，防止结果串号。"""
    control.execute_idempotent(
        scope="s", key="k", payload={"a": 1}, operation=lambda: {"ok": True}
    )
    with pytest.raises(ValueError):
        control.execute_idempotent(
            scope="s", key="k", payload={"a": 2}, operation=lambda: {"ok": True}
        )


def test_idempotent_failed_operation_retries_next_call(control):
    """上一次失败不缓存失败结果，下次请求重新执行。"""
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise RuntimeError("boom")

    def succeeding():
        calls["n"] += 1
        return {"ok": True}

    with pytest.raises(RuntimeError):
        control.execute_idempotent(scope="s", key="k", payload={}, operation=failing)
    result = control.execute_idempotent(scope="s", key="k", payload={}, operation=succeeding)
    assert result["result"] == {"ok": True}
    assert calls["n"] == 2


def test_circuit_breaker_opens_and_short_circuits(control):
    """连续失败达到阈值后熔断打开；冷却期内不调下游，直接走兜底。"""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise TimeoutError("upstream timeout")

    def fallback(error):
        return {"answer": "degraded"}

    # 第一次调用：max_attempts=2 内部试 2 次，累积 2 次失败（未达阈值 3）
    outcome = control.execute_resilient(
        operation_name="chat_llm", operation=flaky, fallback=fallback,
        max_attempts=2, failure_threshold=3, cooldown_seconds=60,
    )
    assert outcome["degraded"] is True
    assert outcome["circuit_state"] == "closed"
    assert calls["n"] == 2

    # 第二次调用：再累积 2 次 → 4 >= 3，熔断打开
    outcome = control.execute_resilient(
        operation_name="chat_llm", operation=flaky, fallback=fallback,
        max_attempts=2, failure_threshold=3, cooldown_seconds=60,
    )
    assert outcome["circuit_state"] == "open"
    assert calls["n"] == 4

    # 冷却期内第三次调用：直接兜底，不再调下游
    before = calls["n"]
    outcome = control.execute_resilient(
        operation_name="chat_llm", operation=flaky, fallback=fallback,
        max_attempts=2, failure_threshold=3, cooldown_seconds=60,
    )
    assert outcome["result"] == {"answer": "degraded"}
    assert calls["n"] == before, "熔断期间不应调用下游"

    # 非网络类异常不计数也不兜底：直接抛出（只捕获 ConnectionError/TimeoutError）
    def hard_failure():
        raise ValueError("config bug")

    with pytest.raises(ValueError):
        control.execute_resilient(
            operation_name="other", operation=hard_failure, fallback=fallback,
            max_attempts=1, failure_threshold=3, cooldown_seconds=60,
        )


def test_audit_events_recorded(control):
    """每个授权动作都留审计痕迹（含单用户放行）。"""
    control.authorize(
        tenant_id="local", user_id="u1",
        resource_type="conversation_thread", resource_id="t1", action="invoke",
    )
    events = control.list_audit_events(limit=10)
    assert len(events) >= 1
    event = events[0]
    assert event["user_id"] == "u1"
    assert event["action"] == "invoke"
    assert event["decision"] == "allow"
    assert event["reason"].startswith("single-user mode")


def test_trace_spans_records_component_timing(control):
    """trace_span 记录节点耗时与状态，可按 trace_id 聚合。"""
    trace_id = "trace-abc"
    with control.trace_span(trace_id, "agent_runtime", "conversation_graph"):
        pass
    spans = control.list_spans(trace_id)
    assert len(spans) == 1
    assert spans[0]["trace_id"] == trace_id
    assert spans[0]["component"] == "agent_runtime"
    assert spans[0]["status"] == "success"
