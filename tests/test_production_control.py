"""生产控制面测试：幂等 / 熔断 / 审计。

直接在 ProductionControlPlane 上测试，不依赖 LLM，全部离线。
"""
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

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


# ---------- P4: 指数退避 + full jitter（AWS 标准） ----------

def test_retry_uses_exponential_backoff_with_full_jitter(control, monkeypatch):
    """重试间隔必须是 full jitter: sleep ∈ [0, base * 2^(attempt-1)]，绝不立即连打。"""
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("research_agent.research_production.time.sleep", fake_sleep)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise TimeoutError("upstream timeout")

    fallback = lambda error: {"answer": "degraded"}  # noqa: E731

    control.execute_resilient(
        operation_name="backoff_test", operation=flaky, fallback=fallback,
        max_attempts=3, failure_threshold=99, cooldown_seconds=60,
        backoff_base_seconds=0.5,
    )
    # 3 次尝试 → 2 次失败后各睡一次
    assert len(sleeps) == 2, f"应 sleep 2 次（最后一次失败不睡）, got {sleeps}"
    # attempt=1 → cap=0.5*2^0=0.5; attempt=2 → cap=0.5*2^1=1.0
    assert 0.0 <= sleeps[0] < 0.5, f"第一次退避应在 [0, 0.5), got {sleeps[0]}"
    assert 0.0 <= sleeps[1] < 1.0, f"第二次退避应在 [0, 1.0), got {sleeps[1]}"
    assert calls["n"] == 3


def test_retry_backoff_zero_base_no_sleep(control, monkeypatch):
    """backoff_base_seconds=0 时不睡眠（测试/同步场景的逃生阀）。"""
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("research_agent.research_production.time.sleep", fake_sleep)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise TimeoutError("boom")

    fallback = lambda error: {"answer": "degraded"}  # noqa: E731

    control.execute_resilient(
        operation_name="backoff_zero", operation=flaky, fallback=fallback,
        max_attempts=2, failure_threshold=99, cooldown_seconds=60,
        backoff_base_seconds=0.0,
    )
    assert sleeps == [0.0], "base=0 时退避为 0（2 次尝试仅 1 次失败间隙），立即重试"


# ---------- P5: half-open 半开探测（AWS 三态） ----------

def test_circuit_half_open_probe_success_closes(control):
    """open 冷却期过后进入 half-open，探测成功 → closed，链路恢复。"""
    calls = {"n": 0}

    def fail_then_succeed():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("boom")
        return {"answer": "recovered"}

    fallback = lambda error: {"answer": "degraded"}  # noqa: E731

    # 打到 open（threshold=2, 每次 1 次尝试）
    for _ in range(2):
        control.execute_resilient(
            operation_name="half_open_ok", operation=fail_then_succeed, fallback=fallback,
            max_attempts=1, failure_threshold=2, cooldown_seconds=999,
        )
    # 冷却期改为 0（模拟 cooldown 已到点）→ 进入 half-open 探测
    outcome = control.execute_resilient(
        operation_name="half_open_ok", operation=fail_then_succeed, fallback=fallback,
        max_attempts=1, failure_threshold=2, cooldown_seconds=0,
    )
    assert outcome["result"] == {"answer": "recovered"}
    assert outcome["circuit_state"] == "closed"
    assert control._get_circuit("half_open_ok")["state"] == "closed"


def test_circuit_half_open_probe_failure_reopens(control):
    """探测失败 → 立即重新 open，后续请求在冷却期内继续被拒。"""
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise TimeoutError("boom")

    fallback = lambda error: {"answer": "degraded"}  # noqa: E731
    name = "half_open_fail"

    for _ in range(2):
        control.execute_resilient(
            operation_name=name, operation=always_fail, fallback=fallback,
            max_attempts=1, failure_threshold=2, cooldown_seconds=999,
        )
    # 把 opened_at 拨回过去，模拟冷却期已过 → 探测发生并失败
    control._set_circuit(name, "open", 2, time.time() - 100)
    outcome = control.execute_resilient(
        operation_name=name, operation=always_fail, fallback=fallback,
        max_attempts=1, failure_threshold=2, cooldown_seconds=999,
    )
    assert outcome["circuit_state"] == "open", "探测失败应立即重新 open"
    assert outcome["result"] == {"answer": "degraded"}
    # 重新 open 后（opened_at=now），冷却期内后续请求被拒，不再调下游
    before = calls["n"]
    outcome2 = control.execute_resilient(
        operation_name=name, operation=always_fail, fallback=fallback,
        max_attempts=1, failure_threshold=2, cooldown_seconds=999,
    )
    assert outcome2["circuit_state"] == "open"
    assert calls["n"] == before, "冷却期内不应再调下游"


def test_circuit_half_open_rejects_concurrent_probes(control):
    """half-open 已有一个探测在途时，其他请求直接拒绝（原子转换只放行一个）。"""
    name = "half_open_concurrent"
    fallback = lambda error: {"answer": "degraded"}  # noqa: E731
    # 直接构造 open 且冷却期已过（opened_at 100s 前 > cooldown 10s）
    control._set_circuit(name, "open", 3, time.time() - 100)
    # 第一个请求抢到探测权
    outcome = control.execute_resilient(
        operation_name=name, operation=lambda: {"ok": 1}, fallback=fallback,
        max_attempts=1, failure_threshold=3, cooldown_seconds=10,
    )
    assert outcome["circuit_state"] == "closed"
    # 人为再设回 half_open（模拟另一请求在探测途中），新请求必须被拒
    control._set_circuit(name, "half_open", 3, time.time() - 100)
    outcome2 = control.execute_resilient(
        operation_name=name, operation=lambda: {"ok": 2}, fallback=fallback,
        max_attempts=1, failure_threshold=3, cooldown_seconds=10,
    )
    assert outcome2["circuit_state"] == "half_open"
    assert outcome2["result"] == {"answer": "degraded"}


def test_half_open_probe_business_error_closes(control):
    """探测请求收到业务错误（非瞬时）→ 链路视为通，恢复 closed 后抛出。"""
    name = "half_open_biz"
    control._set_circuit(name, "open", 3, time.time() - 100)

    def bad_request():
        raise ValueError("4xx config error")

    with pytest.raises(ValueError):
        control.execute_resilient(
            operation_name=name, operation=bad_request,
            max_attempts=1, failure_threshold=3, cooldown_seconds=10,
        )
    assert control._get_circuit(name)["state"] == "closed", \
        "业务错误说明链路通，不应残留 half_open"


# ---------- P6: 幂等 24h 保留窗口（Stripe 同款） ----------

def test_idempotency_retention_purges_expired_records(control):
    """超过保留窗口（默认 24h）的记录被清理，同 key 可重新执行。"""
    scope, key, payload = "s", "retention-key", {"m": 1}
    control.execute_idempotent(
        scope=scope, key=key, payload=payload, operation=lambda: {"ok": 1}
    )
    # 把记录的 updated_at 拨回 48h 前（模拟过期）
    with control._connect() as connection:
        connection.execute(
            "UPDATE idempotency SET updated_at=? WHERE scope=? AND request_key=?",
            ((datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(), scope, key),
        )
    # 再次同 key 调用 → 旧记录被 retention 清理，重新认领执行（不是 replay）
    calls = {"n": 0}
    result = control.execute_idempotent(
        scope=scope, key=key, payload=payload,
        operation=lambda: (calls.__setitem__("n", calls["n"] + 1), {"ok": 2})[1],
    )
    assert result["idempotent_replay"] is False, "过期记录应被清理并重新执行"
    assert result["result"] == {"ok": 2}
    assert calls["n"] == 1


def test_idempotency_retention_keeps_fresh_records(control):
    """保留窗口内的记录不受影响，仍走 replay。"""
    control.execute_idempotent(
        scope="s", key="fresh", payload={"m": 1}, operation=lambda: {"ok": 1}
    )
    result = control.execute_idempotent(
        scope="s", key="fresh", payload={"m": 1}, operation=lambda: {"ok": 2}
    )
    assert result["idempotent_replay"] is True
    assert result["result"] == {"ok": 1}
