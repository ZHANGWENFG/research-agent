"""B1 追加: research_service.py（38%）任务状态机测试。

任务生命周期 queued→running→succeeded/failed、并发上限、stale 恢复、
四种 run_mode（fake/manual/fail/research）、错误红action。全部离线，用临时目录。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_service import ResearchTaskService, _parse_timestamp, _redact_error


@pytest.fixture
def service(tmp_path):
    return ResearchTaskService(str(tmp_path), max_concurrent_tasks=1)


def test_submit_creates_queued_task(service, tmp_path):
    task = service.submit_research_task("pim 天线", run_mode="fake")
    assert task["status"] == "queued"
    assert task["run_mode"] == "fake"
    state_file = tmp_path / "tasks" / "{0}.json".format(task["task_id"])
    assert state_file.exists()
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["topic"] == "pim 天线"
    assert persisted["expected_keywords"] == []


def test_run_fake_mode_succeeds(service):
    task = service.submit_research_task("主题", run_mode="fake")
    result = service.run_task(task["task_id"])
    assert result["status"] == "succeeded"
    assert result["finished_at"]


def test_run_fail_mode_records_error(service):
    """fail 模式: 任务失败但 run_task 不抛——错误进状态。"""
    task = service.submit_research_task("主题", run_mode="fail")
    result = service.run_task(task["task_id"])
    assert result["status"] == "failed"
    assert "simulated" in result["error"]


def test_run_manual_mode_returns_immediately(service):
    """manual 模式: 不改状态直接返回（人工接管）。"""
    task = service.submit_research_task("主题", run_mode="manual")
    result = service.run_task(task["task_id"])
    assert result["status"] == "running"  # _run_task_locked 先置 running 再短路
    assert "manual" in str(result.get("run_mode"))


def test_run_invalid_mode_fails_without_throwing(service):
    task = service.submit_research_task("主题", run_mode="bogus")
    result = service.run_task(task["task_id"])
    assert result["status"] == "failed"
    assert "run modes" in result["error"]


def test_complete_task_success_and_failure(service):
    task = service.submit_research_task("主题", run_mode="fake")
    ok = service.complete_task(task["task_id"], success=True)
    assert ok["status"] == "succeeded"
    bad = service.complete_task(task["task_id"], success=False, error="e2e 失败")
    assert bad["status"] == "failed"
    assert "e2e 失败" in bad["error"]


def test_worker_tick_respects_concurrency_cap(service):
    """max_concurrent_tasks=1: 2 个 queued 只启动 1 个。"""
    t1 = service.submit_research_task("a")
    t2 = service.submit_research_task("b")
    tick = service.worker_tick()
    assert tick["started_count"] == 1
    assert tick["queued_count"] == 1
    assert len(tick["started_task_ids"]) == 1


def test_worker_tick_does_not_exceed_capacity_when_running(service):
    t1 = service.submit_research_task("a")
    service.worker_tick()  # 启动 1 个（占满容量）
    t2 = service.submit_research_task("b")
    tick = service.worker_tick()
    assert tick["started_count"] == 0  # 无空闲容量
    assert service.get_task(t2["task_id"])["status"] == "queued"


def test_recover_stale_running_tasks(service, tmp_path):
    """超时 running 任务 → 恢复为 failed；未超时不受影响。"""
    fresh = service.submit_research_task("fresh")
    service.run_task(fresh["task_id"])  # 转 succeeded，不干扰
    # 手工伪造一个超时的 running 任务（task_id 须为 32 位 hex——白名单校验）
    stale_id = "a" * 32
    state = {
        "task_id": stale_id, "status": "running",
        "started_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    (tmp_path / "tasks" / "{0}.json".format(stale_id)).write_text(
        json.dumps(state), encoding="utf-8"
    )
    recovered = service.recover_stale_running_tasks(max_age_seconds=60)
    assert recovered["failed_task_ids"] == [stale_id]
    persisted = service.get_task(stale_id)
    assert persisted["status"] == "failed"
    assert "stale" in persisted["error"]


def test_recover_skips_fresh_running_task(service, tmp_path):
    """started_at 距今 < 阈值 → 不动。"""
    now = time.time()
    fresh_id = "b" * 32
    state = {
        "task_id": fresh_id, "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime(now)),
        "updated_at": "",
    }
    (tmp_path / "tasks" / "{0}.json".format(fresh_id)).write_text(
        json.dumps(state), encoding="utf-8"
    )
    recovered = service.recover_stale_running_tasks(max_age_seconds=3600)
    assert recovered["failed_task_ids"] == []
    assert service.get_task(fresh_id)["status"] == "running"


def test_list_tasks_filter_by_status(service):
    t1 = service.submit_research_task("a")
    t2 = service.submit_research_task("b")
    service.run_task(t1["task_id"])
    succeeded = service.list_tasks(status="succeeded")
    queued = service.list_tasks(status="queued")
    assert [t["task_id"] for t in succeeded] == [t1["task_id"]]
    assert [t["task_id"] for t in queued] == [t2["task_id"]]


def test_redact_error_removes_sensitive_markers():
    err = "RuntimeError: API key sk-123456 failed at https://api.x.com/v1"
    cleaned = _redact_error(err)
    assert "sk-123456" not in cleaned
    assert "RuntimeError" in cleaned


def test_parse_timestamp_handles_iso():
    assert _parse_timestamp("2020-01-01T00:00:00Z") == 1577836800.0
    assert _parse_timestamp("garbage") is None
    assert _parse_timestamp(None) is None
