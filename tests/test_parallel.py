"""缺口 4: 多代理并行测试（research_parallel.py）。

ThreadPoolExecutor 并行视角：验证真并发（mock 慢任务计时）、失败隔离、
聚合顺序（按视角原序）、证据去重、workers 上限。
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_parallel import run_parallel_perspectives


def _evidence(doc_id, content):
    return {"document_id": doc_id, "title": "T" + doc_id, "content": content, "score": 0.8}


def _llm_echo():
    """记录调用线程与返回内容的 LLM mock。"""
    state = {"calls": [], "threads": set()}

    def llm(prompt):
        state["threads"].add(threading.get_ident())
        state["calls"].append(prompt)
        return "合成段落 [1][2]"

    return llm, state


def test_parallel_runs_each_perspective_once():
    llm, state = _llm_echo()
    result = run_parallel_perspectives(
        "pim 天线", ["机制", "临床", "安全"], llm_call=llm,
        search=lambda q, k: [_evidence("d1", "content")],
    )
    assert len(result["sections"]) == 3
    assert len(llm.calls if hasattr(llm, "calls") else state["calls"]) == 3
    assert result["converged"] is True
    assert result["errors"] == []


def test_parallel_true_concurrency_with_slow_tasks():
    """3 个各 0.4s 的慢视角，并行应显著快于串行 1.2s。"""
    def slow_search(query, top_k):
        time.sleep(0.4)
        return [_evidence("d" + query[-1], "content")]

    llm, _ = _llm_echo()
    started = time.monotonic()
    result = run_parallel_perspectives(
        "topic", ["v1", "v2", "v3"], llm_call=llm, search=slow_search, max_workers=3
    )
    elapsed = time.monotonic() - started
    assert result["converged"] is True
    assert elapsed < 1.1  # 并行: ~0.4s 而非串行 ~1.2s（给调度留余量）


def test_parallel_section_order_preserved():
    """聚合段落按视角原顺序拼装（乱序返回也不影响）。"""
    def slow_search(query, top_k):
        # 视角 2 比视角 1 慢 → 若不保序会出现乱序
        time.sleep(0.3 if "v2" in query else 0.0)
        return [_evidence("d1", "content")]

    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", ["v1", "v2", "v3"], llm_call=llm, search=slow_search, max_workers=3
    )
    order = [s["perspective"] for s in result["sections"]]
    assert order == ["v1", "v2", "v3"]


def test_parallel_failure_isolation():
    """单个视角子任务失败 → 记 error 继续，其他视角正常。"""
    def flaky_search(query, top_k):
        if "坏视角" in query:
            raise RuntimeError("index corrupted")
        return [_evidence("d1", "content")]

    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", ["好视角", "坏视角", "另一个好视角"],
        llm_call=llm, search=flaky_search, max_workers=3,
    )
    assert len(result["errors"]) == 1
    assert "坏视角" in result["errors"][0]["perspective"]
    assert result["converged"] is False
    # 失败视角占位，好视角照常出段落
    ok_sections = [s for s in result["sections"] if s["paragraph"]]
    assert len(ok_sections) == 2
    failed_section = next(s for s in result["sections"] if not s["paragraph"])
    assert "失败" in failed_section["error"]


def test_parallel_evidence_merged_and_dedup():
    """多视角共享同一证据 → 聚合去重（document_id 唯一）。"""
    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", ["v1", "v2"],
        llm_call=llm,
        search=lambda q, k: [_evidence("shared", "content"), _evidence("unique", "x")],
        max_workers=2,
    )
    ids = [e["document_id"] for e in result["evidence"]]
    assert len(ids) == len(set(ids))
    assert "shared" in ids and "unique" in ids


def test_parallel_empty_perspectives_defaults():
    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", [], llm_call=llm, search=lambda q, k: [_evidence("d1", "x")]
    )
    assert result["perspectives"] == ["通用视角"]
    assert len(result["sections"]) == 1


def test_parallel_workers_capped_by_perspectives():
    """workers 不会超过视角数。"""
    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", ["only"], llm_call=llm,
        search=lambda q, k: [_evidence("d1", "x")],
        max_workers=10,
    )
    assert result["parallel"]["workers"] == 1


def test_parallel_synthesis_failure_isolated():
    """LLM 合成抛异常 → 该视角失败隔离，不拖垮整体。"""
    def flaky_llm(prompt):
        if "坏" in prompt:
            raise RuntimeError("llm timeout")
        return "正常段落"

    result = run_parallel_perspectives(
        "topic", ["好视角", "坏视角"],
        llm_call=flaky_llm,
        search=lambda q, k: [_evidence("d1", "x")],
        max_workers=2,
    )
    assert len(result["errors"]) == 1
    assert result["converged"] is False


def test_parallel_evidence_by_view():
    """evidence_by_view 按视角分组保留。"""
    def search(query, top_k):
        return [_evidence("d" + query[-1], "content")]

    llm, _ = _llm_echo()
    result = run_parallel_perspectives(
        "topic", ["v1", "v2"], llm_call=llm, search=search, max_workers=2
    )
    assert set(result["evidence_by_view"].keys()) == {"v1", "v2"}
    assert result["evidence_by_view"]["v1"][0]["document_id"] == "d1"
    assert result["evidence_by_view"]["v2"][0]["document_id"] == "d2"
