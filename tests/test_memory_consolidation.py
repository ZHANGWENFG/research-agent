"""缺口④：记忆自动沉淀单测。

锁定语义：同主题记录按术语重叠聚类（min_episodes 阈值）；LLM 总结优先、
无 LLM 规则回退；源记录打标防重复（二次调用不新增）；空记忆返回空。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_memory import MemoryRecord, ResearchMemoryStore  # noqa: E402
from research_agent.research_memory_consolidation import (  # noqa: E402
    CONSOLIDATED_FLAG,
    _cluster_records,
    consolidate_memories,
)

PIM_EPISODES = [
    "SciFact 基准上 PIM 检索准确率提升明显",
    "SciFact 数据集 PIM 表现优于基线",
    "PIM 在 SciFact 评测中准确率最高",
]
MMR_EPISODES = [
    "MMR 重排序增加结果多样性",
    "MMR 多样性重排在长尾查询上有效",
    "MMR 重排避免结果冗余",
]


def make_store(contents, kind="episodic"):
    records = [MemoryRecord(kind=kind, content=c, id="r{0}".format(i))
               for i, c in enumerate(contents)]
    return ResearchMemoryStore(episodic=records), records


class TestClusterRecords:
    def test_same_topic_merged(self):
        clusters = _cluster_records(
            [MemoryRecord(kind="episodic", content=c) for c in PIM_EPISODES]
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_topics_separated(self):
        all_records = [MemoryRecord(kind="episodic", content=c)
                       for c in PIM_EPISODES + MMR_EPISODES]
        clusters = _cluster_records(all_records)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [3, 3]

    def test_below_threshold_dropped(self):
        clusters = _cluster_records(
            [MemoryRecord(kind="episodic", content=c) for c in PIM_EPISODES[:2]],
            min_episodes=3,
        )
        assert clusters == []

    def test_threshold_parameter(self):
        clusters = _cluster_records(
            [MemoryRecord(kind="episodic", content=c) for c in PIM_EPISODES[:2]],
            min_episodes=2,
        )
        assert len(clusters) == 1


class TestConsolidateMemories:
    def test_rule_fallback_no_llm(self):
        store, _ = make_store(PIM_EPISODES)
        created = consolidate_memories(store)
        assert len(created) == 1
        item = created[0]
        assert "SciFact" in item["content"] or "PIM" in item["content"]
        assert item["tags"]
        assert len(item["source_episode_ids"]) == 3
        assert len(store.semantic) == 1
        assert store.semantic[0].metadata[CONSOLIDATED_FLAG]

    def test_llm_summary_mock(self):
        store, _ = make_store(PIM_EPISODES)

        def fake_llm(prompt):
            assert "semantic" in prompt and "记录" in prompt
            return "PIM 在 SciFact 基准上表现优异\n#pim,#scifact"

        created = consolidate_memories(store, llm_call=fake_llm)
        assert len(created) == 1
        assert created[0]["content"] == "PIM 在 SciFact 基准上表现优异"
        assert set(created[0]["tags"]) == {"pim", "scifact"}
        assert store.semantic[0].metadata["tags"] == ["pim", "scifact"]

    def test_no_duplicate_on_second_call(self):
        store, _ = make_store(PIM_EPISODES)
        consolidate_memories(store)
        second = consolidate_memories(store)
        assert second == []
        assert len(store.semantic) == 1

    def test_empty_store(self):
        created = consolidate_memories(ResearchMemoryStore())
        assert created == []
        assert len(ResearchMemoryStore().semantic) == 0

    def test_min_episodes_threshold(self):
        store, _ = make_store(PIM_EPISODES[:2])
        assert consolidate_memories(store, min_episodes=3) == []
        store2, _ = make_store(PIM_EPISODES[:2])
        created = consolidate_memories(store2, min_episodes=2)
        assert len(created) == 1

    def test_already_marked_records_skipped(self):
        records = [MemoryRecord(kind="episodic", content=c,
                                metadata={CONSOLIDATED_FLAG: "2026-01-01T00:00:00+00:00"})
                   for c in PIM_EPISODES]
        store = ResearchMemoryStore(episodic=records)
        created = consolidate_memories(store)
        assert created == []

    def test_source_episode_ids_recorded(self):
        store, records = make_store(PIM_EPISODES)
        consolidate_memories(store)
        meta = store.semantic[0].metadata
        assert set(meta["source_episode_ids"]) == {r.id for r in records}
        assert all(r.metadata.get(CONSOLIDATED_FLAG) for r in records)

    def test_working_and_episodic_both_considered(self):
        working = [MemoryRecord(kind="working", content=c) for c in PIM_EPISODES]
        store = ResearchMemoryStore(working=working)
        created = consolidate_memories(store)
        assert len(created) == 1


# ---------- 挂接④语义（research_service._consolidate_memories_after_run） ----------

from research_agent.research_service import ResearchTaskService  # noqa: E402


def _service_with(tmp_path):
    return ResearchTaskService(str(tmp_path))


def _episode(store, content):
    store.remember_episode(content)


def test_consolidate_hook_disabled_no_side_effect(tmp_path, monkeypatch):
    """MEMORY_CONSOLIDATE 未设 → 挂接零行为（不建文件、不写记录）。"""
    monkeypatch.delenv("MEMORY_CONSOLIDATE", raising=False)
    service = _service_with(tmp_path)
    state = {"task_id": "t1", "topic": "PIM"}
    result = {"qc_passed": True, "scorecard": {}}
    service._consolidate_memories_after_run(state, result, lambda p: "ok")
    assert not (tmp_path / "memory.json").exists()


def test_consolidate_hook_enabled_records_episodes(tmp_path, monkeypatch):
    """启用后：每次任务写入 2 条 episodic；未达阈值不沉淀但积累原料。"""
    monkeypatch.setenv("MEMORY_CONSOLIDATE", "1")
    service = _service_with(tmp_path)
    state = {"task_id": "t1", "topic": "PIM 检索优化"}
    result = {"qc_passed": True, "scorecard": {"coverage": 0.9}}
    service._consolidate_memories_after_run(state, result, lambda p: "ok")
    store = ResearchMemoryStore.load(str(tmp_path / "memory.json"))
    assert len(store.episodic) == 2
    assert store.semantic == []


def test_consolidate_hook_accumulates_and_saves(tmp_path, monkeypatch):
    """预置同主题记录 + 本次记录 → 聚类沉淀 1 条 semantic 并落盘。"""
    monkeypatch.setenv("MEMORY_CONSOLIDATE", "1")
    memory_path = tmp_path / "memory.json"
    pre = ResearchMemoryStore()
    for c in [
        "SciFact 基准上 PIM 检索准确率提升明显",
        "SciFact 数据集 PIM 表现优于基线",
        "PIM 在 SciFact 评测中准确率最高",
    ]:
        _episode(pre, c)
    pre.save(str(memory_path))

    service = _service_with(tmp_path)
    state = {"task_id": "t9", "topic": "PIM 检索优化"}
    result = {"qc_passed": True, "scorecard": {}}
    service._consolidate_memories_after_run(state, result, lambda p: "PIM 结论\n#pim")
    store = ResearchMemoryStore.load(str(memory_path))
    assert len(store.semantic) == 1
    assert store.semantic[0].metadata["source_episode_ids"]


def test_consolidate_hook_failure_silent(tmp_path, monkeypatch):
    """沉淀异常 → warning 记录，不向外抛出（LLM 调用抛错被吞）。"""
    monkeypatch.setenv("MEMORY_CONSOLIDATE", "1")
    memory_path = tmp_path / "memory.json"
    pre = ResearchMemoryStore()
    for c in [
        "SciFact 基准上 PIM 检索准确率提升明显",
        "SciFact 数据集 PIM 表现优于基线",
        "PIM 在 SciFact 评测中准确率最高",
    ]:
        _episode(pre, c)
    pre.save(str(memory_path))

    service = _service_with(tmp_path)
    state = {"task_id": "t1", "topic": "PIM 检索优化"}
    result = {"qc_passed": True, "scorecard": {}}

    def boom(prompt):
        raise RuntimeError("llm down")

    # llm_call 抛异常路径：异常被吞，memory.json 不更新（save 未执行）
    service._consolidate_memories_after_run(state, result, boom)
    store = ResearchMemoryStore.load(str(memory_path))
    assert store.semantic == []
