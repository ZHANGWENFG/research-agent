"""缺口 1/2/4 挂接测试：开关语义 + 回退安全 + 零行为变化。

核心验收：
- 开关未设置 → 原流水线零变化（mock 验证调用序列一致）
- mode="agent" 无 LLM → 安全回退固定流水线
- RESEARCH_QUERY_REWRITE=1 → 证据不足时触发改写重搜
- RESEARCH_PARALLEL=1 → _run_research_loop 走并行分支
"""
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_qa import ResearchQAAgent
from research_agent.research_service import ResearchTaskService


# ---------- 工具 ----------

def _sufficient_evidence():
    return [
        {"id": "e1", "chunk_id": "e1", "title": "PIM 论文",
         "content": "passive intermodulation antenna analysis", "score": 0.9},
    ]


def _empty_answer():
    return {"answer": "", "evidence": [], "citations": [], "grounded": False,
            "memory_context": {}}


def _kb_answer(evidence):
    return {"answer": "基于证据的答案 [1]", "evidence": evidence,
            "citations": [{"id": "e1"}], "grounded": True, "memory_context": {}}


class _FakeTaskService:
    """最小 task_service mock：状态 + 知识库查询。"""

    def __init__(self, kb=lambda q, k: _kb_answer(_sufficient_evidence())):
        self._kb = kb
        self.tasks = {}
        self.queries = []

    def submit_research_task(self, topic, **kwargs):
        task_id = "a" * 32
        self.tasks[task_id] = {"task_id": task_id, "topic": topic, "status": "succeeded"}
        return self.tasks[task_id]

    def run_task(self, task_id):
        self.tasks[task_id]["status"] = "succeeded"
        return self.tasks[task_id]

    def get_task(self, task_id):
        return self.tasks.get(task_id) or {"task_id": task_id, "status": "succeeded"}

    def query_knowledge_base(self, task_id, question, top_k=3):
        self.queries.append(question)
        return self._kb(question, top_k)


def _no_llm(monkeypatch):
    """build_chat_llm_callable 返回 None（未配置）。"""
    monkeypatch.setattr(
        "research_agent.research_router_llm.build_chat_llm_callable",
        lambda enabled=None: None,
    )


def _fake_llm(monkeypatch, decisions):
    """build_chat_llm_callable 返回假 LLM：依次返回 decisions，耗尽后返回 answer JSON。"""
    queue = list(decisions)
    calls = {"n": 0}

    def chat_llm(prompt):
        if calls["n"] < len(queue):
            item = queue[calls["n"]]
            calls["n"] += 1
            return item
        return json.dumps({"tool": "answer", "args": {"question": "q"}})

    monkeypatch.setattr(
        "research_agent.research_router_llm.build_chat_llm_callable",
        lambda enabled=None: chat_llm,
    )
    return calls


# ---------- 缺口 1: agent 模式 ----------

def test_ask_default_mode_zero_change(monkeypatch):
    """mode 默认 auto 且 env 未设 → 不触发 agent/adaptive 分支。"""
    fake = _FakeTaskService()
    agent = ResearchQAAgent(fake)
    with mock.patch("research_agent.research_qa.os.getenv", return_value="") as env_mock:
        result = agent.ask("pim 天线是什么", mode="auto")
    # 原路径: 无 task_id → submit → run_task → query_knowledge_base
    assert result["decision"]["action"] in ("retrieve_then_answer",)
    assert len(fake.queries) == 1
    env_mock.assert_any_call("RESEARCH_AGENT_LOOP")
    env_mock.assert_any_call("RESEARCH_QUERY_REWRITE")


def test_ask_agent_mode_falls_back_without_llm(monkeypatch):
    """mode="agent" 但无 LLM 配置 → 安全回退固定流水线。"""
    _no_llm(monkeypatch)
    fake = _FakeTaskService()
    agent = ResearchQAAgent(fake)
    result = agent.ask("pim 天线", mode="agent")
    assert result["decision"]["action"] == "retrieve_then_answer"  # 固定流水线行为
    assert "agent_loop" not in result


def test_ask_agent_mode_runs_loop(monkeypatch):
    """mode="agent" 有 LLM → 走 AgentLoop，决策 action=agent_loop。"""
    calls = _fake_llm(monkeypatch, [
        json.dumps({"reasoning": "先检索", "tool": "search", "args": {"query": "pim"}}),
        json.dumps({"reasoning": "够了", "tool": "answer", "args": {"question": "pim 是啥"}}),
    ])
    fake = _FakeTaskService()
    agent = ResearchQAAgent(fake)
    result = agent.ask("pim 天线", mode="agent", top_k=3)
    assert result["decision"]["action"] == "agent_loop"
    assert result["agent_loop"]["termination"] == "agent_decided_answer"
    assert len(fake.queries) >= 1  # search 工具调用了知识库
    assert calls["n"] == 2


def test_ask_agent_mode_env_enabled(monkeypatch):
    """RESEARCH_AGENT_LOOP=1（env 开关）→ 与 mode="agent" 等效。"""
    calls = _fake_llm(monkeypatch, [
        json.dumps({"tool": "search", "args": {"query": "pim"}}),
    ])
    fake = _FakeTaskService()
    agent = ResearchQAAgent(fake)
    with mock.patch("research_agent.research_qa.os.getenv",
                    side_effect=lambda k: "1" if k == "RESEARCH_AGENT_LOOP" else ""):
        result = agent.ask("pim 天线", mode="auto")
    assert result["decision"]["action"] == "agent_loop"
    assert calls["n"] >= 1


# ---------- 缺口 2: 查询改写 ----------

def test_ask_rewrite_off_no_adaptive(monkeypatch):
    """RESEARCH_QUERY_REWRITE 未设 → 证据不足也只查一次，不触发改写。"""
    fake = _FakeTaskService(kb=lambda q, k: _empty_answer())
    agent = ResearchQAAgent(fake)
    with mock.patch("research_agent.research_qa.os.getenv", return_value=""):
        result = agent.ask("pim 天线", mode="auto")
    assert len(fake.queries) == 1
    assert result["decision"]["action"] in ("reject_low_confidence", "retrieve_then_answer")


def test_ask_rewrite_on_triggers_adaptive(monkeypatch):
    """RESEARCH_QUERY_REWRITE=1 + 证据不足 → adaptive 改写重搜并收敛。"""
    # 改写路径的 LLM 返回 rewritten_query JSON（与 agent 决策 JSON 不同）
    _fake_llm(monkeypatch, [
        '{"variant": "expansion", "rewritten_query": "pim antenna rf interference"}',
    ])
    state = {"n": 0}

    def kb(query, top_k):
        state["n"] += 1
        if query == "pim antenna 是什么":
            return _empty_answer()  # 第一轮不足
        return _kb_answer([
            {"id": "e1", "chunk_id": "e1", "title": "PIM 论文",
             "content": "pim antenna analysis rf interference", "score": 0.9},
        ])  # 改写后充足

    fake = _FakeTaskService(kb=kb)
    agent = ResearchQAAgent(fake)
    monkeypatch.setenv("RESEARCH_QUERY_REWRITE", "1")
    result = agent.ask("pim antenna 是什么", mode="auto")
    monkeypatch.delenv("RESEARCH_QUERY_REWRITE", raising=False)
    assert state["n"] >= 2  # 首轮 + 改写后
    assert result["decision"]["reason"] == "query rewrite improved retrieval"
    assert result["evidence"]  # 改写后拿到证据


def test_ask_rewrite_on_no_llm_degrades(monkeypatch):
    """改写开关开但无 LLM → 单轮评估不改写（安全退化），不崩。"""
    _no_llm(monkeypatch)
    fake = _FakeTaskService(kb=lambda q, k: _empty_answer())
    agent = ResearchQAAgent(fake)
    monkeypatch.setenv("RESEARCH_QUERY_REWRITE", "1")
    result = agent.ask("pim 天线", mode="auto")  # 不抛异常即可
    monkeypatch.delenv("RESEARCH_QUERY_REWRITE", raising=False)
    assert result["evidence_sufficiency"]["sufficient"] is False


# ---------- 缺口 4: 并行 ----------

def test_service_parallel_off_default(monkeypatch, tmp_path):
    """RESEARCH_PARALLEL 未设 → _run_research_loop 调串行 run_research_loop。"""
    monkeypatch.setattr(
        "research_agent.research_router_llm.build_chat_llm_callable",
        lambda enabled=None: None,
    )
    service = ResearchTaskService(str(tmp_path))
    with mock.patch.object(service, "_run_parallel_research") as parallel_mock, \
         mock.patch("research_agent.research_loop.run_research_loop",
                    return_value={"article": "串行文章", "perspectives": [],
                                  "scorecard": {}, "qc_passed": True}):
        service._run_research_loop(
            {"task_id": "b" * 32, "topic": "pim", "output_dir": str(tmp_path / "out"),
             "retriever": "arxiv", "options": {}}
        )
    parallel_mock.assert_not_called()


def test_service_parallel_on_uses_parallel_branch(monkeypatch, tmp_path):
    """RESEARCH_PARALLEL=1 → _run_research_loop 调用 _run_parallel_research。"""
    monkeypatch.setattr(
        "research_agent.research_router_llm.build_chat_llm_callable",
        lambda enabled=None: None,
    )
    monkeypatch.setenv("RESEARCH_PARALLEL", "1")
    try:
        service = ResearchTaskService(str(tmp_path))
        with mock.patch.object(service, "_run_parallel_research",
                               return_value={"article": "并行文章", "perspectives": [],
                                             "scorecard": {}, "qc_passed": True,
                                             "citation_pool": []}) as parallel_mock:
            service._run_research_loop(
                {"task_id": "c" * 32, "topic": "pim", "output_dir": str(tmp_path / "out"),
                 "retriever": "arxiv", "options": {}}
            )
        parallel_mock.assert_called_once()
    finally:
        monkeypatch.delenv("RESEARCH_PARALLEL", raising=False)


def test_run_parallel_research_writes_article(tmp_path):
    """并行调研: 视角生成 → 并行 → 聚合写文章文件。"""
    service = ResearchTaskService(str(tmp_path))
    fake_llm = lambda prompt: "段落 [1]"

    def fake_search(query, top_k):
        return [{"url": "http://x/1", "title": "T1", "description": "d", "snippets": []}]

    with mock.patch("research_agent.research_loop.generate_perspectives",
                    side_effect=lambda s, llm_call, search, fulltext, skill_context: (
                        s.update({"perspectives": ["机制", "安全"]}) or s
                    )), \
         mock.patch("research_agent.research_parallel.run_parallel_perspectives",
                    return_value={
                        "perspectives": ["机制", "安全"],
                        "sections": [
                            {"perspective": "机制", "paragraph": "机制段落"},
                            {"perspective": "安全", "paragraph": "安全段落"},
                        ],
                        "evidence": [{"document_id": "d1"}],
                        "evidence_by_view": {},
                        "errors": [],
                        "parallel": {"workers": 2, "perspective_count": 2},
                        "converged": True,
                    }):
        result = service._run_parallel_research(
            {"task_id": "d" * 32, "output_dir": str(tmp_path)},
            "pim 天线", fake_llm, fake_search, "", "arxiv",
        )
    article_file = tmp_path / "myagent_article_polished.txt"
    assert article_file.exists()
    content = article_file.read_text(encoding="utf-8")
    assert "## 机制" in content
    assert "## 安全" in content
    assert result["qc_passed"] is True
