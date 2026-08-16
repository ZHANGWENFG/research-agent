"""B3: research_benchmarks.py（38%）BenchmarkRegistry 纯逻辑覆盖。

注册表决定"哪个基准可以跑、缺什么输入、命令怎么拼"——生产 CI 与
手工跑基准都走这里，用临时根目录离线验证 ready/blocked/命令组装。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_benchmarks import (
    BenchmarkRegistry,
    _discover_inputs,
    _input_exists,
    _latest_result_path,
    _resolve_benchmark_root,
)


def _make_scifact(root: Path):
    scifact = root / "datasets" / "scifact"
    (scifact / "qrels").mkdir(parents=True, exist_ok=True)
    (scifact / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (scifact / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (scifact / "qrels" / "test.tsv").write_text("qid\tdocid\trel\n", encoding="utf-8")
    return scifact


def test_input_exists_scifact_requires_three_files(tmp_path):
    scifact = _make_scifact(tmp_path)
    assert _input_exists("scifact_dir", scifact) is True
    (scifact / "queries.jsonl").unlink()
    assert _input_exists("scifact_dir", scifact) is False  # 缺一个文件即不可用


def test_discover_inputs_only_existing(tmp_path):
    _make_scifact(tmp_path)
    inputs = _discover_inputs(tmp_path)
    assert "scifact_dir" in inputs
    # qasper_json 指向 tmp 根下的 qasper 数据——不存在则不被发现
    assert "qasper_json" not in inputs
    assert "longbench_json" not in inputs


def test_resolve_benchmark_root_explicit(tmp_path):
    assert _resolve_benchmark_root(str(tmp_path)) == tmp_path.resolve()


def test_resolve_benchmark_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_BENCHMARK_ROOT", str(tmp_path))
    assert _resolve_benchmark_root(None) == tmp_path.resolve()


def test_catalog_scifact_ready(tmp_path):
    _make_scifact(tmp_path)
    catalog = BenchmarkRegistry(str(tmp_path)).catalog()
    scifact = next(item for item in catalog["benchmarks"] if item["id"] == "scifact-retrieval-v55")
    assert scifact["ready"] is True
    assert scifact["status"] == "ready"
    assert scifact["inputs"][0]["available"] is True
    assert scifact["profiles"] == ["smoke", "quality"]


def test_catalog_empty_root_blocked(tmp_path):
    """空根目录: 所有基准 blocked 并说明缺什么。"""
    catalog = BenchmarkRegistry(str(tmp_path)).catalog()
    for item in catalog["benchmarks"]:
        assert item["ready"] is False
        assert item["status"] == "blocked"
    scifact = next(item for item in catalog["benchmarks"] if item["id"] == "scifact-retrieval-v55")
    assert "缺少本地输入" in scifact["blocker"]


def test_definition_unknown_id_raises(tmp_path):
    registry = BenchmarkRegistry(str(tmp_path))
    with pytest.raises(KeyError, match="unknown benchmark"):
        registry.definition("no-such-benchmark")


def test_build_command_scifact_smoke(tmp_path):
    """scifact smoke 命令: hash embedding + smoke-limit 20 + bm25/hybrid。"""
    _make_scifact(tmp_path)
    registry = BenchmarkRegistry(str(tmp_path))
    command = registry.build_command("scifact-retrieval-v55", tmp_path / "out", profile="smoke")
    joined = " ".join(command)
    assert "--embedding hash" in joined
    assert "--smoke-limit 20" in joined
    assert "--modes bm25 hybrid" in joined
    assert "--reranker" not in joined  # smoke 不加重排


def test_build_command_scifact_quality_uses_reranker(tmp_path):
    _make_scifact(tmp_path)
    registry = BenchmarkRegistry(str(tmp_path))
    command = registry.build_command("scifact-retrieval-v55", tmp_path / "out", profile="quality")
    joined = " ".join(command)
    assert "--embedding real" in joined
    assert "--reranker" in joined
    assert "hybrid_rerank" in joined


def test_build_command_bad_profile(tmp_path):
    _make_scifact(tmp_path)
    registry = BenchmarkRegistry(str(tmp_path))
    with pytest.raises(ValueError, match="profile"):
        registry.build_command("scifact-retrieval-v55", tmp_path / "out", profile="turbo")


def test_build_command_missing_input_raises(tmp_path):
    """输入缺失时 build_command 拒绝（而非拼出会失败的命令）。"""
    registry = BenchmarkRegistry(str(tmp_path))
    with pytest.raises(ValueError, match="缺少本地输入"):
        registry.build_command("scifact-retrieval-v55", tmp_path / "out")


def test_build_command_llm_benchmark_requires_key(monkeypatch, tmp_path):
    """付费 LLM 基准: 未显式确认/无 key 都拒绝。"""
    _make_scifact(tmp_path)
    registry = BenchmarkRegistry(str(tmp_path))
    # 注入 qasper_rankings 输入，让命令拼装能走到 LLM 校验分支
    (tmp_path / "fake_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    registry.inputs["qasper_rankings"] = tmp_path / "fake_predictions.jsonl"
    with pytest.raises(ValueError, match="显式确认"):
        registry.build_command("qasper-answer-v55", tmp_path / "out", allow_paid_llm=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        registry.build_command("qasper-answer-v55", tmp_path / "out", allow_paid_llm=True)


def test_latest_result_path_returns_none_when_absent():
    assert _latest_result_path("scifact-retrieval-v55", Path("nope")) is None
