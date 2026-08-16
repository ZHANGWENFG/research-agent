"""P1 补测: research_router_llm.py（32%）纯逻辑——缓存容量/toml 环境加载。

意图路由是"问题 → 研究策略"的第一道门，配置加载逻辑离线可测。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_router_llm import (
    _load_flat_toml_env,
    _router_cache_size,
)


def test_router_cache_size_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_ROUTER_CACHE_SIZE", "128")
    assert _router_cache_size() == 128
    monkeypatch.setenv("RESEARCH_ROUTER_CACHE_SIZE", "abc")
    assert _router_cache_size() == 512  # ValueError 回退默认
    monkeypatch.setenv("RESEARCH_ROUTER_CACHE_SIZE", "0")
    assert _router_cache_size() == 0  # 0 = 禁用缓存


def test_load_flat_toml_env_parses_and_setdefault(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text(
        'DEEPSEEK_API_KEY = "sk-abc"\nOPENAI_API_BASE = "https://x/v1"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    _load_flat_toml_env(str(secrets))
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-abc"
    assert os.environ["OPENAI_API_BASE"] == "https://x/v1"


def test_load_flat_toml_env_does_not_override_existing(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('DEEPSEEK_API_KEY = "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    _load_flat_toml_env(str(secrets))
    assert os.environ["DEEPSEEK_API_KEY"] == "from-env"  # setdefault: 已存在不覆盖


def test_load_flat_toml_env_missing_file_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("RESEARCH_SECRETS_PATH", raising=False)
    _load_flat_toml_env(str(tmp_path / "nope.toml"))  # 不抛错即可
