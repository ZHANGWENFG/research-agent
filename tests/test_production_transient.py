"""P1 补测: research_production.py 瞬时错误判定（64% → 拉高）。

熔断器只对瞬时错误计数——业务错误（参数/鉴权）误计数会熔断一切，
这个分类函数是可靠性的第一道防线，纯逻辑可测。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_production import _is_transient_llm_error


def test_transient_standard_library():
    assert _is_transient_llm_error(ConnectionError("conn")) is True
    assert _is_transient_llm_error(TimeoutError("t")) is True


def test_transient_requests():
    import requests
    assert _is_transient_llm_error(requests.exceptions.ConnectionError("c")) is True
    assert _is_transient_llm_error(requests.exceptions.Timeout("t")) is True
    assert _is_transient_llm_error(requests.exceptions.RequestException("generic")) is True


def test_not_transient_business_errors():
    assert _is_transient_llm_error(ValueError("400 bad request")) is False
    assert _is_transient_llm_error(KeyError("missing")) is False
    assert _is_transient_llm_error(RuntimeError("boom")) is False
    assert _is_transient_llm_error(None) is False
