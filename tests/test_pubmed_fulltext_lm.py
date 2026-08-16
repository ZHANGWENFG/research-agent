"""B1 追加: pubmed(28%)/fulltext(20%)/lm(35%) 三个低覆盖模块离线测试。

pubmed: NCBI 响应解析 + 节流 + 参数组装（mock requests，不发网络）
fulltext: 域名白名单 + 下载三重校验 + 文件头检查（纯逻辑 + mock 流式下载）
lm: 瞬时错误分类 + 指数退避重试（mock 函数计数，不等真实 sleep）
"""
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent import research_fulltext as ft
from research_agent import research_pubmed as pm
from research_agent import research_retrieval_runtime as runtime
from research_agent.lm import _call_with_retry, _is_transient


# ================= pubmed =================

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP {0}".format(self.status_code))

    def json(self):
        return self._payload

    @property
    def text(self):
        return self._payload if isinstance(self._payload, str) else ""


def test_pubmed_get_injects_tool_email():
    """_get 自动注入 tool/email，有 api_key 时注入 api_key。"""
    rm = pm.PubMedRM(email="me@x.com", api_key="KEY")
    with mock.patch.object(rm, "_throttle"), mock.patch.object(
        rm, "_last_request_at", 0.0
    ), mock.patch("research_agent.research_pubmed.requests.get",
                  return_value=_FakeResponse({"ok": True})) as mocked_get:
        data = rm._get("http://x", {"term": "q"})
    assert data == {"ok": True}
    _, kwargs = mocked_get.call_args
    assert kwargs["params"]["tool"] == "research-agent"
    assert kwargs["params"]["email"] == "me@x.com"
    assert kwargs["params"]["api_key"] == "KEY"


def test_pubmed_throttle_sleeps_between_requests():
    """两次请求间隔小于 throttle → 补睡；大于 → 不睡。"""
    rm = pm.PubMedRM(throttle=5.0)
    with mock.patch("research_agent.research_pubmed.time.sleep") as sleep_mock:
        rm._last_request_at = time.monotonic()  # 刚请求过
        rm._throttle()
        assert sleep_mock.called
        # 假装已经过了 6 秒
        rm._last_request_at = time.monotonic() - 6.0
        rm._throttle()
        assert sleep_mock.call_count == 1  # 第二次没再睡


def test_pubmed_esearch_parses_idlist():
    """esearch JSON 解析: idlist 提取。"""
    rm = pm.PubMedRM()
    with mock.patch.object(rm, "_get", return_value={
        "esearchresult": {"idlist": ["1", "2"]}
    }):
        assert rm._esearch("pim antenna", retmax=2) == ["1", "2"]


def test_pubmed_esummary_empty_pmids():
    """空 PMID 列表 → 空 dict，不请求。"""
    rm = pm.PubMedRM()
    with mock.patch.object(rm, "_get") as get_mock:
        assert rm._esummary([]) == {}
        get_mock.assert_not_called()


def test_pubmed_efetch_parses_text_blocks():
    """NCBI efetch 纯文本按 PMID 切块: '1. 12345' 起始行是分隔符。"""
    rm = pm.PubMedRM()
    text = (
        "1. 34567890\n"
        "Title of paper one\n"
        "Abstract line one.\n"
        "Abstract line two.\n"
        "\n"
        "2. 99887766\n"
        "Title of paper two\n"
        "Abstract line three.\n"
    )
    with mock.patch.object(rm, "_throttle"), mock.patch.object(
        rm, "_last_request_at", 0.0
    ), mock.patch("research_agent.research_pubmed.requests.get",
                  return_value=_FakeResponse(text)):
        blocks = rm._efetch_abstracts(["34567890", "99887766"])
    assert "34567890" in blocks
    assert "Abstract line one." in blocks["34567890"]
    assert "99887766" in blocks
    assert "Abstract line three." in blocks["99887766"]
    assert "1. 34567890" not in blocks["34567890"]  # 分隔符本身不进正文


def test_pubmed_efetch_empty_pmids():
    rm = pm.PubMedRM()
    with mock.patch("research_agent.research_pubmed.requests.get") as get_mock:
        assert rm._efetch_abstracts([]) == {}
        get_mock.assert_not_called()


# ================= fulltext =================

def test_domain_of():
    assert ft._domain_of("https://pmc.ncbi.nlm.nih.gov/articles/x") == "pmc.ncbi.nlm.nih.gov"
    assert ft._domain_of("http://arxiv.org/abs/2401.1") == "arxiv.org"
    assert ft._domain_of("") == ""
    assert ft._domain_of("not a url") == ""


def test_is_whitelisted_domain_and_subdomain():
    assert ft.is_whitelisted("https://arxiv.org/abs/2401.1") is True
    assert ft.is_whitelisted("https://pmc.ncbi.nlm.nih.gov/a") is True
    assert ft.is_whitelisted("https://europepmc.org/a") is True
    assert ft.is_whitelisted("https://evil.com/arxiv.org-phish") is False
    assert ft.is_whitelisted("https://sub.europepmc.org/x") is True  # 子域
    assert ft.is_whitelisted("") is False


def test_validate_response_content_type():
    """内容类型不在白名单 → 拒绝。"""
    resp = mock.Mock()
    resp.headers = {"content-type": "text/html"}
    assert "content-type not allowed" in ft._validate_response(resp)


def test_validate_response_size():
    """超过 20MB → 拒绝。"""
    resp = mock.Mock()
    resp.headers.get = lambda key, default=None: (
        "application/pdf" if key == "content-type" else str(21 * 1024 * 1024)
    )
    assert "file too large" in ft._validate_response(resp)


def test_validate_response_ok():
    resp = mock.Mock()
    resp.headers.get = lambda key, default=None: (
        "application/pdf" if key == "content-type" else "1024"
    )
    assert ft._validate_response(resp) is None


def test_download_file_rejects_bad_header_and_cleans_up(tmp_path):
    """流式下载: PDF 头校验失败 → 返回错误并删除残留文件。"""
    def fake_iter():
        yield b"NOTPDF content"  # 头 4 字节非 %PDF
        yield b"more bytes"

    resp = mock.Mock()
    resp.headers.get = lambda key, default=None: (
        "application/pdf" if key == "content-type" else str(len(b"x" * 4))
    )
    resp.iter_content = lambda chunk_size=8192: fake_iter()
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=resp):
        result = ft.download_file("https://arxiv.org/pdf/2401.1.pdf", str(tmp_path))
    assert result["ok"] is False
    assert "file header" in result["error"]
    assert not list(tmp_path.glob("*.pdf"))  # 残留已清理


def test_download_file_size_limit_during_stream(tmp_path):
    """流式下载中途超 max_size → 中止并删除。"""
    def fake_iter():
        yield b"a" * 1024
        yield b"b" * 1024

    resp = mock.Mock()
    resp.headers.get = lambda key, default=None: (
        "application/pdf" if key == "content-type" else "2048"
    )
    resp.iter_content = lambda chunk_size=8192: fake_iter()
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=resp):
        result = ft.download_file("https://arxiv.org/pdf/x.pdf", str(tmp_path), max_size=1500)
    assert result["ok"] is False
    assert "size exceeded" in result["error"]


def test_download_file_ok_pdf(tmp_path):
    """合法 %PDF 文件 → 下载成功、路径落盘、preview 字段存在。"""
    resp = mock.Mock()
    resp.headers.get = lambda key, default=None: (
        "application/pdf" if key == "content-type" else str(len(b"%PDF fake"))
    )
    resp.iter_content = lambda chunk_size=8192: iter([b"%PDF fake"])
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=resp):
        result = ft.download_file("https://arxiv.org/pdf/paper.pdf", str(tmp_path))
    assert result["ok"] is True
    assert Path(result["path"]).exists()
    assert "preview" in result


# ================= lm.py =================

def test_is_transient_classifies():
    """瞬时错误（网络/限流）→ True；业务错误 → False。"""
    import requests
    assert _is_transient(requests.exceptions.Timeout("t")) is True
    assert _is_transient(requests.exceptions.ConnectionError("c")) is True
    assert _is_transient(ValueError("bad request")) is False
    assert _is_transient(RuntimeError("boom")) is False


def test_is_transient_unwraps_cause_chain():
    """litellm 包装错误: cause 链里有瞬时错误 → 判瞬时。"""
    import requests
    inner = requests.exceptions.Timeout("timeout")
    outer = RuntimeError("litellm wrapper")
    outer.__cause__ = inner
    assert _is_transient(outer) is True


def test_call_with_retry_retries_then_succeeds():
    """前两次瞬时失败、第三次成功 → 重试 2 次后返回。"""
    import requests
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.Timeout("t")
        return "ok"

    with mock.patch("research_agent.lm.time.sleep"):
        result = _call_with_retry(flaky, max_retries=5)
    assert result == "ok"
    assert calls["n"] == 3


def test_call_with_retry_gives_up_after_max():
    """一直瞬时失败 → 达到 max_retries 后上抛。"""
    import requests

    def always_fail(**kwargs):
        raise requests.exceptions.Timeout("t")

    with mock.patch("research_agent.lm.time.sleep"):
        with pytest.raises(requests.exceptions.Timeout):
            _call_with_retry(always_fail, max_retries=2)


def test_call_with_retry_non_transient_no_retry():
    """非瞬时错误 → 一次都不重试。"""
    calls = {"n": 0}

    def bad(**kwargs):
        calls["n"] += 1
        raise ValueError("400 bad request")

    with pytest.raises(ValueError):
        _call_with_retry(bad, max_retries=3)
    assert calls["n"] == 1


def test_call_with_retry_exponential_backoff_cap(monkeypatch):
    """退避封顶: 超过 cap 后不再增长（不 assert 具体值，只验证 cap 生效）。"""
    import requests
    sleeps = []
    monkeypatch.setattr("research_agent.lm.LLM_RETRY_MAX_DELAY", 2.0)
    monkeypatch.setattr("research_agent.lm.LLM_RETRY_BASE_DELAY", 1.0)
    monkeypatch.setattr("research_agent.lm.random.uniform", lambda a, b: 0.0)

    def flaky(**kwargs):
        raise requests.exceptions.Timeout("t")

    with mock.patch("research_agent.lm.time.sleep", side_effect=lambda s: sleeps.append(s)):
        with pytest.raises(requests.exceptions.Timeout):
            _call_with_retry(flaky, max_retries=4)
    # 退避序列 1,2,2,2（第 3 次起被 cap=2.0 截断）
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


# ================= runtime 附加（顺手补行） =================

def test_runtime_embedding_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_RETRIEVAL_EMBEDDING", "hash")
    assert runtime.runtime_embedding() == "hash"
