"""P1 补测: research_fulltext.py 文本接口（fetch_pmc/europepmc/unpaywall）。

合规获取层"文本接口优先"——mock requests，验证每种响应分支
（成功 XML / 过短 / 无结果 / 非 200 / 网络异常 / 空入参）。
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent import research_fulltext as ft


class _FakeTextResponse:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP {0}".format(self.status_code))

    def json(self):
        import json
        return json.loads(self.text) if isinstance(self.text, str) else self.text


def test_fetch_pmc_empty_id():
    assert ft.fetch_pmc_fulltext("") is None
    assert ft.fetch_pmc_fulltext(None) is None


def test_fetch_pmc_ok_returns_xml():
    xml = "<article>" + "x" * 500 + "</article>"
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse(xml)):
        assert ft.fetch_pmc_fulltext("PMC1234567") == xml


def test_fetch_pmc_too_short_returns_none():
    """NCBI 返回过短文本（无实际全文）→ None，不产出垃圾。"""
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse("<empty/>")):
        assert ft.fetch_pmc_fulltext("PMC123") is None


def test_fetch_pmc_network_error_returns_none():
    with mock.patch("research_agent.research_fulltext.requests.get",
                    side_effect=RuntimeError("timeout")):
        assert ft.fetch_pmc_fulltext("PMC123") is None


def test_fetch_europepmc_no_result_returns_none():
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse('{"resultList": {"result": []}}')):
        assert ft.fetch_europepmc_fulltext("34567890") is None


def test_fetch_europepmc_ok_returns_xml():
    payload = '{"resultList": {"result": [{"fullTextXml": "' + "x" * 500 + '"}]}}'
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse(payload)):
        assert ft.fetch_europepmc_fulltext("34567890") == "x" * 500


def test_fetch_europepmc_falls_back_to_whitelist_url():
    """XML 过短 → 退回 fullTextUrlList 里的白名单链接。"""
    payload = (
        '{"resultList": {"result": [{"fullTextXml": "short", '
        '"fullTextUrlList": {"fullTextUrl": ['
        '{"url": "https://evil.com/x"}, '
        '{"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1"}]}}]}}'
    )
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse(payload)):
        result = ft.fetch_europepmc_fulltext("34567890")
    assert result == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1"


def test_fetch_europepmc_empty_id():
    assert ft.fetch_europepmc_fulltext("") is None


def test_lookup_unpaywall_empty_doi():
    assert ft.lookup_unpaywall("") is None


def test_lookup_unpaywall_ok_pdf_url_first():
    payload = (
        '{"best_oa_location": {"url_for_pdf": "https://oa.org/pdf", '
        '"url": "https://oa.org/landing"}}'
    )
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse(payload)):
        assert ft.lookup_unpaywall("10.1000/xyz") == "https://oa.org/pdf"


def test_lookup_unpaywall_404_returns_none():
    with mock.patch("research_agent.research_fulltext.requests.get",
                    return_value=_FakeTextResponse("", status=404)):
        assert ft.lookup_unpaywall("10.1000/missing") is None
