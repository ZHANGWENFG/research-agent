"""P1 补测: get_fulltext 四级优先级（fulltext 40% → 拉高）。

全链路 mock，验证:
1. pmcid 文本接口命中 → fulltext/source=pmc
2. pmid → EuropePMC XML → fulltext
3. EuropePMC 返回白名单 URL → 走下载
4. doi → Unpaywall OA 链接 → 白名单下载
5. 非白名单 + approval_queue → pending_approval
6. 非白名单 + 无队列 → link 兜底
7. 下载失败 → failed 且不拖垮调用
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from research_agent.research_fulltext import get_fulltext


def _article(meta=None, url="", title="论文标题"):
    return {"meta": meta or {}, "url": url, "title": title}


def test_get_fulltext_pmc_first_priority():
    """pmcid 存在 → 直接 pmc 全文，不碰后续逻辑。"""
    with mock.patch("research_agent.research_fulltext.fetch_pmc_fulltext",
                    return_value="<article>" + "x" * 500 + "</article>"):
        result = get_fulltext(_article({"pmcid": "PMC123"}), "out")
    assert result["status"] == "fulltext"
    assert result["source"] == "pmc"
    assert result["chars"] > 500


def test_get_fulltext_pmc_fail_then_europepmc():
    """pmc 失败 → 降级 europepmc（单点失败不拖垮整轮）。"""
    pmc = mock.MagicMock(side_effect=RuntimeError("pmc down"))
    epmc = mock.MagicMock(return_value="<full>" + "y" * 500 + "</full>")
    with mock.patch("research_agent.research_fulltext.fetch_pmc_fulltext", pmc), \
         mock.patch("research_agent.research_fulltext.fetch_europepmc_fulltext", epmc):
        result = get_fulltext(
            _article({"pmcid": "PMC1", "pmid": "34567890"}), "out"
        )
    assert result["source"] == "europepmc"
    assert result["status"] == "fulltext"


def test_get_fulltext_europepmc_url_falls_to_download():
    """EuropePMC 返回白名单 URL → 转下载（whitelist_download）。"""
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9"
    with mock.patch("research_agent.research_fulltext.fetch_pmc_fulltext",
                    return_value=None), \
         mock.patch("research_agent.research_fulltext.fetch_europepmc_fulltext",
                    return_value=url), \
         mock.patch("research_agent.research_fulltext.download_file",
                    return_value={"ok": True, "path": "/tmp/x.pdf", "size": 1024}):
        result = get_fulltext(_article({"pmid": "34567890"}), "out")
    assert result["source"] == "whitelist_download"
    assert result["status"] == "fulltext"


def test_get_fulltext_unpaywall_oa_link_downloads():
    """doi → Unpaywall OA 链接（arxiv 白名单）→ 自动下载。"""
    oa_url = "https://arxiv.org/pdf/2401.12345"
    with mock.patch("research_agent.research_fulltext.fetch_pmc_fulltext",
                    return_value=None), \
         mock.patch("research_agent.research_fulltext.fetch_europepmc_fulltext",
                    return_value=None), \
         mock.patch("research_agent.research_fulltext.lookup_unpaywall",
                    return_value=oa_url), \
         mock.patch("research_agent.research_fulltext.download_file",
                    return_value={"ok": True, "path": "/tmp/p.pdf", "size": 2048}):
        result = get_fulltext(_article({"doi": "10.1000/xyz"}), "out")
    assert result["source"] == "whitelist_download"


def test_get_fulltext_non_whitelist_creates_approval():
    """非白名单 + 审批队列 → pending_approval。"""
    queue = mock.MagicMock()
    queue.create.return_value = {"id": "a1", "status": "pending"}
    result = get_fulltext(
        _article({}, url="https://journal.evil.com/x"),
        "out",
        approval_queue=queue,
    )
    assert result["status"] == "pending_approval"
    assert result["approval_id"] == "a1"
    assert "人工审批" in result["message"]
    queue.create.assert_called_once()


def test_get_fulltext_non_whitelist_no_queue_link_fallback():
    """非白名单 + 无审批队列 → 只给链接（四级兜底）。"""
    result = get_fulltext(_article({}, url="https://elsewhere.org/x"), "out")
    assert result["status"] == "link"
    assert result["source"] == "none"
    assert result["url"] == "https://elsewhere.org/x"


def test_get_fulltext_empty_article_link():
    result = get_fulltext({}, "out")
    assert result["status"] == "link"
    assert result["url"] == ""


def test_get_fulltext_whitelist_download_failure_reported():
    """白名单下载失败 → status=failed + 错误信息，不抛异常。"""
    with mock.patch("research_agent.research_fulltext.fetch_pmc_fulltext",
                    return_value=None), \
         mock.patch("research_agent.research_fulltext.fetch_europepmc_fulltext",
                    return_value=None), \
         mock.patch("research_agent.research_fulltext.download_file",
                    return_value={"ok": False, "error": "size exceeded"}):
        result = get_fulltext(_article({}, url="https://arxiv.org/pdf/x.pdf"), "out")
    assert result["status"] == "failed"
    assert result["message"] == "size exceeded"
