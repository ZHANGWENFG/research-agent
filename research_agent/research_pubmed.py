"""PubMed 检索器（改造新增，主线数据源之一）。

流程：esearch 按关键词拿 PMID 列表 → esummary 批量拿元数据（标题/期刊/作者/日期）
     → efetch 拿摘要文本；PMC 编号直接带入元数据，供全文获取层第一步使用。
返回格式与 ArxivRM 统一：{url, title, description(摘要), snippets, meta{...}}，
下游检索、证据裁判、引用组装全部无感。

【改造要点】0.4 秒节流（PubMed 免费接口约 3 次/秒，防 429）；单用户模式无 API Key 也可用。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_THROTTLE = 0.4  # 秒，PubMed 免费接口限流间隔


class PubMedRM:
    """Retrieve paper metadata + abstracts from PubMed via NCBI E-utilities."""

    def __init__(
        self,
        k: int = 3,
        email: str = "",
        api_key: str = "",
        throttle: float = DEFAULT_THROTTLE,
        timeout: float = 20.0,
    ):
        self.k = int(k)
        self.email = email or "local@example.com"
        self.api_key = api_key
        self.throttle = float(throttle)
        self.timeout = float(timeout)
        self.usage = 0
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    # ---------- 节流 ----------
    def _throttle(self):
        """每个请求之间至少间隔 throttle 秒（防 PubMed 限流）。"""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < self.throttle:
                time.sleep(self.throttle - elapsed)
            self._last_request_at = time.monotonic()

    def _get(self, url: str, params: Dict) -> dict:
        self._throttle()
        self.usage += 1
        params.setdefault("tool", "research-agent")
        params.setdefault("email", self.email)
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ---------- 三步取数 ----------
    def _esearch(self, term: str, retmax: int) -> List[str]:
        """按关键词搜，返回 PMID 列表。"""
        data = self._get(
            f"{EUTILS_BASE}/esearch.fcgi",
            {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax},
        )
        return data.get("esearchresult", {}).get("idlist", [])

    def _esummary(self, pmids: List[str]) -> Dict[str, dict]:
        """批量拿元数据：标题/期刊/日期/作者/PMCID。"""
        if not pmids:
            return {}
        data = self._get(
            f"{EUTILS_BASE}/esummary.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        )
        return data.get("result", {})

    def _efetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """批量拿摘要文本（efetch rettype=abstract）。"""
        if not pmids:
            return {}
        self._throttle()
        self.usage += 1
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
            "tool": "research-agent",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        response = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=self.timeout)
        response.raise_for_status()
        # 按 PMID 切分文本块（NCBI 输出以 PMID 分隔）
        text = response.text
        blocks: Dict[str, str] = {}
        current_pmid: Optional[str] = None
        current_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # NCBI 输出以 "1. PMID" / "2. PMID" ... 编号分隔——旧实现只认
            # "1. " 导致多篇摘要的后续分隔符被当正文吞掉（真实 bug，B1 测试抓到）
            if re.match(r"^\d+\. \d+$", stripped) or stripped.isdigit():
                if current_pmid and current_lines:
                    blocks[current_pmid] = "\n".join(current_lines).strip()
                current_pmid = stripped.split(". ")[-1].strip()
                current_lines = []
            else:
                if current_pmid:
                    current_lines.append(line)
        if current_pmid and current_lines:
            blocks[current_pmid] = "\n".join(current_lines).strip()
        return blocks

    # ---------- 统一 forward 接口（与 ArxivRM 对齐） ----------
    def forward(self, query: str) -> List[Dict]:
        pmids = self._esearch(query, retmax=self.k)
        if not pmids:
            return []
        summary = self._esummary(pmids)
        abstracts = self._efetch_abstracts(pmids)
        results: List[Dict] = []
        for pmid in pmids:
            item = summary.get(pmid, {})
            if not item:
                continue
            title = item.get("title") or ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            abstract = abstracts.get(pmid, "")
            description = abstract or (item.get("summary") or "")
            # 提取 PMC 编号（全文获取层第一步就靠它）
            pmcid = ""
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "pmc":
                    pmcid = aid.get("value", "")
            results.append(
                {
                    "url": url,
                    "title": title,
                    "description": description,
                    "snippets": [abstract] if abstract else [],
                    "meta": {
                        "source_type": "pubmed",
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "journal": item.get("fulljournalname") or item.get("source") or "",
                        "pubdate": item.get("pubdate") or item.get("epubdate") or "",
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                    },
                }
            )
        return results

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"PubMedRM": usage}


def search_pubmed(query: str, k: int = 3) -> List[Dict]:
    """便捷入口：直接按查询词搜 PubMed。"""
    return PubMedRM(k=k).forward(query)
