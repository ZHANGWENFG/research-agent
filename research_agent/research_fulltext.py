"""全文获取层（改造新增，合规获取全文）。

优先级四级：
1. 文本接口优先（不产生文件）：PMC efetch（有 pmcid）/ Europe PMC REST（有 pmid）/ Unpaywall（有 doi）
2. 白名单下载自动放行：域名在可信清单（PMC/EuropePMC/arXiv）→ 三重校验后直接下载
3. 非白名单下载人工审批：生成审批请求（SSE 推前端）→ 用户批准/拒绝（普通 POST 回传）→ 批准才下载
4. 兜底：只给原文链接

可信判定：PMC 编号 / Unpaywall 返回 = 权威凭证（不用正则判可信，正则只校验格式）。

【改造要点】download 是危险操作：白名单自动放行 + 非白名单人工审批；全程审计。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# 可信下载白名单（自动放行）
WHITELIST_DOMAINS = ("pmc.ncbi.nlm.nih.gov", "europepmc.org", "arxiv.org")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_CONTENT_TYPES = ("application/pdf", "application/xml", "text/xml")
PDF_HEADER = b"%PDF"

# 审批状态机
PENDING, APPROVED, REJECTED, DOWNLOADING, DONE, FAILED = (
    "pending", "approved", "rejected", "downloading", "done", "failed",
)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


def _domain_of(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url or "")
    return match.group(1).lower() if match else ""


def is_whitelisted(url: str) -> bool:
    domain = _domain_of(url)
    return any(domain == d or domain.endswith("." + d) for d in WHITELIST_DOMAINS)


class ApprovalQueue:
    """非白名单下载人工审批队列（人在环，human-in-the-loop）。

    状态机：pending → approved → downloading → done / failed
                  └──→ rejected（只保留链接，不下载）
    存储：SQLite（approvals 表），线程安全；审批结果回传走普通 HTTP 接口。
    """

    def __init__(self, database_path: str):
        self.database_path = str(database_path)
        self._lock = threading.Lock()
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    url TEXT,
                    source TEXT,
                    size_hint INTEGER,
                    risk_level TEXT,
                    status TEXT,
                    reason TEXT,
                    requested_at TEXT,
                    decided_at TEXT
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.database_path)

    def create(self, url: str, source: str, size_hint: int = 0, task_id: str = "") -> Dict:
        with self._lock, self._connect() as conn:
            row = {
                "id": uuid.uuid4().hex,
                "task_id": task_id,
                "url": url,
                "source": source,
                "size_hint": size_hint,
                "risk_level": "high" if not is_whitelisted(url) else "low",
                "status": PENDING,
                "reason": "",
                "requested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "decided_at": "",
            }
            conn.execute(
                """
                INSERT INTO approvals (id, task_id, url, source, size_hint, risk_level,
                                       status, reason, requested_at, decided_at)
                VALUES (:id, :task_id, :url, :source, :size_hint, :risk_level,
                        :status, :reason, :requested_at, :decided_at)
                """,
                row,
            )
        return row

    def resolve(self, approval_id: str, approve: bool, reason: str = "") -> Optional[Dict]:
        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,))
            row = cur.fetchone()
            if not row:
                return None
            columns = [d[0] for d in cur.description]
            record = dict(zip(columns, row))
            if record["status"] != PENDING:
                return record
            status = APPROVED if approve else REJECTED
            conn.execute(
                "UPDATE approvals SET status=?, reason=?, decided_at=? WHERE id=?",
                (status, reason, time.strftime("%Y-%m-%d %H:%M:%S"), approval_id),
            )
            record["status"] = status
            record["reason"] = reason
        return record

    def mark(self, approval_id: str, status: str):
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE approvals SET status=? WHERE id=?", (status, approval_id))

    def list_pending(self) -> List[Dict]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM approvals WHERE status=?", (PENDING,))
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def _validate_response(response: requests.Response) -> Optional[str]:
    """下载三重校验：内容类型 / 大小 / （下载后文件头）。返回错误信息或 None（通过）。"""
    content_type = (response.headers.get("content-type") or "").lower()
    if content_type and not any(ct in content_type for ct in ALLOWED_CONTENT_TYPES):
        return f"content-type not allowed: {content_type}"
    length = int(response.headers.get("content-length") or 0)
    if length > MAX_FILE_SIZE:
        return f"file too large: {length} bytes"
    return None


def download_file(url: str, output_dir: str, max_size: int = MAX_FILE_SIZE) -> Dict:
    """白名单/已审批下载：流式写入 + 大小限制 + 文件头校验。"""
    response = requests.get(url, stream=True, timeout=30.0)
    response.raise_for_status()
    error = _validate_response(response)
    if error:
        return {"ok": False, "error": error}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1].split("?")[0] or "download.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    target = Path(output_dir) / name
    size = 0
    with open(target, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            size += len(chunk)
            if size > max_size:
                handle.close()
                target.unlink(missing_ok=True)
                return {"ok": False, "error": "size exceeded during stream"}
            handle.write(chunk)
    # 文件头校验（PDF 必须是 %PDF 开头）
    with open(target, "rb") as handle:
        header = handle.read(4)
    if target.suffix.lower() == ".pdf" and not header.startswith(PDF_HEADER):
        target.unlink(missing_ok=True)
        return {"ok": False, "error": "file header is not %PDF"}
    # 提取文本（主流做法: 下载不是终点，全文要进生成链路才能支撑引用）
    # 提取失败不致命——保留下载成功 + path，preview 降级
    preview = ""
    if target.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(target))
            extracted = []
            for page in reader.pages[:5]:  # 只读前 5 页，防超大 PDF 拖慢
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    extracted.append(page_text)
            preview = "\n".join(extracted)[:800]
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf text extraction failed %s: %s", target.name, exc)
    return {"ok": True, "path": str(target), "size": size, "preview": preview}


# ---------- 文本接口（不产生文件，最干净） ----------

def fetch_pmc_fulltext(pmcid: str) -> Optional[str]:
    """PMC efetch：按 PMC 编号直接返回全文 XML 文本。"""
    if not pmcid:
        return None
    try:
        response = requests.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params={"db": "pmc", "id": pmcid, "rettype": "xml", "retmode": "xml"},
            timeout=30.0,
        )
        response.raise_for_status()
        text = response.text
        return text if len(text.strip()) > 200 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("PMC fulltext failed for %s: %s", pmcid, exc)
        return None


def fetch_europepmc_fulltext(pmid: str) -> Optional[str]:
    """Europe PMC REST：按 PMID 返回全文 XML/文本。"""
    if not pmid:
        return None
    try:
        response = requests.get(
            f"{EUROPE_PMC_BASE}/search",
            params={"query": f"EXT_ID:{pmid}", "resultType": "core", "format": "json"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        result = (data.get("resultList") or {}).get("result") or []
        if not result:
            return None
        full_text_xml = result[0].get("fullTextXml") or ""
        if len(full_text_xml.strip()) > 200:
            return full_text_xml
        # 退回 fullTextUrlList 里的白名单链接
        for item in result[0].get("fullTextUrlList", {}).get("fullTextUrl", []):
            url = item.get("url", "")
            if url and is_whitelisted(url):
                return url
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("EuropePMC fulltext failed for %s: %s", pmid, exc)
        return None


def lookup_unpaywall(doi: str, email: str = "local@example.com") -> Optional[str]:
    """Unpaywall：DOI 换合法开放获取链接（权威凭证，学术标准做法）。"""
    if not doi:
        return None
    try:
        response = requests.get(
            f"{UNPAYWALL_BASE}/{doi}", params={"email": email}, timeout=20.0
        )
        if response.status_code != 200:
            return None
        data = response.json()
        location = data.get("best_oa_location") or {}
        return location.get("url_for_pdf") or location.get("url") or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unpaywall lookup failed for %s: %s", doi, exc)
        return None


# ---------- 统一入口 ----------

def get_fulltext(
    article: Dict,
    download_dir: str,
    approval_queue: Optional[ApprovalQueue] = None,
    task_id: str = "",
    email: str = "local@example.com",
) -> Dict:
    """按优先级尝试拿全文（四级方案）。

    article 元数据字段：pmcid / pmid / doi / url（来自 PubMedRM/ArxivRM 的 meta）。
    返回结构兼容旧版闭包契约（ok/source/chars/preview），并带 status 字段:
      {ok: bool, status: fulltext|link|pending_approval|failed|none,
       source: pmc|europepmc|whitelist_download|approval_required|none,
       content?/path?/preview?, approval_id?, url?, message?}

    这是全项目唯一的 get_fulltext 实现（2026-08-16 收敛）：
    原 service/graph_adapter 各有一份闭包重写，已出现行为漂移——
    本函数吸收了两份的健壮性（逐步 try/except 降级，单点失败不拖垮整轮检索）。
    """
    pmcid = (article.get("meta") or {}).get("pmcid") or ""
    pmid = (article.get("meta") or {}).get("pmid") or ""
    doi = (article.get("meta") or {}).get("doi") or ""
    url = article.get("url") or ""
    title = str(article.get("title") or "")[:120]

    # 1. 文本接口优先（不产生文件）
    if pmcid:
        try:
            text = fetch_pmc_fulltext(pmcid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pmc fulltext failed %s: %s", pmcid, exc)
            text = None
        if text:
            return {"ok": True, "status": "fulltext", "source": "pmc",
                    "pmcid": pmcid, "chars": len(text), "preview": text[:800],
                    "content": text, "url": url}
    if pmid:
        try:
            text = fetch_europepmc_fulltext(pmid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("europepmc fulltext failed %s: %s", pmid, exc)
            text = None
        if text and text.startswith("<"):
            return {"ok": True, "status": "fulltext", "source": "europepmc",
                    "pmid": pmid, "chars": len(text), "preview": text[:800],
                    "content": text, "url": url}
        if text and text.startswith("http"):
            url = text  # EuropePMC 返回了白名单链接，走下载

    # 2. Unpaywall 找 OA 链接（权威凭证）
    if doi:
        try:
            oa_url = lookup_unpaywall(doi, email=email)
        except Exception as exc:  # noqa: BLE001
            logger.warning("unpaywall lookup failed %s: %s", doi, exc)
            oa_url = None
        if oa_url:
            url = oa_url

    # 3/4. 下载：白名单自动放行，非白名单人工审批
    if url and is_whitelisted(url):
        try:
            result = download_file(url, download_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("whitelist download failed %s: %s", url, exc)
            result = {"ok": False, "error": repr(exc)}
        if result.get("ok"):
            return {"ok": True, "status": "fulltext", "source": "whitelist_download",
                    "path": result["path"], "url": url,
                    "chars": int(result.get("size") or 0),
                    "preview": f"downloaded:{result.get('size')}B"}
        return {"ok": False, "status": "failed", "source": "whitelist_download",
                "message": result.get("error"), "url": url}

    if url and approval_queue is not None:
        try:
            approval = approval_queue.create(
                url=url, source=title, size_hint=0, task_id=task_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval queue create failed %s: %s", url, exc)
            return {"ok": False, "status": "failed", "source": "approval_required",
                    "url": url, "message": repr(exc)}
        return {"ok": False, "status": "pending_approval",
                "source": "approval_required", "approval_id": approval["id"],
                "status_code": approval["status"], "url": url,
                "message": "非白名单下载，等待人工审批", "approval": approval}

    # 兜底：只给链接
    return {"ok": False, "status": "link", "source": "none", "url": url,
            "message": "仅提供原文链接（文本接口未命中且无白名单/审批下载）"}
