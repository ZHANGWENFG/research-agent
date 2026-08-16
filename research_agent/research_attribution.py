"""事实性验证（attribution）层（缺口①，2026-08-16 新增）。

来源: Attributed QA（Rashkin et al., 2021, arXiv:2112.11961）——答案句子
逐句回链到证据，无法回链的句子显式标注；Self-RAG（Asai et al., 2023）
的"验证断言是否被证据支撑"思想。

三个层次：
1. `split_sentences(text)` — 中英文句号/感叹/问号拆句（纯函数，引用编号 [n] 不被拆断）
2. `verify_claim(claim, evidence_pool)` — 单句回链 {supported, evidence_ids, score}
   - 信号 = 有意义术语重叠（复用 `meaningful_terms`，与检索层口径一致）
     + 数字命中（句子里的数字/百分比在证据中出现）加权
     + 专有名词（全大写英文词如 PIM/RF/MMR）命中加权
3. `verify_article(article, evidence_pool)` — 全文逐句验证，输出覆盖比率与
   unsupported 明细。**不删文只标注**——保持"证据驱动的诚实"。

设计约束: 纯函数 + 依赖注入（evidence 全部显式传入），离线可测；
默认关闭（由挂接方 RESEARCH_ATTRIBUTION=1 启用），不改动默认执行路径。
"""
from __future__ import annotations

import re
from typing import Dict, List

# 单句判定阈值: 有意义术语重叠 >=2 → supported；1 个重叠但带数字命中 → supported
MIN_TERM_OVERLAP = 2
PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9]*\b")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")

# 句子边界: 中英文句末标点后切分（英文句号后带空格、中文连续句无空格都处理）
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)(?!\d)\s+")


def split_sentences(text: str) -> List[str]:
    """中英文拆句。引用编号 [n] 位于句内不被拆断。

    "A 方法有效[1]。B 方法无效。" → ["A 方法有效[1]。", "B 方法无效。"]
    "First works. Second fails." → ["First works.", "Second fails."]
    """
    text = str(text or "").strip()
    if not text:
        return []
    segments = _SENTENCE_BOUNDARY.split(text)
    return [seg.strip() for seg in segments if seg.strip()]


def _numbers(text: str) -> set:
    return set(NUMBER_PATTERN.findall(text))


def _proper_nouns(text: str) -> set:
    return set(PROPER_NOUN_PATTERN.findall(text))


def _term_overlap(sentence: str, evidence_text: str):
    """有意义术语重叠（复用检索层 meaningful_terms）。"""
    from .research_retrieval_runtime import meaningful_terms

    sentence_terms = meaningful_terms(sentence)
    evidence_terms = meaningful_terms(evidence_text)
    # 纯数字 token（如 "95"）不算术语重叠——数字命中由 _numbers 单独加权
    return sorted(
        t for t in (sentence_terms & evidence_terms)
        if not re.fullmatch(r"\d+(?:\.\d+)?%?", t)
    )


def verify_claim(
    claim: str,
    evidence_pool: List[Dict],
    min_term_overlap: int = MIN_TERM_OVERLAP,
) -> Dict:
    """单句回链验证。

    evidence_pool: [{id: str|int, content: str}]（content 为证据文本）。
    返回 {supported, evidence_ids, score, reason}。
    - supported: 有足够术语重叠，或数字命中且至少有部分重叠
    - 空证据池 → supported=False, reason="no evidence"
    """
    claim = str(claim or "").strip()
    if not claim:
        return {"supported": False, "evidence_ids": [], "score": 0.0,
                "reason": "empty claim"}
    if not evidence_pool:
        return {"supported": False, "evidence_ids": [], "score": 0.0,
                "reason": "no evidence"}

    claim_numbers = _numbers(claim)
    claim_nouns = _proper_nouns(claim)

    best_score = 0.0
    best_ids: List = []
    best_overlap: List = []
    best_number_hits: List = []
    best_noun_hits: List = []

    for item in evidence_pool:
        content = str(item.get("content") or "")
        overlap = _term_overlap(claim, content)
        number_hits = sorted(claim_numbers & _numbers(content))
        noun_hits = sorted(claim_nouns & _proper_nouns(content))
        score = len(overlap) * 1.0 + len(number_hits) * 2.0 + len(noun_hits) * 1.0
        if score > best_score:
            best_score = score
            best_ids = [item.get("id")]
            best_overlap = overlap
            best_number_hits = number_hits
            best_noun_hits = noun_hits
        elif score == best_score and best_ids and item.get("id") not in best_ids:
            best_ids.append(item.get("id"))

    overlap_ok = len(best_overlap) >= min_term_overlap
    number_ok = bool(best_number_hits) and bool(best_overlap)
    supported = overlap_ok or number_ok

    if not supported:
        if not best_overlap:
            reason = "no meaningful overlap with any evidence"
        elif best_number_hits:
            reason = "partial overlap（有数字命中但术语重叠不足）"
        else:
            reason = "partial overlap（术语重叠不足）"
    else:
        reason = "supported by evidence"
    return {
        "supported": supported,
        "evidence_ids": best_ids,
        "score": round(best_score, 4),
        "reason": reason,
        "overlap": best_overlap,
        "number_hits": best_number_hits,
        "proper_noun_hits": best_noun_hits,
    }


def verify_article(article: str, evidence_pool: List[Dict]) -> Dict:
    """全文逐句验证。返回:
    {total_sentences, supported_count, unsupported: [{sentence, evidence_ids,
      score, reason}], coverage_ratio, results: [逐句结果]}。
    空证据池 → 全部句子 unsupported + reason="no evidence"（不删文）。
    """
    sentences = split_sentences(article)
    total = len(sentences)
    results = [verify_claim(sentence, evidence_pool) for sentence in sentences]
    supported_count = sum(1 for r in results if r["supported"])
    unsupported = [
        {
            "sentence": sentence,
            "evidence_ids": r["evidence_ids"],
            "score": r["score"],
            "reason": r["reason"],
        }
        for sentence, r in zip(sentences, results)
        if not r["supported"]
    ]
    return {
        "total_sentences": total,
        "supported_count": supported_count,
        "unsupported": unsupported,
        "coverage_ratio": round(supported_count / total, 4) if total else 0.0,
        "results": results,
    }
