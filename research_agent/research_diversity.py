"""MMR 多样性重排（P2-A2, 2026-08-16 新增）。

来源: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for
Reordering Documents and Producing Summaries", SIGIR'98.

MMR(d) = λ · rel(d) − (1−λ) · max_{s∈S} sim(d, s)

- rel(d): 相关性（默认取候选自带 score 字段，零依赖）
- sim(d,s): 相似度，默认词集 Jaccard（无 embedding 依赖、离线可测）；
  调用方可注入任意相似度函数（如 cosine embedding sim）。
- λ ∈ [0.5, 0.8] 为原文建议区间，主流默认 λ=0.7。

用途: 检索/重排结果去冗余——多篇讲同一方法的论文只保留相关性最强的一篇，
其余位置让给不同视角的证据（多样性），提升最终报告的覆盖面。
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

# 默认多样性强度（原文建议 0.5–0.8，0.7 为生产默认）
DEFAULT_MMR_LAMBDA = 0.7

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_TOKEN = re.compile(r"[a-zA-Z0-9_\-]+")


def token_set(text) -> set:
    """检索用词集: 英文/数字 token + 中文连续串（与检索层口径一致）。"""
    text = str(text or "").lower()
    tokens = set(_LATIN_TOKEN.findall(text))
    for run in _CJK_RUN.findall(text):
        if run:
            tokens.add(run)
    tokens.discard("")
    return tokens


def jaccard_similarity(a, b) -> float:
    """词集 Jaccard: |A∩B| / |A∪B|，空集对任意文本 = 0（无信息量）。"""
    set_a = token_set(a)
    set_b = token_set(b)
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def mmr_rerank(
    ranked: List[Dict],
    *,
    relevance_key: str = "score",
    lambda_: float = DEFAULT_MMR_LAMBDA,
    top_k: Optional[int] = None,
    similarity: Optional[Callable[[Dict, Dict], float]] = None,
    text_key: str = "content",
) -> List[Dict]:
    """对已排序候选做 MMR 多样性重排。

    贪心选择: 每轮选 MMR 分数最高者入列，已选集合参与相似度惩罚。
    空输入 → 空列表；单元素 → 原样返回。
    """
    if not ranked:
        return []
    lambda_ = float(lambda_)
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    sim = similarity or (lambda d1, d2: jaccard_similarity(
        d1.get(text_key, ""), d2.get(text_key, "")
    ))
    remaining = list(ranked)
    selected: List[Dict] = []
    while remaining and (top_k is None or len(selected) < top_k):
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(remaining):
            relevance = float(candidate.get(relevance_key) or 0.0)
            redundancy = 0.0
            if selected:
                redundancy = max(sim(candidate, picked) for picked in selected)
            mmr = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected
