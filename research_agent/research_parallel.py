"""真·多代理并行（缺口 4，2026-08-16 新增）。

来源: STORM（Shao et al., 2024, arXiv:2402.14207）——多视角并行研究后聚合，
但这里把"LLM 扮演多专家"升级为"真线程并行执行"（每个视角一个独立子任务，
独立检索+独立合成），墙钟时间随视角数近似反比下降。

设计:
- `run_parallel_perspectives` — ThreadPoolExecutor 并行跑每个视角
  - 每个子任务: 检索（视角聚焦查询）→ 合成段落（llm_call）
  - 聚合: 合并证据引用、按视角顺序拼装、按 document_id 去重
  - 失败隔离: 单子任务失败记 error 继续，不拖垮整体
- max_workers 上限保护（默认 3），避免线程风暴

设计约束:
- 全部依赖注入（llm_call / search），离线可用 mock 测并发与隔离
- 默认关闭: 由调用方显式启用，现有串行路径零改动
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 3


def run_parallel_perspectives(
    topic: str,
    perspectives: List[str],
    llm_call: Callable[[str], str],
    search: Callable[[str, int], List[Dict]],
    max_workers: int = DEFAULT_MAX_WORKERS,
    top_k: int = 5,
) -> Dict:
    """并行跑每个视角（检索+合成），失败隔离，聚合结果。"""
    perspectives = [p for p in (perspectives or []) if p and str(p).strip()]
    if not perspectives:
        perspectives = ["通用视角"]
    workers = max(1, min(int(max_workers or 1), len(perspectives)))
    results: List[Optional[Dict]] = [None] * len(perspectives)
    errors: List[Dict] = []
    evidence_by_view: Dict[str, List[Dict]] = {}

    def run_one(index: int, perspective: str) -> Dict:
        return _run_single_perspective(
            topic, perspective, llm_call=llm_call, search=search, top_k=top_k
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_one, index, perspective): (index, perspective)
            for index, perspective in enumerate(perspectives)
        }
        for future in concurrent.futures.as_completed(future_map):
            index, perspective = future_map[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 —— 失败隔离：单视角失败不影响整体
                logger.warning("perspective %s failed: %s", perspective, exc)
                errors.append({"perspective": perspective, "error": repr(exc)})

    # 聚合（按视角顺序拼装，失败视角留占位说明）
    sections = []
    seen_evidence: set = set()
    merged_evidence: List[Dict] = []
    for index, perspective in enumerate(perspectives):
        outcome = results[index]
        if outcome is None:
            sections.append(
                {
                    "perspective": perspective,
                    "paragraph": "",
                    "error": "子任务失败（已隔离，不影响其他视角）",
                }
            )
            continue
        sections.append(
            {"perspective": perspective, "paragraph": outcome["paragraph"]}
        )
        evidence_by_view[perspective] = outcome["evidence"]
        for item in outcome["evidence"]:
            key = item.get("document_id") or item.get("chunk_id") or item.get("url")
            if key is None or key not in seen_evidence:
                merged_evidence.append(item)
                if key is not None:
                    seen_evidence.add(key)

    return {
        "topic": topic,
        "perspectives": perspectives,
        "sections": sections,
        "evidence": merged_evidence,
        "evidence_by_view": evidence_by_view,
        "errors": errors,
        "parallel": {"workers": workers, "perspective_count": len(perspectives)},
        "converged": not errors,  # 无失败视角 → 完整收敛
    }


def _run_single_perspective(
    topic: str,
    perspective: str,
    *,
    llm_call: Callable[[str], str],
    search: Callable[[str, int], List[Dict]],
    top_k: int,
) -> Dict:
    """单个视角子任务: 视角聚焦检索 → 合成一段（可独立失败）。"""
    focus_query = "{0} 视角：{1}".format(topic, perspective)
    ranked = search(focus_query, top_k)
    evidence = [dict(item) for item in ranked]
    excerpt = "\n".join(
        "{0}\n{1}".format(item.get("title", ""), str(item.get("content") or "")[:500])
        for item in evidence[:5]
    )
    prompt = (
        "你是论文调研专家，负责「{0}」视角。基于以下证据写一段中文综述（120 字内，"
        "可引用 [1][2] 编号）：\n主题：{1}\n证据：\n{2}".format(perspective, topic, excerpt)
    )
    try:
        reply = llm_call(prompt)
        paragraph = (
            reply[0] if isinstance(reply, list) and reply else reply
        )
        paragraph = str(paragraph or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("perspective %s synthesis failed: %s", perspective, exc)
        paragraph = ""
    if not paragraph:
        raise RuntimeError("empty synthesis for perspective: {0}".format(perspective))
    return {"perspective": perspective, "paragraph": paragraph, "evidence": evidence}
