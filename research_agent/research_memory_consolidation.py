"""记忆自动沉淀（缺口④，2026-08-16 新增）。

依据 MEMGPT（Packer et al., 2023, arXiv:2310.08560）"睡眠期间记忆被
蒸馏成更抽象的语义记忆"的思想：把零散的 working/episodic 记录按主题
聚类，同类合并为一条 semantic 记忆。

流程: 候选记录（未打标）→ 术语重叠聚类（复用 meaningful_terms，与
检索层口径一致）→ 每簇总结（LLM 优先，可降级规则提取）→ 写入
`store.remember_semantic`，metadata 记 consolidated_at + source_episode_ids，
并给源记录打标防重复。

接口:
- `_cluster_records(records, min_episodes=3)`: 贪心聚类，过滤 < 阈值的簇
- `consolidate_memories(store, llm_call=None, min_episodes=3)`: 主入口，
  返回 [{content, tags, source_episode_ids}]；llm_call 缺省时规则提取

设计约束: 只依赖 ResearchMemoryStore 的内存接口（不新增持久化）；
默认关闭（由挂接方 MEMORY_CONSOLIDATE=1 启用）。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .research_memory import MemoryRecord
from .research_retrieval_runtime import meaningful_terms

CONSOLIDATED_FLAG = "consolidated_at"
RULE_EXCERPT_CHARS = 100
RULE_TOP_TERMS = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_terms(record: MemoryRecord) -> set:
    text = record.content
    for value in (record.metadata or {}).values():
        text += " " + str(value)
    return meaningful_terms(text)


def _cluster_records(
    records: List[MemoryRecord], min_episodes: int = 3
) -> List[List[MemoryRecord]]:
    """按术语重叠贪心聚类：新记录与簇内任一记录共享 >=1 个有意义术语即并入。

    返回满足 len(cluster) >= min_episodes 的簇列表。
    """
    clusters: List[List[MemoryRecord]] = []
    for record in records:
        terms = _record_terms(record)
        if not terms:
            continue
        placed = False
        for cluster in clusters:
            cluster_terms = set()
            for member in cluster:
                cluster_terms |= _record_terms(member)
            if terms & cluster_terms:
                cluster.append(record)
                placed = True
                break
        if not placed:
            clusters.append([record])
    return [c for c in clusters if len(c) >= min_episodes]


def _frequent_terms(cluster: List[MemoryRecord], top: int = RULE_TOP_TERMS) -> List[str]:
    counter: Counter = Counter()
    for record in cluster:
        counter.update(_record_terms(record))
    return [term for term, _ in counter.most_common(top)]


def _rule_summarize(cluster: List[MemoryRecord]) -> tuple:
    """无 LLM 时规则提取：最高频术语 + 首条记录摘录前 100 字。"""
    terms = _frequent_terms(cluster)
    excerpt = str(cluster[0].content).strip()[:RULE_EXCERPT_CHARS]
    content = "主题：{0}。摘录：{1}".format("、".join(terms), excerpt)
    return content, list(terms)


def _llm_summarize(cluster: List[MemoryRecord], llm_call: Callable[[str], str]) -> tuple:
    """调用 LLM 总结。prompt 要求输出一句话 + tags；解析容错（无 tags 行则空）。"""
    records_text = "\n".join(
        "- {0}".format(str(r.content).strip()[:200]) for r in cluster
    )
    prompt = (
        "你是记忆沉淀模块。把下面 {0} 条同主题记录总结成一句话 semantic 记忆，"
        "第二行输出逗号分隔的 #tags。\n记录：\n{1}\n输出："
    ).format(len(cluster), records_text)
    response = str(llm_call(prompt) or "").strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    content = lines[0] if lines else "（LLM 返回为空）"
    tags = []
    for line in lines[1:]:
        tags += [tag.lstrip("#").strip() for tag in line.split(",") if tag.strip()]
    return content, tags


def consolidate_memories(
    store,
    llm_call: Optional[Callable[[str], str]] = None,
    min_episodes: int = 3,
) -> List[Dict]:
    """对 store 的 working+episodic 未打标记录做聚类沉淀。

    每簇: 总结 → remember_semantic(content, metadata={consolidated_at,
    source_episode_ids}, tags) → 源记录打标。返回创建的沉淀条目列表。
    """
    candidates = [
        r
        for r in list(store.working) + list(store.episodic)
        if not (r.metadata or {}).get(CONSOLIDATED_FLAG)
    ]
    clusters = _cluster_records(candidates, min_episodes=min_episodes)
    created: List[Dict] = []
    stamp = _now_iso()
    for cluster in clusters:
        if llm_call is not None:
            content, tags = _llm_summarize(cluster, llm_call)
        else:
            content, tags = _rule_summarize(cluster)
        source_ids = [r.id for r in cluster]
        store.remember_semantic(
            content,
            metadata={
                CONSOLIDATED_FLAG: stamp,
                "source_episode_ids": source_ids,
            },
            tags=tags,
        )
        for record in cluster:
            record.metadata[CONSOLIDATED_FLAG] = stamp
        created.append(
            {"content": content, "tags": tags, "source_episode_ids": source_ids}
        )
    return created
