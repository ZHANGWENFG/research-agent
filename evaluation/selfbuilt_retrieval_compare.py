# -*- coding: utf-8 -*-
"""自建语料：新旧检索栈对比（legacy vs V4.1 hybrid）——可复现评测。

测试设定（2026-08-13 实测）：
- 语料：项目文档（my-agent-项目全解.md + README.md）按 ## 小节切块，31 节
- 用例：19 个口语化中文查询（模拟真实用户提问，非照抄标题）
- 嵌入：HashEmbeddingProvider(64 维)，零外部依赖，无需 LLM/API
- 指标：Recall@5 / MRR / nDCG@5 / 平均延迟

实测结果（本机，Python 3.13）：
- 旧栈(legacy)  : Recall@5=0.1579  MRR=0.0833  nDCG@5=0.0965  延迟=2.96ms
- 新栈(V4.1)    : Recall@5=0.2632  MRR=0.1404  nDCG@5=0.1554  延迟=1.37ms
- 提升          : Recall@5 +66.69%  MRR +68.55%  nDCG@5 +61.04%  延迟 -54%

运行：python evaluation/selfbuilt_retrieval_compare.py
依赖：numpy、rank-bm25（轻量）；其余项目重依赖（litellm 等）在脚本内 stub，不影响本评测。
"""

import sys
import types
import re
import time
import statistics
from pathlib import Path


def _stub(name):
    sys.modules[name] = types.ModuleType(name)


def _install_stubs():
    """顶掉 lm.py/utils.py 的重依赖（本评测只跑检索，不碰 LLM）。"""
    for mod in [
        "litellm", "openai", "anthropic", "transformers", "requests", "ujson",
        "regex", "toml", "tqdm", "httpx", "trafilatura", "langchain_text_splitters",
    ]:
        _stub(mod)
    litellm = sys.modules["litellm"]
    caching = types.ModuleType("litellm.caching")
    caching_pkg = types.ModuleType("litellm.caching.caching")

    class Cache:
        def __init__(self, *a, **k):
            pass

    caching_pkg.Cache = Cache
    caching.caching = caching_pkg
    litellm.caching = caching
    sys.modules["litellm.caching"] = caching
    sys.modules["litellm.caching.caching"] = caching_pkg
    litellm.drop_params = True
    litellm.telemetry = False
    litellm.cache = None
    litellm.completion = lambda *a, **k: None
    litellm.text_completion = lambda *a, **k: None
    openai = sys.modules["openai"]
    openai.OpenAI = type("OpenAI", (), {})
    openai.AzureOpenAI = type("AzureOpenAI", (), {})
    sys.modules["ujson"].dumps = lambda *a, **k: ""
    sys.modules["ujson"].loads = lambda *a, **k: {}
    sys.modules["tqdm"].tqdm = type("tqdm", (), {})
    sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = type(
        "RCTS", (), {}
    )
    sys.modules["trafilatura"].extract = lambda *a, **k: ""


def load_sections(path):
    """按 '## ' 标题切块，返回 [(标题, 正文), ...]。"""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"\n## ", text)
    sections = []
    for p in parts[1:]:
        title_line = p.splitlines()[0] if p.splitlines() else ""
        title = re.sub(r"[#*`|\[\]]", "", title_line).strip()
        if not title or len(title) < 3:
            continue
        body = "\n".join(p.splitlines()[1:]).strip()
        if len(body) < 100:
            continue
        sections.append((title, body))
    return sections


# 口语化中文查询 -> 目标节标题关键词（模拟真实用户提问）
QA = [
    ("长对话聊到几十轮模型会忘前面的内容，怎么压缩成摘要还能找回来", "模块 8"),
    ("同一个请求被重复提交会不会重复执行重复花钱", "模块 9"),
    ("怎么防止模型自己编造引用编号和假出处", "模块 2"),
    ("中文的检索召回率低，有什么办法提升", "模块 6"),
    ("调研跑到一半程序崩了，能不能接着跑", "模块 10"),
    ("用户之前说要中文，现在改口要英文，记忆怎么处理", "模块 7"),
    ("怎么合规地拿到论文全文，不只是摘要", "模块 1"),
    ("怎么判断手里的证据够不够回答用户的问题", "模块 2"),
    ("问、答、写、审四个角色是怎么分工协作的", "模块 2"),
    ("每次改完代码怎么保证系统质量不下降", "模块 4"),
    ("PubMed 文献怎么检索，会不会被限流", "模块 3"),
    ("多用户用同一个系统，记忆会不会互相串", "模块 7"),
    ("调研是自动跑的，成本怎么控制不烧钱", "模块 2"),
    ("用户输入乱七八糟的内容怎么拦截", "第 1 步"),
    ("检索出来的结果怎么排序怎么融合", "模块 6"),
    ("论文切成小块检索，每块多大合适", "模块 6"),
    ("系统怎么判断用户是想闲聊还是要查资料", "第 3 步"),
    ("调研产出的报告能不能复用来回答类似问题", "第 8 步"),
    ("深度调研的并发怎么控制，怕不怕接口限流", "第 7 步"),
    ("怎么判断回答有没有依据，能不能溯源", "第 8 步"),
]


def build_dataset():
    """切块 + 构造用例，返回 (chunks, cases)。"""
    root = Path(__file__).resolve().parents[1]
    sections = load_sections(root / "my-agent-项目全解.md") + load_sections(
        root / "README.md"
    )
    title_to_id = {}
    chunks = []
    for i, (title, body) in enumerate(sections):
        cid = f"sec-{i+1}"
        title_to_id[title] = cid
        chunks.append(
            {
                "chunk_id": cid,
                "title": title[:60],
                "content": body,
                "retrieval_content": f"{title}\n{body[:2000]}",
            }
        )
    cases = []
    for q, marker in QA:
        found = None
        for t in title_to_id:
            if marker in t:
                found = title_to_id[t]
                break
        if found:
            cases.append({"query": q, "relevant": [found]})
    return chunks, cases


def run(index, cases, top_k=5):
    hits, mrrs, ndcgs, lats = [], [], [], []
    for case in cases:
        relevant = set(case["relevant"])
        t0 = time.perf_counter()
        ranked = index.search(case["query"], top_k=top_k)
        lats.append((time.perf_counter() - t0) * 1000)
        ranked_ids = [str(r.get("chunk_id") or "") for r in ranked[:top_k]]
        hit = 1 if any(rid in relevant for rid in ranked_ids) else 0
        mrr = 0.0
        for j, rid in enumerate(ranked_ids, start=1):
            if rid in relevant:
                mrr = 1.0 / j
                break
        ndcg = 0.0
        for j, rid in enumerate(ranked_ids, start=1):
            if rid in relevant:
                ndcg += 1.0 / (j if j <= 2 else j * 0.7)
        hits.append(hit)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
    return {
        "Recall@5": round(statistics.mean(hits), 4),
        "MRR": round(statistics.mean(mrrs), 4),
        "nDCG@5": round(statistics.mean(ndcgs), 4),
        "avg_ms": round(statistics.mean(lats), 2),
    }


def main():
    _install_stubs()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from research_agent.research_retrieval_common import (
        HashEmbeddingProvider,
        ResearchRAGIndex,
    )
    from research_agent.research_retrieval_index import HybridPaperIndex

    chunks, cases = build_dataset()
    print(f"语料: {len(chunks)} 节 | 用例: {len(cases)} 个口语化中文查询")

    provider = HashEmbeddingProvider(dim=64)
    embeds = provider.embed([c["content"] for c in chunks])
    legacy_chunks = [dict(c, embedding=embeds[i]) for i, c in enumerate(chunks)]
    legacy_index = ResearchRAGIndex(chunks=legacy_chunks, embedding_provider=provider)
    new_index = HybridPaperIndex(chunks, embedding_provider=provider)

    legacy = run(legacy_index, cases)
    new = run(new_index, cases)

    print("\n=== 对比结果 ===")
    print(
        f"旧检索栈(legacy): Recall@5={legacy['Recall@5']} MRR={legacy['MRR']} "
        f"nDCG@5={legacy['nDCG@5']} 延迟={legacy['avg_ms']}ms"
    )
    print(
        f"新检索栈(Hybrid)  : Recall@5={new['Recall@5']} MRR={new['MRR']} "
        f"nDCG@5={new['nDCG@5']} 延迟={new['avg_ms']}ms"
    )
    print("\n=== 提升 ===")
    for k in ["Recall@5", "MRR", "nDCG@5"]:
        old, new = legacy[k], new[k]
        if old > 0:
            print(f"{k}: {old} -> {new}  (相对提升 {(new - old) / old * 100:+.2f}%)")
    print(f"延迟变化: Δ{new['avg_ms'] - legacy['avg_ms']:+.2f}ms")


if __name__ == "__main__":
    main()
