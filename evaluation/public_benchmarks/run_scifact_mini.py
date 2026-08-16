"""P2-B2: SciFact 小批量 A/B —— MMR 多样性对 top-k 召回的影响。

动机: 全量 5183 篇 CPU embedding 单次超 10 分钟（环境算力限制），
方案降级为 500 篇子集 + bootstrap 置信区间（统计有效性兜底）。

对比: hybrid(top_k=k) 原始排序 vs hybrid + MMR(λ=0.7) 后重排截断。
MMR 不改变候选集（只重排+截断），其价值体现在 top-k 更小（k=3/5）时
"召回更全"：重复视角被多样性惩罚，给不同相关证据腾位置。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.public_benchmarks.beir_scifact import download_scifact, load_scifact
from evaluation.public_benchmarks.runner import retrieval_metrics
from research_agent.research_diversity import mmr_rerank
from research_agent.research_eval_pipeline import bootstrap_mean_ci
from research_agent.research_retrieval_index import HybridPaperIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="语料子集篇数")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.7)
    parser.add_argument("--output", default="storage/benchmark_runs/scifact-mini-mmr")
    args = parser.parse_args()

    started = time.time()
    print("① 数据集（缓存）…", flush=True)
    dataset_dir = download_scifact(str(ROOT / "storage" / "datasets"))
    dataset = load_scifact(dataset_dir, split="test")
    docs = dataset.documents[: args.limit]
    print("   语料子集 {0}/{1} 篇 | 用例 {2} 条".format(
        len(docs), len(dataset.documents), len(dataset.cases)), flush=True)

    print("② 真实向量 embedding（{0}）…".format(args.model), flush=True)
    from research_agent.research_retrieval_common import (
        SentenceTransformerEmbeddingProvider,
    )
    provider = SentenceTransformerEmbeddingProvider(model_name=args.model)
    chunks = [
        {
            "chunk_id": document.document_id,
            "document_id": document.document_id,
            "title": document.title,
            "content": document.text,
            "retrieval_content": document.text,
            "metadata": dict(document.metadata),
        }
        for document in docs
    ]
    index = HybridPaperIndex(chunks, embedding_provider=provider)

    print("③ 逐用例对比 hybrid vs hybrid+MMR…", flush=True)
    base_metrics, mmr_metrics = [], []
    for case in dataset.cases:
        base = index.search(case.query, mode="hybrid", top_k=args.top_k)
        mmr = mmr_rerank(base, lambda_=args.lambda_, top_k=args.top_k)
        relevant = set(case.relevant_document_ids)
        base_metrics.append(dict(
            retrieval_metrics(
                [item["document_id"] for item in base], relevant, top_k=args.top_k
            ),
            latency_ms=0.0,
        ))
        mmr_metrics.append(dict(
            retrieval_metrics(
                [item["document_id"] for item in mmr], relevant, top_k=args.top_k
            ),
            latency_ms=0.0,
        ))

    def _summarize(rows):
        return {
            "case_count": len(rows),
            "recall": bootstrap_mean_ci([r["recall_at_{0}".format(args.top_k)] for r in rows]),
            "hit": bootstrap_mean_ci([float(r["hit_at_{0}".format(args.top_k)]) for r in rows]),
            "mrr": bootstrap_mean_ci([r["mrr"] for r in rows]),
            "ndcg": bootstrap_mean_ci([r["ndcg_at_{0}".format(args.top_k)] for r in rows]),
        }

    base_summary = _summarize(base_metrics)
    mmr_summary = _summarize(mmr_metrics)

    def _delta(s1, s2):
        return {k: round(s2[k]["mean"] - s1[k]["mean"], 4) for k in ("recall", "hit", "mrr", "ndcg")}

    report = {
        "title": "SciFact mini A/B: hybrid vs hybrid+MMR (top-k={0})".format(args.top_k),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_limit": args.limit,
        "corpus_total": len(dataset.documents),
        "case_count": len(dataset.cases),
        "embedding_model": provider.name,
        "top_k": args.top_k,
        "lambda": args.lambda_,
        "baseline": base_summary,
        "mmr": mmr_summary,
        "delta": _delta(base_summary, mmr_summary),
    }
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def _fmt(metric):
        b = base_summary[metric]["mean"]
        m = mmr_summary[metric]["mean"]
        return "| {0} | {1:.4f} [{2:.4f},{3:.4f}] | {4:.4f} [{5:.4f},{6:.4f}] | {7:+.4f} |".format(
            metric, b, base_summary[metric]["low"], base_summary[metric]["high"],
            m, mmr_summary[metric]["low"], mmr_summary[metric]["high"],
            report["delta"][metric],
        )

    md = [
        "# SciFact mini A/B：MMR 多样性对 top-{0} 召回的影响".format(args.top_k),
        "",
        "- 语料子集 {0}/{1} 篇（CPU 环境全量超时降级，bootstrap CI 兜底统计性）".format(
            args.limit, len(dataset.documents)),
        "- 用例 {0} 条 | embedding: {1} | λ={2}".format(
            len(dataset.cases), provider.name, args.lambda_),
        "- 报告: `{0}/report.json`".format(output_dir),
        "",
        "| 指标 | baseline (hybrid) | MMR λ={0} | Δ |".format(args.lambda_),
        "| --- | --- | --- | --- |",
        _fmt("recall"), _fmt("hit"), _fmt("mrr"), _fmt("ndcg"),
        "",
        "耗时 {0:.0f}s".format(time.time() - started),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
