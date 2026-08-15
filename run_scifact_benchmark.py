# -*- coding: utf-8 -*-
"""SciFact 公开基准评测入口：下载 BEIR 官方数据 → 跑检索 → 落产物到 storage/benchmark_runs/。

用法：
    python run_scifact_benchmark.py          # hash 嵌入（离线 smoke，~1 分钟）
    python run_scifact_benchmark.py --real   # sentence-transformers 真实向量（需下载模型，慢）
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.public_benchmarks.beir_scifact import download_scifact, load_scifact
from evaluation.public_benchmarks.runner import (
    HashEmbeddingProvider,
    run_retrieval_benchmark,
)
from evaluation.public_benchmarks.report import write_benchmark_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="用真实向量（sentence-transformers）")
    parser.add_argument("--output", default=str(ROOT / "storage" / "benchmark_runs" / "scifact"))
    parser.add_argument("--cache", default=str(ROOT / "storage" / "datasets"))
    args = parser.parse_args()

    started = time.time()
    print("① 下载/校验 BEIR SciFact 数据集…")
    dataset_dir = download_scifact(args.cache)
    print("   数据集就绪:", dataset_dir)

    print("② 加载 test split…")
    dataset = load_scifact(dataset_dir, split="test")
    print(f"   语料 {len(dataset.documents)} 篇 | 用例 {len(dataset.cases)} 个")

    if args.real:
        from research_agent.research_retrieval_common import (
            sentence_transformers_embedding_provider,
        )
        provider = sentence_transformers_embedding_provider()
    else:
        provider = HashEmbeddingProvider(dim=128)
    print(f"③ 嵌入后端: {provider.name}")

    print("④ 跑检索基准（bm25 / dense / hybrid）…")
    report = run_retrieval_benchmark(
        dataset,
        provider,
        modes=("bm25", "dense", "hybrid"),
        top_k=10,
        seed=55,
    )
    manifest = report["manifest"]
    manifest["embedding_backend"] = provider.name
    manifest["real_vectors"] = args.real

    output_dir = Path(args.output)
    write_benchmark_artifacts(output_dir, manifest, report, [], [])
    elapsed = time.time() - started
    print(f"⑤ 产物已写入: {output_dir}")
    print(f"   耗时 {elapsed:.1f}s")
    print("\n=== 指标摘要 ===")
    for mode, metrics in report["modes"].items():
        print(
            f"  [{mode}] Recall@10={metrics.get('recall_at_10')} "
            f"MRR@10={metrics.get('mrr_at_10')} nDCG@10={metrics.get('ndcg_at_10')}"
        )


if __name__ == "__main__":
    main()
