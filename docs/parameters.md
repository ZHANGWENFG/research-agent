# 参数审计对照表：现状 vs 主流 vs 文献依据

> 生成：2026-08-16 · 目的：让每个关键参数可辩护（面试/评审能引文献），标出差距项供 P2-A2 改造
> 判定标准：✅ 已对齐主流 / 🟡 有依据但有更优空间 / 🔴 与主流有差距（待 P2 处理）

## 检索层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| RRF 融合常数 k | `rank_constant=60` | k=60（生产默认） | Cormack, Clarke & Büttcher, SIGIR'09《Reciprocal Rank Fusion》原文设定 k=60；LanceDB/Weaviate/Elastic 生产默认 k=60。k 越大越平滑，对低频排序噪声鲁棒 | ✅ 已对齐 |
| 检索 top_k（知识库问答） | `top_k=3`（`query_knowledge_base`） | 5–10（BEIR 惯例 recall@5/10） | BEIR 基准（Thakur et al., 2021）以 recall@5/10/100 报告；top_k=3 在 0/1 命中率上容错太薄——单条证据漂移即丢 | 🟡 3 偏保守，P2 建议 ≥5 再压缩 |
| 检索 top_k（运行时基准） | `top_k=5`（`run_retrieval_benchmark`） | 5（hit@5 报告口径） | BEIR 惯例 | ✅ |
| 检索模式 | `hybrid`（BM25+Dense+RRF） | 混合检索为主流默认 | Cormack RRF + 工业混合检索共识 | ✅ |

## 分块层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| chunk_size | 500 字符（字符级切分） | 256–512 token | NVIDIA《Finding the Best Chunking Strategy》(2024)：factoid 查询 256–512 token 最优；arXiv 2505.21700：长文档 64–128 token 更优。500 中文字符 ≈ 350–500 token，落在主流窗口上沿 | 🟡 在窗口内但偏大，长尾段落信息密度低；P2 评估 300/500 两档 |
| chunk_overlap | 50（10%）/ 100（20%）字符 | 10–20%（有争议） | arXiv 2601.14123：overlap 无显著端到端收益；DLR (COINS'24)：512 窗口 + 200 overlap 达最高 IoU≈0.099。现状 10–20% 在主流区间 | 🟡 保留，P2 用 A/B 验证是否可降 |
| 分块单位 | 字符级 | token/句子级 | NVIDIA：语义分块 vs 固定分块因数据集而异 | 🟡 字符级对中文可用（1 字≈1 token），够用 |

## 重排层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| 重排启用门禁（nDCG↑） | `ndcg_at_5` 提升才启用 | 质量门禁模式 | Cross-Encoder rerank（Nogueira & Cho, 2019）常规做 top-k 精排 | ✅ 设计有亮点 |
| 重排延迟预算 | `p95_latency_ms ≤ 500` | 500ms 级（在线 RAG SLO） | 工业 RAG 延迟预算 300–800ms | ✅ |
| 重排召回容忍 | `max_recall_drop=0.02` | 允许 0–5% 召回损失换取 nDCG | rerank 检视实践 | ✅ 参数可辩护 |

## LLM 调用层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| 重试次数 | `max_retries=3` | 3（OpenAI 官方指数退避示例） | OpenAI Cookbook：max 3、base 1s、cap 60s | ✅ |
| 退避基线与封顶 | base=1.0s，cap=30s | base=1s，cap=30–60s | 指数退避 + jitter（AWS 重试白皮书）——jitter 防重试风暴 | ✅ |
| 瞬时错误白名单 | 429/5xx/连接失败/超时 | 同类 | 主流一致 | ✅ |
| 熔断 | 半开探测 + 失败计数（production 层） | Fowler 熔断器 | Fowler, *CircuitBreaker*（martinfowler.com） | ✅ |

## 数据获取层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| PubMed 节流 | `throttle=0.4s`（2.5 req/s） | 免费接口 3 req/s（无 key） | NCBI E-utilities 官方限流：无 API key 3 req/s | ✅ 0.4s 略保守于 0.33s，安全 |
| 下载大小上限 | 20MB | 10–50MB 视场景 | 全文下载常规上限 | ✅ |
| 白名单域名 | PMC/europepmc/arxiv | 权威 OA 源 | 合规获取设计 | ✅ 有亮点 |

## 系统层

| 参数 | 现状值 | 主流值/范围 | 文献依据 | 判定 |
|---|---|---|---|---|
| 并发任务上限 | `max_concurrent_tasks=1` | 按资源配额（CPU 推理场景 1–4） | 本地推理无 GPU 时 1 最稳 | 🟡 保守但安全，留参数可调 |
| 索引 LRU 容量 | env `RESEARCH_RETRIEVAL_INDEX_CACHE_SIZE=16` | 进程内缓存 10–32 | 缓存容量经验值 | ✅ |
| stale 任务恢复阈值 | 调用方传参 | 分钟级（2–10min） | 分布式任务恢复惯例 | ✅ |

## 缺失项（P2-A2 新增，均有文献依据）

| 能力 | 主流方法 | 依据 | 计划值 |
|---|---|---|---|
| 查询改写 | HyDE/LLM query expansion | Gao et al., 2022《Precise Zero-Shot Dense Retrieval without Relevance Labels》(HyDE)；工业 RAG 查询改写提升 recall | P2 接入意图路由后 LLM 改写，A/B 验证 |
| 多样性去重 | **MMR**（Maximal Marginal Relevance） | **Carbonell & Goldstein, SIGIR'98**：MMR 公式 λ·sim(q,d) − (1−λ)·max sim(d,已选)，原文建议 λ 0.5–0.8 | λ=0.7（主流默认），top-10 内去重 |
| 证据压缩 | LongLLMLingua/上下文压缩 | Jiang et al., 2023《LLMLingua》；压缩后上下文减 50%+ 不掉精度 | 首轮：top-k 证据按相关度截断排序 |

## 汇总

- ✅ 完全对齐主流且有文献：RRF k=60、重试退避、熔断、节流、延迟门禁、白名单下载
- 🟡 有依据但可再论证：top_k=3 偏小、chunk=500 偏大、并发=1 保守
- 🔴 能力缺失（P2 补）：查询改写、MMR 多样性、证据压缩——三项都是主流 RAG 流水线标准件
