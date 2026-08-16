"""research_retrieval_common —— 检索相关的「共享零件库」。

这里集中放被多个模块共用、且不依赖具体业务逻辑的检索基础零件：
- 文本切块 / 分词 / 相似度等工具函数
- 本地哈希向量（HashEmbeddingProvider）等嵌入 Provider
- 本地 RAG 索引（ResearchRAGIndex）
- 上下文压缩检索器（ContextCompressionRetriever）
- 跨会话长期记忆索引（ResearchLongTermMemoryIndex）

设计约束（防止循环引用）：
- 本文件只允许 import 标准库 / 第三方库 / `research_memory`；
- 严禁 `from .research_rag import ...` 或 `from .research_retrieval_index import ...`，
  否则会与旧检索机、新检索机互相引用，启动即报循环导入错误。
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .research_memory import compress_context

# ---------------------------------------------------------------------------
# 调研产物文件名（自研命名；读取端兼容历史 STORM 命名的产物目录，保证旧评测背书可复现）
# ---------------------------------------------------------------------------
ARTICLE_FILENAME = "myagent_article_polished.txt"
OUTLINE_FILENAME = "myagent_outline.txt"
_LEGACY_ARTICLE_FILENAMES = ("storm_gen_article_polished.txt", "storm_gen_article.txt")
_LEGACY_OUTLINE_FILENAME = "storm_gen_outline.txt"


def resolve_article_path(run_dir) -> Path:
    """新命名优先；历史产物目录（storm_gen_*）回退兜底。"""
    run_dir = Path(run_dir)
    for name in (ARTICLE_FILENAME,) + _LEGACY_ARTICLE_FILENAMES:
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / ARTICLE_FILENAME


def resolve_outline_path(run_dir) -> Path:
    run_dir = Path(run_dir)
    for name in (OUTLINE_FILENAME, _LEGACY_OUTLINE_FILENAME):
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / OUTLINE_FILENAME


class HashEmbeddingProvider:
    """Deterministic local embedding baseline."""

    name = "hash_embedding"

    def __init__(self, dim: int = 64):
        self.dim = int(dim or 64)

    def embed(self, texts: Iterable[str]):
        return [hash_embedding(text, dim=self.dim) for text in texts]

    def embed_query(self, text: str):
        return self.embed([text])[0]


class CallableEmbeddingProvider:
    """Adapter for real embedding backends without forcing a dependency."""

    def __init__(self, name: str, dim: int, embed_fn: Callable[[List[str]], List[List[float]]]):
        self.name = name
        self.dim = int(dim)
        self.embed_fn = embed_fn

    def embed(self, texts: Iterable[str]):
        vectors = self.embed_fn([str(text or "") for text in texts])
        return [_normalize_vector(vector, self.dim) for vector in vectors]

    def embed_query(self, text: str):
        return self.embed([text])[0]


class SentenceTransformerEmbeddingProvider:
    """Optional sentence-transformers provider.

    This is deliberately lazy-imported so normal tests and fake demos do not
    require installing sentence-transformers.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer

        self.name = "sentence_transformers:{0}".format(model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Iterable[str]):
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str):
        return self.embed([text])[0]


def build_embedding_provider(provider: Optional[str] = None, embedding_dim: int = 64):
    provider = (provider or os.getenv("RESEARCH_EMBEDDING_PROVIDER") or "hash").lower()
    if provider in {"hash", "local", "baseline"}:
        return HashEmbeddingProvider(dim=embedding_dim)
    if provider in {"sentence-transformers", "sentence_transformers", "bge", "bge-m3"}:
        model = os.getenv("RESEARCH_EMBEDDING_MODEL") or "BAAI/bge-m3"
        return SentenceTransformerEmbeddingProvider(model_name=model)
    raise ValueError("Unsupported embedding provider: {0}".format(provider))


class ResearchRAGIndex:
    """Local RAG index with lexical + hash-embedding hybrid retrieval."""

    def __init__(
        self,
        chunks: Optional[List[Dict]] = None,
        embedding_dim: int = 64,
        config: Optional[Dict] = None,
        embedding_provider=None,
    ):
        self.chunks = chunks or []
        self.embedding_dim = int(embedding_dim or 64)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(self.embedding_dim)
        self.config = config or {
            "index_type": "local_json_hash_embedding",
            "ann": "linear_scan_baseline",
            "hnsw_ready_params": {"M": 16, "ef_construct": 100, "ef_search": 64},
            "embedding_provider": self.embedding_provider.name,
        }

    @classmethod
    def from_run_dir(
        cls,
        run_dir,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_dim: int = 64,
        embedding_provider=None,
    ):
        run_dir = Path(run_dir)
        documents = []
        article_path = resolve_article_path(run_dir)
        article = (
            article_path.read_text(encoding="utf-8", errors="replace")
            if article_path.exists()
            else ""
        )
        if article:
            documents.append(
                {
                    "document_id": "generated_article",
                    "title": "Generated Research Article",
                    "text": article,
                    "source_type": "article",
                    "url": str(article_path),
                }
            )
        for index, result in enumerate(_read_json(run_dir / "raw_search_results.json", []), start=1):
            snippets = result.get("snippets") or []
            text = "\n".join(
                [
                    str(result.get("title") or ""),
                    str(result.get("description") or ""),
                    "\n".join(str(item) for item in snippets),
                ]
            ).strip()
            if text:
                documents.append(
                    {
                        "document_id": "retrieval-{0}".format(index),
                        "title": result.get("title") or "Retrieved source {0}".format(index),
                        "text": text,
                        "source_type": result.get("source_type") or "retrieval",
                        "url": result.get("url") or "",
                        "metadata": {
                            "result_index": index,
                            "query": result.get("query", ""),
                        },
                    }
                )
        return cls.from_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=embedding_dim,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Dict],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_dim: int = 64,
        embedding_provider=None,
    ):
        chunks = []
        embedding_provider = embedding_provider or HashEmbeddingProvider(embedding_dim)
        embedding_dim = int(getattr(embedding_provider, "dim", embedding_dim) or embedding_dim)
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        for doc_index, document in enumerate(documents or [], start=1):
            text = str(document.get("text") or "")
            document_id = document.get("document_id") or "doc-{0}".format(doc_index)
            for chunk_index, content in enumerate(
                chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                start=1,
            ):
                chunk_id = "{0}-chunk-{1}".format(document_id, chunk_index)
                embedding = embedding_provider.embed([content])[0]
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "title": document.get("title") or document_id,
                        "content": content,
                        "url": document.get("url") or "",
                        "source_type": document.get("source_type") or "document",
                        "metadata": dict(document.get("metadata") or {}, chunk_index=chunk_index),
                        "embedding": embedding,
                        "token_count_estimate": estimate_tokens(content),
                    }
                )
        return cls(
            chunks=chunks,
            embedding_dim=embedding_dim,
            config={
                "index_type": "local_json_hash_embedding",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_provider": embedding_provider.name,
                "ann": "linear_scan_baseline",
                "hnsw_ready_params": {"M": 16, "ef_construct": 100, "ef_search": 64},
            },
            embedding_provider=embedding_provider,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "embedding_dim": self.embedding_dim,
                    "config": self.config,
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path, embedding_provider=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        provider = embedding_provider or HashEmbeddingProvider(data.get("embedding_dim") or 64)
        return cls(
            chunks=data.get("chunks") or [],
            embedding_dim=data.get("embedding_dim") or 64,
            config=data.get("config") or {},
            embedding_provider=provider,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.65,
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        rerank: bool = True,
    ):
        query_embedding = self.embedding_provider.embed_query(query)
        query_terms = tokenize(query)
        expected_keywords = expected_keywords or []
        forbidden_keywords = forbidden_keywords or []
        scored = []
        for index, chunk in enumerate(self.chunks):
            text = "{0}\n{1}".format(chunk.get("title", ""), chunk.get("content", ""))
            lexical = lexical_score(query_terms, tokenize(text))
            vector = cosine_similarity(query_embedding, chunk.get("embedding") or [])
            hybrid = alpha * vector + (1.0 - alpha) * lexical
            rerank_score = hybrid
            expected_hits = keyword_hits(text, expected_keywords)
            forbidden_hits = keyword_hits(text, forbidden_keywords)
            if rerank:
                rerank_score += min(0.25, 0.08 * len(expected_hits))
                rerank_score -= min(0.35, 0.12 * len(forbidden_hits))
            enriched = dict(chunk)
            enriched.update(
                {
                    "lexical_score": round(lexical, 6),
                    "vector_score": round(vector, 6),
                    "hybrid_score": round(hybrid, 6),
                    "rerank_score": round(rerank_score, 6),
                    "expected_keyword_hits": expected_hits,
                    "forbidden_keyword_hits": forbidden_hits,
                    "score": round(rerank_score, 6),
                }
            )
            scored.append((rerank_score, index, enriched))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [item for score, _, item in scored[:top_k] if score > 0]


class ContextCompressionRetriever:
    """Wrapper between retriever/index and prompt assembly."""

    def __init__(
        self,
        index: ResearchRAGIndex,
        max_context_chars: int = 2400,
        history_ratio: float = 0.3,
        evidence_ratio: float = 0.7,
        candidate_multiplier: int = 4,
        llm_compressor: Optional[Callable[[Dict], str]] = None,
    ):
        self.index = index
        self.max_context_chars = int(max_context_chars)
        self.history_ratio = float(history_ratio)
        self.evidence_ratio = float(evidence_ratio)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.llm_compressor = llm_compressor

    def retrieve(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 5,
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
    ):
        expected_keywords = expected_keywords or []
        forbidden_keywords = forbidden_keywords or []
        candidates = self.index.search(
            query,
            top_k=max(top_k, top_k * self.candidate_multiplier),
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
        )
        filtered = [
            chunk
            for chunk in candidates
            if not chunk.get("forbidden_keyword_hits") or chunk.get("expected_keyword_hits")
        ]
        if not filtered:
            filtered = candidates
        selected = filtered[:top_k]
        history_budget = int(self.max_context_chars * self.history_ratio)
        evidence_budget = self.max_context_chars - history_budget
        compressed_history = compress_context(
            history or [],
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            max_chars=max(120, history_budget),
        )
        compression_mode = "coarse_filter_then_rule_sentence_extract"
        if self.llm_compressor:
            compressed_evidence = _compress_with_llm(
                self.llm_compressor,
                selected,
                query=query,
                max_chars=max(120, evidence_budget),
                expected_keywords=expected_keywords,
                forbidden_keywords=forbidden_keywords,
            )
            compression_mode = "llm_context_compressor"
        else:
            compressed_evidence = _compress_chunks(
                selected,
                query=query,
                max_chars=max(120, evidence_budget),
                expected_keywords=expected_keywords,
            )
        prompt_context = _trim_join(
            [
                compressed_history.get("summary", ""),
                compressed_evidence,
            ],
            max_chars=self.max_context_chars,
        )
        return {
            "query": query,
            "chunks": selected,
            "compressed_history": compressed_history,
            "compressed_evidence": compressed_evidence,
            "prompt_context": prompt_context,
            "budget": {
                "max_context_chars": self.max_context_chars,
                "history_ratio": self.history_ratio,
                "evidence_ratio": self.evidence_ratio,
                "history_chars": len(compressed_history.get("summary", "")),
                "evidence_chars": len(compressed_evidence),
            },
            "audit": {
                "candidate_count": len(candidates),
                "coarse_filtered_count": len(candidates) - len(filtered),
                "selected_count": len(selected),
                "compression": compression_mode,
            },
        }


class ResearchLongTermMemoryIndex:
    """Persistent local vector-like memory index for cross-session recall."""

    def __init__(self, records: Optional[List[Dict]] = None, embedding_dim: int = 64):
        self.records = records or []
        self.embedding_dim = int(embedding_dim)

    @classmethod
    def from_memory_store(cls, memory_store, embedding_dim: int = 64):
        records = []
        data = memory_store.to_dict()
        for kind in ("semantic", "episodic", "working"):
            for item in data.get(kind, []):
                record = dict(item)
                record["kind"] = kind
                record["embedding"] = hash_embedding(record.get("content", ""), dim=embedding_dim)
                records.append(record)
        if data.get("preferences"):
            records.append(
                {
                    "kind": "preferences",
                    "content": json.dumps(data["preferences"], ensure_ascii=False),
                    "metadata": {},
                    "embedding": hash_embedding(
                        json.dumps(data["preferences"], ensure_ascii=False),
                        dim=embedding_dim,
                    ),
                }
            )
        return cls(records=records, embedding_dim=embedding_dim)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"embedding_dim": self.embedding_dim, "records": self.records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            records=data.get("records") or [],
            embedding_dim=data.get("embedding_dim") or 64,
        )

    def recall(self, query: str, top_k: int = 5):
        query_embedding = hash_embedding(query, dim=self.embedding_dim)
        query_terms = tokenize(query)
        scored = []
        for index, record in enumerate(self.records):
            content = record.get("content", "")
            score = 0.7 * cosine_similarity(query_embedding, record.get("embedding") or [])
            score += 0.3 * lexical_score(query_terms, tokenize(content))
            enriched = dict(record)
            enriched["score"] = round(score, 6)
            scored.append((score, index, enriched))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [item for score, _, item in scored[:top_k] if score > 0]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """按文档结构切分文本（主流 RAG 做法: chunking 是产品决策，尊重段落/句子边界）。

    与旧版"纯字符窗口"的关键区别:
      1. 段落优先 —— 文本先按换行分块，大段（超过 chunk_size）才继续切，
         小段（不足 chunk_size）与相邻段合并，保留语义单元；
      2. 句子边界对齐 —— 必须切分时，在句末标点（。.!?！？）附近找断点，
         不让英文单词被拦腰截断、不让中文句子被从中间切开；
      3. 保留段落结构 —— 不再先把空白全部塌缩成单行。

    参数语义与旧版一致: chunk_size 为目标长度（字符），chunk_overlap 为相邻
    chunk 的重叠字符数（在超长块内部滑动时生效）。
    """
    text = str(text or "")
    if not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # 第 1 步: 按空行/换行切段落，再按目标长度做结构化合并
    paragraphs = _split_paragraphs(text)
    units: List[str] = []
    current = ""
    for para in paragraphs:
        if len(para) >= chunk_size:
            # 超长段落独立处理: 先把已累积的 unit 落盘
            if current:
                units.append(current.strip())
                current = ""
            units.extend(_split_long_unit(para, chunk_size, chunk_overlap))
        elif len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip() if current else para
        else:
            if current:
                units.append(current.strip())
            current = para
    if current:
        units.append(current.strip())

    # 第 2 步: 单段仍超长时，在句子边界处回溯切分（不硬切字符）
    final: List[str] = []
    for unit in units:
        if len(unit) <= chunk_size:
            final.append(unit)
        else:
            final.extend(_split_long_unit(unit, chunk_size, chunk_overlap))
    return [c for c in final if c.strip()]


# 段落按空行或换行切分（保留结构，不塌缩空白）
_PARAGRAPH_RE = re.compile(r"\n\s*\n|\n+")


def _split_paragraphs(text: str) -> List[str]:
    parts = [p for p in _PARAGRAPH_RE.split(text) if p.strip()]
    return parts or [text.strip()]


# 句末标点: 中英文句号/感叹/问号（主流切分点）
_SENTENCE_END_RE = re.compile(r"[。.!?！？；;]")
# 避免在句末标点后紧跟数字/缩写处切断（如 "3.14"、"U.S."）
_NO_SPLIT_AFTER = re.compile(r"(?<=\.)(?=\d)|(?<=\.[A-Za-z])(?=[A-Za-z])")


def _split_long_unit(unit: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """超长单元在句子边界处切分，窗口内尽量以句末标点收尾。"""
    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
    length = len(unit)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:  # 不是最后一块 → 尝试回溯到句末标点
            boundary = _sentence_boundary_before(unit, start, end)
            if boundary is not None:
                end = boundary
        chunk = unit[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)  # 保证前进
    return chunks


def _sentence_boundary_before(text: str, start: int, end: int) -> Optional[int]:
    """在 [start, end] 内找最后一个句末标点后的断点；找不到返回 None（硬切）。"""
    # 从 end 往前找，但至少保留 chunk_size 的 60% 长度，避免切出极短块
    min_boundary = start + int((end - start) * 0.6)
    best = None
    for m in _SENTENCE_END_RE.finditer(text, start, end):
        pos = m.end()
        if pos < min_boundary:
            continue
        if _NO_SPLIT_AFTER.search(text, pos - 2, pos + 2):
            continue
        best = pos
    return best


def hash_embedding(text: str, dim: int = 64):
    vector = [0.0] * int(dim)
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def _normalize_vector(vector: Iterable[float], dim: int):
    values = [float(value) for value in vector]
    if len(values) < dim:
        values = values + [0.0] * (dim - len(values))
    if len(values) > dim:
        values = values[:dim]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    # 修复: 必须迭代补齐/截断后的 values，而不是原始 vector——
    # 否则短向量的 padding、长向量的截断全部丢失，维度契约被破坏
    return [round(value / norm, 8) for value in values]


def cosine_similarity(left: List[float], right: List[float]):
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    return max(0.0, sum(left[i] * right[i] for i in range(length)))


def lexical_score(query_terms, text_terms):
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


def tokenize(text: str):
    """统一分词（2026-08-16 收敛）: 与 multilingual_tokenize 语义对齐——
    Latin 词 + CJK unigram/bigram，CJK 范围统一 \u4e00-\u9fff（基本汉字区）。
    返回 set（去重，兼容旧调用方 hash_embedding 的遍历语义）。
    """
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(sequence)  # unigram（逐字符，与 multilingual_tokenize 一致）
        tokens.extend(
            sequence[index : index + 2] for index in range(len(sequence) - 1)
        )
    return set(tokens)


def keyword_hits(text: str, keywords: Iterable[str]):
    lowered = str(text or "").lower()
    return [keyword for keyword in keywords or [] if keyword and keyword.lower() in lowered]


def estimate_tokens(text: str):
    return max(1, int(len(str(text or "")) / 4))


def _compress_chunks(chunks: List[Dict], query: str, max_chars: int, expected_keywords: List[str]):
    query_terms = tokenize(query)
    lines = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[。.!?])\s+", chunk.get("content", ""))
        kept = []
        for sentence in sentences:
            if not sentence:
                continue
            if tokenize(sentence) & query_terms or keyword_hits(sentence, expected_keywords):
                kept.append(sentence)
        if not kept:
            kept = [chunk.get("content", "")[:220]]
        lines.append(
            "[{0}] {1}".format(chunk.get("chunk_id", ""), " ".join(kept).strip())
        )
    return _trim_join(lines, max_chars=max_chars)


def _compress_with_llm(
    compressor: Callable[[Dict], str],
    chunks: List[Dict],
    query: str,
    max_chars: int,
    expected_keywords: List[str],
    forbidden_keywords: List[str],
):
    payload = {
        "query": query,
        "max_chars": max_chars,
        "expected_keywords": expected_keywords,
        "forbidden_keywords": forbidden_keywords,
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "title": chunk.get("title", ""),
                "content": chunk.get("content", ""),
                "score": chunk.get("score", 0),
            }
            for chunk in chunks
        ],
    }
    try:
        compressed = str(compressor(payload) or "").strip()
    except Exception:
        compressed = ""
    if not compressed:
        compressed = _compress_chunks(
            chunks,
            query=query,
            max_chars=max_chars,
            expected_keywords=expected_keywords,
        )
    return _trim_join([compressed], max_chars=max_chars)


def _trim_join(parts: List[str], max_chars: int):
    text = "\n".join(part for part in parts if part)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _read_first_existing(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path).read_text(encoding="utf-8", errors="replace")
    return ""


def _read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
