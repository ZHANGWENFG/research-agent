import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .research_retrieval_common import resolve_article_path, resolve_outline_path


@dataclass
class EvalCase:
    topic: str
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    expected_language: str = "original"
    min_sources: int = 1

    @classmethod
    def from_dict(cls, data):
        return cls(
            topic=data["topic"],
            expected_keywords=data.get("expected_keywords") or [],
            forbidden_keywords=data.get("forbidden_keywords") or [],
            expected_language=data.get("expected_language", "original"),
            min_sources=int(data.get("min_sources", 1)),
        )


def load_eval_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase.from_dict(item) for item in data]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _iter_result_texts(raw_results) -> Iterable[str]:
    for result in raw_results or []:
        if isinstance(result, dict):
            yield str(result.get("title") or "")
            yield str(result.get("description") or "")
            for snippet in result.get("snippets") or []:
                yield str(snippet)
        else:
            yield str(result)


def _count_offtopic_results(raw_results, forbidden_keywords: Iterable[str]) -> int:
    count = 0
    for result in raw_results or []:
        text = "\n".join(_iter_result_texts([result]))
        if _count_keyword_hits(text, forbidden_keywords):
            count += 1
    return count


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if keyword and keyword.lower() in lowered:
            hits.append(keyword)
    return hits


def _chinese_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    visible_count = len(re.findall(r"\S", text))
    if visible_count == 0:
        return 0.0
    return chinese_count / visible_count


def _load_trace_events(path: Path) -> List[Dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "trace_decode_error", "raw": line})
    return events


def _score_completion(checks: Dict[str, bool]) -> float:
    points = 0.0
    points += 8.0 if checks["has_article"] else 0.0
    points += 5.0 if checks["has_outline"] else 0.0
    points += 4.0 if checks["has_search_results"] else 0.0
    points += 3.0 if checks["run_success"] else 0.0
    return points


# ------------------------------------------------------------------
# Faithfulness（忠实度）—— RAGAS 四维指标的核心维度
# ------------------------------------------------------------------
# 主流定义（RAGAS）: 回答中的每个论断 (claim) 都应能被检索上下文 (context)
# 支持；忠实度 = 可被支持的论断数 / 论断总数。
# 项目里没有 LLM 论断抽取器，因此用确定性近似:
#   1. 把文章按句子切分为"论断"
#   2. 每个论断若与任一检索结果（标题/摘要/snippet）有实质关键词重叠，
#      视为"有证据支持"
#   3. faithfulness = 有支持的论断 / 总论断
# 这是无 LLM 依赖的保守近似，避免评测本身引入模型成本与不确定性；
# 若未来接入 RAGAS 可用 LLM 论断抽取替换，接口保持一致。
# ------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.!?！？])\s*|\n+")


def _split_claims(article: str) -> List[str]:
    """把文章切成论断（句子级），过滤空串和过短片段。"""
    claims = []
    for chunk in _SENTENCE_SPLIT_RE.split(article):
        chunk = chunk.strip()
        # 过滤太短的片段（如 "3."、"参考文献"这类非论断）
        if len(chunk) >= 10:
            claims.append(chunk)
    return claims


def _evidence_overlap(claim: str, retrieval_text: str) -> bool:
    """判断论断与检索证据是否有实质重叠。

    主流做法用 LLM 判断"论断能否从上下文推出"；这里用确定性近似:
    论断中 2+ 个实词（英文词或中文 bigram）出现在证据文本里即视为有支持。
    保守近似宁可漏判（faithfulness 偏低），不虚高。
    """
    claim_terms = _meaningful_terms(claim)
    if not claim_terms:
        return False
    evidence = retrieval_text.lower()
    hits = [t for t in claim_terms if t in evidence]
    # 至少 2 个实词命中（同一 bigram 的重复命中只算一次）
    return len(set(hits)) >= 2


_TERM_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "at", "for", "and", "or", "but", "with", "as", "by", "from",
    "this", "that", "it", "its", "we", "they", "our", "their", "研究", "我们",
    "本文", "该", "在", "对", "与", "了", "的", "和", "是", "有", "而", "中",
}


def _meaningful_terms(text: str) -> List[str]:
    """提取论断中的实词（英文单词含 2 字符缩写 + 中文连续 2 字符 bigram），去停用词。

    中文 bigram 方案（主流检索做法）: "医疗领域" → ["医疗", "疗领", "领域"]，
    其中实词 bigram 能在证据文本中精确匹配，比 4 字符窗口鲁棒。
    """
    terms = []
    for m in re.finditer(r"[a-zA-Z][a-zA-Z\-]{1,}|[\u4e00-\u9fff]+", text.lower()):
        token = m.group(0)
        if token[0].isascii():  # 英文词
            if token not in _TERM_STOPWORDS:
                terms.append(token)
        else:  # 中文串 → 连续 2 字符 bigram
            for i in range(len(token) - 1):
                bigram = token[i : i + 2]
                if bigram not in _TERM_STOPWORDS:
                    terms.append(bigram)
    return terms


def _score_faithfulness(article: str, retrieval_text: str) -> Tuple[float, float, float]:
    """计算忠实度得分。

    返回 (得分0-10, 有支持的论断数, 论断总数)。文章为空时得 0。
    """
    claims = _split_claims(article)
    if not claims:
        return 0.0, 0.0, 0.0
    supported = sum(1 for c in claims if _evidence_overlap(c, retrieval_text))
    ratio = supported / len(claims)
    return round(10.0 * ratio, 2), supported, len(claims)


def _score_retrieval(expected_hits: List[str], expected_total: int, source_count: int, min_sources: int) -> float:
    if expected_total == 0:
        keyword_score = 15.0
    else:
        keyword_score = 20.0 * min(1.0, len(expected_hits) / expected_total)
    source_score = 10.0 * min(1.0, source_count / max(1, min_sources))
    return keyword_score + source_score


def _score_article(article: str, expected_hits: List[str], expected_total: int, chinese_ratio: float, expected_language: str) -> float:
    length_score = 6.0 if len(article.strip()) >= 80 else (3.0 if article.strip() else 0.0)
    if expected_total == 0:
        coverage_score = 7.0
    else:
        coverage_score = 7.0 * min(1.0, len(expected_hits) / expected_total)
    language_score = 7.0
    if expected_language == "zh":
        language_score = 7.0 * min(1.0, chinese_ratio / 0.35)
    return length_score + coverage_score + language_score


def _score_trace(checks: Dict[str, bool], trace_events: List[Dict]) -> float:
    event_names = {event.get("event") for event in trace_events}
    points = 0.0
    points += 4.0 if checks["has_trace"] else 0.0
    points += 3.0 if "run_start" in event_names else 0.0
    points += 3.0 if "run_end" in event_names else 0.0
    points += 3.0 if "retrieval_start" in event_names or "tool_start" in event_names else 0.0
    points += 2.0 if "retrieval_end" in event_names or "tool_end" in event_names else 0.0
    return points


def evaluate_run(run_dir, case: EvalCase):
    run_dir = Path(run_dir)
    raw_results = _read_json(run_dir / "raw_search_results.json", [])
    summary = _read_json(run_dir / "run_summary.json", {})
    outline = _read_text(resolve_outline_path(run_dir))
    article = _read_text(resolve_article_path(run_dir))
    trace_events = _load_trace_events(run_dir / "research_trace.jsonl")

    retrieval_text = "\n".join(_iter_result_texts(raw_results))
    # 检索维度只评估检索结果本身（主流做法: 检索与生成分开测，
    # 避免"文章写得好掩盖检索退化"——H2 修复）
    retrieval_expected_hits = _count_keyword_hits(retrieval_text, case.expected_keywords)
    article_and_retrieval = article + "\n" + retrieval_text
    expected_hits = _count_keyword_hits(article_and_retrieval, case.expected_keywords)
    forbidden_hits = _count_keyword_hits(article_and_retrieval, case.forbidden_keywords)
    offtopic_result_count = _count_offtopic_results(raw_results, case.forbidden_keywords)
    chinese_ratio = _chinese_char_ratio(article)
    source_count = len(raw_results) if isinstance(raw_results, list) else 0

    # Faithfulness（RAGAS 核心维度）: 文章的论断有多少能被检索证据支持
    faithfulness, supported_claims, total_claims = _score_faithfulness(
        article, retrieval_text
    )

    checks = {
        "has_article": bool(article.strip()),
        "has_outline": bool(outline.strip()),
        "has_search_results": source_count > 0,
        "has_trace": len(trace_events) > 0,
        "run_success": bool(summary.get("success")) or any(
            event.get("event") == "run_end" and event.get("success")
            for event in trace_events
        ),
    }

    completion = _score_completion(checks)
    # 检索维度用"只来自检索结果"的命中（H2 修复）
    retrieval = _score_retrieval(
        expected_hits=retrieval_expected_hits,
        expected_total=len(case.expected_keywords),
        source_count=source_count,
        min_sources=case.min_sources,
    )
    offtopic_penalty = 15.0 * min(1.0, offtopic_result_count / max(1, source_count))
    article_score = _score_article(
        article=article,
        expected_hits=expected_hits,
        expected_total=len(case.expected_keywords),
        chinese_ratio=chinese_ratio,
        expected_language=case.expected_language,
    )
    trace_score = _score_trace(checks, trace_events)
    # 总分: 各维度权重上限合计 95（completion 20 + retrieval 30 + article 20 + trace 15
    # + faithfulness 10），归一化到 100 分制——"满分 100"不再有死代码误导
    MAX_TOTAL = 95.0
    total = completion + retrieval + article_score + trace_score + faithfulness - offtopic_penalty
    total = max(0.0, min(MAX_TOTAL, total)) / MAX_TOTAL * 100.0

    notes = _build_notes(checks, forbidden_hits, expected_hits, case, source_count)
    return {
        "topic": case.topic,
        "scores": {
            "total": round(total, 2),
            "task_completion": round(completion, 2),
            "retrieval_quality": round(retrieval, 2),
            "offtopic_penalty": round(offtopic_penalty, 2),
            "article_quality": round(article_score, 2),
            "runtime_observability": round(trace_score, 2),
            "faithfulness": faithfulness,
        },
        "metrics": {
            "source_count": source_count,
            "expected_hits": expected_hits,
            "forbidden_hits": forbidden_hits,
            "offtopic_result_count": offtopic_result_count,
            "chinese_char_ratio": round(chinese_ratio, 4),
            "trace_event_count": len(trace_events),
            "supported_claims": supported_claims,
            "total_claims": total_claims,
        },
        "checks": checks,
        "notes": notes,
    }


def evaluate_qa_artifact(run_dir, case: EvalCase):
    run_dir = Path(run_dir)
    qa = _read_json(run_dir / "qa_answer.json", {})
    answer = str(qa.get("answer") or "")
    citations = qa.get("citations") or []
    grounded = bool(qa.get("grounded"))
    expected_hits = _count_keyword_hits(answer, case.expected_keywords)
    forbidden_hits = _count_keyword_hits(answer, case.forbidden_keywords)
    chinese_ratio = _chinese_char_ratio(answer)

    keyword_score = 12.0
    if case.expected_keywords:
        keyword_score = 12.0 * min(1.0, len(expected_hits) / len(case.expected_keywords))
    citation_score = 8.0 if citations else 0.0
    grounded_score = 6.0 if grounded else 0.0
    language_score = 4.0
    if case.expected_language == "zh":
        language_score = 4.0 * min(1.0, chinese_ratio / 0.25)
    forbidden_penalty = 10.0 if forbidden_hits else 0.0
    qa_quality = keyword_score + citation_score + grounded_score + language_score
    # QA 维度权重上限 30（12+8+6+4），归一化到 100 分制（同 evaluate_run 的 95→100 修正）
    QA_MAX_TOTAL = 30.0
    total = max(0.0, min(QA_MAX_TOTAL, qa_quality - forbidden_penalty)) / QA_MAX_TOTAL * 100.0

    checks = {
        "qa_exists": bool(qa),
        "qa_has_answer": bool(answer.strip()),
        "qa_has_citation": bool(citations),
        "qa_grounded": grounded,
    }
    notes = []
    if not checks["qa_exists"]:
        notes.append("缺少 qa_answer.json。")
    if not checks["qa_has_citation"]:
        notes.append("问答缺少引用，不能证明答案来自知识库证据。")
    if forbidden_hits:
        notes.append("问答中出现跑题关键词：" + ", ".join(forbidden_hits))
    missing = [item for item in case.expected_keywords if item not in expected_hits]
    if missing:
        notes.append("问答未覆盖期望关键词：" + ", ".join(missing))
    if not notes:
        notes.append("问答结果满足当前知识库 QA 评估规则。")

    return {
        "topic": case.topic,
        "scores": {
            "total": round(total, 2),
            "qa_quality": round(qa_quality, 2),
            "forbidden_penalty": round(forbidden_penalty, 2),
        },
        "metrics": {
            "expected_hits": expected_hits,
            "forbidden_hits": forbidden_hits,
            "citation_count": len(citations),
            "chinese_char_ratio": round(chinese_ratio, 4),
        },
        "checks": checks,
        "notes": notes,
    }


def _build_notes(checks, forbidden_hits, expected_hits, case, source_count) -> List[str]:
    notes = []
    if not checks["has_article"]:
        notes.append("缺少文章产物。")
    if not checks["has_outline"]:
        notes.append("缺少大纲产物。")
    if not checks["has_trace"]:
        notes.append("缺少 runtime trace，难以复盘工具调用链路。")
    if forbidden_hits:
        notes.append("检索或文章中出现跑题关键词：" + ", ".join(forbidden_hits))
    missing = [item for item in case.expected_keywords if item not in expected_hits]
    if missing:
        notes.append("未覆盖期望关键词：" + ", ".join(missing))
    if source_count < case.min_sources:
        notes.append("来源数量不足，当前 {0}，期望至少 {1}。".format(source_count, case.min_sources))
    if not notes:
        notes.append("本次运行的核心指标满足当前规则评估要求。")
    return notes


def write_scorecards(run_dir, scorecard) -> Tuple[Path, Path]:
    run_dir = Path(run_dir)
    json_path = run_dir / "scorecard.json"
    md_path = run_dir / "scorecard.md"
    json_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(_render_scorecard_markdown(scorecard), encoding="utf-8")
    return json_path, md_path


def _render_scorecard_markdown(scorecard) -> str:
    lines = [
        "# Research Eval Scorecard",
        "",
        "## Summary",
        "",
        "- Topic: {0}".format(scorecard.get("topic", "")),
        "- Total Score: {0}".format(scorecard.get("scores", {}).get("total", 0)),
        "",
        "## Scores",
        "",
    ]
    for name, value in scorecard.get("scores", {}).items():
        lines.append("- {0}: {1}".format(name, value))
    lines.extend(["", "## Metrics", ""])
    for name, value in scorecard.get("metrics", {}).items():
        lines.append("- {0}: {1}".format(name, value))
    lines.extend(["", "## Checks", ""])
    for name, value in scorecard.get("checks", {}).items():
        lines.append("- {0}: {1}".format(name, value))
    lines.extend(["", "## Notes", ""])
    for note in scorecard.get("notes", []):
        lines.append("- {0}".format(note))
    lines.append("")
    return "\n".join(lines)
