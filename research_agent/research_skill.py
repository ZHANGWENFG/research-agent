"""skill 功能（改造新增，领域增强层）。

skill = 一份 SKILL.md 领域知识包（术语/视角/检索词/注意事项），放项目 skills/ 目录。
机制三步：扫描 skills/ → 按主题匹配（关键词规则先筛 + LLM 可选确认）→ 注入调研流程。

与工具解耦：skill 管"怎么调研"（流程/知识/术语），工具管"能做什么"（检索/下载/问答），
不依赖 MCP——skill 是项目内置机制。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

SKILL_MD = "SKILL.md"


def parse_skill_md(path: Path) -> Optional[Dict]:
    """解析 SKILL.md frontmatter（name/description/triggers）+ 正文。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    meta: Dict = {"name": path.parent.name, "description": "", "triggers": []}
    body = text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if match:
        frontmatter, body = match.group(1), match.group(2)
        for line in frontmatter.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip().strip('"')
                if key in ("name", "description"):
                    meta[key] = value
                elif key == "triggers":
                    meta["triggers"] = [t.strip() for t in value.replace("，", ",").split(",") if t.strip()]
    meta["path"] = str(path)
    meta["content"] = body.strip()
    return meta


def scan_skills(skills_dir: str) -> List[Dict]:
    """扫描 skills/ 目录，返回所有 skill 清单。"""
    root = Path(skills_dir)
    if not root.exists():
        return []
    skills = []
    for skill_path in root.glob(f"*/{SKILL_MD}"):
        parsed = parse_skill_md(skill_path)
        if parsed:
            skills.append(parsed)
    return skills


def match_skills(topic: str, skills: List[Dict],
                 llm_confirm: Optional[Callable] = None,
                 max_hits: int = 2) -> List[Dict]:
    """按主题匹配 skill：关键词规则先筛（触发词出现在主题里），LLM 可选确认。

    max_hits：一次最多注入 1~2 个 skill（配置）。
    """
    if not skills:
        return []
    topic_lower = topic.lower()
    hits = []
    for skill in skills:
        triggers = [t.lower() for t in skill.get("triggers", [])]
        if any(t and t in topic_lower for t in triggers):
            hits.append(skill)
    # LLM 确认（可选）：在规则命中基础上做领域确认
    if llm_confirm is not None and hits:
        confirmed = []
        for skill in hits:
            try:
                prompt = (
                    f"主题：{topic}。判断该主题是否属于领域 skill「{skill.get('name')}」"
                    f"（描述：{skill.get('description')}）。只回答 yes 或 no。"
                )
                reply = llm_confirm(prompt, max_tokens=10, temperature=0.0)
                text = (reply[0] if isinstance(reply, list) else reply).strip().lower()
                if text.startswith("yes"):
                    confirmed.append(skill)
            except Exception:  # noqa: BLE001
                confirmed.append(skill)  # LLM 失败按规则结果保留
        hits = confirmed
    return hits[:max_hits]


def load_skill_context(skill: Dict) -> Dict[str, str]:
    """提取 skill 的可注入内容：术语/视角/检索词/注意事项。"""
    content = skill.get("content", "")
    return {
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "terms": _extract_section(content, "术语"),
        "perspectives": _extract_section(content, "视角"),
        "search_hints": _extract_section(content, "检索"),
        "notes": _extract_section(content, "注意"),
    }


def _extract_section(content: str, heading: str) -> str:
    """从 SKILL.md 正文按小标题提取片段（术语/视角/检索/注意）。"""
    lines = content.splitlines()
    capture = False
    parts = []
    for line in lines:
        if re.match(rf"^#{{1,4}}\s*{heading}", line.strip()):
            capture = True
            continue
        if capture and line.strip().startswith("#"):
            break
        if capture and line.strip():
            parts.append(line.strip())
    return "\n".join(parts)


def inject_skill(prompt_builder: Callable, skill: Optional[Dict]) -> str:
    """把 skill 内容注入视角生成/作家/专家的提示词（如无匹配 skill 则原样返回）。"""
    if not skill:
        return ""
    ctx = load_skill_context(skill)
    return (
        f"\n[领域增强 skill：{ctx['name']}]\n"
        f"领域描述：{ctx['description']}\n"
        f"术语表：{ctx['terms']}\n"
        f"推荐视角：{ctx['perspectives']}\n"
        f"检索提示：{ctx['search_hints']}\n"
        f"注意事项：{ctx['notes']}\n"
    )
