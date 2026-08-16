"""skill 功能单元测试（2026-08-16 新增）。

research_skill.py 是纯文件解析 + 字符串匹配，无 LLM 依赖（llm_confirm 可选），
此前 0 测试拉低覆盖率；本文件覆盖解析/扫描/匹配/LLM 确认全路径。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_skill import match_skills, parse_skill_md, scan_skills


def _write_skill(root: Path, name: str, description: str, triggers: str) -> Path:
    """构造一个带 frontmatter 的 SKILL.md。"""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"triggers: {triggers}\n"
        "---\n"
        "正文：该领域的调研要点。",
        encoding="utf-8",
    )
    return md


def test_parse_skill_md_full(tmp_path):
    """完整 frontmatter 解析：name/description/triggers/正文分离。"""
    path = _write_skill(tmp_path, "quantum", "量子计算领域", "量子, 量子比特, superposition")
    meta = parse_skill_md(path)
    assert meta["name"] == "quantum"
    assert meta["description"] == "量子计算领域"
    assert "量子" in meta["triggers"]
    assert "量子比特" in meta["triggers"]
    assert "正文" in meta["content"]


def test_parse_skill_md_missing_frontmatter(tmp_path):
    """无 frontmatter 的 SKILL.md 也能解析（name 回退目录名，正文=全文）。"""
    path = tmp_path / "plain" / "SKILL.md"
    path.parent.mkdir()
    path.write_text("没有 frontmatter 的正文", encoding="utf-8")
    meta = parse_skill_md(path)
    assert meta["name"] == "plain"
    assert meta["description"] == ""
    assert meta["content"].startswith("没有 frontmatter")


def test_parse_skill_md_unreadable(tmp_path):
    """读取失败返回 None（不崩溃）。"""
    missing = tmp_path / "nope" / "SKILL.md"
    assert parse_skill_md(missing) is None


def test_parse_skill_md_chinese_comma_triggers(tmp_path):
    """中文逗号分隔的 triggers 也应正确拆分。"""
    path = _write_skill(tmp_path, "med", "医疗领域", "医疗，影像，诊断")
    meta = parse_skill_md(path)
    assert "医疗" in meta["triggers"]
    assert "影像" in meta["triggers"]


def test_scan_skills_finds_all(tmp_path):
    """scan_skills 递归找到所有 SKILL.md。"""
    _write_skill(tmp_path, "a", "A领域", "alpha")
    _write_skill(tmp_path, "b", "B领域", "beta")
    skills = scan_skills(str(tmp_path))
    assert len(skills) == 2
    assert {s["name"] for s in skills} == {"a", "b"}


def test_scan_skills_missing_dir_returns_empty():
    """目录不存在返回空列表（不崩溃）。"""
    assert scan_skills("/nonexistent/skills") == []


def test_match_skills_trigger_hit():
    """触发词命中主题 → 匹配。"""
    skills = [
        {"name": "ai", "description": "", "triggers": ["机器学习", "深度学习"]},
        {"name": "bio", "description": "", "triggers": ["基因"]},
    ]
    hits = match_skills("深度学习在医疗的应用", skills)
    assert len(hits) == 1
    assert hits[0]["name"] == "ai"


def test_match_skills_empty_inputs():
    """空 skills / 空主题不崩溃。"""
    assert match_skills("anything", []) == []
    assert match_skills("", [{"name": "a", "triggers": ["x"]}]) == []


def test_match_skills_llm_confirm_filters():
    """LLM 确认回调按协议（yes 开头保留）把规则命中缩小。"""
    skills = [
        {"name": "keep", "triggers": ["量子"]},
        {"name": "drop", "triggers": ["量子"]},
    ]

    def confirm(prompt, **kwargs):
        # 真实协议：回调收到拼好的 prompt，返回文本以 yes/no 开头
        return "yes" if "「keep」" in prompt else "no"

    hits = match_skills("量子计算", skills, llm_confirm=confirm)
    assert [h["name"] for h in hits] == ["keep"]


def test_match_skills_max_hits_limits():
    """max_hits 限制一次注入数量。"""
    skills = [{"name": f"s{i}", "triggers": ["量子"]} for i in range(5)]
    hits = match_skills("量子计算", skills, max_hits=2)
    assert len(hits) == 2
