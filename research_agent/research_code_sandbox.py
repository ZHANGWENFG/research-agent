"""代码执行沙箱（缺口②，2026-08-16 新增）。

依据 Code Interpreter 模式：LLM 生成 Python 代码 → 受限环境执行 →
结果回传 agent。实现策略为 **subprocess + 超时强杀 + 输出截断 + 危险
模式白名单拒绝**，零第三方依赖、离线可测。

安全边界（如实声明）：这是"尽力而为"的规则层防护，**不是隔离容器**。
防御恶意代码（提权、逃逸、资源耗尽）不在本层范围内——本层用于研究
agent 验算论文数字、跑小型实验脚本；FORBIDDEN_PATTERNS 拦截最常见的
文件/系统/网络/任意执行入口。

接口:
- `FORBIDDEN_PATTERNS`: [(compiled_pattern, 说明), ...] 白名单拒绝表
- `_is_blocked(code)` → {blocked, reason}
- `run_python_sandbox(code, timeout=10, max_output_bytes=65536)`
  → {ok, stdout, stderr, exit_code, duration_ms, blocked, reason}
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from typing import Dict, Tuple

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_OUTPUT_BYTES = 65536

# 危险模式白名单拒绝表: (正则, 说明)
FORBIDDEN_PATTERNS = [
    (re.compile(r"^\s*import\s+os\b", re.M), "import os（文件/系统操作）"),
    (re.compile(r"^\s*from\s+os\b", re.M), "from os（文件/系统操作）"),
    (re.compile(r"^\s*import\s+subprocess\b", re.M), "import subprocess（子进程执行）"),
    (re.compile(r"^\s*from\s+subprocess\b", re.M), "from subprocess（子进程执行）"),
    (re.compile(r"open\s*\("), "open(（文件读写）"),
    (re.compile(r"__import__\s*\("), "__import__(（动态导入）"),
    (re.compile(
        r"^\s*import\s+(socket|requests|urllib|http|ftplib|smtplib|aiohttp|httpx)\b",
        re.M,
    ), "网络库导入（socket/requests/urllib/http 等）"),
    (re.compile(
        r"^\s*from\s+(socket|requests|urllib|http|ftplib|smtplib|aiohttp|httpx)\b",
        re.M,
    ), "网络库导入（socket/requests/urllib/http 等）"),
    (re.compile(r"^\s*import\s+(shutil|glob|pathlib|tempfile)\b", re.M), "文件系统库导入"),
    (re.compile(r"^\s*from\s+(shutil|glob|pathlib|tempfile)\b", re.M), "文件系统库导入"),
    (re.compile(r"eval\s*\(|exec\s*\("), "eval/exec（任意代码执行）"),
    (re.compile(r"os\.(system|popen|remove|unlink|rmdir|rename)"), "os 危险调用"),
    (re.compile(r"\bcompile\s*\("), "compile(（动态编译执行）"),
    (re.compile(r"\binput\s*\("), "input(（阻塞等待输入）"),
]


def _is_blocked(code) -> Dict:
    """白名单拒绝判定。返回 {blocked: bool, reason: str}。"""
    if not code or not str(code).strip():
        return {"blocked": False, "reason": ""}
    for pattern, label in FORBIDDEN_PATTERNS:
        if pattern.search(str(code)):
            return {"blocked": True, "reason": label}
    return {"blocked": False, "reason": ""}


def _truncate(text: str, max_output_bytes: int) -> str:
    text = text or ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_output_bytes:
        return text
    head = encoded[:max_output_bytes].decode("utf-8", errors="ignore")
    return head + "\n...[truncated]"


def run_python_sandbox(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> Dict:
    """在子进程 Python 中执行代码，超时强杀、输出截断。

    返回 {ok, stdout, stderr, exit_code, duration_ms, blocked, reason}。
    - blocked=True: 白名单拒绝，未执行任何代码，exit_code=None
    - 超时: reason="timeout after {timeout}s"，exit_code=None
    - 正常/异常退出: exit_code=returncode，ok=(returncode==0)
    """
    start = time.monotonic()
    blocked = _is_blocked(code)
    if blocked["blocked"]:
        return {
            "ok": False,
            "stdout": "",
            "stderr": blocked["reason"],
            "exit_code": None,
            "duration_ms": 0,
            "blocked": True,
            "reason": blocked["reason"],
        }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", str(code)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": False,
            "stdout": "",
            "stderr": "timeout after {0}s".format(timeout),
            "exit_code": None,
            "duration_ms": duration_ms,
            "blocked": False,
            "reason": "timeout",
        }
    duration_ms = int((time.monotonic() - start) * 1000)
    return {
        "ok": proc.returncode == 0,
        "stdout": _truncate(proc.stdout, max_output_bytes),
        "stderr": _truncate(proc.stderr, max_output_bytes),
        "exit_code": proc.returncode,
        "duration_ms": duration_ms,
        "blocked": False,
        "reason": "",
    }
