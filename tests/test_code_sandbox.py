"""缺口②：代码执行沙箱单测。

锁定语义：危险模式白名单拒绝（不执行）；time.sleep 超时强杀；
大输出截断；语法错误 exit_code 非 0；正常代码返回 stdout。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.research_code_sandbox import (  # noqa: E402
    _is_blocked,
    run_python_sandbox,
)


class TestIsBlocked:
    def test_golden_allowed(self):
        for code in ["print('42')", "x = 1\ny = x + 2\nprint(y)", "2**10", ""]:
            assert _is_blocked(code)["blocked"] is False, code

    def test_golden_forbidden(self):
        cases = {
            "import os": "import os",
            "from os import path": "from os",
            "import subprocess": "import subprocess",
            "with open('f') as fh: pass": "open(",
            "import requests": "网络库",
            "from urllib.request import urlopen": "网络库",
            "eval('1+1')": "eval/exec",
            "exec('x=1')": "eval/exec",
            "import pathlib": "文件系统",
        }
        for code, expect_part in cases.items():
            result = _is_blocked(code)
            assert result["blocked"] is True, code
            assert expect_part in result["reason"], (code, result["reason"])


class TestRunPythonSandbox:
    def test_print_42_ok(self):
        result = run_python_sandbox("print('42')")
        assert result["ok"] is True
        assert result["stdout"].strip() == "42"
        assert result["exit_code"] == 0
        assert result["blocked"] is False

    def test_clean_math_allowed(self):
        result = run_python_sandbox("print(2**10)")
        assert result["ok"] is True
        assert result["stdout"].strip() == "1024"

    def test_import_os_rejected_before_execution(self):
        result = run_python_sandbox("import os\nprint('should not run')")
        assert result["blocked"] is True
        assert result["exit_code"] is None
        assert result["stdout"] == ""
        assert "import os" in result["reason"]

    def test_network_import_rejected(self):
        result = run_python_sandbox("import requests\nprint('x')")
        assert result["blocked"] is True
        assert "网络库" in result["reason"]

    def test_timeout_kills_sleep(self):
        # sleep(30) 必须被 timeout 强杀，不能等 30 秒
        result = run_python_sandbox("import time\ntime.sleep(30)", timeout=3)
        assert result["ok"] is False
        assert result["reason"] == "timeout"
        assert result["exit_code"] is None
        assert result["duration_ms"] < 20000

    def test_large_output_truncated(self):
        result = run_python_sandbox(
            "print('x' * 200000)", max_output_bytes=1024
        )
        assert result["ok"] is True
        assert "...[truncated]" in result["stdout"]
        assert len(result["stdout"]) < 4096

    def test_syntax_error_nonzero_exit(self):
        result = run_python_sandbox("def foo(:\n    pass")
        assert result["ok"] is False
        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"] or "语法错误" in result["stderr"]

    def test_explicit_exit_code(self):
        result = run_python_sandbox("import sys\nsys.exit(3)")
        assert result["ok"] is False
        assert result["exit_code"] == 3

    def test_stderr_captured(self):
        result = run_python_sandbox("import sys\nprint('err', file=sys.stderr)")
        assert result["stderr"].strip() == "err"

    def test_blocked_reports_duration_zero(self):
        result = run_python_sandbox("open('/etc/passwd')")
        assert result["blocked"] is True
        assert result["duration_ms"] == 0
