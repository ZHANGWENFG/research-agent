"""调研任务状态机枚举 —— 全项目唯一状态定义。

契约：`queued -> running -> succeeded | failed`
- `queued`     提交成功，等待调度
- `running`    任务执行中
- `succeeded`  完成（产物已落盘）
- `failed`     失败（产物/日志已落盘，可查原因）

注意：`research_benchmarks.py`（benchmark 子进程状态）与
`research_fulltext.py`（全文下载状态）是各自独立的状态机，语义不同，
不应混用本枚举。
"""


class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    # SSE 流的结束态：任务到此不再有新事件
    TERMINAL = (SUCCEEDED, FAILED)
