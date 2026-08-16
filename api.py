"""my-agent 后端 API（改造新增：FastAPI + SSE + 审批 + skill）。

接口：
  POST /api/research                    提交调研任务
  GET  /api/research/{task_id}          查任务状态 / 产物
  GET  /api/research/{task_id}/stream   SSE 事件流（task_status / evidence / approval_request / heartbeat）
  POST /api/research/{task_id}/approve  审批回传（批准 / 拒绝）
  POST /api/chat                        问答（走生产运行时 → 主编排图）
  POST /api/kb/query                    知识库查询
  POST /api/sessions                    建会话（完整多轮会话）
  GET  /api/sessions                    列会话
  GET  /api/sessions/{chat_id}          查会话
  POST /api/sessions/{chat_id}/message  会话内发消息（上下文 + 记忆 + 主编排图）
  POST /api/sessions/{chat_id}/regenerate  重新生成上一条回复
  POST /api/sessions/{chat_id}/compact  压缩上下文（防爆窗，原文可还原）
  POST /api/sessions/{chat_id}/restore  还原压缩前的原文
  GET  /api/sessions/{chat_id}/context  查上下文状态（token / 压缩信息）
  GET  /api/approvals/pending           待审批列表
  GET  /api/skills                      已安装 skill 列表
  GET  /api/admin/status                治理面板：控制面健康状态
  GET  /api/admin/audit                 治理面板：操作审计留痕
  GET  /api/admin/spans?trace_id=       治理面板：全链路 Span 查询
  GET  /api/health                      健康检查

启动：uvicorn api:app --host 127.0.0.1 --port 8000
推送用 SSE、回传用普通 POST——不需要 WebSocket。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="my-agent Paper Research Agent", version="1.0.0")

# 运行根目录（存储：任务 / SQLite / 产物）
ROOT_DIR = Path(__file__).resolve().parent / "storage"
ROOT_DIR.mkdir(parents=True, exist_ok=True)

from research_agent.research_fulltext import ApprovalQueue  # noqa: E402
from research_agent.research_task_status import TaskStatus  # noqa: E402
from research_agent.research_service import ResearchTaskService  # noqa: E402
from research_agent.research_skill import scan_skills  # noqa: E402
from research_agent.research_chat_agent import ResearchChatAgent  # noqa: E402

service = ResearchTaskService(root_dir=str(ROOT_DIR))
approval_queue = ApprovalQueue(str(ROOT_DIR / "approvals.sqlite"))

# 启动时恢复孤儿任务: 上次进程崩溃/被杀后遗留的 running 任务
# （daemon 线程退出时状态文件会停在 running，SSE 客户端永远等不到终态）
# 超过 30 分钟视为 stale → 标记 failed，客户端轮询即可看到终态
try:
    _stale_recovery = service.recover_stale_running_tasks(max_age_seconds=1800)
    if _stale_recovery.get("failed_count"):
        logger.warning(
            "启动恢复 %s 个孤儿 running 任务: %s",
            _stale_recovery["failed_count"],
            _stale_recovery["failed_task_ids"],
        )
except Exception:  # noqa: BLE001
    logger.exception("启动时恢复 stale 任务失败（不影响服务启动）")

# 会话层：完整会话管理（建会话 / 发消息 / 压缩 / 还原 / 重新生成）
# 与 /api/chat 的无状态问答不同，这里维护持久化的多轮会话（chat_sessions/*.json）
chat_agent = ResearchChatAgent(task_service=service)


# ---------- 请求模型 ----------

class ResearchRequest(BaseModel):
    topic: str = Field(..., description="调研主题")
    run_mode: str = "research"
    retriever: str = "pubmed"
    max_perspectives: int = 1
    max_conv_turn: int = 1


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    request_id: Optional[str] = None
    topic: Optional[str] = None
    run_mode: str = "research"
    allow_deep_research: bool = True


class ApproveRequest(BaseModel):
    approval_id: str = Field(..., description="待审批项 ID")
    approve: bool = True
    reason: str = ""


class KBQueryRequest(BaseModel):
    task_id: str
    question: str
    top_k: int = 3


class SessionCreateRequest(BaseModel):
    title: str = ""
    topic: str = ""
    run_mode: str = "research"
    retriever: str = "pubmed"
    output_language: str = "zh"
    context_window_size: int = 6
    context_token_limit: int = 4096
    expected_keywords: Optional[List[str]] = None
    forbidden_keywords: Optional[List[str]] = None


class SessionMessageRequest(BaseModel):
    message: str = Field(..., description="用户消息")


class SessionRestoreRequest(BaseModel):
    compaction_id: str = Field(..., description="压缩记录 ID（由压缩/发消息返回）")


# ---------- 调研任务 ----------

@app.post("/api/research")
def submit_research(request: ResearchRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="topic 不能为空")
    state = service.submit_research_task(
        topic=request.topic,
        run_mode=request.run_mode,
        retriever=request.retriever,
        max_perspectives=request.max_perspectives,
        max_conv_turn=request.max_conv_turn,
    )
    task_id = state["task_id"]

    def _run():
        try:
            service.run_task(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s failed: %s", task_id, exc)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "status": TaskStatus.QUEUED}


@app.get("/api/research/{task_id}")
def get_research_task(task_id: str):
    try:
        return service.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/research/{task_id}/stream")
def stream_research_task(task_id: str):
    """SSE 推送任务状态；心跳防断线；succeeded/failed 结束。"""

    def event_stream():
        heartbeat_at = time.time()
        while True:
            try:
                state = service.get_task(task_id)
            except Exception:  # noqa: BLE001
                yield _sse("task_status", {"task_id": task_id, "status": "not_found"})
                break
            status = state.get("status", "unknown")
            yield _sse("task_status", {"task_id": task_id, "status": status,
                                       "summary": state.get("result_summary")})
            if status in TaskStatus.TERMINAL:
                break
            if time.time() - heartbeat_at > 15:
                yield _sse("heartbeat", {"ts": time.time()})
                heartbeat_at = time.time()
            time.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/research/{task_id}/approve")
def approve_download(task_id: str, request: ApproveRequest):
    """审批回传：推送用 SSE（approval_request），回传用普通 POST——无需 WebSocket。"""
    record = approval_queue.resolve(request.approval_id, request.approve, request.reason)
    if record is None:
        raise HTTPException(status_code=404, detail="审批项不存在")
    return {"task_id": task_id, "resolved": record}


@app.get("/api/approvals/pending")
def list_pending_approvals():
    return {"approvals": approval_queue.list_pending()}


# ---------- 问答 ----------

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    thread_id = request.thread_id or uuid.uuid4().hex
    request_id = request.request_id or uuid.uuid4().hex
    result = service.invoke_conversation_graph(
        tenant_id="local",
        thread_id=thread_id,
        request_id=request_id,
        user_id="local-user",
        message=request.message,
        topic=request.topic or request.message[:80],
        run_mode=request.run_mode,
        allow_deep_research=request.allow_deep_research,
    )
    return {
        "thread_id": thread_id,
        "request_id": request_id,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "grounded": result.get("grounded", False),
        "retrieval_stack": result.get("retrieval_stack", ""),
        "governance": result.get("governance", {}),
    }


@app.post("/api/kb/query")
def query_kb(request: KBQueryRequest):
    answer = service.query_knowledge_base(
        request.task_id, request.question, top_k=request.top_k
    )
    return answer


# ---------- 会话管理（完整多轮会话：建会话 / 发消息 / 压缩 / 还原 / 重新生成） ----------

@app.post("/api/sessions")
def create_session(request: SessionCreateRequest):
    """建会话：返回 chat_id，后续所有会话操作都用它。"""
    session = chat_agent.create_session(
        title=request.title,
        topic=request.topic,
        run_mode=request.run_mode,
        retriever=request.retriever,
        output_language=request.output_language,
        context_window_size=request.context_window_size,
        context_token_limit=request.context_token_limit,
        expected_keywords=request.expected_keywords,
        forbidden_keywords=request.forbidden_keywords,
    )
    return {"chat_id": session["chat_id"], "session": session}


@app.get("/api/sessions")
def list_sessions(limit: int = 50):
    """列会话：按更新时间倒序，带消息数和最后一条预览。"""
    limit = max(1, min(200, int(limit)))
    return {"sessions": chat_agent.list_sessions(limit=limit)}


@app.get("/api/sessions/{chat_id}")
def get_session(chat_id: str):
    """查会话完整内容（消息 / 压缩状态 / 上下文视图）。"""
    try:
        return chat_agent.get_session(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/sessions/{chat_id}/message")
def send_session_message(chat_id: str, request: SessionMessageRequest):
    """发消息：走完整会话链（上下文窗口 + 记忆召回 + 主编排图），返回带引用的回答。
    注意：该接口同步执行，深度调研场景可能耗时较长。"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    try:
        return chat_agent.send_message(chat_id, request.message)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{chat_id}/regenerate")
def regenerate_session_last(chat_id: str):
    """重新生成上一条回复：基于最近一条用户消息再答一次，旧回复标记版本号保留。"""
    try:
        return chat_agent.regenerate_last(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{chat_id}/compact")
def compact_session(chat_id: str, force: bool = True):
    """压缩上下文：把旧消息压成摘要（防爆窗口），原文存档可还原。"""
    try:
        return chat_agent.compact_context(chat_id, force=force)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/sessions/{chat_id}/restore")
def restore_session(chat_id: str, request: SessionRestoreRequest):
    """还原上下文：按压缩记录 ID 把摘要 100% 还原成原文。"""
    try:
        return chat_agent.restore_context(chat_id, request.compaction_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{chat_id}/context")
def get_session_context(chat_id: str):
    """查上下文状态：token 用量 / 压缩信息 / 当前上下文视图。"""
    try:
        return chat_agent.get_context(chat_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------- skill ----------

@app.get("/api/skills")
def list_skills():
    skills_dir = Path(__file__).resolve().parent / "skills"
    skills = scan_skills(str(skills_dir))
    return {"skills": [{"name": s["name"], "description": s["description"],
                        "triggers": s["triggers"]} for s in skills]}


# ---------- 后台治理面板（SQLite WAL 控制面查询） ----------

@app.get("/api/admin/status")
def admin_status():
    """治理面板：控制面健康状态（后端 / 表行数 / WAL 模式）。"""
    return service.get_production_status()


@app.get("/api/admin/audit")
def admin_audit(limit: int = 100):
    """治理面板：操作审计留痕（幂等/熔断/授权等事件）。"""
    limit = max(1, min(500, int(limit)))
    return {"audit_events": service.list_production_audit_events(limit=limit)}


@app.get("/api/admin/spans")
def admin_spans(trace_id: str = ""):
    """治理面板：全链路 Span 查询（按 trace_id 聚合，无则返回提示）。"""
    if not trace_id:
        return {"spans": [], "message": "缺少 trace_id 参数，先查 /api/chat 返回的 governance 字段"}
    return {"trace_id": trace_id, "spans": service.list_production_spans(trace_id)}


# ---------- 健康检查 ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "my-agent", "root_dir": str(ROOT_DIR)}


def _sse(event: str, payload: dict) -> str:
    return "event: {0}\ndata: {1}\n\n".format(event, json.dumps(payload, ensure_ascii=False))
