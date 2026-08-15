"""主编排状态图（改造：LangGraph 10 节点 + 3 条件路由）。

节点：classify → memory_recall → (casual_chat | memory_answer | knowledge_retrieval)
      → evidence_grade → (answer_with_citations | deep_research | refuse_or_clarify)
      → memory_candidate_write → final_trace → END

挂载：SqliteSaver 检查点（崩溃可恢复）；RetryPolicy 只挂联网节点
      （knowledge_retrieval / deep_research，仅网络类错误重试 2 次）；
      deep_research 节点超时 300 秒。

复用原项目模块：intent_router（意图路由）、memory（长期记忆）、qa（知识库问答）、
research_qa（证据裁判）、router_llm（LLM 闭包）；deep_research 用自研多角色循环。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

logger = logging.getLogger(__name__)

RETRY_NETWORK_ERRORS = (ConnectionError, TimeoutError)

# 联网节点重试策略：网络类错误重试 2 次（共 3 次尝试），指数退避
NETWORK_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=30.0,
    retry_on=RETRY_NETWORK_ERRORS,
    jitter=True,
)


class ConversationState(TypedDict, total=False):
    thread_id: str
    message: str
    topic: str
    user_id: str
    router_decision: Dict[str, Any]
    memory_hits: list
    knowledge_result: Dict[str, Any]
    evidence_grade: Dict[str, Any]
    deep_research_result: Dict[str, Any]
    answer: str
    citations: list
    grounded: bool
    retrieval_stack: str
    trace_id: str
    allow_deep_research: bool
    conversation_history: list


class ResearchLangGraph:
    """主编排运行时（改造版）。依赖全部注入，便于测试与替换。"""

    def __init__(
        self,
        intent_router: Callable,
        memory_service: Any,
        kb_service: Any,
        evidence_judge: Callable,
        chat_llm: Callable,
        research_runner: Callable,
        memory_writer: Callable,
        checkpointer: Any = None,
    ):
        self.intent_router = intent_router
        self.memory_service = memory_service
        self.kb_service = kb_service
        self.evidence_judge = evidence_judge
        self.chat_llm = chat_llm
        self.research_runner = research_runner
        self.memory_writer = memory_writer
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    # ---------- 节点 ----------

    def _classify(self, state: ConversationState) -> dict:
        router = self.intent_router
        message = state.get("message", "")
        if callable(router):
            decision = router(message=message, topic=state.get("topic", ""),
                              history=state.get("conversation_history", []))
        else:  # ResearchIntentRouter 对象 → .route(message, session, context_window)
            decision = router.route(
                message=message,
                session={"topic": state.get("topic", "")},
                context_window=state.get("conversation_history", []),
            )
        return {"router_decision": decision}

    def _memory_recall(self, state: ConversationState) -> dict:
        hits = []
        if self.memory_service is not None:
            try:
                result = self.memory_service.search(
                    namespace=state.get("user_id", "local"),
                    query=state.get("message", ""),
                    top_k=5,
                )
                hits = result.get("results", []) if isinstance(result, dict) else (result or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory recall failed: %s", exc)
        return {"memory_hits": hits or []}

    def _casual_chat(self, state: ConversationState) -> dict:
        reply = self.chat_llm(state.get("message", ""), max_tokens=400, temperature=0.7)
        text = reply[0] if isinstance(reply, list) else reply
        return {"answer": text, "grounded": False, "retrieval_stack": "casual_chat"}

    def _memory_answer(self, state: ConversationState) -> dict:
        hits = state.get("memory_hits", [])
        if not hits:
            return {"answer": "我暂时没在记忆中找到相关内容。", "grounded": False,
                    "retrieval_stack": "memory_answer"}
        content = "\n".join(f"- {h.get('content', '')}" for h in hits[:5])
        prompt = f"基于以下记忆内容回答用户问题。问题：{state.get('message', '')}\n记忆：\n{content}"
        reply = self.chat_llm(prompt, max_tokens=300, temperature=0.3)
        return {"answer": reply[0] if isinstance(reply, list) else reply,
                "grounded": True, "retrieval_stack": "memory_answer"}

    def _knowledge_retrieval(self, state: ConversationState) -> dict:
        result = {"evidence": [], "citations": [], "answer": "", "grounded": False}
        if self.kb_service is not None:
            try:
                if callable(self.kb_service):
                    result = self.kb_service(
                        question=state.get("message", ""),
                        top_k=3,
                        task_id=state.get("topic", ""),
                    )
                else:  # ResearchKnowledgeBase 风格对象 → .query()
                    result = self.kb_service.query(
                        state.get("message", ""),
                        top_k=3,
                        task_id=state.get("topic", ""),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("knowledge retrieval failed: %s", exc)
        return {"knowledge_result": result}

    def _evidence_grade(self, state: ConversationState) -> dict:
        evidence = (state.get("knowledge_result") or {}).get("evidence", [])
        if not evidence:
            return {"evidence_grade": {"sufficient": False, "reason": "no evidence"}}
        try:
            decision = self.evidence_judge(
                question=state.get("message", ""), evidence=evidence[:5]
            )
            return {"evidence_grade": decision}
        except Exception as exc:  # noqa: BLE001
            logger.warning("evidence judge failed, fallback to rule: %s", exc)
            sufficient = len(evidence) >= 2
            return {"evidence_grade": {"sufficient": sufficient, "reason": "rule fallback"}}

    def _deep_research(self, state: ConversationState) -> dict:
        try:
            result = self.research_runner(state.get("message", ""))
            return {"deep_research_result": result}
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep research failed: %s", exc)
            return {"deep_research_result": {"error": str(exc), "article": ""}}

    def _answer_with_citations(self, state: ConversationState) -> dict:
        research = state.get("deep_research_result")
        kb = state.get("knowledge_result") or {}
        if research and research.get("article"):
            citations = research.get("citation_pool", [])
            return {
                "answer": research["article"],
                "citations": citations,
                "grounded": bool(citations),
                "retrieval_stack": "myagent_deep_research_loop",
            }
        if kb.get("answer"):
            return {
                "answer": kb["answer"],
                "citations": kb.get("citations", []),
                "grounded": kb.get("grounded", False),
                "retrieval_stack": "knowledge_base",
            }
        return {"answer": "现有资料不足以回答该问题。", "citations": [], "grounded": False,
                "retrieval_stack": "none"}

    def _memory_candidate_write(self, state: ConversationState) -> dict:
        if self.memory_writer is not None:
            try:
                self.memory_writer(
                    message=state.get("message", ""),
                    answer=state.get("answer", ""),
                    user_id=state.get("user_id", "local"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory write failed: %s", exc)
        return {}

    def _final_trace(self, state: ConversationState) -> dict:
        trace = {
            "trace_id": state.get("trace_id") or uuid.uuid4().hex,
            "answer_len": len(state.get("answer", "")),
            "citations": len(state.get("citations", [])),
            "stack": state.get("retrieval_stack", ""),
            "ts": time.time(),
        }
        logger.debug("trace: %s", trace)
        return {"trace_id": trace["trace_id"]}

    # ---------- 条件路由 ----------

    def _route_after_classify(self, state: ConversationState) -> str:
        intent = (state.get("router_decision") or {}).get("intent", "knowledge")
        if intent in ("casual", "chitchat"):
            return "casual_chat"
        if intent in ("memory", "remember"):
            return "memory_answer"
        return "knowledge_retrieval"

    def _route_after_evidence_grade(self, state: ConversationState) -> str:
        grade = state.get("evidence_grade") or {}
        if grade.get("sufficient"):
            return "answer_with_citations"
        if state.get("allow_deep_research", True):
            return "deep_research"
        return "refuse_or_clarify"

    # ---------- 图装配 ----------

    def _build_graph(self):
        builder = StateGraph(ConversationState)
        builder.add_node("classify", self._classify)
        builder.add_node("memory_recall", self._memory_recall)
        builder.add_node("casual_chat", self._casual_chat)
        builder.add_node("memory_answer", self._memory_answer)
        builder.add_node("knowledge_retrieval", self._knowledge_retrieval,
                         retry_policy=NETWORK_RETRY)
        builder.add_node("evidence_grade", self._evidence_grade)
        # 注意：langgraph 1.2.9 同步节点不支持 timeout（仅 async 节点可安全取消），
        # 故联网调研节点只挂网络错误重试；整体超时由调用方/服务层兜底。
        builder.add_node(
            "deep_research", self._deep_research, retry_policy=NETWORK_RETRY
        )
        builder.add_node("refuse_or_clarify", lambda s: {
            "answer": "当前无法回答，请补充信息或明确调研主题。", "grounded": False,
            "retrieval_stack": "refuse_or_clarify"})
        builder.add_node("answer_with_citations", self._answer_with_citations)
        builder.add_node("memory_candidate_write", self._memory_candidate_write)
        builder.add_node("final_trace", self._final_trace)

        builder.add_edge(START, "classify")
        builder.add_edge("classify", "memory_recall")
        builder.add_conditional_edges(
            "memory_recall",
            self._route_after_classify,
            {"casual_chat": "casual_chat", "memory_answer": "memory_answer",
             "knowledge_retrieval": "knowledge_retrieval"},
        )
        builder.add_edge("knowledge_retrieval", "evidence_grade")
        builder.add_conditional_edges(
            "evidence_grade",
            self._route_after_evidence_grade,
            {"answer_with_citations": "answer_with_citations",
             "deep_research": "deep_research",
             "refuse_or_clarify": "refuse_or_clarify"},
        )
        builder.add_edge("deep_research", "answer_with_citations")
        for node in ("casual_chat", "memory_answer", "answer_with_citations",
                     "refuse_or_clarify"):
            builder.add_edge(node, "memory_candidate_write")
        builder.add_edge("memory_candidate_write", "final_trace")
        builder.add_edge("final_trace", END)

        return builder.compile(checkpointer=self.checkpointer)

    def invoke(self, initial: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        return self.graph.invoke(initial, config=config)
