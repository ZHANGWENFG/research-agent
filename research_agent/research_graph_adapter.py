"""主编排图适配器（改造新增）：兼容接口包装自研主编排图。

生产运行时（ResearchProductionRuntime）原本以
  graph_runtime_class(root_dir, task_service, intent_router=, chat_llm=, evidence_judge=)
的方式实例化 v44 运行时并调用 invoke(**payload) 返回 dict（含 node_events 等）。
本适配器保持同一接口，内部组装 主编排图（ResearchLangGraph）：
  - memory_service：LongTermMemoryService
  - kb_service：task_service.query_knowledge_base 包装
  - research_runner：自研多角色循环 run_research_loop
  - memory_writer：LongTermMemoryService.ingest_message 包装
  - checkpointer：SqliteSaver（SQLite 断点续跑）
返回结构与原接口兼容（answer/citations/grounded/retrieval_stack/node_events/runtime）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResearchGraphAdapter:
    """兼容接口的图适配器（供生产运行时直接使用）。"""

    def __init__(
        self,
        root_dir,
        task_service,
        intent_router=None,
        chat_llm=None,
        evidence_judge=None,
        retriever: str = "pubmed",
    ):
        self.root_dir = root_dir
        self.task_service = task_service
        self.intent_router = intent_router
        self.chat_llm = chat_llm
        self.evidence_judge = evidence_judge
        self.retriever = retriever
        self._runtime = None

    def _build_runtime(self, run_mode: str = "research"):
        from .research_langgraph import ResearchLangGraph
        from .research_longterm_memory import LongTermMemoryService
        from .research_loop import run_research_loop
        from .research_router_llm import (
            build_chat_llm_callable,
            build_intent_router,
            build_judge_llm_callable,
        )

        real_mode = run_mode == "research"
        intent_router = self.intent_router or build_intent_router(run_mode=run_mode)
        chat_llm = self.chat_llm or build_chat_llm_callable(enabled=real_mode)
        evidence_judge = self.evidence_judge or build_judge_llm_callable(
            enabled=real_mode
        )
        memory_service = LongTermMemoryService(self.root_dir / "memory_service")

        def kb_query(question: str, top_k: int = 3, task_id: str = ""):
            try:
                return self.task_service.query_knowledge_base(
                    task_id, question, top_k=top_k
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("kb query failed: %s", exc)
                return {"evidence": [], "citations": [], "answer": "", "grounded": False}

        def research_runner(topic: str):
            """深度调研：三源检索切换 + 全文获取 + skill 注入（与 service 主线一致）。"""
            from .research_fulltext import (
                ApprovalQueue,
                download_file,
                fetch_europepmc_fulltext,
                fetch_pmc_fulltext,
                is_whitelisted,
            )
            from .research_skill import inject_skill, match_skills, scan_skills

            retriever_name = str(self.retriever or "pubmed").strip().lower()
            if retriever_name in ("arxiv", "arxivrm"):
                from .rm import ArxivRM

                rm = ArxivRM(k=3)
            elif retriever_name in ("local-pdf", "localpdf", "local_pdf", "local"):
                from .rm import LocalPDFRM

                rm = LocalPDFRM(k=3, pdf_dir=str(self.task_service.root_dir / "pdfs"))
            else:
                from .research_pubmed import PubMedRM

                rm = PubMedRM(k=3)

            # 审批库与 api.py / service 共用同一文件，保证审批闭环可见
            approval_queue = ApprovalQueue(
                str(self.task_service.root_dir / "approvals.sqlite")
            )

            def get_fulltext(evidence_item):
                meta = evidence_item.get("meta") or {}
                pmcid = str(meta.get("pmcid") or "")
                pmid = str(meta.get("pmid") or "")
                url = str(evidence_item.get("url") or "")
                if pmcid:
                    text = fetch_pmc_fulltext(pmcid)
                    if text:
                        return {"ok": True, "source": "pmc", "pmcid": pmcid,
                                "chars": len(text), "preview": text[:800]}
                if pmid:
                    text = fetch_europepmc_fulltext(pmid)
                    if text:
                        return {"ok": True, "source": "europepmc", "pmid": pmid,
                                "chars": len(text), "preview": text[:800]}
                if url and is_whitelisted(url):
                    downloaded = download_file(
                        url, str(self.root_dir / "fulltext")
                    )
                    return dict(downloaded, source="whitelist_download", url=url)
                if url:
                    record = approval_queue.create(
                        url=url,
                        source=str(evidence_item.get("title") or "")[:120],
                        size_hint=0,
                        task_id=topic,
                    )
                    return {"ok": False, "source": "approval_required",
                            "approval_id": record["id"], "status": record["status"],
                            "url": url}
                return {"ok": False, "source": "none"}

            skill_context = ""
            skills_dir = self.task_service.root_dir.parent / "skills"
            if skills_dir.is_dir():
                try:
                    skills = scan_skills(str(skills_dir))
                    hits = match_skills(topic, skills)
                    if hits:
                        skill_context = "".join(
                            inject_skill(None, skill) for skill in hits
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("skill inject skipped: %s", exc)

            return run_research_loop(
                topic,
                llm_call=chat_llm,
                search=rm.forward,
                fulltext=get_fulltext,
                skill_context=skill_context,
            )

        def memory_writer(message: str, answer: str, user_id: str = "local"):
            try:
                memory_service.ingest_message(
                    namespace=user_id, message=message
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory write failed: %s", exc)

        # 检查点：SQLite 持久化断点续跑（langgraph-checkpoint-sqlite 3.x）。
        # 连接由本类长期持有（check_same_thread=False），进程生命周期内可反复读写；
        # 换 SqliteSaver 后线程状态/历史跨进程重启仍可恢复。失败时降级 InMemorySaver。
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpoint_path = self.root_dir / "checkpoints.sqlite"
            checkpoint_conn = sqlite3.connect(
                str(checkpoint_path), check_same_thread=False
            )
            checkpointer = SqliteSaver(checkpoint_conn)
            logger.info("checkpointer: SqliteSaver(%s)", checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SqliteSaver unavailable, fallback InMemorySaver: %s", exc
            )
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()

        return ResearchLangGraph(
            intent_router=intent_router,
            memory_service=memory_service,
            kb_service=kb_query,
            evidence_judge=evidence_judge,
            chat_llm=chat_llm,
            research_runner=research_runner,
            memory_writer=memory_writer,
            checkpointer=checkpointer,
        )

    def invoke(self, **payload) -> Dict[str, Any]:
        run_mode = payload.get("run_mode", "research")
        retriever = payload.get("retriever") or self.retriever
        if retriever:
            self.retriever = retriever
        runtime = self._build_runtime(run_mode)
        thread_id = payload.get("thread_id", "default")
        initial = {
            "thread_id": thread_id,
            "message": payload.get("message", ""),
            "topic": payload.get("topic", "") or payload.get("message", "")[:80],
            "user_id": payload.get("user_id", "local-user"),
            "allow_deep_research": payload.get("allow_deep_research", True),
            "conversation_history": payload.get("conversation_history", []),
        }
        final = runtime.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        return {
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "grounded": final.get("grounded", False),
            "retrieval_stack": final.get("retrieval_stack", "myagent"),
            "node_events": [],
            "runtime": "myagent",
            "router_decision": final.get("router_decision"),
            "memory_hits": final.get("memory_hits", []),
        }

    def get_thread_state(self, thread_id: str):
        try:
            runtime = self._build_runtime()
            return runtime.graph.get_state({"configurable": {"thread_id": thread_id}})
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def get_thread_history(self, thread_id: str, limit: int = 50):
        try:
            runtime = self._build_runtime()
            history = list(runtime.graph.get_state_history(
                {"configurable": {"thread_id": thread_id}}
            ))
            return [{"values": h.values} for h in history[:limit]]
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
