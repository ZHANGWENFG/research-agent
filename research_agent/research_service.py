import json
import threading
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .research_eval import EvalCase, evaluate_run, write_scorecards
from .research_kb_qa import ResearchKnowledgeBase, write_qa_artifact
from .research_retrieval_common import (
    ARTICLE_FILENAME,
    OUTLINE_FILENAME,
    resolve_article_path,
    resolve_outline_path,
)

logger = logging.getLogger(__name__)


class ResearchTaskService:
    """File-backed service core for Research task APIs."""

    def __init__(self, root_dir, max_concurrent_tasks: int = 1, pipeline_runner=None):
        from .research_benchmarks import BenchmarkRunManager

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir = self.root_dir / "tasks"
        self.results_dir = self.root_dir / "results"
        self.tasks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.max_concurrent_tasks = max(1, int(max_concurrent_tasks))
        self._task_slot = threading.BoundedSemaphore(self.max_concurrent_tasks)
        self.pipeline_runner = pipeline_runner
        self.benchmark_runs = BenchmarkRunManager(self.root_dir)

    def get_benchmark_catalog(self):
        return self.benchmark_runs.catalog()

    def start_benchmark_run(
        self,
        benchmark_id: str,
        profile: str = "smoke",
        allow_paid_llm: bool = False,
    ):
        return self.benchmark_runs.start(
            benchmark_id,
            profile=profile,
            allow_paid_llm=allow_paid_llm,
        )

    def get_benchmark_run(self, run_id: str):
        return self.benchmark_runs.get(run_id)

    def cancel_benchmark_run(self, run_id: str):
        return self.benchmark_runs.cancel(run_id)

    def submit_research_task(
        self,
        topic: str,
        retriever: str = "arxiv",
        output_language: str = "zh",
        run_mode: str = "fake",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **options,
    ):
        task_id = uuid.uuid4().hex
        output_dir = self.results_dir / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "task_id": task_id,
            "topic": topic,
            "retriever": retriever,
            "output_language": output_language,
            "run_mode": run_mode,
            "status": "queued",
            "output_dir": str(output_dir),
            "created_at": _now(),
            "updated_at": _now(),
            "queue_index": self._next_queue_index(),
            "expected_keywords": expected_keywords or [],
            "forbidden_keywords": forbidden_keywords or [],
            "options": _redact(options),
        }
        self._write_state(task_id, state)
        return state

    def get_task(self, task_id: str):
        return self._read_state(task_id)

    def list_tasks(self, status: Optional[str] = None):
        tasks = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if status is None or state.get("status") == status:
                tasks.append(state)
        return sorted(
            tasks,
            key=lambda item: (
                int(item.get("queue_index", 0)),
                item.get("created_at", ""),
            ),
        )

    def run_task(self, task_id: str):
        # 并发上限：同时最多跑 max_concurrent_tasks 个任务，其余排队等待
        self._task_slot.acquire()
        try:
            return self._run_task_locked(task_id)
        finally:
            self._task_slot.release()

    def _run_task_locked(self, task_id: str):
        state = self._read_state(task_id)
        state["status"] = "running"
        state["started_at"] = _now()
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        try:
            if state.get("run_mode") == "fail":
                raise RuntimeError("simulated task failure for service testing")
            if state.get("run_mode") == "manual":
                return state
            if state.get("run_mode") == "research":
                self._run_research_loop(state)
            elif state.get("run_mode") != "fake":
                raise ValueError(
                    "Supported run modes are 'fake', 'research', 'manual', and 'fail'."
                )
            else:
                self._run_fake_research(state)
            state["status"] = "succeeded"
            state["finished_at"] = _now()
        except Exception as error:
            state["status"] = "failed"
            state["finished_at"] = _now()
            state["error"] = _redact_error(str(error))
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        return state

    def worker_tick(self):
        running = self._list_tasks_by_status("running")
        capacity = max(0, self.max_concurrent_tasks - len(running))
        queued = self._list_tasks_by_status("queued")
        started = []
        for state in queued[:capacity]:
            state["status"] = "running"
            state["started_at"] = _now()
            state["updated_at"] = _now()
            self._write_state(state["task_id"], state)
            started.append(state["task_id"])
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "running_count": len(running) + len(started),
            "queued_count": max(0, len(queued) - len(started)),
            "started_count": len(started),
            "started_task_ids": started,
        }

    def complete_task(self, task_id: str, success: bool = True, error: str = ""):
        state = self._read_state(task_id)
        state["status"] = "succeeded" if success else "failed"
        if not success:
            state["error"] = _redact_error(error or "task failed")
        state["finished_at"] = _now()
        state["updated_at"] = _now()
        self._write_state(task_id, state)
        return state

    def recover_stale_running_tasks(self, max_age_seconds: float):
        now_ts = time.time()
        failed = []
        for state in self._list_tasks_by_status("running"):
            started_at = _parse_timestamp(state.get("started_at")) or 0.0
            if now_ts - started_at >= max_age_seconds:
                state["status"] = "failed"
                state["error"] = "stale running task recovered after timeout"
                state["finished_at"] = _now()
                state["updated_at"] = _now()
                self._write_state(state["task_id"], state)
                failed.append(state["task_id"])
        return {"failed_count": len(failed), "failed_task_ids": failed}

    def run_stress_benchmark(self, total_tasks: int, fail_every: int = 0):
        created = []
        latencies = []
        max_observed_running = 0
        for index in range(total_tasks):
            mode = "fail" if fail_every and (index + 1) % fail_every == 0 else "fake"
            created.append(
                self.submit_research_task(
                    topic="stress topic {0}".format(index + 1),
                    run_mode=mode,
                )["task_id"]
            )
        while True:
            queued = self._list_tasks_by_status("queued")
            running = self._list_tasks_by_status("running")
            max_observed_running = max(max_observed_running, len(running))
            if not queued and not running:
                break
            batch = self.worker_tick()
            max_observed_running = max(max_observed_running, batch["running_count"])
            for task_id in batch["started_task_ids"]:
                start = time.time()
                state = self._read_state(task_id)
                if state.get("run_mode") == "manual":
                    self.complete_task(task_id, success=True)
                else:
                    self.run_task(task_id)
                latencies.append(time.time() - start)
        states = [self._read_state(task_id) for task_id in created]
        succeeded = len([state for state in states if state["status"] == "succeeded"])
        failed = len([state for state in states if state["status"] == "failed"])
        report = {
            "total_tasks": total_tasks,
            "succeeded": succeeded,
            "failed": failed,
            "failure_rate": round(failed / max(1, total_tasks), 4),
            "avg_latency_sec": round(sum(latencies) / max(1, len(latencies)), 4),
            "p95_latency_sec": round(_percentile(latencies, 0.95), 4),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_observed_running": max_observed_running,
            "retry_count": 0,
        }
        (self.root_dir / "stress_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def get_article(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        path = resolve_article_path(output_dir)
        if not path.exists():
            path = None
        return {
            "task_id": task_id,
            "path": str(path) if path else "",
            "content": (
                path.read_text(encoding="utf-8", errors="replace") if path else ""
            ),
        }

    def get_scorecard(self, task_id: str):
        state = self._read_state(task_id)
        return _read_json(Path(state["output_dir"]) / "scorecard.json", {})

    def get_trace(self, task_id: str):
        state = self._read_state(task_id)
        trace_path = Path(state["output_dir"]) / "research_trace.jsonl"
        return {"task_id": task_id, "events": _load_jsonl(trace_path)}

    def get_dashboard_bundle(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        return {
            "project": {
                "name": "Research Agent",
                "version": "v5.6",
                "description": "Service-backed Research dashboard snapshot",
            },
            "tasks": [state],
            "article": self.get_article(task_id),
            "qa": _read_json(output_dir / "qa_answer.json", {}),
            "scorecard": self.get_scorecard(task_id),
            "trace": self.get_trace(task_id),
            "process": self.get_process_artifacts(task_id),
            "pipeline_worker": _read_json(output_dir / "pipeline_worker.json", {}),
            "service_snapshot": {
                "task_id": task_id,
                "output_dir": str(output_dir),
                "status": state.get("status", ""),
                "run_mode": state.get("run_mode", ""),
                "retriever": state.get("retriever", ""),
                "updated_at": state.get("updated_at", ""),
            },
        }

    def query_knowledge_base(self, task_id: str, question: str, top_k: int = 3):
        from .research_router_llm import build_chat_llm_callable

        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        kb = ResearchKnowledgeBase.from_run_dir(output_dir)
        answer = kb.answer_question(
            question,
            top_k=top_k,
            answer_generator=build_chat_llm_callable(
                enabled=state.get("run_mode") == "research"
            ),
        )
        write_qa_artifact(output_dir, answer)
        return answer

    def ask_research_agent(
        self,
        question: str,
        topic: Optional[str] = None,
        task_id: Optional[str] = None,
        mode: str = "auto",
        top_k: int = 3,
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **options,
    ):
        from .research_qa import ResearchQAAgent

        return ResearchQAAgent(self).ask(
            question=question,
            topic=topic,
            task_id=task_id,
            mode=mode,
            top_k=top_k,
            run_mode=run_mode,
            retriever=retriever,
            output_language=output_language,
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            **options,
        )

    def create_chat_session(
        self,
        title: str = "",
        topic: str = "",
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        context_window_size: int = 6,
        context_token_limit: int = 4096,
        user_id: str = "local-user",
        tenant_id: str = "local",
        memory_enabled: bool = True,
        **options,
    ):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).create_session(
            title=title,
            topic=topic,
            run_mode=run_mode,
            retriever=retriever,
            output_language=output_language,
            expected_keywords=expected_keywords,
            forbidden_keywords=forbidden_keywords,
            context_window_size=context_window_size,
            context_token_limit=context_token_limit,
            user_id=user_id,
            tenant_id=tenant_id,
            memory_enabled=memory_enabled,
            **options,
        )

    def get_chat_session(self, chat_id: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).get_session(chat_id)

    def send_chat_message(self, chat_id: str, message: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).send_message(chat_id, message)

    def list_chat_sessions(self, limit: int = 50):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).list_sessions(limit=limit)

    def regenerate_chat_message(self, chat_id: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).regenerate_last(chat_id)

    def stop_chat_generation(self, chat_id: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).stop_generation(chat_id)

    def get_chat_context(self, chat_id: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).get_context(chat_id)

    def compact_chat_context(self, chat_id: str, force: bool = True):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).compact_context(chat_id, force=force)

    def restore_chat_context(self, chat_id: str, compaction_id: str):
        from .research_chat_agent import ResearchChatAgent

        return ResearchChatAgent(self).restore_context(chat_id, compaction_id)

    def create_memory(self, **payload):
        return self._memory_service_v43().upsert(**payload)

    def list_memories(self, namespace: str, include_inactive: bool = False):
        return {
            "namespace": namespace,
            "memories": self._memory_service_v43().list_memories(
                namespace, include_inactive=include_inactive
            ),
        }

    def search_memories(self, namespace: str, query: str, top_k: int = 5):
        return self._memory_service_v43().search(namespace, query, top_k=top_k)

    def edit_memory(self, namespace: str, memory_id: str, content: str, **updates):
        return self._memory_service_v43().edit(
            namespace=namespace, memory_id=memory_id, content=content, **updates
        )

    def delete_memory(
        self, namespace: str, memory_id: str, reason: str = "user_request"
    ):
        return self._memory_service_v43().delete(namespace, memory_id, reason=reason)

    def export_memories(self, namespace: str):
        return self._memory_service_v43().export_namespace(namespace)

    def set_memory_enabled(self, namespace: str, enabled: bool):
        return self._memory_service_v43().set_enabled(namespace, enabled)

    def invoke_conversation_graph(self, **payload):
        return self._production_runtime().invoke(**payload)

    def get_conversation_graph_spec(self):
        return self._production_runtime().get_graph_spec()

    def get_conversation_thread_state(
        self, thread_id: str, tenant_id: str = "local", user_id: str = "local-user"
    ):
        self._production_control().authorize(
            tenant_id, user_id, "conversation_thread", thread_id, "read_state"
        )
        return self._production_runtime().get_thread_state(thread_id)

    def get_conversation_thread_history(
        self,
        thread_id: str,
        limit: int = 50,
        tenant_id: str = "local",
        user_id: str = "local-user",
    ):
        self._production_control().authorize(
            tenant_id, user_id, "conversation_thread", thread_id, "read_history"
        )
        return self._production_runtime().get_thread_history(thread_id, limit=limit)

    def get_production_trace(self, trace_id: str, tenant_id: str, user_id: str):
        self._production_control().authorize(
            tenant_id, user_id, "trace", trace_id, "read"
        )
        return {
            "trace_id": trace_id,
            "spans": self._production_control().list_spans(trace_id),
        }

    def get_production_status(self):
        return self._production_control().status()

    def list_production_audit_events(self, limit: int = 100):
        """后台治理：操作审计留痕查询（治理面板接口用）。"""
        return self._production_control().list_audit_events(limit=limit)

    def list_production_spans(self, trace_id: str):
        """后台治理：按 trace_id 查全链路 Span（治理面板接口用）。"""
        return self._production_control().list_spans(trace_id)

    def import_evaluation_dataset(self, dataset_path: str):
        from .research_eval_pipeline import AnnotationStore

        source = Path(dataset_path)
        if not source.exists():
            raise ValueError("找不到 v5.4 候选数据集：{0}".format(source))
        dataset = json.loads(source.read_text(encoding="utf-8"))
        if not dataset.get("cases") or not dataset.get("corpus"):
            raise ValueError("v5.4 数据集必须同时包含 cases 和 corpus")
        root = self._evaluation_root()
        target = root / "candidate_dataset.json"
        target.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress = AnnotationStore(root, dataset).progress()
        return dict(progress, configured=True)

    def get_evaluation_status(self):
        from .research_eval_pipeline import AnnotationStore

        dataset = self._evaluation_dataset(required=False)
        if not dataset:
            return {
                "configured": False,
                "trust_level": "candidate",
                "candidate_count": 0,
                "reviewed_count": 0,
                "valid_reviewed_test_count": 0,
                "frozen_test_allowed": False,
                "message": "尚未导入 v5.4 候选数据集。",
            }
        return dict(AnnotationStore(self._evaluation_root(), dataset).progress(), configured=True)

    def list_evaluation_annotations(self, offset: int = 0, limit: int = 50):
        from .research_eval_pipeline import AnnotationStore

        dataset = self._evaluation_dataset()
        store = AnnotationStore(self._evaluation_root(), dataset)
        cases = store.list_cases()
        offset = max(0, int(offset))
        limit = max(1, min(200, int(limit)))
        return {
            "cases": cases[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "total": len(cases),
            "progress": store.progress(),
        }

    def save_evaluation_review(self, case_id: str, review: Dict):
        from .research_eval_pipeline import AnnotationStore

        dataset = self._evaluation_dataset()
        payload = dict(review or {}, case_id=case_id)
        return AnnotationStore(self._evaluation_root(), dataset).save_review(payload)

    def run_evaluation_context(self):
        from .research_eval_pipeline import (
            AnnotationStore,
            enrich_context_cases,
            evaluate_context_scenarios,
            normalize_corpus,
        )

        dataset = self._evaluation_dataset()
        store = AnnotationStore(self._evaluation_root(), dataset)
        reviewed = store.export_reviewed_dataset()["cases"]
        cases = reviewed or store.list_cases()[: min(20, len(dataset.get("cases") or []))]
        cases = enrich_context_cases(cases, normalize_corpus(dataset))
        report = evaluate_context_scenarios(cases)
        report["trust"] = store.progress()
        self._write_evaluation_report("context", report)
        return report

    def run_evaluation_retrieval(
        self,
        embedding: str = "hash",
        top_k: int = 5,
        configurations: Optional[List[str]] = None,
        candidate_k: int = 20,
        enable_reranker: bool = False,
    ):
        from .research_eval_pipeline import (
            AnnotationStore,
            normalize_corpus,
            ranked_document_ids,
            run_retrieval_benchmark,
        )
        from .research_retrieval_runtime import _dense_provider
        from .research_retrieval_index import CrossEncoderReranker, HybridPaperIndex

        dataset = self._evaluation_dataset()
        store = AnnotationStore(self._evaluation_root(), dataset)
        evaluated_dataset = dict(dataset, cases=store.list_cases())
        provider = _dense_provider(embedding)
        index = HybridPaperIndex(normalize_corpus(dataset), provider)
        requested = list(configurations or ["bm25", "dense", "hybrid"])
        skipped = {}
        reranker = None
        if enable_reranker and "hybrid_rerank" not in requested:
            requested.append("hybrid_rerank")
        if "hybrid_rerank" in requested:
            try:
                reranker = CrossEncoderReranker()
            except Exception as error:
                requested.remove("hybrid_rerank")
                skipped["hybrid_rerank"] = str(error)
        if not requested:
            raise ValueError("没有可运行的检索配置")

        def search(case, mode, retrieve_k):
            started = time.perf_counter()
            chunks = index.search(
                case.get("query") or "",
                mode=mode,
                top_k=retrieve_k,
                candidate_k=max(int(candidate_k), retrieve_k),
                reranker=reranker if mode == "hybrid_rerank" else None,
            )
            return {
                "ranked_document_ids": ranked_document_ids(chunks),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }

        progress = store.progress()
        report = run_retrieval_benchmark(
            evaluated_dataset,
            search_fn=search,
            configurations=requested,
            top_k=top_k,
            trust_level=progress["trust_level"],
        )
        report["models"] = {
            "embedding": str(getattr(provider, "name", embedding)),
            "reranker": str(getattr(reranker, "model_name", "")) if reranker else None,
        }
        report["skipped_configurations"] = skipped
        report["trust"] = progress
        self._write_evaluation_report("retrieval", report)
        return report

    def get_evaluation_latest(self):
        from .research_eval_pipeline import sanitize_report

        root = self._evaluation_root()
        return sanitize_report(
            {
                "project": "Research v5.4 Benchmark Console",
                "status": self.get_evaluation_status(),
                "retrieval": _read_json(root / "retrieval_report.json", {}),
                "context": _read_json(root / "context_report.json", {}),
            }
        )

    def _evaluation_root(self):
        root = self.root_dir / "evaluations"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _evaluation_dataset(self, required: bool = True):
        dataset = _read_json(
            self._evaluation_root() / "candidate_dataset.json", {}
        )
        if required and not dataset:
            raise ValueError("尚未导入 v5.4 候选数据集")
        return dataset

    def _write_evaluation_report(self, name: str, report: Dict):
        path = self._evaluation_root() / "{0}_report.json".format(name)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _production_runtime(self):
        from .research_production import ResearchProductionRuntime

        return ResearchProductionRuntime(
            root_dir=self.root_dir / "production_runtime",
            task_service=self,
            control_plane=self._production_control(),
        )

    def _production_control(self):
        from .research_production import ProductionControlPlane

        return ProductionControlPlane(
            self.root_dir / "production_control.sqlite"
        )

    def _memory_service_v43(self):
        from .research_longterm_memory import LongTermMemoryService

        return LongTermMemoryService(self.root_dir / "memory_service")

    def _run_fake_research(self, state: Dict):
        output_dir = Path(state["output_dir"])
        topic = state["topic"]
        topic_lower = str(topic or "").lower()
        pim_topic = (
            "pim" in topic_lower
            or "无源互调" in topic_lower
            or "passive intermodulation" in topic_lower
        )
        if pim_topic:
            keyword = "passive intermodulation"
            framing = "是 RF 系统中由无源器件非线性导致的互调杂散问题"
        else:
            keyword = topic
            framing = "是该方向的核心研究问题"
        article = (
            "# {topic}\n\n"
            "围绕“{topic}”的调研结论：{keyword} {framing}；"
            "模型驱动与数据驱动（神经网络）方法可用于建模、抑制与对消，"
            "并需要可复现的 benchmark 验证效果。[1]\n\n"
            "工程化要点：混合检索（BM25+Dense+RRF）、证据门控、可恢复上下文"
            "与跨会话记忆共同保证回答可溯源。[2]\n"
        ).format(topic=topic, keyword=keyword, framing=framing)
        raw_results = [
            {
                "title": "{0} 研究综述".format(topic),
                "description": "围绕 {0} 的检索结果与关键方法。".format(topic),
                "url": "https://example.com/topic-0",
                "snippets": ["{0} 的模型驱动与数据驱动方法对比。".format(topic)],
            },
            {
                "title": "Neural methods for {0}".format(topic),
                "description": "Neural network approaches for {0} modeling and suppression.".format(
                    topic
                ),
                "url": "https://example.com/topic-1",
                "snippets": ["Neural modeling and cancellation for {0}.".format(topic)],
            },
        ]
        summary = {
            "success": True,
            "task_id": state["task_id"],
            "topic": topic,
            "artifacts": [
                ARTICLE_FILENAME,
                "raw_search_results.json",
                "research_trace.jsonl",
            ],
        }
        (output_dir / OUTLINE_FILENAME).write_text(
            "# {0}\n## 定义\n## 神经网络抑制".format(topic),
            encoding="utf-8",
        )
        (output_dir / ARTICLE_FILENAME).write_text(
            article,
            encoding="utf-8",
        )
        (output_dir / "conversation_log.json").write_text(
            json.dumps(
                (
                    [
                        {
                            "role": "researcher",
                            "message": "如何定义 RF 场景下的 PIM？",
                        },
                        {
                            "role": "expert",
                            "message": "这里 PIM 指 passive intermodulation，不是 processing-in-memory。",
                        },
                    ]
                    if pim_topic
                    else [
                        {
                            "role": "researcher",
                            "message": "如何界定这个主题的核心问题？",
                        },
                        {
                            "role": "expert",
                            "message": "围绕“{0}”，先给出定义与关键方法，"
                            "再比较模型驱动与数据驱动路线。".format(topic),
                        },
                    ]
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "raw_search_results.json").write_text(
            json.dumps(raw_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "reflection.txt").write_text(
            (
                "Critic: 检索结果需要过滤 processing-in-memory / DRAM 语义，"
                "保留 RF passive intermodulation 方向。"
                if pim_topic
                else "Critic: 检索结果需要区分相关与跑题内容，围绕 {0} "
                "保留模型驱动与数据驱动两条主线，并记录证据来源。".format(topic)
            ),
            encoding="utf-8",
        )
        trace_events = [
            {
                "event": "run_start",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "success": True,
            },
            {
                "event": "tool_start",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "tool": "fake_research",
            },
            {
                "event": "tool_end",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "tool": "fake_research",
            },
            {
                "event": "run_end",
                "task_id": state["task_id"],
                "timestamp": _now(),
                "success": True,
            },
        ]
        (output_dir / "research_trace.jsonl").write_text(
            "\n".join(json.dumps(event, ensure_ascii=False) for event in trace_events)
            + "\n",
            encoding="utf-8",
        )
        case = EvalCase(
            topic=topic,
            expected_keywords=state.get("expected_keywords")
            or ["passive intermodulation", "RF"],
            forbidden_keywords=state.get("forbidden_keywords")
            or ["processing-in-memory", "DRAM", "RAM"],
            expected_language=state.get("output_language", "zh"),
            min_sources=1,
        )
        write_scorecards(output_dir, evaluate_run(output_dir, case))

    def get_process_artifacts(self, task_id: str):
        state = self._read_state(task_id)
        output_dir = Path(state["output_dir"])
        return {
            "outline": _read_text(resolve_outline_path(output_dir)),
            "conversation": _read_text(output_dir / "conversation_log.json"),
            "reflection": _read_text(output_dir / "reflection.txt"),
            "run_summary": _read_text(output_dir / "run_summary.json"),
            "raw_search_results": _read_text(output_dir / "raw_search_results.json"),
            "plan": _read_text(output_dir / "query_plan.json")
            or _read_text(output_dir / "raw_search_results.json"),
        }

    def _run_research_loop(self, state: Dict):
        """【改造】主线调研任务：自研多角色循环（替代原 ResearchAgent pipeline）。

        三个接线口子已闭合：
          口子3 检索器三源切换：按 state["retriever"] 选 pubmed / arxiv / local-pdf
          口子1 全文获取：专家回答命中文献时触发 get_fulltext 回调——
                 PMC/EuropePMC 文本接口优先 → 白名单自动下载 → 非白名单进审批队列
          口子2 skill 注入：调研开始前扫描 skills/ 按主题匹配，注入视角生成/专家回答
        成稿写到 {ARTICLE_FILENAME}（知识库问答 from_run_dir 依赖此文件名）。
        """
        from .research_fulltext import (
            ApprovalQueue,
            get_fulltext as _get_fulltext_module,
        )
        from .research_loop import run_research_loop
        from .research_router_llm import build_chat_llm_callable
        from .research_skill import inject_skill, match_skills, scan_skills

        topic = state.get("topic") or ""
        output_dir = Path(state["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        chat_llm = build_chat_llm_callable(enabled=True)

        # ① 检索器三源切换（口子3）：返回格式统一（url/title/description/snippets/meta）
        retriever_name = str(state.get("retriever") or "pubmed").strip().lower()
        if retriever_name in ("arxiv", "arxivrm"):
            from .rm import ArxivRM

            rm = ArxivRM(k=3)
            retriever_used = "arxiv"
        elif retriever_name in ("local-pdf", "localpdf", "local_pdf", "local"):
            from .rm import LocalPDFRM

            pdf_dir = (state.get("options") or {}).get("pdf_dir") or str(
                self.root_dir / "pdfs"
            )
            rm = LocalPDFRM(k=3, pdf_dir=pdf_dir)
            retriever_used = "local-pdf"
        else:
            from .research_pubmed import PubMedRM

            rm = PubMedRM(k=3)
            retriever_used = "pubmed"
        search = rm.forward

        # ② 全文获取（口子1）：审批库与 api.py 共用同一文件（root_dir/approvals.sqlite）
        approval_queue = ApprovalQueue(str(self.root_dir / "approvals.sqlite"))
        fulltext_dir = output_dir / "fulltext"
        # 收敛（2026-08-16）: 全文获取唯一实现是 research_fulltext.get_fulltext，
        # 这里只做薄封装注入依赖（审批队列/下载目录/task_id），不再重写逻辑
        def get_fulltext(evidence_item: Dict) -> Dict:
            return _get_fulltext_module(
                evidence_item,
                download_dir=str(fulltext_dir),
                approval_queue=approval_queue,
                task_id=state["task_id"],
            )

        # ③ skill 注入（口子2）：调研开始前扫描 skills/ 按主题匹配
        skill_context = ""
        skills_dir = self.root_dir.parent / "skills"
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

        # 缺口 4 挂接（2026-08-16）: RESEARCH_PARALLEL=1 走真·多视角并行
        # （ThreadPoolExecutor），否则走原串行 STORM 循环——零行为变化
        if os.getenv("RESEARCH_PARALLEL") == "1":
            result = self._run_parallel_research(
                state, topic, chat_llm, search, skill_context, retriever_used,
            )
        else:
            result = run_research_loop(
                topic,
                llm_call=chat_llm,
                search=search,
                fulltext=get_fulltext,
                skill_context=skill_context,
            )
        article = result.get("article") or ""
        (output_dir / ARTICLE_FILENAME).write_text(
            article, encoding="utf-8"
        )
        (output_dir / "research_loop_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state["result_summary"] = {
            "loop": "self-research-loop",
            "retriever": retriever_used,
            "skill_injected": bool(skill_context),
            "perspectives": result.get("perspectives", []),
            "scorecard": result.get("scorecard", {}),
            "qc_passed": result.get("qc_passed", False),
        }

    # ---------- 缺口 4: 真·多代理并行（RESEARCH_PARALLEL=1 启用） ----------

    def _run_parallel_research(
        self,
        state: Dict,
        topic: str,
        chat_llm,
        search: Callable,
        skill_context: str,
        retriever_used: str,
    ) -> Dict:
        """并行版本调研：多视角线程并行（检索+合成）→ 聚合文章。

        复用 generate_perspectives 产出视角（STORM），执行阶段升级为
        ThreadPoolExecutor 真并行；失败隔离保证单视角失败不拖垮整体。
        """
        from .research_loop import generate_perspectives
        from .research_parallel import run_parallel_perspectives

        output_dir = Path(state["output_dir"])
        # 视角生成（与串行同一入口，失败回退 1 视角 + 并发 1）
        plan_state = {"topic": topic}
        generate_perspectives(
            plan_state,
            llm_call=chat_llm,
            search=search,
            fulltext=None,
            skill_context=skill_context,
        )
        perspectives = plan_state.get("perspectives") or ["通用视角"]

        def search_fn(query: str, top_k: int):
            results = search(query)  # rm.forward(query) → 统一格式 list
            return [dict(item) for item in (results or [])]

        parallel = run_parallel_perspectives(
            topic,
            perspectives,
            llm_call=chat_llm,
            search=search_fn,
            max_workers=3,
            top_k=3,
        )
        sections = [
            "## {0}\n\n{1}".format(s["perspective"], s["paragraph"])
            for s in parallel["sections"]
            if s["paragraph"]
        ]
        article = "\n\n".join(sections) or "（并行调研未产出内容）"
        (output_dir / ARTICLE_FILENAME).write_text(article, encoding="utf-8")
        (output_dir / "research_parallel_result.json").write_text(
            json.dumps(parallel, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        state["result_summary"] = {
            "loop": "parallel-perspectives",
            "retriever": retriever_used,
            "skill_injected": bool(skill_context),
            "perspectives": perspectives,
            "parallel_workers": parallel["parallel"]["workers"],
            "failed_views": len(parallel["errors"]),
            "converged": parallel["converged"],
        }
        return {
            "article": article,
            "perspectives": perspectives,
            "scorecard": {"converged": parallel["converged"]},
            "qc_passed": parallel["converged"],
            "citation_pool": parallel["evidence"],
        }

    def _state_path(self, task_id: str):
        # 安全校验: task_id 必须是 32 位 hex（uuid4().hex 的格式），
        # 防止 "../../etc/passwd" 这类路径遍历注入（主流安全实践: 入口白名单校验）
        if not isinstance(task_id, str) or not re.fullmatch(r"[0-9a-f]{32}", task_id):
            raise KeyError("Invalid task_id: {0!r}".format(task_id))
        return self.tasks_dir / "{0}.json".format(task_id)

    def _read_state(self, task_id: str):
        path = self._state_path(task_id)
        if not path.exists():
            raise KeyError("Unknown task_id: {0}".format(task_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_state(self, task_id: str, state: Dict):
        # 原子写（主流做法）: 先写临时文件再 os.replace，
        # 避免多线程/多 worker 并发写同一任务文件时写坏 JSON
        path = self._state_path(task_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(_redact(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _list_tasks_by_status(self, status: str):
        tasks = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == status:
                tasks.append(state)
        return sorted(
            tasks,
            key=lambda item: (
                int(item.get("queue_index", 0)),
                item.get("created_at", ""),
            ),
        )

    def _next_queue_index(self):
        return len(list(self.tasks_dir.glob("*.json"))) + 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact_secret(key, _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_secret(key, value):
    lowered = str(key).lower()
    if lowered in {"api_key", "apikey", "access_key", "secret_key"}:
        return "***REDACTED***"
    if (
        "token" in lowered
        or "secret" in lowered
        or "password" in lowered
        or lowered.endswith("_key")
    ):
        return "***REDACTED***"
    return value


def _redact_error(message: str):
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***REDACTED***", message)


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _read_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "decode_error", "raw": line})
    return events
