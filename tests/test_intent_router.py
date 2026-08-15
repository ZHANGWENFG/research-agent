"""意图路由测试：规则层（离线、不花钱）。

意图枚举：casual_chat | system_help | research_qa | run_research | clarify
"""
import pytest

from research_agent.research_intent_router import route_by_rules


def _route(message, topic="", history=None):
    return route_by_rules(message, {"topic": topic}, history or [])


def test_research_keyword_detected():
    decision = _route("帮我调研 LangGraph 和 LangChain 的区别", topic="LangGraph")
    assert decision.get("intent") in ("research_qa", "run_research")


def test_knowledge_question_detected():
    # 知识问题需要领域关键词（rag/pim/神经网络等）+ 疑问词
    decision = _route("RAG 是什么？能解决什么问题？", topic="RAG")
    assert decision.get("intent") in ("research_qa", "run_research")


def test_direct_research_request_detected():
    decision = _route("帮我调研一下 RAG 的原理和实现", topic="RAG")
    assert decision.get("intent") == "run_research"


def test_casual_chat_detected():
    decision = _route("你好")
    assert decision.get("intent") == "casual_chat"


def test_system_help_detected():
    decision = _route("你是谁？你能做什么？")
    assert decision.get("intent") == "system_help"


def test_router_output_has_confidence_and_reason():
    decision = _route("帮我调研一下记忆系统", topic="记忆")
    assert "confidence" in decision
    assert "reason" in decision or "rationale" in decision
