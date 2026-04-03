"""
用户意图规划工作流（LangGraph）

用 LangChain 生态的 **LangGraph** 将「天气子智能体 → LLM 规划 → 规则编排」固化为状态图，
与原先 `UserIntentAgent._plan` 顺序与语义一致，便于可视化扩展与维护。

说明：子智能体业务逻辑仍在各 `*Agent` 类中；本模块只负责 **编排**，不替代路况/路径等规则实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph


class IntentPlanState(TypedDict, total=False):
    """LangGraph 在节点间传递的状态（单轮规划）。"""

    message: str
    history: List[Dict[str, Any]]
    rag_hits: List[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    llm_attempted: bool


def compile_user_intent_planning_graph(user_intent_agent: Any):
    """
    编译意图规划图。`user_intent_agent` 为已构造的 `UserIntentAgent` 实例（避免循环 import 不写类型注解）。
    """
    agent = user_intent_agent

    def node_weather(state: IntentPlanState) -> Dict[str, Any]:
        msg = state.get("message") or ""
        hist = state.get("history") or []
        wx = agent._weather_dialog.try_plan(msg, hist)
        if wx is not None:
            plan = agent._post_process_llm_plan(wx, msg, hist)
            return {"plan": plan}
        return {}

    def node_llm(state: IntentPlanState) -> Dict[str, Any]:
        if not agent._svc.llm_enabled or agent._svc.llm is None:
            return {}
        msg = state.get("message") or ""
        hist = state.get("history") or []
        rag = state.get("rag_hits") or []
        try:
            llm_plan = agent._plan_by_llm(msg, hist, rag)
            llm_plan = agent._post_process_llm_plan(llm_plan, msg, hist)
            if agent._is_usable_plan(llm_plan):
                return {"plan": llm_plan, "llm_attempted": True}
        except Exception:
            pass
        return {"llm_attempted": True}

    def node_rules(state: IntentPlanState) -> Dict[str, Any]:
        msg = state.get("message") or ""
        hist = state.get("history") or []
        rule_plan = agent._plan_by_rules(msg, hist)
        if agent._svc.llm_enabled and bool(state.get("llm_attempted")):
            rule_plan = {**rule_plan, "used_llm": True}
        return {"plan": rule_plan}

    def node_finalize(state: IntentPlanState) -> Dict[str, Any]:
        return {}

    def route_after_weather(s: IntentPlanState) -> str:
        return "finalize" if s.get("plan") is not None else "llm"

    def route_after_llm(s: IntentPlanState) -> str:
        return "finalize" if s.get("plan") is not None else "rules"

    graph = StateGraph(IntentPlanState)
    graph.add_node("weather", node_weather)
    graph.add_node("llm", node_llm)
    graph.add_node("rules", node_rules)
    graph.add_node("finalize", node_finalize)
    graph.set_entry_point("weather")
    graph.add_conditional_edges(
        "weather",
        route_after_weather,
        {"finalize": "finalize", "llm": "llm"},
    )
    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"finalize": "finalize", "rules": "rules"},
    )
    graph.add_edge("rules", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
