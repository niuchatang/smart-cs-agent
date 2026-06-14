"""
Planner LangGraph：memory 上下文 → 是否规划 → 生成任务 → 聚合为 plan。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from memory.models import MemoryContext


class PlannerState(TypedDict, total=False):
    message: str
    enriched_message: str
    history: List[Dict[str, Any]]
    user_id: str
    memory_context: Optional[MemoryContext]
    use_planner: bool
    plan: Optional[Dict[str, Any]]
    planner_goal: str


def compile_planner_decision_graph(planner_agent: Any, memory_agent: Any):
    """轻量决策图：加载记忆、补全消息、判断是否走 Planner。"""

    def node_memory_load(state: PlannerState) -> Dict[str, Any]:
        ctx = state.get("memory_context")
        if ctx is None:
            uid = str(state.get("user_id") or "")
            if uid and memory_agent is not None:
                ctx = memory_agent.load_context(uid)
        return {"memory_context": ctx}

    def node_enrich(state: PlannerState) -> Dict[str, Any]:
        msg = state.get("message") or ""
        ctx = state.get("memory_context")
        enriched = msg
        if ctx is not None and memory_agent is not None:
            enriched = memory_agent.enrich_message(msg, ctx)
        return {"enriched_message": enriched}

    def node_planner(state: PlannerState) -> Dict[str, Any]:
        msg = state.get("message") or ""
        hist = state.get("history") or []
        ctx = state.get("memory_context")
        if not planner_agent.should_plan(msg, hist, ctx):
            return {"use_planner": False}
        plan = planner_agent.run(msg, hist, ctx)
        if plan is None:
            return {"use_planner": False}
        goal = str((plan.get("meta") or {}).get("goal") or "")
        return {"use_planner": True, "plan": plan, "planner_goal": goal}

    graph = StateGraph(PlannerState)
    graph.add_node("memory_load", node_memory_load)
    graph.add_node("enrich", node_enrich)
    graph.add_node("planner", node_planner)
    graph.set_entry_point("memory_load")
    graph.add_edge("memory_load", "enrich")
    graph.add_edge("enrich", "planner")
    graph.add_edge("planner", END)
    return graph.compile()
