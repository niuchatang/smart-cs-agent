"""
认知编排层：Memory → Planner → 既有 Intent LangGraph。

在不改动各子 Agent 实现的前提下，为单轮 parse 增加长期记忆与多任务规划能力。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from memory import MemoryAgent
from memory.models import MemoryContext
from planner import PlannerAgent
from planner.planning_graph import compile_planner_decision_graph


class CognitiveOrchestrator:
    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent
        self.memory = MemoryAgent(service_agent)
        self.planner = PlannerAgent(service_agent)
        self.enabled = os.getenv("COGNITIVE_ENABLE", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._graph: Any = None

    def _ensure_graph(self) -> Any:
        if self._graph is None:
            self._graph = compile_planner_decision_graph(self.planner, self.memory)
        return self._graph

    def parse(
        self,
        message: str,
        history: List[Dict[str, Any]],
        rag_hits: List[Dict[str, Any]],
        *,
        user_id: str = "",
    ) -> Dict[str, Any]:
        if not self.enabled or not user_id:
            return self._svc.intent_agent.parse(message, history, rag_hits)

        ctx = self.memory.load_context(user_id)
        enriched = self.memory.enrich_message(message, ctx)
        out = self._ensure_graph().invoke(
            {
                "message": message,
                "enriched_message": enriched,
                "history": history or [],
                "memory_context": ctx,
                "user_id": user_id,
            }
        )
        if out.get("use_planner") and isinstance(out.get("plan"), dict):
            plan = dict(out["plan"])
            plan.setdefault("meta", {})
            plan["meta"]["memory_context"] = self.memory.context_prompt(ctx)
            if ctx.proactive_hint:
                plan["meta"]["proactive_hint"] = ctx.proactive_hint
            return plan

        plan = self._svc.intent_agent.parse(enriched, history, rag_hits)
        plan.setdefault("meta", {})
        mem_prompt = self.memory.context_prompt(ctx)
        if mem_prompt:
            plan["meta"]["memory_context"] = mem_prompt
        if ctx.proactive_hint:
            plan["meta"]["proactive_hint"] = ctx.proactive_hint
        return plan

    def after_turn(
        self,
        user_id: str,
        message: str,
        plan: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        reply: str = "",
    ) -> None:
        if not user_id:
            return
        self.memory.extract_and_save(user_id, message, plan, tool_results, reply=reply)
