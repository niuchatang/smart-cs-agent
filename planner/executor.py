from __future__ import annotations

from typing import Any, Dict, List

from .models import Plan


class TaskExecutor:
    """将 Planner 产出的 Plan 转为 main 可执行的标准 plan dict。"""

    def to_executable_plan(self, plan: Plan, *, confidence: float = 0.9) -> Dict[str, Any]:
        base = plan.to_plan_dict(confidence=confidence)
        base["meta"]["decomposed_agents"] = self._decomposed_view(plan)
        return base

    @staticmethod
    def _decomposed_view(plan: Plan) -> List[Dict[str, Any]]:
        """给用户/日志展示的逻辑子 Agent 视图（可与实际 tool 合并执行）。"""
        goal = plan.goal
        if any(t.tool == "query_travel_decision" for t in plan.tasks):
            return [
                {"agent": "RouteAgent", "status": "bundled"},
                {"agent": "WeatherAgent", "status": "bundled"},
                {"agent": "AQIAgent", "status": "bundled"},
                {"agent": "GISAgent", "status": "bundled"},
                {"agent": "RiskAgent", "status": "bundled"},
                {"goal": goal},
            ]
        return [{"agent": t.agent, "tool": t.tool} for t in plan.tasks]

    def merge_tool_results(self, tool_results: List[Dict[str, Any]], plan: Plan) -> List[Dict[str, Any]]:
        if not plan.tasks:
            return tool_results
        tagged = list(tool_results)
        for tr in tagged:
            tr.setdefault("planner_goal", plan.goal)
        return tagged
