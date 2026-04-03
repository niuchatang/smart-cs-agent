"""
调度中枢智能体（Intent Orchestrator Agent）

统一编排子智能体调用顺序，保持与原 UserIntentAgent._plan_by_rules 等价的优先级：

1. 路况：显式高速编号
2. 通用：天气规则
3. 路况：肯定答复 / 起终点+路况（走廊）
4. 路径：纯路径规划
5. 路况：历史路线高速、泛路况等
6. 通用：公交实时、票价、工单、unknown

天气多轮对话仍由 WeatherDialogAgent 在 UserIntentAgent 最前拦截（不经本调度器规则链）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .general_intent_agent import GeneralIntentAgent
from .road_condition_agent import RoadConditionAgent
from .route_planning_agent import RoutePlanningAgent


class IntentOrchestratorAgent:
    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent
        self._road = RoadConditionAgent(service_agent)
        self._route = RoutePlanningAgent(service_agent)
        self._general = GeneralIntentAgent(service_agent)

    def plan_rules(self, message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        hist = history or []
        p = self._road.try_explicit_highway(message, hist)
        if p is not None:
            return p
        p = self._general.try_weather_rules(message, hist)
        if p is not None:
            return p
        p = self._road.try_mid_od_corridor(message, hist)
        if p is not None:
            return p
        p = self._route.try_rule_plan(message, hist)
        if p is not None:
            return p
        p = self._road.try_late_corridor(message, hist)
        if p is not None:
            return p
        return self._general.try_tail_rules(message, hist)
