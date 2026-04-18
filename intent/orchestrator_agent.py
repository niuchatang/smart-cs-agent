"""
调度中枢智能体（Intent Orchestrator Agent）

统一编排子智能体调用顺序：

1. 路况：显式高速编号
2. 通用：天气规则
3. 路况：肯定答复 / 起终点+路况（走廊）
4. 扩展：`AgentRegistry` 注册的可插拔子智能体（ETC / 服务区 / 出发时间 /
   事故告警 / 天气影响 / 无障碍 / 澄清 等）。放在纯路径规划**之前**，
   避免「成都到重庆的最佳出行时间」这类含 OD 的垂直领域提问被路径规划
   以 OD 为由抢先返回；扩展链内部按 priority 升序尝试。
5. 路径：纯路径规划
6. 路况：历史路线高速、泛路况等
7. 通用：公交实时、票价、工单、unknown。

天气多轮对话仍由 WeatherDialogAgent 在 UserIntentAgent 最前拦截（不经本调度器规则链）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_registry import AgentRegistry, default_extension_factories
from .general_intent_agent import GeneralIntentAgent
from .road_condition_agent import RoadConditionAgent
from .route_planning_agent import RoutePlanningAgent


class IntentOrchestratorAgent:
    def __init__(
        self,
        service_agent: Any,
        *,
        registry: Optional[AgentRegistry] = None,
        enable_extensions: bool = True,
    ) -> None:
        self._svc = service_agent
        self._road = RoadConditionAgent(service_agent)
        self._route = RoutePlanningAgent(service_agent)
        self._general = GeneralIntentAgent(service_agent)
        self._registry = registry if registry is not None else AgentRegistry()
        if enable_extensions and registry is None:
            for factory in default_extension_factories():
                try:
                    self._registry.register(factory(service_agent))
                except Exception:
                    continue

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

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
        p = self._registry.try_plan(message, hist)
        if p is not None:
            return p
        p = self._route.try_rule_plan(message, hist)
        if p is not None:
            return p
        p = self._road.try_late_corridor(message, hist)
        if p is not None:
            return p
        return self._general.try_tail_rules(message, hist)
