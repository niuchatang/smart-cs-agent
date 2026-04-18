"""
可插拔子智能体注册表（Agent Registry）

- 每个扩展子智能体声明：`name / priority / try_plan(message, history)`；
- `plan_rules_extended` 在现有 orchestrator 规则链尾部之前按优先级尝试扩展智能体；
- 不改变既有 WeatherDialog / RoadCondition / RoutePlanning / General 的默认顺序；
- 所有扩展 Agent 返回 Optional[Dict]，None 表示不接管；命中则直接返回该计划。

约定：
- priority 数值越小越先尝试（与 orchestrator 早期分支一致）；
- 同优先级按注册顺序；
- Agent 可读写 `service_agent`（主智能体）暴露的辅助函数（`_clean_place`、
  `_extract_last_route_cities_from_history` 等），但不应直接改写主智能体状态；
- 扩展返回的 intent 必须仍在 main.IntentType Literal 内，否则主进程 Pydantic 校验会失败。
  新业务主题（如 ETC/服务区）请复用既有 intent（如 fare_policy / route_planning 等），
  通过 `llm_reply` 或 `actions` 承载差异。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol


class ExtensionAgent(Protocol):
    """扩展智能体协议：最小只需实现 try_plan。"""

    name: str
    priority: int

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]: ...


class AgentRegistry:
    """扩展子智能体注册表（单进程，线程不安全；仅在 orchestrator 构造期写入）。"""

    def __init__(self) -> None:
        self._items: List[ExtensionAgent] = []

    def register(self, agent: ExtensionAgent) -> None:
        self._items.append(agent)
        self._items.sort(key=lambda a: getattr(a, "priority", 100))

    def extend(self, agents: List[ExtensionAgent]) -> None:
        for a in agents:
            self.register(a)

    def agents(self) -> List[ExtensionAgent]:
        return list(self._items)

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for a in self._items:
            try:
                plan = a.try_plan(message, history)
            except Exception:
                plan = None
            if plan is not None:
                plan.setdefault("used_llm", False)
                if not isinstance(plan.get("meta"), dict):
                    plan["meta"] = {}
                plan["meta"]["ext_agent"] = getattr(a, "name", a.__class__.__name__)
                return plan
        return None


AgentFactory = Callable[[Any], ExtensionAgent]


def default_extension_factories() -> List[AgentFactory]:
    """默认扩展集合：惰性导入，避免启动时循环引用。

    顺序（priority）说明见各 Agent 内常量；大致为：
    20 ETCCharge → 30 ServiceArea → 35 DepartureTime → 40 TrafficIncident →
    50 WeatherImpact → 60 Accessibility → 85 Clarify（未分类兜底最后兜）
    """

    def _etc(svc: Any):
        from .etc_charge_agent import ETCChargeAgent

        return ETCChargeAgent(svc)

    def _sa(svc: Any):
        from .service_area_agent import ServiceAreaAgent

        return ServiceAreaAgent(svc)

    def _dt(svc: Any):
        from .departure_time_agent import DepartureTimeAgent

        return DepartureTimeAgent(svc)

    def _ti(svc: Any):
        from .traffic_incident_agent import TrafficIncidentAgent

        return TrafficIncidentAgent(svc)

    def _wi(svc: Any):
        from .weather_impact_agent import WeatherImpactAgent

        return WeatherImpactAgent(svc)

    def _acc(svc: Any):
        from .accessibility_agent import AccessibilityAgent

        return AccessibilityAgent(svc)

    def _clarify(svc: Any):
        from .clarify_agent import ClarifyAgent

        return ClarifyAgent(svc)

    return [_etc, _sa, _dt, _ti, _wi, _acc, _clarify]
