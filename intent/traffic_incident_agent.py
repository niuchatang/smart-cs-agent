"""
突发事件 / 事故告警智能体（Traffic Incident Agent）

- 作用：当用户问「有没有事故/封路/交通管制/最新路况公告」等，
  直接复用 `query_highway_condition` 工具的能力，按历史路径途经高速批量拉取；
- 与 RoadConditionAgent 的差异：
  - RoadConditionAgent 依赖用户显式说「高速/路况」+ 起终点；
  - 本智能体面向更泛的「告警」「突发」「管制公告」「事故报告」等口语；
- 若历史没有路线，降级到询问起终点或高速编号。

意图归属：`highway_condition`。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_TRIGGERS = (
    "有没有事故",
    "事故公告",
    "事故报告",
    "突发",
    "最新路况",
    "最新事故",
    "最新管制",
    "临时管制",
    "封路",
    "封闭公告",
    "应急",
    "改道",
    "绕行",
    "临时绕行",
    "注意事项",
)


class TrafficIncidentAgent:
    name = "traffic_incident"
    priority = 40

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        if not any(k in text for k in _TRIGGERS):
            return None

        explicit_target = None
        if hasattr(self._svc, "_extract_highway_target"):
            explicit_target = self._svc._extract_highway_target(text)
        route_highways: List[str] = []
        route_points: List[Dict[str, Any]] = []
        if hasattr(self._svc, "_extract_last_route_highways_from_history"):
            route_highways = list(
                self._svc._extract_last_route_highways_from_history(history) or []
            )
        if hasattr(self._svc, "_extract_last_route_probe_points_from_history"):
            route_points = list(
                self._svc._extract_last_route_probe_points_from_history(history) or []
            )

        actions: List[Dict[str, Any]] = []
        if explicit_target:
            params: Dict[str, Any] = {"target": explicit_target}
            if route_points:
                params["context_points"] = route_points
            actions.append({"tool": "query_highway_condition", "params": params})
        elif route_highways:
            for hw in route_highways[:4]:
                params = {"target": hw}
                if route_points:
                    params["context_points"] = route_points
                actions.append({"tool": "query_highway_condition", "params": params})
        else:
            return {
                "intent": "highway_condition",
                "confidence": 0.72,
                "actions": [],
                "llm_reply": (
                    "可以帮你查事故/管制类突发公告。请告诉我高速编号（如「G2」「G40」）"
                    "或先规划路线，我会按沿途高速逐条汇总。"
                ),
                "used_llm": False,
            }

        return {
            "intent": "highway_condition",
            "confidence": 0.89,
            "actions": actions,
            "used_llm": False,
        }
