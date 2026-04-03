"""
路径规划智能体（Route Planning Agent）

识别「从 A 到 B 怎么走」等纯规划诉求，产出 route_planning + query_route_plan。
起终点+路况由路况智能体优先处理，本智能体在调度中枢中在其后执行。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RoutePlanningAgent:
    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_rule_plan(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        text = message.lower()
        multi = self._svc._extract_multi_stop_places(message)
        if multi and len(multi) >= 3:
            origin, destination = multi[0], multi[-1]
        else:
            origin, destination = self._svc._extract_route_endpoints(message)
        road_query = any(k in text for k in ["高速", "事故", "管制", "封闭", "拥堵", "路况"])
        if road_query and origin and destination:
            return None
        has_route_keywords = any(
            k in text for k in ["怎么走", "路线", "换乘", "导航", "到达", "到...怎么去", "出行方案"]
        )
        if (origin and destination) or has_route_keywords:
            actions: List[Dict[str, Any]] = []
            if origin and destination:
                rp2: Dict[str, Any] = {"origin": origin, "destination": destination, "mode": "driving"}
                if multi and len(multi) >= 3:
                    rp2["waypoints"] = multi[1:-1]
                actions.append({"tool": "query_route_plan", "params": rp2})
            return {
                "intent": "route_planning",
                "confidence": 0.9 if origin and destination else 0.84,
                "actions": actions,
                "used_llm": False,
            }
        return None
