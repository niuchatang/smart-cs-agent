"""
路况智能体（Road Condition Agent）

负责高速/道路路况、起终点走廊路况（先 query_route_plan 再查沿途高速）等。
与路径规划智能体分离，由 IntentOrchestratorAgent 按与原规则一致的顺序调度。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RoadConditionAgent:
    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_explicit_highway(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        explicit_highway_target = self._svc._extract_highway_target(message)
        if not explicit_highway_target:
            return None
        route_points = self._svc._extract_last_route_probe_points_from_history(history)
        direct_highway_params: Dict[str, Any] = {"target": explicit_highway_target}
        if route_points:
            direct_highway_params["context_points"] = route_points
        return {
            "intent": "highway_condition",
            "confidence": 0.92,
            "actions": [{"tool": "query_highway_condition", "params": direct_highway_params}],
            "used_llm": False,
        }

    def try_mid_od_corridor(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """肯定答复高速编号、起终点+路况（先于纯路径规划分支）。"""
        text = message.lower()
        route_points = self._svc._extract_last_route_probe_points_from_history(history)
        recent_highway_codes = self._svc._extract_last_highway_codes_from_history(history)

        if self._svc._is_affirmative_message(message) and recent_highway_codes:
            if len(recent_highway_codes) > 1:
                return {
                    "intent": "highway_condition",
                    "confidence": 0.88,
                    "actions": [],
                    "llm_reply": (
                        f"可以。你上一条涉及多条高速：{', '.join(recent_highway_codes[:4])}。"
                        "请告诉我要展开哪一条（例如：查看G30详情）。"
                    ),
                    "used_llm": False,
                }
            highway_params: Dict[str, Any] = {"target": recent_highway_codes[0]}
            if route_points:
                highway_params["context_points"] = route_points
            return {
                "intent": "highway_condition",
                "confidence": 0.9,
                "actions": [{"tool": "query_highway_condition", "params": highway_params}],
                "used_llm": False,
            }

        multi = self._svc._extract_multi_stop_places(message)
        if multi and len(multi) >= 3:
            origin, destination = multi[0], multi[-1]
        else:
            origin, destination = self._svc._extract_route_endpoints(message)
        road_query = any(k in text for k in ["高速", "事故", "管制", "封闭", "拥堵", "路况"])
        if road_query and origin and destination:
            rp1: Dict[str, Any] = {"origin": origin, "destination": destination, "mode": "driving"}
            if multi and len(multi) >= 3:
                rp1["waypoints"] = multi[1:-1]
            return {
                "intent": "highway_condition",
                "confidence": 0.91,
                "actions": [{"tool": "query_route_plan", "params": rp1}],
                "od_traffic_followup": True,
                "used_llm": False,
            }
        return None

    def try_late_corridor(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """纯路径规划之后剩余的路况分支（历史路线高速、泛路况等）。"""
        text = message.lower()
        explicit_target = self._svc._extract_highway_target(message)
        history_highway_target = self._svc._extract_last_highway_from_history(history)
        route_highways = self._svc._extract_last_route_highways_from_history(history)
        route_points = self._svc._extract_last_route_probe_points_from_history(history)
        road_query = any(k in text for k in ["高速", "事故", "管制", "封闭", "拥堵", "路况"])

        if road_query and explicit_target:
            params: Dict[str, Any] = {"target": explicit_target}
            if route_points:
                params["context_points"] = route_points
            return {
                "intent": "highway_condition",
                "confidence": 0.91,
                "actions": [{"tool": "query_highway_condition", "params": params}],
                "used_llm": False,
            }
        if road_query and route_highways:
            highway_actions: List[Dict[str, Any]] = []
            for hw in route_highways[:4]:
                params = {"target": hw}
                if route_points:
                    params["context_points"] = route_points
                highway_actions.append({"tool": "query_highway_condition", "params": params})
            return {
                "intent": "highway_condition",
                "confidence": 0.9,
                "actions": highway_actions,
                "used_llm": False,
            }
        if road_query and history_highway_target:
            params = {"target": history_highway_target}
            if route_points:
                params["context_points"] = route_points
            return {
                "intent": "highway_condition",
                "confidence": 0.86,
                "actions": [{"tool": "query_highway_condition", "params": params}],
                "used_llm": False,
            }
        if road_query:
            return {
                "intent": "highway_condition",
                "confidence": 0.78,
                "actions": [],
                "used_llm": False,
            }
        return None
