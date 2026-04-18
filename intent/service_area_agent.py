"""
服务区 / 充电桩 / 加油智能体（Service Area Agent）

- 依托既有 `query_route_plan` 返回的 `highways` / `cities_along_route` 给出通用建议；
- 不对外拉真实 POI（避免占用配额与未授权内部接口）。若想接真实数据，可在后续
  把 `_mock_service_area_hint` 替换为 tool 调用，外层结构不变；
- 触发词：「服务区」「充电桩」「充电」「加油」「歇一下」「休息区」。

意图归属：`route_planning`（复用现有意图），差异通过 `llm_reply` 表达。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

_SA_KW = ("服务区", "休息区", "加油站", "充电桩", "充电", "加油", "歇一下", "歇脚", "停一下")


def _mock_service_area_hint(highway: str, city: str) -> str:
    kinds = ["综合型服务区（餐饮/卫生间/便利店）", "含新能源充电（直流快充）", "有加油站与维修点"]
    return f"{highway}·{city}附近 — {random.choice(kinds)}"


class ServiceAreaAgent:
    name = "service_area"
    priority = 30

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text or not any(k in text for k in _SA_KW):
            return None

        rc: List[str] = []
        hw: List[str] = []
        if hasattr(self._svc, "_extract_last_route_cities_from_history"):
            rc = list(self._svc._extract_last_route_cities_from_history(history) or [])
        if hasattr(self._svc, "_extract_last_route_highways_from_history"):
            hw = list(self._svc._extract_last_route_highways_from_history(history) or [])

        want_ev = any(k in text for k in ("充电", "充电桩", "新能源"))
        want_fuel = any(k in text for k in ("加油", "加油站"))

        if rc:
            pool = hw[:4] if hw else ["主要高速"]
            lines: List[str] = []
            random.seed(len(rc) + len(text))
            for i, c in enumerate(rc[:6]):
                base = _mock_service_area_hint(pool[i % len(pool)], c)
                if want_ev:
                    base += "；建议优先选带「快充」图标的服务区。"
                elif want_fuel:
                    base += "；建议在油量剩 1/4 前预加满。"
                lines.append(f"- {base}")
            block = "\n".join(lines)
            reply = (
                f"沿最近一次路线，给你一份服务区建议（示意，供规划参考，真实 POI 请以导航 App 为准）：\n{block}\n\n"
                "如需我更具体到某段高速（例如「G40 上的充电桩」），告诉我编号即可。"
            )
            conf = 0.86
        else:
            reply = (
                "要查服务区/充电/加油，请先告诉我起终点（例如「南京到合肥」），"
                "或复用最近一次路径规划结果；之后我会按途经高速与城市给出分段建议。"
            )
            conf = 0.7

        return {
            "intent": "route_planning",
            "confidence": conf,
            "actions": [],
            "llm_reply": reply,
            "used_llm": False,
        }
