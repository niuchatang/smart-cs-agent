"""
最佳出发时间智能体（Departure Time Agent）

- 基于规则的通勤早晚高峰 + 节假日常识，给出「什么时候走更顺」的建议；
- 不触达外部实时路况 API（已有 query_highway_condition 可配合使用），
  仅返回建议文本，后续用户可再点击路况芯片获取更精准的当下情况；
- 会读取当前消息或历史中最近一次成功路径规划的 OD，针对该走廊给出更具体建议；
- 归入 `route_planning` 意图。

触发词：「几点出发」「什么时候走」「最佳出行时间」「最佳出发时间」
「出发时间」「错峰」「早高峰」「晚高峰」等。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_TRIGGERS = (
    "最佳出行时间",
    "最佳出发时间",
    "最佳时间",
    "最优出行时间",
    "最合适的时间",
    "最合适时间",
    "出行时间",
    "出发时间",
    "何时出发",
    "几点出发",
    "几点走",
    "什么时候走",
    "什么时候出发",
    "什么时候动身",
    "错峰",
    "早高峰",
    "晚高峰",
    "堵不堵",
    "现在走合适吗",
    "现在出发",
    "早点走",
    "晚点走",
)

_WEEKDAY_PEAK = "周一至周五 7:00-9:30、17:00-19:30 一般为城市早晚高峰，出入城主干道与环线可能较慢。"
_WEEKEND_PEAK = "周末早 9:00-11:00 出城、下午 15:00-19:00 返城时段易拥堵，短途出游建议错开。"
_HOLIDAY_PEAK = "节假日首日 7-10 点出城、末日 15-20 点返程最集中，建议提前半天或错后 2 小时。"


class DepartureTimeAgent:
    name = "departure_time"
    priority = 35

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

        prefer_arrival = bool(re.search(r"(?:几点|哪个点|什么时候).{0,6}(?:到|抵达)", text))

        origin, destination = self._resolve_od(text, history)
        od_label = f"{origin} → {destination}" if origin and destination else ""

        lines: List[str] = []
        if od_label:
            lines.append(f"针对 {od_label} 的出发时间建议（经验规则，具体以当天实时路况为准）：")
        else:
            lines.append("出发时间建议（经验规则，具体以当天实时路况为准）：")
        lines.append(f"- 工作日：{_WEEKDAY_PEAK}")
        lines.append(f"- 周末：{_WEEKEND_PEAK}")
        lines.append(f"- 节假日：{_HOLIDAY_PEAK}")
        lines.append("- 长途自驾：建议 5:30-6:30 出发，可避开绝大多数走廊第一波拥堵。")

        if od_label:
            lines.append(
                f"\n如果你希望对 {od_label} 更精准，我可以再帮你查一次「沿途高速实时路况」或"
                "「途经城市天气」，告诉我即可。"
            )
        elif prefer_arrival:
            lines.append(
                "\n如果你有到达时间限制，请告诉我希望几点到以及起终点，我会按平均车速倒推建议出发时间。"
            )
        else:
            lines.append(
                "\n告诉我起终点（例如「成都到重庆」）我可以再结合里程给出更贴合的出发时间建议。"
            )

        return {
            "intent": "route_planning",
            "confidence": 0.82 if od_label else 0.75,
            "actions": [],
            "llm_reply": "\n".join(lines),
            "used_llm": False,
        }

    def _resolve_od(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """优先用当前消息显式 OD；否则回溯历史里最近一次成功路径规划的城市序列。"""
        svc = self._svc
        origin = destination = ""
        extractor = getattr(svc, "_extract_route_endpoints", None)
        if callable(extractor):
            try:
                origin, destination = extractor(message)
            except Exception:
                origin = destination = ""
        if origin and destination:
            return origin, destination

        hist_extractor = getattr(svc, "_extract_last_route_cities_from_history", None)
        if callable(hist_extractor):
            try:
                cities = hist_extractor(history) or []
            except Exception:
                cities = []
            if isinstance(cities, list) and len(cities) >= 2:
                return str(cities[0]), str(cities[-1])

        for item in reversed(history or []):
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            txt = str(item.get("content", ""))
            if not txt:
                continue
            try:
                o, d = extractor(txt) if callable(extractor) else ("", "")
            except Exception:
                o = d = ""
            if o and d:
                return o, d
        return "", ""
