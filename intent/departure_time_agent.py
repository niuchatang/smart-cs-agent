"""
最佳出发时间智能体（Departure Time Agent）

- 基于规则的通勤早晚高峰 + 节假日常识，给出「什么时候走更顺」的建议；
- 不触达外部实时路况 API（已有 query_highway_condition 可配合使用），
  仅返回建议文本，后续用户可再点击路况芯片获取更精准的当下情况；
- 归入 `route_planning` 意图。

触发词：「几点出发」「什么时候走」「早高峰」「晚高峰」「错峰」「出发时间」。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TRIGGERS = (
    "几点出发",
    "什么时候走",
    "什么时候出发",
    "出发时间",
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

        lines: List[str] = []
        lines.append("出发时间建议（经验规则，具体以当天实时路况为准）：")
        lines.append(f"- 工作日：{_WEEKDAY_PEAK}")
        lines.append(f"- 周末：{_WEEKEND_PEAK}")
        lines.append(f"- 节假日：{_HOLIDAY_PEAK}")
        lines.append("- 长途自驾：建议 5:30-6:30 出发，可避开绝大多数走廊第一波拥堵。")

        if prefer_arrival:
            lines.append(
                "\n如果你有到达时间限制，请告诉我希望几点到以及起终点，我会按平均车速倒推建议出发时间。"
            )
        else:
            lines.append("\n需要我再补一条路线的高速实时路况（「沿途高速路况」）来辅助决策吗？")

        return {
            "intent": "route_planning",
            "confidence": 0.78,
            "actions": [],
            "llm_reply": "\n".join(lines),
            "used_llm": False,
        }
