"""停车场查询智能体（Parking Lot Agent）

SPEC：docs/specs/parking_lot_agent.md

设计要点（与 ServiceAreaAgent 对齐，便于读者类比）：
- 触发词命中后，优先复用历史路线的 `cities_along_route`，给「沿途分段」建议；
- 无路线上下文时，从 message 抽取地点（"X 到 Y"或"X 附近"等）给单点建议；
- 都没有时返回引导性回复（仍命中本 Agent，confidence 调低）；
- 不联网真实 POI，预留 `_mock_parking_hint` 钩子，后续可换成 tool 调用而不动外层结构。

意图归属：复用 `route_planning`（同 ServiceAreaAgent），通过 `llm_reply` 表达差异；
新增 IntentType 会破坏 main.py Pydantic 校验，故不引入。
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

_PARKING_KW = (
    "停车场",
    "停车位",
    "停个车",
    "停车",
    "车位",
    "泊车",
    "停哪",
    "停在",
)

_EV_KW = ("充电", "充电桩", "新能源", "电动车")
_INDOOR_KW = ("地下", "室内", "地库", "室内停车")
_FREE_KW = ("免费", "不收费")
# 显式"沿途"意图词；命中则即使消息里也提到了某个具体地名，也走沿途分段
_ALONG_KW = ("沿途", "沿路", "沿线", "路上", "途中", "一路上", "一路", "路途")

# "X 附近 / 旁边 / 周围" 的地点抽取
_NEAR_PLACE_RE = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9]{2,15}?)\s*(?:附近|旁边|周边|周围|那边)"
)


def _mock_parking_hint(place: str, *, ev: bool, indoor: bool, free: bool) -> str:
    """单条停车场示意。后续可替换为 tool 调用，签名保持一致。"""
    kinds = [
        "综合型停车场（24h 开放，付费）",
        "地面临时停车区（短停免费 15 分钟内）",
        "地下停车场（遮蔽防晒，付费）",
    ]
    if ev:
        kinds.append("含新能源充电桩（直流快充）")
    if indoor:
        kinds = [k for k in kinds if "地下" in k or "室内" in k] or [
            "地下停车场（遮蔽防晒，付费）"
        ]
    if free:
        kinds.append("免费停车场（车位有限，建议早到）")
    return f"{place} — {random.choice(kinds)}"


class ParkingLotAgent:
    """停车场查询扩展智能体。"""

    name = "parking_lot"
    priority = 32  # 紧随 ServiceArea(30) 之后，先于 DepartureTime(35)

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    # -- public ---------------------------------------------------------------

    def try_plan(
        self,
        message: str,
        history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text or not any(k in text for k in _PARKING_KW):
            return None

        hist = history if isinstance(history, list) else []
        want_ev = any(k in text for k in _EV_KW)
        want_indoor = any(k in text for k in _INDOOR_KW)
        want_free = any(k in text for k in _FREE_KW)
        want_along = any(k in text for k in _ALONG_KW)

        cities = self._extract_route_cities(hist)
        place = self._extract_single_place(text)

        # 优先级（按意图明确度由高到低）：
        # 1. 显式"沿途"语义 + 历史有路线 → 沿途分段
        # 2. 消息里抽到具体地点 → 单点查询（即使历史有路线，也以当前消息为准）
        # 3. 仅有历史路线 → 沿途分段（兜底，假设用户隐式继续聊路线）
        # 4. 都没有 → 引导补充
        if want_along and cities:
            return self._build_along_route_plan(cities, want_ev, want_indoor, want_free)
        if place:
            return self._build_single_place_plan(place, want_ev, want_indoor, want_free)
        if cities:
            return self._build_along_route_plan(cities, want_ev, want_indoor, want_free)
        return self._build_clarify_plan()

    # -- helpers --------------------------------------------------------------

    def _extract_route_cities(self, history: List[Dict[str, Any]]) -> List[str]:
        if hasattr(self._svc, "_extract_last_route_cities_from_history"):
            try:
                return list(
                    self._svc._extract_last_route_cities_from_history(history) or []
                )
            except Exception:
                return []
        return []

    def _extract_single_place(self, text: str) -> str:
        # 1) "X 到 Y" 之 Y（目的地优先）
        if hasattr(self._svc, "_extract_route_endpoints"):
            try:
                origin, dest = self._svc._extract_route_endpoints(text)
                if dest:
                    return dest
                if origin:
                    return origin
            except Exception:
                pass
        # 2) "X 附近 / 旁边"
        m = _NEAR_PLACE_RE.search(text)
        if m:
            return m.group(1)
        # 3) 兜底：去掉触发词后剩余的最长中文片段
        stripped = text
        for kw in _PARKING_KW + _EV_KW + _INDOOR_KW + _FREE_KW:
            stripped = stripped.replace(kw, "")
        m = re.search(r"[\u4e00-\u9fa5A-Za-z0-9]{2,15}", stripped)
        return m.group(0) if m else ""

    def _build_along_route_plan(
        self, cities: List[str], ev: bool, indoor: bool, free: bool
    ) -> Dict[str, Any]:
        random.seed(len(cities) + (1 if ev else 0) + (2 if indoor else 0))
        lines = [
            f"- {_mock_parking_hint(c, ev=ev, indoor=indoor, free=free)}"
            for c in cities[:6]
        ]
        suffix = ""
        if ev:
            suffix += "\n\n（已优先标注带充电桩的选项）"
        elif indoor:
            suffix += "\n\n（已优先标注地下/室内停车）"
        elif free:
            suffix += "\n\n（已优先标注免费停车选项）"
        reply = (
            f"沿你最近一次路线（{cities[0]} → {cities[-1]}），给一份分段停车建议（示意，"
            f"真实停车信息请以导航 App 为准）：\n" + "\n".join(lines) + suffix +
            "\n\n如想找特定城市/服务区的停车场，告诉我编号即可。"
        )
        return self._wrap(0.85, reply)

    def _build_single_place_plan(
        self, place: str, ev: bool, indoor: bool, free: bool
    ) -> Dict[str, Any]:
        random.seed(hash(place) & 0xFFFF)
        items = [
            _mock_parking_hint(place, ev=ev, indoor=indoor, free=free) for _ in range(3)
        ]
        body = "\n".join(f"- {x}" for x in dict.fromkeys(items))  # 去重保序
        hint_tail = []
        if not ev:
            hint_tail.append("充电")
        if not indoor:
            hint_tail.append("室内/地下")
        if not free:
            hint_tail.append("免费")
        suggest = "、".join(hint_tail) if hint_tail else "其他偏好"
        reply = (
            f"关于「{place}」附近的停车场建议（示意，真实停车信息请以导航 App 为准）：\n"
            f"{body}\n\n"
            f"如需更精准信息（具体收费、是否需预约），可告诉我「{suggest}」等偏好，我会进一步过滤。"
        )
        return self._wrap(0.82, reply)

    def _build_clarify_plan(self) -> Dict[str, Any]:
        reply = (
            "要查停车场，请先告诉我地点（例如「南京南站附近停车场」），"
            "或先做一次路径规划后再问「沿途停车」，我可以基于路线给分段建议。"
        )
        return self._wrap(0.7, reply)

    @staticmethod
    def _wrap(confidence: float, reply: str) -> Dict[str, Any]:
        return {
            "intent": "route_planning",
            "confidence": confidence,
            "actions": [],
            "llm_reply": reply,
            "used_llm": False,
        }
