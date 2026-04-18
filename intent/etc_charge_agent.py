"""
ETC / 通行费估算智能体（ETC Charge Agent）

- 复用既有 `fare_policy` 意图承载；
- 触发词：「ETC」「通行费」「过路费」「收费多少」「多少钱」「节假日免费」等；
- 不调用任何外部内部系统；基于公开的一类车 0.45 元/公里 × 折扣系数做粗估，
  并对节假日小客车免费做常识提示；
- 若 history 中存在最近一次 `query_route_plan` 成功结果，则按其里程直接估算，
  不再另发起外部路径请求。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional

# 节假日小客车免通行费的名义日期（公开常识，实际以国务院当年公告为准）
_FREE_HOLIDAYS_NAMES = ("春节", "清明", "劳动节", "国庆")

_PRICE_PER_KM_BY_CLASS = {
    1: 0.45,  # 7座及以下小客车
    2: 0.55,  # 8-19座客车 / 2 轴货车
    3: 0.65,
    4: 0.80,
    5: 0.95,
}


def _etc_discount(price: float) -> float:
    """常见 ETC 小客车 5% 折扣（公开常识），货车不打折。"""
    return round(price * 0.95, 2)


def _extract_distance_km_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    for item in reversed(history or []):
        if item.get("role") != "agent":
            continue
        meta = item.get("meta") or {}
        trs = meta.get("tool_results") if isinstance(meta, dict) else None
        if not isinstance(trs, list):
            continue
        for tr in reversed(trs):
            if not isinstance(tr, dict):
                continue
            if tr.get("tool") == "query_route_plan" and tr.get("success"):
                for key in ("distance_km", "distance", "distance_meters"):
                    v = tr.get(key)
                    if isinstance(v, (int, float)) and v > 0:
                        return float(v) / (1000.0 if key == "distance_meters" else 1.0)
        break
    return None


class ETCChargeAgent:
    name = "etc_charge"
    priority = 20

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        tl = text.lower()
        hit = any(
            k in tl or k in text
            for k in ("etc", "通行费", "过路费", "高速费", "收费多少", "节假日免费")
        ) or ("多少钱" in text and any(k in text for k in ("高速", "驾车", "开车")))
        if not hit:
            return None

        vclass = _infer_vehicle_class(text)
        price_per_km = _PRICE_PER_KM_BY_CLASS.get(vclass, 0.45)
        distance_km = _extract_distance_km_from_history(history)

        holiday_note = ""
        if any(k in text for k in _FREE_HOLIDAYS_NAMES) or "节假日" in text:
            holiday_note = (
                "\n\n提醒：春节、清明、五一、国庆四个全国性法定节假日，"
                "7 座及以下小型客车在收费公路主线收费站无需缴纳通行费（以当年国务院公告为准）。"
            )

        if distance_km:
            full = round(distance_km * price_per_km, 2)
            etc = _etc_discount(full) if vclass == 1 else full
            reply = (
                f"基于你最近一次规划的路线约 {distance_km:.0f} 公里、车型 {vclass} 类，"
                f"粗估通行费 {full} 元；若走 ETC 折后约 {etc} 元。\n"
                "注：实际金额以入出口自动计费为准；部分省际差异与特殊路段另行计算。"
                f"{holiday_note}"
            )
        else:
            reply = (
                "可以估算通行费，请告诉我起终点（例如「南京到上海」），"
                f"我会按 {price_per_km:.2f} 元/公里（{vclass} 类车）给出粗估；"
                "ETC 小客车通常再打约 95 折（以各省政策为准）。"
                f"{holiday_note}"
            )

        return {
            "intent": "fare_policy",
            "confidence": 0.88 if distance_km else 0.82,
            "actions": [],
            "llm_reply": reply,
            "used_llm": False,
        }


def _infer_vehicle_class(text: str) -> int:
    if re.search(r"7\s*座|五座|七座|小客车|小轿车|家用车", text):
        return 1
    if re.search(r"中巴|中型客车|19\s*座", text):
        return 2
    if re.search(r"大巴|大型客车|旅游大巴", text):
        return 3
    if re.search(r"货车|厢式货车|卡车", text):
        return 3
    return 1
