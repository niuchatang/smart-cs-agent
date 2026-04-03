"""
通用意图智能体（General Intent Agent）

承接天气规则、公交实时、票价、工单、转人工及 unknown，由调度中枢在路况/路径子智能体之后调用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .weather_agent import WEATHER_CITY_CLARIFY_REPLY

# 单城天气正则易误匹配：「查询天气」→ 曾把「查询」当成城市名
_WEATHER_NOT_A_CITY = frozenset(
    {
        "途径",
        "途经",
        "沿途",
        "路线",
        "这条路线",
        "刚才路线",
        "查询途径",
        "查询途经",
        "查查途径",
        "查查途经",
        "查询",
        "查查",
        "查下",
        "查查看",
        "想查",
        "看看",
    }
)


class GeneralIntentAgent:
    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_weather_rules(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        text = message.lower()
        weather_kw = any(
            k in message
            for k in ["天气", "气温", "降雨", "下雪", "下雨", "台风", "雾霾", "冷不冷", "热不热"]
        )
        roadish = any(
            k in text
            for k in [
                "路况",
                "拥堵",
                "好走",
                "好走吗",
                "畅通",
                "堵车",
                "堵不堵",
                "事故",
                "管制",
                "封闭",
                "封路",
                "高速",
                "高架",
                "环线",
            ]
        )
        if not weather_kw or roadish:
            return None

        chip = self._svc.parse_cities_from_weather_followup_question(message)
        if chip:
            return {
                "intent": "weather_query",
                "confidence": 0.91,
                "actions": [{"tool": "query_weather", "params": {"cities": chip}}],
                "used_llm": False,
            }
        route_ctx = any(
            k in message
            for k in [
                "沿途",
                "途经",
                "途径",
                "这条路线",
                "刚才路线",
                "刚规划",
                "沿线",
                "路上",
                "刚查的路线",
            ]
        )
        if route_ctx:
            rc = self._svc._extract_last_route_cities_from_history(history)
            route_weather_phrase = any(
                p in message
                for p in ("途径天气", "途经天气", "沿途天气", "路线天气", "路上天气")
            )
            if route_weather_phrase and not rc:
                return {
                    "intent": "weather_query",
                    "confidence": 0.7,
                    "actions": [],
                    "llm_reply": "暂未找到可用的路线城市，请先完成一次路径规划后再试。",
                    "used_llm": False,
                }
            if rc:
                # 「途径/途经」类说法需先让用户选城市，与天气芯片一致，避免一次查满整条路线
                if route_weather_phrase:
                    chain = "、".join(rc[:20])
                    suffix = f"（共 {len(rc)} 城）" if len(rc) > 20 else ""
                    ex = str(rc[0]).strip()
                    return {
                        "intent": "weather_query",
                        "confidence": 0.92,
                        "actions": [],
                        "llm_reply": (
                            f"当前路线途经城市包括：{chain}{suffix}。\n"
                            f"请问要查询哪一座城市的天气？请直接回复城市名（例如「{ex}」）。\n"
                            "若希望从第一站起按顺序逐站查询，请回复「沿途」。"
                        ),
                        "used_llm": False,
                    }
                return {
                    "intent": "weather_query",
                    "confidence": 0.9,
                    "actions": [{"tool": "query_weather", "params": {"cities": rc}}],
                    "used_llm": False,
                }
        multi_w = self._svc._extract_multi_stop_places(message)
        if multi_w and len(multi_w) >= 2:
            return {
                "intent": "weather_query",
                "confidence": 0.88,
                "actions": [{"tool": "query_weather", "params": {"cities": multi_w}}],
                "used_llm": False,
            }
        ow, od = self._svc._extract_route_endpoints(message)
        if ow and od:
            return {
                "intent": "weather_query",
                "confidence": 0.87,
                "actions": [{"tool": "query_weather", "params": {"cities": [ow, od]}}],
                "used_llm": False,
            }
        # 须在 m_one 之前：「A、B天气怎么样」若只走 m_one 会匹配到 B+天气，漏掉 A
        listed = self._svc.parse_weather_city_list_from_message(message)
        if listed and len(listed) >= 2:
            return {
                "intent": "weather_query",
                "confidence": 0.89,
                "actions": [{"tool": "query_weather", "params": {"cities": listed}}],
                "used_llm": False,
            }
        m_one = re.search(
            r"(?:查(?:询|(?:一)?下)|看看)?\s*([\u4e00-\u9fff]{2,12}(?:市|县|区|自治州)?)\s*(?:今天|现在|这两)?\s*(?:天气|气温)",
            message,
        )
        if m_one:
            one_city = self._svc._clean_place(m_one.group(1))
            if one_city and one_city not in _WEATHER_NOT_A_CITY:
                return {
                    "intent": "weather_query",
                    "confidence": 0.86,
                    "actions": [{"tool": "query_weather", "params": {"cities": [one_city]}}],
                    "used_llm": False,
                }
        return {
            "intent": "weather_query",
            "confidence": 0.55,
            "actions": [],
            "llm_reply": WEATHER_CITY_CLARIFY_REPLY,
            "used_llm": False,
        }

    def try_tail_rules(self, message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = message.lower()
        target = self._svc._extract_transit_target(message) or self._svc._extract_last_target_from_history(history)

        if any(k in text for k in ["地铁", "公交", "路况", "拥堵", "延误", "班次", "首班", "末班", "实时"]):
            return {
                "intent": "realtime_status",
                "confidence": 0.9,
                "actions": [{"tool": "query_transit_status", "params": {"target": target}}],
                "used_llm": False,
            }
        if any(k in text for k in ["票价", "多少钱", "收费", "换乘优惠", "学生卡", "次卡", "月票"]):
            return {"intent": "fare_policy", "confidence": 0.86, "actions": [], "used_llm": False}
        if any(k in text for k in ["退票", "退款", "改签", "取消行程"]):
            return {
                "intent": "ticket_refund",
                "confidence": 0.9,
                "actions": [{"tool": "create_transport_ticket", "params": {"issue_type": "ticket_refund", "detail": message}}],
                "used_llm": False,
            }
        if any(k in text for k in ["失物", "遗失", "丢了", "招领", "找回"]):
            return {
                "intent": "lost_and_found",
                "confidence": 0.9,
                "actions": [{"tool": "create_transport_ticket", "params": {"issue_type": "lost_and_found", "detail": message}}],
                "used_llm": False,
            }
        if any(k in text for k in ["人工", "投诉", "态度差", "生气", "服务差"]):
            return {
                "intent": "complaint" if "投诉" in text else "human_handoff",
                "confidence": 0.92,
                "actions": [{"tool": "create_transport_ticket", "params": {"issue_type": "complaint", "detail": message}}],
                "used_llm": False,
            }
        if any(k in text for k in ["转人工", "人工客服", "客服介入"]):
            return {
                "intent": "human_handoff",
                "confidence": 0.95,
                "actions": [{"tool": "handoff_to_human", "params": {"priority": "high"}}],
                "used_llm": False,
            }
        return {"intent": "unknown", "confidence": 0.45, "actions": [], "used_llm": False}
