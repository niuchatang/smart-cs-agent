from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from memory.extractors import extract_route_endpoints, is_time_only_departure_query
from memory.models import MemoryContext

from .models import Plan, PlanTask

FUTURE_SUITABLE = re.compile(r"(未来|接下来|最近|这|明后)([一二三四五六七八九十\d]+)?天.*适合.*去(.+?)吗")
WEEKEND_TRIP = re.compile(r"(周末|两日|三日).*(游|规划|行程|攻略)")
DEPARTURE_OD = re.compile(r"(明天|后天|今天|下周)?.*(从)?(.+?)到(.+?).*(几点|什么时候|适合).*出发")
SIMPLE_WEATHER_TRIP = re.compile(r"(未来|明天|后天).*(天气|空气|AQI|雾霾).*(去|到)(.+?)(吗|？|\?|$)")


def should_plan(message: str, history: List[Dict[str, Any]], ctx: MemoryContext | None = None) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if is_time_only_departure_query(text) and ctx and ctx.profile_get("commute_origin") and ctx.profile_get("commute_dest"):
        return True
    if DEPARTURE_OD.search(text):
        return True
    if FUTURE_SUITABLE.search(text):
        return True
    if WEEKEND_TRIP.search(text):
        return True
    if SIMPLE_WEATHER_TRIP.search(text):
        return True
    score = 0
    for kw in ("路线", "天气", "空气", "AQI", "拥堵", "风险", "几点出发", "规划"):
        if kw in text:
            score += 1
    return score >= 3


def build_plan(message: str, history: List[Dict[str, Any]], ctx: MemoryContext | None = None, service_agent: Any = None) -> Optional[Plan]:
    text = (message or "").strip()
    if not text:
        return None

    if is_time_only_departure_query(text) and ctx:
        origin = ctx.profile_get("commute_origin")
        dest = ctx.profile_get("commute_dest")
        if origin and dest:
            return Plan(
                goal=f"{origin}到{dest}出行决策",
                execution="mixed",
                tasks=[
                    PlanTask(
                        id="td1",
                        agent="TravelDecisionAgent",
                        tool="query_travel_decision",
                        params={
                            "origin": origin,
                            "destination": dest,
                            "travel_mode": ctx.profile_get("travel_mode") or "driving",
                            "depart_time_text": _extract_depart_time(text),
                        },
                        parallel_group=1,
                    ),
                ],
            )

    m = DEPARTURE_OD.search(text)
    if m:
        origin = _clean(m.group(3))
        dest = _clean(m.group(4))
        if not origin or not dest:
            o, d = extract_route_endpoints(text, service_agent)
            origin, dest = o or origin, d or dest
        if ctx and (not origin or not dest):
            origin = origin or (ctx.profile_get("commute_origin") if ctx else "")
            dest = dest or (ctx.profile_get("commute_dest") if ctx else "")
        if origin and dest:
            depart = _extract_depart_time(text)
            return Plan(
                goal=f"{origin}到{dest}出行决策",
                execution="mixed",
                tasks=[
                    PlanTask(
                        id="td1",
                        agent="TravelDecisionAgent",
                        tool="query_travel_decision",
                        params={
                            "origin": origin,
                            "destination": dest,
                            "travel_mode": (ctx.profile_get("travel_mode") if ctx else "") or "driving",
                            "depart_time_text": depart,
                        },
                        parallel_group=1,
                    ),
                ],
            )

    m2 = FUTURE_SUITABLE.search(text)
    if m2:
        dest = _clean(m2.group(3))
        if dest:
            days = 3
            dm = re.search(r"([一二三四五六七八九十\d]+)天", text)
            if dm:
                days = _cn_num(dm.group(1))
            return Plan(
                goal=f"未来{days}天是否适合去{dest}",
                execution="parallel",
                tasks=[
                    PlanTask(
                        id="w1",
                        agent="WeatherAgent",
                        tool="query_weather",
                        params={
                            "cities": [dest],
                            "include_forecast": True,
                            "include_aqi": True,
                            "include_warning": True,
                            "days": days,
                        },
                        parallel_group=1,
                    ),
                ],
            )

    m3 = SIMPLE_WEATHER_TRIP.search(text)
    if m3:
        dest = _clean(m3.group(4))
        if dest:
            days = 3
            dm = re.search(r"([一二三四五六七八九十\d]+)天", text)
            if dm:
                days = _cn_num(dm.group(1))
            return Plan(
                goal=f"未来{days}天是否适合去{dest}",
                execution="parallel",
                tasks=[
                    PlanTask(
                        id="w1",
                        agent="WeatherAgent",
                        tool="query_weather",
                        params={
                            "cities": [dest],
                            "include_forecast": True,
                            "include_aqi": True,
                            "include_warning": True,
                            "days": days,
                        },
                        parallel_group=1,
                    ),
                ],
            )

    if WEEKEND_TRIP.search(text):
        city = _guess_city(text) or "北京"
        return Plan(
            goal=f"{city}周末行程规划",
            execution="mixed",
            tasks=[
                PlanTask(id="w1", agent="WeatherAgent", tool="query_weather", params={"cities": [city], "include_forecast": True, "days": 2}, parallel_group=1),
                PlanTask(id="r1", agent="RouteAgent", tool="query_route_plan", params={"origin": city, "destination": city, "mode": "driving"}, parallel_group=2),
            ],
        )

    o, d = extract_route_endpoints(text, service_agent)
    if o and d and sum(1 for k in ("天气", "空气", "AQI", "路况", "风险") if k in text) >= 2:
        return Plan(
            goal=f"{o}到{d}综合出行分析",
            execution="mixed",
            tasks=[
                PlanTask(
                    id="td1",
                    agent="TravelDecisionAgent",
                    tool="query_travel_decision",
                    params={"origin": o, "destination": d, "travel_mode": "driving"},
                    parallel_group=1,
                ),
            ],
        )
    return None


def _extract_depart_time(text: str) -> str:
    for kw in ("明天", "后天", "今天", "早上", "晚上"):
        if kw in text:
            return kw
    return ""


def _guess_city(text: str) -> str:
    for c in ("北京", "上海", "天津", "重庆", "广州", "深圳", "杭州", "成都", "西安", "南京"):
        if c in text:
            return c
    return ""


def _clean(s: str) -> str:
    return re.sub(r"[，。！？!?的]$", "", (s or "").strip())


def _cn_num(raw: str) -> int:
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if raw.isdigit():
        return max(1, min(int(raw), 7))
    if raw in mapping:
        return mapping[raw]
    return 3
