from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

COMMUTE_PATTERNS = [
    re.compile(r"每天.*?从(.+?)到(.+?)(?:上班|通勤|开车|出发)"),
    re.compile(r"我.*?住在(.+?)"),
    re.compile(r"常住(.+?)"),
    re.compile(r"从(.+?)到(.+?)上班"),
]

TIME_ONLY_PATTERNS = [
    re.compile(r"^(今天|明天|后天)?几点出发"),
    re.compile(r"^(今天|明天|后天)?什么时候出发"),
    re.compile(r"^(今天|明天|后天)?适合几点"),
]


def extract_commute(message: str) -> Optional[Tuple[str, str]]:
    text = (message or "").strip()
    for pat in COMMUTE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 2:
            origin = _clean(m.group(1))
            dest = _clean(m.group(2))
            if origin and dest:
                return origin, dest
        if m.lastindex == 1:
            home = _clean(m.group(1))
            if home:
                return home, ""
    return None


def extract_residence(message: str) -> str:
    text = (message or "").strip()
    for pat in (re.compile(r"住在(.+?)(?:，|。|$)"), re.compile(r"家住(.+?)(?:，|。|$)")):
        m = pat.search(text)
        if m:
            return _clean(m.group(1))
    return ""


def is_time_only_departure_query(message: str) -> bool:
    text = (message or "").strip()
    return any(p.search(text) for p in TIME_ONLY_PATTERNS)


def extract_route_endpoints(message: str, service_agent: Any = None) -> Tuple[str, str]:
    if service_agent is not None and hasattr(service_agent, "_extract_route_endpoints"):
        try:
            result = service_agent._extract_route_endpoints(message)
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return _clean(str(result[0])), _clean(str(result[1]))
        except Exception:
            pass
    m = re.search(r"从(.+?)到(.+?)(?:的|适合|几点|天气|路线|走|出发|$)", message or "")
    if m:
        return _clean(m.group(1)), _clean(m.group(2))
    return "", ""


def worth_remembering(message: str, plan: Dict[str, Any]) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if extract_commute(text):
        return True
    if extract_residence(text):
        return True
    intent = str(plan.get("intent") or "")
    if intent in {"route_planning", "travel_decision", "weather_query"}:
        return True
    if any(k in text for k in ("每天", "经常", "习惯", "偏好", "通勤")):
        return True
    return False


def build_event_content(message: str, plan: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    intent = str(plan.get("intent") or "")
    content: Dict[str, Any] = {"message": message, "intent": intent}
    o, d = extract_route_endpoints(message)
    if o and d:
        content["origin"] = o
        content["destination"] = d
    route = next((x for x in tool_results if x.get("tool") == "query_route_plan" and x.get("success")), None)
    if route:
        content["route_summary"] = {
            "distance_km": route.get("distance_km"),
            "duration_min": route.get("duration_min"),
        }
    td = next((x for x in tool_results if x.get("tool") == "query_travel_decision" and x.get("success")), None)
    if td:
        content["travel_decision"] = td.get("query", {})
        content["risk_level"] = (td.get("risk") or {}).get("level")
    return content


def summarize_turn(message: str, plan: Dict[str, Any], reply: str = "") -> str:
    intent = str(plan.get("intent") or "unknown")
    o, d = extract_route_endpoints(message)
    if o and d:
        return f"讨论 {o} 到 {d} 的出行（{intent}）"
    if intent == "weather_query":
        return f"查询天气：{message[:40]}"
    if reply:
        return reply[:120]
    return message[:80]


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[，。！？!?]$", "", t)
    return t.strip()
