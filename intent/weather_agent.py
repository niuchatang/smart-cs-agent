"""
天气对话智能体（Weather Dialog Agent）

- 芯片为肯定句「查询途经城市天气」：点击后助手先询问要查途经**哪一座**城市，用户回复城市名后再查天气。
- 用户也可回复「沿途」等，从第一站起按顺序逐站查询（与「继续查询下一途经城市」衔接）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 与 main._weather_followup_from_route 中 question 一致（肯定句，非疑问「是否」）
WEATHER_ROUTE_CHIP = "查询途经城市天气"
WEATHER_ROUTE_CHIP_ALT = (
    "查询途径城市天气",
    "查询途经城市的天气",
    "查询途径城市的天气",
)

# 口语/简写：与完整芯片同流程（先问查哪一座途经城市，不当作城市名查询）
WEATHER_ROUTE_CHIP_SHORT = (
    "查询途径天气",
    "查询途经天气",
    "查途径天气",
    "查途经天气",
    "想查途径天气",
    "想查途经天气",
)

# 历史会话里曾使用的芯片（兼容）
WEATHER_ROUTE_CHIP_LEGACY = (
    "是否查询途经城市的天气？",
    "是否查询途径城市的天气？",
    "是否查询途经城市天气？",
)

WEATHER_CITY_CLARIFY_REPLY = (
    "好的。请具体说明要查询哪些城市的天气（可列出多个，用顿号或逗号分隔，例如「南京、武汉」）；"
    "若希望按刚规划的路线依次查询，请回复「沿途」或「途经城市」（从第一站开始，每次查一城）。"
)

# 仅「查天气」、未出现地名时：不调用天气 API，直接追问城市（避免把「查询」等当成城市名）
_BARE_WEATHER_QUERY_CORE = frozenset(
    {
        "查询天气",
        "查天气",
        "查查天气",
        "查下天气",
        "查一下天气",
        "查询一下天气",
        "看看天气",
        "想查天气",
        "我要查天气",
        "我想查天气",
        "帮我查天气",
        "帮我查询天气",
        "帮我查一下天气",
        "请帮我查天气",
        "天气查询",
        "查一查天气",
        "查个天气",
    }
)

WEATHER_CITY_CLARIFY_MARKERS = (
    "请具体说明要查询哪些城市",
    "若希望按您刚规划的整条路线查询",
    "若希望按刚规划的路线依次查询",
)

# 上一轮助手已问「途经哪一座城市」
WEATHER_WHICH_CITY_MARKERS = (
    "请问要查询途经哪一座城市",
    "查询途经哪一座城市的天气",
    "途经哪一座城市的天气",
    "请直接回复城市名",
)


class WeatherDialogAgent:
    """天气多轮对话：try_plan 返回与 UserIntentAgent 相同结构的 plan，或 None 表示不接管。"""

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    @staticmethod
    def _strip_bare_weather_punctuation(s: str) -> str:
        t = re.sub(r"\s+", "", (s or "").strip())
        t = t.rstrip("吗么嘛吧呢呀啊呐噢哦哪嗯")
        return t.rstrip("？?！!。. ")

    @classmethod
    def _is_bare_weather_query(cls, msg: str) -> bool:
        core = cls._strip_bare_weather_punctuation(msg)
        return core in _BARE_WEATHER_QUERY_CORE

    def try_plan(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        msg = (message or "").strip()
        if not msg:
            return None

        if self._is_bare_weather_query(msg):
            return {
                "intent": "weather_query",
                "confidence": 0.9,
                "actions": [],
                "llm_reply": WEATHER_CITY_CLARIFY_REPLY,
                "used_llm": False,
            }

        if self._is_continue_route_weather_message(msg):
            return self._plan_continue_route_weather(history)

        if self._is_short_weather_yes(msg):
            pend0 = self._extract_along_route_pending_from_history(history)
            if isinstance(pend0, dict) and not pend0.get("complete") and pend0.get("next_index") is not None:
                return self._build_query_next_stop_plan(pend0)

        if self._is_route_weather_chip_message(msg):
            return self._plan_chip_route_weather(history)

        # 上一轮是「途经天气第 N 站」类回复时，正文不含「请问要查哪一座城」等标记，原先无法走多城解析；
        # 用户若直接发「南通、苏州、连云港」应一次查多城（普通 query_weather），而非误走 LLM/仅一站。
        if self._last_agent_had_along_route_weather_reply(history):
            plist = self._parse_user_city_list(msg)
            if len(plist) >= 2:
                return self._plan_query_cities(plist[:10])
            if len(plist) == 1:
                pick_one = self._match_route_city_pick(plist[0], history)
                if pick_one is not None:
                    rc, idx = pick_one
                    return self._build_query_at_route_index(rc, idx)
                return self._plan_query_cities(plist[:10])

        if self._last_agent_asked_route_weather_which_city(history):
            if self._is_along_route_token(msg):
                return self._plan_query_route_cities(history)
            pick = self._match_route_city_pick(msg, history)
            if pick is not None:
                rc, idx = pick
                return self._build_query_at_route_index(rc, idx)
            plist = self._parse_user_city_list(msg)
            if len(plist) >= 2:
                return self._plan_query_cities(plist[:10])
            if len(plist) == 1:
                pick_one = self._match_route_city_pick(plist[0], history)
                if pick_one is not None:
                    rc, idx = pick_one
                    return self._build_query_at_route_index(rc, idx)
            if self._is_short_weather_yes(msg):
                return {
                    "intent": "weather_query",
                    "confidence": 0.88,
                    "actions": [],
                    "llm_reply": "请从路线途经列表里回复一座城市名称（例如列表中的第一个城市），或回复「沿途」从第一站起依次查询。",
                    "used_llm": False,
                }
            return None

        if self._last_agent_asked_weather_city_detail(history):
            if self._is_along_route_token(msg):
                return self._plan_query_route_cities(history)
            cities = self._parse_user_city_list(msg)
            if cities:
                return self._plan_query_cities(cities)
            if self._is_short_weather_yes(msg):
                return {
                    "intent": "weather_query",
                    "confidence": 0.88,
                    "actions": [],
                    "llm_reply": "还请补充要查询的城市名称（例如「上海、杭州」），或回复「沿途」按刚规划路线上的城市查询。",
                    "used_llm": False,
                }
            return None

        if self._is_short_weather_yes(msg) and self._last_agent_offered_route_weather(history):
            if self._last_agent_highway_pick_clarification(history):
                return None
            return self._plan_short_yes_after_route_weather_offer(history)

        return None

    def _plan_short_yes_after_route_weather_offer(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        rc = self._svc._extract_last_route_cities_from_history(history)
        if rc:
            chain = "、".join(rc[:20])
            suffix = f"（共 {len(rc)} 城）" if len(rc) > 20 else ""
            return {
                "intent": "weather_query",
                "confidence": 0.9,
                "actions": [],
                "llm_reply": (
                    f"当前路线途经城市包括：{chain}{suffix}。\n"
                    "请问要查询途经哪一座城市的天气？请直接回复城市名；若从第一站起逐站查，请回复「沿途」。"
                ),
                "used_llm": False,
            }
        return {
            "intent": "weather_query",
            "confidence": 0.9,
            "actions": [],
            "llm_reply": WEATHER_CITY_CLARIFY_REPLY,
            "used_llm": False,
        }

    def _plan_chip_route_weather(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        rc = self._svc._extract_last_route_cities_from_history(history)
        if not rc:
            return {
                "intent": "weather_query",
                "confidence": 0.7,
                "actions": [],
                "llm_reply": "暂未找到可用的路线城市，请先完成一次路径规划后再试。",
                "used_llm": False,
            }
        chain = "、".join(rc[:20])
        suffix = f"（共 {len(rc)} 城）" if len(rc) > 20 else ""
        ex = str(rc[0]).strip()
        return {
            "intent": "weather_query",
            "confidence": 0.94,
            "actions": [],
            "llm_reply": (
                f"当前路线途经城市包括：{chain}{suffix}。\n"
                f"请问要查询途经哪一座城市的天气？请直接回复城市名（例如「{ex}」）。\n"
                "若希望从第一站起按顺序逐站查询，请回复「沿途」。"
            ),
            "used_llm": False,
        }

    def _plan_query_route_cities(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        rc = self._svc._extract_last_route_cities_from_history(history)
        if rc:
            return {
                "intent": "weather_query",
                "confidence": 0.93,
                "actions": [
                    {
                        "tool": "query_weather",
                        "params": {
                            "cities": [rc[0]],
                            "along_route_queue": rc,
                            "along_route_index": 0,
                        },
                    }
                ],
                "used_llm": False,
            }
        return {
            "intent": "weather_query",
            "confidence": 0.65,
            "actions": [],
            "llm_reply": "当前会话里还没有成功的路径规划记录，请先规划路线，或直接告诉我城市名称。",
            "used_llm": False,
        }

    def _plan_continue_route_weather(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        pend = self._extract_along_route_pending_from_history(history)
        if isinstance(pend, dict) and not pend.get("complete") and pend.get("next_index") is not None:
            return self._build_query_next_stop_plan(pend)
        return {
            "intent": "weather_query",
            "confidence": 0.72,
            "actions": [],
            "llm_reply": "当前没有待查询的下一途经城市。若需重新开始，请先规划路线并点击「查询途经城市天气」。",
            "used_llm": False,
        }

    def _build_query_at_route_index(self, rc: List[str], idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(rc):
            return {
                "intent": "weather_query",
                "confidence": 0.55,
                "actions": [],
                "llm_reply": "未在路线途经列表中定位到该城市，请核对名称后重试。",
                "used_llm": False,
            }
        city = str(rc[idx]).strip()
        return {
            "intent": "weather_query",
            "confidence": 0.93,
            "actions": [
                {
                    "tool": "query_weather",
                    "params": {
                        "cities": [city],
                        "along_route_queue": rc,
                        "along_route_index": idx,
                    },
                }
            ],
            "used_llm": False,
        }

    def _build_query_next_stop_plan(self, pend: Dict[str, Any]) -> Dict[str, Any]:
        q = pend.get("queue")
        ni = pend.get("next_index")
        if not isinstance(q, list) or not q or ni is None:
            return {
                "intent": "weather_query",
                "confidence": 0.6,
                "actions": [],
                "llm_reply": "无法解析途经城市队列，请重新发起路径规划后的天气查询。",
                "used_llm": False,
            }
        try:
            ix = int(ni)
        except (TypeError, ValueError):
            return {
                "intent": "weather_query",
                "confidence": 0.6,
                "actions": [],
                "llm_reply": "途经天气状态异常，请重新点击追问或发送「继续查询下一途经城市天气」。",
                "used_llm": False,
            }
        if ix < 0 or ix >= len(q):
            return {
                "intent": "weather_query",
                "confidence": 0.6,
                "actions": [],
                "llm_reply": "途经城市序号已越界，请重新开始查询。",
                "used_llm": False,
            }
        city = str(q[ix]).strip()
        return {
            "intent": "weather_query",
            "confidence": 0.93,
            "actions": [
                {
                    "tool": "query_weather",
                    "params": {
                        "cities": [city],
                        "along_route_queue": q,
                        "along_route_index": ix,
                    },
                }
            ],
            "used_llm": False,
        }

    def _match_route_city_pick(self, msg: str, history: List[Dict[str, Any]]) -> Optional[Tuple[List[str], int]]:
        rc = self._svc._extract_last_route_cities_from_history(history)
        if not rc:
            return None
        raw = (msg or "").strip()
        raw = re.sub(r"(的)?天气.*$", "", raw).strip()
        if raw in ("起点", "出发地", "第一站", "第一个", "首个", "开始"):
            return rc, 0
        if raw in ("终点", "目的地", "最后一站", "最后一个", "结尾"):
            return rc, len(rc) - 1
        m = re.match(r"^第\s*(\d+)\s*", raw)
        if m:
            k = int(m.group(1))
            if 1 <= k <= len(rc):
                return rc, k - 1
        cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        m2 = re.match(r"^第\s*([一二三四五六七八九十]+)\s*", raw)
        if m2:
            g = m2.group(1)
            k = cn.get(g, 0)
            if k == 0 and g == "十":
                k = 10
            if 1 <= k <= len(rc):
                return rc, k - 1

        cleaned = self._svc._clean_place(raw)
        cleaned = re.sub(r"^(查|查询|我要查|想查)\s*", "", cleaned).strip()
        if not cleaned:
            return None

        for i, c in enumerate(rc):
            cc = self._svc._clean_place(str(c))
            if cleaned == cc:
                return rc, i

        hits: List[Tuple[int, int]] = []
        for i, c in enumerate(rc):
            cc = self._svc._clean_place(str(c))
            if len(cleaned) >= 2 and (cleaned in cc or cc in cleaned):
                hits.append((i, len(cc)))
        if len(hits) == 1:
            return rc, hits[0][0]
        if hits:
            hits.sort(key=lambda x: -x[1])
            return rc, hits[0][0]

        plist = self._parse_user_city_list(raw)
        if len(plist) == 1:
            one = plist[0]
            for i, c in enumerate(rc):
                cc = self._svc._clean_place(str(c))
                if one == cc or (len(one) >= 2 and (one in cc or cc in one)):
                    return rc, i
        return None

    @staticmethod
    def _is_continue_route_weather_message(msg: str) -> bool:
        s = (msg or "").strip()
        base = "继续查询下一途经城市天气"
        if s == base or s.rstrip("？?") == base:
            return True
        return bool(re.match(r"^继续查询下一途经城市（.+）的天气[？?]?\s*$", s))

    @staticmethod
    def _last_agent_had_along_route_weather_reply(history: List[Dict[str, Any]]) -> bool:
        """最近一条助手消息是否来自「沿途/途经」逐站天气（tool_results 含 along_route_mode）。"""
        for item in reversed(history or []):
            if item.get("role") != "agent":
                continue
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            trs = meta.get("tool_results", [])
            if not isinstance(trs, list):
                return False
            for tr in reversed(trs):
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") == "query_weather" and tr.get("success") and tr.get("along_route_mode"):
                    return True
            return False
        return False

    @staticmethod
    def _extract_along_route_pending_from_history(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for item in reversed(history):
            if item.get("role") != "agent":
                continue
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            trs = meta.get("tool_results", [])
            if not isinstance(trs, list):
                return None
            for tr in reversed(trs):
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_weather" or not tr.get("success"):
                    continue
                pend = tr.get("along_route_pending")
                if isinstance(pend, dict) and pend.get("next_index") is not None and not pend.get("complete"):
                    return pend
            return None
        return None

    def _plan_query_cities(self, cities: List[str]) -> Dict[str, Any]:
        return {
            "intent": "weather_query",
            "confidence": 0.92,
            "actions": [{"tool": "query_weather", "params": {"cities": cities}}],
            "used_llm": False,
        }

    def _is_short_weather_yes(self, msg: str) -> bool:
        t = msg.strip()
        if not t or len(t) > 20:
            return False
        if re.search(r"[、,，;；]", t):
            return False
        if t in {"是", "要", "嗯", "行", "好", "查", "对", "嗯嗯", "好哒"}:
            return True
        return self._svc._is_affirmative_message(t)

    @staticmethod
    def _norm_route_chip_text(s: str) -> str:
        return re.sub(r"\s+", "", (s or "").strip())

    @classmethod
    def _is_route_weather_chip_message(cls, msg: str) -> bool:
        s = msg.strip()
        sn = cls._norm_route_chip_text(s)
        all_c = (
            WEATHER_ROUTE_CHIP,
            *WEATHER_ROUTE_CHIP_ALT,
            *WEATHER_ROUTE_CHIP_LEGACY,
            *WEATHER_ROUTE_CHIP_SHORT,
        )
        for c in all_c:
            c0 = c.strip()
            if s == c0:
                return True
            if sn == cls._norm_route_chip_text(c0):
                return True
            if s.rstrip("。.！!") == c0.rstrip("。.！!？?"):
                return True
            if s.rstrip("？?") == c0.rstrip("？?"):
                return True
        return False

    @staticmethod
    def _is_along_route_token(msg: str) -> bool:
        s = msg.strip().lower()
        keys = (
            "沿途",
            "途经",
            "途经城市",
            "就查途经",
            "刚规划的",
            "这条路线",
            "刚查的路线",
            "路线上的",
            "全部城市",
            "都查",
        )
        return any(k in s for k in keys)

    def _parse_user_city_list(self, msg: str) -> List[str]:
        listed = self._svc.parse_weather_city_list_from_message(msg)
        if listed:
            return listed
        s = re.sub(r"^(就|只|先|仅|帮我|查一下|查询)\s*", "", msg.strip())
        if not s or self._is_along_route_token(s):
            return []
        if self._is_short_weather_yes(s):
            return []
        parts = re.split(r"[、,，;；\s]+", s)
        out: List[str] = []
        seen: set[str] = set()
        skip = {"天气", "气温", "查询", "怎么样", "如何", "帮我", "然后", "还有", "途径", "途经", "沿途"}
        for p in parts:
            t = self._svc._clean_place(p)
            if t in skip:
                continue
            if len(t) >= 2 and len(t) <= 16 and t not in seen:
                seen.add(t)
                out.append(t)
        return out[:10]

    @staticmethod
    def _last_agent_offered_route_weather(history: List[Dict[str, Any]]) -> bool:
        markers = (
            "查询途经城市天气",
            "查询途径城市天气",
            "是否查询途经城市的天气",
            "是否查询途径城市的天气",
            "是否查询途经城市天气",
        )
        for item in reversed(history):
            if item.get("role") != "agent":
                continue
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            fus = meta.get("follow_ups", [])
            if not isinstance(fus, list):
                return False
            for fu in fus:
                if not isinstance(fu, dict):
                    continue
                q = str(fu.get("question", "")).strip()
                if any(m in q for m in markers):
                    return True
            return False
        return False

    @staticmethod
    def _last_agent_asked_route_weather_which_city(history: List[Dict[str, Any]]) -> bool:
        for item in reversed(history):
            if item.get("role") != "agent":
                continue
            content = str(item.get("content", ""))
            if any(m in content for m in WEATHER_WHICH_CITY_MARKERS):
                return True
            return False
        return False

    @staticmethod
    def _last_agent_asked_weather_city_detail(history: List[Dict[str, Any]]) -> bool:
        for item in reversed(history):
            if item.get("role") != "agent":
                continue
            content = str(item.get("content", ""))
            if any(m in content for m in WEATHER_CITY_CLARIFY_MARKERS):
                return True
            if "还请补充要查询的城市名称" in content:
                return True
            return False
        return False

    @staticmethod
    def _last_agent_highway_pick_clarification(history: List[Dict[str, Any]]) -> bool:
        for item in reversed(history):
            if item.get("role") != "agent":
                continue
            c = str(item.get("content", ""))
            if "涉及多条高速" in c or "请告诉我要展开哪一条" in c:
                return True
            return False
        return False
