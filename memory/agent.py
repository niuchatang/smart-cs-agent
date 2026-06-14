from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import (
    build_event_content,
    extract_commute,
    extract_residence,
    extract_route_endpoints,
    is_time_only_departure_query,
    summarize_turn,
    worth_remembering,
)
from .models import MemoryContext
from .repository import MemoryRepository
from .tools import MemoryToolkit


class MemoryAgent:
    """长期记忆 Agent：加载上下文、补全消息、抽取并持久化记忆。"""

    def __init__(self, service_agent: Any = None, repo: MemoryRepository | None = None) -> None:
        self._svc = service_agent
        self._repo = repo or MemoryRepository()
        self.tools = MemoryToolkit(self._repo)

    def load_context(self, user_id: str) -> MemoryContext:
        if not user_id:
            return MemoryContext()
        profiles = self._repo.get_profiles(user_id)
        events = self._repo.list_events(user_id, limit=8)
        summary = self._repo.latest_summary(user_id)
        hint = self._build_proactive_hint(profiles)
        return MemoryContext(
            user_id=user_id,
            profiles=profiles,
            recent_events=events,
            summary=summary,
            proactive_hint=hint,
        )

    def enrich_message(self, message: str, ctx: MemoryContext) -> str:
        text = (message or "").strip()
        if not text or not ctx.profiles:
            return text
        if is_time_only_departure_query(text):
            origin = ctx.profile_get("commute_origin")
            dest = ctx.profile_get("commute_dest")
            if origin and dest:
                return f"从{origin}到{dest}{text}"
        if "回家" in text and ctx.profile_get("home_district"):
            return text.replace("回家", f"回{ctx.profile_get('home_district')}")
        return text

    def context_prompt(self, ctx: MemoryContext) -> str:
        if not ctx.user_id:
            return ""
        lines: List[str] = []
        if ctx.profile_get("commute_origin") and ctx.profile_get("commute_dest"):
            lines.append(
                f"通勤：{ctx.profile_get('commute_origin')} → {ctx.profile_get('commute_dest')}"
            )
        if ctx.profile_get("home_district"):
            lines.append(f"常住：{ctx.profile_get('home_district')}")
        if ctx.profile_get("travel_mode"):
            lines.append(f"出行方式：{ctx.profile_get('travel_mode')}")
        if ctx.summary:
            lines.append(f"近期摘要：{ctx.summary}")
        return " | ".join(lines)

    def extract_and_save(
        self,
        user_id: str,
        message: str,
        plan: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        reply: str = "",
    ) -> None:
        if not user_id:
            return
        text = (message or "").strip()
        commute = extract_commute(text)
        if commute:
            origin, dest = commute
            if origin:
                self._repo.upsert_profile(user_id, "commute_origin", origin)
                self._repo.upsert_profile(user_id, "home_district", origin)
            if dest:
                self._repo.upsert_profile(user_id, "commute_dest", dest)

        residence = extract_residence(text)
        if residence:
            self._repo.upsert_profile(user_id, "home_district", residence)

        if any(k in text for k in ("开车", "驾车", "自驾")):
            self._repo.upsert_profile(user_id, "travel_mode", "driving")
        elif "地铁" in text or "公交" in text:
            self._repo.upsert_profile(user_id, "travel_mode", "transit")

        o, d = extract_route_endpoints(text, self._svc)
        if o and d:
            self._repo.upsert_profile(user_id, "last_origin", o)
            self._repo.upsert_profile(user_id, "last_destination", d)

        if worth_remembering(text, plan):
            event_type = str(plan.get("intent") or "conversation")
            self._repo.add_event(user_id, event_type, build_event_content(text, plan, tool_results))
            self._repo.add_summary(user_id, summarize_turn(text, plan, reply))

    def search(self, user_id: str, query: str = "") -> Dict[str, Any]:
        return self.tools.search.run(user_id, query)

    def _build_proactive_hint(self, profiles: Dict[str, str]) -> str:
        origin = profiles.get("commute_origin", "")
        dest = profiles.get("commute_dest", "")
        if origin and dest:
            return f"已记住您的通勤路线：{origin} → {dest}。询问出发时间时会自动补全起终点。"
        return ""
