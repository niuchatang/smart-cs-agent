"""
对话摘要器（Conversation Summarizer）

- 按 N 轮阈值把更早的历史压成一段事实性摘要（起终点、已查询城市、最近高速编号等），
  降低 `_history_to_text` 对 prompt 长度的压力；
- 纯规则模式**零依赖**即可使用；若 `self._svc.llm` 存在，则可选走 LLM 精炼；
- 摘要不改写 `conversations.json` 的原始数据，只作为「注入上下文」供 `_plan_by_llm` 使用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _fact_from_plan(meta: Dict[str, Any]) -> List[str]:
    facts: List[str] = []
    trs = meta.get("tool_results") if isinstance(meta, dict) else None
    if not isinstance(trs, list):
        return facts
    for tr in trs:
        if not isinstance(tr, dict) or not tr.get("success"):
            continue
        tool = tr.get("tool")
        if tool == "query_route_plan":
            o = tr.get("origin")
            d = tr.get("destination")
            if o and d:
                facts.append(f"曾规划过 {o} → {d}")
        elif tool == "query_weather":
            cities = tr.get("cities") or tr.get("query_cities")
            if isinstance(cities, list) and cities:
                facts.append("曾查询城市天气：" + "、".join(str(x) for x in cities[:5]))
        elif tool == "query_highway_condition":
            code = tr.get("code") or tr.get("target")
            if code:
                facts.append(f"曾查询高速 {code} 路况")
    return facts


class ConversationSummarizer:
    def __init__(self, service_agent: Any, keep_recent: int = 6) -> None:
        self._svc = service_agent
        self._keep_recent = max(2, keep_recent)

    def split(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        hist = history or []
        if len(hist) <= self._keep_recent:
            return {"summary": "", "recent": hist}
        older = hist[: -self._keep_recent]
        recent = hist[-self._keep_recent :]
        return {"summary": self._summarize_rule_based(older), "recent": recent}

    def _summarize_rule_based(self, older: List[Dict[str, Any]]) -> str:
        facts: List[str] = []
        for item in older:
            meta = item.get("meta") or {}
            facts.extend(_fact_from_plan(meta))
            if item.get("role") == "user":
                content = str(item.get("content", ""))
                codes = re.findall(r"[GS]\s*\d{1,4}", content)
                if codes:
                    facts.append("用户提到过高速：" + "、".join(c.replace(" ", "").upper() for c in codes[:3]))
        seen: set[str] = set()
        uniq: List[str] = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        if not uniq:
            return ""
        return "[历史要点] " + "；".join(uniq[-8:])
