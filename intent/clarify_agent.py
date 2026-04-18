"""
澄清提问智能体（Clarify Agent）

定位：在所有规则链都没有高置信度命中、且用户句又显式涉及出行领域时，
与其让主智能体渲染空 `unknown` 回复，不如给出更精准的澄清话术，引导用户补齐关键要素。

不接管场景（返回 None）：
- 已被天气/路况/路径/通用任一前置智能体命中的情况；
- 纯闲聊、问候类（让主智能体走 RAG 知识问答即可）。

命中条件（任一）：
- 含出行关键词（路/车/高速/服务区/ETC/充电/天气/改签等）但缺起终点/时间/城市；
- 用户句极短（<= 4 字）又出现过路径历史。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 注意：此处**故意不包含**"改签/退票/失物/人工/投诉"等词——
# 它们由 `GeneralIntentAgent.try_tail_rules` 直接命中为具体工单/转人工意图，
# 若在此再拦截，会产生先澄清再补充起终点的错位对话。
_TRAVEL_HINT_KW = (
    "路线",
    "导航",
    "怎么走",
    "出行",
    "出差",
    "自驾",
    "驾车",
)

_GREETING = re.compile(r"^(你好|您好|hi|hello|嗨|在吗|在不在|在么)[\s\S]*$", re.I)


class ClarifyAgent:
    name = "clarify"
    priority = 85

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        msg = (message or "").strip()
        if not msg or _GREETING.match(msg):
            return None

        if any(k in msg for k in _TRAVEL_HINT_KW):
            origin, destination = self._svc._extract_route_endpoints(msg) if hasattr(
                self._svc, "_extract_route_endpoints"
            ) else (None, None)
            if origin and destination:
                return None
            return {
                "intent": "unknown",
                "confidence": 0.62,
                "actions": [],
                "llm_reply": (
                    "为了更准确地帮到你，请补充一下这些信息：\n"
                    "1) 出发地与目的地（例如「南京到上海」）；\n"
                    "2) 大致出行时间（例如「今天下午 3 点」）；\n"
                    "3) 是否想看：驾车路线 / 沿途高速路况 / 途经城市天气 / 服务区与充电 / 通行费估算。\n"
                    "直接把其中一两项补齐即可，我来自动继续。"
                ),
                "used_llm": False,
            }

        if len(msg) <= 4 and hasattr(self._svc, "_extract_last_route_cities_from_history"):
            rc = self._svc._extract_last_route_cities_from_history(history)
            if rc:
                ex = rc[0]
                return {
                    "intent": "unknown",
                    "confidence": 0.55,
                    "actions": [],
                    "llm_reply": (
                        f"你最近的路线上有 {len(rc)} 座途经城市（示例：{ex}）。"
                        "想让我做哪一项？可回复「沿途天气」「沿途高速路况」「服务区」或「通行费」。"
                    ),
                    "used_llm": False,
                }
        return None
