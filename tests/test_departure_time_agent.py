"""DepartureTimeAgent 行为回归测试。

覆盖三种场景：
1. 「成都到重庆的最佳出行时间」这类含 OD 的出发时间提问要命中本智能体，
   **不能**被 RoutePlanningAgent 以 OD 为由抢走；
2. 当前消息无 OD 时，能从历史里最近一次成功 query_route_plan 取 OD 回填；
3. 没有任何触发词时返回 None。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from intent.departure_time_agent import DepartureTimeAgent
from intent.orchestrator_agent import IntentOrchestratorAgent


class _StubService:
    """最小化模拟 CustomerServiceAgent，仅暴露本文件用到的两个工具方法。"""

    @staticmethod
    def _extract_route_endpoints(text: str) -> Tuple[str, str]:
        import re

        m = re.search(r"([\u4e00-\u9fa5]{2,6})\s*(?:到|至|去|往)\s*([\u4e00-\u9fa5]{2,6})", text)
        if not m:
            return "", ""
        return m.group(1), m.group(2)

    @staticmethod
    def _extract_multi_stop_places(_text: str) -> List[str]:
        return []

    @staticmethod
    def _extract_last_route_cities_from_history(history: List[Dict[str, Any]]) -> List[str]:
        for item in reversed(history or []):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            for tr in meta.get("tool_results", []) or []:
                if tr.get("tool") == "query_route_plan" and tr.get("success"):
                    cities = tr.get("data", {}).get("cities_along_route") or []
                    if len(cities) >= 2:
                        return list(cities)
        return []


def test_departure_time_agent_triggers_on_best_time_with_od() -> None:
    agent = DepartureTimeAgent(_StubService())
    plan = agent.try_plan("成都到重庆的最佳出行时间", [])
    assert plan is not None
    assert plan["intent"] == "route_planning"
    reply = plan["llm_reply"]
    assert "成都" in reply and "重庆" in reply


def test_departure_time_agent_uses_history_when_no_od_in_message() -> None:
    agent = DepartureTimeAgent(_StubService())
    history = [
        {"role": "user", "content": "从成都到重庆怎么走"},
        {
            "role": "assistant",
            "content": "路径规划完成",
            "meta": {
                "tool_results": [
                    {
                        "tool": "query_route_plan",
                        "success": True,
                        "data": {"cities_along_route": ["成都", "重庆"]},
                    }
                ]
            },
        },
    ]
    plan = agent.try_plan("最佳出行时间", history)
    assert plan is not None
    assert "成都" in plan["llm_reply"] and "重庆" in plan["llm_reply"]


def test_departure_time_agent_no_trigger_returns_none() -> None:
    agent = DepartureTimeAgent(_StubService())
    assert agent.try_plan("今天北京天气怎么样", []) is None


def test_orchestrator_prefers_departure_time_over_route_planning_when_both_apply() -> None:
    """关键回归点：含 OD 的「最佳出行时间」问题应命中 DepartureTimeAgent，
    而不是被 RoutePlanningAgent 的纯 OD 规则抢走。"""
    orchestrator = IntentOrchestratorAgent(_StubService())
    plan = orchestrator.plan_rules("成都到重庆的最佳出行时间", [])
    assert plan["intent"] == "route_planning"
    reply = plan.get("llm_reply") or ""
    assert "出发时间" in reply or "出行时间" in reply
    assert not plan.get("actions"), "本意图不应触发路径规划工具调用"
