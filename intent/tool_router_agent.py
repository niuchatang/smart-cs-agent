"""
工具路由智能体（Tool Router Agent）

- 面向主智能体工具执行阶段：当一条 action 失败 / 数据为空时，按「备选工具链」
  自动改写成另一个工具再试一次（例如 `query_weather` 单城失败 → 同城换一个 provider 参数）；
- 本文件**不**替换 `_execute_actions`，而是提供一个 helper：
  `suggest_fallback(action, result) -> Optional[Action]`，
  由 main.py 在合适位置调用（或后续重构时接入）；
- 若不启用，系统行为不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


FALLBACK_RULES: List[Dict[str, Any]] = [
    {
        "when_tool": "query_weather",
        "when_failed": True,
        "to": {"tool": "query_weather", "extra_params": {"provider": "open-meteo"}},
    },
    {
        "when_tool": "query_route_plan",
        "when_failed": True,
        "to": {"tool": "query_route_plan", "extra_params": {"engine": "osrm"}},
    },
    {
        "when_tool": "query_highway_condition",
        "when_empty_incidents": True,
        "to": {"tool": "query_highway_condition", "extra_params": {"include_forecast": True}},
    },
]


class ToolRouterAgent:
    name = "tool_router"
    priority = 1000

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def suggest_fallback(
        self, action: Dict[str, Any], result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        tool = str(action.get("tool", ""))
        params = dict(action.get("params", {}) or {})
        ok = bool(result.get("success"))
        empty_inc = tool == "query_highway_condition" and ok and not result.get("incidents")

        for rule in FALLBACK_RULES:
            if rule.get("when_tool") != tool:
                continue
            if rule.get("when_failed") and ok:
                continue
            if rule.get("when_empty_incidents") and not empty_inc:
                continue
            target = rule["to"]
            new_params = {**params, **target.get("extra_params", {})}
            if new_params == params:
                return None
            return {"tool": target["tool"], "params": new_params}
        return None
