"""Travel Decision Agent.

Parses OD + departure-time + travel-mode questions and produces a
query_travel_decision tool action. The actual route/weather/GIS/risk workflow
is executed by CustomerServiceAgent so it can reuse existing tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools_infra.travel_decision_tools import TravelQueryParser


class TravelDecisionAgent:
    name = "travel_decision"
    priority = 15

    def __init__(self, service_agent: Any, parser: TravelQueryParser | None = None) -> None:
        self._svc = service_agent
        self._parser = parser or TravelQueryParser()

    def try_plan(self, message: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        msg = (message or "").strip()
        if not msg or not self._parser.is_travel_decision_query(msg):
            return None
        parsed = self._parser.parse(msg)
        if parsed is None or not parsed.origin or not parsed.destination:
            return None
        return {
            "intent": "travel_decision",
            "confidence": 0.92,
            "actions": [{"tool": "query_travel_decision", "params": parsed.model_dump()}],
            "used_llm": False,
            "meta": {"travel_decision_query": parsed.model_dump()},
        }
