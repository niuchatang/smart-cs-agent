import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent.cognitive_orchestrator import CognitiveOrchestrator


def test_cognitive_orchestrator_planner_branch():
    svc = MagicMock()
    svc.intent_agent.parse.return_value = {
        "intent": "unknown",
        "confidence": 0.5,
        "actions": [],
        "used_llm": False,
    }
    orch = CognitiveOrchestrator(svc)
    plan = orch.parse(
        "明天从北京到天津适合几点出发",
        [],
        [],
        user_id="u1",
    )
    assert plan.get("meta", {}).get("planner") is True
    assert plan["actions"][0]["tool"] == "query_travel_decision"


def test_cognitive_orchestrator_memory_enrich_fallback():
    svc = MagicMock()
    svc.intent_agent.parse.return_value = {
        "intent": "weather_query",
        "confidence": 0.8,
        "actions": [{"tool": "query_weather", "params": {"cities": ["北京"]}}],
        "used_llm": False,
    }
    orch = CognitiveOrchestrator(svc)
    orch.memory.extract_and_save("u2", "我每天从朝阳区到亦庄上班", {"intent": "route_planning"}, [])
    plan = orch.parse("今天几点出发", [], [], user_id="u2")
    assert plan.get("meta", {}).get("planner") is True
    assert plan["actions"][0]["params"]["origin"] == "朝阳区"
    assert plan["actions"][0]["params"]["destination"] == "亦庄"
