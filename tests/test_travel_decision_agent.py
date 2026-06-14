"""Travel Decision Agent tests."""

from __future__ import annotations

from intent.travel_decision_agent import TravelDecisionAgent
from tools_infra.travel_decision_tools import RiskScorer, TravelQueryParser


def test_travel_query_parser_extracts_od_time_mode() -> None:
    parser = TravelQueryParser()
    q = parser.parse("明天上午从北京市朝阳区开车去天津市西青区，适合几点出发？")
    assert q is not None
    assert q.origin == "北京市朝阳区"
    assert q.destination == "天津市西青区"
    assert q.depart_time_text == "明天上午"
    assert q.travel_mode == "driving"


def test_travel_decision_agent_builds_tool_action() -> None:
    plan = TravelDecisionAgent(object()).try_plan("明天上午从北京市朝阳区开车去天津市西青区，适合几点出发？", [])
    assert plan is not None
    assert plan["intent"] == "travel_decision"
    assert plan["actions"][0]["tool"] == "query_travel_decision"
    params = plan["actions"][0]["params"]
    assert params["origin"] == "北京市朝阳区"
    assert params["destination"] == "天津市西青区"


def test_risk_scorer_weather_and_traffic_rules() -> None:
    risk = RiskScorer.score(
        route_result={"success": True},
        weather_blocks=[
            {
                "ok": True,
                "city": "天津市西青区",
                "live": {"weather": "暴雨", "visibility": "300"},
                "forecast": [],
                "aqi": {"aqi": "220"},
            }
        ],
        highway_results=[{"success": True, "target": "G2", "congestion_level": "中度拥堵"}],
    )
    assert risk.risk_score >= 140
    assert risk.risk_level == "Critical"
    assert any("暴雨" in x for x in risk.main_risks)
