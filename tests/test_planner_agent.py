import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.models import MemoryContext
from planner.rules import build_plan, should_plan


def test_should_plan_departure_od():
    msg = "明天从北京到天津适合几点出发"
    assert should_plan(msg, []) is True
    plan = build_plan(msg, [], MemoryContext())
    assert plan is not None
    assert plan.tasks[0].tool == "query_travel_decision"


def test_should_plan_future_trip():
    msg = "未来三天适合去天津吗"
    assert should_plan(msg, []) is True
    plan = build_plan(msg, [], MemoryContext())
    assert plan is not None
    assert plan.tasks[0].tool == "query_weather"


def test_planner_uses_memory_commute():
    ctx = MemoryContext(profiles={"commute_origin": "朝阳区", "commute_dest": "亦庄"})
    plan = build_plan("明天适合几点出发", [], ctx)
    assert plan is not None
    assert plan.tasks[0].params.get("origin") == "朝阳区"
    assert plan.tasks[0].params.get("destination") == "亦庄"
