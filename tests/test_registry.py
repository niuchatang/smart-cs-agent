"""AgentRegistry 行为测试。"""

from __future__ import annotations

from intent.agent_registry import AgentRegistry


class _A:
    name = "a"
    priority = 10

    def try_plan(self, msg, hist):
        if "a" in msg:
            return {"intent": "unknown", "actions": [], "confidence": 0.6}
        return None


class _B:
    name = "b"
    priority = 5

    def try_plan(self, msg, hist):
        if "b" in msg:
            return {"intent": "unknown", "actions": [], "confidence": 0.7}
        return None


def test_priority_order() -> None:
    reg = AgentRegistry()
    reg.register(_A())
    reg.register(_B())
    names = [a.name for a in reg.agents()]
    assert names == ["b", "a"]


def test_try_plan_first_match_wins() -> None:
    reg = AgentRegistry()
    reg.register(_A())
    reg.register(_B())
    plan = reg.try_plan("a", [])
    assert plan is not None
    assert plan["meta"]["ext_agent"] == "a"


def test_try_plan_no_match_returns_none() -> None:
    reg = AgentRegistry()
    reg.register(_A())
    assert reg.try_plan("nothing", []) is None
