import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.agent import MemoryAgent
from memory.models import MemoryContext


class _Svc:
    def _extract_route_endpoints(self, message: str):
        if "朝阳区" in message and "亦庄" in message:
            return "朝阳区", "亦庄"
        return "", ""


def test_memory_commute_extract_and_enrich():
    agent = MemoryAgent(_Svc())
    user = "test_user_memory"
    agent.extract_and_save(
        user,
        "我每天从朝阳区到亦庄上班",
        {"intent": "route_planning"},
        [],
    )
    ctx = agent.load_context(user)
    assert ctx.profile_get("commute_origin") == "朝阳区"
    assert ctx.profile_get("commute_dest") == "亦庄"
    enriched = agent.enrich_message("今天几点出发", ctx)
    assert "朝阳区" in enriched and "亦庄" in enriched


def test_memory_search_tool():
    agent = MemoryAgent()
    user = "test_user_search"
    agent.tools.save.run(user, "travel_mode", "driving")
    hit = agent.search(user, "travel")
    assert hit["success"] is True
    assert any(p["key"] == "travel_mode" for p in hit["profiles"])
