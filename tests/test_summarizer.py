"""ConversationSummarizer 基础测试。"""

from __future__ import annotations

from intent.summarizer import ConversationSummarizer


def _mk_hist(n: int):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"用户消息{i}", "meta": {}})
        out.append(
            {
                "role": "agent",
                "content": f"助手回复{i}",
                "meta": {
                    "tool_results": [
                        {"tool": "query_route_plan", "success": True, "origin": "南京", "destination": "上海"}
                    ]
                }
                if i == 0
                else {},
            }
        )
    return out


def test_no_summary_when_short() -> None:
    s = ConversationSummarizer(service_agent=None, keep_recent=6)
    h = _mk_hist(2)
    out = s.split(h)
    assert out["summary"] == ""
    assert len(out["recent"]) == len(h)


def test_summarize_older_history() -> None:
    s = ConversationSummarizer(service_agent=None, keep_recent=4)
    h = _mk_hist(8)
    out = s.split(h)
    assert out["summary"].startswith("[历史要点]")
    assert "南京" in out["summary"]
    assert len(out["recent"]) == 4
