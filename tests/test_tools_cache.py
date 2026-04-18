"""tools_infra.cache / registry 基础测试。"""

from __future__ import annotations

import time

from tools_infra.cache import TTLCache, cached
from tools_infra.registry import ToolRunner, ToolSpec


def test_ttl_cache_expire() -> None:
    c = TTLCache(max_size=8, default_ttl=0.05)
    c.set("k", 1)
    assert c.get("k") == 1
    time.sleep(0.06)
    assert c.get("k") is None


def test_cached_decorator_reuses_value() -> None:
    c = TTLCache(max_size=8, default_ttl=10)
    calls = {"n": 0}

    @cached(c, key_fn=lambda x: ("f", x))
    def f(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1


def test_tool_runner_retries_then_fails() -> None:
    calls = {"n": 0}

    def raising(**kw):
        calls["n"] += 1
        raise RuntimeError("boom")

    spec = ToolSpec(name="x", call=raising, max_retry=2, retry_backoff=0.0)
    runner = ToolRunner()
    raised = False
    try:
        runner.call(spec)
    except RuntimeError:
        raised = True
    assert raised
    assert calls["n"] == 3


def test_tool_runner_fallback_used() -> None:
    def raising(**kw):
        raise RuntimeError("boom")

    def fb(**kw):
        return "ok"

    spec = ToolSpec(name="y", call=raising, max_retry=0, fallback=fb)
    runner = ToolRunner()
    assert runner.call(spec) == "ok"
