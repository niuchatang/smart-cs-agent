"""ParkingLotAgent 行为回归测试（按 SPEC docs/specs/parking_lot_agent.md）。

测试矩阵 T1~T7，覆盖：
- 单点查询（含/不含修饰词）
- 沿途停车（基于历史路线）
- 触发词但无上下文 → 引导
- 无触发词 → None
- 异常 history → 不抛
- orchestrator 集成命中
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from intent.orchestrator_agent import IntentOrchestratorAgent
from intent.parking_lot_agent import ParkingLotAgent


class _StubService:
    """最小化 service_agent，仅暴露本测试用到的辅助方法。

    其余所有 `_extract_*` / `_is_*` 类辅助方法走 `__getattr__` 兜底返回安全默认值，
    这样 orchestrator 沿规则链调用时不会因缺方法抛 AttributeError。
    """

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

    def __getattr__(self, name: str):
        if name.startswith("_is_"):
            return lambda *_a, **_kw: False
        if name.startswith("_extract_") or name.startswith("_clean_") or name.startswith("_get_"):
            return lambda *_a, **_kw: []
        raise AttributeError(name)


def _hist_with_route(cities: List[str]) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "从 {} 到 {}".format(cities[0], cities[-1])},
        {
            "role": "assistant",
            "content": "已规划路径",
            "meta": {
                "tool_results": [
                    {
                        "tool": "query_route_plan",
                        "success": True,
                        "data": {"cities_along_route": cities},
                    }
                ]
            },
        },
    ]


# ---- T1: 单点停车查询 ---------------------------------------------------------

def test_t1_single_place_parking_query() -> None:
    agent = ParkingLotAgent(_StubService())
    plan = agent.try_plan("南京南站附近有什么停车场", [])
    assert plan is not None
    assert plan["intent"] == "route_planning"
    assert plan["actions"] == []
    assert plan["used_llm"] is False
    assert "南京南站" in plan["llm_reply"]


# ---- T2: 沿途停车（基于历史路线） ---------------------------------------------

def test_t2_along_route_uses_history() -> None:
    agent = ParkingLotAgent(_StubService())
    plan = agent.try_plan("沿途哪里能停车", _hist_with_route(["成都", "内江", "重庆"]))
    assert plan is not None
    reply = plan["llm_reply"]
    assert "成都" in reply and "重庆" in reply
    assert plan["confidence"] >= 0.8


# ---- T3: 充电类停车 -----------------------------------------------------------

def test_t3_ev_charging_parking() -> None:
    agent = ParkingLotAgent(_StubService())
    plan = agent.try_plan("沿途停车带充电的", _hist_with_route(["成都", "内江", "重庆"]))
    assert plan is not None
    assert "充电" in plan["llm_reply"]


# ---- T4: 触发词但无任何地点 ---------------------------------------------------

def test_t4_trigger_only_no_place_returns_guidance() -> None:
    agent = ParkingLotAgent(_StubService())
    plan = agent.try_plan("停车", [])
    assert plan is not None
    assert plan["confidence"] <= 0.75
    assert ("地点" in plan["llm_reply"]) or ("起终点" in plan["llm_reply"])


# ---- T5: 无触发词必返 None ---------------------------------------------------

def test_t5_no_trigger_returns_none() -> None:
    agent = ParkingLotAgent(_StubService())
    assert agent.try_plan("今天北京天气怎么样", []) is None
    assert agent.try_plan("", []) is None


# ---- T6: 异常 history 不抛 ----------------------------------------------------

def test_t6_robust_to_bad_history() -> None:
    agent = ParkingLotAgent(_StubService())
    plan = agent.try_plan("哪里停车", None)  # type: ignore[arg-type]
    assert plan is not None  # 走 fallback
    assert plan["used_llm"] is False


# ---- T7: orchestrator 集成 ---------------------------------------------------

def test_t7_orchestrator_routes_to_parking_lot() -> None:
    orch = IntentOrchestratorAgent(_StubService())
    plan = orch.plan_rules("南京南站附近停车场", [])
    assert plan is not None
    assert plan.get("meta", {}).get("ext_agent") == "parking_lot"
    assert plan["intent"] == "route_planning"


# ---- T8: 当前消息含具体地点时，必须以"地点"为准，不被历史路线劫持 ---------
# 真实回归 case：用户先问过 北京→南京 路径规划（cities 进了历史），
# 接着问"南京南站附近的停车场" —— 必须命中单点查询，**不能**因为历史有
# cities 就被沿途模式抢走（这是用户在 v1 上观察到的 bug）。

def test_t8_explicit_place_overrides_history_cities() -> None:
    agent = ParkingLotAgent(_StubService())
    history = _hist_with_route(["北京", "东城", "廊坊", "沧州", "德州", "泰安", "南京"])
    plan = agent.try_plan("南京南站附近的停车场", history)
    assert plan is not None
    reply = plan["llm_reply"]
    # 必须谈"南京南站"，不能去渲染北京/廊坊/沧州那串沿途城市
    assert "南京南站" in reply
    assert "北京" not in reply, "存在历史路线时不应被沿途模式劫持"
    assert "廊坊" not in reply
    assert "沧州" not in reply


# ---- T9: 显式"沿途"语义即使消息里也提到地名，仍走沿途模式 ------------------

def test_t9_along_keyword_keeps_route_mode_even_with_place_in_text() -> None:
    agent = ParkingLotAgent(_StubService())
    history = _hist_with_route(["成都", "内江", "重庆"])
    # "沿途" 关键字 + 提到了"重庆" —— 应当走沿途分段，而不是只看重庆
    plan = agent.try_plan("沿途有重庆的停车场吗", history)
    assert plan is not None
    reply = plan["llm_reply"]
    assert "成都" in reply and "内江" in reply and "重庆" in reply
