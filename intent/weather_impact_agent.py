"""
天气-行车影响智能体（Weather Impact Agent）

- 专注把「天气」转成「行车风险 + 行动建议」；与现有 `WeatherDialogAgent` 不冲突：
  后者做多轮选城市/沿途循环；本智能体处理「下雨/雾/风/雪 → 能不能开/要不要绕/几点出发」类句；
- 不调用任何外部系统，只依据关键词给规则化建议；
- 意图归属：`weather_query`（主意图仍由既有模板渲染天气摘要），我们通过 `llm_reply`
  输出**行车影响**段落，替代默认的「请补充城市」类兜底。

触发词（需要**同时**出现天气 + 行车 / 影响 类词）：
  天气/雨/雪/雾/风/霾 + 能开吗/影响/走不走/要不要/绕行/推迟/等雨/高速/山路 等。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_WX_TOKENS = ("天气", "下雨", "降雨", "暴雨", "小雨", "中雨", "大雨", "雾", "大雾", "雪", "降雪", "台风", "大风", "霾", "高温")
_IMPACT_TOKENS = (
    "能开吗",
    "能走吗",
    "开不开",
    "走不走",
    "要不要绕",
    "影响",
    "推迟",
    "等等",
    "等雨",
    "停不停",
    "安全吗",
    "能出行吗",
    "适合出行",
    "还能自驾",
    "风险",
)


def _summary(wx: str) -> str:
    if any(k in wx for k in ("暴雨", "大雨")):
        return (
            "大到暴雨会显著降低能见度、制动距离变长且易形成积水。建议：\n"
            "- 时速降 20 km/h，打开近光灯与雾灯，勿开远光；\n"
            "- 避开路面积水与下凹立交；\n"
            "- 如降雨时段不长（1-2 小时内）可等雨后再走；\n"
            "- 山区/隧道多路段建议延后或绕行国道。"
        )
    if any(k in wx for k in ("大雾", "浓雾", "雾")):
        return (
            "有雾路段高速通常会管制或降速，请提前查询交管公告：\n"
            "- 能见度 <200m 建议推迟出行；\n"
            "- 必须出发时保持前车 3-5 倍日常车距，开启雾灯与示廓灯，严禁远光；\n"
            "- 勿并线超车，注意服务区汇入车辆。"
        )
    if any(k in wx for k in ("降雪", "大雪", "雪", "冰冻")):
        return (
            "冬季降雪/结冰路段需要尤其谨慎：\n"
            "- 起步缓加油，避免急制动；\n"
            "- 下坡与桥面易积冰，提前减速并拉大跟车距离；\n"
            "- 车辆需配冬季胎或链条；高速可能临时封闭，出发前请查路况。"
        )
    if any(k in wx for k in ("台风", "大风")):
        return (
            "强风天高速侧风易影响车辆稳定：\n"
            "- 双手握方向盘，避免单手；\n"
            "- 减少超车与变道，远离大型挂车；\n"
            "- 如发布台风红色预警，建议推迟出行或改乘公共交通。"
        )
    if "高温" in wx:
        return (
            "高温长途行驶请注意爆胎与疲劳驾驶：\n"
            "- 出发前检查胎压，建议低 0.1-0.2 bar；\n"
            "- 每 2 小时进一次服务区，适量补水；\n"
            "- 停车避免暴晒，留意车内儿童/宠物。"
        )
    if "霾" in wx:
        return "雾霾能见度不佳与雾天类似；请提前开启雾灯与示廓灯，降速慢行。"
    return "如需更具体建议，请说明当天天气（如「明早大雨」）和行车路线。"


class WeatherImpactAgent:
    name = "weather_impact"
    priority = 50

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        has_wx = any(k in text for k in _WX_TOKENS)
        has_impact = any(k in text for k in _IMPACT_TOKENS) or bool(
            re.search(r"(?:还|能|可以).{0,3}(?:出行|出发|走|开)", text)
        )
        if not (has_wx and has_impact):
            return None

        wx = next((k for k in _WX_TOKENS if k in text), "天气")
        body = _summary(wx)
        return {
            "intent": "weather_query",
            "confidence": 0.82,
            "actions": [],
            "llm_reply": f"关于「{wx}」对出行的影响建议：\n{body}",
            "used_llm": False,
        }
