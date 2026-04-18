"""
无障碍出行智能体（Accessibility Agent）

- 回答轮椅、盲道、无障碍卫生间、车站无障碍通道、导盲犬乘车等常见问题；
- 纯规则回复，仅在相关话题下触发，避免干扰主流程；
- 意图归属：`unknown`（作为知识型回复），将通过 `llm_reply` 直接输出内容。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_TRIGGERS = (
    "无障碍",
    "轮椅",
    "坡道",
    "盲道",
    "导盲犬",
    "无障碍卫生间",
    "升降平台",
    "行动不便",
    "老人出行",
    "视障",
    "听障",
    "残疾人",
    "带老人",
    "母婴",
)


class AccessibilityAgent:
    name = "accessibility"
    priority = 60

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def try_plan(
        self, message: str, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        text = (message or "").strip()
        if not text:
            return None
        if not any(k in text for k in _TRIGGERS):
            return None

        lines: List[str] = ["关于无障碍/随行出行，给你一些通用建议："]
        if any(k in text for k in ("轮椅", "行动不便", "残疾人", "带老人")):
            lines += [
                "- 选择带「直梯/无障碍通道」标识的车站出入口；",
                "- 高铁/机场可提前 48 小时通过 12306 或航司官网预约轮椅与专人接送；",
                "- 出租/网约车可优先选择平台的「无障碍」或「关爱」车型（保有量有限，建议提前下单）。",
            ]
        if any(k in text for k in ("盲道", "视障", "导盲犬")):
            lines += [
                "- 根据现行《残疾人保障法》与多地条例，导盲犬可进入公共交通工具与站点；",
                "- 地铁闸机旁通常设有盲文按钮与工作人员呼叫按钮；",
                "- 高铁站请优先到「重点旅客候车区」，可预约全程陪同。",
            ]
        if any(k in text for k in ("无障碍卫生间", "母婴")):
            lines += [
                "- 高速服务区大多提供无障碍卫生间与第三卫生间；",
                "- 大型综合枢纽（机场/高铁南北站）设独立母婴室，位置见站内指示牌。",
            ]
        if "听障" in text:
            lines += [
                "- 可使用 12306 / 交通出行类 App 内的「在线字幕/图示」功能；",
                "- 车站广播信息也会在大屏同步显示，建议留意显示屏与手环提醒设置。",
            ]
        lines.append(
            "\n如果你有具体起终点或车站，我可以再给出该站/路线的无障碍设施与联系方式（需你先规划路线）。"
        )

        return {
            "intent": "unknown",
            "confidence": 0.8,
            "actions": [],
            "llm_reply": "\n".join(lines),
            "used_llm": False,
        }
