"""用户消息信号：闲聊 vs 出行知识问句，供意图回落与回复渲染共用。"""

from __future__ import annotations

_CASUAL_MARKERS = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "早上好",
    "晚上好",
    "晚安",
    "聊聊",
    "聊聊天",
    "随便聊",
    "心情",
    "不开心",
    "郁闷",
    "无聊",
    "烦",
    "累",
)

_TRAVEL_KNOWLEDGE_MARKERS = (
    "路线",
    "规划",
    "地铁",
    "公交",
    "高速",
    "路况",
    "票价",
    "收费",
    "退票",
    "改签",
    "失物",
    "投诉",
    "天气",
    "拥堵",
    "事故",
    "管制",
    "机场",
    "快线",
    "换乘",
    "站点",
    "出发",
    "到达",
    "多久",
    "多少钱",
    "怎么坐",
    "怎么走",
    "在哪",
    "哪里",
)


def is_travel_knowledge_query(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return any(k in text for k in _TRAVEL_KNOWLEDGE_MARKERS)


def is_casual_chitchat(message: str) -> bool:
    text = (message or "").strip()
    if not text or is_travel_knowledge_query(text):
        return False
    if any(k in text for k in _CASUAL_MARKERS):
        return True
    if len(text) <= 24 and not any(ch.isdigit() for ch in text):
        if "?" not in text and "？" not in text and "吗" not in text:
            if not any(k in text for k in ("怎么", "如何", "多少", "哪", "几")):
                return True
    return False


CASUAL_CHITCHAT_FALLBACK_REPLY = (
    "你好呀，我是智慧交通客服助手。出行相关的问题随时可以问我；"
    "要是想随便聊聊，我也愿意听你说说。"
)
