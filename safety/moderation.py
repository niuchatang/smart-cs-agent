"""
出站文本审核（Moderation）

- 规则：过滤夸大/承诺性/法律性承诺关键词；
- 不对外呼接真实内容审核 API（本仓库专注自包含）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

_FORBIDDEN: Tuple[str, ...] = (
    "保证百分百通过",
    "包过",
    "一定会退款",
    "必胜",
    "官方承诺",
    "绝对安全",
    "绝对不会",
)

_REPLACEMENT = "[已移除的承诺性表述]"


@dataclass
class ModerationResult:
    safe: bool = True
    hits: List[str] = field(default_factory=list)
    sanitized: str = ""


def scan_forbidden(text: str) -> ModerationResult:
    hits = [kw for kw in _FORBIDDEN if kw in (text or "")]
    res = ModerationResult(safe=not hits, hits=hits, sanitized=text or "")
    if hits:
        res.sanitized = sanitize_reply(text or "")
    return res


def sanitize_reply(text: str) -> str:
    out = text or ""
    for kw in _FORBIDDEN:
        out = out.replace(kw, _REPLACEMENT)
    return out
