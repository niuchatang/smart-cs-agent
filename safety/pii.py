"""
PII 脱敏工具函数

与 `intent.guardrail_agent._PII_PATTERNS` 规则等价；分离到独立模块供
日志写入、历史持久化、对外返回体裁剪等位置复用。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

PII_KINDS: Tuple[str, ...] = ("phone", "idcard", "bankcard", "email", "plate", "addr_detail")

_PII_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    (
        "idcard",
        re.compile(
            r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dxX](?!\d)"
        ),
    ),
    ("bankcard", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("plate", re.compile(r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}")),
    ("addr_detail", re.compile(r"\d+(?:号楼|号|栋|单元|室|层|户)")),
]


def mask_pii(text: str) -> str:
    out = text or ""
    for label, pat in _PII_PATTERNS:
        out = pat.sub(lambda m, lb=label: f"[{lb.upper()}]", out)
    return out


def pii_hits(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, pat in _PII_PATTERNS:
        n = len(pat.findall(text or ""))
        if n > 0:
            counts[label] = n
    return counts
