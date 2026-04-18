"""
安全护栏智能体（Guardrail Agent）

功能：
1) **入站检测**：提示词注入（「忽略以上指令 / 扮演 / 打印 system prompt」等）；
2) **PII 脱敏**：身份证号、手机号、车牌、银行卡号、邮箱、详细门牌号；
3) **出站过滤**：承诺性/法律性 / 违规话术关键词检查。

所有结果均为 **纯规则**，`main.py` 可选择在：
- `chat()` 入口调用 `scan_inbound(message)`；
- 发给 LLM 前调用 `mask_pii(text)`；
- `_render_reply` 末端调用 `scan_outbound(reply)`。

均不依赖外部服务；若需更严格审核，可以再接真实内容安全 API（本项目忽略）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List


_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|above)\s+instructions"),
    re.compile(r"忽略(以上|之前|上面)(的)?(所有)?指令"),
    re.compile(r"(?i)you\s+are\s+now|act\s+as|pretend\s+to\s+be"),
    re.compile(r"(现在起|接下来)你(要|必须|就是)(扮演|假装|充当)"),
    re.compile(r"(?i)print\s+(your|the)\s+system\s+prompt"),
    re.compile(r"(显示|打印|输出)(system\s*prompt|你的提示词|系统提示)"),
    re.compile(r"(?i)reveal\s+(your|the)\s+api\s+key"),
]

_PII_PATTERNS = [
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("idcard", re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dxX](?!\d)")),
    ("bankcard", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("plate", re.compile(r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}")),
    ("addr_detail", re.compile(r"\d+(?:号楼|号|栋|单元|室|层|户)")),
]

_OUTBOUND_FORBIDDEN = [
    "保证百分百通过",
    "包过",
    "一定会退款",
    "必胜",
    "官方承诺",
]


@dataclass
class GuardrailInboundResult:
    safe: bool = True
    injection: bool = False
    reasons: List[str] = field(default_factory=list)
    masked_message: str = ""


@dataclass
class GuardrailOutboundResult:
    safe: bool = True
    reasons: List[str] = field(default_factory=list)
    sanitized: str = ""


class GuardrailAgent:
    name = "guardrail"
    priority = 0

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def scan_inbound(self, message: str) -> GuardrailInboundResult:
        text = message or ""
        res = GuardrailInboundResult(safe=True, injection=False, masked_message=text)
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                res.safe = False
                res.injection = True
                res.reasons.append(f"injection: {pat.pattern[:40]}…")
                break
        res.masked_message = self.mask_pii(text)
        return res

    def mask_pii(self, text: str) -> str:
        out = text or ""
        for label, pat in _PII_PATTERNS:
            out = pat.sub(lambda m, lb=label: f"[{lb.upper()}]", out)
        return out

    def scan_outbound(self, reply: str) -> GuardrailOutboundResult:
        text = reply or ""
        res = GuardrailOutboundResult(safe=True, sanitized=text)
        for kw in _OUTBOUND_FORBIDDEN:
            if kw in text:
                res.safe = False
                res.reasons.append(f"forbidden: {kw}")
        if not res.safe:
            sanitized = text
            for kw in _OUTBOUND_FORBIDDEN:
                sanitized = sanitized.replace(kw, "[已移除的承诺性表述]")
            res.sanitized = sanitized
        return res

    def soft_decline_reply(self) -> str:
        return (
            "为了安全起见，我不会按绕开系统指令/泄露内部提示/伪装身份等方式回复。"
            "如果你有具体的出行/客服问题（路况、路径、天气、退改签、失物、投诉等），可以直接告诉我。"
        )
