"""
安全层（safety/）

- `pii.py`：PII 脱敏（手机号、身份证、车牌、银行卡、邮箱、门牌号）；
- `moderation.py`：出站承诺/违规词检测。

`intent.guardrail_agent` 已内置同类规则，作为完整智能体使用；本模块提供
**纯函数接口**，便于在 main.py 日志/持久化/出站渲染等非智能体位置直接复用。
"""

from .moderation import ModerationResult, sanitize_reply, scan_forbidden
from .pii import PII_KINDS, mask_pii, pii_hits

__all__ = [
    "mask_pii",
    "pii_hits",
    "PII_KINDS",
    "sanitize_reply",
    "scan_forbidden",
    "ModerationResult",
]
