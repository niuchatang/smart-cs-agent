"""
投诉分级智能体（Complaint Triage Agent）

- 输入：用户消息；
- 输出：`{severity: low/medium/high/urgent, reasons: [...], suggest_handoff: bool}`；
- 实现：优先关键词规则；若主智能体已启用 LLM，则再调用一次 `ChatOpenAI` 做情感/紧急度复核；
- 不直接接管意图规划（保留给现有 General tail 规则）；由 main.py 在渲染投诉/转人工
  相关话术时调用，提升分级与处理建议的一致性。

使用方法（`main.py` 伪码）：

```python
from intent.complaint_triage_agent import ComplaintTriageAgent
triager = ComplaintTriageAgent(self)
triage = triager.triage(message)
if triage["severity"] == "urgent":
    # 自动转人工 + 高优先级
```
"""

from __future__ import annotations

from typing import Any, Dict, List

_URGENT = ("生命", "危险", "受伤", "受伤了", "急救", "起火", "着火", "爆炸", "有人晕倒")
_HIGH = ("严重", "非常生气", "投诉到底", "曝光", "起诉", "报警", "媒体")
_MEDIUM = ("投诉", "不满", "差评", "多次", "反复", "延误", "退款")
_LOW = ("建议", "咨询", "问下", "了解")


class ComplaintTriageAgent:
    name = "complaint_triage"
    priority = 100  # 不自动注册到规则链，仅供主流程显式调用

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def triage(self, message: str) -> Dict[str, Any]:
        text = (message or "").strip()
        if not text:
            return {"severity": "low", "reasons": [], "suggest_handoff": False, "source": "empty"}

        reasons: List[str] = []
        severity = "low"
        if any(k in text for k in _URGENT):
            severity = "urgent"
            reasons.append("包含安全/紧急关键词")
        elif any(k in text for k in _HIGH):
            severity = "high"
            reasons.append("含强烈情绪或法律动作关键词")
        elif any(k in text for k in _MEDIUM):
            severity = "medium"
            reasons.append("含一般投诉/退款/延误关键词")
        elif any(k in text for k in _LOW):
            severity = "low"
            reasons.append("偏咨询/建议")

        handoff = severity in ("urgent", "high")

        llm_checked = False
        if getattr(self._svc, "llm", None) is not None and severity in ("medium", "high"):
            try:
                llm_checked = True
                verdict = self._llm_recheck(text)
                if verdict == "upgrade" and severity != "urgent":
                    severity = "high"
                    reasons.append("LLM 复核升级")
                    handoff = True
                elif verdict == "downgrade" and severity == "high":
                    severity = "medium"
                    reasons.append("LLM 复核下调")
                    handoff = False
            except Exception:
                pass

        return {
            "severity": severity,
            "reasons": reasons,
            "suggest_handoff": handoff,
            "llm_checked": llm_checked,
        }

    def _llm_recheck(self, text: str) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            """
判断用户发言的投诉紧急度。只输出单词之一：keep / upgrade / downgrade。
- upgrade：用户表达了威胁性或安全问题被忽视的强烈情绪；
- downgrade：更像建议或一般咨询，不需要立刻人工介入；
- keep：维持既有规则判定。

用户原文：{text}
""".strip()
        )
        chain = prompt | self._svc.llm | StrOutputParser()
        out = str(chain.invoke({"text": text})).strip().lower()
        if "upgrade" in out:
            return "upgrade"
        if "downgrade" in out:
            return "downgrade"
        return "keep"
