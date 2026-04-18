"""
多语言 / 方言前处理智能体（Multilingual Agent）

- 在进入 `UserIntentAgent._plan` 之前做「语种/方言检测」；
- 如果检测到非中文主流语种，返回提示用户所用语种，并把「翻译成普通话」的版本作为
  `normalized_text` 返回给调用侧。主智能体可选择用 `normalized_text` 进入原有流程；
- 使用纯启发式 + 可选 LLM（若启用）完成翻译，不依赖外部翻译 API。

主要用法：
```python
from intent.multilingual_agent import MultilingualAgent
mla = MultilingualAgent(self)
info = mla.detect(message)
if info["action"] == "translate":
    message = info["normalized_text"] or message
```
"""

from __future__ import annotations

import re
from typing import Any, Dict

_CJK = re.compile(r"[\u4e00-\u9fff]")
_HIRAKATA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[a-zA-Z]")
_CANTONESE_HINT = ("嘅", "喺", "係", "佢", "冇", "咁", "啲", "嘢", "點解")


class MultilingualAgent:
    name = "multilingual"
    priority = 5  # 预处理阶段（不参与常规 try_plan 调度）

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def detect(self, message: str) -> Dict[str, Any]:
        text = (message or "").strip()
        if not text:
            return {"language": "unknown", "action": "passthrough"}

        lang = "zh"
        if _HIRAKATA.search(text):
            lang = "ja"
        elif _HANGUL.search(text):
            lang = "ko"
        elif _CYRILLIC.search(text):
            lang = "ru"
        elif _LATIN.search(text) and not _CJK.search(text) and len(text) >= 6:
            lang = "en"

        dialect = None
        if lang == "zh" and any(k in text for k in _CANTONESE_HINT):
            dialect = "yue"

        action = "passthrough"
        normalized_text = None
        if lang != "zh" or dialect == "yue":
            action = "translate"
            normalized_text = self._translate_to_zh(text, lang if lang != "zh" else "yue")

        return {
            "language": lang,
            "dialect": dialect,
            "action": action,
            "normalized_text": normalized_text,
        }

    def _translate_to_zh(self, text: str, src: str) -> str:
        if getattr(self._svc, "llm", None) is None:
            return text
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                "将以下文本翻译成简体普通话的短句，只输出译文，不要解释。原语种：{src}\n原文：{text}"
            )
            chain = prompt | self._svc.llm | StrOutputParser()
            out = str(chain.invoke({"src": src, "text": text})).strip()
            return out or text
        except Exception:
            return text
