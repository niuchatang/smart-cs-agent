"""
满意度 / 对话闭环智能体（Satisfaction Agent）

- 纯离线/显式调用；不参与规则链自动路由；
- 在会话结束（N 分钟无消息 / 用户说「没其他问题」）时，由主智能体触发
  `prompt_survey()` 生成满意度提问；用户回复分数后 `record_score()` 持久化；
- 评分低于 3 分的对话可在批处理时送 `EvalAgent` 做样本分析。

持久化：默认写入 `data/satisfaction.jsonl`（每行 {session_id, score, comment, ts}）。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional


_CLOSE_SIGNALS = (
    "没其他问题",
    "没其他问题了",
    "没了",
    "不用了",
    "谢谢",
    "多谢",
    "好的谢谢",
    "可以了",
    "结束",
    "再见",
    "拜拜",
)


class SatisfactionAgent:
    name = "satisfaction"
    priority = 200

    def __init__(self, service_agent: Any, data_path: str = "data/satisfaction.jsonl") -> None:
        self._svc = service_agent
        self._path = Path(data_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def should_prompt_survey(message: str) -> bool:
        t = (message or "").strip()
        return any(s in t for s in _CLOSE_SIGNALS)

    def prompt_survey(self) -> str:
        return (
            "本次服务就到这里啦。请为本次对话打个分（1-5 分，5 为非常满意），"
            "也欢迎一句话告诉我可以怎么改进，例如「4 分，回答还能更简洁些」。"
        )

    _RE = re.compile(r"([1-5])\s*分?")

    def parse_score(self, message: str) -> Optional[Dict[str, Any]]:
        t = (message or "").strip()
        m = self._RE.search(t)
        if not m:
            return None
        score = int(m.group(1))
        comment = t.replace(m.group(0), "").strip(" ,，。.")
        return {"score": score, "comment": comment}

    def record_score(self, session_id: str, score: int, comment: str = "") -> None:
        line = {
            "session_id": session_id,
            "score": int(score),
            "comment": comment or "",
            "ts": int(time.time()),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
