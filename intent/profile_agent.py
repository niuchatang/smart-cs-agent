"""
用户画像智能体（Profile Agent，离线/异步）

- 在每轮对话**结束后**异步调用，把最新消息里的出行偏好抽取并合并到画像文件；
- 不参与实时规划；主智能体可读 `profile.get(user_id)` 在下一轮 prompt 中注入一段偏好摘要；
- 存储：`data/user_profiles.json`（简单字典，线程不安全但对单进程 demo 足够）。

抽取内容（纯规则，避免依赖 LLM）：
- 最近 10 条起终点；
- 最近 5 条高速编号；
- 常用出发时间段（早/晚高峰偏好）；
- 是否偏好高速/国道；
- 是否需要充电桩/母婴室 / 无障碍。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List


def _dedup(seq: List[str], limit: int) -> List[str]:
    out: List[str] = []
    for x in seq:
        t = (x or "").strip()
        if t and t not in out:
            out.append(t)
    return out[-limit:]


class ProfileAgent:
    name = "profile"
    priority = 500

    def __init__(self, service_agent: Any, path: str = "data/user_profiles.json") -> None:
        self._svc = service_agent
        self._path = Path(path)
        self._lock = RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def get(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._cache.get(user_id, {}))

    def summary(self, user_id: str) -> str:
        p = self.get(user_id)
        if not p:
            return ""
        lines: List[str] = []
        if p.get("frequent_od"):
            lines.append("常用路线：" + "；".join(p["frequent_od"][-5:]))
        if p.get("frequent_highways"):
            lines.append("常关注高速：" + "、".join(p["frequent_highways"][-5:]))
        if p.get("time_pref"):
            lines.append("出发偏好：" + p["time_pref"])
        if p.get("flags"):
            lines.append("偏好/需求：" + "、".join(p["flags"]))
        return " | ".join(lines)

    def update_from_message(self, user_id: str, message: str, plan: Dict[str, Any]) -> None:
        if not user_id:
            return
        text = (message or "").strip()
        with self._lock:
            p = self._cache.setdefault(user_id, {})
            frequent_od: List[str] = list(p.get("frequent_od", []))
            frequent_highways: List[str] = list(p.get("frequent_highways", []))
            flags: List[str] = list(p.get("flags", []))

            if hasattr(self._svc, "_extract_route_endpoints"):
                o, d = self._svc._extract_route_endpoints(text)
                if o and d:
                    frequent_od.append(f"{o} → {d}")

            for m in re.findall(r"[GS]\s*\d{1,4}", text):
                frequent_highways.append(m.replace(" ", "").upper())

            if any(k in text for k in ("早高峰", "早点走", "一早", "清早")):
                p["time_pref"] = "偏好早高峰前出发"
            elif any(k in text for k in ("晚点走", "错峰", "夜里")):
                p["time_pref"] = "偏好错峰/夜间出发"

            for k, flag in (
                ("充电", "新能源/需要充电桩"),
                ("加油", "燃油/关注油站"),
                ("母婴", "母婴出行"),
                ("无障碍", "无障碍出行"),
                ("轮椅", "无障碍出行"),
            ):
                if k in text and flag not in flags:
                    flags.append(flag)

            p["frequent_od"] = _dedup(frequent_od, 10)
            p["frequent_highways"] = _dedup(frequent_highways, 5)
            p["flags"] = flags[-8:]

            self._flush()

    def _flush(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
