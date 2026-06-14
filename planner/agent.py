from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from memory.models import MemoryContext

from .executor import TaskExecutor
from .models import Plan
from .rules import build_plan, should_plan


class PlannerAgent:
    """任务规划 Agent：分析目标、拆解任务、生成可执行 plan。"""

    def __init__(self, service_agent: Any = None) -> None:
        self._svc = service_agent
        self._executor = TaskExecutor()
        self.enabled = os.getenv("PLANNER_ENABLE", "true").strip().lower() in {"1", "true", "yes", "on"}

    def should_plan(self, message: str, history: List[Dict[str, Any]], ctx: MemoryContext | None = None) -> bool:
        if not self.enabled:
            return False
        return should_plan(message, history, ctx)

    def build_plan(
        self,
        message: str,
        history: List[Dict[str, Any]],
        ctx: MemoryContext | None = None,
    ) -> Optional[Plan]:
        return build_plan(message, history, ctx, self._svc)

    def run(
        self,
        message: str,
        history: List[Dict[str, Any]],
        ctx: MemoryContext | None = None,
    ) -> Optional[Dict[str, Any]]:
        plan = self.build_plan(message, history, ctx)
        if plan is None:
            return None
        return self._executor.to_executable_plan(plan)
