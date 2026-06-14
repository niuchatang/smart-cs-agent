from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

ExecutionMode = Literal["serial", "parallel", "mixed"]


class PlanTask(BaseModel):
    id: str
    agent: str
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    parallel_group: int = 1
    depends_on: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    tasks: List[PlanTask] = Field(default_factory=list)
    execution: ExecutionMode = "mixed"
    used_llm: bool = False

    def to_plan_dict(self, confidence: float = 0.9) -> Dict[str, Any]:
        actions = [{"tool": t.tool, "params": dict(t.params)} for t in self.tasks if t.tool]
        intent = "travel_decision" if any(t.tool == "query_travel_decision" for t in self.tasks) else "route_planning"
        if any(t.agent == "WeatherAgent" for t in self.tasks) and not any(
            t.tool == "query_travel_decision" for t in self.tasks
        ):
            intent = "weather_query"
        return {
            "intent": intent,
            "confidence": confidence,
            "actions": actions,
            "used_llm": self.used_llm,
            "llm_reply": "",
            "meta": {
                "planner": True,
                "goal": self.goal,
                "tasks": [t.model_dump() for t in self.tasks],
                "execution": self.execution,
            },
        }
