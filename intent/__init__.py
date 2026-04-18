"""
用户意图解析子系统：将口语化输入解析为标准化执行计划（intent + actions），供客服主智能体调度工具与生成回复。

## 子智能体分层

- **主链路（默认启用，顺序与原 `_plan_by_rules` 一致）**：
  `WeatherDialogAgent` → LLM 规划 → `IntentOrchestratorAgent`（内部编排
  `RoadConditionAgent` / `RoutePlanningAgent` / `GeneralIntentAgent`）。
- **扩展层（默认启用，位于 orchestrator 规则链尾部之前）**：
  `AgentRegistry` 托管 `ETCChargeAgent` / `ServiceAreaAgent` / `DepartureTimeAgent` /
  `TrafficIncidentAgent` / `WeatherImpactAgent` / `AccessibilityAgent` / `ClarifyAgent`。
- **预处理 / 跨阶段工具（按需显式调用）**：`MultilingualAgent`、`ComplaintTriageAgent`、
  `FAQAgent`、`ProfileAgent`、`ToolRouterAgent`、`GuardrailAgent`、`ConversationSummarizer`、
  `EvalAgent`（离线评估）、`SatisfactionAgent`（对话闭环）。
"""

from .accessibility_agent import AccessibilityAgent
from .agent_registry import AgentRegistry, default_extension_factories
from .clarify_agent import ClarifyAgent
from .complaint_triage_agent import ComplaintTriageAgent
from .departure_time_agent import DepartureTimeAgent
from .etc_charge_agent import ETCChargeAgent
from .eval_agent import EvalAgent
from .faq_agent import FAQAgent
from .general_intent_agent import GeneralIntentAgent
from .guardrail_agent import GuardrailAgent
from .multilingual_agent import MultilingualAgent
from .orchestrator_agent import IntentOrchestratorAgent
from .profile_agent import ProfileAgent
from .road_condition_agent import RoadConditionAgent
from .route_planning_agent import RoutePlanningAgent
from .satisfaction_agent import SatisfactionAgent
from .service_area_agent import ServiceAreaAgent
from .summarizer import ConversationSummarizer
from .tool_router_agent import ToolRouterAgent
from .traffic_incident_agent import TrafficIncidentAgent
from .user_intent_agent import UserIntentAgent
from .weather_impact_agent import WeatherImpactAgent

__all__ = [
    "AccessibilityAgent",
    "AgentRegistry",
    "ClarifyAgent",
    "ComplaintTriageAgent",
    "ConversationSummarizer",
    "DepartureTimeAgent",
    "ETCChargeAgent",
    "EvalAgent",
    "FAQAgent",
    "GeneralIntentAgent",
    "GuardrailAgent",
    "IntentOrchestratorAgent",
    "MultilingualAgent",
    "ProfileAgent",
    "RoadConditionAgent",
    "RoutePlanningAgent",
    "SatisfactionAgent",
    "ServiceAreaAgent",
    "ToolRouterAgent",
    "TrafficIncidentAgent",
    "UserIntentAgent",
    "WeatherImpactAgent",
    "default_extension_factories",
]
