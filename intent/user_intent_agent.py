"""
用户意图解析智能体（User Intent Agent）

核心职责：
- 识别用户口语化指令，区分路况/路径/公交地铁实时/票价工单/转人工等；
- 优先使用大模型输出结构化 JSON 计划，失败时回落规则引擎；
- 输出统一结构的 plan：intent、confidence、actions、可选 llm_reply / od_traffic_followup，
  由 CustomerServiceAgent 执行工具并渲染话术。

上下文抽取（地名、历史路径、高速编号等）仍由主智能体提供，本模块只负责「理解与标准化计划」。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .intent_planning_graph import compile_user_intent_planning_graph
from .orchestrator_agent import IntentOrchestratorAgent
from .weather_agent import WeatherDialogAgent


class UserIntentAgent:
    """用户意图解析智能体：parse() 返回与原先 CustomerServiceAgent._plan 相同结构的 dict。"""

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent
        self._weather_dialog = WeatherDialogAgent(service_agent)
        self._orchestrator = IntentOrchestratorAgent(service_agent)
        self._planning_graph: Any = None

    def parse(self, message: str, history: List[Dict[str, Any]], rag_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解析用户消息，输出标准化 plan（供工具执行层消费）。"""
        return self._plan(message, history, rag_hits)

    def _ensure_planning_graph(self) -> Any:
        if self._planning_graph is None:
            self._planning_graph = compile_user_intent_planning_graph(self)
        return self._planning_graph

    def _plan(self, message: str, history: List[Dict[str, Any]], rag_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        hist = history or []
        out = self._ensure_planning_graph().invoke(
            {
                "message": message,
                "history": hist,
                "rag_hits": rag_hits or [],
                "llm_attempted": False,
            }
        )
        plan = out.get("plan")
        if isinstance(plan, dict):
            return plan
        return self._plan_by_rules(message, hist)

    @staticmethod
    def _should_prefer_llm(message: str, history: List[Dict[str, Any]]) -> bool:
        if not history:
            return False
        text = message.strip()
        if not text:
            return False
        if len(text) <= 12 and any(k in text for k in ["怎么样", "如何", "咋样", "这中间", "这段", "这条", "路况"]):
            return True
        return False

    def _plan_by_llm(self, message: str, history: List[Dict[str, Any]], rag_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        history_text = self._svc._history_to_text(history)
        rag_text = self._svc._rag_to_text(rag_hits)
        recent_route_context = self._svc._extract_recent_route_context(history)
        if self._svc.llm is None:
            return self._plan_by_rules(message, history)
        planner_prompt = ChatPromptTemplate.from_template(
            """
你是智慧交通客服智能体中的「用户意图解析」模块：只负责识别用户想做什么，并输出可执行的工具计划。
仅输出 JSON，格式如下：
{{
  "intent": "route_planning|realtime_status|highway_condition|weather_query|fare_policy|ticket_refund|lost_and_found|complaint|human_handoff|unknown",
  "confidence": 0.0,
  "actions": [{{"tool":"query_transit_status|query_highway_condition|query_weather|calculate_fare|create_transport_ticket|handoff_to_human","params":{{}}}}],
  "llm_reply": "string"
}}

规则：
1) 查询地铁/公交实时状态优先调用 query_transit_status。
2) 查询高速或道路事故/管制/拥堵/路况优先调用 query_highway_condition。
3) 天气多轮由「天气对话智能体」优先处理：追问芯片为「查询途经城市天气」，点击后先问途经哪一座城市再查；「沿途」为逐站查询；短「是」在无待查下一站时会引导选城市。显式「上海天气」等仍可用 query_weather。
4) 票价咨询可调用 calculate_fare。
5) 退票、失物招领、投诉应调用 create_transport_ticket。
6) 明确要求人工时调用 handoff_to_human。
7) confidence 在 [0,1]。
8) 对“这中间路况/这段路况/路况怎么样/有交通事故吗”这类追问，若未明确给出高速编号，请结合“最近一次路径规划上下文”里的途经高速，生成多个 query_highway_condition 动作，逐条查询并返回。
9) intent 为 unknown 且用户只是在闲聊、问候、与出行无关时，actions 可为空，并在 llm_reply 给一句简短友好回复（后续主体会结合知识库再润色）。

最近对话上下文：
{history_text}

最近一次路径规划上下文（结构化）：
{recent_route_context}

检索到的知识库片段（RAG）：
{rag_text}

用户消息：{message}
""".strip()
        )
        chain = planner_prompt | self._svc.llm | StrOutputParser()
        text = chain.invoke(
            {
                "history_text": history_text,
                "recent_route_context": recent_route_context,
                "rag_text": rag_text,
                "message": message,
            }
        ).strip()
        parsed = self._parse_llm_plan_json(text)
        return {
            "intent": parsed.get("intent", "unknown"),
            "confidence": float(parsed.get("confidence", 0.6)),
            "actions": parsed.get("actions", []),
            "llm_reply": parsed.get("llm_reply", ""),
            "used_llm": True,
        }

    @staticmethod
    def _parse_llm_plan_json(text: str) -> Dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            block = m.group(0)
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        raise ValueError("llm output is not valid planning json")

    def _post_process_llm_plan(
        self, llm_plan: Dict[str, Any], message: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        plan = dict(llm_plan)
        plan["actions"] = list(plan.get("actions", [])) if isinstance(plan.get("actions"), list) else []
        intent = str(plan.get("intent", "unknown"))
        road_query = any(k in message for k in ["高速", "事故", "管制", "封闭", "拥堵", "路况"])
        transit_query = any(k in message for k in ["地铁", "公交", "班次", "首班", "末班", "延误"])

        if road_query and not transit_query and intent == "realtime_status":
            intent = "highway_condition"
            plan["intent"] = intent
            plan["confidence"] = max(float(plan.get("confidence", 0.0)), 0.75)

        explicit_target = self._svc._extract_highway_target(message)
        if explicit_target:
            intent = "highway_condition"
            plan["intent"] = intent
            plan["confidence"] = max(float(plan.get("confidence", 0.0)), 0.9)

        multi_od = self._svc._extract_multi_stop_places(message)
        if multi_od and len(multi_od) >= 3:
            origin_od, dest_od = multi_od[0], multi_od[-1]
        else:
            origin_od, dest_od = self._svc._extract_route_endpoints(message)
        if (
            road_query
            and not transit_query
            and origin_od
            and dest_od
            and not explicit_target
        ):
            intent = "highway_condition"
            plan["intent"] = "highway_condition"
            plan["confidence"] = max(float(plan.get("confidence", 0.0)), 0.88)
            plan["od_traffic_followup"] = True
            rp_od: Dict[str, Any] = {"origin": origin_od, "destination": dest_od, "mode": "driving"}
            if multi_od and len(multi_od) >= 3:
                rp_od["waypoints"] = multi_od[1:-1]
            plan["actions"] = [{"tool": "query_route_plan", "params": rp_od}]

        if intent == "unknown" and self._svc._is_affirmative_message(message):
            recent_codes = self._svc._extract_last_highway_codes_from_history(history)
            if recent_codes:
                plan["intent"] = "highway_condition"
                plan["confidence"] = max(float(plan.get("confidence", 0.0)), 0.82)
                if len(recent_codes) > 1:
                    plan["actions"] = []
                    plan["llm_reply"] = (
                        f"可以。你上一条涉及多条高速：{', '.join(recent_codes[:4])}。"
                        "请告诉我你想展开哪一条（例如：查看G30详情）。"
                    )
                else:
                    plan["actions"] = [{"tool": "query_highway_condition", "params": {"target": recent_codes[0]}}]

        if intent == "route_planning" and not plan["actions"]:
            multi_rp = self._svc._extract_multi_stop_places(message)
            if multi_rp and len(multi_rp) >= 3:
                plan["actions"] = [
                    {
                        "tool": "query_route_plan",
                        "params": {
                            "origin": multi_rp[0],
                            "destination": multi_rp[-1],
                            "waypoints": multi_rp[1:-1],
                            "mode": "driving",
                        },
                    }
                ]
            else:
                origin, destination = self._svc._extract_route_endpoints(message)
                if origin and destination:
                    plan["actions"] = [
                        {
                            "tool": "query_route_plan",
                            "params": {"origin": origin, "destination": destination, "mode": "driving"},
                        }
                    ]

        if intent == "highway_condition" and not plan.get("od_traffic_followup"):
            fallback_target = explicit_target or self._svc._extract_last_highway_from_history(history)
            route_highways = self._svc._extract_last_route_highways_from_history(history)
            route_points = self._svc._extract_last_route_probe_points_from_history(history)
            normalized_actions: List[Dict[str, Any]] = []
            for action in plan["actions"]:
                if not isinstance(action, dict):
                    continue
                tool = str(action.get("tool", ""))
                params = action.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                if tool == "query_highway_condition":
                    target = str(params.get("target", "")).strip() or fallback_target
                    if target:
                        out_params: Dict[str, Any] = {"target": target}
                        if isinstance(params.get("context_points"), list):
                            out_params["context_points"] = params.get("context_points")
                        elif isinstance(params.get("context_point"), dict):
                            out_params["context_points"] = [params.get("context_point")]
                        elif route_points:
                            out_params["context_points"] = route_points
                        normalized_actions.append({"tool": "query_highway_condition", "params": out_params})
                else:
                    normalized_actions.append({"tool": tool, "params": params})
            if not explicit_target and route_highways:
                normalized_actions = []
                for hw in route_highways[:4]:
                    out_params = {"target": hw}
                    if route_points:
                        out_params["context_points"] = route_points
                    normalized_actions.append({"tool": "query_highway_condition", "params": out_params})
            elif not normalized_actions:
                if fallback_target:
                    out_params = {"target": fallback_target}
                    if route_points:
                        out_params["context_points"] = route_points
                    normalized_actions.append({"tool": "query_highway_condition", "params": out_params})
                elif route_highways:
                    for hw in route_highways[:4]:
                        out_params = {"target": hw}
                        if route_points:
                            out_params["context_points"] = route_points
                        normalized_actions.append({"tool": "query_highway_condition", "params": out_params})
            plan["actions"] = normalized_actions

        normalized_wx: List[Dict[str, Any]] = []
        for action in plan["actions"]:
            if not isinstance(action, dict):
                continue
            if str(action.get("tool", "")) == "query_weather":
                params = action.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                cities = params.get("cities")
                clist = [str(x).strip() for x in cities] if isinstance(cities, list) else []
                wp: Dict[str, Any] = {"cities": clist}
                arq = params.get("along_route_queue")
                if isinstance(arq, list) and arq:
                    wp["along_route_queue"] = [str(x).strip() for x in arq if str(x).strip()]
                ari = params.get("along_route_index")
                if ari is not None:
                    try:
                        wp["along_route_index"] = int(ari)
                    except (TypeError, ValueError):
                        pass
                normalized_wx.append({"tool": "query_weather", "params": wp})
            else:
                normalized_wx.append(action)
        plan["actions"] = normalized_wx
        if any(isinstance(a, dict) and a.get("tool") == "query_weather" for a in plan["actions"]):
            plan["intent"] = "weather_query"

        return plan

    @staticmethod
    def _is_usable_plan(plan: Dict[str, Any]) -> bool:
        intent = str(plan.get("intent", "")).strip()
        actions = plan.get("actions", [])
        if intent in {
            "route_planning",
            "realtime_status",
            "highway_condition",
            "weather_query",
            "fare_policy",
            "ticket_refund",
            "lost_and_found",
            "complaint",
            "human_handoff",
            "unknown",
        }:
            return isinstance(actions, list)
        return False

    def _plan_by_rules(self, message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """规则回落：由 IntentOrchestratorAgent 按子智能体顺序编排（路况 → 天气 → 路径 → 通用）。"""
        return self._orchestrator.plan_rules(message, history)
