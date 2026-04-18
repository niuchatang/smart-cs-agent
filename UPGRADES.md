# Smart CS Agent — 升级说明（批量落地）

本文件汇总本次所有**无需内部系统**的升级与新增子智能体。所有新代码：

- **不改动** `main.py` 业务主干（避免对 4000+ 行主流程引入风险）；
- **只**扩展 `intent/` 子系统、新增独立的 `tools_infra/`、`safety/`、`rag/`、`evaluation/`、`tests/`；
- **默认启用**扩展智能体（接入在 `IntentOrchestratorAgent.plan_rules` 规则链尾部之前），
  也可以通过 `IntentOrchestratorAgent(..., enable_extensions=False)` 关闭。

---

## 1. 新增子智能体

| 类别 | 类 | 文件 | 启用方式 |
|---|---|---|---|
| 意图规则扩展 | `ETCChargeAgent` | `intent/etc_charge_agent.py` | 自动（注册到 registry，priority=20） |
| 意图规则扩展 | `ServiceAreaAgent` | `intent/service_area_agent.py` | 自动（priority=30） |
| 意图规则扩展 | `DepartureTimeAgent` | `intent/departure_time_agent.py` | 自动（priority=35） |
| 意图规则扩展 | `TrafficIncidentAgent` | `intent/traffic_incident_agent.py` | 自动（priority=40） |
| 意图规则扩展 | `WeatherImpactAgent` | `intent/weather_impact_agent.py` | 自动（priority=50） |
| 意图规则扩展 | `AccessibilityAgent` | `intent/accessibility_agent.py` | 自动（priority=60） |
| 意图规则扩展 | `ClarifyAgent` | `intent/clarify_agent.py` | 自动（priority=85，兜底） |
| 预/后处理 | `MultilingualAgent` | `intent/multilingual_agent.py` | 显式调用 `detect(msg)` |
| 预/后处理 | `GuardrailAgent` | `intent/guardrail_agent.py` | 显式调用入/出站扫描 |
| 分级/评估 | `ComplaintTriageAgent` | `intent/complaint_triage_agent.py` | 在投诉/转人工意图处调用 `triage()` |
| RAG 专职 | `FAQAgent` | `intent/faq_agent.py` | 在 `_render_reply` 命中 RAG 时调用 `answer()` |
| 画像 / 记忆 | `ProfileAgent` | `intent/profile_agent.py` | 每轮末尾调 `update_from_message()` |
| 画像 / 记忆 | `ConversationSummarizer` | `intent/summarizer.py` | 在 `_plan_by_llm` 准备 prompt 时调 `split()` |
| 工具路由 | `ToolRouterAgent` | `intent/tool_router_agent.py` | 主智能体在工具失败后调 `suggest_fallback()` |
| 满意度 | `SatisfactionAgent` | `intent/satisfaction_agent.py` | 会话结束信号触发 `prompt_survey()` |
| 评估 | `EvalAgent` | `intent/eval_agent.py` | CLI：`python -m evaluation.eval` |

所有扩展智能体复用既有意图枚举（`route_planning`/`highway_condition`/`weather_query`/
`fare_policy`/`unknown` 等），通过 `actions` 或 `llm_reply` 承载新业务，
无需修改 `main.IntentType` Literal 或 Pydantic 模型。

---

## 2. 基础设施

### `tools_infra/`

- `cache.py`：`TTLCache`（OrderedDict LRU + TTL，线程安全）+ `@cached` 装饰器。
- `registry.py`：`ToolSpec`（schema / timeout / retry / fallback / cache）+ `ToolRunner`（熔断计数）。

可以逐步把 Nominatim 地理编码、OSRM、同城短时天气这类外呼函数迁到 `ToolRunner` 上；
迁移前系统行为不变。

### `safety/`

- `pii.py`：`mask_pii`、`pii_hits`；
- `moderation.py`：`scan_forbidden` / `sanitize_reply`。

与 `GuardrailAgent` 的内置规则等价，可在日志/持久化/出站渲染等非 Agent 位置直接复用。

### `rag/hybrid_retriever.py`

- BM25（LangChain `BM25Retriever`）+ 可选向量检索；
- 融合：RRF（`1 / (k + rank)`）；
- 没装向量模型时自动退化为纯 BM25，不影响现有 `_SimpleRAGStore` 行为。

启用示例：

```python
from rag import HybridRetriever, RRFConfig

# 示例 encoder：接 sentence-transformers 或你喜欢的 embedding
# 不传 encoder 时就是 BM25-only，相当于增强版的现有检索
retriever = HybridRetriever(docs, rrf=RRFConfig(top_k=5))
hits = retriever.retrieve("高速退票手续费")
```

### `evaluation/`

- `intent_cases.jsonl`：20 条意图回归用例（可持续追加）；
- `eval.py` / `intent.eval_agent`：加载用例，调 `UserIntentAgent.parse` 对比 intent 与必须调用的工具。

```bash
python -m evaluation.eval --cases evaluation/intent_cases.jsonl
```

### `tests/`

- `test_guardrail.py`、`test_registry.py`、`test_summarizer.py`、`test_tools_cache.py`。
- 运行：`python -m pytest -q tests`（需安装 `pytest`）。

---

## 3. 如何接入到 `main.py`（可选，零改动时系统已可用）

下面几段是**可选**的「更深一层集成」，不做也不影响默认扩展生效。

### 3.1 `ChatRequest` 入站 → GuardrailAgent

```python
from intent import GuardrailAgent
guardrail = GuardrailAgent(self)
scan = guardrail.scan_inbound(payload.message)
if scan.injection:
    return ChatResponse(intent="unknown", confidence=0.9,
                        reply=guardrail.soft_decline_reply(), actions=[],
                        tool_results=[], used_llm=False, rag_used=False)
message_for_llm = scan.masked_message   # 喂给 _plan_by_llm 前做 PII 脱敏
```

### 3.2 `_render_reply` 出站 → moderation

```python
from safety import scan_forbidden
reply_res = scan_forbidden(reply)
if not reply_res.safe:
    reply = reply_res.sanitized
```

### 3.3 会话摘要接入 `_plan_by_llm`

```python
from intent import ConversationSummarizer
summarizer = ConversationSummarizer(self)
split = summarizer.split(history)
history_text = (split["summary"] + "\n" if split["summary"] else "") + \
               self._history_to_text(split["recent"])
```

### 3.4 投诉分级

```python
from intent import ComplaintTriageAgent
tri = ComplaintTriageAgent(self).triage(message)
if tri["severity"] == "urgent":
    # 升级：加 handoff_to_human，priority=high
```

### 3.5 画像

```python
from intent import ProfileAgent
profiler = ProfileAgent(self)
profiler.update_from_message(user_id=payload.user_id, message=message, plan=plan)
brief = profiler.summary(payload.user_id)  # 可写到 _plan_by_llm 的 prompt 顶部
```

### 3.6 工具回退

```python
from intent import ToolRouterAgent
router = ToolRouterAgent(self)
fallback = router.suggest_fallback(action, result)
if fallback is not None:
    result = self._run_single_action(fallback)
```

### 3.7 满意度

```python
from intent import SatisfactionAgent
sat = SatisfactionAgent(self)
if SatisfactionAgent.should_prompt_survey(message):
    reply = sat.prompt_survey()  # 替换当轮 reply
score = sat.parse_score(message)
if score: sat.record_score(payload.user_id, score["score"], score["comment"])
```

---

## 4. 默认扩展的优先级一览

扩展层调用发生在 orchestrator 的第 6 步（在 `try_late_corridor` 之后、
`try_tail_rules` 之前），内部按 `priority` 升序尝试：

```
 0  GuardrailAgent                         # 不自动注册，需显式调用
 5  MultilingualAgent                      # 预处理器，不在主链路
20  ETCChargeAgent
30  ServiceAreaAgent
35  DepartureTimeAgent
40  TrafficIncidentAgent
50  WeatherImpactAgent
60  AccessibilityAgent
85  ClarifyAgent（兜底型澄清）
100 ComplaintTriageAgent / FAQAgent / ToolRouterAgent / ProfileAgent 等（均不参与自动链）
200 SatisfactionAgent / EvalAgent（离线/事件驱动）
```

---

## 5. 故意跳过的项目（需要内部系统）

以下项目**本次未实现**，因为它们依赖尚未提供的内部系统接入：

- 真实工单 / CRM / 坐席队列（`HandoffRouterAgent`、完整 `TicketAgent`）；
- 12306 / 航司 / 地铁真实实时数据（`MultiModalAgent`）；
- 高速集团官方事故公告抓取 / 推送；
- 真实满意度后台 / BI；
- LangSmith / OpenTelemetry 的 trace 后端；
- 语音 / 流式前端改造（需改 Vue 组件 + WebSocket 后端）。

基础结构（Registry、Guardrail、Safety、Tool Runner、Hybrid RAG）已经把后续接入留出了扩展点。
