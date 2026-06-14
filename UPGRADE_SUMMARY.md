# Smart CS Agent 升级汇总（全记录）

> **完整架构讲解**见 [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)（推荐先读）。  
> 扩展 Agent 接入示例见 [`UPGRADES.md`](./UPGRADES.md)。  
> Memory + Planner 专项设计见 [`docs/specs/memory_planner_upgrade.md`](./docs/specs/memory_planner_upgrade.md)。

---

## 升级时间线

| 阶段 | 名称 | 文档章节 |
|:---:|---|---|
| A | 基础客服 + 工具 + RAG | ARCHITECTURE §二 |
| B | LangGraph 意图图 + Orchestrator | ARCHITECTURE §四 |
| C | AgentRegistry + 16 扩展 Agent + 基础设施 | 本文 §一～三 |
| D | TravelDecision + GIS 天气区域 | 本文 §六 |
| E | **Memory Agent + Planner Agent** | 本文 §七 |

---

## 一、阶段 C — 可插拔扩展（原「本轮升级」）

> 当时约束：**尽量不改动** `main.py` 主干；专属系统能力跳过。  
> 后续阶段 D/E 为增强出行与认知能力，对 `main.py` 做了**最小接入**（`CognitiveOrchestrator`）。

所有改造通过 `intent/` 子系统 + 四个新模块（`tools_infra/`、`safety/`、`rag/`、`evaluation/`）落地，并用 `tests/` 兜单测。

---

### 1. 新增子智能体（16 个）

在 `IntentOrchestratorAgent.plan_rules` 中，于 `try_late_corridor` 之后、`try_tail_rules` 之前调用 `AgentRegistry.try_plan(...)`，按 `priority` 升序尝试：

| 优先级 | 智能体 | 文件 | 职责 |
|---|---|---|---|
| 20 | `ETCChargeAgent` | `intent/etc_charge_agent.py` | ETC / 过路费估算，节假日免费提醒 |
| 30 | `ServiceAreaAgent` | `intent/service_area_agent.py` | 路线上的服务区 / 充电站 / 加油站 |
| 35 | `DepartureTimeAgent` | `intent/departure_time_agent.py` | 最佳出发时间（早晚高峰 / 节假日启发式） |
| 40 | `TrafficIncidentAgent` | `intent/traffic_incident_agent.py` | 事故 / 管制 / 封路主动查询 |
| 50 | `WeatherImpactAgent` | `intent/weather_impact_agent.py` | 雨 / 雾 / 雪 / 大风 / 高温对驾驶的建议 |
| 60 | `AccessibilityAgent` | `intent/accessibility_agent.py` | 无障碍出行（轮椅、导盲犬、无障碍设施） |
| 85 | `ClarifyAgent` | `intent/clarify_agent.py` | 低置信度时主动追问缺失要素 |

可通过 `IntentOrchestratorAgent(..., enable_extensions=False)` 一键关闭整条扩展链。

全部位于 `smart-cs-agent/intent/`。

#### 自动挂到规则链上（默认生效）

| 智能体 | 文件 | 职责 |
|---|---|---|
| `GuardrailAgent` | `intent/guardrail_agent.py` | 入站查 prompt injection + PII 脱敏；出站扫描承诺性话术 |
| `MultilingualAgent` | `intent/multilingual_agent.py` | 中 / 日 / 韩 / 俄 / 英 / 粤语识别 + 可选翻译 |
| `ComplaintTriageAgent` | `intent/complaint_triage_agent.py` | 投诉分级（low / medium / high / urgent） |
| `FAQAgent` | `intent/faq_agent.py` | 专职 RAG 问答 + 引用渲染 |
| `ProfileAgent` | `intent/profile_agent.py` | 从消息里抽偏好，落盘 `data/user_profiles.json` |
| `ConversationSummarizer` | `intent/summarizer.py` | 历史折叠成摘要，缩 prompt |
| `ToolRouterAgent` | `intent/tool_router_agent.py` | 工具失败 / 空结果时给回退方案 |
| `SatisfactionAgent` | `intent/satisfaction_agent.py` | 会话结束弹满意度，写 `data/satisfaction.jsonl` |
| `EvalAgent` | `intent/eval_agent.py` | 离线意图回归评测，CLI 可调用 |

#### 按需显式调用（留接入点，未强绑主流程）

- `intent/agent_registry.py`：`AgentRegistry` + `ExtensionAgent` 协议（支持 `priority` / `name` / `try_plan`）；
- `intent/orchestrator_agent.py`：多插一步 `self._registry.try_plan(message, history)`，把第 1 组自动接上；
- 所有扩展**复用既有 `IntentType` 枚举**（`route_planning` / `highway_condition` / `weather_query` / `fare_policy` / `unknown`），不改 Pydantic schema。

---

#### 让它们能插拔的底座

| 模块 | 关键能力 | 关键文件 |
|---|---|---|
| `tools_infra/` | `TTLCache`（OrderedDict + LRU + TTL，线程安全）、`@cached` 装饰器；`ToolSpec`（schema / timeout / retry / fallback / cache）+ `ToolRunner`（熔断计数） | `tools_infra/cache.py`、`tools_infra/registry.py` |
| `safety/` | `mask_pii` / `pii_hits` / `scan_forbidden` / `sanitize_reply`，与 Guardrail 规则等价，可在非 Agent 处复用 | `safety/pii.py`、`safety/moderation.py` |
| `rag/` | `HybridRetriever`：BM25 + 可选向量检索，用 RRF 融合；无向量模型时自动退化为纯 BM25 | `rag/hybrid_retriever.py` |
| `evaluation/` | 离线意图回归：`intent_cases.jsonl`（20 条）+ `python -m evaluation.eval` | `evaluation/eval.py`、`evaluation/intent_cases.jsonl` |
| `tests/` | 单测覆盖 guardrail / registry / summarizer / tools 缓存 | `tests/test_guardrail.py` 等 |

---

## 二、阶段 C — 新增基础设施（4 个模块 + tests）

| 跳过项 | 原因 | 已留好的扩展点 |
|---|---|---|
| 真实工单 / CRM / 坐席队列（`HandoffRouterAgent`、完整 `TicketAgent`） | 需要内部 CRM API | `ComplaintTriageAgent` + `AgentRegistry` |
| 12306 / 航司 / 地铁实时数据（`MultiModalAgent`） | 需要专属数据源授权 | `ToolRunner` + `AgentRegistry` |
| 高速集团官方事故公告抓取 / 推送 | 需要内部公告系统 | `TrafficIncidentAgent` + `ToolRunner` |
| 真实满意度后台 / BI | 需要内部报表系统 | `SatisfactionAgent` 已把数据落到 jsonl |
| LangSmith / OpenTelemetry trace | 需要外部账号 | `ToolRunner` 有打点位 |
| 语音 / 流式前端 | 需要改 Vue + WebSocket | 暂未改前端 |

---

## 三、阶段 C — 故意没做的（需要内部系统）

### 单测

```bash
cd smart-cs-agent
source .venv/bin/activate
python -m pytest -q tests
```

### 离线意图回归

```bash
python -m evaluation.eval --cases evaluation/intent_cases.jsonl
```

### 接口手测（启动后）

启动：

```bash
cd smart-cs-agent
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

在前端（http://127.0.0.1:8000/）或 `/docs` 里丢这些句子验证对应智能体：

| 触发话术 | 应命中 |
|---|---|
| `广州到深圳ETC大概多少钱` | `ETCChargeAgent` |
| `这条路线上有哪些服务区` | `ServiceAreaAgent` |
| `明天几点出发最合适` | `DepartureTimeAgent` |
| `下雨天开高速要注意什么` | `WeatherImpactAgent` |
| `G4京港澳今天有事故吗` | `TrafficIncidentAgent` |
| `轮椅能上高铁吗` | `AccessibilityAgent` |
| `查路线`（信息不全） | `ClarifyAgent` |

---

## 四、阶段 C — 验证方式

## 五、阶段 C — 文件清单
├── intent/
│   ├── agent_registry.py          # 新：Registry + ExtensionAgent 协议
│   ├── orchestrator_agent.py      # 改：接入 registry
│   ├── __init__.py                # 改：导出新 agent
│   ├── clarify_agent.py           # 新
│   ├── etc_charge_agent.py        # 新
│   ├── service_area_agent.py      # 新
│   ├── departure_time_agent.py    # 新
│   ├── weather_impact_agent.py    # 新
│   ├── traffic_incident_agent.py  # 新
│   ├── accessibility_agent.py     # 新
│   ├── complaint_triage_agent.py  # 新
│   ├── multilingual_agent.py      # 新
│   ├── satisfaction_agent.py      # 新
│   ├── faq_agent.py               # 新
│   ├── profile_agent.py           # 新
│   ├── tool_router_agent.py       # 新
│   ├── guardrail_agent.py         # 新
│   ├── summarizer.py              # 新
│   └── eval_agent.py              # 新
├── tools_infra/                   # 新模块：TTLCache / ToolSpec / ToolRunner
├── safety/                        # 新模块：PII 脱敏 / 出站审核
├── rag/                           # 新模块：HybridRetriever (BM25 + 向量)
├── evaluation/                    # 新模块：离线评测
│   ├── eval.py
│   └── intent_cases.jsonl
├── tests/                         # 新模块：单测
├── UPGRADES.md                    # 新：详细集成示例
└── UPGRADE_SUMMARY.md             # 新：本文件
```

---

## 六、阶段 D — 出行决策 + GIS 天气区域（2026-06）

| 升级项 | 文件 | 说明 |
|---|---|---|
| TravelDecisionAgent | `intent/travel_decision_agent.py` | 识别「从 A 到 B 几点出发」类意图 |
| query_travel_decision | `main.py`, `tools_infra/travel_decision_tools.py` | 一站式：路线 + 天气 + 路况 + 风险评分 |
| LangGraph 入口节点 | `intent/intent_planning_graph.py` | `travel_decision_node` 置于图最前 |
| GIS MySQL 空间表 | `database/weather_gis_schema.sql` | `weather_admin_boundary` + `weather_admin_alias` |
| GISLocationTool | `tools_infra/gis_location_tool.py` | 地名/坐标 → adcode，优先 MySQL ST_Contains |
| 边界数据脚本 | `scripts/download_admin_boundaries.py` | 从阿里云 DataV 下载全国区县 GeoJSON |
| | `scripts/import_admin_boundaries.py` | 导入 MySQL（509 区县） |

**启用**：见 [`docs/specs/weather_gis.md`](./docs/specs/weather_gis.md)

**验证话术**：`西青区天气` · `经度117.01 纬度39.14天气` · `明天从北京到天津适合几点出发`

---

## 七、阶段 E — Memory Agent + Planner Agent（2026-06 最新）

| 升级项 | 文件 | 说明 |
|---|---|---|
| Memory 三表 | `database/memory_schema.sql` | profile / event / summary |
| memory/ 模块 | `memory/agent.py` 等 | 长期记忆读写、规则抽取、4 个 Memory Tool |
| planner/ 模块 | `planner/agent.py` 等 | 任务规划、规则拆解、LangGraph 决策图 |
| CognitiveOrchestrator | `intent/cognitive_orchestrator.py` | Memory → Planner → Intent 认知编排 |
| main.py 接入 | `CustomerServiceAgent.chat()` | `cognitive.parse()` + `after_turn()` |
| 单测 | `tests/test_memory_agent.py` 等 | 7 条用例 |

**环境变量**：

```env
MEMORY_ENABLE_MYSQL=true
PLANNER_ENABLE=true
COGNITIVE_ENABLE=true
```

**验证话术**：

| 话术 | 能力 |
|---|---|
| `我每天从朝阳区到亦庄上班` | 写入通勤记忆 |
| `今天几点出发` | Memory 补全 OD + Planner 出行决策 |
| `未来三天适合去天津吗` | Planner 天气/AQI 综合 |

**详细设计**：[`docs/specs/memory_planner_upgrade.md`](./docs/specs/memory_planner_upgrade.md)

---

## 八、当前推荐文档阅读顺序

1. [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — **系统架构 + 全升级记录 + 扩展指南**
2. [`UPGRADE_SUMMARY.md`](./UPGRADE_SUMMARY.md) — 本文，按阶段速览
3. [`UPGRADES.md`](./UPGRADES.md) — 扩展 Agent 接入代码示例
4. [`docs/specs/weather_gis.md`](./docs/specs/weather_gis.md) — GIS 专项
5. [`docs/specs/memory_planner_upgrade.md`](./docs/specs/memory_planner_upgrade.md) — Memory/Planner 专项
