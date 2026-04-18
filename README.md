## 预览

<img width="1508" height="905" alt="截屏2026-04-02 17 47 41" src="https://github.com/user-attachments/assets/620e09e3-eb27-4476-b181-3f4797d2456c" />

<img width="1508" height="905" alt="截屏2026-04-02 17 48 25" src="https://github.com/user-attachments/assets/922bcf0e-62a2-4f3b-a9e7-165535663f85" />

<img width="1508" height="905" alt="截屏2026-04-02 17 48 06" src="https://github.com/user-attachments/assets/668a958b-91ab-4096-8948-da5f7a2e0b0e" />

<img width="1508" height="905" alt="截屏2026-04-02 17 48 25" src="https://github.com/user-attachments/assets/8eb83ec3-96e6-40bc-9d94-80c76903e046" />


# 智慧交通客服智能体（Smart CS Agent）

面向**高速公路 / 出行咨询**场景的对话式系统：对外表现为**智能客服**（多轮问答、业务引导），工程上实现为 **编排式多智能体 + 工具型主智能体**——**主智能体** `CustomerServiceAgent`（`main.py`）负责检索上下文、执行工具与渲染回复；**意图侧**由多个**子智能体**分工协作（见下节「架构说明」），规则路径由 **`IntentOrchestratorAgent`** 统一调度，天气多轮由 **`WeatherDialogAgent`** 优先接管，复杂句可再经 **同一 LLM** 输出结构化计划。这与「多个大模型角色自主对话、互相指派任务」的典型多智能体演示不同，本项目子智能体以 **规则与状态机** 为主、**可选单一 LLM** 辅助规划。后端基于 **FastAPI + LangChain**；**意图规划主链路**用 **LangGraph**（`intent/intent_planning_graph.py`）编排「天气子智能体 → LLM → 规则中枢」状态图；**对话模型使用 DeepSeek**（官方 OpenAI 兼容 `/v1` 接口）。实现上通过 LangChain 的 **`ChatOpenAI`** 客户端调用该协议——这是类名/包名（`langchain-openai`），**不表示调用 OpenAI 官方**。未配置 DeepSeek 时，仍可通过 `OPENAI_*` 环境变量接入其它 **同一协议** 的厂商作为备选。前端提供 **Vue 3（主）** 与 **纯静态 HTML（兜底）** 两套界面。

## 使用声明

本仓库**仅供个人学习使用**（含本地研究、试验与修改）。**商业使用**（营利性产品或服务、对外收费或内部经营场景等）**不在默示许可范围内**；如有商用需求，**请先与作者联系并取得书面同意后再使用**。

**联系作者**：可通过 GitHub [@niuchatang](https://github.com/niuchatang) 联系，或在本仓库提交 Issue 说明商用意向与用途。

---

## 项目定位

| 维度 | 说明 |
|------|------|
| **业务** | 路径规划、路况/管制、票价规则、退改签、失物招领、投诉与转人工等交通客服常见诉求 |
| **架构** | **编排式多智能体**：主智能体执行工具与话术；意图层含调度中枢与路况/路径/通用/天气等子智能体（见「架构说明」） |
| **技术** | RAG（BM25）增强知识问答；**LangGraph** 编排意图工作流；规则引擎优先、按需调用 LLM；结构化工具返回供前端地图与卡片展示 |
| **用户** | 需注册/登录（MySQL 存用户与会话）；`user_id` 区分同一账号下的不同对话线程 |

---

## 功能概览

### 对话与知识

- **多轮会话**：按用户与会话 ID 持久化到 `data/conversations.json`（与登录用户名组合作用域，避免串号）。
- **RAG**：从 `data/knowledge_base.json` 加载文档，使用 **LangChain BM25Retriever** 本地检索，低延迟。
- **意图路由（多智能体协作）**：`UserIntentAgent` 为意图总入口；内部通过 **LangGraph** 按序执行 **天气子智能体** →（未命中则）**LLM 结构化规划** →（仍未采用则）**规则编排**；**天气多轮**仍由 `WeatherDialogAgent` 在图首节点拦截；规则阶段由 **`IntentOrchestratorAgent`** 调度 **路况** `RoadConditionAgent`、**路径** `RoutePlanningAgent`、**通用** `GeneralIntentAgent`。各节点产出统一的 `intent + actions` 计划，由主智能体执行。
- **追问建议**：回复可带 `follow_ups`，前端以「你可能还想问」芯片展示，点击即发送。

### 路径与路况（工具侧）

- **驾车/出行路径**（`query_route_plan`）：支持「从 A 到 B」类说法；地理编码（如 Nominatim）+ 路线服务（**高德路径 API**（需 `AMAP_API_KEY`）、**OSRM 公共实例**、失败时本地估算等策略，以实际 `main.py` 为准）。
- **结构化结果**：含里程、耗时、折线点、途经高速等信息；后端会沿折线**逆地理编码**生成 `cities_along_route`（推断途经城市序列），供「查询途经城市天气」等使用；前端用 **Leaflet** 展示路线。
- **行程提示**（`trip_hints`）：由后端生成，与路径结果一并返回；前端在「路径规划」区展示卡片（非服务区 POI 推荐链路）。
- **高速路况**（`query_highway_condition`）：与路线/路段探测结合时可在地图上叠加拥堵示意（具体字段以前端与工具返回为准）。
- **天气**（`query_weather` + **天气对话智能体** `intent/weather_agent.py`）：有 `AMAP_API_KEY` 时优先高德天气，否则用 **Open-Meteo**。路径规划后的追问为肯定句 **「查询途经城市天气」**：点击后助手先**询问要查途经哪一座城市**，用户回复城市名后再查；也可回复 **「沿途」** 从第一站起逐站查，并用 **「继续查询下一途经城市（…）」** /「继续查询下一途经城市天气」/ 短「是」进下一站。实况缺失时仍展示明日预报。显式多城（如「上海、北京天气」）仍可一次查多城。成功出摘要后，回复中会附带基于气温与晴雨等的 **衣着建议**（规则模板，见上文「近期更新说明」）。

### 其他工具能力

- **公交/线路状态**（`query_transit_status`）：示例或内置线路数据（可扩展接真实 API）。
- **票价估算**（`calculate_fare`）。
- **工单类**（`create_transport_ticket`）：退票、失物、投诉等诉求的结构化登记。
- **转人工**（`handoff_to_human`）。

### 前端体验

- **Vue 应用**（构建输出目录 `static-vue/`，挂载在 `/static-vue/`）：登录页 + 对话主页；左侧路径/地图/途经高速/行程提示，右侧对话与输入；日夜间主题。
- **静态页兜底**：未构建 Vue 时使用 `static/index.html`、`static/login.html`。
- **标题区公路小游戏**（彩蛋 🥚）：顶部标题里藏着一个可交互的迷你避车游戏，详见「[彩蛋：顶部标题里的避车小游戏](#彩蛋顶部标题里的避车小游戏)」。

### 安全与运维

- **会话 Cookie**：HttpOnly，登录态校验保护 `/chat`、历史记录等接口。
- **`GET /health`**：服务与 `llm_enabled`、模型名。
- **`GET /debug/config`**：脱敏诊断（如 `llm_api_configured`、高德、MySQL 等，不返回密钥）。

---

## 近期更新说明

以下为实现层近期补充，便于对照代码与排查行为差异。

### 意图层多智能体（`intent/`）

- **LangGraph 意图工作流**（`intent/intent_planning_graph.py`）：将单轮规划写成 **weather / llm / rules / finalize** 节点，条件跳转与原先 `_plan` 一致；业务仍由各子智能体类实现，图只负责 **编排**。
- **`IntentOrchestratorAgent`**（`intent/orchestrator_agent.py`）：**调度中枢**；在 **LLM 规划未采用或不可用** 时，由规则回落统一调用各 **子智能体**，**顺序与原先单文件内规则一致**：显式高速编号 → 通用天气规则 → 路况中段（肯定答复高速、起终点+路况）→ **纯路径规划** → 路况末段（历史路线高速、泛路况等）→ 通用尾部（公交实时、票价、工单、转人工等）。
- **`RoadConditionAgent`**（`intent/road_condition_agent.py`）、**`RoutePlanningAgent`**（`intent/route_planning_agent.py`）、**`GeneralIntentAgent`**（`intent/general_intent_agent.py`）：路况 / 路径 / 通用 **子智能体**；**`UserIntentAgent._plan_by_rules`** 仅委托 **`orchestrator.plan_rules(...)`**。
- **`WeatherDialogAgent`**：**天气子智能体**，在 **`UserIntentAgent._plan` 最前**执行（早于 LLM 与编排器规则），负责途经天气芯片、沿途逐站队列、短「是」衔接等；**不经过** `IntentOrchestratorAgent` 的规则链。

### 天气问询纠偏

- **无具体城市**（如「查询天气」「帮我查天气」）：避免正则把 **「查询」** 等当成城市名调用高德；由 **`WeatherDialogAgent`** 识别 **裸查天气短句** 并直接返回澄清话术（`WEATHER_CITY_CLARIFY_REPLY`），通用规则侧增加 **非城市词** 集合过滤，天气规则兜底也附带同一类澄清文案。
- **「途径」与「途经」简写**：扩展与 **「查询途经城市天气」** 等价的 **短句芯片**（如「查询途径天气」）及去空白匹配；**「途径」** 纳入路线相关语境，**「途径天气」类** 在有路线城市时先 **追问查哪一城**（与芯片流程一致），避免误查虚构地名。

### 天气回复中的衣着建议

- 在 **`main.py`** 的 **`_render_reply`** 中，当 **`intent == weather_query`** 且 **`query_weather` 成功** 时，在「城市天气摘要」之后、「长途出行…」提示之前，追加 **「衣着建议」** 段落。
- 实现为 **`CustomerServiceAgent._weather_block_effective_temp_and_desc`**（从高德实况/预报或 Open-Meteo 抽取代表气温与描述）与 **`_outfit_recommend_cn`**（按温区与晴雨/大风等关键词的 **规则模板**），**不额外调用大模型**，多城则按城市分条列出。

---

## 多智能体扩展（本轮升级）

本轮升级遵循两条硬约束：**不改动 `main.py` 主干** + **跳过需要专属/内部系统的能力**。所有改造通过 `intent/` 子系统和四个新模块（`tools_infra/`、`safety/`、`rag/`、`evaluation/`）落地，并用 `tests/` 兜单测。详细接入示例见 **`UPGRADES.md`**，速览见 **`UPGRADE_SUMMARY.md`**。

### 1. 可插拔扩展底座：`AgentRegistry`

- **文件**：`intent/agent_registry.py`（`AgentRegistry` + `ExtensionAgent` 协议）、`intent/orchestrator_agent.py`（接入 registry）。
- **挂点**：`IntentOrchestratorAgent.plan_rules` 中，于 `try_late_corridor` 之后、`try_tail_rules` 之前，按 `priority` 升序尝试 `registry.try_plan(message, history)`。
- **关闭方式**：`IntentOrchestratorAgent(..., enable_extensions=False)`。
- **设计理念**：所有扩展复用既有 `IntentType` 枚举，通过 `actions` / `llm_reply` 承载新业务，**不修改 `main.IntentType` 与 Pydantic 模型**。

### 2. 新增子智能体（16 个）

**自动挂到规则链（按优先级生效）：**

| 优先级 | 智能体 | 文件 | 职责 |
|---|---|---|---|
| 20 | `ETCChargeAgent` | `intent/etc_charge_agent.py` | ETC / 过路费估算，节假日免费提醒 |
| 30 | `ServiceAreaAgent` | `intent/service_area_agent.py` | 路线上的服务区 / 充电站 / 加油站 |
| 35 | `DepartureTimeAgent` | `intent/departure_time_agent.py` | 最佳出发时间（早晚高峰 / 节假日启发式） |
| 40 | `TrafficIncidentAgent` | `intent/traffic_incident_agent.py` | 事故 / 管制 / 封路主动查询 |
| 50 | `WeatherImpactAgent` | `intent/weather_impact_agent.py` | 雨 / 雾 / 雪 / 大风 / 高温对驾驶的建议 |
| 60 | `AccessibilityAgent` | `intent/accessibility_agent.py` | 无障碍出行（轮椅、导盲犬、无障碍设施） |
| 85 | `ClarifyAgent` | `intent/clarify_agent.py` | 低置信度时主动追问缺失要素 |

**按需显式调用（留接入点，未强绑主流程）：**

| 智能体 | 文件 | 职责 |
|---|---|---|
| `GuardrailAgent` | `intent/guardrail_agent.py` | 入站查 prompt injection + PII 脱敏；出站扫描承诺性话术 |
| `MultilingualAgent` | `intent/multilingual_agent.py` | 中 / 日 / 韩 / 俄 / 英 / 粤语识别 + 可选翻译 |
| `ComplaintTriageAgent` | `intent/complaint_triage_agent.py` | 投诉分级（low / medium / high / urgent） |
| `FAQAgent` | `intent/faq_agent.py` | 专职 RAG 问答 + 引用渲染 |
| `ProfileAgent` | `intent/profile_agent.py` | 从消息抽用户偏好，落盘 `data/user_profiles.json` |
| `ConversationSummarizer` | `intent/summarizer.py` | 历史折叠成摘要，缩 prompt 长度 |
| `ToolRouterAgent` | `intent/tool_router_agent.py` | 工具失败 / 空结果时给回退方案 |
| `SatisfactionAgent` | `intent/satisfaction_agent.py` | 会话结束弹满意度，写 `data/satisfaction.jsonl` |
| `EvalAgent` | `intent/eval_agent.py` | 离线意图回归评测，CLI 可调用 |

### 3. 新增基础设施模块

| 模块 | 关键能力 | 关键文件 |
|---|---|---|
| `tools_infra/` | `TTLCache`（OrderedDict + LRU + TTL，线程安全）、`@cached` 装饰器；`ToolSpec`（schema / timeout / retry / fallback / cache）+ `ToolRunner`（熔断计数） | `tools_infra/cache.py`、`tools_infra/registry.py` |
| `safety/` | `mask_pii` / `pii_hits` / `scan_forbidden` / `sanitize_reply`，与 Guardrail 规则等价，可在非 Agent 处复用 | `safety/pii.py`、`safety/moderation.py` |
| `rag/` | `HybridRetriever`：BM25 + 可选向量检索，用 **RRF** 融合；无向量模型时自动退化为纯 BM25 | `rag/hybrid_retriever.py` |
| `evaluation/` | 离线意图回归：`intent_cases.jsonl`（20 条用例）+ `python -m evaluation.eval` | `evaluation/eval.py`、`evaluation/intent_cases.jsonl` |
| `tests/` | 单测覆盖 guardrail / registry / summarizer / tools 缓存 | `tests/test_*.py` |

### 4. 触发示例（快速验证）

启动后在前端或 `/docs` 直接丢以下话术：

| 话术 | 应命中 |
|---|---|
| `广州到深圳ETC大概多少钱` | `ETCChargeAgent` |
| `这条路线上有哪些服务区` | `ServiceAreaAgent` |
| `明天几点出发最合适` | `DepartureTimeAgent` |
| `下雨天开高速要注意什么` | `WeatherImpactAgent` |
| `G4京港澳今天有事故吗` | `TrafficIncidentAgent` |
| `轮椅能上高铁吗` | `AccessibilityAgent` |
| `查路线`（信息不全） | `ClarifyAgent` |

命令行验证：

```bash
python -m pytest -q tests
python -m evaluation.eval --cases evaluation/intent_cases.jsonl
```

### 5. 故意没做的（依赖内部系统）

| 跳过项 | 原因 | 已留好的扩展点 |
|---|---|---|
| 真实工单 / CRM / 坐席队列 | 需要内部 CRM API | `ComplaintTriageAgent` + `AgentRegistry` |
| 12306 / 航司 / 地铁实时数据 | 需要专属数据源授权 | `ToolRunner` + `AgentRegistry` |
| 高速集团官方事故公告抓取 | 需要内部公告系统 | `TrafficIncidentAgent` + `ToolRunner` |
| 真实满意度后台 / BI | 需要内部报表系统 | `SatisfactionAgent` 已把数据落到 jsonl |
| LangSmith / OpenTelemetry trace | 需要外部账号 | `ToolRunner` 有打点位 |
| 语音 / 流式前端 | 需要改 Vue + WebSocket | 暂未改前端 |

---

## 彩蛋：顶部标题里的避车小游戏 🥚

打开对话页之后，别急着输入问题——看一眼**顶部标题栏下方的那条双向高速**。那不是一张静态贴图，而是一个**被藏起来的迷你小游戏**。

### 怎么玩

1. 把鼠标**移到标题区（那条高速路面上）**，会看到中间那辆小车跟着你的鼠标走。
2. **鼠标横向移动** → 小车左右变道（硬约束在路面内）；
3. **鼠标纵向越过中线** → 小车**切换上下车道**（两条车道方向相反）；
4. 每隔约 **2.1 秒**会从左右两边**随机刷出一辆对向车**，同车道的车之间自动保持最小间距，不会"叠模型"；
5. **撞上任何一辆对向车** → 当前回合立即重置：所有来车清掉，小车回到中央，计时从 -0.8 秒开始（给你一小段「神圣时间」喘口气）。

### 它在哪儿

- **Vue 版前端**：`frontend/src/views/ChatView.vue` 里的 `initTitleCarAnimation()`（约第 422 行）。
- **静态兜底前端**：`static/index.html` 同名函数（约第 1312 行）。
- 两边的 DOM 节点分别是 `highwaySceneRef` / `carRef` / `trafficLayerRef`（Vue）和对应的 id/class（静态版）。

### 技术要点（给想看看代码的人）

- **60 FPS**：用 `requestAnimationFrame` 自己驱动，不依赖 CSS 动画，撞击判定每帧都跑。
- **横向位移只用 `transform: translate3d`**，避免每帧改 `left` 触发 layout 抖动；子像素保留两位小数减少抖动感。
- **车道几何**：两条车道的 `bottom` 值故意错开到 14 / 54 px，配合 22 px 的车体高度，保证两车道的碰撞盒不跨线重叠。
- **最小车距**：每条车道内按方向排序后，逐个把后车推到「前车尾 + `minCarGap(24)`」之后，既写实又不撞。
- **撞击判定**：经典 AABB（轴对齐矩形）`intersects()`，任意一辆来车命中即 `resetRun()`。
- **和业务零耦合**：不调任何 API、不写 localStorage、不影响对话状态；纯前端装饰，可随时删除不破坏功能。

### 彩蛋玩法建议

- **看你能不能连撞 10 秒不挂**——没有计分板，纯靠手感（想给它加个 high score 也就几行事）；
- **双屏 / 触控板用户**：纵向手势过于灵敏时，可以把鼠标停在路面边缘练习单道连续行驶；
- **审美小建议**：夜间主题下路面和车灯对比更强，体验更好——右上角切换主题按钮顺手一试。

> 提示：这个小游戏只是**顺手写的装饰交互**，当初是为了让空荡的标题区不那么死板。如果你想扩展它（计分 / 道具 / 排行榜），入口就是 `initTitleCarAnimation()`，改动完全不会影响智能客服主流程。

---

## 意图类型（Intent）

代码中与规划、回复逻辑一致的意图包括：

- `route_planning` — 路线规划
- `realtime_status` — 实时线路/班次类（与纯路况说法可能归并到高速路况意图）
- `highway_condition` — 高速事故、管制、拥堵等
- `weather_query` — 城市/沿途天气查询（`query_weather`）
- `fare_policy` — 票价与优惠规则
- `ticket_refund` — 退票/改签
- `lost_and_found` — 失物招领
- `complaint` — 投诉
- `human_handoff` — 转人工
- `unknown` — 未命中或通用知识问答（部分场景会走 RAG）

---

## 架构说明

```
用户浏览器
    ↓
FastAPI（main.py）
    ├─ 鉴权（MySQL：用户、会话）
    ├─ POST /chat → CustomerServiceAgent（主智能体）.chat()
    │       ├─ 读历史（conversations.json）
    │       ├─ RAG 检索（BM25；可选升级为 HybridRetriever = BM25 + 向量 + RRF）
    │       ├─ UserIntentAgent.parse() → LangGraph 意图状态图（intent/intent_planning_graph.py）
    │       │       weather 节点：WeatherDialogAgent
    │       │       → llm 节点：可选结构化规划
    │       │       → rules 节点：IntentOrchestratorAgent
    │       │               ├─ Road / Route / General 既有子智能体
    │       │               └─ AgentRegistry（本轮升级）
    │       │                   ETC / ServiceArea / DepartureTime /
    │       │                   TrafficIncident / WeatherImpact /
    │       │                   Accessibility / Clarify …… 按 priority 尝试
    │       ├─ 执行工具 → tool_results[]（可选走 ToolRunner：cache / retry / 熔断 / fallback）
    │       ├─ Guardrail / ComplaintTriage / ProfileAgent（按需显式调用）
    │       └─ 渲染 reply + follow_ups
    └─ 静态资源：/static-vue/ 或 static/
```

- **编排式多智能体**：**主智能体** `CustomerServiceAgent` 持有对话闭环与工具执行；**意图层**在 `intent/` 内由 **`UserIntentAgent`** 调用 **LangGraph** 编译图（`compile_user_intent_planning_graph`），节点顺序为 **天气 → LLM → 规则编排**，与原先 `if/else` 逻辑等价；输出统一结构的计划后仍由主智能体执行（单进程、非分布式）。
- **单文件核心**：路由、鉴权、工具实现与话术模板主体在 `main.py`；意图解析按上式拆分为多类 **Agent** 文件，便于维护与扩展。
- **前端构建**：`frontend/` 为 Vite + Vue 3 源码，`npm run build` 输出到 `static-vue/`。

---

## 技术栈详解与涉及文件

以下按技术领域分点说明：**技术是什么**、**在本项目里做什么**、**主要落在哪些文件**（后端大逻辑均在 `main.py` 内按类/函数组织，下文写到的「位置」以该文件内的职责为准）。

### 1. 语言与运行时

- **Python 3**  
  - 后端唯一实现语言。  
  - **文件**：`main.py`（全集）、`requirements.txt`（依赖清单）。

### 2. Web 服务层

- **FastAPI**  
  - 提供 REST API、依赖注入（如登录用户 `Depends(get_current_user)`）、请求/响应模型。  
  - **文件**：`main.py`（`app = FastAPI()`、各 `@app.get` / `@app.post` 等路由）。
- **Uvicorn**  
  - ASGI 服务器，用于启动 FastAPI。  
  - **文件**：命令行启动；依赖见 `requirements.txt`。
- **Starlette**（FastAPI 底层）  
  - `Request` / `Response`、重定向、静态文件挂载等。  
  - **文件**：`main.py`（`FileResponse`、`RedirectResponse`、`StaticFiles` 挂载）。
- **Pydantic v2**  
  - 定义 `ChatRequest`、`ChatResponse`、`AuthUser` 等模型，做校验与序列化。  
  - **文件**：`main.py` 前部各 `class X(BaseModel)`。

### 3. 配置与环境

- **python-dotenv**  
  - 启动时加载 `.env`，避免把密钥写进代码。  
  - **文件**：`main.py`（`load_dotenv()`）、`.env`（本地，勿提交）、`.env.example`（变量模板）。

### 4. 数据存储

- **MySQL**  
  - 存储注册用户、密码哈希、盐值、登录会话（Token 与过期时间）。  
  - **文件**：`main.py` 中 `AuthStore` 类（连接、`CREATE TABLE`、注册/登录/会话校验）；环境变量见 `.env.example`。
- **PyMySQL**  
  - MySQL 驱动，字典游标查询。  
  - **文件**：`main.py`（`import pymysql`、`DictCursor`）。
- **JSON 文件（本地持久化）**  
  - **会话消息**：`data/conversations.json`，按会话 ID（与用户名作用域组合）追加消息与 `meta`（意图、`tool_results`、`follow_ups` 等）。  
  - **RAG 语料**：`data/knowledge_base.json`，加载为多条 `Document`。  
  - **文件**：`main.py` 中对话存储 / RAG 加载逻辑；数据实体为上述两个 JSON。

### 5. 大模型与 LangChain 生态

- **对话模型（DeepSeek + `langchain-openai` 包内的 `ChatOpenAI` 客户端）**  
  - **默认厂商为 DeepSeek**：环境变量 **`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL_ID`**（解析优先级与默认值见 `main.py` 中 `_resolve_*`）。  
  - **`ChatOpenAI`** 仅表示「OpenAI **兼容** HTTP API」的 LangChain 封装；**本项目主线不依赖 OpenAI 官方账号**。  
  - 用于两类调用：**① 用户意图解析（结构化 JSON 计划，见 `intent/user_intent_agent.py`）**；**② 在特定意图下基于 RAG 片段生成短回答**。  
  - **文件**：`intent/user_intent_agent.py`（`UserIntentAgent.parse` → LLM/规则规划）；`main.py` 中 `CustomerServiceAgent`（`_build_llm`、`_build_answer_chain`、`_render_reply`、工具执行等）。  
  - **流程级说明**：见下文 **「大模型（LLM）工作流详解」**。
- **`langchain-core`**  
  - `ChatPromptTemplate` 拼装系统/用户提示；`StrOutputParser` 取模型文本；`Document` 表示知识库条目（与 BM25 配合，**不经过向量 embedding 模型**）。  
  - **文件**：`main.py`。
- **`langchain-community` → `BM25Retriever`**  
  - 本地 **BM25** 稀疏检索（依赖 `rank-bm25` 等）；检索阶段**不调用大模型**。  
  - **文件**：`main.py`（`SimpleRAGStore`）。
- **LangGraph**  
  - LangChain 生态下的 **状态图 / 工作流** 库：将 `UserIntentAgent` 内「天气优先 → LLM → 规则」固化为 **StateGraph** 节点与条件边，便于与 LangChain 工具链统一维护。  
  - **文件**：`intent/intent_planning_graph.py`（`compile_user_intent_planning_graph`）；`intent/user_intent_agent.py`（懒编译图、`invoke` 取 `plan`）。  
  - **依赖**：`requirements.txt` 中 `langgraph>=0.2.0,<0.3`（大版本升级时需对照 LangGraph 迁移说明调整 API）。
- **依赖包**  
  - **文件**：`requirements.txt`（`langchain`、`langgraph`、`langchain-openai`、`langchain-community`、`rank-bm25`）。

### 6. HTTP 与外部地理 / 路径服务

- **`requests`**  
  - 调用高德、OSRM、Nominatim 等 HTTP API，以及其它需客户端发起的请求。  
  - **文件**：`main.py`（路径规划、地理编码、路况等工具函数内）。
- **高德开放平台（REST）**  
  - 配置 `AMAP_API_KEY` 时用于驾车路径等（具体接口以 `main.py` 实现为准）。  
  - **文件**：`main.py`；密钥 **`.env` / `.env.example`**。
- **OSRM（公共路由服务）**  
  - 无密钥或降级时的路径计算备选。  
  - **文件**：`main.py`。
- **OpenStreetMap Nominatim（地理编码）**  
  - 将地名转为经纬度（需遵守 Nominatim 使用政策）。  
  - **文件**：`main.py`。
- **OpenStreetMap 瓦片（地图展示）**  
  - 前端 Leaflet 使用 OSM 标准瓦片 URL 渲染底图。  
  - **文件**：`frontend/src/views/ChatView.vue`（或构建后的 `static-vue` 内联脚本）；静态版见 `static/index.html` 中地图初始化。

### 7. 安全与密码学（标准库）

- **`secrets` / `hashlib` / `hmac`**  
  - 密码盐、哈希、会话 Token 生成等。  
  - **文件**：`main.py`（`AuthStore` 内）。
- **HttpOnly Cookie**  
  - 登录态通过 Cookie 传递，减少 XSS 窃取风险。  
  - **文件**：`main.py`（登录/登出时 `set_cookie` / `delete_cookie`）。

### 8. 前端：构建工具与框架

- **Vite**  
  - 开发服务器、打包 ES 模块、产出到 `static-vue/`。  
  - **文件**：`frontend/vite.config.js`（`outDir: '../static-vue'`、`base: '/static-vue/'`）、`frontend/package.json`（`npm run dev` / `build`）。
- **Vue 3**  
  - 组合式 API、单文件组件（SFC）。  
  - **文件**：`frontend/src/main.js`（`createApp`、挂载路由）、`frontend/src/App.vue`（根组件）、`frontend/src/views/ChatView.vue`（主聊天与地图）、`frontend/src/views/LoginView.vue`（登录注册）。
- **Vue Router**  
  - 前端路由：`/` 对话、`/login` 登录。  
  - **文件**：`frontend/src/router.js`。
- **原生 `fetch` + Cookie**  
  - 与后端同域携带 `credentials: 'same-origin'`。  
  - **文件**：`frontend/src/api.js`（`apiGet` / `apiPost` / `apiDelete`）。
- **Leaflet**  
  - 展示路径折线、起终点标记、拥堵叠加层等。  
  - **文件**：`frontend/src/main.js`（`import 'leaflet/dist/leaflet.css'`）、`frontend/src/views/ChatView.vue`（`import L from 'leaflet'` 及地图逻辑）。
- **全局样式与主题**  
  - 日夜间主题通过 `body.theme-day` 与 `localStorage('ui-theme')` 切换。  
  - **文件**：`frontend/src/style.css`、`frontend/index.html`（首屏脚本切换 class）、`ChatView.vue` / `LoginView.vue` 内按钮与逻辑。

### 9. 前端业务工具模块

- **`routeDisplay.js`**  
  - 解析后端路径结果、途经高速卡片数据、多段路线等，供界面展示。  
  - **文件**：`frontend/src/utils/routeDisplay.js`；由 `ChatView.vue` 引用。

### 10. 静态资源与其它前端文件

- **HTML 入口（开发）**  
  - **文件**：`frontend/index.html`（挂载 `#app`、引入 `main.js`）。
- **脚手架残留组件（可选）**  
  - **文件**：`frontend/src/components/HelloWorld.vue`（若未在路由中使用可忽略或删除）。
- **图标与静态公共资源**  
  - **文件**：`frontend/public/favicon.svg`、`frontend/public/icons.svg`；`frontend/src/assets/vue.svg`、`vite.svg` 等。
- **构建产物**  
  - **目录**：`static-vue/`（`index.html`、`assets/*.js`、`assets/*.css` 等），由 `npm run build` 生成，**勿手改**，应改 `frontend/` 源码。

### 11. 无构建兜底前端（CDN + 单文件）

- **Vue 3（CDN ESM）+ 内联逻辑**  
  - 不依赖 `npm run build` 也能使用完整聊天与地图流程（与 Vue 工程版功能对齐度以仓库为准）。  
  - **文件**：`static/index.html`（样式、`createApp`、路由模拟、Leaflet、`initTitleCarAnimation` 等）、`static/login.html`（登录注册页）。

### 12. 运维与诊断接口

- **健康检查 / 配置探测**  
  - **文件**：`main.py`（`GET /health`、`GET /debug/config`）。

---

## 大模型（LLM）工作流详解

本节对应 **`main.py` 中 `CustomerServiceAgent`** 的实现逻辑，便于对照代码阅读。

### 1. 何时启用大模型（DeepSeek 为主，`OPENAI_*` 为备选）

本项目**以 DeepSeek 为默认对话模型**。在 `.env` 中配置 **`DEEPSEEK_API_KEY`**，以及（建议）**`DEEPSEEK_BASE_URL`**（如 `https://api.deepseek.com`，代码会自动补上 `/v1`）、**`DEEPSEEK_MODEL_ID`**（如 `deepseek-chat`）即可。

**`main.py` 中 `_resolve_api_key` / `_resolve_base_url` / `_resolve_model` 的约定：**

- **API Key**：**优先 `DEEPSEEK_API_KEY`**，否则 **`OPENAI_API_KEY`**（供其它 OpenAI 兼容厂商或自建网关使用）。  
- **Base URL**：**优先 `DEEPSEEK_BASE_URL`（及 `/v1` 处理）**；否则 **`OPENAI_BASE_URL`**；若两者都未配置，则根据**最终生效的是哪类 Key**选择默认端点——仅配 DeepSeek Key 时为 `https://api.deepseek.com/v1`，仅配 `OPENAI_API_KEY` 时为 `https://api.openai.com/v1`。  
- **模型 ID**：**优先 `DEEPSEEK_MODEL_ID`**，否则 **`OPENAI_MODEL`**；若都未写，则根据 Key 类型默认 **`deepseek-chat`** 或 **`gpt-4o-mini`**；无任何 Key 时占位默认仍为 **`deepseek-chat`**（此时不会真正发起请求，因 `_build_llm` 要求 Key 非空）。

**未配置任何可用 Key** 时：`self.llm` 为 `None`，**`llm_enabled` 为假**，意图与动作完全走 **`_plan_by_rules`**（经 **`IntentOrchestratorAgent`** 规则链）；`GET /health` 会反映是否启用模型。

**`GET /debug/config`** 中布尔字段为 **`llm_api_configured`**（表示是否配置了任一兼容 Chat API 的 Key），不再使用 `openai_configured` 命名。

### 2. 模型客户端与参数

- **`_build_llm()`**：在存在 API Key 时构造 **`ChatOpenAI`**：`temperature=0.1`（规划更稳）、`timeout=20`（秒）。  
- **`_build_answer_chain()`**：在 `self.llm` 非空时返回 **`ChatPromptTemplate | llm | StrOutputParser`** 链，专用于 **RAG 约束回答**；若未配置 LLM 则为 `None`。

### 3. 每一轮用户消息的整体顺序（与 LLM 相关部分）

在 **`chat()`** 中（简化）：

1. **`rag_hits = self.rag_store.retrieve(message, k=3)`** — 纯 **BM25**，不调用大模型。  
2. **`plan = self._plan(message, history, rag_hits)`** — 此处决定是否、如何用 LLM 做规划（见下节）。  
3. **`tool_results = self._execute_actions(plan["actions"])`** — 执行工具，与 LLM 无关。  
4. **`reply = self._render_reply(..., llm_reply=plan.get("llm_reply"), rag_hits=...)`** — 可能再次调用 **`answer_chain`**（见 §6）。  
5. 返回 **`ChatResponse`**，其中 **`used_llm`** 来自 `plan.get("used_llm")`，表示本轮规划阶段是否走了模型（若尝试过但解析失败回落规则，仍可能标记为已尝试，见 `_plan` 内逻辑）。

### 4. 规划阶段：`_plan` → `_plan_by_llm` → `_post_process_llm_plan`

- **`_plan`**  
  - 若 **`llm_enabled`**：依次调用 **`_plan_by_llm`** → **`_post_process_llm_plan`**，若结果满足 **`_is_usable_plan`**（意图在枚举内且 `actions` 为列表等），则**采用该计划**。  
  - 若模型未启用、调用抛错、或 JSON 无效、或计划不可用：回落 **`_plan_by_rules`**（内部委托 **`IntentOrchestratorAgent.plan_rules`**，按路况/路径/通用子模块顺序匹配关键词与历史启发式）。  
  - 若本已启用 LLM 且**尝试过**但回落到规则，会把 **`used_llm=True`** 写入规则计划，便于前端/日志区分「模型参与过但未果」。

- **`_plan_by_llm`**  
  - 将 **最近对话**、**最近一次路径规划结构化摘要**、**RAG 检索片段文本** 与用户当前句一并写入提示模板。  
  - 要求模型**仅输出 JSON**，字段包括：`intent`、`confidence`、`actions`（含 `tool` / `params`）、**`llm_reply`**（可选，若填则可能在 `_render_reply` 中直接作为最终回复）。  
  - 使用 LangChain 链：**`planner_prompt | self.llm | StrOutputParser()`**，得到字符串后再 **`_parse_llm_plan_json`** 解析。  
  - **`_parse_llm_plan_json`**：去掉 Markdown 代码围栏、处理 ```json；若仍失败则用正则从正文中**抠出第一段 `{...}`**，以兼容部分模型在 JSON 外夹杂说明文字的情况。

- **`_post_process_llm_plan`**  
  - 在模型输出基础上做**业务向修正**：例如将明显「高速路况」说法从 `realtime_status` 纠到 `highway_condition`、根据起终点补充 **`query_route_plan`** 与 **`od_traffic_followup`**、处理「好的/可以」类短确认与历史中多条高速编号等。  
  - 具体分支以 **`main.py` 源码为准**，README 只概括其作用。

### 5. `llm_reply` 与最终展示文案

- 在 **`_render_reply`** 中：**若 `llm_reply` 非空字符串，则直接作为返回给用户的 `reply`**，不再走后续模板拼接（优先级最高）。  
- 否则按 **`intent`** 与 **`tool_results`** 使用大量 **Python 字符串模板** 生成回复（路径成功/失败、高速路况列表、工单号等）。

### 6. RAG + 大模型的「知识问答链」`answer_chain`

- **触发条件**：`_render_reply` 中，当 **`intent` 属于 `fare_policy`、`route_planning` 或 `unknown`**，且存在 **`rag_hits`**，且 **`answer_chain` 已构建**。  
- **行为**：`answer_chain.invoke({"question": message, "context": _rag_to_text(rag_hits)})`，提示词要求**严格依据给定片段**、先给结论、控制在约 1～3 句，并带「知识依据」式说明。  
- **失败兜底**：若调用异常或链为 `None`，则退回为**拼接第一条 RAG 命中正文**（带标题来源）。

### 7. 明确不交给大模型生成的部分

- **追问芯片 `follow_ups`**：由 **`_build_follow_ups` / `_follow_ups_fallback`** 用**规则模板**生成，注释中说明为避免模型**编造点击后无法兑现**的追问。  
- **工具内部实现**（路径、高德、OSRM、Nominatim 等）均为代码逻辑 + HTTP，**不由 LLM 直接写 API 结果**。

### 8. 与 LLM 相关的主要符号索引（均在 `main.py`）

| 符号 | 作用 |
|------|------|
| `_resolve_api_key` / `_resolve_base_url` / `_resolve_model` | 从环境变量解析 Key、Base URL、模型名 |
| `llm_enabled`（property） | 是否已创建 `ChatOpenAI` 实例 |
| `_build_llm` / `_build_answer_chain` | 构造规划用与 RAG 答问用链 |
| `chat` | 编排检索、规划、工具、渲染与 `used_llm` |
| `_plan` / `_plan_by_llm` / `_plan_by_rules` | 规划主入口与两路实现；规则回落委托 `IntentOrchestratorAgent` |
| `_post_process_llm_plan` / `_parse_llm_plan_json` / `_is_usable_plan` | 解析与校验模型规划输出 |
| `_render_reply` | 合并 `llm_reply`、工具结果模板与 `answer_chain` |
| `_history_to_text` / `_extract_recent_route_context` / `_rag_to_text` | 拼进规划提示的上下文 |

---

## 目录结构（速查）

| 路径 | 说明 |
|------|------|
| `main.py` | 后端单体：智能体、RAG、工具、路径与路况、天气渲染（含衣着建议）、鉴权、路由、静态入口逻辑 |
| `intent/` | **编排式多智能体·意图层**：`UserIntentAgent`、`intent_planning_graph.py`（LangGraph）、调度中枢、16 个子智能体、`AgentRegistry` 扩展底座 |
| `tools_infra/` | **本轮新增**：`TTLCache` + `ToolSpec` + `ToolRunner`（重试 / 熔断 / 回退 / 缓存） |
| `safety/` | **本轮新增**：PII 脱敏、出站内容审核（可在非 Agent 侧复用） |
| `rag/` | **本轮新增**：`HybridRetriever`（BM25 + 可选向量检索 + RRF 融合） |
| `evaluation/` | **本轮新增**：离线意图回归评测用例与 CLI |
| `tests/` | **本轮新增**：guardrail / registry / summarizer / tools 单测 |
| `UPGRADES.md` / `UPGRADE_SUMMARY.md` | **本轮新增**：升级接入手册与速览 |
| `frontend/` | Vue 3 + Vite 源码 |
| `frontend/vite.config.js` | 构建到 `static-vue/`、`base` 配置 |
| `static-vue/` | 构建产物（存在则 `GET /` 优先返回此 SPA） |
| `static/index.html`、`static/login.html` | 无构建时的双页前端 |
| `data/knowledge_base.json` | RAG 知识库 |
| `data/conversations.json` | 多轮对话记录（可很大，注意备份与隐私；已加入 `.gitignore`） |
| `data/user_profiles.json` / `data/satisfaction.jsonl` | **本轮新增**：用户画像与满意度落盘（运行期生成，已 `.gitignore`） |
| `requirements.txt` | Python 依赖 |
| `.env.example` / `.env` | 环境变量模板与实际配置 |
| `Dockerfile` / `docker-compose.yml` | 容器化构建与一键编排（含 MySQL） |
| `.dockerignore` | Docker 构建忽略项 |

更细的前后端文件职责见上一节 **「技术栈详解与涉及文件」**。

---

## 环境变量（`.env`）

复制 `.env.example` 为 `.env` 后按需填写：

| 变量 | 作用 |
|------|------|
| **`DEEPSEEK_API_KEY`** / **`DEEPSEEK_BASE_URL`** / **`DEEPSEEK_MODEL_ID`** | **主用**：DeepSeek 官方兼容接口（规划 + RAG 答问链）。示例：`DEEPSEEK_BASE_URL=https://api.deepseek.com`，`DEEPSEEK_MODEL_ID=deepseek-chat`。 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | **可选**：其它 OpenAI 兼容服务；**仅当未使用 DeepSeek 或需回退时再配**（优先级见上文「大模型工作流」§1）。 |
| `AMAP_API_KEY` | 高德路径等服务；未配置时倾向 OSRM / 回退方案 |
| `AMAP_BYPASS_PROXY` | 高德请求是否绕过代理（按部署环境调整） |
| `MYSQL_*` | 用户注册登录、会话；连接失败时鉴权不可用（注册/登录将报错） |

---

## 安装与运行

### 1. Python 后端

```bash
cd smart-cs-agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

复制环境变量模板并配置 **DeepSeek**（推荐）：

```bash
cp .env.example .env
# 编辑 .env：至少填写 DEEPSEEK_API_KEY；DEEPSEEK_BASE_URL / DEEPSEEK_MODEL_ID 可按 .env.example 默认
```

### 2. MySQL

创建数据库（名称与 `.env` 中 `MYSQL_DATABASE` 一致，默认可为 `smart_cs_agent`），保证 `MYSQL_USER` / `MYSQL_PASSWORD` 可连接；首次启动会由代码自动建表。

### 3. 前端（可选，推荐）

```bash
cd smart-cs-agent/frontend
npm install
npm run build
```

构建完成后，根目录会出现/更新 `static-vue/`，访问根路径将使用 Vue 界面。

### 4. 启动服务

```bash
cd smart-cs-agent
uvicorn main:app --reload --port 8010
```

- 主页：<http://127.0.0.1:8010>（未登录会跳转登录页）
- 开发调试前端：`frontend` 目录下 `npm run dev`（`vite.config.js` 已把 `/auth`、`/chat`、`/history` 等代理到本机 `http://127.0.0.1:8010`，需先启动后端；端口不同可设环境变量 `VITE_API_TARGET`）

---

## Docker 部署

仓库提供 **`Dockerfile`**（多阶段：构建 Vue → 安装 Python 依赖 → 运行 Uvicorn）与 **`docker-compose.yml`**（**MySQL 8** + **应用** 两个服务）。

### 1. 准备 `.env`

在**项目根目录**保留或创建 `.env`，至少包含：

- **`DEEPSEEK_API_KEY`**、**`DEEPSEEK_BASE_URL`**、**`DEEPSEEK_MODEL_ID`**（或你实际使用的大模型变量）
- **`AMAP_API_KEY`**（可选，影响高德路径/路况）
- **`MYSQL_ROOT_PASSWORD`**：**Docker Compose 专用**，作为 Compose 里 MySQL 容器的 **root 初始密码**；`docker-compose.yml` 会把同一值注入应用容器的 **`MYSQL_PASSWORD`**，使应用用 `root` 连接数据库

本地开发若已有 **`MYSQL_PASSWORD`**，部署 Compose 时建议在 `.env` 里**增加一行** `MYSQL_ROOT_PASSWORD=`**与之相同**，避免两套密码不一致。

> `docker-compose.yml` 会**强制覆盖**容器内 **`MYSQL_HOST=mysql`**、**`MYSQL_UNIX_SOCKET=`**（空），避免沿用你本机 `127.0.0.1` / `socket` 配置导致连不上库。

### 2. 构建并启动

```bash
cd smart-cs-agent
docker compose up -d --build
```

- 应用：<http://127.0.0.1:8010>（端口可用环境变量 **`APP_PORT`** 修改，默认 `8010`）
- MySQL 默认映射到本机 **`MYSQL_PUBLISH_PORT`**（默认 `3306`），仅调试或外部客户端需要时可改端口避免与宿主机已有 MySQL 冲突

### 3. 数据持久化

- **`mysql_data` 卷**：数据库文件  
- **`app_data` 卷**：挂载到容器内 **`/app/data`**，存放 **`conversations.json`** 等。首次启动时 **`docker/entrypoint.sh`** 若发现卷中尚无 **`knowledge_base.json`**，会从镜像内默认文件复制一份，避免空卷覆盖导致 RAG 无数据。

### 4. 仅构建镜像（不用 Compose）

```bash
docker build -t smart-cs-agent:latest .
docker run --rm -p 8010:8010 \
  -e MYSQL_HOST=你的MySQL主机 \
  -e MYSQL_PORT=3306 \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=... \
  -e MYSQL_DATABASE=smart_cs_agent \
  -e MYSQL_UNIX_SOCKET= \
  -e DEEPSEEK_API_KEY=... \
  smart-cs-agent:latest
```

需自备可访问的 MySQL，且不要用宿主机 `127.0.0.1` 指代「容器自己」；在 Linux 上可用 **`host.docker.internal`** 或 Docker 网络别名。

### 5. 相关文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 前端 `npm run build` + Python 镜像运行 `main.py` |
| `docker/entrypoint.sh` | 启动前补全 `knowledge_base.json`（命名卷为空时） |
| `docker-compose.yml` | `mysql` + `app`，健康检查与卷 |
| `.dockerignore` | 减小构建上下文（排除 `node_modules`、本地 `.env` 不进镜像层；运行时用 `env_file` 挂载） |

---

## API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查与 LLM 是否启用 |
| `GET` | `/debug/config` | 配置诊断（无密钥） |
| `POST` | `/auth/register` | 注册并设置 Cookie |
| `POST` | `/auth/login` | 登录 |
| `POST` | `/auth/logout` | 登出 |
| `GET` | `/auth/me` | 当前用户（需登录） |
| `POST` | `/chat` | 发送消息；Body: `{ "user_id": "<会话ID>", "message": "..." }`；返回意图、置信度、`tool_results`、`follow_ups`、RAG 元数据等 |
| `GET` | `/history/{user_id}` | 该会话历史（需登录，且会话属于当前用户作用域） |
| `DELETE` | `/history/{user_id}` | 清空该会话历史 |

---

## 响应为何能较快

- **BM25 本地检索**，无需远程向量库。
- **规则优先**：常见意图不调用大模型。
- **LLM 按需**：仅在规划或生成需要时调用。
- **工具本地化或 HTTP**：状态与工单多为本地逻辑或可配置的外部 API。

---

## 路线规划触发示例（自然语言）

用户可在对话中尝试类似说法（具体解析规则见 `main.py`）：

- 「从北京到上海怎么走」
- 「南通到北京驾车路线」

流程概览：抽取起终点 → 地理编码 → 路径计算 → 组装 `trip_hints` 等结构化字段 → 自然语言回复；失败时会提示细化地名或检查 API 配置。

---

## 后续可扩展方向

### 本轮已落地（详见「多智能体扩展」章节）

- ✅ 多意图扩展与插拔（`AgentRegistry` + 16 个新子智能体）
- ✅ BM25 之外的向量检索接入点（`rag/HybridRetriever`，缺向量模型时自动退化）
- ✅ 工具层统一化（`tools_infra/` 的 cache / retry / circuit breaker / fallback）
- ✅ 安全层（`safety/` PII 脱敏 + 出站审核 + `GuardrailAgent`）
- ✅ 离线评测框架（`evaluation/` + `EvalAgent`）
- ✅ 单元测试骨架（`tests/`）

### 下一步（需要内部系统或前端改造）

- 接入真实公交 / 地铁 / 路况数据源与工单系统（CRM）。
- 接入 LangSmith / OpenTelemetry 做链路追踪与审计日志。
- 前端与地图样式、无障碍与国际化；语音 / 流式输出（需要 Vue + WebSocket 改造）。
- 把 Nominatim / 高德 / OSRM / Open-Meteo 等外呼迁到 `ToolRunner`，让 cache / retry / fallback 统一生效。

---

## 使用声明

本项目用于学习与演示智慧交通客服场景。调用高德、OSRM、Nominatim 等第三方服务时，请遵守各平台服务条款、配额与合规要求。
