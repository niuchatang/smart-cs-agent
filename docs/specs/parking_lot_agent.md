# SPEC · ParkingLotAgent（停车场查询智能体）

> **状态**：Draft → Implemented  
> **作者**：（按 spec-driven-development skill 自动生成）  
> **关联 skill**：`spec-driven-development`、`api-and-interface-design`

---

## 1. 目标 Objective

让用户用自然语言查询「停车场 / 车位 / 在哪停车」，复用既有路径规划上下文，给出**沿途**或**目的地附近**的停车建议；不联网查真实 POI（避免占用配额、维持现有"模拟数据 + 真实结构"约定，与 `ServiceAreaAgent` 一致）。

## 2. 非目标 Non-Goals

- ❌ 不接入第三方真实停车场 POI（高德 / 百度 API）—— 用 mock，但保留可替换的接口形状
- ❌ 不做停车费支付 / 预订
- ❌ 不做车牌号解析
- ❌ 不替代 RoutePlanningAgent 做完整路线规划

## 3. 用户故事 User Stories

| ID | 故事 | 验收 |
|---|---|---|
| US-1 | 用户问「南京南站附近有什么停车场」→ 给附近停车场建议 | reply 含「南京南站」+ 至少 1 条停车选项 |
| US-2 | 用户先做了「成都到重庆」路线规划，再问「沿途哪里能停车」→ 基于历史路线给沿途停车建议 | reply 含成都、重庆等沿路城市 + 停车信息 |
| US-3 | 用户单独问「停车」无任何上下文 → 引导补充地点 | reply 询问"请告诉我地点或起终点" |
| US-4 | 用户问「电动车停车带充电的」→ 优先推荐带充电桩的停车场 | reply 含"带充电"标识 |
| US-5 | 用户问"今天天气" → 不应触发本 Agent | `try_plan` 返回 None |

## 4. 接口契约 Contract（按 api-and-interface-design）

### 4.1 类签名
```python
class ParkingLotAgent:
    name: str = "parking_lot"
    priority: int = 32      # 紧随 ServiceArea(30) 之后

    def __init__(self, service_agent: Any) -> None: ...

    def try_plan(
        self,
        message: str,
        history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]: ...
```

### 4.2 输入 Input
- `message`: 用户最新消息文本（必须）
- `history`: 历史对话列表（可空），元素结构同既有 Agent

### 4.3 输出 Output（命中场景）

```python
{
    "intent": "route_planning",   # 复用既有 IntentType Literal（参考 ServiceAreaAgent）
    "confidence": float,          # 0.70~0.86，按上下文丰富度浮动
    "actions": [],                # 不触发额外工具
    "llm_reply": str,             # 给用户的自然语言回复
    "used_llm": False,            # 纯规则
}
```

返回 `None` 表示本 Agent 不接管，由 orchestrator 继续往下试。

### 4.4 触发词 Trigger Keywords
```python
("停车", "停车场", "车位", "泊车", "停哪", "停在", "停车位", "停个车")
```

### 4.5 场景修饰词 Modifiers
- 充电类：`("充电", "充电桩", "新能源", "电动车")` → reply 加充电桩优先
- 室内/地下：`("地下", "室内", "地库")` → reply 加遮蔽建议
- 免费：`("免费", "不收费")` → reply 加免费场推荐
- 沿途意图词：`("沿途", "沿路", "沿线", "路上", "途中", "一路上", "一路", "路途")` → 强制走沿途分段（即使消息里也提到了某个具体地名）

### 4.6 决策优先级 Decision Priority（v2，修复 v1 历史路线劫持 bug）

输入命中触发词后，按以下顺序选择回复模式：

| 顺序 | 条件 | 模式 |
|---|---|---|
| 1 | `want_along=True` 且 历史有 cities | 沿途分段 `_build_along_route_plan` |
| 2 | 当前消息抽到具体地点 `place` | **单点查询 `_build_single_place_plan`（无视历史 cities）** |
| 3 | 仅有历史 cities，无具体地点 | 沿途分段（隐式继续聊路线） |
| 4 | 都没有 | 引导补充 `_build_clarify_plan` |

**关键不变量**：消息里抽到 `place` 时，**当前消息优先级高于历史上下文**。这是 v1 的一个已知 bug：用户先做"北京→南京"路径规划，再问"南京南站附近的停车场"，v1 会因为历史有 cities 而错误地渲染北京/廊坊/沧州的沿途分段，把"南京南站"完全忽略。v2 通过加入"显式沿途意图词" + "place 优先于 cities" 修复。

## 5. 不变量 Invariants

1. **空消息 / 无触发词必返 None**——和现有所有 ExtensionAgent 一致
2. **`intent` 必须 ∈ main.IntentType Literal**——这里固定 `"route_planning"`
3. **`actions` 必为空列表**——本 Agent 不调用工具
4. **`used_llm: False`** 必须显式标
5. **不修改 `service_agent` 状态**——只读它的 `_extract_last_route_cities_from_history` 等辅助方法
6. **异常静默吞掉**：被 `AgentRegistry.try_plan` 包了 try-except，本 Agent 实现可正常 raise，外层会兜

## 6. 错误语义 Error Semantics
- 输入异常（如 history 不是 list）→ 返回 None，**不抛异常**
- service_agent 缺少辅助方法 → 走 fallback 路径（仅基于 message 给通用建议）

## 7. 测试矩阵 Test Matrix

| ID | 场景 | 输入 message | 输入 history | 期望 |
|---|---|---|---|---|
| T1 | 单点停车查询 | "南京南站附近有什么停车场" | [] | 命中，reply 含「南京南站」 |
| T2 | 沿路停车（基于历史路线）| "沿途哪里能停车" | 含 `cities_along_route=["成都","重庆"]` 的工具结果 | 命中，reply 含成都+重庆 |
| T3 | 充电类停车 | "停车带充电的" | T2 的 history | 命中，reply 含「充电」 |
| T4 | 触发词但无地点上下文 | "停车" | [] | 命中（confidence 偏低），reply 提示"请告诉我地点" |
| T5 | 无触发词 | "今天北京天气" | [] | 返回 None |
| T6 | 异常 history | "哪里停车" | None | 不抛异常，按 T4 fallback 处理 |
| T7 | Orchestrator 集成 | "南京南站停车" | [] | 走 `IntentOrchestratorAgent.plan_rules`，命中 ext_agent=`parking_lot` |
| T8 | 历史劫持回归（v2） | "南京南站附近的停车场" | 含 `cities=[北京...南京]` | 走单点，reply 含"南京南站"，**不含**北京/廊坊 |
| T9 | "沿途" 关键字优先 | "沿途有重庆的停车场吗" | 含 `cities=[成都,内江,重庆]` | 走沿途，reply 含成都+内江+重庆 |

## 8. 样例输出 Sample Outputs

### Case T1
```text
关于「南京南站」附近的停车场建议（示意，真实停车信息请以导航 App 为准）：
- 南京南站·北广场地下停车场（综合型，24h 开放，付费）
- 南京南站·南广场地面停车场（含新能源充电桩）
- 南京南站·临时落客区（短停 15 分钟内免费）

如需更精准信息（具体收费、是否需预约），可告诉我「室内/地下/免费」等偏好，我会进一步过滤。
```

### Case T2
```text
沿你最近一次路线（成都 → 重庆），给一份分段停车建议（示意，供规划参考）：
- 成都 — 综合型停车场（含新能源充电）
- 内江 — 高速服务区附近的临时停车区
- 重庆 — 综合型停车场（地下，24h 开放）

如想找特定城市/服务区的停车场，告诉我编号即可。
```

### Case T4
```text
要查停车场，请先告诉我地点（例如「南京南站附近停车场」），
或先做一次路径规划后再问"沿途停车"，我可以基于路线给分段建议。
```

## 9. Out of Scope（可在后续 PR 引入）
- 接入真实 POI（替换 `_mock_parking_hint` 为 tool 调用）
- 收费区间筛选
- 残疾人停车位筛选（可与 `AccessibilityAgent` 联动）
- 多车型尺寸（小型车 / SUV / 卡车）匹配

## 10. 验收 Verification

实现完成后必须满足：
- [ ] 7 条测试全通
- [ ] `pytest tests/` 整体回归无新增失败
- [ ] orchestrator 注册表里出现 `parking_lot`
- [ ] `_StubService` 测试不依赖真实 service_agent
- [ ] commit message 引用本 SPEC 路径
