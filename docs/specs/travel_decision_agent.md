# Travel Decision Agent

Travel Decision Agent analyzes whether and when a user should travel between
two places by combining route planning, route GIS, weather, AQI, warning, and
traffic-risk signals.

## Workflow

```text
User
-> Intent Agent
-> travel_decision_node
-> TravelDecisionAgent
-> query_travel_decision
-> Route Agent
-> RouteGISService
-> Weather/AQI/Warning queries
-> RiskScorer
-> TravelDecisionFormatter
-> User
```

## LangGraph Nodes

```text
travel_decision_node
-> weather_node
-> llm
-> rules
-> finalize
```

The detailed route/weather/risk chain is executed by `query_travel_decision`
because it reuses existing service tools.

## Cache Strategy

- Route: 30 minutes, key `travel-route:{mode}:{origin}:{destination}`.
- Weather: 10 minutes via `WeatherCache`.
- AQI: 10 minutes via `WeatherCache`.
- GIS: database is permanent; runtime cache is 24 hours for resolved points/names.

## Risk Rules

| Signal | Score |
| --- | ---: |
| 大雨 | +30 |
| 暴雨 | +50 |
| 大风 | +20 |
| 大雾 | +40 |
| AQI > 200 | +20 |
| 能见度 < 500m | +40 |
| 高速拥堵 | +30 |

Risk levels:

- `Low`: 0-29
- `Medium`: 30-69
- `High`: 70-99
- `Critical`: >=100
