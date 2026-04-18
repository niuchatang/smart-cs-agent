"""
离线评估智能体（Eval Agent）

- 读取 `evaluation/intent_cases.jsonl`（每行 `{user, expected_intent, must_call_tool?}`）；
- 对每条用例把 `user` 喂给 `UserIntentAgent.parse()`，比对 intent 与工具调用；
- 输出汇总 {pass, fail, fail_cases[]}；
- 可选：若主智能体启用了 LLM，用 LLM 复核「reply 是否明显偏题」。

命令行用法：
```bash
python -m intent.eval_agent --cases evaluation/intent_cases.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            print(f"[warn] line {i}: invalid json ({exc})", file=sys.stderr)
    return out


class EvalAgent:
    name = "eval"

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def evaluate(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        passed = 0
        fails: List[Dict[str, Any]] = []
        for case in cases:
            user = str(case.get("user", "")).strip()
            expected_intent = str(case.get("expected_intent", "")).strip()
            must_call_tool = case.get("must_call_tool")
            if not user or not expected_intent:
                continue
            plan = self._parse(user)
            got_intent = str(plan.get("intent", ""))
            tools_used = {a.get("tool") for a in (plan.get("actions") or []) if isinstance(a, dict)}

            ok_intent = got_intent == expected_intent
            ok_tool = True
            if isinstance(must_call_tool, str) and must_call_tool:
                ok_tool = must_call_tool in tools_used
            if ok_intent and ok_tool:
                passed += 1
            else:
                fails.append(
                    {
                        "user": user,
                        "expected_intent": expected_intent,
                        "got_intent": got_intent,
                        "must_call_tool": must_call_tool,
                        "tools_used": sorted(t for t in tools_used if t),
                    }
                )
        return {"total": len(cases), "pass": passed, "fail": len(fails), "fails": fails}

    def _parse(self, message: str) -> Dict[str, Any]:
        agent = getattr(self._svc, "intent_agent", None)
        if agent is None:
            return {"intent": "unknown", "actions": []}
        try:
            return agent.parse(message, [], [])
        except Exception:
            return {"intent": "unknown", "actions": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/intent_cases.jsonl")
    args = parser.parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"cases file not found: {cases_path}", file=sys.stderr)
        return 2
    cases = _load_cases(cases_path)

    try:
        from main import CustomerServiceAgent  # type: ignore
        svc = CustomerServiceAgent()
    except Exception as exc:
        print(f"failed to instantiate CustomerServiceAgent: {exc}", file=sys.stderr)
        return 3
    agent = EvalAgent(svc)
    result = agent.evaluate(cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
