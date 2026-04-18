"""
`evaluation.eval` 入口（与 `intent.eval_agent.main` 等价），便于在 CI/脚手架中统一调用：

```bash
python -m evaluation.eval --cases evaluation/intent_cases.jsonl
```
"""

from __future__ import annotations

from intent.eval_agent import main

if __name__ == "__main__":
    raise SystemExit(main())
