"""
工具规格与统一执行器（Tool Spec / Tool Runner）

- `ToolSpec`：声明式描述一个工具：名称、参数 schema、超时、重试次数与可选 fallback；
- `ToolRunner`：把函数调用包一层超时/重试/熔断/缓存；
- 本模块**不**自动接管 `CustomerServiceAgent._execute_actions`；
  主智能体可在需要加硬化的工具（如 Nominatim / OSRM）上显式替换调用。

示例：
```python
from tools_infra.registry import ToolSpec, ToolRunner
from tools_infra.cache import TTLCache

geo_cache = TTLCache(max_size=1024, default_ttl=1800)

def _geocode(address: str) -> dict: ...

GEOCODE = ToolSpec(
    name="geocode",
    call=_geocode,
    params_schema={"address": {"type": "str", "required": True}},
    timeout=8,
    max_retry=2,
    cache=geo_cache,
    cache_key=lambda address: ("geocode", address.strip().lower()),
)

runner = ToolRunner(default_retry=1)
result = runner.call(GEOCODE, address="南京")
```
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Hashable, Optional

from .cache import TTLCache

log = logging.getLogger("tools_infra")


@dataclass
class ToolSpec:
    name: str
    call: Callable[..., Any]
    params_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeout: float = 10.0
    max_retry: int = 1
    retry_backoff: float = 0.5
    cache: Optional[TTLCache] = None
    cache_key: Optional[Callable[..., Hashable]] = None
    fallback: Optional[Callable[..., Any]] = None

    def validate(self, params: Dict[str, Any]) -> None:
        for key, meta in self.params_schema.items():
            if meta.get("required") and (key not in params or params[key] in (None, "")):
                raise ValueError(f"tool[{self.name}] missing required param: {key}")


class _Circuit:
    def __init__(self, fail_threshold: int = 5, reset_after: float = 60.0) -> None:
        self._fail = 0
        self._opened_at = 0.0
        self._fail_threshold = fail_threshold
        self._reset_after = reset_after
        self._lock = RLock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self._fail < self._fail_threshold:
                return False
            if time.time() - self._opened_at > self._reset_after:
                self._fail = 0
                self._opened_at = 0.0
                return False
            return True

    def on_success(self) -> None:
        with self._lock:
            self._fail = 0
            self._opened_at = 0.0

    def on_failure(self) -> None:
        with self._lock:
            self._fail += 1
            if self._fail == self._fail_threshold:
                self._opened_at = time.time()


class ToolRunner:
    def __init__(self, default_retry: int = 1) -> None:
        self._default_retry = default_retry
        self._circuits: Dict[str, _Circuit] = {}

    def call(self, spec: ToolSpec, **params: Any) -> Any:
        spec.validate(params)

        if spec.cache is not None and spec.cache_key is not None:
            try:
                key = spec.cache_key(**params)
            except Exception:
                key = None
            if key is not None:
                hit = spec.cache.get(key)
                if hit is not None:
                    return hit

        circuit = self._circuits.setdefault(spec.name, _Circuit())
        if circuit.open:
            if spec.fallback is not None:
                return spec.fallback(**params)
            raise RuntimeError(f"tool[{spec.name}] circuit open")

        retries = max(0, int(spec.max_retry if spec.max_retry is not None else self._default_retry))
        last_exc: Optional[BaseException] = None
        for attempt in range(retries + 1):
            try:
                value = spec.call(**params)
                circuit.on_success()
                if spec.cache is not None and spec.cache_key is not None:
                    try:
                        cache_key = spec.cache_key(**params)
                        spec.cache.set(cache_key, value)
                    except Exception:
                        pass
                return value
            except Exception as exc:
                last_exc = exc
                circuit.on_failure()
                log.warning(
                    "tool[%s] attempt %d/%d failed: %s", spec.name, attempt + 1, retries + 1, exc
                )
                if attempt < retries:
                    time.sleep(spec.retry_backoff * (2 ** attempt))

        if spec.fallback is not None:
            try:
                return spec.fallback(**params)
            except Exception as fb_exc:
                last_exc = fb_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"tool[{spec.name}] unknown failure")
