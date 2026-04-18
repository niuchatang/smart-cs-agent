"""
简易 TTL + LRU 缓存（线程安全）

- 进程内内存缓存，无外部依赖；
- 适合给 Nominatim 地理编码、OSRM 路径、同城短时天气等高频只读请求减压；
- 使用：
  ```python
  from tools_infra.cache import TTLCache, cached
  geo_cache = TTLCache(max_size=2048, default_ttl=1800)

  @cached(geo_cache, key_fn=lambda q: ("geocode", q.strip().lower()))
  def geocode(q: str): ...
  ```
"""

from __future__ import annotations

import time
from collections import OrderedDict
from functools import wraps
from threading import RLock
from typing import Any, Callable, Hashable, Optional, Tuple, TypeVar

T = TypeVar("T")


class TTLCache:
    def __init__(self, max_size: int = 1024, default_ttl: float = 600.0) -> None:
        self._max = max(16, int(max_size))
        self._ttl = max(1.0, float(default_ttl))
        self._store: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: Hashable, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            expires = time.time() + (self._ttl if ttl is None else float(ttl))
            self._store[key] = (expires, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def cached(
    cache: TTLCache,
    key_fn: Callable[..., Hashable],
    ttl: Optional[float] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                key = key_fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit  # type: ignore[return-value]
            value = fn(*args, **kwargs)
            try:
                cache.set(key, value, ttl=ttl)
            except Exception:
                pass
            return value

        return wrapper

    return deco
