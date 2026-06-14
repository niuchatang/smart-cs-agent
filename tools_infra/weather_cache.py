"""Weather cache with Redis first and in-memory fallback.

Weather data is short lived. This cache keeps the public interface small so
tools can use Redis in deployed environments while tests and local runs keep
working without Redis installed or running.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .cache import TTLCache

try:  # Optional dependency: project can still run without Redis locally.
    import redis  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - depends on optional package presence
    redis = None  # type: ignore[assignment]


class WeatherCache:
    """JSON cache for weather/geocode responses.

    Redis is used when REDIS_URL or REDIS_HOST is configured and the client can
    connect quickly. Otherwise a process-local TTL cache provides the same API.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 600,
        namespace: str = "weather",
        redis_url: str | None = None,
    ) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.namespace = (namespace or "weather").strip(":") or "weather"
        self._memory = TTLCache(max_size=2048, default_ttl=float(self.ttl_seconds))
        self._redis = self._build_redis(redis_url)

    @staticmethod
    def _build_redis(redis_url: str | None) -> Any:
        if redis is None:
            return None
        url = (redis_url or os.getenv("REDIS_URL", "")).strip()
        host = os.getenv("REDIS_HOST", "").strip()
        if not url and not host:
            return None
        try:
            if url:
                client = redis.Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=0.3,
                    socket_timeout=0.5,
                )
            else:
                client = redis.Redis(
                    host=host or "127.0.0.1",
                    port=int(os.getenv("REDIS_PORT", "6379") or 6379),
                    db=int(os.getenv("REDIS_DB", "0") or 0),
                    password=os.getenv("REDIS_PASSWORD", "") or None,
                    decode_responses=True,
                    socket_connect_timeout=0.3,
                    socket_timeout=0.5,
                )
            client.ping()
            return client
        except Exception:
            return None

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get_json(self, key: str) -> Optional[Any]:
        cache_key = self._key(key)
        if self._redis is not None:
            try:
                raw = self._redis.get(cache_key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self._memory.get(cache_key)

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        cache_key = self._key(key)
        ttl = max(1, int(ttl_seconds or self.ttl_seconds))
        if self._redis is not None:
            try:
                self._redis.setex(cache_key, ttl, json.dumps(value, ensure_ascii=False))
                return
            except Exception:
                pass
        self._memory.set(cache_key, value, ttl=ttl)

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"
