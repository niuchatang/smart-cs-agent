"""
工具基础设施（Tool Infrastructure）

- `cache.py`：基于 (name, args) 的 TTL + LRU 内存缓存；
- `registry.py`：`ToolSpec`（schema/timeout/retry/fallback）与 `ToolRunner`；
- 不替换 main.py 既有 `_execute_actions` 逻辑，作为可选的 helper 层，
  在需要包一层鲁棒性的场景显式使用（例如地理编码、OSRM）。
"""

from .cache import TTLCache, cached
from .registry import ToolRunner, ToolSpec

__all__ = ["TTLCache", "cached", "ToolSpec", "ToolRunner"]
