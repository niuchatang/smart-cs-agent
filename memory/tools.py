from __future__ import annotations

from typing import Any, Dict, List

from .repository import MemoryRepository


class MemorySearchTool:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def run(self, user_id: str, query: str = "") -> Dict[str, Any]:
        profiles = self._repo.search_profiles(user_id, query)
        events = self._repo.list_events(user_id, limit=10)
        return {
            "tool": "memory_search",
            "success": True,
            "profiles": [p.model_dump() for p in profiles],
            "events": [e.model_dump() for e in events],
            "summary": self._repo.latest_summary(user_id),
        }


class MemorySaveTool:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def run(self, user_id: str, key: str, value: str, event_type: str = "", event_content: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if key:
            self._repo.upsert_profile(user_id, key, value)
        if event_type:
            self._repo.add_event(user_id, event_type, event_content or {})
        return {"tool": "memory_save", "success": True, "key": key, "event_type": event_type}


class MemoryUpdateTool:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def run(self, user_id: str, key: str, value: str) -> Dict[str, Any]:
        self._repo.upsert_profile(user_id, key, value)
        return {"tool": "memory_update", "success": True, "key": key}


class MemoryDeleteTool:
    def __init__(self, repo: MemoryRepository) -> None:
        self._repo = repo

    def run(self, user_id: str, key: str) -> Dict[str, Any]:
        ok = self._repo.delete_profile(user_id, key)
        return {"tool": "memory_delete", "success": ok, "key": key}


class MemoryToolkit:
    def __init__(self, repo: MemoryRepository | None = None) -> None:
        self.repo = repo or MemoryRepository()
        self.search = MemorySearchTool(self.repo)
        self.save = MemorySaveTool(self.repo)
        self.update = MemoryUpdateTool(self.repo)
        self.delete = MemoryDeleteTool(self.repo)
