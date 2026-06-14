from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryProfileItem(BaseModel):
    key: str
    value: str
    updated_at: Optional[datetime] = None


class MemoryEventItem(BaseModel):
    event_type: str
    content: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class MemoryContext(BaseModel):
    user_id: str = ""
    profiles: Dict[str, str] = Field(default_factory=dict)
    recent_events: List[MemoryEventItem] = Field(default_factory=list)
    summary: str = ""
    proactive_hint: str = ""

    def profile_get(self, key: str, default: str = "") -> str:
        return self.profiles.get(key, default)
