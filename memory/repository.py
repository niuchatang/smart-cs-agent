from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymysql  # type: ignore[import-untyped]
from dotenv import load_dotenv
from pymysql.cursors import DictCursor  # type: ignore[import-untyped]

from .models import MemoryEventItem, MemoryProfileItem

load_dotenv()


class MemoryRepository:
    """MySQL 记忆仓储；不可用时降级 JSON 文件。"""

    def __init__(self, json_fallback: str = "data/memory_store.json") -> None:
        self.host = os.getenv("MYSQL_HOST", "127.0.0.1")
        self.port = int(os.getenv("MYSQL_PORT", "3306") or 3306)
        self.user = os.getenv("MYSQL_USER", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "")
        self.database = os.getenv("MYSQL_DATABASE", "smart_cs_agent")
        self.unix_socket = os.getenv("MYSQL_UNIX_SOCKET", "").strip()
        self.mysql_enabled = os.getenv("MEMORY_ENABLE_MYSQL", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._json_path = Path(json_fallback)
        self._json_cache: Dict[str, Any] = {}
        self._schema_ready = False
        self._load_json()

    def ensure_schema(self) -> None:
        if self._schema_ready or not self.mysql_enabled:
            return
        schema_path = Path(__file__).resolve().parents[1] / "database" / "memory_schema.sql"
        if not schema_path.exists():
            self._schema_ready = True
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            sql = schema_path.read_text(encoding="utf-8")
            statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
            conn.commit()
            self._schema_ready = True
        except Exception:
            pass
        finally:
            conn.close()

    def get_profiles(self, user_id: str) -> Dict[str, str]:
        if not user_id:
            return {}
        self.ensure_schema()
        conn = self._connect()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT `key`, value FROM memory_profile WHERE user_id=%s",
                        (user_id,),
                    )
                    rows = cur.fetchall() or []
                return {str(r["key"]): str(r["value"]) for r in rows}
            except Exception:
                pass
            finally:
                conn.close()
        bucket = self._json_cache.setdefault("profiles", {}).setdefault(user_id, {})
        return dict(bucket)

    def upsert_profile(self, user_id: str, key: str, value: str) -> None:
        if not user_id or not key:
            return
        self.ensure_schema()
        conn = self._connect()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_profile (user_id, `key`, value)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE value=VALUES(value), updated_at=CURRENT_TIMESTAMP
                        """,
                        (user_id, key, value),
                    )
                conn.commit()
                return
            except Exception:
                pass
            finally:
                conn.close()
        bucket = self._json_cache.setdefault("profiles", {}).setdefault(user_id, {})
        bucket[key] = value
        self._flush_json()

    def delete_profile(self, user_id: str, key: str) -> bool:
        if not user_id or not key:
            return False
        conn = self._connect()
        ok = False
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_profile WHERE user_id=%s AND `key`=%s",
                        (user_id, key),
                    )
                conn.commit()
                ok = cur.rowcount > 0
            except Exception:
                ok = False
            finally:
                conn.close()
        bucket = self._json_cache.setdefault("profiles", {}).get(user_id, {})
        if key in bucket:
            del bucket[key]
            self._flush_json()
            ok = True
        return ok

    def search_profiles(self, user_id: str, query: str) -> List[MemoryProfileItem]:
        profiles = self.get_profiles(user_id)
        q = (query or "").strip().lower()
        out: List[MemoryProfileItem] = []
        for k, v in profiles.items():
            if not q or q in k.lower() or q in v.lower():
                out.append(MemoryProfileItem(key=k, value=v))
        return out

    def add_event(self, user_id: str, event_type: str, content: Dict[str, Any]) -> None:
        if not user_id:
            return
        self.ensure_schema()
        payload = json.dumps(content, ensure_ascii=False)
        conn = self._connect()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_event (user_id, event_type, content) VALUES (%s, %s, %s)",
                        (user_id, event_type, payload),
                    )
                conn.commit()
                return
            except Exception:
                pass
            finally:
                conn.close()
        events = self._json_cache.setdefault("events", {}).setdefault(user_id, [])
        events.append({"event_type": event_type, "content": content})
        events[:] = events[-200:]
        self._flush_json()

    def list_events(self, user_id: str, limit: int = 20, event_type: str = "") -> List[MemoryEventItem]:
        self.ensure_schema()
        conn = self._connect()
        rows: List[Dict[str, Any]] = []
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    if event_type:
                        cur.execute(
                            """
                            SELECT event_type, content, created_at
                            FROM memory_event
                            WHERE user_id=%s AND event_type=%s
                            ORDER BY id DESC LIMIT %s
                            """,
                            (user_id, event_type, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT event_type, content, created_at
                            FROM memory_event
                            WHERE user_id=%s
                            ORDER BY id DESC LIMIT %s
                            """,
                            (user_id, limit),
                        )
                    rows = cur.fetchall() or []
            except Exception:
                rows = []
            finally:
                conn.close()
        if rows:
            out: List[MemoryEventItem] = []
            for r in rows:
                content = r.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"raw": content}
                if not isinstance(content, dict):
                    content = {}
                out.append(
                    MemoryEventItem(
                        event_type=str(r.get("event_type") or ""),
                        content=content,
                        created_at=r.get("created_at"),
                    )
                )
            return out
        events = self._json_cache.setdefault("events", {}).get(user_id, [])
        out = []
        for e in events[-limit:][::-1]:
            if event_type and e.get("event_type") != event_type:
                continue
            out.append(MemoryEventItem(event_type=e.get("event_type", ""), content=e.get("content", {})))
        return out

    def add_summary(self, user_id: str, summary: str) -> None:
        if not user_id or not summary.strip():
            return
        self.ensure_schema()
        conn = self._connect()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_summary (user_id, summary) VALUES (%s, %s)",
                        (user_id, summary.strip()),
                    )
                conn.commit()
                return
            except Exception:
                pass
            finally:
                conn.close()
        sums = self._json_cache.setdefault("summaries", {}).setdefault(user_id, [])
        sums.append(summary.strip())
        sums[:] = sums[-20:]
        self._flush_json()

    def latest_summary(self, user_id: str) -> str:
        conn = self._connect()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT summary FROM memory_summary WHERE user_id=%s ORDER BY id DESC LIMIT 1",
                        (user_id,),
                    )
                    row = cur.fetchone()
                if row:
                    return str(row.get("summary") or "")
            except Exception:
                pass
            finally:
                conn.close()
        sums = self._json_cache.setdefault("summaries", {}).get(user_id, [])
        return sums[-1] if sums else ""

    def _connect(self) -> Optional[Any]:
        if not self.mysql_enabled:
            return None
        kwargs: Dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
        }
        if self.unix_socket and Path(self.unix_socket).exists():
            kwargs.pop("host", None)
            kwargs.pop("port", None)
            kwargs["unix_socket"] = self.unix_socket
        try:
            return pymysql.connect(**kwargs)
        except Exception:
            return None

    def _load_json(self) -> None:
        if not self._json_path.exists():
            return
        try:
            self._json_cache = json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            self._json_cache = {}

    def _flush_json(self) -> None:
        try:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_path.write_text(
                json.dumps(self._json_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
