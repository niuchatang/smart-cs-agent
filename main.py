from __future__ import annotations

import json
import math
import os
import re
import secrets
import time
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

import pymysql  # type: ignore[reportMissingImports]
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pymysql.cursors import DictCursor  # type: ignore[reportMissingImports]
from pydantic import BaseModel, Field

load_dotenv()

from intent.user_intent_agent import UserIntentAgent

IntentType = Literal[
    "route_planning",
    "realtime_status",
    "highway_condition",
    "weather_query",
    "fare_policy",
    "ticket_refund",
    "lost_and_found",
    "complaint",
    "human_handoff",
    "unknown",
]


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., description="User message")


class AgentAction(BaseModel):
    tool: str
    params: Dict[str, Any]


class FollowUpItem(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)
    answer: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    reply: str
    actions: List[AgentAction]
    tool_results: List[Dict[str, Any]]
    used_llm: bool
    rag_used: bool
    rag_hits: List[Dict[str, Any]] = Field(default_factory=list)
    follow_ups: List[FollowUpItem] = Field(default_factory=list)


class HistoryItem(BaseModel):
    role: Literal["user", "agent"]
    content: str
    timestamp: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    user_id: str
    items: List[HistoryItem]


class LoginRequest(BaseModel):
    """登录：密码至少 6 位（与历史账号兼容）。"""

    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_\-]+$")
    password: str = Field(..., min_length=6, max_length=64)


class RegisterRequest(BaseModel):
    """注册：密码至少 8 位，与 AuthStore._validate_register_password 一致，避免仅前端校验导致 422 含义不清。"""

    username: str = Field(..., min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_\-]+$")
    password: str = Field(..., min_length=8, max_length=64)


class AuthUser(BaseModel):
    user_id: int
    username: str


class AuthResponse(BaseModel):
    ok: bool
    user: AuthUser


class AuthStore:
    def __init__(self) -> None:
        self.host = os.getenv("MYSQL_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
        self.user = os.getenv("MYSQL_USER", "root").strip() or "root"
        self.password = os.getenv("MYSQL_PASSWORD", "").strip()
        self.database = os.getenv("MYSQL_DATABASE", "smart_cs_agent").strip() or "smart_cs_agent"
        self.unix_socket = self._resolve_unix_socket()
        self.enabled = True
        self.init_error = ""
        try:
            self._ensure_schema()
        except Exception as exc:
            self.enabled = False
            self.init_error = str(exc)
            print(f"[auth] MySQL init failed: {self.init_error}")

    def register(self, username: str, password: str) -> Dict[str, Any]:
        self._assert_enabled()
        uname = self._normalize_username(username)
        self._validate_register_password(password)
        salt = secrets.token_hex(16)
        pwd_hash = self._hash_password(password, salt)
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username=%s", (uname,))
                if cur.fetchone():
                    raise ValueError("用户名已存在")
                cur.execute(
                    "INSERT INTO users(username, password_hash, password_salt) VALUES(%s, %s, %s)",
                    (uname, pwd_hash, salt),
                )
                conn.commit()
                user_id = int(cur.lastrowid)
        return {"id": user_id, "username": uname}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        self._assert_enabled()
        uname = self._normalize_username(username)
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, password_hash, password_salt FROM users WHERE username=%s",
                    (uname,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("用户名或密码错误")
                expected = self._hash_password(password, str(row["password_salt"]))
                if not hmac.compare_digest(expected, str(row["password_hash"])):
                    raise ValueError("用户名或密码错误")
                return {"id": int(row["id"]), "username": str(row["username"])}

    def create_session(self, user_id: int) -> tuple[str, datetime]:
        self._assert_enabled()
        token = secrets.token_urlsafe(32)
        # 使用 UTC 写入 DATETIME，与查询中 UTC_TIMESTAMP() 一致，避免本机/容器与 MySQL 时区不一致导致「刚登录就失效」
        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=7)
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO user_sessions(user_id, token, expires_at) VALUES(%s, %s, %s)", (user_id, token, expires_at))
                conn.commit()
        return token, expires_at

    def resolve_user_by_token(self, token: str) -> Dict[str, Any] | None:
        self._assert_enabled()
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT u.id AS user_id, u.username AS username
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.token=%s AND s.expires_at > UTC_TIMESTAMP()
                    """,
                    (token,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {"id": int(row["user_id"]), "username": str(row["username"])}

    def delete_session(self, token: str) -> None:
        if not token or not self.enabled:
            return
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_sessions WHERE token=%s", (token,))
                conn.commit()

    def _ensure_schema(self) -> None:
        with self._connect_server() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
        with self._connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                      id BIGINT PRIMARY KEY AUTO_INCREMENT,
                      username VARCHAR(32) NOT NULL UNIQUE,
                      password_hash CHAR(64) NOT NULL,
                      password_salt CHAR(32) NOT NULL,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_sessions (
                      id BIGINT PRIMARY KEY AUTO_INCREMENT,
                      user_id BIGINT NOT NULL,
                      token VARCHAR(128) NOT NULL UNIQUE,
                      expires_at DATETIME NOT NULL,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      INDEX idx_user_sessions_token(token),
                      INDEX idx_user_sessions_expires(expires_at),
                      CONSTRAINT fk_user_sessions_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute("DELETE FROM user_sessions WHERE expires_at <= UTC_TIMESTAMP()")
                conn.commit()

    def _connect_server(self) -> Any:
        kwargs = self._base_connect_kwargs()
        return pymysql.connect(**kwargs)

    def _connect_db(self) -> Any:
        kwargs = self._base_connect_kwargs()
        kwargs["database"] = self.database
        return pymysql.connect(**kwargs)

    def _assert_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("登录服务暂不可用，请检查本地MySQL配置")

    @staticmethod
    def _normalize_username(username: str) -> str:
        value = username.strip().lower()
        if not re.match(r"^[a-zA-Z0-9_\-]{3,32}$", value):
            raise ValueError("用户名仅支持3-32位字母、数字、下划线、中划线")
        return value

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
        return digest.hex()

    @staticmethod
    def _validate_register_password(password: str) -> None:
        if len(password) < 8:
            raise ValueError("注册密码长度不能少于8位")
        if not re.search(r"[A-Z]", password):
            raise ValueError("注册密码需包含至少1个大写字母")
        if not re.search(r"[a-z]", password):
            raise ValueError("注册密码需包含至少1个小写字母")
        if not re.search(r"\d", password):
            raise ValueError("注册密码需包含至少1个数字")

    def _resolve_unix_socket(self) -> str:
        configured = os.getenv("MYSQL_UNIX_SOCKET", "").strip()
        if configured:
            return configured
        for path in ["/tmp/mysql.sock", "/var/mysql/mysql.sock"]:
            if Path(path).exists():
                return path
        return ""

    def _base_connect_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "user": self.user,
            "password": self.password,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
        }
        # Prefer unix socket for local official MySQL installs on macOS.
        if self.unix_socket and self.host in {"127.0.0.1", "localhost"}:
            kwargs["unix_socket"] = self.unix_socket
        else:
            kwargs["host"] = self.host
            kwargs["port"] = self.port
        return kwargs


class ConversationStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("{}", encoding="utf-8")

    def get(self, user_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
        data = self._load()
        items = data.get(user_id, [])
        if limit is None:
            return list(items)
        return list(items[-limit:])

    def append(self, user_id: str, role: str, content: str, meta: Dict[str, Any] | None = None) -> None:
        data = self._load()
        items = data.setdefault(user_id, [])
        items.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "meta": meta or {},
            }
        )
        data[user_id] = items[-100:]
        self._save(data)

    def clear(self, user_id: str) -> None:
        data = self._load()
        data[user_id] = []
        self._save(data)

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            return json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class SimpleRAGStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")
        self.docs = self._load_docs()
        self.retriever = self._build_retriever(self.docs)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        docs = self.retriever.invoke(query)[:k]
        hits: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            hits.append(
                {
                    "id": str(doc.metadata.get("id", "")),
                    "title": str(doc.metadata.get("title", "")),
                    "content": doc.page_content,
                    "score": round(1.0 - (idx - 1) * 0.1, 3),
                }
            )
        return hits

    def _load_docs(self) -> List[Dict[str, Any]]:
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    @staticmethod
    def _build_retriever(raw_docs: List[Dict[str, Any]]) -> BM25Retriever:
        docs: List[Document] = []
        for it in raw_docs:
            docs.append(
                Document(
                    page_content=str(it.get("content", "")),
                    metadata={"id": str(it.get("id", "")), "title": str(it.get("title", ""))},
                )
            )
        if not docs:
            docs.append(Document(page_content="暂无知识库内容。", metadata={"id": "empty", "title": "empty"}))
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 3
        return retriever


class CustomerServiceAgent:
    # 服务区：距起点 150km、距终点 100km 内不推荐；中间段按约 250km（200～300）间隔抽样。
    SERVICE_AREA_EXCLUDE_START_KM = 150.0
    SERVICE_AREA_EXCLUDE_END_KM = 100.0
    SERVICE_AREA_MIDDLE_SPACING_KM = 250.0
    # 长途全程（含多段合并）最多推荐的服务区数量；在走廊内尽量均匀分布
    SERVICE_AREA_MAX_RECOMMEND = 6

    def __init__(self) -> None:
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()
        self.model = self._resolve_model()
        self.amap_api_key = os.getenv("AMAP_API_KEY", "").strip()
        self.amap_bypass_proxy = os.getenv("AMAP_BYPASS_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.transit_status = {
            "地铁2号线": {"status": "正常", "next_train_min": 3, "notice": "高峰期行车间隔约3-4分钟"},
            "地铁4号线": {"status": "轻微延误", "next_train_min": 8, "notice": "设备检修导致部分区段降速"},
            "S1高架": {"status": "拥堵", "next_train_min": None, "notice": "建议绕行东三环"},
            "机场快线": {"status": "正常", "next_train_min": 6, "notice": "当前运行平稳"},
        }
        self.highway_conditions = {
            "G2": {
                "name": "京沪高速",
                "congestion_level": "中度拥堵",
                "has_accident": True,
                "incident": "K1062 附近发生两车追尾，已占用应急车道",
                "controls": ["事故路段限速 60km/h", "建议提前从临近出口绕行"],
                "advice": "建议避开高峰通过事故区间，或改走 G25 联络线",
            },
            "G15": {
                "name": "沈海高速",
                "congestion_level": "基本畅通",
                "has_accident": False,
                "incident": "暂无事故上报",
                "controls": ["部分路段夜间施工，间歇性单车道通行"],
                "advice": "夜间通行请关注可变情报板提示",
            },
            "G25": {
                "name": "长深高速",
                "congestion_level": "轻度缓行",
                "has_accident": False,
                "incident": "暂无事故上报",
                "controls": ["节假日前后可能分时段限流"],
                "advice": "可作为 G2 拥堵时的替代走廊",
            },
            "G30": {
                "name": "连霍高速",
                "congestion_level": "通行基本正常",
                "has_accident": False,
                "incident": "暂无事故上报",
                "controls": ["部分区段存在养护作业，建议按现场标志减速慢行"],
                "advice": "长距离通行建议提前规划补给点与休息区",
            },
            "G40": {
                "name": "沪陕高速",
                "congestion_level": "基本畅通",
                "has_accident": False,
                "incident": "暂无事故上报",
                "controls": ["局部路段车流波动较大，建议保持安全车距"],
                "advice": "跨江通道高峰波动明显，建议错峰通过关键枢纽",
            },
            "S1高架": {
                "name": "S1高架",
                "congestion_level": "重度拥堵",
                "has_accident": True,
                "incident": "主线进城方向多车剐蹭，排队约 4km",
                "controls": ["高峰时段入口匝道控流", "建议错峰或改道"],
                "advice": "建议改走地面快速路或地铁通勤",
            },
        }
        self.rag_store = SimpleRAGStore(Path(__file__).parent / "data" / "knowledge_base.json")
        self.llm = self._build_llm()
        self.answer_chain = self._build_answer_chain()
        self.intent_agent = UserIntentAgent(self)

    @staticmethod
    def _resolve_api_key() -> str:
        # 默认优先 DeepSeek；仍支持 OPENAI_* 以便接入其它 OpenAI 兼容端点。
        return (
            os.getenv("DEEPSEEK_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )

    @staticmethod
    def _resolve_base_url() -> str:
        deepseek_base = os.getenv("DEEPSEEK_BASE_URL", "").strip().rstrip("/")
        if deepseek_base:
            if deepseek_base.endswith("/v1"):
                return deepseek_base
            return f"{deepseek_base}/v1"
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip()
        if openai_base:
            return openai_base.rstrip("/")
        # 未写 URL 时：谁配了 Key 就默认谁的官方兼容端点（避免只配 OPENAI_KEY 却指向 DeepSeek）
        if os.getenv("DEEPSEEK_API_KEY", "").strip():
            return "https://api.deepseek.com/v1"
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "https://api.openai.com/v1"
        return "https://api.deepseek.com/v1"

    @staticmethod
    def _resolve_model() -> str:
        explicit = (
            os.getenv("DEEPSEEK_MODEL_ID", "").strip()
            or os.getenv("OPENAI_MODEL", "").strip()
        )
        if explicit:
            return explicit
        if os.getenv("DEEPSEEK_API_KEY", "").strip():
            return "deepseek-chat"
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "gpt-4o-mini"
        return "deepseek-chat"

    @property
    def llm_enabled(self) -> bool:
        return self.llm is not None

    def chat(self, message: str, history: List[Dict[str, Any]] | None = None) -> ChatResponse:
        rag_hits = self.rag_store.retrieve(message, k=3)
        plan = self.intent_agent.parse(message, history or [], rag_hits)
        tool_results = self._execute_actions(plan["actions"])
        if plan.get("od_traffic_followup"):
            tool_results = self._append_highway_queries_for_od_route(tool_results)
        reply = self._render_reply(
            intent=plan["intent"],
            message=message,
            tool_results=tool_results,
            llm_reply=plan.get("llm_reply", ""),
            rag_hits=rag_hits,
        )
        rag_visible_for_intent = plan["intent"] in {"fare_policy", "unknown"}
        visible_rag_hits = rag_hits if rag_visible_for_intent else []
        follow_items = self._build_follow_ups(message, reply, plan["intent"], tool_results)
        return ChatResponse(
            intent=plan["intent"],
            confidence=plan["confidence"],
            reply=reply,
            actions=[AgentAction(**a) for a in plan["actions"]],
            tool_results=tool_results,
            used_llm=bool(plan.get("used_llm", False)),
            rag_used=bool(visible_rag_hits),
            rag_hits=[{"id": x["id"], "title": x["title"], "score": x["score"]} for x in visible_rag_hits],
            follow_ups=[FollowUpItem(**x) for x in follow_items],
        )

    def _build_follow_ups(
        self,
        user_message: str,
        reply: str,
        intent: str,
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """生成最多 3 条追问；仅用与当前工具能力一致的规则模板，避免 LLM 杜撰点击后无法兑现的问题。"""
        fallback = self._follow_ups_fallback(user_message, reply, intent, tool_results)
        merged: List[Dict[str, str]] = []
        seen_q: set[str] = set()
        for it in fallback:
            q = str(it.get("question", "")).strip()
            a = str(it.get("answer", "")).strip()
            if len(q) < 2 or len(a) < 4:
                continue
            key = q.lower()[:80]
            if key in seen_q:
                continue
            seen_q.add(key)
            merged.append({"question": q[:200], "answer": a[:4000]})
            if len(merged) >= 3:
                break
        return merged[:3]

    def _follow_ups_fallback(
        self,
        user_message: str,
        reply: str,
        intent: str,
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        route = next((x for x in tool_results if x.get("tool") == "query_route_plan" and x.get("success")), None)
        hw = next((x for x in tool_results if x.get("tool") == "query_highway_condition" and x.get("success")), None)
        wx_last = next(
            (x for x in reversed(tool_results) if isinstance(x, dict) and x.get("tool") == "query_weather" and x.get("success")),
            None,
        )
        if isinstance(wx_last, dict):
            pend = wx_last.get("along_route_pending")
            if isinstance(pend, dict) and pend.get("next_city") and pend.get("next_index") is not None:
                nc = str(pend["next_city"]).strip()
                return [
                    {
                        "question": f"继续查询下一途经城市（{nc}）的天气？"[:200],
                        "answer": "点击后查询路线下一站的天气（依次进行）。",
                    },
                    {
                        "question": "从成都到重庆怎么走？",
                        "answer": "点击后发起驾车路径规划（可将城市换成你的起终点）。",
                    },
                    {
                        "question": "G2京沪高速现在好走吗？",
                        "answer": "点击后查询 G2 拥堵、事故与管制摘要。",
                    },
                ]
        if intent == "route_planning" and isinstance(route, dict):
            o, d = str(route.get("origin", "")).strip(), str(route.get("destination", "")).strip()
            via = route.get("waypoints")
            via_s = ""
            if isinstance(via, list) and via:
                via_s = "、".join(str(x).strip() for x in via if str(x).strip())
            leg_hint = f"（经停：{via_s}）" if via_s else ""
            hw_list = route.get("highways") if isinstance(route.get("highways"), list) else []
            hw0 = str(hw_list[0]).strip() if hw_list else ""
            q_hw = (hw0 + "现在好走吗？") if hw0 and len(hw0) <= 18 else "G2京沪高速现在好走吗？"
            weather_item = self._weather_followup_from_route(route)
            return [
                {
                    "question": f"{o}到{d}{leg_hint}沿途高速路况怎么样？" if o and d else "这条路线沿途高速路况怎么样？",
                    "answer": "点击后会再识别起终点并查询沿途高速路况（地点需能解析）。也可直接发「G40 路况」。",
                },
                weather_item,
                {
                    "question": q_hw,
                    "answer": "点击后按高速路况工具查询拥堵、事故与管制摘要。",
                },
            ]
        if intent in {"highway_condition", "realtime_status"} and isinstance(hw, dict):
            code = str(hw.get("code") or "").strip()
            other = "G15沈海高速现在好走吗？" if code.upper() != "G15" else "G2京沪高速现在好走吗？"
            items = [
                {
                    "question": other,
                    "answer": "点击后查询另一条示例高速的路况摘要，便于对比走廊。",
                },
                {
                    "question": "上海到青岛怎么走？",
                    "answer": "点击后发起驾车路径规划，可再改成你的起终点。",
                },
                {
                    "question": f"{code}路况" if code else "G2路况",
                    "answer": f"点击后再次查询{'该高速' if code else 'G2'}路况摘要。",
                },
            ]
            if isinstance(route, dict) and route.get("success"):
                w_item = self._weather_followup_from_route(route)
                items = [w_item, items[0], items[2]]
            return items
        if intent in {"fare_policy", "ticket_refund"}:
            return [
                {
                    "question": "高铁退票手续费一般怎么算？",
                    "answer": "点击后我按知识库与常见规则说明；具体以 12306 或车站当日公告为准。",
                },
                {
                    "question": "从南京到上海怎么走？",
                    "answer": "点击后发起驾车路径规划示例，你可再改成自己的起终点。",
                },
                {
                    "question": "G2京沪高速现在好走吗？",
                    "answer": "点击后查询 G2 路况摘要。",
                },
            ]
        if intent in {"lost_and_found", "complaint", "human_handoff"}:
            return [
                {
                    "question": "北京到天津怎么走？",
                    "answer": "点击后发起驾车路径规划示例。",
                },
                {
                    "question": "G2京沪高速现在好走吗？",
                    "answer": "点击后查询高速路况摘要。",
                },
                {
                    "question": "地铁2号线现在怎么样？",
                    "answer": "点击后查询本助手内置的地铁2号线示例状态。",
                },
            ]
        return [
            {
                "question": "从成都到重庆怎么走？",
                "answer": "点击后发起驾车路径规划（可将城市换成你的起终点）。",
            },
            {
                "question": "G2京沪高速现在好走吗？",
                "answer": "点击后查询 G2 拥堵、事故与管制摘要。",
            },
            {
                "question": "地铁2号线现在怎么样？",
                "answer": "点击后查询内置示例：地铁2号线运行状态（其他线路需你输入全名尝试）。",
            },
        ]

    def _execute_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        last_highway_query_ts = 0.0
        for action in actions:
            tool = action.get("tool", "")
            params = action.get("params", {})
            if tool == "query_route_plan":
                wp = params.get("waypoints")
                if not isinstance(wp, list):
                    wp = None
                results.append(
                    self._query_route_plan(
                        origin=params.get("origin", ""),
                        destination=params.get("destination", ""),
                        mode=params.get("mode", "driving"),
                        waypoints=wp,
                    )
                )
            elif tool == "query_transit_status":
                results.append(self._query_transit_status(params.get("target", "")))
            elif tool == "query_highway_condition":
                # Prevent triggering AMap QPS limits when querying multiple highways in one turn.
                now = time.time()
                wait_s = 0.35 - (now - last_highway_query_ts)
                if wait_s > 0:
                    time.sleep(wait_s)
                context_points = params.get("context_points")
                parsed_points: List[Dict[str, Any]] = []
                if isinstance(context_points, list):
                    for cp in context_points:
                        if isinstance(cp, dict):
                            parsed_points.append(cp)
                legacy_point = params.get("context_point")
                if isinstance(legacy_point, dict):
                    parsed_points.append(legacy_point)
                results.append(self._query_highway_condition(params.get("target", ""), context_points=parsed_points))
                last_highway_query_ts = time.time()
            elif tool == "calculate_fare":
                results.append(
                    self._calculate_fare(
                        origin=params.get("origin", ""),
                        destination=params.get("destination", ""),
                        mode=params.get("mode", "metro"),
                    )
                )
            elif tool == "create_transport_ticket":
                results.append(self._create_transport_ticket(params.get("issue_type", "general"), params.get("detail", "")))
            elif tool == "handoff_to_human":
                results.append(self._handoff_to_human(params.get("priority", "normal")))
            elif tool == "query_weather":
                raw_cities = params.get("cities")
                cities_list: List[str] = [str(x).strip() for x in raw_cities] if isinstance(raw_cities, list) else []
                arq = params.get("along_route_queue")
                ari = params.get("along_route_index")
                results.append(
                    self._query_weather(
                        cities_list,
                        along_route_queue=arq if isinstance(arq, list) else None,
                        along_route_index=ari,
                    )
                )
            else:
                results.append({"tool": tool, "success": False, "error": f"unknown tool: {tool}"})
        return results

    @staticmethod
    def _route_city_sequence(route: Dict[str, Any]) -> List[str]:
        car = route.get("cities_along_route")
        if isinstance(car, list) and len(car) >= 2:
            seq = [str(x).strip() for x in car if str(x).strip()]
            if len(seq) >= 2:
                return seq
        o = str(route.get("origin", "")).strip()
        d = str(route.get("destination", "")).strip()
        via = route.get("waypoints")
        via_list = [str(x).strip() for x in via] if isinstance(via, list) else []
        out: List[str] = []
        seen: set[str] = set()
        for x in [o] + via_list + [d]:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _weather_followup_from_route(self, route: Dict[str, Any]) -> Dict[str, str]:
        return {
            "question": "查询途经城市天气",
            "answer": "点击后助手会先列出路线途经城市，并询问您要查哪一座；回复城市名后再查天气。也可回复「沿途」从第一站起逐站查询。",
        }

    def _append_highway_queries_for_od_route(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """After query_route_plan for OD+路况, query each corridor highway with route probe points."""
        if not tool_results:
            return tool_results
        route = tool_results[0]
        if route.get("tool") != "query_route_plan" or not route.get("success"):
            return tool_results
        pts = route.get("route_points", [])
        ctx = CustomerServiceAgent._probe_points_from_route_points_list(pts)
        codes = self._highway_codes_from_route_plan_result(route)
        if not codes:
            return tool_results
        out = list(tool_results)
        last_hw_ts = 0.0
        for hw in codes[:4]:
            now = time.time()
            wait_s = 0.35 - (now - last_hw_ts)
            if wait_s > 0:
                time.sleep(wait_s)
            out.append(self._query_highway_condition(hw, context_points=ctx))
            last_hw_ts = time.time()
        return out

    def _highway_codes_from_route_plan_result(self, tr: Dict[str, Any]) -> List[str]:
        highs = tr.get("highways", [])
        if not isinstance(highs, list):
            return []
        out: List[str] = []
        seen: set[str] = set()
        for h in highs:
            s = str(h).strip()
            if not s:
                continue
            code = self._extract_highway_target(s) or self._normalize_highway_code(s)
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        if out:
            out.sort(key=lambda x: (0 if str(x).upper().startswith("G") else 1, len(str(x))))
        return out

    @staticmethod
    def _probe_points_from_route_points_list(pts: Any) -> List[Dict[str, Any]]:
        if not isinstance(pts, list) or len(pts) < 2:
            return []
        out: List[Dict[str, Any]] = []
        for frac in [0.0, 0.08, 0.2, 0.5, 0.8, 1.0]:
            idx = min(len(pts) - 1, max(0, int(round((len(pts) - 1) * frac))))
            p = pts[idx]
            if not isinstance(p, list) or len(p) < 2:
                continue
            if not isinstance(p[0], (int, float)) or not isinstance(p[1], (int, float)):
                continue
            out.append({"lat": float(p[0]), "lon": float(p[1])})
        return out

    def _render_reply(
        self,
        intent: IntentType,
        message: str,
        tool_results: List[Dict[str, Any]],
        llm_reply: str,
        rag_hits: List[Dict[str, Any]],
    ) -> str:
        if llm_reply.strip():
            return llm_reply.strip()
        if intent == "weather_query":
            w = next((x for x in tool_results if x.get("tool") == "query_weather"), None)
            if isinstance(w, dict) and w.get("success"):
                lines: List[str] = []
                if w.get("along_route_mode"):
                    pend_raw = w.get("along_route_pending")
                    pend = cast(Dict[str, Any], pend_raw) if isinstance(pend_raw, dict) else {}
                    tot = int(pend.get("total_stops", 0) or 0)
                    cur = int(pend.get("current_stop", 0) or 0)
                    c0 = str((w.get("cities") or ["未知"])[0])
                    if tot > 0 and cur > 0:
                        lines.append(f"途经天气（第 {cur}/{tot} 站：{c0}）")
                    else:
                        lines.append(f"途经天气：{c0}")
                    if pend.get("complete"):
                        lines.append("本条路线途经城市已依次查完。")
                else:
                    lines.append("已查询以下城市天气摘要：")
                for block in w.get("forecasts", []) if isinstance(w.get("forecasts"), list) else []:
                    if not isinstance(block, dict) or not block.get("ok"):
                        err = str(block.get("error", "暂无数据") if isinstance(block, dict) else "暂无数据")
                        cname = str(block.get("city", "") if isinstance(block, dict) else "")
                        lines.append(f"- {cname or '某城市'}：{err}")
                        continue
                    cname = str(block.get("city", "")).strip()
                    src = str(block.get("source", ""))
                    if src == "amap_weather":
                        live_raw = block.get("live")
                        live = cast(Dict[str, Any], live_raw) if isinstance(live_raw, dict) else {}
                        weather = str(live.get("weather", "")).strip()
                        temp = str(live.get("temperature", "")).strip()
                        wind = str(live.get("winddirection", "")).strip() + str(live.get("windpower", "")).strip()
                        hum = str(live.get("humidity", "")).strip()
                        live_bits: List[str] = []
                        if temp:
                            live_bits.append(f"气温{temp}℃")
                        if weather:
                            live_bits.append(weather)
                        if hum:
                            live_bits.append(f"湿度{hum}%")
                        if wind:
                            live_bits.append(wind)
                        cast0_raw = block.get("cast0")
                        cast0 = cast(Dict[str, Any], cast0_raw) if isinstance(cast0_raw, dict) else {}
                        fc_line = ""
                        if cast0:
                            dayw = str(cast0.get("dayweather", "")).strip()
                            nt = str(cast0.get("nighttemp", "")).strip()
                            dt = str(cast0.get("daytemp", "")).strip()
                            if dayw or nt or dt:
                                fc_line = f"明日预报 {dayw or '—'} {nt}～{dt}℃".replace("～℃", "℃")
                        segm: List[str] = []
                        if live_bits:
                            segm.append("，".join(live_bits))
                        if fc_line:
                            segm.append(fc_line)
                        extra = "；".join(segm) if segm else "详见气象服务"
                        lines.append(f"- {cname}：{extra}（高德天气）")
                    else:
                        cur_raw = block.get("current")
                        cur = cast(Dict[str, Any], cur_raw) if isinstance(cur_raw, dict) else {}
                        t2 = cur.get("temperature_2m")
                        desc = str(block.get("weather_desc", "")).strip()
                        tline = f"当前约 {t2}℃" if t2 is not None else ""
                        lines.append(f"- {cname}：{desc or '天气'}{('，' + tline) if tline else ''}（Open-Meteo）")
                ok_forecasts = [
                    b
                    for b in (w.get("forecasts") or [])
                    if isinstance(b, dict) and b.get("ok")
                ]
                if ok_forecasts:
                    if len(ok_forecasts) == 1:
                        t_eff, wb = CustomerServiceAgent._weather_block_effective_temp_and_desc(ok_forecasts[0])
                        hint = CustomerServiceAgent._outfit_recommend_cn(t_eff, wb)
                        lines.append(f"衣着建议：{hint}")
                    else:
                        lines.append("衣着建议：")
                        for b in ok_forecasts:
                            cn = str(b.get("city", "")).strip() or "当地"
                            t_eff, wb = CustomerServiceAgent._weather_block_effective_temp_and_desc(b)
                            hint = CustomerServiceAgent._outfit_recommend_cn(t_eff, wb)
                            lines.append(f"- {cn}：{hint}")
                pend2_raw = w.get("along_route_pending")
                pend2 = cast(Dict[str, Any], pend2_raw) if isinstance(pend2_raw, dict) else {}
                if w.get("along_route_mode") and pend2.get("next_city") and not pend2.get("complete"):
                    lines.append(
                        "要继续查询下一站，请点击下方「继续查询下一途经城市」追问，或发送「继续查询下一途经城市天气」。"
                    )
                    lines.append(
                        "若想一次查询多个城市，可直接回复城市名并用顿号或逗号分隔（例如「南通、苏州、连云港」），将合并为本轮天气摘要。"
                    )
                lines.append("长途出行请结合沿途实时预警与导航提示。")
                return "\n".join(lines).strip()
            err = str(w.get("error", "查询失败") if isinstance(w, dict) else "查询失败")
            return (
                f"暂时查不到天气数据。\n{err}\n"
                "可尝试写清城市名（如「上海天气」），或先规划路线再点「你可能还想问」里的天气追问。"
            )
        if intent == "route_planning":
            if tool_results and tool_results[0].get("success"):
                d = tool_results[0]
                src = str(d.get("source", ""))
                source_hint = (
                    "（实时API）" if src in {"osrm", "amap_direction", "multi_segment"} or d.get("multi_segment") else "（估算）"
                )
                highways = d.get("highways") or []
                highway_text = "、".join(highways[:4]) if highways else "暂无"
                hints_raw = d.get("trip_hints") or []
                hint_lines: List[str] = []
                if isinstance(hints_raw, list):
                    for h in hints_raw:
                        if not isinstance(h, dict):
                            continue
                        title = str(h.get("title", "")).strip()
                        detail = str(h.get("detail", "")).strip()
                        if title and detail:
                            hint_lines.append(f"· {title}：{detail}")
                hints_text = "\n".join(hint_lines) if hint_lines else "（详见左侧「行程提示」）"
                if d.get("multi_segment"):
                    waypoints_raw = d.get("waypoints")
                    via_list: List[Any] = waypoints_raw if isinstance(waypoints_raw, list) else []
                    chain = " → ".join(
                        [str(d.get("origin", ""))] + [str(x) for x in via_list] + [str(d.get("destination", ""))]
                    )
                    leg_blocks: List[str] = []
                    details = d.get("leg_details") if isinstance(d.get("leg_details"), list) else []
                    if details:
                        for ld in details:
                            if not isinstance(ld, dict):
                                continue
                            lf = str(ld.get("from", "")).strip()
                            lt = str(ld.get("to", "")).strip()
                            dk = ld.get("distance_km", "?")
                            dm = ld.get("duration_min", "?")
                            leg_blocks.append(
                                f"【{lf} → {lt}】约 {dk} 公里，约 {dm} 分钟"
                            )
                            _hw = ld.get("highways")
                            hw_list: List[Any] = _hw if isinstance(_hw, list) else []
                            hw_s = "、".join(str(x).strip() for x in hw_list[:5] if str(x).strip()) or "暂无"
                            leg_blocks.append(f"  本段途经高速：{hw_s}")
                    else:
                        leg_lines = d.get("leg_summaries") if isinstance(d.get("leg_summaries"), list) else []
                        if leg_lines:
                            leg_blocks.append("\n".join(f"  - {x}" for x in leg_lines))
                    legs_txt = "\n".join(leg_blocks) + "\n" if leg_blocks else ""
                    body = (
                        f"给你规划好多段路线：{chain} {source_hint}，分段说明如下：\n"
                        f"{legs_txt}"
                        f"- 合计里程：约 {d['distance_km']} 公里\n"
                        f"- 合计时长：约 {d['duration_min']} 分钟\n"
                        f"- 全程说明：{d['summary']}\n"
                        f"- 沿途主要高速（合并参考）：{highway_text}\n"
                        f"- 行程提示：\n{hints_text}\n"
                        "出发前建议用导航按分段复核路线，并留意实时路况。"
                    )
                    return body.strip()
                return (
                    f"给你规划好了，从 {d['origin']} 到 {d['destination']} {source_hint}：\n"
                    f"- 预计里程：约 {d['distance_km']} 公里\n"
                    f"- 预计时长：约 {d['duration_min']} 分钟\n"
                    f"- 路线建议：{d['summary']}\n"
                    f"- 途经高速：{highway_text}\n"
                    f"- 行程提示：\n{hints_text}\n"
                    "出发前建议再看一次实时路况，避免临时拥堵影响行程。"
                )
            if tool_results and tool_results[0].get("error"):
                return (
                    "抱歉，这次路线规划没成功。\n"
                    f"原因：{tool_results[0]['error']}\n"
                    "你可以稍后重试，或者把起终点写得更具体一些（例如“南通站 -> 北京南站”）。"
                )
            if rag_hits:
                best = rag_hits[0]
                return f"{best['content']}\n\n（知识来源：{best['title']}）"
            return "可以的，请告诉我起点和终点，比如：从南通到北京怎么走。"
        if intent == "realtime_status":
            if tool_results and tool_results[0].get("success"):
                d = tool_results[0]
                if d.get("next_train_min") is None:
                    return f"帮你查到了：{d['target']} 目前是【{d['status']}】。{d['notice']}。"
                return (
                    f"帮你查到了：{d['target']} 目前是【{d['status']}】。\n"
                    f"预计下一班约 {d['next_train_min']} 分钟，{d['notice']}。"
                )
            return "请告诉我要查的线路或路段，比如“地铁2号线”或“S1高架”。"
        if intent == "highway_condition":
            route_ok = next(
                (x for x in (tool_results or []) if x.get("tool") == "query_route_plan" and x.get("success")),
                None,
            )
            hw_success = [
                x for x in (tool_results or []) if x.get("tool") == "query_highway_condition" and x.get("success")
            ]
            source_map = {
                "amap_api": "高德实时路况",
                "amap_circle_api": "高德实时路况",
                "mock_realtime_feed": "本地样例库（API不可用时）",
                "fallback_no_api": "综合路况估算",
            }
            if route_ok and hw_success:
                lines = [
                    f"已按 **{route_ok.get('origin', '')} → {route_ok.get('destination', '')}** 驾车走廊查询沿途主要高速路况："
                ]
                for d in hw_success[:6]:
                    has_accident = bool(d.get("has_accident"))
                    source_text = source_map.get(str(d.get("source", "")), "未知来源")
                    status_text = "有事故" if has_accident else "未发现事故/管制异常"
                    lines.append(
                        f"- {d.get('name', '未知高速')}（{d.get('code', '--')}）："
                        f"{d.get('congestion_level', '未知')}，{status_text}，来源={source_text}"
                    )
                lines.append(
                    f"路线参考：约 {route_ok.get('distance_km', '?')} 公里、约 {route_ok.get('duration_min', '?')} 分钟；"
                    "出发前建议再用导航确认实时拥堵与管制。"
                )
                return "\n".join(lines)
            if route_ok and not hw_success:
                return (
                    f"已规划 {route_ok.get('origin', '')} → {route_ok.get('destination', '')}（约 "
                    f"{route_ok.get('distance_km', '?')} 公里，约 {route_ok.get('duration_min', '?')} 分钟），"
                    "但沿途高速路况暂时未能拉取，请稍后重试或确认已配置路况查询（如 AMap Key）。"
                )
            route_fail = next(
                (
                    x
                    for x in (tool_results or [])
                    if x.get("tool") == "query_route_plan" and not x.get("success")
                ),
                None,
            )
            if route_fail:
                err = str(route_fail.get("error", "未知错误"))
                if "geocode" in err.lower():
                    return (
                        "没能根据你提供的起、终点解析出位置，所以还无法规划驾车走廊，也就查不了「沿途高速」路况。\n"
                        f"（系统提示：{err}）\n"
                        "你可以：① 写全名或带省名，如「南京市→昆明市」；口语里单独一个「宁」一般指南京，系统已尽量自动识别；"
                        "② 配置环境变量 `AMAP_API_KEY` 国内解析更稳；③ 若只关心某条高速，可直接问「G60 路况」等。"
                    )
                return (
                    "路线规划这一步没有成功，暂时无法按起终点沿途查询高速路况。\n"
                    f"原因：{err}\n"
                    "请换个说法或稍后重试。"
                )
            if tool_results and any(x.get("success") for x in tool_results):
                success_items = [
                    x for x in tool_results if x.get("success") and x.get("tool") != "query_route_plan"
                ]
                if not success_items:
                    return "请提供要查询的高速名称或编号，例如：G2、京沪高速、S1高架。"
                if len(success_items) == 1:
                    d = success_items[0]
                    controls = d.get("controls", []) if isinstance(d.get("controls"), list) else []
                    has_controls = any(str(x).strip() and "未发现" not in str(x) for x in controls)
                    incident_text = str(d.get("incident", "暂无事故上报")).strip()
                    has_accident = bool(d.get("has_accident"))
                    if not has_accident and ("未发现" in incident_text or "暂无" in incident_text):
                        incident_text = "当前未发现事故上报"
                    control_text = "；".join(controls) if controls else "当前未发现交通管制信息"
                    if not has_controls and ("暂无" in control_text or "未发现" in control_text):
                        control_text = "当前未发现交通管制信息"
                    source_text = source_map.get(str(d.get("source", "")), "未知来源")
                    return (
                        f"已为你查询 {d['name']}（{d['code']}）当前路况：\n"
                        f"- 拥堵等级：{d['congestion_level']}\n"
                        f"- 事故情况：{incident_text}\n"
                        f"- 管制信息：{control_text}\n"
                        f"- 出行建议：{d['advice']}\n"
                        f"- 数据来源：{source_text}"
                    )
                lines = ["已根据你最近规划路线，逐条查询途经高速路况："]
                for d in success_items[:6]:
                    has_accident = bool(d.get("has_accident"))
                    source_text = source_map.get(str(d.get("source", "")), "未知来源")
                    status_text = "有事故" if has_accident else "未发现事故/管制异常"
                    lines.append(
                        f"- {d.get('name','未知高速')}（{d.get('code','--')}）："
                        f"{d.get('congestion_level','未知')}，{status_text}，来源={source_text}"
                    )
                lines.append("如果你要，我可以继续展开某一条高速的事故详情和管制信息。")
                return "\n".join(lines)
            if tool_results and tool_results[0].get("error"):
                first_err = str(tool_results[0].get("error", ""))
                return (
                    "这条高速我暂时没查到有效路况数据。\n"
                    f"原因：{first_err}\n"
                    "你可以换一个表达再试（例如：G15 沈海高速今天有事故吗），"
                    "或者先在配置里设置 `AMAP_API_KEY` 以启用高德实时路况。"
                )
            return "请提供要查询的高速名称或编号，例如：G2、京沪高速、S1高架。"
        if intent in {"ticket_refund", "lost_and_found", "complaint"}:
            ticket = tool_results[0]["ticket_id"] if tool_results else "N/A"
            return f"已帮你提交服务工单（{ticket}）。我们会尽快跟进处理，请留意后续通知。"
        if intent in {"fare_policy", "route_planning", "unknown"} and rag_hits:
            # Use RAG answer chain only for knowledge-heavy intents.
            if self.answer_chain is not None:
                try:
                    return self.answer_chain.invoke(
                        {"question": message, "context": self._rag_to_text(rag_hits)}
                    ).strip()
                except Exception:
                    pass
            best = rag_hits[0]
            return f"{best['content']}\n\n（知识来源：{best['title']}）"
        if intent == "human_handoff":
            return "好的，已为你转接人工交通客服，请稍候，系统会优先处理你的问题。"
        return (
            f"我收到了你的问题：{message[:40]}。\n"
            "你可以再补充一下线路、站点、出发时间或目的地，我就能给你更准确的建议。"
        )

    @staticmethod
    def _wmo_weather_desc(code: int) -> str:
        m = {
            0: "晴",
            1: "晴",
            2: "多云",
            3: "阴",
            45: "雾",
            48: "雾",
            51: "小毛毛雨",
            53: "毛毛雨",
            55: "强毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            80: "阵雨",
            81: "阵雨",
            82: "暴雨",
            95: "雷雨",
            96: "雷雨",
            99: "雷雨",
        }
        return m.get(int(code), f"天气代码{code}")

    @staticmethod
    def _weather_block_effective_temp_and_desc(block: Dict[str, Any]) -> tuple[float | None, str]:
        """从单城天气结果块提取用于穿搭建议的代表气温与天气描述文本。"""
        if not isinstance(block, dict) or not block.get("ok"):
            return None, ""
        src = str(block.get("source", ""))
        desc_parts: List[str] = []
        eff: float | None = None
        if src == "amap_weather":
            live_raw = block.get("live")
            cast0_raw = block.get("cast0")
            live: Dict[str, Any] = live_raw if isinstance(live_raw, dict) else {}
            cast0: Dict[str, Any] = cast0_raw if isinstance(cast0_raw, dict) else {}
            wt = str(live.get("weather", "")).strip()
            if wt:
                desc_parts.append(wt)
            dw = str(cast0.get("dayweather", "")).strip()
            if dw:
                desc_parts.append(dw)
            ts = str(live.get("temperature", "")).strip()
            if ts:
                try:
                    eff = float(ts)
                except ValueError:
                    pass
            if eff is None:
                nums: List[float] = []
                for key in ("daytemp", "nighttemp"):
                    x = str(cast0.get(key, "")).strip()
                    if not x:
                        continue
                    try:
                        nums.append(float(x))
                    except ValueError:
                        pass
                if nums:
                    eff = sum(nums) / len(nums)
        else:
            cur_raw = block.get("current")
            cur: Dict[str, Any] = cur_raw if isinstance(cur_raw, dict) else {}
            t2 = cur.get("temperature_2m")
            if t2 is not None:
                try:
                    eff = float(t2)
                except (TypeError, ValueError):
                    pass
            d = str(block.get("weather_desc", "")).strip()
            if d:
                desc_parts.append(d)
        return eff, " ".join(desc_parts)

    @staticmethod
    def _outfit_recommend_cn(temp_c: float | None, weather_blob: str) -> str:
        """根据气温与天气描述生成简短中文衣着建议（规则模板，不调用模型）。"""
        w = weather_blob or ""
        has_rain = any(
            k in w
            for k in (
                "雨",
                "雪",
                "雹",
                "阵雪",
                "阵雨",
                "雷阵雨",
                "毛毛雨",
                "冻雨",
                "小到中雨",
                "中到大雨",
            )
        )
        has_strong_wind = ("风" in w) and any(x in w for x in ("大", "强", "阵", "沙尘", "飓"))

        if temp_c is None:
            base = "请关注当地气温变化，适时增减衣物。"
            if has_rain:
                base += "有降水时建议携带雨具、穿防滑鞋。"
            return base

        parts: List[str] = []
        if temp_c >= 30:
            parts.append("天气炎热，宜轻薄透气的短袖、短裤或裙装，注意防晒与补水。")
        elif temp_c >= 25:
            parts.append("气温偏高，可选短袖或薄长袖，搭配防晒衣、遮阳帽更舒适。")
        elif temp_c >= 18:
            parts.append("体感较舒适，长袖或T恤外加薄外套即可，早晚偏凉可搭一件开衫。")
        elif temp_c >= 10:
            parts.append("天气偏凉，建议外套、卫衣或针织衫，注意腰腹与足部保暖。")
        elif temp_c >= 0:
            parts.append("气温较低，厚外套、棉衣或轻羽绒更合适，可备围巾、手套。")
        else:
            parts.append("天气严寒，建议羽绒服或厚棉衣分层穿着，戴帽围巾并做好防风。")

        if has_rain:
            parts.append("有雨雪时建议带折叠伞或雨衣，鞋子选防滑、易打理的款式。")
        if has_strong_wind and temp_c < 22:
            parts.append("风力较大时可加一件防风外套。")

        return "".join(parts)

    def _query_weather_amap(self, city: str) -> Dict[str, Any]:
        if not self.amap_api_key:
            return {"city": city, "ok": False, "error": "no amap key"}
        try:
            ad = self._amap_geocode_adcode(city) or city
            url = "https://restapi.amap.com/v3/weather/weatherInfo"
            params = {"key": self.amap_api_key, "city": ad, "extensions": "all"}
            resp = self._http_get(url, params=params, timeout=12, bypass_proxy=self.amap_bypass_proxy)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return {
                    "city": city,
                    "ok": False,
                    "error": str(payload.get("info", "amap weather failed")),
                }
            lives = payload.get("lives", [])
            forecasts = payload.get("forecasts", [])
            live: Dict[str, Any] = {}
            if lives and isinstance(lives[0], dict):
                live = dict(lives[0])
            cast0: Dict[str, Any] = {}
            if forecasts and isinstance(forecasts[0], dict):
                casts = forecasts[0].get("casts", [])
                if isinstance(casts, list) and casts and isinstance(casts[0], dict):
                    cast0 = dict(casts[0])
            return {"city": city, "ok": True, "source": "amap_weather", "live": live, "cast0": cast0}
        except Exception as e:
            return {"city": city, "ok": False, "error": str(e)}

    def _query_weather_openmeteo(self, city: str, lat: float, lon: float) -> Dict[str, Any]:
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "timezone": "Asia/Shanghai",
                "forecast_days": 2,
            }
            resp = requests.get(url, params=params, timeout=14)
            resp.raise_for_status()
            data = resp.json()
            cur = data.get("current") if isinstance(data.get("current"), dict) else {}
            wc = cur.get("weather_code")
            desc = CustomerServiceAgent._wmo_weather_desc(int(wc)) if wc is not None else "天气"
            return {"city": city, "ok": True, "source": "open_meteo", "current": cur, "weather_desc": desc}
        except Exception as e:
            return {"city": city, "ok": False, "error": str(e)}

    def _query_weather_single_city(self, city: str) -> Dict[str, Any]:
        if self.amap_api_key:
            hit = self._query_weather_amap(city)
            if hit.get("ok"):
                return hit
        geo = self._geocode_place(self._normalize_place_name(city)) or self._lookup_builtin_coord(city)
        if not geo:
            return {"city": city, "ok": False, "error": "无法解析城市位置"}
        return self._query_weather_openmeteo(city, float(geo["lat"]), float(geo["lon"]))

    def _query_weather(
        self,
        cities: List[str],
        along_route_queue: List[Any] | None = None,
        along_route_index: Any = None,
    ) -> Dict[str, Any]:
        if isinstance(along_route_queue, list) and along_route_queue:
            try:
                idx = int(along_route_index) if along_route_index is not None else 0
            except (TypeError, ValueError):
                idx = 0
            cleaned_queue: List[str] = []
            seen_q: set[str] = set()
            for c in along_route_queue:
                t = CustomerServiceAgent._clean_place(str(c).strip())
                if t and t not in seen_q:
                    seen_q.add(t)
                    cleaned_queue.append(t)
            if not cleaned_queue:
                return {
                    "tool": "query_weather",
                    "success": False,
                    "cities": [],
                    "forecasts": [],
                    "along_route_mode": True,
                    "along_route_pending": None,
                    "error": "路线城市列表为空",
                }
            if idx < 0 or idx >= len(cleaned_queue):
                return {
                    "tool": "query_weather",
                    "success": False,
                    "cities": [],
                    "forecasts": [],
                    "along_route_mode": True,
                    "along_route_pending": None,
                    "error": "途经天气序号无效",
                }
            target_city = cleaned_queue[idx]
            forecasts = [self._query_weather_single_city(target_city)]
            ok_any = any(bool(x.get("ok")) for x in forecasts)
            if idx + 1 < len(cleaned_queue):
                along_route_pending: Dict[str, Any] = {
                    "queue": cleaned_queue,
                    "next_index": idx + 1,
                    "next_city": cleaned_queue[idx + 1],
                    "total_stops": len(cleaned_queue),
                    "current_stop": idx + 1,
                }
            else:
                along_route_pending = {
                    "complete": True,
                    "queue": cleaned_queue,
                    "total_stops": len(cleaned_queue),
                    "current_stop": idx + 1,
                }
            return {
                "tool": "query_weather",
                "success": ok_any,
                "cities": [target_city],
                "forecasts": forecasts,
                "along_route_mode": True,
                "along_route_pending": along_route_pending,
                "error": "" if ok_any else "未能获取天气数据",
            }

        cleaned: List[str] = []
        seen: set[str] = set()
        for c in cities:
            t = CustomerServiceAgent._clean_place(str(c).strip())
            if t and t not in seen:
                seen.add(t)
                cleaned.append(t)
        if not cleaned:
            return {
                "tool": "query_weather",
                "success": False,
                "cities": [],
                "forecasts": [],
                "error": "请说明要查的城市，或先规划路线再点追问里的天气。",
            }
        forecasts = []
        for city in cleaned[:10]:
            forecasts.append(self._query_weather_single_city(city))
        ok_any = any(bool(x.get("ok")) for x in forecasts)
        return {
            "tool": "query_weather",
            "success": ok_any,
            "cities": cleaned[:10],
            "forecasts": forecasts,
            "error": "" if ok_any else "未能获取天气数据",
        }

    @staticmethod
    def parse_cities_from_weather_followup_question(message: str) -> List[str]:
        """解析「你可能还想问」天气芯片发送的短句，得到城市列表；空列表表示需结合历史路线。"""
        raw = (message or "").strip()
        m = re.match(r"^查一下(.+?)的天气吗[？?]?\s*$", raw)
        if not m:
            return []
        body = m.group(1).strip()
        if "沿途城市" in body or "刚才路线" in body:
            return []
        parts = re.split(r"[、,，]", body)
        return [p.strip() for p in parts if p.strip() and len(p.strip()) <= 24]

    @staticmethod
    def parse_weather_city_list_from_message(message: str) -> List[str]:
        """
        从「大同、忻州天气怎么样」等句解析多城：先去掉句末天气问法，再按顿号/逗号切分。
        避免单城正则只匹配到「忻州」而漏掉「大同」。
        """
        raw = (message or "").strip()
        if not raw:
            return []
        s = re.sub(
            r"(?:的)?(?:天气|气温)(?:怎么样|如何|怎样|好吗|好么|呢)?\s*$",
            "",
            raw,
        ).strip()
        s = re.sub(r"\s*怎么样\s*$", "", s).strip()
        s = re.sub(r"\s*如何\s*$", "", s).strip()
        if not re.search(r"[、,，;；]", s):
            return []
        parts = re.split(r"[、,，;；]+", s)
        skip = {
            "天气",
            "气温",
            "查询",
            "怎么样",
            "如何",
            "帮我",
            "然后",
            "还有",
            "途径",
            "途经",
            "沿途",
            "今天",
            "现在",
            "明日",
            "这周",
        }
        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            t = CustomerServiceAgent._clean_place(
                CustomerServiceAgent._trim_route_context_suffix_from_place_token(p.strip())
            )
            if not t or t in skip:
                continue
            if len(t) < 2 or len(t) > 16:
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:10]

    def _query_transit_status(self, target: str) -> Dict[str, Any]:
        if not target:
            return {"tool": "query_transit_status", "success": False, "error": "missing target"}
        if target not in self.transit_status:
            return {"tool": "query_transit_status", "success": False, "error": f"target not found: {target}"}
        d = self.transit_status[target]
        return {"tool": "query_transit_status", "success": True, "target": target, **d}

    def _query_highway_condition(
        self, target: str, context_points: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        code = self._normalize_highway_code(target)
        if not code:
            return {"tool": "query_highway_condition", "success": False, "error": "missing or unknown highway target"}
        normalized_context_points = self._normalize_probe_points(context_points)
        # Prefer real-time API first when configured.
        api_result = self._query_highway_condition_by_api(
            code=code,
            target=target,
            context_points=normalized_context_points,
        )
        if api_result.get("success"):
            return api_result

        d = self.highway_conditions.get(code)
        if not d:
            return self._fallback_highway_condition(
                code=code,
                target=target,
                probe_points=normalized_context_points,
                reason=f"highway not found in local data: {target}; api_error={api_result.get('error','unknown')}",
            )
        return {
            "tool": "query_highway_condition",
            "success": True,
            "code": code,
            "name": d["name"],
            "congestion_level": d["congestion_level"],
            "has_accident": d["has_accident"],
            "incident": d["incident"],
            "controls": d["controls"],
            "advice": d["advice"],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "mock_realtime_feed",
            "warning": f"api_error={api_result.get('error','unknown')}",
            "probe_points": normalized_context_points,
        }

    def _fallback_highway_condition(
        self,
        code: str,
        target: str,
        reason: str,
        probe_points: List[Dict[str, float]] | None = None,
    ) -> Dict[str, Any]:
        name = self._highway_name_from_code(code) or target
        return {
            "tool": "query_highway_condition",
            "success": True,
            "code": code,
            "name": name,
            "congestion_level": "暂无实时数据",
            "has_accident": False,
            "incident": "当前未发现明确事故上报",
            "controls": ["当前未发现明确交通管制信息"],
            "advice": "建议出发前再次查看官方路况播报，按现场指引通行",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "fallback_no_api",
            "warning": reason,
            "probe_points": self._normalize_probe_points(probe_points),
        }

    def _query_highway_condition_by_api(
        self, code: str, target: str, context_points: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """
        Query highway traffic via AMap circle traffic API.
        """
        if not self.amap_api_key:
            return {"tool": "query_highway_condition", "success": False, "error": "AMAP_API_KEY is not configured"}
        try:
            query_points: List[Dict[str, float]] = self._normalize_probe_points(context_points)
            for q in self._build_highway_query_candidates(code=code, target=target):
                guessed = self._geocode_place_by_amap(q)
                if guessed:
                    self._append_probe_point(query_points, float(guessed["lat"]), float(guessed["lon"]))
                for searched in self._search_place_points_by_amap(q, max_points=5):
                    self._append_probe_point(query_points, float(searched["lat"]), float(searched["lon"]))
            if not query_points:
                return {"tool": "query_highway_condition", "success": False, "error": "missing query location for circle api"}

            url = "https://restapi.amap.com/v3/traffic/status/circle"
            payload: Dict[str, Any] = {}
            last_error = "unknown"
            for qp in query_points:
                for radius in ("5000", "12000", "25000"):
                    params = {
                        "key": self.amap_api_key,
                        "location": f"{qp['lon']},{qp['lat']}",
                        "radius": radius,
                    }
                    # Retry a couple of times for QPS-throttled responses.
                    temp: Dict[str, Any] = {}
                    for _ in range(3):
                        resp = self._http_get(url, params=params, timeout=10, bypass_proxy=self.amap_bypass_proxy)
                        resp.raise_for_status()
                        temp = resp.json()
                        info = str(temp.get("info", ""))
                        if info == "CUQPS_HAS_EXCEEDED_THE_LIMIT":
                            time.sleep(0.4)
                            continue
                        break
                    if str(temp.get("status", "0")) == "1":
                        payload = temp
                        break
                    last_error = str(temp.get("info", "unknown"))
                if payload:
                    break
            if not payload:
                return {"tool": "query_highway_condition", "success": False, "error": f"amap circle failed: {last_error}"}

            info = payload.get("trafficinfo", {}) if isinstance(payload, dict) else {}
            description = ""
            evaluation_text = ""
            if isinstance(info, dict):
                desc_val = info.get("description", "")
                if isinstance(desc_val, str):
                    description = desc_val.strip()
                elif isinstance(desc_val, list):
                    description = "；".join([str(x).strip() for x in desc_val if str(x).strip()])
                eval_val = info.get("evaluation", {})
                if isinstance(eval_val, dict):
                    evaluation_text = str(eval_val.get("description", "")).strip()
                elif isinstance(eval_val, str):
                    evaluation_text = eval_val.strip()

            congestion_level = evaluation_text if evaluation_text else "未知"
            has_accident = any(k in description for k in ["事故", "追尾", "碰撞", "剐蹭"])
            controls: List[str] = []
            if any(k in description for k in ["管制", "封闭", "限行", "限速"]):
                controls.append("存在临时交通管制，请关注现场指引")
            if not controls:
                controls.append("未发现明确管制信息")
            advice = "建议出发前再次刷新路况，必要时选择替代高速或错峰出行"
            return {
                "tool": "query_highway_condition",
                "success": True,
                "code": code,
                "name": self._highway_name_from_code(code) or target,
                "congestion_level": congestion_level,
                "has_accident": has_accident,
                "incident": description or "暂无事故上报",
                "controls": controls,
                "advice": advice,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "amap_circle_api",
                "probe_points": query_points[:8],
            }
        except Exception as e:
            return {"tool": "query_highway_condition", "success": False, "error": str(e)}

    @staticmethod
    def _append_probe_point(points: List[Dict[str, float]], lat: float, lon: float) -> None:
        key = (round(lat, 4), round(lon, 4))
        for p in points:
            if (round(float(p.get("lat", 0.0)), 4), round(float(p.get("lon", 0.0)), 4)) == key:
                return
        points.append({"lat": lat, "lon": lon})

    def _normalize_probe_points(self, points: List[Dict[str, Any]] | None) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        if not isinstance(points, list):
            return out
        for cp in points:
            if not isinstance(cp, dict):
                continue
            if isinstance(cp.get("lat"), (int, float)) and isinstance(cp.get("lon"), (int, float)):
                self._append_probe_point(out, float(cp["lat"]), float(cp["lon"]))
        return out

    def _build_highway_query_candidates(self, code: str, target: str) -> List[str]:
        candidates = [
            self._highway_name_from_code(code),
            f"{code}高速",
            f"{code}高速公路",
            code,
            target,
            f"{target}高速",
        ]
        out: List[str] = []
        seen: set[str] = set()
        for c in candidates:
            n = str(c).strip()
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _search_place_point_by_amap(self, keyword: str) -> Dict[str, float] | None:
        points = self._search_place_points_by_amap(keyword, max_points=1)
        return points[0] if points else None

    def _search_place_points_by_amap(self, keyword: str, max_points: int = 5) -> List[Dict[str, float]]:
        if not self.amap_api_key:
            return []
        try:
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "key": self.amap_api_key,
                "keywords": keyword,
                "offset": str(max(1, min(max_points, 20))),
                "page": "1",
                "extensions": "base",
            }
            resp = self._http_get(url, params=params, timeout=10, bypass_proxy=self.amap_bypass_proxy)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return []
            pois = payload.get("pois", [])
            if not isinstance(pois, list) or not pois:
                return []
            out: List[Dict[str, float]] = []
            seen: set[tuple[float, float]] = set()
            for p in pois:
                if not isinstance(p, dict):
                    continue
                loc = str(p.get("location", "")).strip()
                if "," not in loc:
                    continue
                lon_s, lat_s = loc.split(",", 1)
                try:
                    lat = float(lat_s)
                    lon = float(lon_s)
                except Exception:
                    continue
                key = (round(lat, 4), round(lon, 4))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"lat": lat, "lon": lon})
                if len(out) >= max_points:
                    break
            return out
        except Exception:
            return []

    @staticmethod
    def _http_get(url: str, params: Dict[str, Any], timeout: int = 10, bypass_proxy: bool = False) -> requests.Response:
        if not bypass_proxy:
            return requests.get(url, params=params, timeout=timeout)
        with requests.Session() as session:
            session.trust_env = False
            return session.get(url, params=params, timeout=timeout)

    def _build_trip_hints(
        self,
        *,
        distance_km: float,
        duration_min: int,
        origin: str,
        destination: str,
        highways: List[str],
        multi_segment: bool = False,
        waypoints: List[str] | None = None,
        source: str = "",
        avoid_ferry: bool = False,
    ) -> List[Dict[str, str]]:
        """基于里程/耗时/路段信息的出行提示（不依赖服务区 POI）。"""
        hints: List[Dict[str, str]] = []
        dk = max(float(distance_km), 0.1)
        dm = max(int(duration_min), 1)
        hours = dm / 60.0
        if hours >= 3.0:
            hints.append(
                {
                    "title": "休息安排",
                    "detail": f"预计全程驾驶约 {hours:.1f} 小时，建议每 2 小时左右在安全区域休息，避免疲劳驾驶。",
                }
            )
        elif hours >= 1.5:
            hints.append(
                {
                    "title": "休息安排",
                    "detail": "中短途也请适时休息，保持注意力与车距。",
                }
            )
        else:
            hints.append(
                {
                    "title": "安全驾驶",
                    "detail": "请勿分心驾驶，遵守限速与信号灯。",
                }
            )
        hints.append(
            {
                "title": "导航与路况",
                "detail": "出发前用导航再次确认路线；途中关注拥堵、事故与临时管制，必要时调整行程。",
            }
        )
        if multi_segment and waypoints:
            via = "、".join(str(x).strip() for x in waypoints[:6] if str(x).strip())
            if via:
                hints.append(
                    {
                        "title": "多段途经点",
                        "detail": f"途经「{via}」，请在导航中核对途经顺序与下道口。",
                    }
                )
        if avoid_ferry or source == "amap_direction":
            hints.append(
                {
                    "title": "陆路路线",
                    "detail": "当前方案按陆路连续路径；若地图上有跨海直线多为示意，请以导航实走为准。",
                }
            )
        hw_preview = "、".join(str(x) for x in highways[:5] if str(x).strip()) if highways else "沿途主干道"
        hints.append(
            {
                "title": "高速与收费",
                "detail": f"参考途经：{hw_preview}。ETC/收费政策以当地为准，高峰时段预留排队时间。",
            }
        )
        if dk >= 400:
            hints.append(
                {
                    "title": "长途车况",
                    "detail": f"单程约 {dk:.0f} 公里，建议检查轮胎、油液与灯光，携带证件与应急用品。",
                }
            )
        hints.append(
            {
                "title": "限行与法规",
                "detail": "各地货车限行、单双号等政策可能调整，请以交管与导航当日公告为准。",
            }
        )
        return hints[:8]

    def _attach_trip_hints_to_route_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("success") or result.get("tool") != "query_route_plan":
            return result
        out = dict(result)
        out["service_areas"] = []
        ld = out.get("leg_details")
        if isinstance(ld, list):
            new_ld: List[Dict[str, Any]] = []
            for item in ld:
                if not isinstance(item, dict):
                    continue
                z = dict(item)
                z["service_areas"] = []
                new_ld.append(z)
            out["leg_details"] = new_ld
        wp = out.get("waypoints")
        way_list = list(wp) if isinstance(wp, list) else None
        out["trip_hints"] = self._build_trip_hints(
            distance_km=float(out.get("distance_km") or 0),
            duration_min=int(out.get("duration_min") or 0),
            origin=str(out.get("origin") or ""),
            destination=str(out.get("destination") or ""),
            highways=list(out.get("highways") or []) if isinstance(out.get("highways"), list) else [],
            multi_segment=bool(out.get("multi_segment")),
            waypoints=way_list,
            source=str(out.get("source") or ""),
            avoid_ferry=bool(out.get("avoid_ferry")),
        )
        return self._attach_cities_along_route_to_result(out)

    def _attach_cities_along_route_to_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """沿折线采样逆地理编码，得到途经城市（地级市/区县）名称，供途经天气等使用。"""
        out = dict(result)
        pts = out.get("route_points")
        if not isinstance(pts, list) or len(pts) < 2:
            return out
        try:
            cities = self._sample_cities_along_route_polyline(
                pts,
                str(out.get("origin", "")).strip(),
                str(out.get("destination", "")).strip(),
            )
            if cities:
                out["cities_along_route"] = cities
        except Exception:
            pass
        return out

    def _city_labels_fuzzy_equal(self, a: str, b: str) -> bool:
        ca = CustomerServiceAgent._clean_place(str(a or "").strip())
        cb = CustomerServiceAgent._clean_place(str(b or "").strip())
        if not ca or not cb:
            return False
        if ca == cb:
            return True
        if len(ca) >= 2 and (ca in cb or cb in ca):
            return True
        return False

    def _sample_cities_along_route_polyline(
        self,
        route_points: List[List[float]],
        origin_name: str,
        dest_name: str,
        max_geo_calls: int = 12,
    ) -> List[str]:
        if len(route_points) < 2:
            return []
        cum_km = CustomerServiceAgent._route_vertex_cumulative_km(route_points)
        if not cum_km:
            return []
        total = float(cum_km[-1])
        nvert = len(route_points)
        if total < 5:
            fracs = [0.0, 0.55, 1.0]
        elif total < 45:
            fracs = [0.0, 0.18, 0.38, 0.58, 0.78, 1.0]
        elif total < 150:
            fracs = [0.0, 0.1, 0.22, 0.38, 0.52, 0.65, 0.78, 0.9, 1.0]
        else:
            k = min(max_geo_calls, max(9, int(total / 60) + 6))
            fracs = [i / max(1, k - 1) for i in range(k)] if k > 1 else [0.0, 1.0]

        ordered_idx: List[int] = []
        last_i = -1
        for f in fracs:
            target_km = total * float(f)
            best_i = nvert - 1
            for i, ck in enumerate(cum_km):
                if float(ck) >= target_km:
                    best_i = i
                    break
            if best_i == last_i:
                continue
            ordered_idx.append(best_i)
            last_i = best_i

        ordered_idx = ordered_idx[:max_geo_calls]
        raw_names: List[str] = []
        for idx_i, vi in enumerate(ordered_idx):
            if idx_i > 0:
                time.sleep(0.09 if self.amap_api_key else 0.35)
            lat = float(route_points[vi][0])
            lon = float(route_points[vi][1])
            nm = self._reverse_geocode_short_name(lat, lon).strip()
            if nm and len(nm) >= 2 and nm not in ("中国", "中华人民共和国"):
                raw_names.append(CustomerServiceAgent._clean_place(nm))

        dedup: List[str] = []
        for nm in raw_names:
            if dedup and self._city_labels_fuzzy_equal(dedup[-1], nm):
                continue
            dedup.append(nm)

        o = CustomerServiceAgent._clean_place(str(origin_name).strip())
        d = CustomerServiceAgent._clean_place(str(dest_name).strip())
        if o and (not dedup or not self._city_labels_fuzzy_equal(o, dedup[0])):
            dedup.insert(0, o)
        if d and (not dedup or not self._city_labels_fuzzy_equal(d, dedup[-1])):
            dedup.append(d)

        out_names: List[str] = []
        for x in dedup:
            t = str(x).strip()
            if not t:
                continue
            if out_names and self._city_labels_fuzzy_equal(out_names[-1], t):
                continue
            out_names.append(t)
        if len(out_names) < 2 and o and d and o != d:
            return [o, d]
        return out_names

    def _query_route_plan(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        waypoints: List[str] | None = None,
    ) -> Dict[str, Any]:
        origin = CustomerServiceAgent._clean_place(str(origin or "").strip())
        destination = CustomerServiceAgent._clean_place(str(destination or "").strip())
        if waypoints is not None:
            stops = [
                CustomerServiceAgent._clean_place(str(x).strip())
                for x in waypoints
                if str(x).strip()
            ]
            if stops:
                places = [origin] + stops + [destination]
                if len(places) >= 3:
                    return self._query_route_plan_multi(places, mode)
        return self._query_route_plan_single_leg(origin, destination, mode)

    def _query_route_plan_multi(self, places: List[str], mode: str = "driving") -> Dict[str, Any]:
        legs: List[Dict[str, Any]] = []
        for i in range(len(places) - 1):
            legs.append(self._query_route_plan_single_leg(places[i], places[i + 1], mode))
        if any(not x.get("success") for x in legs):
            err = next((x.get("error", "unknown") for x in legs if not x.get("success")), "leg failed")
            return {"tool": "query_route_plan", "success": False, "error": f"多段路线其中一段失败：{err}"}
        total_km = round(sum(float(x.get("distance_km", 0) or 0) for x in legs), 1)
        total_min = max(1, int(sum(int(x.get("duration_min", 0) or 0) for x in legs)))
        merged_points: List[List[float]] = []
        for leg in legs:
            pts = leg.get("route_points", [])
            if not isinstance(pts, list):
                continue
            for p in pts:
                if not isinstance(p, list) or len(p) < 2:
                    continue
                if merged_points and abs(merged_points[-1][0] - p[0]) < 1e-5 and abs(merged_points[-1][1] - p[1]) < 1e-5:
                    continue
                merged_points.append([float(p[0]), float(p[1])])
        seen_hw: set[str] = set()
        merged_hw: List[str] = []
        for leg in legs:
            for h in leg.get("highways", []) or []:
                s = str(h).strip()
                if s and s not in seen_hw:
                    seen_hw.add(s)
                    merged_hw.append(s)
        # 多段路线：汇总各段高速与里程；出行提示见 trip_hints（不再推荐服务区 POI）。
        leg_details: List[Dict[str, Any]] = []
        for i, leg in enumerate(legs):
            o, dest = places[i], places[i + 1]
            hw_leg = [str(x).strip() for x in (leg.get("highways") or []) if str(x).strip()][:8]
            leg_details.append(
                {
                    "from": o,
                    "to": dest,
                    "distance_km": leg.get("distance_km"),
                    "duration_min": leg.get("duration_min"),
                    "highways": hw_leg,
                    "service_areas": [],
                }
            )
        leg_lines = [
            f"{places[i]} → {places[i + 1]}：约 {legs[i].get('distance_km', '?')} 公里，约 {legs[i].get('duration_min', '?')} 分钟"
            for i in range(len(legs))
        ]
        via = places[1:-1]
        summary = (
            f"多段驾车路线（经停：{'、'.join(via)}），各段已分别规划；建议出发前用导航按途经点复核"
            if via
            else "多段驾车路线，各段已分别规划"
        )
        srcs = {str(x.get("source", "")) for x in legs}
        source = "amap_direction" if "amap_direction" in srcs else (next(iter(srcs)) if len(srcs) == 1 else "multi_segment")
        merged = {
            "tool": "query_route_plan",
            "success": True,
            "origin": places[0],
            "destination": places[-1],
            "waypoints": via,
            "multi_segment": True,
            "leg_summaries": leg_lines,
            "leg_details": leg_details,
            "mode": "driving" if mode not in {"walking", "cycling"} else mode,
            "distance_km": total_km,
            "duration_min": total_min,
            "summary": summary,
            "highways": merged_hw[:12],
            "service_areas": [],
            "route_points": merged_points,
            "source": source,
            "warning": "route_merged_from_legs",
        }
        return self._attach_trip_hints_to_route_result(merged)

    def _query_route_plan_single_leg(self, origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
        origin = CustomerServiceAgent._clean_place(str(origin or "").strip())
        destination = CustomerServiceAgent._clean_place(str(destination or "").strip())
        if not origin or not destination:
            return {"tool": "query_route_plan", "success": False, "error": "missing origin or destination"}
        try:
            a = self._geocode_place(self._normalize_place_name(origin)) or self._lookup_builtin_coord(origin)
            b = self._geocode_place(self._normalize_place_name(destination)) or self._lookup_builtin_coord(destination)
            if not a or not b:
                return {
                    "tool": "query_route_plan",
                    "success": False,
                    "error": "geocode failed for origin or destination",
                }
            # Prefer AMap direction API for domestic route quality.
            amap_result = self._query_route_plan_by_amap(origin=origin, destination=destination, a=a, b=b)
            if amap_result.get("success"):
                return amap_result

            # Fallback to OSRM if AMap API unavailable.
            profile = "driving" if mode not in {"walking", "cycling"} else mode
            url = (
                "https://router.project-osrm.org/route/v1/"
                f"{profile}/{a['lon']},{a['lat']};{b['lon']},{b['lat']}"
                "?overview=full&geometries=geojson&alternatives=false&steps=true"
            )
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            routes = data.get("routes", [])
            if not routes:
                return self._fallback_route_estimate(origin, destination, a["lat"], a["lon"], b["lat"], b["lon"])
            route = routes[0]
            distance_km = round(float(route.get("distance", 0)) / 1000, 1)
            duration_min = max(1, int(round(float(route.get("duration", 0)) / 60)))
            summary = "主干道优先，注意高峰拥堵与限行策略"
            geometry = route.get("geometry", {}) if isinstance(route, dict) else {}
            coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
            route_points = self._normalize_route_points(coords)

            extracted_highways = self._extract_highways_from_route(route)
            fallback_highways, _ = self._build_route_annotations(origin, destination, distance_km, duration_min)
            highways = extracted_highways or fallback_highways
            warning = "route_source=osrm_fallback"
            raw = {
                "tool": "query_route_plan",
                "success": True,
                "origin": origin,
                "destination": destination,
                "mode": profile,
                "distance_km": distance_km,
                "duration_min": duration_min,
                "summary": summary,
                "highways": highways,
                "service_areas": [],
                "route_points": route_points,
                "source": "osrm",
                "warning": warning,
            }
            return self._attach_trip_hints_to_route_result(raw)
        except Exception as e:
            # Network-restricted environments can still provide an approximate route.
            a = self._lookup_builtin_coord(origin)
            b = self._lookup_builtin_coord(destination)
            if a and b:
                fb = self._fallback_route_estimate(origin, destination, a["lat"], a["lon"], b["lat"], b["lon"])
                fb["warning"] = f"osrm unavailable: {e}"
                return fb
            return {"tool": "query_route_plan", "success": False, "error": str(e)}

    @staticmethod
    def _geocode_place_nominatim_query(q: str) -> Dict[str, Any] | None:
        if not (q or "").strip():
            return None
        try:
            url = "https://nominatim.openstreetmap.org/search"
            headers = {"User-Agent": "SmartTransportCSAgent/1.1 (domestic-route-dev)"}
            params = {"q": q.strip(), "format": "jsonv2", "limit": 1}
            resp = requests.get(url, params=params, headers=headers, timeout=12)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return None
            row = rows[0]
            return {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "display_name": row.get("display_name", q),
            }
        except Exception:
            return None

    def _geocode_place(self, name: str) -> Dict[str, Any] | None:
        # Prefer AMap geocode for Chinese place names.
        amap_hit = self._geocode_place_by_amap(name)
        if amap_hit is not None:
            return amap_hit
        n = (name or "").strip()
        if not n:
            return None
        candidates: List[str] = []
        for base in (n, self._normalize_place_name(n)):
            b = (base or "").strip()
            if not b:
                continue
            for q in (
                b,
                f"{b}, China",
                f"{b}, 中国",
                f"{b}市, China",
                f"{b}市, 中国",
            ):
                if q not in candidates:
                    candidates.append(q)
        for q in candidates:
            hit = CustomerServiceAgent._geocode_place_nominatim_query(q)
            if hit:
                return hit
        return None

    def _geocode_place_by_amap(self, name: str) -> Dict[str, Any] | None:
        if not self.amap_api_key:
            return None
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {
                "key": self.amap_api_key,
                "address": name,
            }
            resp = self._http_get(url, params=params, timeout=10, bypass_proxy=self.amap_bypass_proxy)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return None
            geocodes = payload.get("geocodes", [])
            if not isinstance(geocodes, list) or not geocodes:
                return None
            location = str(geocodes[0].get("location", "")).strip()
            if "," not in location:
                return None
            lon_s, lat_s = location.split(",", 1)
            return {"lat": float(lat_s), "lon": float(lon_s), "display_name": name}
        except Exception:
            return None

    def _amap_geocode_adcode(self, name: str) -> str | None:
        if not self.amap_api_key:
            return None
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {"key": self.amap_api_key, "address": name}
            resp = self._http_get(url, params=params, timeout=10, bypass_proxy=self.amap_bypass_proxy)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return None
            geocodes = payload.get("geocodes", [])
            if not isinstance(geocodes, list) or not geocodes:
                return None
            ad = geocodes[0].get("adcode") if isinstance(geocodes[0], dict) else None
            if ad:
                return str(ad).strip()
        except Exception:
            pass
        return None

    def _query_route_plan_by_amap(
        self, origin: str, destination: str, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.amap_api_key:
            return {"tool": "query_route_plan", "success": False, "error": "AMAP_API_KEY is not configured"}
        try:
            url = "https://restapi.amap.com/v3/direction/driving"
            params = {
                "key": self.amap_api_key,
                "origin": f"{a['lon']},{a['lat']}",
                "destination": f"{b['lon']},{b['lat']}",
                "extensions": "all",
                "strategy": "0",
                # 0=可使用轮渡（polyline 常在海上画直线，易被误认为「路线画进海里」）；1=不走轮渡，陆路连续，便于与地图路网对照。
                "ferry": "1",
            }
            resp = self._http_get(url, params=params, timeout=15, bypass_proxy=self.amap_bypass_proxy)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return {
                    "tool": "query_route_plan",
                    "success": False,
                    "error": f"amap direction failed: {payload.get('info', 'unknown')}",
                }
            route = payload.get("route", {}) if isinstance(payload, dict) else {}
            paths = route.get("paths", []) if isinstance(route, dict) else []
            if not isinstance(paths, list) or not paths:
                return {"tool": "query_route_plan", "success": False, "error": "amap route empty"}
            path0 = paths[0] if isinstance(paths[0], dict) else {}
            distance_km = round(float(path0.get("distance", 0)) / 1000, 1)
            duration_min = max(1, int(round(float(path0.get("duration", 0)) / 60)))
            steps = path0.get("steps", [])
            route_points = self._decode_route_points_from_amap_steps(steps)
            if len(route_points) < 2:
                route_points = self._build_fallback_route_points(
                    float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]), count=24
                )
            extracted_highways = self._extract_highways_from_amap_steps(steps)
            fallback_highways, _ = self._build_route_annotations(origin, destination, distance_km, duration_min)
            highways = extracted_highways or fallback_highways
            summary = (
                "高德路径规划结果（已选「不走轮渡」，路线沿陆路绕行；里程/耗时可能与含渤海轮渡的方案不同）。"
                "建议结合实时路况动态调整出发时间。"
            )
            raw = {
                "tool": "query_route_plan",
                "success": True,
                "origin": origin,
                "destination": destination,
                "mode": "driving",
                "distance_km": distance_km,
                "duration_min": duration_min,
                "summary": summary,
                "highways": highways,
                "service_areas": [],
                "route_points": route_points,
                "source": "amap_direction",
                "warning": "",
                "avoid_ferry": True,
            }
            return self._attach_trip_hints_to_route_result(raw)
        except Exception as e:
            return {"tool": "query_route_plan", "success": False, "error": str(e)}

    @staticmethod
    def _decode_route_points_from_amap_steps(steps: Any) -> List[List[float]]:
        if not isinstance(steps, list):
            return []
        points: List[List[float]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            polyline = str(step.get("polyline", "")).strip()
            if not polyline:
                continue
            chunks = [x.strip() for x in polyline.split(";") if x.strip()]
            for c in chunks:
                if "," not in c:
                    continue
                lon_s, lat_s = c.split(",", 1)
                try:
                    lat = float(lat_s)
                    lon = float(lon_s)
                except Exception:
                    continue
                if points and abs(points[-1][0] - lat) < 1e-7 and abs(points[-1][1] - lon) < 1e-7:
                    continue
                points.append([lat, lon])
        return points

    def _extract_highways_from_amap_steps(self, steps: Any) -> List[str]:
        if not isinstance(steps, list):
            return []
        found: List[str] = []
        seen: set[str] = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            road = str(step.get("road", "")).strip()
            instruction = str(step.get("instruction", "")).strip()
            text = f"{road} {instruction}".strip()
            if not text:
                continue
            code = self._extract_highway_target(text)
            if code:
                name = self._highway_name_from_code(code)
                label = f"{code}{name}" if name != code else code
                if label not in seen:
                    seen.add(label)
                    found.append(label)
                continue
            # Keep short named highways even without explicit G/S code,
            # avoid swallowing full instruction sentences.
            if road and "高速" in road:
                name = road.strip()
                if len(name) <= 18 and name not in seen:
                    seen.add(name)
                    found.append(name)
        return found[:6]

    def _is_valid_service_area_poi_name(self, name: str) -> bool:
        n = str(name).strip()
        if not n or "服务区" not in n:
            return False
        if n in {"服务区", "高速服务区", "停车区", "停车服务区"}:
            return False
        if len(n) < 4:
            return False
        return True

    def _query_service_areas_by_amap(self, route_points: List[List[float]], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        沿路线按「里程比例」多点采样，每段最多取 1～2 个 POI，避免长距离路线只在某一截搜满 5 条。
        """
        if not self.amap_api_key or len(route_points) < 3:
            return []
        # 从起点到终点均匀布点；长路线略加密
        cum_km = self._route_vertex_cumulative_km(route_points)
        route_len_km = cum_km[-1] if cum_km else 0.0
        if route_len_km > 1600:
            n_bands = 10
        elif route_len_km > 900:
            n_bands = 9
        elif route_len_km > 650:
            n_bands = 8
        elif route_len_km > 400:
            n_bands = 7
        else:
            n_bands = 6
        fracs = [(i + 1) / (n_bands + 1) for i in range(n_bands)]
        sample_idx: List[int] = []
        for f in fracs:
            sample_idx.append(int(round((len(route_points) - 1) * f)))
        # 路线前段额外加密采样，避免首段（尤其北方稀疏路段）候选过少
        lead_fracs: List[float] = []
        if route_len_km > 350:
            lead_fracs = [0.015, 0.03, 0.05, 0.08, 0.12]
        # 后段加密：避免长路段只在 0～30% 里程内有 POI 命中
        tail_fracs: List[float] = []
        if route_len_km > 520:
            tail_fracs = [0.68, 0.76, 0.84, 0.91, 0.97]
        seen_i: set[int] = set()
        ordered_idx: List[int] = []
        n_pts = len(route_points) - 1
        for f in lead_fracs:
            idx = min(max(0, int(round(n_pts * f))), len(route_points) - 1)
            if idx not in seen_i:
                seen_i.add(idx)
                ordered_idx.append(idx)
        for idx in sample_idx:
            idx = min(max(0, idx), len(route_points) - 1)
            if idx not in seen_i:
                seen_i.add(idx)
                ordered_idx.append(idx)
        for f in tail_fracs:
            idx = min(max(0, int(round(n_pts * f))), len(route_points) - 1)
            if idx not in seen_i:
                seen_i.add(idx)
                ordered_idx.append(idx)
        # 超长路线在中段再加密，减少「两省之间一大段没有 POI 命中」导致的推荐空白。
        if route_len_km > 1100:
            for f in (0.32, 0.38, 0.44, 0.52, 0.60):
                idx = min(max(0, int(round(n_pts * f))), len(route_points) - 1)
                if idx not in seen_i:
                    seen_i.add(idx)
                    ordered_idx.append(idx)
        results: List[Dict[str, Any]] = []
        seen_base: set[str] = set()
        max_collect = max(max_results * 3, 14)
        if route_len_km > 1300:
            max_collect = max(max_collect, 36)
        elif route_len_km > 1000:
            max_collect = max(max_collect, 28)
        elif route_len_km > 650:
            max_collect = max(max_collect, 20)
        try:
            for idx in ordered_idx:
                pt = route_points[idx]
                lat, lon = float(pt[0]), float(pt[1])
                url = "https://restapi.amap.com/v3/place/around"
                params = {
                    "key": self.amap_api_key,
                    "location": f"{lon},{lat}",
                    "keywords": "高速服务区",
                    "radius": "20000",
                    "offset": "10",
                    "page": "1",
                    "extensions": "base",
                }
                resp = self._http_get(url, params=params, timeout=12, bypass_proxy=self.amap_bypass_proxy)
                resp.raise_for_status()
                payload = resp.json()
                if str(payload.get("status", "0")) != "1":
                    params["keywords"] = "服务区"
                    resp = self._http_get(url, params=params, timeout=12, bypass_proxy=self.amap_bypass_proxy)
                    resp.raise_for_status()
                    payload = resp.json()
                if str(payload.get("status", "0")) != "1":
                    continue
                pois = payload.get("pois", [])
                if not isinstance(pois, list):
                    continue
                picked_here = 0
                for p in pois:
                    if not isinstance(p, dict):
                        continue
                    name = str(p.get("name", "")).strip()
                    if not self._is_valid_service_area_poi_name(name):
                        continue
                    base_key = self._service_area_dedup_key(name)
                    if base_key in seen_base:
                        continue
                    loc = str(p.get("location", "")).strip()
                    if "," not in loc:
                        continue
                    lon_s, lat_s = loc.split(",", 1)
                    try:
                        poi_lat = float(lat_s)
                        poi_lon = float(lon_s)
                    except Exception:
                        continue
                    seen_base.add(base_key)
                    results.append(
                        {
                            "name": name,
                            "lat": round(poi_lat, 6),
                            "lon": round(poi_lon, 6),
                            "facilities": ["卫生间", "餐饮", "加油站"],
                        }
                    )
                    picked_here += 1
                    if len(results) >= max_collect:
                        return results
                    if picked_here >= 1:
                        break
            return results
        except Exception:
            return []

    @staticmethod
    def _route_vertex_cumulative_km(route_points: List[List[float]]) -> List[float]:
        if not route_points:
            return []
        cum: List[float] = [0.0]
        for i in range(1, len(route_points)):
            la0, lo0 = float(route_points[i - 1][0]), float(route_points[i - 1][1])
            la1, lo1 = float(route_points[i][0]), float(route_points[i][1])
            cum.append(cum[-1] + CustomerServiceAgent._haversine_km(la0, lo0, la1, lo1))
        return cum

    @staticmethod
    def _snap_to_polyline_km(
        route_points: List[List[float]],
        cum_km: List[float],
        lat: float,
        lon: float,
    ) -> tuple[float, float]:
        """
        将 (lat,lon) 投影到折线各边上的垂足，返回（沿折线累积弧长 km, 到折线的球面距离 km）。
        若仅用「最近顶点」在长稀疏 polyline 上会把中部 POI 全判到终点附近，导致服务区推荐扎堆。
        """
        n = len(route_points)
        if n < 2 or len(cum_km) != n:
            return 0.0, 1e18
        best_off = 1e18
        best_along = 0.0
        for i in range(n - 1):
            la0, lo0 = float(route_points[i][0]), float(route_points[i][1])
            la1, lo1 = float(route_points[i + 1][0]), float(route_points[i + 1][1])
            seg_len = float(cum_km[i + 1]) - float(cum_km[i])
            if seg_len < 1e-9:
                off0 = CustomerServiceAgent._haversine_km(lat, lon, la0, lo0)
                if off0 < best_off:
                    best_off, best_along = off0, float(cum_km[i])
                continue
            ml = math.radians((la0 + la1) / 2.0)
            clat = math.cos(ml)
            px = math.radians(lon - lo0) * clat
            py = math.radians(lat - la0)
            sx = math.radians(lo1 - lo0) * clat
            sy = math.radians(la1 - la0)
            seg2 = sx * sx + sy * sy
            if seg2 < 1e-22:
                off0 = CustomerServiceAgent._haversine_km(lat, lon, la0, lo0)
                if off0 < best_off:
                    best_off, best_along = off0, float(cum_km[i])
                continue
            t = (px * sx + py * sy) / seg2
            t = max(0.0, min(1.0, t))
            cla = la0 + t * (la1 - la0)
            clo = lo0 + t * (lo1 - lo0)
            off = CustomerServiceAgent._haversine_km(lat, lon, cla, clo)
            along = float(cum_km[i]) + t * seg_len
            if off < best_off:
                best_off = off
                best_along = along
        return best_along, best_off

    @staticmethod
    def _distance_along_route_km(
        route_points: List[List[float]], cum_km: List[float], lat: float, lon: float
    ) -> float:
        if not route_points or not cum_km:
            return 0.0
        along, _off = CustomerServiceAgent._snap_to_polyline_km(route_points, cum_km, lat, lon)
        return along

    def _attach_route_progress_for_service_areas(
        self,
        service_areas: List[Dict[str, Any]],
        route_points: List[List[float]],
        total_distance_km: float,
        total_duration_min: int,
        max_pick: int | None = 5,
    ) -> List[Dict[str, Any]]:
        if not service_areas or not route_points:
            return []
        cum_km = self._route_vertex_cumulative_km(route_points)
        geom_len = float(cum_km[-1]) if cum_km else 0.0
        geom_len = max(geom_len, 1e-6)
        decl_km = max(float(total_distance_km), geom_len, 1.0)
        scale_odom = decl_km / geom_len
        normalized: List[Dict[str, Any]] = []
        for sa in service_areas:
            if not isinstance(sa, dict):
                continue
            if not isinstance(sa.get("lat"), (int, float)) or not isinstance(sa.get("lon"), (int, float)):
                continue
            lat_f, lon_f = float(sa["lat"]), float(sa["lon"])
            d_geom, _off = CustomerServiceAgent._snap_to_polyline_km(route_points, cum_km, lat_f, lon_f)
            d_along = d_geom * scale_odom
            progress = min(1.0, max(0.0, d_geom / geom_len))
            normalized.append(
                {
                    "name": str(sa.get("name", "服务区")).strip() or "服务区",
                    "distance_km_from_start": round(d_along, 1),
                    "eta_min_from_start": max(1, int(round(total_duration_min * progress))),
                    "facilities": sa.get("facilities", ["卫生间", "餐饮", "加油站"]),
                    "lat": round(lat_f, 6),
                    "lon": round(lon_f, 6),
                }
            )
        normalized.sort(key=lambda x: x["distance_km_from_start"])
        dedup: List[Dict[str, Any]] = []
        seen_full: set[str] = set()
        seen_base: set[str] = set()
        for item in normalized:
            n = str(item["name"])
            if not self._is_valid_service_area_poi_name(n):
                continue
            if n in seen_full:
                continue
            base_key = self._service_area_dedup_key(n)
            if base_key in seen_base:
                continue
            seen_full.add(n)
            seen_base.add(base_key)
            # 展示用短名：去掉「(某某方向)」等括号说明，避免同站多方向重复刷屏
            display = self._service_area_display_name(n)
            item = {**item, "name": display}
            dedup.append(item)
        if max_pick is None:
            return dedup
        return self._pick_service_areas_even_spaced(dedup, decl_km, max_pick)

    @staticmethod
    def _service_area_usable_corridor_km(
        leg_km: float, is_first_leg: bool = True, is_last_leg: bool = True
    ) -> float:
        lk = max(0.0, float(leg_km))
        cut_s = CustomerServiceAgent.SERVICE_AREA_EXCLUDE_START_KM if is_first_leg else 0.0
        cut_e = CustomerServiceAgent.SERVICE_AREA_EXCLUDE_END_KM if is_last_leg else 0.0
        return max(0.0, lk - cut_s - cut_e)

    @staticmethod
    def _service_area_slots_for_corridor_km(usable_km: float) -> int:
        """可推荐走廊内约每 250km（200～300 的中值）一处；走廊无效则为 0。"""
        u = max(0.0, float(usable_km))
        if u < 1.0:
            return 0
        n = int(round(u / CustomerServiceAgent.SERVICE_AREA_MIDDLE_SPACING_KM))
        return max(0, min(CustomerServiceAgent.SERVICE_AREA_MAX_RECOMMEND, n))

    @staticmethod
    def _service_area_slots_for_leg_km(
        leg_km: float, is_first_leg: bool = True, is_last_leg: bool = True
    ) -> int:
        usable = CustomerServiceAgent._service_area_usable_corridor_km(
            leg_km, is_first_leg=is_first_leg, is_last_leg=is_last_leg
        )
        return CustomerServiceAgent._service_area_slots_for_corridor_km(usable)

    @staticmethod
    def _leg_cumulative_km_on_geometry(
        legs: List[Dict[str, Any]], route_points: List[List[float]]
    ) -> List[float]:
        """按各段声明里程占比，把分段边界映射到合并几何累计长度（km）。"""
        cum_km = CustomerServiceAgent._route_vertex_cumulative_km(route_points)
        geom_len = float(cum_km[-1]) if cum_km else 0.0
        total_decl = sum(float(l.get("distance_km", 0) or 0) for l in legs)
        if geom_len <= 0 or total_decl <= 0:
            return [geom_len] if geom_len > 0 else []
        acc = 0.0
        bounds: List[float] = []
        for leg in legs:
            acc += float(leg.get("distance_km", 0) or 0)
            bounds.append(geom_len * (acc / total_decl))
        return bounds

    @staticmethod
    def _allocate_service_area_slots_by_leg_km(leg_kms: List[float], max_results: int) -> List[int]:
        """按各段里程占比分配推荐条数（最大余数法），长段多、短段少。"""
        n = len(leg_kms)
        if n <= 0 or max_results <= 0:
            return []
        w = [max(0.0, float(x)) for x in leg_kms]
        tw = sum(w) or 1.0
        exact = [max_results * (wi / tw) for wi in w]
        alloc = [int(e) for e in exact]
        rem = max_results - sum(alloc)
        order = sorted(range(n), key=lambda i: exact[i] - alloc[i], reverse=True)
        t = 0
        while rem > 0 and t < n * max_results + 10:
            alloc[order[t % n]] += 1
            rem -= 1
            t += 1
        if max_results >= n:
            for i in range(n):
                if alloc[i] < 1:
                    j = max(range(n), key=lambda k: alloc[k])
                    if alloc[j] > 1:
                        alloc[j] -= 1
                        alloc[i] = 1
        guard = 0
        while sum(alloc) > max_results and guard < n * max_results + 10:
            j = max(range(n), key=lambda k: (alloc[k], w[k]))
            if alloc[j] <= 1:
                break
            alloc[j] -= 1
            guard += 1
        guard = 0
        while sum(alloc) < max_results and guard < n * max_results + 10:
            j = max(range(n), key=lambda k: w[k] / max(float(alloc[k]), 1.0))
            alloc[j] += 1
            guard += 1
        return alloc

    def _pick_service_areas_proportional_by_legs(
        self,
        items: List[Dict[str, Any]],
        segment_end_km: List[float],
        leg_km_weights: List[float],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """多段路线：按路段长度比例分配名额，每段内在该段里程区间内均匀选点，避免全堆在首段或尾段。"""
        if not items:
            return []
        ref = max(
            float(segment_end_km[-1]) if segment_end_km else 0.0,
            max(float(x["distance_km_from_start"]) for x in items),
            1.0,
        )
        n = len(segment_end_km)
        if n < 2 or len(leg_km_weights) != n:
            return self._pick_service_areas_even_spaced(items, ref, max_results)
        alloc = CustomerServiceAgent._allocate_service_area_slots_by_leg_km(leg_km_weights, max_results)
        picked: List[Dict[str, Any]] = []
        prev = 0.0
        for i in range(n):
            end = float(segment_end_km[i])
            need = alloc[i] if i < len(alloc) else 0
            if need <= 0:
                prev = end
                continue
            seg_items = [
                x
                for x in items
                if prev - 0.5 <= float(x["distance_km_from_start"]) <= end + 0.5
            ]
            if not seg_items:
                prev = end
                continue
            seg_len = max(end - prev, 1.0)
            local: List[Dict[str, Any]] = []
            for x in seg_items:
                loc_d = max(0.0, float(x["distance_km_from_start"]) - prev)
                local.append({**x, "distance_km_from_start": loc_d})
            take = min(need, len(seg_items))
            sub = self._pick_service_areas_even_spaced(local, seg_len, take)
            for p in sub:
                gd = round(float(p["distance_km_from_start"]) + prev, 1)
                picked.append({**p, "distance_km_from_start": gd})
            prev = end
        used_keys = {self._service_area_dedup_key(str(x.get("name", ""))) for x in picked}
        pool = [x for x in items if self._service_area_dedup_key(str(x.get("name", ""))) not in used_keys]
        prev = 0.0
        for i in range(n):
            end = float(segment_end_km[i])
            need = alloc[i] if i < len(alloc) else 0
            got = sum(
                1
                for p in picked
                if prev - 0.5 <= float(p["distance_km_from_start"]) <= end + 0.5
            )
            deficit = max(0, need - got)
            mid = (prev + end) / 2.0
            for _ in range(deficit):
                if not pool:
                    break
                best = min(pool, key=lambda z: abs(float(z["distance_km_from_start"]) - mid))
                k = self._service_area_dedup_key(str(best.get("name", "")))
                used_keys.add(k)
                picked.append(dict(best))
                pool = [x for x in pool if self._service_area_dedup_key(str(x.get("name", ""))) not in used_keys]
            prev = end
        if len(picked) < max_results:
            pool2 = [
                x
                for x in items
                if self._service_area_dedup_key(str(x.get("name", "")))
                not in {self._service_area_dedup_key(str(z.get("name", ""))) for z in picked}
            ]
            extra = self._pick_service_areas_even_spaced(pool2, ref, max_results - len(picked))
            picked.extend(extra)
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        for x in sorted(picked, key=lambda z: float(z["distance_km_from_start"])):
            k = self._service_area_dedup_key(str(x.get("name", "")))
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
            if len(out) >= max_results:
                break
        return out[:max_results]

    @staticmethod
    def _dedupe_service_areas_skip_close_on_route(
        items: List[Dict[str, Any]], min_delta_km: float
    ) -> List[Dict[str, Any]]:
        """沿路线里程排序后，过近的两点只保留先出现的一条（避免涪陵西/涪陵双推）。"""
        if not items or min_delta_km <= 0:
            return items
        sorted_items = sorted(items, key=lambda x: float(x["distance_km_from_start"]))
        out: List[Dict[str, Any]] = []
        for it in sorted_items:
            d = float(it["distance_km_from_start"])
            if not out:
                out.append(it)
                continue
            if d - float(out[-1]["distance_km_from_start"]) >= min_delta_km:
                out.append(it)
        return out

    def _pick_service_areas_uniform_bins(
        self,
        items: List[Dict[str, Any]],
        corridor_lo: float,
        corridor_max_km: float,
        n_bins: int,
    ) -> List[Dict[str, Any]]:
        """
        将可推荐走廊 [lo, hi] 均分为 n_bins 段，每段选 1 个服务区（离该段中心里程最近），
        保证推荐点沿程大致等间距；避免原先「全体抢最近目标点 + 空隙补点」导致扎堆或忽近忽远。
        """
        if n_bins <= 0 or not items:
            return []
        lo = float(corridor_lo)
        hi = float(corridor_max_km)
        span = hi - lo
        if span < 1.0:
            return []
        in_corridor = [
            dict(x)
            for x in items
            if isinstance(x, dict)
            and lo - 0.5 <= float(x.get("distance_km_from_start", 0) or 0) <= hi + 0.5
        ]
        if not in_corridor:
            return []
        in_corridor.sort(key=lambda z: float(z.get("distance_km_from_start", 0) or 0))
        deduped: List[Dict[str, Any]] = []
        seen_k: set[str] = set()
        for x in in_corridor:
            k = self._service_area_dedup_key(str(x.get("name", "")))
            if k in seen_k:
                continue
            seen_k.add(k)
            deduped.append(x)
        # 相邻推荐期望间距约 span/n；回退选点时至少隔开一半桶宽，减少两桶都回退到同一簇。
        # 仅在对「全走廊回退」选点时使用，避免相邻两桶都回退到同一 POI 簇；本桶内不施加，以免误伤桶边界附近的合法点。
        min_sep = max(90.0, min(240.0, span / (2.0 * float(n_bins))))
        bin_w = span / float(n_bins)
        used_keys: set[str] = set()
        picked_dists: List[float] = []
        picked: List[Dict[str, Any]] = []

        def pool_eligible(
            candidates: List[Dict[str, Any]],
            center: float,
            *,
            enforce_sep: bool,
            relax_sep: bool,
        ) -> List[Dict[str, Any]]:
            ms = min_sep if (enforce_sep and not relax_sep) else 0.0
            out: List[Dict[str, Any]] = []
            for x in candidates:
                k = self._service_area_dedup_key(str(x.get("name", "")))
                if k in used_keys:
                    continue
                d = float(x.get("distance_km_from_start", 0) or 0)
                if ms > 0 and picked_dists:
                    if any(abs(d - pd) < ms for pd in picked_dists):
                        continue
                out.append(x)
            if not out and enforce_sep and not relax_sep:
                return pool_eligible(candidates, center, enforce_sep=True, relax_sep=True)
            return out

        for i in range(n_bins):
            b0 = lo + i * bin_w
            b1 = hi if i == n_bins - 1 else lo + (i + 1) * bin_w
            # 锚点略靠本桶前段（约 35% 桶宽），避免「离中点最近」总选桶内偏后的服务区，造成与前一点的间距过大。
            center = b0 + 0.35 * (b1 - b0)
            if i == n_bins - 1:
                in_bin = [
                    x
                    for x in deduped
                    if self._service_area_dedup_key(str(x.get("name", ""))) not in used_keys
                    and b0 - 1e-6 <= float(x.get("distance_km_from_start", 0) or 0) <= hi + 0.5
                ]
            else:
                in_bin = [
                    x
                    for x in deduped
                    if self._service_area_dedup_key(str(x.get("name", ""))) not in used_keys
                    and b0 <= float(x.get("distance_km_from_start", 0) or 0) < b1
                ]
            pool = pool_eligible(in_bin, center, enforce_sep=False, relax_sep=False)
            if not pool:
                pool = pool_eligible(
                    [
                        x
                        for x in deduped
                        if self._service_area_dedup_key(str(x.get("name", ""))) not in used_keys
                    ],
                    center,
                    enforce_sep=True,
                    relax_sep=False,
                )
            if not pool:
                break
            best = min(pool, key=lambda z: abs(float(z.get("distance_km_from_start", 0) or 0) - center))
            k = self._service_area_dedup_key(str(best.get("name", "")))
            used_keys.add(k)
            d_chosen = float(best.get("distance_km_from_start", 0) or 0)
            picked_dists.append(d_chosen)
            picked.append(dict(best))
        picked.sort(key=lambda z: float(z["distance_km_from_start"]))
        return [
            {**z, "distance_km_from_start": round(float(z["distance_km_from_start"]), 1)} for z in picked
        ]

    def _pick_service_areas_in_corridor(
        self,
        items: List[Dict[str, Any]],
        corridor_min_km: float,
        corridor_max_km: float,
        max_results: int,
        min_inter_pick_km: float | None = None,
    ) -> List[Dict[str, Any]]:
        """仅在 [corridor_min_km, corridor_max_km] 内，按里程均分桶各选 1 处，间距最均匀。"""
        _ = min_inter_pick_km  # 保留参数兼容旧调用；间距由分桶保证
        if max_results <= 0 or not items:
            return []
        return self._pick_service_areas_uniform_bins(
            items, corridor_min_km, corridor_max_km, max_results
        )

    def _pick_service_areas_even_spaced(
        self,
        items: List[Dict[str, Any]],
        reference_total_km: float,
        max_results: int,
        min_inter_pick_km: float | None = None,
    ) -> List[Dict[str, Any]]:
        """在整条路线上近似均匀选取若干服务区；可选 min_inter_pick_km 强制点与点之间沿路线间隔。"""
        if not items:
            return []
        ref = max(float(reference_total_km), max(float(x["distance_km_from_start"]) for x in items), 1.0)
        targets = [ref * (i + 1) / (max_results + 1) for i in range(max_results)]
        sorted_items = sorted(items, key=lambda x: float(x["distance_km_from_start"]))
        used: set[int] = set()
        picked: List[Dict[str, Any]] = []
        for t in targets:
            if len(picked) >= max_results:
                break
            chosen_idx: int | None = None
            for relax in (False, True):
                try_idx: int | None = None
                best_diff = 1e18
                for i, it in enumerate(sorted_items):
                    if i in used:
                        continue
                    d = float(it["distance_km_from_start"])
                    if not relax and min_inter_pick_km and picked:
                        if any(
                            abs(d - float(p["distance_km_from_start"])) < min_inter_pick_km
                            for p in picked
                        ):
                            continue
                    diff = abs(d - t)
                    if diff < best_diff:
                        best_diff = diff
                        try_idx = i
                if try_idx is not None:
                    chosen_idx = try_idx
                    break
            if chosen_idx is not None:
                used.add(chosen_idx)
                picked.append(sorted_items[chosen_idx])

        while len(picked) < max_results:
            best_i: int | None = None
            best_score = -1.0
            for i, it in enumerate(sorted_items):
                if i in used:
                    continue
                d = float(it["distance_km_from_start"])
                if min_inter_pick_km and picked:
                    md = min(abs(d - float(p["distance_km_from_start"])) for p in picked)
                    if md < min_inter_pick_km * 0.70:
                        continue
                    score = md
                elif picked:
                    score = min(abs(d - float(p["distance_km_from_start"])) for p in picked)
                else:
                    score = d
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                for i, it in enumerate(sorted_items):
                    if i not in used:
                        best_i = i
                        break
            if best_i is None:
                break
            used.add(best_i)
            picked.append(sorted_items[best_i])
        return sorted(picked, key=lambda x: float(x["distance_km_from_start"]))[:max_results]

    def _cap_service_areas_even_along_total_route(
        self,
        items: List[Dict[str, Any]],
        total_km: float,
        cap: int,
    ) -> List[Dict[str, Any]]:
        """按全程累计里程去重后，若超过 cap 条则在走廊上均匀抽样（用于多段合并与展示上限）。"""
        if not items or cap <= 0:
            return []
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for x in sorted(items, key=lambda z: float(z.get("distance_km_from_start", 0) or 0)):
            if not isinstance(x, dict):
                continue
            k = CustomerServiceAgent._service_area_dedup_key(str(x.get("name", "")).strip())
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(x)
        if len(deduped) <= cap:
            return deduped
        ref = max(
            float(total_km),
            max(float(z.get("distance_km_from_start", 0) or 0) for z in deduped),
            1.0,
        )
        min_inter = max(55.0, ref / float(cap + 1))
        return self._pick_service_areas_even_spaced(deduped, ref, cap, min_inter_pick_km=min_inter)

    def _assign_merged_service_areas_to_leg_details(
        self,
        merged: List[Dict[str, Any]],
        legs: List[Dict[str, Any]],
        places: List[str],
        leg_details: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """把全程合并后的服务区（累计里程）按路段拆回各 leg，卡片内里程/时间为相对本段起点。"""
        bounds: List[tuple[float, float]] = []
        acc = 0.0
        for leg in legs:
            lk = float(leg.get("distance_km", 0) or 0)
            bounds.append((acc, acc + lk))
            acc += lk
        per_leg: List[List[Dict[str, Any]]] = [[] for _ in legs]
        if not merged:
            return [
                {**{k: v for k, v in d.items() if k != "service_areas"}, "service_areas": []}
                for d in leg_details
            ]
        for sa in merged:
            if not isinstance(sa, dict):
                continue
            d = float(sa.get("distance_km_from_start", 0) or 0)
            best_i = 0
            found = False
            for i, (a, b) in enumerate(bounds):
                if a - 0.5 <= d <= b + 0.5:
                    best_i = i
                    found = True
                    break
            if not found:
                best_i = min(
                    range(len(bounds)),
                    key=lambda i: min(abs(d - bounds[i][0]), abs(d - bounds[i][1])),
                )
            a, _b = bounds[best_i]
            leg_k = float(legs[best_i].get("distance_km", 0) or 0) or 1.0
            leg_min = int(legs[best_i].get("duration_min", 0) or 0)
            local_d = round(max(0.0, d - a), 1)
            local_eta = max(1, int(round(leg_min * min(1.0, local_d / max(leg_k, 1e-6)))))
            row = dict(sa)
            row["distance_km_from_start"] = local_d
            row["eta_min_from_start"] = local_eta
            row["leg_from"] = places[best_i]
            row["leg_to"] = places[best_i + 1]
            row["km_on_leg"] = local_d
            per_leg[best_i].append(row)
        out: List[Dict[str, Any]] = []
        for i, det in enumerate(leg_details):
            block = {k: v for k, v in det.items() if k != "service_areas"}
            sas = sorted(per_leg[i], key=lambda z: float(z.get("distance_km_from_start", 0) or 0))
            block["service_areas"] = sas
            out.append(block)
        return out

    @staticmethod
    def _service_area_min_spacing_km(leg_km: float, n_picks: int) -> float:
        """相邻推荐点沿路线期望间距，约束在约 200～300km（随本段长度与处数略调）。"""
        lk = max(float(leg_km), 1.0)
        n = max(1, int(n_picks))
        step = lk / float(n)
        return max(200.0, min(300.0, step))

    def _finalize_leg_service_areas(
        self,
        items: List[Dict[str, Any]],
        leg_km: float,
        is_first_leg: bool = True,
        is_last_leg: bool = True,
    ) -> List[Dict[str, Any]]:
        """本段：去掉起点 150km、终点前 100km 内点位，再在中间走廊按约 200～300km 间隔抽样。"""
        if not items:
            return []
        lk = max(float(leg_km), 1.0)
        min_d = self.SERVICE_AREA_EXCLUDE_START_KM if is_first_leg else 0.0
        max_d = lk - self.SERVICE_AREA_EXCLUDE_END_KM if is_last_leg else lk
        # 首/尾短段时「150km 内 + 终点前 100km」可能把走廊挤没；改为路段中段仍可推荐
        if max_d <= min_d + 1.0:
            pad = max(1.0, min(lk * 0.12, lk / 2.0 - 1.0))
            if lk > 2.0 * pad + 2.0:
                min_d = pad
                max_d = lk - pad
            else:
                min_d = max(0.0, lk * 0.15)
                max_d = min(lk, lk * 0.85)
        if max_d <= min_d + 1.0:
            min_d, max_d = 0.0, lk
        usable = max_d - min_d
        n_pick = CustomerServiceAgent._service_area_slots_for_corridor_km(usable)
        # 短段按里程算出来 0 槽，但已有 POI/候选时仍给至少 1 条，避免某一段「空白」
        if n_pick <= 0 and items and lk >= 2.5:
            n_pick = 1
        if n_pick <= 0:
            return []
        cluster_gap = max(35.0, min(52.0, lk * 0.038))
        thinned = CustomerServiceAgent._dedupe_service_areas_skip_close_on_route(items, cluster_gap)
        corridor_items = [
            x
            for x in thinned
            if min_d - 0.5 <= float(x.get("distance_km_from_start", 0) or 0) <= max_d + 0.5
        ]
        if not corridor_items:
            return []
        min_inter = CustomerServiceAgent._service_area_min_spacing_km(usable, n_pick) * 0.72
        return self._pick_service_areas_in_corridor(
            corridor_items,
            min_d,
            max_d,
            n_pick,
            min_inter_pick_km=max(200.0, min_inter),
        )

    def _finalize_single_route_service_areas(
        self,
        candidates: List[Dict[str, Any]],
        route_points: List[List[float]],
        distance_km: float,
        duration_min: int,
    ) -> List[Dict[str, Any]]:
        """单段全程：起点 150km、终点前 100km 内不推荐；中间走廊按约每 250km 一处均匀抽样。"""
        if not candidates or not route_points or len(route_points) < 2:
            return []
        total = max(float(distance_km), 1.0)
        cmin = self.SERVICE_AREA_EXCLUDE_START_KM
        cmax = total - self.SERVICE_AREA_EXCLUDE_END_KM
        if cmax <= cmin + 1.0:
            return []
        usable = cmax - cmin
        n_pick = CustomerServiceAgent._service_area_slots_for_corridor_km(usable)
        if n_pick <= 0:
            return []
        full = self._attach_route_progress_for_service_areas(
            candidates, route_points, distance_km, duration_min, max_pick=None
        )
        if not full:
            return []
        cluster_gap = max(35.0, min(52.0, total * 0.038))
        thinned = CustomerServiceAgent._dedupe_service_areas_skip_close_on_route(full, cluster_gap)
        min_inter = CustomerServiceAgent._service_area_min_spacing_km(usable, n_pick) * 0.72
        return self._pick_service_areas_in_corridor(
            thinned,
            cmin,
            cmax,
            n_pick,
            min_inter_pick_km=max(200.0, min_inter),
        )

    def _merge_service_areas_from_legs(
        self,
        legs: List[Dict[str, Any]],
        total_km: float,
        total_min: int,
        max_results: int = 6,
    ) -> List[Dict[str, Any]]:
        """多段路线：把各段服务区里程/时间平移到全程坐标，再均匀抽样，避免结果全堆在第一段。"""
        offset_km = 0.0
        offset_min = 0
        collected: List[Dict[str, Any]] = []
        for leg in legs:
            las = leg.get("service_areas") or []
            if not isinstance(las, list):
                las = []
            for sa in las:
                if not isinstance(sa, dict):
                    continue
                lat, lon = sa.get("lat"), sa.get("lon")
                if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                    continue
                name = str(sa.get("name", "")).strip()
                if not self._is_valid_service_area_poi_name(name):
                    continue
                dist = float(sa.get("distance_km_from_start", 0) or 0) + offset_km
                eta = max(1, int(sa.get("eta_min_from_start", 0) or 0) + offset_min)
                display = self._service_area_display_name(name)
                collected.append(
                    {
                        "name": display,
                        "distance_km_from_start": round(dist, 1),
                        "eta_min_from_start": eta,
                        "facilities": sa.get("facilities", ["卫生间", "餐饮", "加油站"]),
                        "lat": round(float(lat), 6),
                        "lon": round(float(lon), 6),
                    }
                )
            offset_km += float(leg.get("distance_km", 0) or 0)
            offset_min += int(leg.get("duration_min", 0) or 0)
        collected.sort(key=lambda x: float(x["distance_km_from_start"]))
        dedup: List[Dict[str, Any]] = []
        seen_base: set[str] = set()
        for item in collected:
            base_key = self._service_area_dedup_key(str(item["name"]))
            if base_key in seen_base:
                continue
            seen_base.add(base_key)
            dedup.append(item)
        total_f = max(float(total_km), 1.0)
        cmin = CustomerServiceAgent.SERVICE_AREA_EXCLUDE_START_KM
        cmax = total_f - CustomerServiceAgent.SERVICE_AREA_EXCLUDE_END_KM
        usable = max(0.0, cmax - cmin)
        n_pick = min(
            max_results,
            CustomerServiceAgent._service_area_slots_for_corridor_km(usable),
        )
        if n_pick <= 0:
            return []
        min_inter = CustomerServiceAgent._service_area_min_spacing_km(usable, n_pick) * 0.72
        return self._pick_service_areas_in_corridor(
            dedup,
            cmin,
            cmax,
            n_pick,
            min_inter_pick_km=max(200.0, min_inter),
        )

    @staticmethod
    def _service_area_dedup_key(name: str) -> str:
        """同一路侧服务区常带不同方向后缀，按主体名去重。"""
        n = str(name).strip()
        if "(" in n:
            return n.split("(", 1)[0].strip().lower()
        return n.lower()

    @staticmethod
    def _service_area_display_name(name: str) -> str:
        n = str(name).strip()
        if "(" in n and n.endswith(")"):
            return n.split("(", 1)[0].strip() or n
        return n

    def _calculate_fare(self, origin: str, destination: str, mode: str = "metro") -> Dict[str, Any]:
        if not origin or not destination:
            return {"tool": "calculate_fare", "success": False, "error": "missing origin or destination"}
        base = 2 if mode == "metro" else 3
        delta = max(0, min(4, abs(len(origin) - len(destination))))
        fare = base + delta
        return {"tool": "calculate_fare", "success": True, "origin": origin, "destination": destination, "mode": mode, "fare": fare}

    def _create_transport_ticket(self, issue_type: str, detail: str) -> Dict[str, Any]:
        ticket_id = f"TR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return {
            "tool": "create_transport_ticket",
            "success": True,
            "ticket_id": ticket_id,
            "issue_type": issue_type,
            "detail_preview": detail[:100],
        }

    def _handoff_to_human(self, priority: str) -> Dict[str, Any]:
        case_id = f"HF-{datetime.now().strftime('%H%M%S')}"
        return {"tool": "handoff_to_human", "success": True, "case_id": case_id, "priority": priority}

    @staticmethod
    def _extract_transit_target(text: str) -> str:
        m = re.search(r"(地铁\d+号线|S\d+高架|机场快线)", text, flags=re.IGNORECASE)
        if not m:
            return ""
        value = m.group(0)
        value = re.sub(r"^s", "S", value, flags=re.IGNORECASE)
        return value

    @staticmethod
    def _extract_highway_target(text: str) -> str:
        upper = text.upper()
        # Do not use \b here because Chinese chars are treated as \w in Unicode mode.
        m = re.search(r"(?<![A-Z0-9])([GS]\d{1,4})(?![A-Z0-9])", upper)
        if m:
            return m.group(1)
        for name in ["京沪高速", "沈海高速", "长深高速", "沪陕高速", "S1高架"]:
            if name in text:
                return name
        return ""

    @staticmethod
    def _is_affirmative_message(message: str) -> bool:
        t = message.strip().lower()
        if not t:
            return False
        affirmatives = {
            "可以",
            "好的",
            "好",
            "行",
            "继续",
            "嗯",
            "是的",
            "对",
            "ok",
            "okay",
            "yes",
            "y",
        }
        return t in affirmatives

    def _extract_last_highway_codes_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if not isinstance(tool_results, list):
                continue
            codes: List[str] = []
            seen: set[str] = set()
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_highway_condition" or not tr.get("success"):
                    continue
                code = str(tr.get("code", "")).strip().upper()
                if code and code not in seen:
                    seen.add(code)
                    codes.append(code)
            if codes:
                return codes
        return []

    @staticmethod
    def _normalize_highway_code(target: str) -> str:
        t = target.strip().upper()
        if re.fullmatch(r"[GS]\d{1,4}", t):
            return t
        name_map = {
            "京沪高速": "G2",
            "京哈高速": "G1",
            "绥满高速": "G10",
            "沈海高速": "G15",
            "长深高速": "G25",
            "连霍高速": "G30",
            "沪蓉高速": "G42",
            "沪陕高速": "G40",
            "大广高速": "G45",
            "S1高架": "S1高架",
        }
        return name_map.get(target.strip(), "")

    @staticmethod
    def _highway_name_from_code(code: str) -> str:
        code_map = {
            "G1": "京哈高速",
            "G2": "京沪高速",
            "G10": "绥满高速",
            "G15": "沈海高速",
            "G25": "长深高速",
            "G30": "连霍高速",
            "G42": "沪蓉高速",
            "G40": "沪陕高速",
            "G45": "大广高速",
            "S1高架": "S1高架",
        }
        return code_map.get(code, code)

    @staticmethod
    def _extract_multi_stop_places(text: str) -> List[str] | None:
        """Parse chains like 兰州到成都再到宜昌 or 兰州到成都到宜昌 → ordered place names."""
        raw = text.strip()
        raw = re.sub(r"(的路线|路线|怎么走|如何走|怎么去|出行方案)$", "", raw)
        raw = re.sub(r"^从\s*", "", raw)
        if re.search(r"再到|然后\s*到|接着\s*到", raw):
            temp = re.sub(r"\s*(?:再到|然后\s*到|接着\s*到)\s*", "▸", raw)
            chunks = [c.strip() for c in temp.split("▸") if c.strip()]
            if len(chunks) < 2:
                return None
            m = re.match(r"(.+?)到(.+)$", chunks[0].strip())
            if not m:
                return None
            a = CustomerServiceAgent._clean_place(m.group(1))
            b = CustomerServiceAgent._clean_place(m.group(2))
            if not a or not b:
                return None
            out: List[str] = [a, b]
            for c in chunks[1:]:
                cc = CustomerServiceAgent._clean_place(c.strip())
                if not cc:
                    return None
                out.append(cc)
            return out if len(out) >= 3 else None
        if raw.count("到") >= 2:
            parts = re.split(r"\s*到\s*", raw)
            cleaned: List[str] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                cc = CustomerServiceAgent._clean_place(p)
                if cc:
                    cleaned.append(cc)
            if len(cleaned) >= 3:
                return cleaned
        return None

    # 起终点正则用「非空白」连续匹配时，会把「贵阳沿途高速路况…」整段吃进终点，导致 geocode 失败；在路线/路况用语前截断。
    # 含「的路段/拥堵」等，避免「北京到上海的路段拥堵吗」把终点吃成「上海的路段拥堵吗」导致 geocode 失败（意图识别往往是对的，坏在起终点抽取）。
    _ROUTE_CONTEXT_IN_PLACE_TOKEN = re.compile(
        r"(?:沿途|经停|路过|高速公路|高速路|高速|国道|省道|快速路|"
        r"的路段|路段|拥堵|堵不堵|塞车|通畅|顺畅|"
        r"帮我|帮|选条|选一条|选一种|选个|最快的路|最快的|最快|"
        r"推荐一下|推荐|规划一下|规划|路径|方案|走哪条|哪条快|"
        r"路况|交通情况|通行情况|交通状况|道路情况|怎么样|怎么走|怎么去|如何走|如何|导航|自驾|开车|驾车|吗)"
    )

    @staticmethod
    def _trim_route_context_suffix_from_place_token(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        m = CustomerServiceAgent._ROUTE_CONTEXT_IN_PLACE_TOKEN.search(s)
        if m and m.start() > 0:
            return s[: m.start()].strip()
        return s

    @staticmethod
    def _extract_route_endpoints(text: str) -> tuple[str, str]:
        m = re.search(r"从\s*([^\s，。；,.]+?)\s*到\s*([^\s，。；,.]+)", text)
        if m:
            o = CustomerServiceAgent._trim_route_context_suffix_from_place_token(m.group(1))
            d = CustomerServiceAgent._trim_route_context_suffix_from_place_token(m.group(2))
            return CustomerServiceAgent._clean_place(o), CustomerServiceAgent._clean_place(d)
        m2 = re.search(r"([^\s，。；,.]+)\s*到\s*([^\s，。；,.]+)", text)
        if m2:
            o = CustomerServiceAgent._trim_route_context_suffix_from_place_token(m2.group(1))
            d = CustomerServiceAgent._trim_route_context_suffix_from_place_token(m2.group(2))
            return CustomerServiceAgent._clean_place(o), CustomerServiceAgent._clean_place(d)
        return "", ""

    @staticmethod
    def _normalize_place_name(name: str) -> str:
        mapping = {
            "北京": "Beijing, China",
            "北京南站": "Beijing South Railway Station, China",
            "北京站": "Beijing Railway Station, China",
            "南通": "Nantong, Jiangsu, China",
            "南通站": "Nantong Railway Station, Jiangsu, China",
            "上海": "Shanghai, China",
            "南京": "Nanjing, China",
            "宁": "Nanjing, Jiangsu, China",
            "广州": "Guangzhou, China",
            "深圳": "Shenzhen, China",
            "宜昌": "Yichang, Hubei, China",
            "宜昌市": "Yichang, Hubei, China",
            "盐城": "Yancheng, Jiangsu, China",
            "盐城市": "Yancheng, Jiangsu, China",
            "武汉": "Wuhan, Hubei, China",
            "杭州": "Hangzhou, Zhejiang, China",
            "成都": "Chengdu, Sichuan, China",
            "重庆": "Chongqing, China",
            "西安": "Xi'an, Shaanxi, China",
            "郑州": "Zhengzhou, Henan, China",
            "长沙": "Changsha, Hunan, China",
            "合肥": "Hefei, Anhui, China",
            "南昌": "Nanchang, Jiangxi, China",
            "福州": "Fuzhou, Fujian, China",
            "厦门": "Xiamen, Fujian, China",
            "青岛": "Qingdao, Shandong, China",
            "济南": "Jinan, Shandong, China",
            "石家庄": "Shijiazhuang, Hebei, China",
            "太原": "Taiyuan, Shanxi, China",
            "沈阳": "Shenyang, Liaoning, China",
            "长春": "Changchun, Jilin, China",
            "哈尔滨": "Harbin, Heilongjiang, China",
            "昆明": "Kunming, Yunnan, China",
            "贵阳": "Guiyang, Guizhou, China",
            "南宁": "Nanning, Guangxi, China",
            "海口": "Haikou, Hainan, China",
            "兰州": "Lanzhou, Gansu, China",
            "乌鲁木齐": "Urumqi, Xinjiang, China",
            "银川": "Yinchuan, Ningxia, China",
            "西宁": "Xining, Qinghai, China",
            "拉萨": "Lhasa, Tibet, China",
            "呼和浩特": "Hohhot, Inner Mongolia, China",
            "秦皇岛": "Qinhuangdao, Hebei, China",
            "秦皇岛市": "Qinhuangdao, Hebei, China",
        }
        n = name.strip()
        return mapping.get(n, n)

    @staticmethod
    def _expand_place_alias(name: str) -> str:
        """口语/简称 → 规划用地名（如「宁」在「宁到昆明」里常指南京）。"""
        n = (name or "").strip()
        if not n:
            return n
        aliases = {
            "宁": "南京",
            "京": "北京",
            "沪": "上海",
            "穗": "广州",
            "蓉": "成都",
            "渝": "重庆",
            "哈": "哈尔滨",
            "沈": "沈阳",
            "杭": "杭州",
            "甬": "宁波",
            "鹏": "深圳",
            "邕": "南宁",
        }
        return aliases.get(n, n)

    @staticmethod
    def _clean_place(name: str) -> str:
        n = name.strip()
        # Remove common route-query suffixes.
        n = re.sub(r"(的路线|路线|怎么走|如何走|怎么去|出行方案)$", "", n)
        n = re.sub(r"(?:的路况|路况|交通情况|通行情况|交通状况|道路情况)$", "", n)
        n = re.sub(r"(?:的路段|路段)(?:拥堵|通畅|堵不堵|怎么样|如何|吗)?$", "", n)
        # 「秦皇岛帮我选条最快的路」类：截断正则未覆盖时的兜底
        n = re.sub(r"(?:帮我|帮).*$", "", n)
        n = re.sub(r"(?:选条|选一条|选一种).*$", "", n)
        n = re.sub(r"(?:最快的路|最快路|最快的|最快)$", "", n)
        n = re.sub(r"^(去|到)", "", n)
        n = n.strip()
        return CustomerServiceAgent._expand_place_alias(n)

    @staticmethod
    def _lookup_builtin_coord(name: str) -> Dict[str, Any] | None:
        coords = {
            "北京": (39.9042, 116.4074),
            "北京南站": (39.8652, 116.3785),
            "北京站": (39.9022, 116.4273),
            "南通": (31.9802, 120.8943),
            "南通站": (32.0647, 120.8717),
            "上海": (31.2304, 121.4737),
            "南京": (32.0603, 118.7969),
            "宁": (32.0603, 118.7969),
            "广州": (23.1291, 113.2644),
            "深圳": (22.5431, 114.0579),
            "宜昌": (30.6974, 111.2810),
            "宜昌市": (30.6974, 111.2810),
            "盐城": (33.3478, 120.1628),
            "盐城市": (33.3478, 120.1628),
            "武汉": (30.5928, 114.3055),
            "武汉市": (30.5928, 114.3055),
            "杭州": (30.2741, 120.1551),
            "杭州市": (30.2741, 120.1551),
            "成都": (30.5728, 104.0668),
            "成都市": (30.5728, 104.0668),
            "重庆": (29.5630, 106.5516),
            "重庆市": (29.5630, 106.5516),
            "西安": (34.3416, 108.9398),
            "西安市": (34.3416, 108.9398),
            "郑州": (34.7466, 113.6254),
            "郑州市": (34.7466, 113.6254),
            "长沙": (28.2280, 112.9388),
            "长沙市": (28.2280, 112.9388),
            "合肥": (31.8206, 117.2272),
            "合肥市": (31.8206, 117.2272),
            "南昌": (28.6820, 115.8579),
            "南昌市": (28.6820, 115.8579),
            "福州": (26.0745, 119.2965),
            "福州市": (26.0745, 119.2965),
            "厦门": (24.4798, 118.0819),
            "厦门市": (24.4798, 118.0819),
            "青岛": (36.0671, 120.3826),
            "青岛市": (36.0671, 120.3826),
            "济南": (36.6512, 117.1201),
            "济南市": (36.6512, 117.1201),
            "石家庄": (38.0428, 114.5149),
            "石家庄市": (38.0428, 114.5149),
            "太原": (37.8706, 112.5489),
            "太原市": (37.8706, 112.5489),
            "沈阳": (41.8057, 123.4315),
            "沈阳市": (41.8057, 123.4315),
            "长春": (43.8171, 125.3235),
            "长春市": (43.8171, 125.3235),
            "哈尔滨": (45.8038, 126.5350),
            "哈尔滨市": (45.8038, 126.5350),
            "昆明": (25.0389, 102.7183),
            "昆明市": (25.0389, 102.7183),
            "贵阳": (26.6470, 106.6302),
            "贵阳市": (26.6470, 106.6302),
            "南宁": (22.8170, 108.3669),
            "南宁市": (22.8170, 108.3669),
            "海口": (20.0440, 110.1999),
            "海口市": (20.0440, 110.1999),
            "兰州": (36.0611, 103.8343),
            "兰州市": (36.0611, 103.8343),
            "乌鲁木齐": (43.8256, 87.6168),
            "乌鲁木齐市": (43.8256, 87.6168),
            "银川": (38.4872, 106.2309),
            "银川市": (38.4872, 106.2309),
            "西宁": (36.6171, 101.7782),
            "西宁市": (36.6171, 101.7782),
            "拉萨": (29.6500, 91.1000),
            "呼和浩特市": (40.8414, 111.7519),
            "呼和浩特": (40.8414, 111.7519),
            "秦皇岛": (39.9354, 119.6003),
            "秦皇岛市": (39.9354, 119.6003),
        }
        n = name.strip()
        if n not in coords:
            return None
        lat, lon = coords[n]
        return {"lat": lat, "lon": lon, "display_name": n}

    def _fallback_route_estimate(
        self,
        origin: str,
        destination: str,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> Dict[str, Any]:
        # Straight-line distance * factor ~= coarse road distance
        km = CustomerServiceAgent._haversine_km(lat1, lon1, lat2, lon2)
        distance_km = round(km * 1.25, 1)
        duration_min = max(1, int(round(distance_km / 80 * 60)))
        highways, _ = CustomerServiceAgent._build_route_annotations(
            origin, destination, distance_km, duration_min
        )
        route_points = CustomerServiceAgent._build_fallback_route_points(lat1, lon1, lat2, lon2, count=24)
        raw = {
            "tool": "query_route_plan",
            "success": True,
            "origin": origin,
            "destination": destination,
            "mode": "driving",
            "distance_km": distance_km,
            "duration_min": duration_min,
            "summary": "按城市间道路距离估算，建议出发前再次确认实时路况",
            "highways": highways,
            "service_areas": [],
            "route_points": route_points,
            "source": "fallback_estimation",
        }
        return self._attach_trip_hints_to_route_result(raw)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    @staticmethod
    def _build_route_annotations(
        origin: str, destination: str, distance_km: float, duration_min: int
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        o = CustomerServiceAgent._clean_place(origin)
        d = CustomerServiceAgent._clean_place(destination)
        key = f"{o}->{d}"
        rev_key = f"{d}->{o}"
        corridor = {
            "南通->北京": {
                "highways": ["G15沈海高速", "G2京沪高速", "G1京哈高速（进京段联络）"],
                "service_areas": ["如皋服务区", "沭阳服务区", "临沂服务区", "德州服务区", "廊坊服务区"],
            },
            "上海->北京": {
                "highways": ["G2京沪高速", "G42沪蓉高速（联络段）"],
                "service_areas": ["昆山服务区", "淮安服务区", "济南服务区", "沧州服务区"],
            },
            "南京->海安": {
                "highways": ["G40沪陕高速", "S28启扬高速（联络段）"],
                "service_areas": ["六合服务区", "江都服务区", "如皋服务区"],
            },
            "广州->深圳": {
                "highways": ["G4京港澳高速", "S31龙大高速（联络）"],
                "service_areas": ["虎门服务区", "光明服务区"],
            },
        }
        profile = corridor.get(key) or corridor.get(rev_key) or {
            "highways": ["国家高速主通道", "省级联络高速"],
            "service_areas": CustomerServiceAgent._build_generic_service_area_names(o, d),
        }
        highways: List[str] = list(profile["highways"])
        names: List[str] = list(profile["service_areas"])
        slots = min(len(names), 5)
        service_areas: List[Dict[str, Any]] = []
        for i in range(slots):
            frac = (i + 1) / (slots + 1)
            km = round(distance_km * frac, 1)
            eta = max(1, int(round(duration_min * frac)))
            service_areas.append(
                {
                    "name": names[i],
                    "distance_km_from_start": km,
                    "eta_min_from_start": eta,
                    "facilities": ["卫生间", "餐饮", "充电桩", "加油站"],
                }
            )
        return highways, service_areas

    @staticmethod
    def _build_generic_service_area_names(origin: str, destination: str) -> List[str]:
        o = CustomerServiceAgent._clean_place(origin) or "起点"
        d = CustomerServiceAgent._clean_place(destination) or "终点"
        return [
            f"{o}北服务区",
            f"{o}-{d}中途服务区",
            f"{d}南服务区",
        ]

    def _extract_highways_from_route(self, route: Dict[str, Any]) -> List[str]:
        legs = route.get("legs", []) if isinstance(route, dict) else []
        if not isinstance(legs, list):
            return []
        found: List[str] = []
        seen: set[str] = set()
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            steps = leg.get("steps", [])
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                candidates = [
                    str(step.get("name", "")).strip(),
                    str(step.get("ref", "")).strip(),
                ]
                for c in candidates:
                    if not c:
                        continue
                    code = self._extract_highway_target(c)
                    if code:
                        name = self._highway_name_from_code(code)
                        label = f"{code}{name}" if name != code else code
                        if label not in seen:
                            seen.add(label)
                            found.append(label)
                        continue
                    normalized = self._normalize_highway_code(c)
                    if normalized:
                        name = self._highway_name_from_code(normalized)
                        label = f"{normalized}{name}" if name != normalized else normalized
                        if label not in seen:
                            seen.add(label)
                            found.append(label)
        return found[:5]

    def _reverse_geocode_short_name(self, lat: float, lon: float) -> str:
        if self.amap_api_key:
            try:
                url = "https://restapi.amap.com/v3/geocode/regeo"
                params = {
                    "key": self.amap_api_key,
                    "location": f"{lon},{lat}",
                    "extensions": "base",
                }
                resp = self._http_get(url, params=params, timeout=10, bypass_proxy=self.amap_bypass_proxy)
                resp.raise_for_status()
                payload = resp.json()
                if str(payload.get("status", "0")) == "1":
                    regeocode = payload.get("regeocode", {}) if isinstance(payload, dict) else {}
                    comp = regeocode.get("addressComponent", {}) if isinstance(regeocode, dict) else {}
                    if isinstance(comp, dict):
                        for key in ["city", "district", "township", "province"]:
                            value = comp.get(key, "")
                            if isinstance(value, list):
                                value = value[0] if value else ""
                            v = str(value).strip()
                            if v:
                                return re.sub(r"(市|区|县|自治州|地区)$", "", v).strip()
            except Exception:
                pass
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            headers = {"User-Agent": "smart-transport-agent/1.0"}
            params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            addr = data.get("address", {}) if isinstance(data, dict) else {}
            if isinstance(addr, dict):
                for key in ["city", "town", "county", "state_district", "state"]:
                    v = str(addr.get(key, "")).strip()
                    if v:
                        v = re.sub(r"(市|区|县|自治州|地区)$", "", v).strip()
                        return v
            dn = str(data.get("display_name", "")).strip() if isinstance(data, dict) else ""
            if dn:
                return dn.split(",")[0].strip()
            return ""
        except Exception:
            return ""

    def _infer_service_areas_by_reverse_geocode(
        self,
        route_points: List[List[float]],
        total_distance_km: float,
        total_duration_min: int,
        origin: str,
        destination: str,
    ) -> List[Dict[str, Any]]:
        if len(route_points) < 2:
            return []
        fractions = [0.22, 0.5, 0.78]
        seen_names: set[str] = set()
        results: List[Dict[str, Any]] = []
        for frac in fractions:
            idx = min(len(route_points) - 1, max(0, int(round((len(route_points) - 1) * frac))))
            lat = float(route_points[idx][0])
            lon = float(route_points[idx][1])
            short = self._reverse_geocode_short_name(lat, lon)
            if not short:
                continue
            name = f"{short}服务区"
            if name in seen_names:
                continue
            seen_names.add(name)
            results.append(
                {
                    "name": name,
                    "distance_km_from_start": round(total_distance_km * frac, 1),
                    "eta_min_from_start": max(1, int(round(total_duration_min * frac))),
                    "facilities": ["卫生间", "餐饮", "加油站"],
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                }
            )
        if results:
            return results
        # 最后兜底：仍给具体可读名称，而不是东/中/西。
        generic_names = self._build_generic_service_area_names(origin, destination)
        fallback: List[Dict[str, Any]] = []
        for i, n in enumerate(generic_names, start=1):
            frac = i / (len(generic_names) + 1)
            idx = min(len(route_points) - 1, max(0, int(round((len(route_points) - 1) * frac))))
            lat = float(route_points[idx][0])
            lon = float(route_points[idx][1])
            fallback.append(
                {
                    "name": n,
                    "distance_km_from_start": round(total_distance_km * frac, 1),
                    "eta_min_from_start": max(1, int(round(total_duration_min * frac))),
                    "facilities": ["卫生间", "餐饮", "加油站"],
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                }
            )
        return fallback

    @staticmethod
    def _normalize_route_points(coords: Any) -> List[List[float]]:
        points: List[List[float]] = []
        if not isinstance(coords, list):
            return points
        for item in coords:
            if not isinstance(item, list) or len(item) < 2:
                continue
            lon = item[0]
            lat = item[1]
            if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                continue
            points.append([float(lat), float(lon)])
        return points

    @staticmethod
    def _build_fallback_route_points(lat1: float, lon1: float, lat2: float, lon2: float, count: int = 24) -> List[List[float]]:
        n = max(2, count)
        points: List[List[float]] = []
        for i in range(n):
            frac = i / (n - 1)
            lat = lat1 + (lat2 - lat1) * frac
            lon = lon1 + (lon2 - lon1) * frac
            points.append([round(lat, 6), round(lon, 6)])
        return points

    def _query_service_areas_by_overpass(
        self,
        route_points: List[List[float]],
        total_distance_km: float,
        total_duration_min: int,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if len(route_points) < 2:
            return []
        try:
            lats = [p[0] for p in route_points]
            lons = [p[1] for p in route_points]
            min_lat = min(lats) - 0.08
            max_lat = max(lats) + 0.08
            min_lon = min(lons) - 0.08
            max_lon = max(lons) + 0.08
            query = f"""
[out:json][timeout:20];
(
  node["highway"="services"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["highway"="services"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["highway"="services"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out center tags;
""".strip()
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query.encode("utf-8"),
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            elements = payload.get("elements", [])
            if not isinstance(elements, list):
                return []
            cum_route = CustomerServiceAgent._route_vertex_cumulative_km(route_points)
            geom_len = float(cum_route[-1]) if cum_route else 1.0
            geom_len = max(geom_len, 1e-6)
            scale_od = float(total_distance_km) / geom_len
            scored: List[Dict[str, Any]] = []
            for elem in elements:
                if not isinstance(elem, dict):
                    continue
                tags = elem.get("tags", {}) if isinstance(elem.get("tags"), dict) else {}
                name = str(tags.get("name") or tags.get("name:zh") or "").strip()
                lat: float | None = None
                lon: float | None = None
                if isinstance(elem.get("lat"), (int, float)) and isinstance(elem.get("lon"), (int, float)):
                    lat = float(elem["lat"])
                    lon = float(elem["lon"])
                else:
                    center = elem.get("center", {})
                    if isinstance(center, dict) and isinstance(center.get("lat"), (int, float)) and isinstance(
                        center.get("lon"), (int, float)
                    ):
                        lat = float(center["lat"])
                        lon = float(center["lon"])
                if lat is None or lon is None:
                    continue
                along_g, nearest_km = CustomerServiceAgent._snap_to_polyline_km(
                    route_points, cum_route, lat, lon
                )
                if nearest_km > 10:
                    continue
                if not name:
                    name = f"服务区({elem.get('id', 'unknown')})"
                frac = along_g / geom_len
                dist_km = along_g * scale_od
                scored.append(
                    {
                        "name": name,
                        "distance_km_from_start": round(dist_km, 1),
                        "eta_min_from_start": max(1, int(round(total_duration_min * frac))),
                        "facilities": ["卫生间", "餐饮", "加油站"],
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "_near_km": nearest_km,
                    }
                )
            if not scored:
                return []
            scored.sort(key=lambda x: (x["distance_km_from_start"], x["_near_km"]))
            final_items: List[Dict[str, Any]] = []
            seen_names: set[str] = set()
            for item in scored:
                n = str(item["name"]).strip()
                if n in seen_names:
                    continue
                seen_names.add(n)
                clean_item = {
                    "name": item["name"],
                    "distance_km_from_start": item["distance_km_from_start"],
                    "eta_min_from_start": item["eta_min_from_start"],
                    "facilities": item["facilities"],
                    "lat": item["lat"],
                    "lon": item["lon"],
                }
                final_items.append(clean_item)
                if len(final_items) >= max_results:
                    break
            return final_items
        except Exception:
            return []

    @staticmethod
    def _nearest_point_on_route(route_points: List[List[float]], lat: float, lon: float) -> tuple[int, float]:
        best_idx = 0
        best_dist = 1e18
        for idx, pt in enumerate(route_points):
            d = CustomerServiceAgent._haversine_km(lat, lon, float(pt[0]), float(pt[1]))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx, best_dist

    def _extract_last_target_from_history(self, history: List[Dict[str, Any]]) -> str:
        for item in reversed(history):
            tgt = self._extract_transit_target(str(item.get("content", "")))
            if tgt:
                return tgt
        return ""

    def _extract_last_highway_from_history(self, history: List[Dict[str, Any]]) -> str:
        candidates: List[str] = []
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if isinstance(tool_results, list):
                for tr in tool_results:
                    if not isinstance(tr, dict):
                        continue
                    if tr.get("tool") != "query_route_plan" or not tr.get("success"):
                        continue
                    highs = tr.get("highways", [])
                    if not isinstance(highs, list):
                        continue
                    for h in highs:
                        ht = self._extract_highway_target(str(h))
                        if ht:
                            candidates.append(ht)
                        normalized = self._normalize_highway_code(str(h).strip())
                        if normalized:
                            candidates.append(normalized)
            content = str(item.get("content", ""))
            # 从回复文本里提取 "途经高速：..." 后的第一个可识别高速
            m = re.search(r"途经高速[:：]\s*([^\n]+)", content)
            if m:
                segment = m.group(1)
                for part in re.split(r"[、，,；;]", segment):
                    p = part.strip()
                    if not p:
                        continue
                    ht = self._extract_highway_target(p)
                    if ht:
                        candidates.append(ht)
                    normalized = self._normalize_highway_code(p)
                    if normalized:
                        candidates.append(normalized)
        if not candidates:
            return ""
        # Prefer national G-series highways for stable realtime coverage.
        dedup: List[str] = []
        seen: set[str] = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        dedup.sort(key=lambda x: (0 if str(x).upper().startswith("G") else 1, len(str(x))))
        return dedup[0]

    def _extract_last_route_highways_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if not isinstance(tool_results, list):
                continue
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_route_plan" or not tr.get("success"):
                    continue
                highs = tr.get("highways", [])
                if not isinstance(highs, list):
                    continue
                out: List[str] = []
                seen: set[str] = set()
                for h in highs:
                    s = str(h).strip()
                    if not s:
                        continue
                    code = self._extract_highway_target(s) or self._normalize_highway_code(s)
                    if code and code not in seen:
                        seen.add(code)
                        out.append(code)
                if out:
                    out.sort(key=lambda x: (0 if str(x).upper().startswith("G") else 1, len(str(x))))
                    return out
        return []

    @staticmethod
    def _extract_last_route_probe_points_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if not isinstance(tool_results, list):
                continue
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_route_plan" or not tr.get("success"):
                    continue
                pts = tr.get("route_points", [])
                out = CustomerServiceAgent._probe_points_from_route_points_list(pts)
                if out:
                    return out
        return []

    @staticmethod
    def _extract_recent_route_context(history: List[Dict[str, Any]]) -> str:
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if not isinstance(tool_results, list):
                continue
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_route_plan" or not tr.get("success"):
                    continue
                origin = str(tr.get("origin", "")).strip()
                destination = str(tr.get("destination", "")).strip()
                highs = tr.get("highways", [])
                highways: List[str] = []
                if isinstance(highs, list):
                    highways = [str(x).strip() for x in highs if str(x).strip()]
                th = tr.get("trip_hints", [])
                hint_titles: List[str] = []
                if isinstance(th, list):
                    for h in th:
                        if isinstance(h, dict):
                            t = str(h.get("title", "")).strip()
                            if t:
                                hint_titles.append(t)
                car = tr.get("cities_along_route", [])
                cities_line = ""
                if isinstance(car, list) and car:
                    cities_line = f"; cities_along_route={','.join(str(x).strip() for x in car if str(x).strip())}"
                return (
                    f"origin={origin or 'unknown'}; destination={destination or 'unknown'}; "
                    f"highways={','.join(highways) if highways else 'none'}; "
                    f"trip_hints={','.join(hint_titles) if hint_titles else 'none'}"
                    f"{cities_line}"
                )
        return "none"

    @staticmethod
    def _extract_last_route_cities_from_history(history: List[Dict[str, Any]]) -> List[str]:
        for item in reversed(history):
            meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
            tool_results = meta.get("tool_results", [])
            if not isinstance(tool_results, list):
                continue
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                if tr.get("tool") != "query_route_plan" or not tr.get("success"):
                    continue
                seq = CustomerServiceAgent._route_city_sequence(tr)
                if seq:
                    return seq
        return []

    @staticmethod
    def _history_to_text(history: List[Dict[str, Any]]) -> str:
        if not history:
            return "无"
        lines: List[str] = []
        for item in history[-6:]:
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "无"

    @staticmethod
    def _rag_to_text(rag_hits: List[Dict[str, Any]]) -> str:
        if not rag_hits:
            return "无"
        lines: List[str] = []
        for idx, hit in enumerate(rag_hits[:3], start=1):
            lines.append(f"{idx}. [{hit.get('title','')}] {hit.get('content','')}")
        return "\n".join(lines)

    def _build_llm(self) -> ChatOpenAI | None:
        if not self.api_key:
            return None
        return ChatOpenAI(
            model=self.model,
            api_key=cast(Any, self.api_key),
            base_url=self.base_url,
            temperature=0.1,
            timeout=20,
        )

    def _build_answer_chain(self) -> Any:
        if self.llm is None:
            return None
        prompt = ChatPromptTemplate.from_template(
            """
你是智慧交通客服助手。请严格基于给定知识片段回答，不要编造。
回答要求：
- 先给直接结论，再补充关键规则
- 1-3句，简洁
- 末尾标注“知识依据：<标题关键词>”

用户问题：
{question}

知识片段：
{context}
""".strip()
        )
        return prompt | self.llm | StrOutputParser()


app = FastAPI(title="Smart Transportation Service Agent", version="1.0.0")
agent = CustomerServiceAgent()
STATIC_DIR = Path(__file__).parent / "static"
VUE_DIST_DIR = Path(__file__).parent / "static-vue"
conversation_store = ConversationStore(Path(__file__).parent / "data" / "conversations.json")
auth_store = AuthStore()
SESSION_COOKIE = "st_session"

if VUE_DIST_DIR.exists():
    app.mount("/static-vue", StaticFiles(directory=VUE_DIST_DIR), name="static-vue")


def _set_session_cookie(resp: Response, token: str, expires_at: datetime) -> None:
    # 与 create_session 的 7 天一致；用固定秒数避免本机时间与 DB 比较时的边界误差导致 max_age=0
    max_age = 7 * 24 * 60 * 60
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def _scoped_conversation_id(username: str, conversation_id: str) -> str:
    cid = conversation_id.strip()[:80] if conversation_id else "default"
    if not cid:
        cid = "default"
    cid = re.sub(r"[^a-zA-Z0-9_\-:]", "_", cid)
    return f"{username}::{cid}"


def get_current_user(request: Request) -> AuthUser:
    token = request.cookies.get(SESSION_COOKIE, "").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    user = auth_store.resolve_user_by_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态已失效，请重新登录")
    return AuthUser(user_id=int(user["id"]), username=str(user["username"]))


@app.get("/")
def index(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE, "").strip()
    user = auth_store.resolve_user_by_token(token) if token else None
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    vue_entry = VUE_DIST_DIR / "index.html"
    if vue_entry.exists():
        return FileResponse(vue_entry)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
def login_page(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE, "").strip()
    user = auth_store.resolve_user_by_token(token) if token else None
    if user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    vue_entry = VUE_DIST_DIR / "index.html"
    if vue_entry.exists():
        return FileResponse(vue_entry)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, response: Response) -> AuthResponse:
    try:
        user = auth_store.register(payload.username, payload.password)
        token, expires_at = auth_store.create_session(int(user["id"]))
        _set_session_cookie(response, token, expires_at)
        return AuthResponse(ok=True, user=AuthUser(user_id=int(user["id"]), username=str(user["username"])))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response) -> AuthResponse:
    try:
        user = auth_store.login(payload.username, payload.password)
        token, expires_at = auth_store.create_session(int(user["id"]))
        _set_session_cookie(response, token, expires_at)
        return AuthResponse(ok=True, user=AuthUser(user_id=int(user["id"]), username=str(user["username"])))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.post("/auth/logout")
def logout(request: Request, response: Response) -> Dict[str, Any]:
    token = request.cookies.get(SESSION_COOKIE, "").strip()
    if token:
        auth_store.delete_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthResponse)
def auth_me(user: AuthUser = Depends(get_current_user)) -> AuthResponse:
    return AuthResponse(ok=True, user=user)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "llm_enabled": agent.llm_enabled, "model": agent.model}


@app.get("/debug/config")
def debug_config() -> Dict[str, Any]:
    """
    Safe config diagnostics endpoint.
    Returns only boolean/status info, never returns secrets.
    """
    return {
        "llm_api_configured": bool(agent.api_key),
        "amap_configured": bool(agent.amap_api_key),
        "llm_enabled": agent.llm_enabled,
        "model": agent.model,
        "mysql_enabled": auth_store.enabled,
        "mysql_database": auth_store.database if auth_store.enabled else "",
        "mysql_host": auth_store.host,
        "mysql_port": auth_store.port,
        "mysql_socket": auth_store.unix_socket,
        "mysql_init_error": auth_store.init_error,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: AuthUser = Depends(get_current_user)) -> ChatResponse:
    scoped_id = _scoped_conversation_id(user.username, payload.user_id)
    history = conversation_store.get(scoped_id, limit=10)
    result = agent.chat(payload.message, history=history)
    conversation_store.append(scoped_id, "user", payload.message)
    conversation_store.append(
        scoped_id,
        "agent",
        result.reply,
        meta={
            "intent": result.intent,
            "confidence": result.confidence,
            "used_llm": result.used_llm,
            "tool_results": result.tool_results,
            "follow_ups": [x.model_dump() for x in result.follow_ups],
        },
    )
    return result


@app.get("/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str, user: AuthUser = Depends(get_current_user)) -> HistoryResponse:
    scoped_id = _scoped_conversation_id(user.username, user_id)
    items = [HistoryItem(**it) for it in conversation_store.get(scoped_id)]
    return HistoryResponse(user_id=user_id, items=items)


@app.delete("/history/{user_id}")
def clear_history(user_id: str, user: AuthUser = Depends(get_current_user)) -> Dict[str, Any]:
    scoped_id = _scoped_conversation_id(user.username, user_id)
    conversation_store.clear(scoped_id)
    return {"status": "ok", "user_id": user_id}
