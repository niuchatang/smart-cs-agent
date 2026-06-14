"""tracking-system 接入适配层。

把外部独立的 tracking-system 引到本项目里使用，并提供安全包装：
  - 任何埋点异常都不会冒泡到主流程
  - 路径未找到 / 包未安装时静默降级
  - 全局单例，避免重复创建

业务侧（main.py）只需调用这里暴露的 4 个函数：
  - track_session_start(session_id, user_id)
  - track_session_end(session_id, user_id)  # 会关闭该用户下所有对话槽位的未完成 Task
  - track_turn(...)        # agent_task_* + agent_tool_call + agent_turn（Task=用户目标）
  - track_task_result(...) # 单独工具任务上报（可选）

层级：Session → 多个 Task（用户目标）→ 多个 Turn → 多个 Tool Call。
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_TRACKING_SYSTEM_PATH = "/Users/apple/Desktop/算法/tracking-system"
_TRACKING_SYSTEM_PATH = os.environ.get(
    "TRACKING_SYSTEM_PATH", _DEFAULT_TRACKING_SYSTEM_PATH
)

_RECORD_PAYLOAD = os.environ.get("TRACKING_RECORD_PAYLOAD", "1") not in ("0", "false", "False", "")
_MAX_TEXT_LEN = int(os.environ.get("TRACKING_MAX_TEXT_LEN", "2000"))
_MASK_PII = os.environ.get("TRACKING_MASK_PII", "1") not in ("0", "false", "False", "")

_client = None
_disabled = False

# session 键与 /chat 一致：``{username}::{conversation_id}``；登录埋点仍用 username。
_SESSION_TASK: Dict[str, Dict[str, Any]] = {}


def _load_smart_cs_dotenv() -> None:
    """确保能读到 ``smart-cs-agent/.env`` 里的 MYSQL_*（与 main.py 同目录）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


try:
    from safety.pii import mask_pii as _mask_pii_impl, pii_hits as _pii_hits_impl
except Exception:
    _mask_pii_impl = None
    _pii_hits_impl = None


def _mask(text: str) -> str:
    if not _MASK_PII or _mask_pii_impl is None:
        return text
    try:
        return _mask_pii_impl(text)
    except Exception:
        return text


def _pii_hits(text: str) -> Dict[str, int]:
    if not _MASK_PII or _pii_hits_impl is None:
        return {}
    try:
        return _pii_hits_impl(text) or {}
    except Exception:
        return {}


def _clip(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _mask(text)
    if len(text) <= _MAX_TEXT_LEN:
        return text
    return text[:_MAX_TEXT_LEN] + f"...[+{len(text) - _MAX_TEXT_LEN}]"


def _get_client():
    """懒加载 TrackingClient；首次失败后标记 disabled，后续直接返回 None。"""
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client

    path = Path(_TRACKING_SYSTEM_PATH)
    if not path.exists():
        _disabled = True
        return None

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

    _load_smart_cs_dotenv()

    try:
        from tracking import TrackingClient
        from tracking.sinks import JsonlFileSink, MultiSink
        from tracking.sinks_sqlite import SqliteSink

        events_file = path / "data" / "events.jsonl"
        csv_file = Path(
            os.environ.get("TRACKING_CSV_PATH", str(path / "data" / "events.csv"))
        )
        db_file = Path(
            os.environ.get(
                "TRACKING_DB_PATH", str(path / "data" / "tracking.db")
            )
        )

        storage = os.environ.get("TRACKING_STORAGE", "mysql").strip().lower()
        if storage == "sqlite":
            sink = SqliteSink(str(db_file))
        elif storage in ("both", "jsonl+sqlite", "dual"):
            sink = MultiSink(
                JsonlFileSink(str(events_file)),
                SqliteSink(str(db_file)),
            )
        elif storage == "csv":
            from tracking.sinks_csv import CsvFileSink

            sink = CsvFileSink(str(csv_file))
        elif storage in ("csv_jsonl", "csv+jsonl"):
            from tracking.sinks_csv import CsvFileSink

            sink = MultiSink(
                CsvFileSink(str(csv_file)),
                JsonlFileSink(str(events_file)),
            )
        elif storage in ("csv_mysql", "csv+mysql"):
            from tracking.sinks_csv import CsvFileSink

            try:
                from tracking.sinks_mysql import MySQLSink

                sink = MultiSink(CsvFileSink(str(csv_file)), MySQLSink())
            except ImportError:
                print("[tracking] csv+mysql 需要 pymysql，已仅写 CSV")
                sink = CsvFileSink(str(csv_file))
        elif storage == "mysql":
            try:
                from tracking.sinks_mysql import MySQLSink

                sink = MySQLSink()
            except ImportError:
                print(
                    "[tracking] TRACKING_STORAGE=mysql 需要 pymysql，"
                    "已回退为 jsonl。请执行: pip install pymysql"
                )
                sink = JsonlFileSink(str(events_file))
        elif storage in ("mysql_jsonl", "mysql+jsonl", "mysql_dual"):
            try:
                from tracking.sinks_mysql import MySQLSink

                sink = MultiSink(
                    JsonlFileSink(str(events_file)),
                    MySQLSink(),
                )
            except ImportError:
                print(
                    "[tracking] mysql+jsonl 需要 pymysql，"
                    "已仅使用 jsonl。请执行: pip install pymysql"
                )
                sink = JsonlFileSink(str(events_file))
        else:
            sink = JsonlFileSink(str(events_file))

        _client = TrackingClient(
            app_name="smart-cs-agent",
            env=os.environ.get("APP_ENV", "dev"),
            sink=sink,
            file_path=str(events_file),
        )
        return _client
    except Exception:
        _disabled = True
        return None


def _safe_call(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except Exception:
        return False


def _session_task_state(session_id: str) -> Dict[str, Any]:
    if session_id not in _SESSION_TASK:
        _SESSION_TASK[session_id] = {
            "open": None,
            "counters": {},
        }
    return _SESSION_TASK[session_id]


def _classify_task_type(intent: str, user_message: str) -> str:
    """将识别意图 + 文本线索映射为分析用 task_type（用户目标粒度）。"""
    msg = user_message or ""
    low = msg.lower()
    if any(k in msg for k in ("充电", "充电桩", "充电站", "换电站")):
        return "charging_station_query"
    if intent == "route_planning":
        return "route_planning"
    if intent == "highway_condition":
        route_hw = ("沿途", "一路", "全程", "各个高速", "哪几条", "路径上", "走高速", "各段")
        if any(h in msg for h in route_hw):
            return "highway_route_condition"
        return "single_highway_condition"
    if intent == "realtime_status":
        return "transit_realtime_status"
    if intent == "weather_query":
        return "weather_query"
    if intent == "fare_policy":
        return "fare_policy"
    if intent == "ticket_refund":
        return "ticket_refund"
    if intent in ("lost_and_found", "complaint", "human_handoff"):
        return "customer_support"
    if intent == "unknown" and low:
        return "unknown"
    return "unknown"


def _next_semantic_task_id(st: Dict[str, Any], task_type: str) -> str:
    ctr: Dict[str, int] = st["counters"]
    n = int(ctr.get(task_type, 0)) + 1
    ctr[task_type] = n
    return f"task_{task_type}_{n:03d}"


def _emit_agent_task_start(
    client,
    *,
    session_id: str,
    user_id: Optional[str],
    task_id: str,
    task_type: str,
    start_ms: int,
) -> None:
    props: Dict[str, Any] = {
        "session_id": session_id,
        "task_id": task_id,
        "task_type": task_type,
        "task_status": "running",
        "start_time_ms": start_ms,
        "end_time_ms": None,
        "duration_ms": 0,
        "turn_count": 0,
        "tool_call_count": 0,
        "success_flag": True,
        "event_schema": "agent_task_v1",
    }
    _safe_call(
        client.track,
        "agent_task_start",
        event_type="event",
        session_id=session_id,
        task_id=task_id,
        user_id=user_id,
        properties=props,
    )


def _emit_agent_task_end(
    client,
    *,
    session_id: str,
    user_id: Optional[str],
    open_task: Dict[str, Any],
    end_ms: int,
    reason: str,
) -> None:
    start_ms = int(open_task["start_ms"])
    duration = max(0, end_ms - start_ms)
    any_tf = bool(open_task.get("any_turn_error"))
    any_tool = bool(open_task.get("any_tool_fail"))
    success_flag = not any_tf and not any_tool
    task_status = "completed" if success_flag else "failed"
    props: Dict[str, Any] = {
        "session_id": session_id,
        "task_id": open_task["task_id"],
        "task_type": open_task["task_type"],
        "task_status": task_status,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "duration_ms": duration,
        "turn_count": int(open_task.get("turn_count", 0)),
        "tool_call_count": int(open_task.get("tool_call_count", 0)),
        "success_flag": success_flag,
        "close_reason": reason,
        "event_schema": "agent_task_v1",
    }
    _safe_call(
        client.track,
        "agent_task_end",
        event_type="event",
        session_id=session_id,
        task_id=open_task["task_id"],
        user_id=user_id,
        properties=props,
    )


def _finalize_open_task(
    client,
    session_id: str,
    user_id: Optional[str],
    *,
    reason: str,
) -> None:
    st = _SESSION_TASK.get(session_id)
    if not st:
        return
    op = st.get("open")
    if not op:
        return
    end_ms = int(time.time() * 1000)
    _emit_agent_task_end(
        client,
        session_id=session_id,
        user_id=user_id,
        open_task=op,
        end_ms=end_ms,
        reason=reason,
    )
    st["open"] = None


def _matching_session_keys_for_logout(session_id: str) -> List[str]:
    """logout 时 session_id 为 username，需关闭所有 ``username::*`` 对话上的 Task。"""
    keys = [k for k in list(_SESSION_TASK.keys()) if k == session_id or k.startswith(f"{session_id}::")]
    return keys


def _resolve_user_task(
    client,
    *,
    session_id: str,
    user_id: Optional[str],
    intent: str,
    user_message: str,
) -> Tuple[str, str]:
    """返回 (semantic_task_id, task_type)，并在需要时结束旧 Task、开启新 Task。"""
    task_type = _classify_task_type(intent, user_message)
    st = _session_task_state(session_id)
    op: Optional[Dict[str, Any]] = st.get("open")

    if op and op.get("task_type") == task_type:
        return str(op["task_id"]), task_type

    if op:
        _finalize_open_task(client, session_id, user_id, reason="task_type_change")

    start_ms = int(time.time() * 1000)
    task_id = _next_semantic_task_id(st, task_type)
    st["open"] = {
        "task_id": task_id,
        "task_type": task_type,
        "start_ms": start_ms,
        "turn_count": 0,
        "tool_call_count": 0,
        "any_turn_error": False,
        "any_tool_fail": False,
    }
    _emit_agent_task_start(
        client,
        session_id=session_id,
        user_id=user_id,
        task_id=task_id,
        task_type=task_type,
        start_ms=start_ms,
    )
    return task_id, task_type


def track_session_start(session_id: str, user_id: Optional[str] = None,
                        properties: Optional[Dict[str, Any]] = None) -> None:
    client = _get_client()
    if client is None:
        return
    _safe_call(
        client.track_session_start,
        session_id,
        user_id=user_id,
        properties=properties or {},
    )


def track_session_end(session_id: str, user_id: Optional[str] = None,
                      properties: Optional[Dict[str, Any]] = None) -> None:
    client = _get_client()
    uid = user_id or session_id
    if client is not None:
        for sid in _matching_session_keys_for_logout(session_id):
            _finalize_open_task(client, sid, uid, reason="session_end")
            _SESSION_TASK.pop(sid, None)
        _safe_call(
            client.track,
            "session_end",
            event_type="session",
            session_id=session_id,
            user_id=user_id,
            properties=properties or {},
        )


def _extract_tool_latency_ms(result: Dict[str, Any]) -> Optional[int]:
    """若工具返回里带耗时字段则抽取（毫秒）。"""
    for k in ("latency_ms", "elapsed_ms", "duration_ms", "cost_ms"):
        v = result.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _turn_status(tools: List[Dict[str, Any]], reply: str) -> str:
    """整轮状态：success / partial / error。"""
    reply_ok = bool((reply or "").strip())
    if not tools:
        return "success" if reply_ok else "error"
    oks: List[bool] = []
    for r in tools:
        if "success" in r:
            oks.append(bool(r.get("success")))
        else:
            oks.append(not bool(r.get("error")))
    if all(oks) and reply_ok:
        return "success"
    if not reply_ok or not any(oks):
        return "error"
    return "partial"


def _params_json_for_tool(
    idx: int,
    actions: Optional[List[Dict[str, Any]]],
    result: Dict[str, Any],
    tool_name: str,
) -> str:
    params: Dict[str, Any] = {}
    raw: Any = None
    if actions:
        if idx < len(actions) and str(actions[idx].get("tool") or "") == tool_name:
            raw = actions[idx].get("params")
        else:
            for a in actions:
                if str(a.get("tool") or "") == tool_name:
                    raw = a.get("params")
                    break
    if isinstance(raw, dict):
        params = raw
    if not params and isinstance(result.get("params"), dict):
        params = result["params"]
    try:
        import json as _json

        return _clip(_json.dumps(params, ensure_ascii=False))
    except Exception:
        return "{}"


def _tool_call_id(turn_id: str, idx: int, tool_name: str) -> str:
    safe = re.sub(r"[^a-z0-9_]", "_", tool_name.lower())[:40]
    return f"tc_{turn_id}_{idx:02d}_{safe}"


def track_turn(
    *,
    session_id: str,
    user_id: Optional[str],
    user_message: str,
    intent: str,
    confidence: float,
    used_llm: bool,
    reply: str,
    tool_results: List[Dict[str, Any]],
    latency_ms: int,
    model_name: Optional[str] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """商业分析对标事件。

    - ``agent_task_start`` / ``agent_task_end``：用户目标（Task）生命周期。
    - ``agent_tool_call``：工具调用，挂 ``task_id`` + ``parent_turn_id``。
    - ``agent_turn``：对话轮次，挂同一 ``task_id``。

    写入顺序：必要时 ``agent_task_end`` → ``agent_task_start`` → ``agent_tool_call``* → ``agent_turn``。
    """
    client = _get_client()
    if client is None:
        return

    task_id, task_type = _resolve_user_task(
        client,
        session_id=session_id,
        user_id=user_id,
        intent=intent,
        user_message=user_message,
    )

    turn_id = uuid.uuid4().hex[:8]
    tools = tool_results or []
    st = _session_task_state(session_id)
    op = st.get("open") or {}

    for idx, result in enumerate(tools):
        tool_name = str(result.get("tool") or result.get("name") or "unknown_tool")
        tool_call_id = _tool_call_id(turn_id, idx, tool_name)
        if "success" in result:
            success = bool(result.get("success"))
        else:
            success = not bool(result.get("error"))
        if not success and st.get("open"):
            st["open"]["any_tool_fail"] = True

        params_json = _params_json_for_tool(idx, actions, result, tool_name)

        props: Dict[str, Any] = {
            "session_id": session_id,
            "task_id": task_id,
            "task_type": task_type,
            "parent_turn_id": turn_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "success": success,
            "params_json": params_json,
            "event_schema": "agent_tool_call_v3",
        }
        tlat = _extract_tool_latency_ms(result)
        if tlat is not None:
            props["latency_ms"] = tlat
        err = str(result.get("error") or "").strip()
        if err:
            props["error"] = err[:500]

        _safe_call(
            client.track,
            "agent_tool_call",
            event_type="task",
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            user_id=user_id,
            properties=props,
        )

    lat = int(latency_ms)
    status = _turn_status(tools, reply)
    if status == "error" and st.get("open"):
        st["open"]["any_turn_error"] = True

    turn_props: Dict[str, Any] = {
        "session_id": session_id,
        "task_id": task_id,
        "task_type": task_type,
        "turn_id": turn_id,
        "intent": intent,
        "confidence": float(confidence or 0.0),
        "latency_ms": lat,
        "tool_count": len(tools),
        "status": status,
        "model_name": (model_name or "").strip() or "unknown",
        "used_llm": bool(used_llm),
        "event_schema": "agent_turn_v3",
        "user_input_len": len(user_message or ""),
        "assistant_output_len": len(reply or ""),
    }
    if _RECORD_PAYLOAD:
        turn_props["user_input"] = _clip(user_message or "")
        turn_props["assistant_output"] = _clip(reply or "")
    pii_in = _pii_hits(user_message or "")
    pii_out = _pii_hits(reply or "")
    if pii_in:
        turn_props["pii_in"] = pii_in
    if pii_out:
        turn_props["pii_out"] = pii_out

    _safe_call(
        client.track,
        "agent_turn",
        event_type="turn",
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        user_id=user_id,
        properties=turn_props,
    )

    if st.get("open"):
        st["open"]["turn_count"] = int(st["open"].get("turn_count", 0)) + 1
        st["open"]["tool_call_count"] = int(st["open"].get("tool_call_count", 0)) + len(tools)


def flush() -> None:
    client = _get_client()
    if client is None:
        return
    _safe_call(client.flush)


def measure_start() -> float:
    return time.perf_counter()


def measure_ms(start_ts: float) -> int:
    return int((time.perf_counter() - start_ts) * 1000)


def track_task_result(task_id: str, **kwargs: Any) -> None:
    """透传 ``TrackingClient.track_task_result``（自定义任务结果，可选）。"""
    client = _get_client()
    if client is None:
        return
    _safe_call(client.track_task_result, task_id, **kwargs)
