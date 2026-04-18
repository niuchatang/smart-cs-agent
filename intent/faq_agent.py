"""
FAQ / 知识问答智能体（FAQ Agent）

- 对 `main.py` 中「`_render_reply` 命中 fare_policy / unknown / route_planning + RAG」逻辑
  做一次抽象封装，便于未来把检索器替换成 **BM25 + 向量** 混合检索；
- 现在只负责：
  1) 构造「严格基于片段回答」的 prompt；
  2) 若 LLM 可用则走链路返回结构化结果（含引用角标）；
  3) 否则回退为「第一条命中 + 标题来源」文本。

输出：`{reply: str, citations: [{id,title,snippet}]}`。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _snippet(hit: Dict[str, Any], max_len: int = 140) -> str:
    text = str(hit.get("content") or hit.get("text") or "")
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


class FAQAgent:
    name = "faq"
    priority = 120

    def __init__(self, service_agent: Any) -> None:
        self._svc = service_agent

    def answer(
        self,
        question: str,
        rag_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hits = rag_hits or []
        if not hits:
            return {"reply": "", "citations": []}

        citations = [
            {
                "id": str(h.get("id", "")),
                "title": str(h.get("title", "")),
                "snippet": _snippet(h),
            }
            for h in hits[:3]
        ]

        chain = getattr(self._svc, "answer_chain", None)
        if chain is None:
            first = hits[0]
            body = str(first.get("content") or first.get("text") or "").strip()
            title = str(first.get("title", "")).strip()
            reply = f"{body}\n\n— 知识依据：{title}" if title else body
            return {"reply": reply, "citations": citations}

        try:
            context = "\n\n".join(
                f"[{i+1}] {h.get('title','')}：{_snippet(h, 240)}" for i, h in enumerate(hits[:3])
            )
            text = chain.invoke({"question": question, "context": context})
            reply = str(text).strip()
            if not reply:
                raise ValueError("empty llm answer")
            if "[1]" not in reply and citations:
                reply = f"{reply}\n\n— 知识依据：{citations[0]['title']}"
            return {"reply": reply, "citations": citations}
        except Exception:
            first = hits[0]
            body = str(first.get("content") or first.get("text") or "").strip()
            title = str(first.get("title", "")).strip()
            reply = f"{body}\n\n— 知识依据：{title}" if title else body
            return {"reply": reply, "citations": citations}
