"""
混合检索器（Hybrid Retriever）

- 输入：一组已有文档（`[{id,title,content}]`）；
- 输出：合并 BM25 与可选向量检索结果的 Top-K 列表；
- 合并算法：**RRF (Reciprocal Rank Fusion)**，`score = Σ 1 / (k + rank_i)`；
- 无外部 embedding 依赖时自动退化为纯 BM25（与 `main.py._SimpleRAGStore` 行为对齐）。

可选：若安装了 `sentence-transformers`，可传入 `vector_encoder="paraphrase-multilingual-MiniLM-L12-v2"` 等模型；
或传入一个自定义 `encoder: Callable[[List[str]], List[List[float]]]` 做 embedding。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    BM25Retriever = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]


@dataclass
class RRFConfig:
    k: int = 60
    top_k: int = 5
    bm25_weight: float = 1.0
    vector_weight: float = 1.0


class HybridRetriever:
    def __init__(
        self,
        docs: Sequence[Dict[str, Any]],
        *,
        encoder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        rrf: Optional[RRFConfig] = None,
    ) -> None:
        self._docs: List[Dict[str, Any]] = list(docs)
        self._rrf = rrf or RRFConfig()
        self._bm25 = self._build_bm25(self._docs)
        self._encoder = encoder
        self._doc_vecs: Optional[List[List[float]]] = None
        if encoder is not None:
            try:
                self._doc_vecs = encoder([self._doc_text(d) for d in self._docs])
            except Exception:
                self._doc_vecs = None

    @staticmethod
    def _doc_text(d: Dict[str, Any]) -> str:
        return " ".join(
            str(d.get(k, "")) for k in ("title", "content", "text") if d.get(k)
        )

    @staticmethod
    def _build_bm25(docs: List[Dict[str, Any]]) -> Any:
        if BM25Retriever is None or Document is None:
            return None
        lc_docs = [
            Document(
                page_content=HybridRetriever._doc_text(d),
                metadata={"id": d.get("id"), "title": d.get("title", "")},
            )
            for d in docs
        ]
        if not lc_docs:
            return None
        retriever = BM25Retriever.from_documents(lc_docs)
        retriever.k = 20
        return retriever

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        topk = int(k or self._rrf.top_k)
        bm25_ranks = self._bm25_rank(query)
        vec_ranks = self._vector_rank(query)
        if not bm25_ranks and not vec_ranks:
            return []

        scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + (
                self._rrf.bm25_weight / (self._rrf.k + rank + 1)
            )
        for rank, doc_id in enumerate(vec_ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + (
                self._rrf.vector_weight / (self._rrf.k + rank + 1)
            )

        by_id = {str(d.get("id")): d for d in self._docs}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:topk]
        out: List[Dict[str, Any]] = []
        for doc_id, score in ranked:
            doc = by_id.get(str(doc_id))
            if not doc:
                continue
            out.append(
                {
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "content": doc.get("content") or doc.get("text") or "",
                    "score": round(float(score), 6),
                }
            )
        return out

    def _bm25_rank(self, query: str) -> List[str]:
        if self._bm25 is None:
            return []
        try:
            results = self._bm25.get_relevant_documents(query)
        except Exception:
            return []
        return [
            str(r.metadata.get("id"))
            for r in results
            if getattr(r, "metadata", None) and r.metadata.get("id") is not None
        ]

    def _vector_rank(self, query: str) -> List[str]:
        if self._encoder is None or not self._doc_vecs:
            return []
        try:
            qv = self._encoder([query])[0]
        except Exception:
            return []
        sims: List[tuple[float, str]] = []
        for i, dv in enumerate(self._doc_vecs):
            doc_id = str(self._docs[i].get("id"))
            sims.append((self._cosine(qv, dv), doc_id))
        sims.sort(key=lambda x: -x[0])
        return [doc_id for _, doc_id in sims[:20]]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
