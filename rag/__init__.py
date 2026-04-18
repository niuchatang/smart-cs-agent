"""
RAG 扩展模块（`rag/`）

- `hybrid_retriever.py`：BM25（已在 main.py 内使用）与**可选**向量检索的合并；
- 若环境里不存在向量模型（sentence-transformers / 兼容 embedding 依赖），
  混合检索会自动退化为纯 BM25，不影响现有行为。
"""

from .hybrid_retriever import HybridRetriever, RRFConfig

__all__ = ["HybridRetriever", "RRFConfig"]
