"""多向量检索（MultiVectorRetriever）：摘要建索引，原文做召回。

架构（与 LangChain 官方 MultiVectorRetriever 思想一致，此处做了金融场景增强）：

        ┌────────────────┐   embed  ┌──────────────────┐
        │ 摘要（60~150字）├─────────►│  Chroma 向量库    │
        └────────────────┘          │  metadata.doc_id │
                                    └────────┬─────────┘
                                             │ Top-K 相似
                                             ▼
                                    ┌──────────────────┐
                                    │  DocStore(JSON)  │  原文块 / 表格
                                    └────────┬─────────┘
                                             │ doc_id → 原始 Chunk
                                             ▼
                                    送入 LLM 的完整上下文

相比「直接把原文块向量化」的两点提升：
1. 摘要语义密度高，Top-K 命中率明显更好（尤其对"同比增速是多少"这类指标型问题）；
2. 召回后返回原文，LLM 仍能看到完整表格与段落，不损失数字细节。

同时实现为 LangChain ``BaseRetriever``，可直接接入 LCEL 链路。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from finrag.chunking.chunker import Chunk
from finrag.config import Settings

DOC_ID_KEY = "doc_id"


class ChunkDocStore:
    """轻量 JSON 文档仓库：doc_id → Chunk（含原文与表格 HTML）。

    用 JSON 而非 pickle，便于人工检查、版本管理与跨环境迁移。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}

    def mset(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self.data[c.chunk_id] = c.to_dict()

    def mget(self, ids: Iterable[str]) -> list[Chunk | None]:
        out = []
        for i in ids:
            d = self.data.get(i)
            out.append(Chunk.from_dict(d) if d else None)
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.data)


class FinancialMultiVectorRetriever(BaseRetriever):
    """面向金融年报的多向量检索器。

    Parameters
    ----------
    vectorstore:
        存放**摘要向量**的 LangChain VectorStore（本项目为 Chroma）。
    docstore:
        存放**原始 Chunk** 的仓库。
    search_kwargs:
        ``k`` / ``fetch_k`` / ``lambda_mult`` 等检索参数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any = Field(description="摘要向量库")
    docstore: ChunkDocStore = Field(description="原文仓库")
    k: int = 6
    fetch_k: int = 30
    lambda_mult: float = 0.5
    enable_mmr: bool = True
    score_threshold: float = 0.0

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return [d for d, _ in self.retrieve_with_scores(query, **kwargs)]

    # ------------------------------------------------------------------ #
    def retrieve_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """检索并返回 (原文Document, 相似度) —— 相似度已归一到 0~1，越大越相关。"""
        k = k or self.k
        # 带分数的候选池（MMR 与精排都要基于它）
        candidates = self.vectorstore.similarity_search_with_score(
            query, k=max(self.fetch_k, k)
        )
        score_map = {
            d.metadata.get(DOC_ID_KEY): s for d, s in candidates if d.metadata.get(DOC_ID_KEY)
        }

        if self.enable_mmr and len(candidates) > k:
            # Chroma 未提供「带分数的 MMR」，因此分两步：
            # 1) 用 MMR 选出多样性的文档；2) 从候选池查表补回相似度分数。
            embedding_fn = getattr(self.vectorstore, "_embedding_function", None)
            if embedding_fn is not None:
                try:
                    qvec = embedding_fn.embed_query(query)
                    docs = self.vectorstore.max_marginal_relevance_search_by_vector(
                        qvec, k=k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
                    )
                    pairs = [(d, score_map.get(d.metadata.get(DOC_ID_KEY), 0.0)) for d in docs]
                except Exception:  # noqa: BLE001  MMR 不可用时退化为纯相似度
                    pairs = candidates[:k]
            else:
                pairs = candidates[:k]
        else:
            pairs = candidates[:k]

        out: list[tuple[Document, float]] = []
        seen: set[str] = set()
        for doc, distance in pairs:
            doc_id = doc.metadata.get(DOC_ID_KEY)
            if not doc_id or doc_id in seen:
                continue
            chunk = self.docstore.mget([doc_id])[0]
            if chunk is None:
                continue
            seen.add(doc_id)
            sim = _distance_to_similarity(distance)
            if sim < self.score_threshold:
                continue
            out.append((self._to_document(chunk, doc.metadata), sim))
        return out

    @staticmethod
    def _to_document(chunk: Chunk, meta: dict | None = None) -> Document:
        return Document(
            page_content=chunk.text,
            metadata={
                DOC_ID_KEY: chunk.chunk_id,
                "category": chunk.category,
                "page": chunk.page,
                "source": chunk.source,
                "section": chunk.section,
                "citation": chunk.citation,
                "html": chunk.html or "",
                **(meta or {}),
            },
        )


def _distance_to_similarity(distance: float | None) -> float:
    """Chroma(cosine) 返回的是平方余弦距离，取值范围 [0, 2]。"""
    if distance is None:
        return 0.0
    # 余弦距离 → 余弦相似度
    sim = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, sim))


# --------------------------------------------------------------------------- #
def build_vectorstore(cfg: Settings, embeddings, collection_name: str | None = None):
    """构建（或连接）Chroma 向量库。"""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name or cfg.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(cfg.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_vectorstore(cfg: Settings, embeddings) -> None:
    vs = build_vectorstore(cfg, embeddings)
    try:
        vs.delete_collection()
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "ChunkDocStore",
    "FinancialMultiVectorRetriever",
    "build_vectorstore",
    "reset_vectorstore",
    "DOC_ID_KEY",
]
