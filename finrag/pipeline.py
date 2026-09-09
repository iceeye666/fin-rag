"""端到端流水线：解析 → 切分 → 摘要 → 多向量入库 → 检索问答。

这是整个项目对外暴露的主入口，CLI / FastAPI / 评测脚本都只依赖它。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from langchain_core.documents import Document

from finrag.chain.qa_chain import AnswerResult, QAChain
from finrag.chunking.chunker import Chunk, chunk_elements, chunk_stats
from finrag.config import Settings
from finrag.parsing.factory import parse_pdf
from finrag.retrieval.multivector import (
    DOC_ID_KEY,
    ChunkDocStore,
    FinancialMultiVectorRetriever,
    build_vectorstore,
)
from finrag.retrieval.rerank import build_reranker
from finrag.summarize.summarizer import Summarizer


@dataclass
class IngestStats:
    files: list[str] = field(default_factory=list)
    n_elements: int = 0
    n_chunks: int = 0
    n_text_chunks: int = 0
    n_table_chunks: int = 0
    chunk_stats: dict[str, Any] = field(default_factory=dict)
    summary_mode: str = ""
    parse_backend: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "n_elements": self.n_elements,
            "n_chunks": self.n_chunks,
            "n_text_chunks": self.n_text_chunks,
            "n_table_chunks": self.n_table_chunks,
            "chunk_stats": self.chunk_stats,
            "summary_mode": self.summary_mode,
            "parse_backend": self.parse_backend,
            "elapsed_s": round(self.elapsed_s, 2),
        }


class FinancialRAGPipeline:
    """金融年报 RAG 主流水线。"""

    def __init__(
        self,
        cfg: Settings | None = None,
        embeddings=None,
        llm=None,
        verbose: bool = True,
    ):
        from finrag.config import get_settings
        from finrag.models import build_chat_model, build_embeddings

        self.cfg = cfg or get_settings()
        self.verbose = verbose
        self.embeddings = embeddings or build_embeddings(self.cfg)
        self.vectorstore = build_vectorstore(self.cfg, self.embeddings)
        self.docstore = ChunkDocStore(self.cfg.docstore_path / "chunks.json")
        self.retriever = FinancialMultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            k=self.cfg.top_k_summary,
            fetch_k=self.cfg.fetch_k,
            lambda_mult=self.cfg.mmr_lambda,
            enable_mmr=self.cfg.enable_mmr,
            score_threshold=self.cfg.score_threshold,
        )
        self.reranker = build_reranker(self.cfg)
        self.llm = llm or build_chat_model(self.cfg)
        self.summarizer = Summarizer(self.cfg)
        self.chain = QAChain(self.retriever, self.llm, self.cfg, self.reranker)
        self._last_stats: IngestStats | None = None

    # ------------------------------------------------------------------ #
    #                              入库                                   #
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        pdf_paths: Iterable[str | Path],
        rebuild: bool = False,
    ) -> IngestStats:
        import time

        t0 = time.perf_counter()
        stats = IngestStats()

        if rebuild:
            self.reset()

        all_chunks: list[Chunk] = []
        for p in pdf_paths:
            p = Path(p)
            if not p.exists():
                print(f"[ingest] 跳过不存在的文件：{p}")
                continue
            self._log(f"[ingest] 解析 {p.name} ...")
            elements, parser = parse_pdf(p, self.cfg)
            stats.parse_backend = getattr(parser, "name", "unknown")
            chunks = chunk_elements(elements, self.cfg)
            self._log(
                f"         元素 {len(elements)} → 切块 {len(chunks)}"
            )
            all_chunks.extend(chunks)
            stats.files.append(p.name)
            stats.n_elements += len(elements)

        if not all_chunks:
            return stats

        stats.chunk_stats = chunk_stats(all_chunks)
        stats.n_chunks = len(all_chunks)
        stats.n_text_chunks = stats.chunk_stats["text"]
        stats.n_table_chunks = stats.chunk_stats["table"]

        # ---- 摘要 ----
        self._log(f"[ingest] 生成摘要（模式：{self.summarizer.mode}）...")
        texts = [c.content for c in all_chunks]
        summaries = self.summarizer.summarize_batch(
            texts,
            progress=lambda d, t: self._log(f"         {d}/{t}") if d % 10 == 0 else None,
        )
        stats.summary_mode = self.summarizer.mode

        # ---- 入库：摘要向量 + 原文 docstore ----
        self._log(f"[ingest] 写入向量库（{len(all_chunks)} 条摘要向量）...")
        summary_docs = []
        for chunk, summary in zip(all_chunks, summaries):
            # 摘要 + 章节名一起作为索引文本，兼顾概括性与主题定位
            index_text = f"{chunk.section}：{summary}" if chunk.section else summary
            summary_docs.append(
                Document(
                    page_content=index_text,
                    metadata={
                        DOC_ID_KEY: chunk.chunk_id,
                        "category": chunk.category,
                        "page": chunk.page,
                        "source": chunk.source,
                        "section": chunk.section,
                    },
                )
            )
        self.vectorstore.add_documents(
            summary_docs, ids=[c.chunk_id for c in all_chunks]
        )
        self.docstore.mset(all_chunks)
        self.docstore.save()

        stats.elapsed_s = time.perf_counter() - t0
        self._last_stats = stats
        self._log(f"[ingest] 完成，用时 {stats.elapsed_s:.1f}s")
        return stats

    # ------------------------------------------------------------------ #
    #                              问答                                   #
    # ------------------------------------------------------------------ #
    def ask(self, question: str, k: int | None = None) -> AnswerResult:
        return self.chain.invoke(question, k=k)

    def stream(self, question: str, k: int | None = None) -> Iterator[dict]:
        return self.chain.stream(question, k=k)

    def retrieve(self, question: str, k: int | None = None):
        return self.chain.retrieve(question, k=k)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """清空向量库与文档仓库。"""
        try:
            self.vectorstore.delete_collection()
        except Exception:  # noqa: BLE001
            pass
        self.vectorstore = build_vectorstore(self.cfg, self.embeddings)
        self.retriever.vectorstore = self.vectorstore
        self.docstore.data.clear()
        self.docstore.save()

    @property
    def is_empty(self) -> bool:
        return len(self.docstore) == 0

    @property
    def stats(self) -> IngestStats | None:
        return self._last_stats

    def index_info(self) -> dict[str, Any]:
        n_vec = 0
        try:
            n_vec = self.vectorstore._collection.count()
        except Exception:  # noqa: BLE001
            pass
        return {
            "docstore_chunks": len(self.docstore),
            "vector_count": n_vec,
            "collection": self.cfg.chroma_collection,
            "persist_dir": str(self.cfg.chroma_dir),
            "last_ingest": self._last_stats.to_dict() if self._last_stats else None,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["FinancialRAGPipeline", "IngestStats"]
