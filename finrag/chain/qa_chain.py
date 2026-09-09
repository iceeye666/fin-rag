"""问答链：检索 → 重排 → 上下文压缩 → 严格约束生成 → 溯源输出。

链路（LCEL 风格，但显式拆分以便插入自定义逻辑与埋点）：

    question ──► [可选] 多轮改写 ──► 多向量检索(摘要→原文)
                                        │
                                        ▼
                                   重排 + 去重 + 截断（上下文压缩）
                                        │
                                        ▼
                              format_context（带来源编号）
                                        │
                                        ▼
                              Qwen-Plus（严格 Prompt）
                                        │
                                        ▼
                        Answer 结果对象（文本 + 引用 + 置信度 + 命中表格）

「上下文压缩」体现在三处：
* 检索阶段：只召回 Top-K，而非全文灌入；
* 重排阶段：按相关性截断到 top_k_final；
* 格式化阶段：单块超出 max_chars 时截断，并为表格打标签避免误读。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from finrag.config import Settings
from finrag.prompt.templates import build_qa_prompt, format_context
from finrag.retrieval.rerank import KeywordOverlapReranker


@dataclass
class Citation:
    """一条引用来源。"""

    index: int
    citation: str
    category: str
    page: int | None
    score: float
    snippet: str
    html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "citation": self.citation,
            "category": self.category,
            "page": self.page,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "html": self.html,
        }


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    latency_ms: int = 0
    retrieval_ms: int = 0
    model: str = ""
    confidence: float = 0.0
    refused: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "latency_ms": self.latency_ms,
            "retrieval_ms": self.retrieval_ms,
            "model": self.model,
            "confidence": round(self.confidence, 4),
            "refused": self.refused,
            "debug": self.debug,
        }


REFUSAL_MARK = "无法回答"


class QAChain:
    """年报问答链。"""

    def __init__(
        self,
        retriever,
        llm: BaseChatModel,
        cfg: Settings | None = None,
        reranker=None,
    ):
        from finrag.config import get_settings

        self.cfg = cfg or get_settings()
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.prompt = build_qa_prompt()
        self.model_name = getattr(llm, "model_name", None) or getattr(
            llm, "model", self.cfg.llm_model
        )

    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or self.cfg.top_k_summary
        pairs = self.retriever.retrieve_with_scores(question, k=k)

        if self.reranker is not None:
            docs = [d for d, _ in pairs]
            base_score = {id(d): s for d, s in pairs}
            reranked = self.reranker.rerank(question, docs)
            # 融合：0.4 * 向量相似度 + 0.6 * 重排分，避免纯关键词把语义相关块压掉
            pairs = [
                (d, 0.4 * base_score.get(id(d), 0.0) + 0.6 * s) for d, s in reranked
            ]
            pairs.sort(key=lambda x: -x[1])
        elif isinstance(self.reranker, type(None)) and self.cfg.enable_rerank:
            pairs = KeywordOverlapReranker().rerank(question, [d for d, _ in pairs])
        return pairs[: self.cfg.top_k_final]

    # ------------------------------------------------------------------ #
    def invoke(self, question: str, k: int | None = None) -> AnswerResult:
        t0 = time.perf_counter()
        pairs = self.retrieve(question, k)
        t1 = time.perf_counter()

        citations: list[Citation] = []
        for i, (doc, score) in enumerate(pairs, start=1):
            md = doc.metadata or {}
            citations.append(
                Citation(
                    index=i,
                    citation=md.get("citation", ""),
                    category=md.get("category", "text"),
                    page=md.get("page"),
                    score=float(score),
                    snippet=doc.page_content[:220],
                    html=md.get("html") or None,
                )
            )

        context = format_context([d for d, _ in pairs])
        if not pairs:
            context = "（本次检索未命中任何内容）"

        messages = self.prompt.format_messages(context=context, question=question)
        resp = self.llm.invoke(messages)
        answer = str(getattr(resp, "content", resp)).strip()
        t2 = time.perf_counter()

        refused = REFUSAL_MARK in answer
        confidence = _confidence(pairs, refused)

        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            latency_ms=int((t2 - t0) * 1000),
            retrieval_ms=int((t1 - t0) * 1000),
            model=self.model_name,
            confidence=confidence,
            refused=refused,
            debug={
                "n_retrieved": len(pairs),
                "context_chars": len(context),
                "top_score": round(pairs[0][1], 4) if pairs else 0.0,
            },
        )

    # ------------------------------------------------------------------ #
    def stream(self, question: str, k: int | None = None):
        """流式生成（Web UI 用）。先返回引用，再逐 token 输出答案。"""
        pairs = self.retrieve(question, k)
        citations = [
            Citation(
                index=i,
                citation=(d.metadata or {}).get("citation", ""),
                category=(d.metadata or {}).get("category", "text"),
                page=(d.metadata or {}).get("page"),
                score=float(s),
                snippet=d.page_content[:220],
                html=(d.metadata or {}).get("html") or None,
            )
            for i, (d, s) in enumerate(pairs, start=1)
        ]
        context = format_context([d for d, _ in pairs]) or "（本次检索未命中任何内容）"
        messages = self.prompt.format_messages(context=context, question=question)

        yield {"type": "citations", "data": [c.to_dict() for c in citations]}
        try:
            for chunk in self.llm.stream(messages):
                delta = getattr(chunk, "content", None)
                if delta:
                    yield {"type": "delta", "data": delta}
        except Exception:  # noqa: BLE001  Mock 模型不支持 stream
            resp = self.llm.invoke(messages)
            yield {"type": "delta", "data": str(getattr(resp, "content", resp))}
        yield {"type": "done", "data": {"n_citations": len(citations)}}


def _confidence(pairs, refused: bool) -> float:
    """简易置信度：结合最高分、命中数量与是否拒答。

    说明：这不是模型校准概率，而是供 UI 展示的启发式指标，
    用于提示用户"检索质量是否可信"。
    """
    if refused or not pairs:
        return 0.0
    top = pairs[0][1]
    coverage = min(1.0, len(pairs) / 3)
    return round(float(min(1.0, 0.6 * top + 0.4 * coverage)), 4)


__all__ = ["QAChain", "AnswerResult", "Citation"]
