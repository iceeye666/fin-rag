"""两级排序：向量召回（粗排）→ 重排（精排）。

向量召回追求「不漏」，Top-K 里常混入主题相近但不含答案的块；
重排用更精细的信号把真正含答案的块提到前面，直接提升 LLM 上下文质量。

提供两种实现：
* ``KeywordOverlapReranker`` —— 零依赖。结合「查询关键词覆盖率」与
  「数字命中率」打分。金融问答高度依赖精确数字，命中查询中的数字
  往往意味着这就是答案所在块，因此权重给得很高。
* ``BGEReranker`` —— 可选，加载 BAAI/bge-reranker-large 做交叉编码，
  效果更好但需要额外权重（默认关闭）。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from langchain_core.documents import Document

_STOP = set("的 了 和 与 及 在 是 为 对 由 从 到 请 请问 多少 什么 哪些 如何 是否 有哪些 公司 年报".split())


def _terms(text: str) -> set[str]:
    """中文二元切词 + 数字，作为轻量语义单元。"""
    t = re.sub(r"[\s，。、；：（）()%“”\"'？?！!]", "", text)
    grams = {t[i : i + 2] for i in range(len(t) - 1)}
    grams |= {w for w in re.findall(r"[A-Za-z]{2,}", text)}
    grams |= set(re.findall(r"\d+\.?\d*", text))
    return {g for g in grams if g not in _STOP}


class KeywordOverlapReranker:
    """零依赖重排器。"""

    def __init__(self, number_weight: float = 0.35, coverage_weight: float = 0.65):
        self.number_weight = number_weight
        self.coverage_weight = coverage_weight

    def rerank(
        self, query: str, docs: Sequence[Document], top_n: int | None = None
    ) -> list[tuple[Document, float]]:
        q_terms = _terms(query)
        q_numbers = set(re.findall(r"\d+\.?\d*", query))
        scored = []
        for d in docs:
            d_terms = _terms(d.page_content)
            d_numbers = set(re.findall(r"\d+\.?\d*", d.page_content))
            coverage = len(q_terms & d_terms) / (len(q_terms) + 1e-9)
            num_hit = (
                len(q_numbers & d_numbers) / (len(q_numbers) + 1e-9) if q_numbers else 0.0
            )
            # 表格块在财报问答中价值更高，轻微加权
            table_bonus = 0.05 if d.metadata.get("category") == "table" else 0.0
            score = self.coverage_weight * coverage + self.number_weight * num_hit + table_bonus
            scored.append((d, float(score)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n] if top_n else scored


class BGEReranker:
    """交叉编码重排（可选，需 BAAI/bge-reranker-large）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: Sequence[Document], top_n: int | None = None):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
        out = [(docs[i], float(scores[i])) for i in order]
        return out[:top_n] if top_n else out


def build_reranker(cfg=None):
    from finrag.config import get_settings

    cfg = cfg or get_settings()
    if not cfg.enable_rerank:
        return None
    try:
        return BGEReranker(device=cfg.embed_device)
    except Exception as e:  # noqa: BLE001
        print(f"[rerank] bge-reranker 不可用，使用关键词重排：{e}")
        return KeywordOverlapReranker()


__all__ = ["KeywordOverlapReranker", "BGEReranker", "build_reranker"]
