"""文本块与表格块的差异化切分。

设计原则：**表格不参与文本切分**。
表格是二维结构，一旦被按字符数切断，行列关系就毁了，检索与生成都会出错。
因此这里把元素分成两条独立流水线：

* 文本流 → 递归字符切分（中文友好分隔符），overlap 保留上下文；
* 表格流 → 保持完整结构，超长时按「行块 + 表头复写」切片。

两类块都带上「章节路径」前缀，用于解决年报中大量指代表述的语义缺失。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.config import Settings
from finrag.parsing.base import ParsedElement

# 中文文档更合理的切分优先级：段落 → 句 → 分句 → 顿号/逗号
CN_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "…", "，", "、", " "]


@dataclass
class Chunk:
    """切分后的最小检索/生成单元。"""

    chunk_id: str
    category: str                       # text | table
    text: str                           # 带章节前缀的完整内容（送入 LLM）
    content: str                        # 原始内容（不含前缀，送入 docstore）
    html: str | None = None
    page: int | None = None
    source: str = ""
    section: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)

    @property
    def citation(self) -> str:
        """人类可读的引用来源，如「2025年报.pdf · P12 · 合并利润表」。"""
        parts = [self.source or "未知文档"]
        if self.page:
            parts.append(f"P{self.page}")
        if self.section:
            parts.append(self.section)
        if self.category == "table":
            parts.append("表格")
        return " · ".join(parts)


def _make_id(source: str, page: int | None, idx: int, text: str) -> str:
    raw = f"{source}|{page}|{idx}|{text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _section_prefix(el: ParsedElement) -> str:
    sec = el.meta.get("section", "")
    sub = el.meta.get("subsection", "")
    parts = [p for p in (sec, sub) if p and p != sec]
    return f"【{' / '.join(parts)}】" if parts else ""


def _build_splitter(cfg: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=CN_SEPARATORS,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


# --------------------------------------------------------------------------- #
def chunk_text_element(el: ParsedElement, cfg: Settings, idx: int) -> list[Chunk]:
    splitter = _build_splitter(cfg)
    prefix = _section_prefix(el)
    base = f"{prefix}{el.text}" if prefix else el.text

    pieces = splitter.split_text(base) if len(base) > cfg.chunk_size else [base]
    out = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if len(piece) < cfg.min_chunk_chars:
            continue
        out.append(
            Chunk(
                chunk_id=_make_id(el.source, el.page, idx * 100 + i, piece),
                category="text",
                text=piece,
                content=piece,
                page=el.page,
                source=el.source,
                section=el.meta.get("section", ""),
                meta={**el.meta, "chunk_index": i},
            )
        )
    return out


def chunk_table_element(el: ParsedElement, cfg: Settings, idx: int) -> list[Chunk]:
    """表格切分：优先整表保留；行数超限则按行块切分，每块复写表头。"""
    from finrag.parsing.postprocess import split_oversized_table

    prefix = _section_prefix(el)
    parts = split_oversized_table(el, max_rows=cfg.table_max_rows_per_chunk)

    out = []
    for i, part in enumerate(parts):
        content = part.text.strip()
        if len(content) < cfg.min_chunk_chars:
            continue
        text = f"{prefix}{content}" if prefix else content
        # 给表格加一句结构化提示，帮助 LLM 理解这是表格而不是乱码
        text = f"[表格内容开始]\n{text}\n[表格内容结束]"
        out.append(
            Chunk(
                chunk_id=_make_id(el.source, el.page, idx * 100 + i, content),
                category="table",
                text=text,
                content=content,
                html=part.html,
                page=el.page,
                source=el.source,
                section=el.meta.get("section", ""),
                meta={**part.meta, "chunk_index": i},
            )
        )
    return out


def chunk_elements(
    elements: Iterable[ParsedElement], cfg: Settings | None = None
) -> list[Chunk]:
    """把解析元素切分为 Chunk 列表（文本流 + 表格流）。"""
    from finrag.config import get_settings

    cfg = cfg or get_settings()
    chunks: list[Chunk] = []
    for idx, el in enumerate(elements):
        if el.category == "table":
            chunks.extend(chunk_table_element(el, cfg, idx))
        elif el.category in {"text", "title", "caption"}:
            chunks.extend(chunk_text_element(el, cfg, idx))
        else:
            continue
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    n_text = sum(1 for c in chunks if c.category == "text")
    n_table = sum(1 for c in chunks if c.category == "table")
    lengths = [len(c.text) for c in chunks] or [0]
    return {
        "total": len(chunks),
        "text": n_text,
        "table": n_table,
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "max_len": max(lengths),
        "min_len": min(lengths),
    }


__all__ = ["Chunk", "chunk_elements", "chunk_text_element", "chunk_table_element", "chunk_stats"]
