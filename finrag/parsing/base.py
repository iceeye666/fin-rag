"""解析层的数据契约与抽象基类。

统一的中间表示（IR）是本项目可维护的关键：
    上游无论用 unstructured(hi_res) 还是 pdfplumber，输出都是 ``ParsedElement``；
    下游切分/摘要/检索只依赖 IR，替换解析后端不需要改动任何业务代码。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# unstructured 元素类别 → 本项目归一化类别
ELEMENT_ALIASES = {
    "NarrativeText": "text",
    "Text": "text",
    "UncategorizedText": "text",
    "ListItem": "text",
    "Title": "title",
    "Header": "title",
    "SubTitle": "title",
    "Table": "table",
    "TableChunk": "table",
    "FigureCaption": "caption",
    "Image": "image",
    "PageBreak": "page_break",
    "Footer": "footer",
    "PageNumber": "footer",
}


@dataclass
class ParsedElement:
    """一个被识别出的文档元素（文本段落 / 标题 / 表格）。"""

    category: str                      # text | title | table | caption | footer
    text: str                          # 纯文本表示（表格为 Markdown 文本化结果）
    html: str | None = None            # 表格的 HTML 表示（保留行列结构）
    page: int | None = None            # 1-based 页码
    source: str = ""                   # 来源文件名
    raw_category: str = ""             # 解析器的原始类别名
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_table(self) -> bool:
        return self.category == "table"

    def __len__(self) -> int:  # 便于按长度过滤
        return len(self.text)


class BaseParser(ABC):
    """PDF 解析器接口。"""

    name: str = "base"

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        ...

    # ---------------------------------------------------------------- #
    @staticmethod
    def normalize_category(raw: str) -> str:
        return ELEMENT_ALIASES.get(raw, "text")

    @staticmethod
    def save(elements: Iterable[ParsedElement], out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in elements]
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_path

    @staticmethod
    def load(path: str | Path) -> list[ParsedElement]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [ParsedElement(**d) for d in data]


def drop_noise(elements: list[ParsedElement], min_chars: int = 2) -> list[ParsedElement]:
    """丢弃页眉页脚、页码、空块等噪声。"""
    keep = []
    for e in elements:
        if e.category in {"footer", "page_break", "image"}:
            continue
        t = e.text.strip()
        if len(t) < min_chars:
            continue
        # 纯页码 / 纯分隔线
        if re_fullmatch_digits(t):
            continue
        keep.append(e)
    return keep


def re_fullmatch_digits(t: str) -> bool:
    t = t.strip()
    return bool(t) and all(c.isdigit() or c in "-—." for c in t)


__all__ = ["ParsedElement", "BaseParser", "ELEMENT_ALIASES", "drop_noise"]
