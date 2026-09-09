"""unstructured + hi_res 高精度解析后端。

hi_res 策略链路：
    PDF → pdf2image 渲染 300DPI 位图 → 布局检测模型（YOLOX）→ 元素分类
        → 表格结构识别（Table Transformer）→ 输出 HTML 表格

优点：能正确分离 Title / NarrativeText / Table，并保留合并单元格等复杂表结构；
代价：慢（每页数秒）且依赖 poppler（pdftoppm）。因此工厂层会在不可用时
自动降级到 pdfplumber，代码路径保持不变。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from finrag.config import Settings
from finrag.parsing.base import BaseParser, ParsedElement


class UnstructuredParser(BaseParser):
    name = "unstructured"

    def __init__(self, cfg: Settings | None = None):
        from finrag.config import get_settings

        self.cfg = cfg or get_settings()

    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 unstructured[pdf]") from e

        pdf_path = Path(pdf_path)
        kwargs: dict[str, Any] = dict(
            filename=str(pdf_path),
            strategy=self.cfg.parse_strategy,
            infer_table_structure=True,   # 关键：输出 text_as_html
            include_page_breaks=False,
            languages=["chi_sim", "eng"],
        )
        if self.cfg.parse_strategy == "hi_res":
            # 关闭图片抽取，年报里图表对问答价值低但耗时高
            kwargs["extract_image_block_types"] = []

        raw_elements = partition_pdf(**kwargs)

        out: list[ParsedElement] = []
        for idx, el in enumerate(raw_elements):
            raw_cat = type(el).__name__
            category = self.normalize_category(raw_cat)
            text = (el.text or "").strip()
            html = None
            if category == "table":
                html = getattr(el.metadata, "text_as_html", None)
                if html:
                    html = re.sub(r"\s+", " ", html).strip()
                # 关键：unstructured 的 el.text 对中文表格会把所有单元格压成一行，
                # 行列关系完全丢失。text_as_html 才是结构化的，必须优先用它还原。
                if html:
                    text = _html_to_text(html) or text
            if category == "text" and not text:
                continue

            meta = _safe_meta(el)
            _attach_coordinates(el, meta)
            out.append(
                ParsedElement(
                    category=category,
                    text=text,
                    html=html,
                    page=meta.get("page_number"),
                    source=pdf_path.name,
                    raw_category=raw_cat,
                    meta={"element_index": idx, **meta},
                )
            )
        return out


def _attach_coordinates(el: Any, meta: dict[str, Any]) -> None:
    """从 unstructured 的 coordinates 提取相对纵向位置。

    有了它，跨页续表合并才能用「前表贴页底 + 后表贴页顶」这条物理判据，
    否则只能靠表头文本猜测，遇到「流动资产合计 / 非流动资产合计」这类
    分页点就会误判为两张独立表。
    """
    try:
        coords = getattr(el.metadata, "coordinates", None)
        if not coords:
            return
        points = getattr(coords, "points", None)
        height = getattr(coords, "layout_height", None)
        if not points or not height:
            return
        ys = [float(p[1]) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not ys:
            return
        meta["y0_rel"] = round(min(ys) / float(height), 4)
        meta["y1_rel"] = round(max(ys) / float(height), 4)
    except Exception:  # noqa: BLE001
        pass


def _safe_meta(el: Any) -> dict[str, Any]:
    try:
        md = el.metadata.to_dict()
    except Exception:  # noqa: BLE001
        return {}
    keep = {k: v for k, v in md.items() if isinstance(v, (str, int, float, bool))}
    return keep


def _html_to_text(html: str) -> str:
    """把 <table> HTML 转成 Markdown-ish 文本，供纯文本检索使用。"""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    lines = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        cells = [c for c in cells if c != ""]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


__all__ = ["UnstructuredParser"]
