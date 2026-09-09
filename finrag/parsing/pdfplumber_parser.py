"""pdfplumber 兜底解析后端（纯 Python，无系统依赖）。

思路：
1. ``page.find_tables()`` 定位表格并拿到精确 bbox；
2. ``page.extract_text_lines()`` 拿到带字号的行，剔除落在表格 bbox 内的行；
3. 剩余行按「字号是否显著大于全文中位数」判定 Title，其余按连续同格式聚合为段落；
4. 表格同时产出 HTML（保留结构，供前端渲染）与竖线文本（供向量检索）。

该后端在**非扫描版**年报上的标题/表格分离效果接近 hi_res，且速度快 1~2 个数量级，
因此非常适合作为 CI / 批量预处理的主力。
"""

from __future__ import annotations

import html as html_lib
import statistics
from pathlib import Path

import pdfplumber

from finrag.parsing.base import BaseParser, ParsedElement


class PdfPlumberParser(BaseParser):
    name = "pdfplumber"

    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        pdf_path = Path(pdf_path)
        out: list[ParsedElement] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            # 先扫描全文档统计「正文字号」。
            # 关键：必须排除表格内的行——年报里表格单元格字号通常最小且数量最多，
            # 若把它们计入中位数，会把正文全部误判成标题。
            all_sizes: list[float] = []
            for page in pdf.pages:
                tb = [t.bbox for t in (page.find_tables() or [])]
                for line in page.extract_text_lines(return_chars=True) or []:
                    if _in_any_bbox(line.get("top", 0), line.get("bottom", 0), tb):
                        continue
                    s = _line_size(line)
                    if s:
                        all_sizes.append(s)
            base_size = statistics.median(all_sizes) if all_sizes else 10.0

            for page_no, page in enumerate(pdf.pages, start=1):
                out.extend(self._parse_page(page, page_no, base_size, pdf_path.name))
        return out

    # ------------------------------------------------------------------ #
    def _parse_page(self, page, page_no: int, base_size: float, source: str):
        elements: list[ParsedElement] = []

        # ---- 表格 ----
        table_bboxes = []
        for tbl in page.find_tables() or []:
            rows = tbl.extract() or []
            rows = [_clean_row(r) for r in rows]
            rows = [r for r in rows if any(c for c in r)]
            if not rows:
                continue
            table_bboxes.append(tbl.bbox)
            elements.append(
                ParsedElement(
                    category="table",
                    text=_rows_to_text(rows),
                    html=_rows_to_html(rows),
                    page=page_no,
                    source=source,
                    raw_category="Table",
                    meta={
                        "n_rows": len(rows),
                        "n_cols": max(len(r) for r in rows),
                        # 记录相对位置，用于跨页续表判定（0=页顶，1=页底）
                        "y0_rel": round(float(tbl.bbox[1]) / page.height, 4),
                        "y1_rel": round(float(tbl.bbox[3]) / page.height, 4),
                    },
                )
            )

        # ---- 文本行（排除表格区域）----
        page_h = page.height or 841.89
        lines = page.extract_text_lines(return_chars=True) or []
        buf: list[dict] = []
        for line in lines:
            top, bottom = line.get("top", 0), line.get("bottom", 0)
            if _in_any_bbox(top, bottom, table_bboxes):
                continue
            size = _line_size(line) or base_size
            text = (line.get("text") or "").strip()
            if not text:
                continue
            # 页眉页脚：位于页面最上/最下的短行。
            # 必须单独识别——它们会夹在两个续表之间，阻断跨页表格合并，
            # 且混入索引后会产生大量与内容无关的噪声块。
            if (bottom / page_h > 0.945 or top / page_h < 0.045) and len(text) <= 60:
                _flush(buf, base_size, elements, page_no, source)
                buf = []
                elements.append(
                    ParsedElement(
                        category="footer",
                        text=text,
                        page=page_no,
                        source=source,
                        raw_category="Footer",
                    )
                )
                continue
            # 用比例阈值：不同 PDF 的绝对字号差异很大，比例更稳
            is_title = size >= base_size * 1.15 or (
                len(text) <= 30 and size >= base_size * 1.06
            )
            if is_title:
                _flush(buf, base_size, elements, page_no, source)
                buf = []
                elements.append(
                    ParsedElement(
                        category="title",
                        text=text,
                        page=page_no,
                        source=source,
                        raw_category="Title",
                        meta={"font_size": size},
                    )
                )
            else:
                buf.append(line)
        _flush(buf, base_size, elements, page_no, source)
        return elements


# --------------------------------------------------------------------------- #
def _line_size(line: dict) -> float:
    """取得一行的有效字号。

    部分 PDF（如 reportlab 生成、或字体信息缺失的扫描件）在
    ``extract_text_lines`` 中不带 ``size``，此时依次回退到：
    字符级 size 均值 → 行高（标题行高通常明显更大）。
    """
    s = float(line.get("size") or 0)
    if s > 0:
        return s
    chars = line.get("chars") or []
    vals = [float(c.get("size") or 0) for c in chars if c.get("size")]
    if vals:
        return sum(vals) / len(vals)
    top, bottom = line.get("top") or 0, line.get("bottom") or 0
    return float(bottom - top)


def _flush(buf, base_size, elements, page_no, source):
    if not buf:
        return
    text = " ".join(l.get("text", "").strip() for l in buf).strip()
    if not text:
        return
    elements.append(
        ParsedElement(
            category="text",
            text=text,
            page=page_no,
            source=source,
            raw_category="NarrativeText",
            meta={"n_lines": len(buf)},
        )
    )


def _in_any_bbox(top: float, bottom: float, bboxes) -> bool:
    for bb in bboxes:
        if bb is None:
            continue
        _, b_top, _, b_bottom = bb
        if b_top is None or b_bottom is None:
            continue
        # 垂直重叠即视为属于表格（表格内文字也会被抽出，需剔除）
        overlap = min(bottom, b_bottom) - max(top, b_top)
        if overlap > 2:
            return True
    return False


def _clean_row(row):
    return [("" if c is None else str(c).replace("\n", " ").strip()) for c in row]


def _rows_to_text(rows) -> str:
    return "\n".join(" | ".join(c for c in r if c) for r in rows if any(r))


def _rows_to_html(rows) -> str:
    esc = html_lib.escape
    parts = ["<table>"]
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cells = "".join(f"<{tag}>{esc(c)}</{tag}>" for c in row)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "".join(parts)


__all__ = ["PdfPlumberParser"]
