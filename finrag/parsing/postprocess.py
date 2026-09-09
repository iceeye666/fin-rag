"""解析后处理：跨页表格合并 + 章节上下文挂载。

这两个步骤直接对应金融年报的两个典型痛点：

1. **跨页上下文丢失**：一张「合并利润表」常常横跨 2~3 页，逐页切分后
   后半部分缺少表头，向量检索时语义残缺。这里按「列数一致 + 首行相同」
   的规则把跨页表格重新拼接，并在每个分片上补全表头。

2. **指代消解缺失**：年报正文中大量「上述情况」「该公司」等指代表述，
   脱离章节标题后无法被检索到。这里把最近的上级标题写入元素元数据，
   切分时拼进 chunk 前缀（见 chunking/ 模块）。
"""

from __future__ import annotations

import re

from finrag.parsing.base import ParsedElement

_TOTAL_PATTERN = re.compile(r"合\s*计|小\s*计|总\s*计|总计")


def merge_cross_page_tables(
    elements: list[ParsedElement], max_gap_pages: int = 1
) -> list[ParsedElement]:
    """把跨页被拆开的表格合并为一个元素。

    判据优先级（从强到弱，任一命中即合并，且必须满足「页码相邻 + 列数相同」）：

    1. **位置证据（最强）**：前表触到页底（y1_rel ≥ 0.80）且后表顶到页顶
       （y0_rel ≤ 0.20）。这是物理事实，几乎不会误判。
    2. **重复表头**：后表首行与前表首行完全一致——排版引擎（reportlab / Word）
       在跨页续表时会自动复写表头，这是续表的强信号。
    3. **未完结**：前表末行不是「合计/总计/小计」——说明表格尚未收尾。

    合并时若首行重复，会丢弃续表的表头行，避免向量中出现冗余。
    """
    out: list[ParsedElement] = []
    pending: ParsedElement | None = None

    for el in elements:
        if el.is_table and pending is not None:
            if _is_continuation(pending, el, max_gap_pages):
                pending = _concat_tables(pending, el)
                continue
            out.append(pending)
            pending = el
            continue

        if el.is_table:
            pending = el
            continue

        if pending is not None:
            out.append(pending)
            pending = None
        out.append(el)

    if pending is not None:
        out.append(pending)
    return out


def _is_continuation(prev: ParsedElement, nxt: ParsedElement, max_gap: int) -> bool:
    gap = (nxt.page or 0) - (prev.page or 0)
    if not (0 <= gap <= max_gap):
        return False
    if _n_cols(nxt) != _n_cols(prev):
        return False

    # 1) 位置证据：前表贴页底 + 后表贴页顶
    prev_bottom = prev.meta.get("y1_rel")
    next_top = nxt.meta.get("y0_rel")
    if prev_bottom is not None and next_top is not None:
        if float(prev_bottom) >= 0.80 and float(next_top) <= 0.20:
            return True

    # 2) 重复表头
    if _first_row(prev) and _first_row(prev) == _first_row(nxt):
        return True

    # 3) 前表未以合计行收尾
    return not _is_total_row(_last_row(prev))


def _last_row(el: ParsedElement) -> str:
    rows = [r for r in el.text.split("\n") if r.strip()]
    return re.sub(r"\s+", "", rows[-1]) if rows else ""


def split_oversized_table(
    el: ParsedElement, max_rows: int = 20, header_rows: int = 1
) -> list[ParsedElement]:
    """超大表格按行切片，每片都带上表头，避免单块过长被截断。"""
    rows = [r for r in el.text.split("\n") if r.strip()]
    if len(rows) <= max_rows + header_rows:
        return [el]

    header = rows[:header_rows]
    body = rows[header_rows:]
    chunks = []
    for i in range(0, len(body), max_rows):
        part = body[i : i + max_rows]
        text = "\n".join(header + part)
        chunks.append(
            ParsedElement(
                category="table",
                text=text,
                html=el.html,
                page=el.page,
                source=el.source,
                raw_category=el.raw_category,
                meta={**el.meta, "table_part": len(chunks) + 1, "split": True},
            )
        )
    return chunks


def attach_section_context(elements: list[ParsedElement]) -> list[ParsedElement]:
    """把最近的一级/二级标题写入每个元素的 ``section`` 元数据。"""
    current_h1 = ""
    current_h2 = ""
    for el in elements:
        if el.category == "title":
            level = el.meta.get("font_size", 0) or 0
            # 无法可靠区分层级时，用「当前 h1 是否已存在」做二级判定
            if not current_h1:
                current_h1 = el.text.strip()
            else:
                current_h2 = el.text.strip()
            el.meta["section"] = current_h1
            continue
        el.meta["section"] = current_h1
        el.meta["subsection"] = current_h2
    return elements


# --------------------------------------------------------------------------- #
def _n_cols(el: ParsedElement) -> int:
    if el.meta.get("n_cols"):
        return int(el.meta["n_cols"])
    first = el.text.split("\n", 1)[0]
    return len([c for c in first.split("|") if c.strip()])


def _first_row(el: ParsedElement) -> str:
    return re.sub(r"\s+", "", el.text.split("\n", 1)[0])


def _is_total_row(row: str) -> bool:
    return bool(_TOTAL_PATTERN.search(row))


def _concat_tables(a: ParsedElement, b: ParsedElement) -> ParsedElement:
    dup_header = _first_row(a) == _first_row(b)
    text = a.text.rstrip() + "\n" + (b.text.split("\n", 1)[1] if dup_header else b.text.lstrip())
    return ParsedElement(
        category="table",
        text=text,
        html=_merge_html(a.html, b.html, drop_header=dup_header),
        page=a.page,
        source=a.source,
        raw_category=a.raw_category,
        meta={
            **a.meta,
            "n_rows": int(a.meta.get("n_rows", 0)) + int(b.meta.get("n_rows", 0)),
            "cross_page": True,
            "page_span": [a.page, b.page],
            # 合并后端部位置沿用后表，保证三段续表能链式合并
            "y1_rel": b.meta.get("y1_rel", a.meta.get("y1_rel")),
        },
    )


def _merge_html(a: str | None, b: str | None, drop_header: bool = True) -> str | None:
    if not a:
        return b
    if not b:
        return a
    rows_b = re.findall(r"<tr.*?</tr>", b, re.S)
    if drop_header and rows_b and re.search(r"<th", rows_b[0], re.I):
        rows_b = rows_b[1:]  # 去掉续表重复表头
    return a.replace("</table>", "".join(rows_b) + "</table>")


__all__ = ["merge_cross_page_tables", "split_oversized_table", "attach_section_context"]
