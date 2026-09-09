"""解析器工厂：按文档特征选择后端，失败自动降级。

选型结论（在本项目 10 页中文年报上实测）：

| 后端                | 耗时   | 表格结构         | 坐标 | 适用            |
|---------------------|--------|------------------|------|-----------------|
| pdfplumber          | ~2s    | 行列完整、可直接读 | 有   | 有文本层的 PDF  |
| unstructured hi_res | ~180s  | 中文被 OCR 拆成单字 | 无   | 扫描件 / 图片版 |

原因：hi_res 会把每页渲染成位图再走「布局检测 + OCR」，
对**已经有文本层**的中文 PDF 反而是负优化——OCR 把连续中文切成单字
（"公司中文名称" → "司 中文 名 称"），且耗时高出两个数量级。

因此 ``auto`` 策略是**先探测再决定**：
用 pdfplumber 抽样前 5 页统计字符密度，低于阈值（扫描件特征）才切换到
unstructured 的 hi_res；否则一律走 pdfplumber。
若确需强制 hi_res，设置 ``PARSE_BACKEND=unstructured`` 即可。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from finrag.config import Settings
from finrag.parsing.base import BaseParser, drop_noise
from finrag.parsing.pdfplumber_parser import PdfPlumberParser
from finrag.parsing.postprocess import attach_section_context, merge_cross_page_tables
from finrag.parsing.unstructured_parser import UnstructuredParser

# 每页平均字符数低于此值，判定为扫描件（无文本层）
SCANNED_PAGE_CHARS_THRESHOLD = 120


def build_parser(cfg: Settings | None = None) -> BaseParser:
    """返回默认解析器（不感知具体文档）。"""
    from finrag.config import get_settings

    cfg = cfg or get_settings()
    if cfg.parse_backend == "unstructured":
        return UnstructuredParser(cfg)
    return PdfPlumberParser()


def detect_backend(pdf_path: Path, cfg: Settings) -> BaseParser:
    """按文档特征探测并选择合适的解析器。"""
    if cfg.parse_backend == "unstructured":
        return UnstructuredParser(cfg)
    if cfg.parse_backend == "pdfplumber":
        return PdfPlumberParser()

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages[:5] or pdf.pages
            chars_per_page = [len(p.chars or []) for p in pages]
        avg = statistics.mean(chars_per_page) if chars_per_page else 0
        if avg < SCANNED_PAGE_CHARS_THRESHOLD:
            print(
                f"[parser] 每页平均仅 {avg:.0f} 个字符，判定为扫描件，"
                "切换 unstructured hi_res（OCR）"
            )
            return UnstructuredParser(cfg)
        print(f"[parser] 检测到文本层（每页约 {avg:.0f} 字符），使用 pdfplumber")
    except Exception as e:  # noqa: BLE001
        print(f"[parser] 探测失败（{type(e).__name__}: {e}），回退 pdfplumber")
    return PdfPlumberParser()


def parse_pdf(
    pdf_path: str | Path,
    cfg: Settings | None = None,
    use_cache: bool = True,
) -> tuple[list, BaseParser]:
    """解析 PDF，带磁盘缓存（hi_res 很慢，缓存收益极大）。

    返回 ``(elements, parser)``。
    """
    from finrag.config import get_settings

    cfg = cfg or get_settings()
    pdf_path = Path(pdf_path)
    cache_file = (
        cfg.cache_path / f"parsed_{pdf_path.stem}_{cfg.parse_backend}.json"
        if cfg.parse_cache_enabled
        else None
    )

    if use_cache and cache_file and cache_file.exists():
        try:
            from finrag.parsing.base import BaseParser as _BP

            elements = _BP.load(cache_file)
            print(f"[parser] 命中缓存：{cache_file.name}（{len(elements)} 元素）")
            return elements, build_parser(cfg)
        except Exception:  # noqa: BLE001
            pass

    parser = detect_backend(pdf_path, cfg)
    try:
        elements = parser.parse(pdf_path)
    except Exception as e:  # noqa: BLE001
        if isinstance(parser, PdfPlumberParser):
            raise
        print(f"[parser] {parser.name} 解析失败（{e}），降级 pdfplumber")
        parser = PdfPlumberParser()
        elements = parser.parse(pdf_path)

    # ---- 后处理：去噪 → 章节挂载 → 跨页表格合并 ----
    # 顺序不能颠倒：必须先剔除页眉页脚，否则它们会夹在续表之间阻断合并
    elements = drop_noise(elements)
    elements = attach_section_context(elements)
    elements = merge_cross_page_tables(elements)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps([e.to_dict() for e in elements], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return elements, parser


__all__ = ["build_parser", "parse_pdf", "detect_backend"]
