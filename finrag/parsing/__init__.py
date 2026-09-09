"""PDF 解析子包。"""

from finrag.parsing.base import BaseParser, ParsedElement
from finrag.parsing.factory import build_parser, parse_pdf
from finrag.parsing.pdfplumber_parser import PdfPlumberParser
from finrag.parsing.postprocess import (
    attach_section_context,
    merge_cross_page_tables,
    split_oversized_table,
)
from finrag.parsing.unstructured_parser import UnstructuredParser

__all__ = [
    "ParsedElement",
    "BaseParser",
    "UnstructuredParser",
    "PdfPlumberParser",
    "build_parser",
    "parse_pdf",
    "merge_cross_page_tables",
    "split_oversized_table",
    "attach_section_context",
]
