"""端到端冒烟测试：不依赖 API Key，验证解析 → 切分 → 入库 → 检索 → 生成全链路。

运行：python -m tests.test_pipeline
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finrag.chunking.chunker import chunk_elements  # noqa: E402
from finrag.config import Settings  # noqa: E402
from finrag.parsing.factory import parse_pdf  # noqa: E402

TMP_STORE = ROOT / "storage" / "_test"


def _cfg() -> Settings:
    """测试用配置：离线可用（hash 嵌入 + mock 生成），存储隔离。"""
    cfg = Settings(
        embed_provider="hash",
        llm_provider="mock",
        parse_backend="pdfplumber",
        chroma_persist_dir=str(TMP_STORE / "chroma"),
        docstore_dir=str(TMP_STORE / "docstore"),
        cache_dir=str(TMP_STORE / "cache"),
    )
    cfg.chroma_collection = "test_collection"
    cfg.ensure_dirs()
    return cfg


def _sample_pdf() -> Path:
    pdfs = sorted((ROOT / "data" / "samples").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("请先运行：python -m scripts.cli gen-sample")
    return pdfs[0]


def test_parse_and_chunk(cfg: Settings) -> list:
    pdf = _sample_pdf()
    elements, parser = parse_pdf(pdf, cfg)
    assert elements, "解析结果为空"
    cats = {e.category for e in elements}
    assert "table" in cats, f"未解析出表格：{cats}"
    assert "title" in cats, f"未解析出标题：{cats}"

    chunks = chunk_elements(elements, cfg)
    assert chunks, "切分结果为空"
    assert any(c.category == "table" for c in chunks), "未产生表格块"
    assert all(c.chunk_id for c in chunks), "存在空 chunk_id"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id 重复"

    print(f"  [ok] 解析 {len(elements)} 元素 / 切分 {len(chunks)} 块（{parser.name}）")
    return chunks


def test_index_and_retrieve(cfg: Settings) -> None:
    from finrag.pipeline import FinancialRAGPipeline

    pipe = FinancialRAGPipeline(cfg, verbose=False)
    stats = pipe.ingest([_sample_pdf()], rebuild=True)
    assert stats.n_chunks > 0, "入库失败"

    pairs = pipe.retrieve("2025 年营业收入是多少？")
    assert pairs, "检索无结果"
    docs = [d.page_content for d, _ in pairs]
    joined = "\n".join(docs)
    assert "12,846,351,207.44" in joined, (
        f"未召回营业收入数值，实际召回：\n{joined[:400]}"
    )

    res = pipe.ask("公司 2025 年的营业收入是多少？")
    assert res.answer, "答案为空"
    assert res.citations, "无引用来源"

    print(f"  [ok] 入库 {stats.n_chunks} 块 / 召回命中营业收入 / 引用 {len(res.citations)} 条")
    print(f"       答案：{res.answer[:100]}")


def test_cross_page_table(cfg: Settings) -> None:
    """验证跨页表格被合并成一张完整的表。"""
    from finrag.parsing.postprocess import merge_cross_page_tables

    elements, _ = parse_pdf(_sample_pdf(), cfg)
    merged = merge_cross_page_tables(elements)
    cross = [e for e in merged if e.meta.get("cross_page")]
    if cross:
        print(f"  [ok] 跨页合并 {len(cross)} 处："
              f"{[(e.meta.get('page_span'), e.meta.get('n_rows')) for e in cross]}")
    else:
        print("  [--] 本文档未出现跨页表格（合并逻辑未被触发）")


def main() -> int:
    print("== 高级 RAG 金融年报 —— 冒烟测试 ==")
    if TMP_STORE.exists():
        shutil.rmtree(TMP_STORE)
    cfg = _cfg()
    try:
        test_parse_and_chunk(cfg)
        test_cross_page_table(cfg)
        test_index_and_retrieve(cfg)
    finally:
        if TMP_STORE.exists():
            shutil.rmtree(TMP_STORE)
    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
