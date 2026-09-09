#!/usr/bin/env python3
"""命令行入口：构建索引 / 交互式问答 / 检索调试。

用法示例
--------
    # 1) 生成模拟年报（可选）
    python -m scripts.cli gen-sample

    # 2) 构建索引（清空重建）
    python -m scripts.cli build --rebuild

    # 3) 交互式问答
    python -m scripts.cli ask

    # 4) 单条提问
    python -m scripts.cli ask -q "2025年营业收入是多少？"

    # 5) 只看检索结果（调试召回质量）
    python -m scripts.cli search -q "毛利率" -k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finrag.config import get_settings  # noqa: E402


def _default_pdfs(cfg) -> list[Path]:
    pdfs = sorted(cfg.sample_dir.glob("*.pdf")) + sorted(cfg.raw_dir.glob("*.pdf"))
    return pdfs


# --------------------------------------------------------------------------- #
def cmd_gen_sample(args) -> int:
    from scripts.gen_sample_report import build_pdf

    path = build_pdf()
    print(f"生成完成：{path}")
    return 0


def cmd_build(args) -> int:
    from finrag.pipeline import FinancialRAGPipeline

    cfg = get_settings()
    pdfs = [Path(p) for p in args.pdf] if args.pdf else _default_pdfs(cfg)
    if not pdfs:
        print(f"未在 {cfg.sample_dir} 或 {cfg.raw_dir} 找到 PDF，请先执行 gen-sample 或指定路径")
        return 1

    pipe = FinancialRAGPipeline(cfg)
    stats = pipe.ingest(pdfs, rebuild=args.rebuild)
    print("\n===== 入库统计 =====")
    for k, v in stats.to_dict().items():
        print(f"  {k}: {v}")
    print("\n===== 索引状态 =====")
    for k, v in pipe.index_info().items():
        print(f"  {k}: {v}")
    return 0


def cmd_ask(args) -> int:
    from finrag.pipeline import FinancialRAGPipeline

    cfg = get_settings()
    pipe = FinancialRAGPipeline(cfg, verbose=False)
    if pipe.is_empty:
        print("索引为空，请先执行 build")
        return 1

    if args.question:
        _print_answer(pipe, args.question, show_context=args.show_context)
        return 0

    print("进入交互模式（输入 exit / q 退出）\n")
    while True:
        try:
            q = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        _print_answer(pipe, q, show_context=args.show_context)
        print()
    return 0


def cmd_search(args) -> int:
    from finrag.pipeline import FinancialRAGPipeline

    cfg = get_settings()
    pipe = FinancialRAGPipeline(cfg, verbose=False)
    pairs = pipe.retrieve(args.question, k=args.k)
    print(f"查询：{args.question}  命中 {len(pairs)} 条\n")
    for i, (doc, score) in enumerate(pairs, 1):
        md = doc.metadata
        print(f"[{i}] score={score:.4f} | {md.get('citation')} | {md.get('category')}")
        print("    " + doc.page_content[:200].replace("\n", " "))
        print()
    return 0


def cmd_web(args) -> int:
    import uvicorn

    cfg = get_settings()
    print(f"启动 Web 服务：http://{cfg.api_host}:{cfg.api_port}")
    uvicorn.run("web.app:app", host=cfg.api_host, port=cfg.api_port, reload=args.reload)
    return 0


def cmd_info(args) -> int:
    from finrag.pipeline import FinancialRAGPipeline

    cfg = get_settings()
    pipe = FinancialRAGPipeline(cfg, verbose=False)
    print("===== 配置 =====")
    for k, v in cfg.model_dump().items():
        if "key" in k.lower() and v:
            v = f"{str(v)[:6]}******"
        print(f"  {k}: {v}")
    print("\n===== 索引 =====")
    for k, v in pipe.index_info().items():
        print(f"  {k}: {v}")
    return 0


# --------------------------------------------------------------------------- #
def _print_answer(pipe, question: str, show_context: bool = False) -> None:
    res = pipe.ask(question)
    print(f"\n问题：{res.question}")
    print(f"答案：{res.answer}")
    if show_context:
        print("\n--- 上下文片段 ---")
        for c in res.citations:
            print(f"[来源{c.index}] {c.citation}  score={c.score:.4f}")
            print(c.snippet)
            print()
    else:
        print("\n引用来源：")
        for c in res.citations:
            print(f"  [{c.index}] {c.citation}  (score={c.score:.3f})")
    print(f"\n耗时 {res.latency_ms}ms（检索 {res.retrieval_ms}ms）| 置信度 {res.confidence}")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="fin-rag", description="高级 RAG 金融年报智能分析助手"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gen-sample", help="生成模拟年报 PDF").set_defaults(func=cmd_gen_sample)

    b = sub.add_parser("build", help="构建向量索引")
    b.add_argument("--pdf", nargs="*", help="PDF 路径，默认取 data/samples 与 data/raw")
    b.add_argument("--rebuild", action="store_true", help="清空后重建")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="问答")
    a.add_argument("-q", "--question", help="直接提问；不填则进入交互模式")
    a.add_argument("--show-context", action="store_true", help="打印召回原文")
    a.set_defaults(func=cmd_ask)

    s = sub.add_parser("search", help="仅检索，调试召回")
    s.add_argument("-q", "--question", required=True)
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    w = sub.add_parser("web", help="启动 Web 服务")
    w.add_argument("--reload", action="store_true", help="开发模式热重载")
    w.set_defaults(func=cmd_web)

    sub.add_parser("info", help="查看配置与索引状态").set_defaults(func=cmd_info)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
