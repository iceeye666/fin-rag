"""FastAPI 服务：文档入库 / 问答（SSE 流式） / 索引状态。

接口一览
--------
POST /api/ingest      上传 PDF 并构建索引（multipart 或 {"paths": [...]}）
POST /api/ingest/local 构建 data/raw + data/samples 下的全部 PDF
POST /api/ask         问答，返回完整结果（含引用、置信度）
POST /api/ask/stream  问答，SSE 流式返回 {citations} → {delta}* → {done}
GET  /api/info        配置与索引状态
DELETE /api/index     清空索引
GET  /                前端页面
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from finrag.config import get_settings  # noqa: E402

app = FastAPI(
    title="高级 RAG 金融年报智能分析助手",
    description="半结构化 PDF 解析 → 多向量检索 → 严格约束生成 → 可溯源问答",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_state: dict[str, Any] = {"pipeline": None}


# --------------------------------------------------------------------------- #
def get_pipeline():
    """懒加载流水线（首次加载嵌入模型较慢）。"""
    if _state["pipeline"] is None:
        from finrag.pipeline import FinancialRAGPipeline

        cfg = get_settings()
        print("[api] 初始化流水线：加载嵌入模型 ...")
        _state["pipeline"] = FinancialRAGPipeline(cfg)
        print("[api] 就绪")
    return _state["pipeline"]


class AskRequest(BaseModel):
    question: str
    k: int | None = None
    show_context: bool = False


class PathsRequest(BaseModel):
    paths: list[str] | None = None
    rebuild: bool = False


# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def api_info():
    cfg = get_settings()
    info = {
        "config": {
            "llm_provider": cfg.llm_provider,
            "llm_model": cfg.llm_model,
            "embed_provider": cfg.embed_provider,
            "embed_model": cfg.embed_model,
            "parse_backend": cfg.parse_backend,
            "parse_strategy": cfg.parse_strategy,
            "chunk_size": cfg.chunk_size,
            "top_k_summary": cfg.top_k_summary,
            "top_k_final": cfg.top_k_final,
            "has_api_key": cfg.has_api_key,
        },
        "index": {"ready": _state["pipeline"] is not None},
    }
    try:
        pipe = get_pipeline()
        info["index"] = pipe.index_info()
        info["index"]["ready"] = True
        info["config"]["actual_embed"] = type(pipe.embeddings).__name__
        info["config"]["actual_llm"] = type(pipe.llm).__name__
        info["config"]["summary_mode"] = pipe.summarizer.mode
    except Exception as e:  # noqa: BLE001
        info["index"]["error"] = f"{type(e).__name__}: {e}"
    return info


@app.post("/api/ingest")
async def api_ingest(files: list[UploadFile] = File(default=[])):
    if not files:
        raise HTTPException(400, "未上传文件")
    cfg = get_settings()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            continue
        dest = cfg.raw_dir / f.filename
        with dest.open("wb") as fp:
            shutil.copyfileobj(f.file, fp)
        saved.append(dest)
    if not saved:
        raise HTTPException(400, "没有有效的 PDF 文件")

    pipe = get_pipeline()
    stats = pipe.ingest(saved, rebuild=False)
    return {"ok": True, "stats": stats.to_dict(), "index": pipe.index_info()}


@app.post("/api/ingest/local")
def api_ingest_local(req: PathsRequest):
    cfg = get_settings()
    if req.paths:
        pdfs = [Path(p) for p in req.paths]
    else:
        pdfs = sorted(cfg.sample_dir.glob("*.pdf")) + sorted(cfg.raw_dir.glob("*.pdf"))
    pdfs = [p for p in pdfs if p.exists()]
    if not pdfs:
        raise HTTPException(404, "未找到可入库的 PDF")

    pipe = get_pipeline()
    stats = pipe.ingest(pdfs, rebuild=req.rebuild)
    return {"ok": True, "stats": stats.to_dict(), "index": pipe.index_info()}


@app.post("/api/ask")
def api_ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题为空")
    pipe = get_pipeline()
    if pipe.is_empty:
        raise HTTPException(409, "索引为空，请先入库 PDF")
    try:
        res = pipe.ask(req.question, k=req.k)
        return res.to_dict()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/retrieve")
def api_retrieve(req: AskRequest):
    """调试用：返回召回的原文块（含分数），不调用 LLM。"""
    pipe = get_pipeline()
    pairs = pipe.retrieve(req.question, k=req.k)
    return {
        "query": req.question,
        "n": len(pairs),
        "chunks": [
            {
                "rank": i + 1,
                "score": round(s, 4),
                "page": (d.metadata or {}).get("page"),
                "category": (d.metadata or {}).get("category"),
                "has_revenue": "营业收入" in d.page_content,
                "content_head": d.page_content[:300],
            }
            for i, (d, s) in enumerate(pairs)
        ],
    }


@app.post("/api/ask/stream")
def api_ask_stream(req: AskRequest):
    pipe = get_pipeline()
    if pipe.is_empty:
        raise HTTPException(409, "索引为空，请先入库 PDF")

    def gen():
        try:
            for evt in pipe.stream(req.question, k=req.k):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.delete("/api/index")
def api_reset():
    pipe = get_pipeline()
    pipe.reset()
    return {"ok": True, "index": pipe.index_info()}


# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run() -> None:
    import uvicorn

    cfg = get_settings()
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port, log_level="info")


if __name__ == "__main__":
    run()
