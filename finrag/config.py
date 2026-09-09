"""全局配置：所有可调参数集中在此，支持 .env / 环境变量 / 代码覆盖。

设计要点：
1. 单一配置源（Single Source of Truth），避免魔法字符串散落各处；
2. 所有路径基于项目根解析，保证 CLI / Web / 测试行为一致；
3. 任何外部依赖（API Key、模型权重）都可在无网络环境下降级，见 models/ 目录。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根：fin-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str | Path) -> Path:
    """相对路径一律基于项目根解析。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 阿里云百炼 DashScope ----------------
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")

    # ---------------- 生成模型 ----------------
    llm_provider: Literal["dashscope", "mock"] = "dashscope"
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.1          # 金融场景要求稳定复现，温度极低
    llm_max_tokens: int = 2048
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---------------- 嵌入模型 ----------------
    embed_provider: Literal["bge", "dashscope", "hash"] = "bge"
    embed_model: str = "BAAI/bge-large-zh-v1.5"
    embed_device: str = "cpu"             # cpu / mps / cuda
    embed_batch_size: int = 16
    embed_normalize: bool = True

    # ---------------- PDF 解析 ----------------
    parse_backend: Literal["auto", "unstructured", "pdfplumber"] = "auto"
    parse_strategy: Literal["hi_res", "fast", "ocr_only"] = "hi_res"
    # hi_res 会把每页渲染成图片跑布局模型，很慢；大文档可先缓存解析结果
    parse_cache_enabled: bool = True

    # ---------------- 切分 ----------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    table_max_rows_per_chunk: int = 20    # 单个表格块最大行数，超出按行切片
    min_chunk_chars: int = 30             # 过短噪声块直接丢弃

    # ---------------- 摘要 ----------------
    summary_max_chars: int = 700          # 单个 chunk 参与摘要的最大字符数
    summary_concurrency: int = 4          # Qwen-Plus 批量摘要并发度
    summary_enabled: bool = True

    # ---------------- 检索 ----------------
    top_k_summary: int = 8                # 摘要向量层召回条数（候选池放大，避免数值表块被 MMR 挤掉）
    top_k_final: int = 6                  # 最终进入 LLM 上下文的条数（财报数值类问题需保留更多表块）
    score_threshold: float = 0.0
    fetch_k: int = 30                     # MMR 候选池
    enable_mmr: bool = True               # 最大边际相关性，去冗余
    mmr_lambda: float = 0.7               # 偏向相关性（数值查证类问题，少做多样性发散）
    enable_rerank: bool = False           # 精排（true 时加载 bge-reranker-large）

    # ---------------- 存储 ----------------
    chroma_persist_dir: str = "storage/chroma"
    chroma_collection: str = "annual_report_multivector"
    docstore_dir: str = "storage/docstore"
    cache_dir: str = "storage/cache"

    # ---------------- 服务 ----------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ---------------- 路径快捷方式 ----------------
    @property
    def chroma_dir(self) -> Path:
        return _abs(self.chroma_persist_dir)

    @property
    def docstore_path(self) -> Path:
        return _abs(self.docstore_dir)

    @property
    def cache_path(self) -> Path:
        return _abs(self.cache_dir)

    @property
    def data_dir(self) -> Path:
        return _abs("data")

    @property
    def raw_dir(self) -> Path:
        return _abs("data/raw")

    @property
    def sample_dir(self) -> Path:
        return _abs("data/samples")

    def ensure_dirs(self) -> None:
        for d in (self.chroma_dir, self.docstore_path, self.cache_path,
                  self.data_dir, self.raw_dir, self.sample_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def has_api_key(self) -> bool:
        return bool(self.dashscope_api_key) and not self.dashscope_api_key.startswith("sk-xxxx")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免重复读取 .env。"""
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
