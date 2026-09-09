"""嵌入模型层（三级降级，保证任何环境都能跑通）。

优先级：
1. ``bge``       —— BAAI/bge-large-zh-v1.5（SentenceTransformers 本地推理，中文金融语义最优）
2. ``dashscope`` —— 通义 text-embedding-v3（OpenAI 兼容接口，无需本地权重）
3. ``hash``      —— 字符 n-gram 哈希嵌入（零依赖、确定性、离线可跑，词面召回能力可用）

关键工程细节：
* BGE 官方要求查询侧加指令前缀 ``"为这个句子生成表示以用于检索相关文章："``，
  文档侧不加。若两侧都加或都不加，召回质量会明显下降——这是很多 RAG 项目
  用 BGE 效果差的首要原因，本项目在 ``embed_query`` 中严格区分。
* 所有向量统一 L2 归一化，配合 Chroma 的 cosine 距离使用。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from langchain_core.embeddings import Embeddings

from finrag.config import Settings

# BGE 中文系列官方查询指令（zh 模型专用）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# --------------------------------------------------------------------------- #
# 1. BGE 本地嵌入
# --------------------------------------------------------------------------- #
class BGEEmbeddings(Embeddings):
    """BAAI/bge-large-zh-v1.5 本地嵌入封装。"""

    def __init__(self, cfg: Settings | None = None):
        from finrag.config import get_settings

        self.cfg = cfg or get_settings()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 sentence-transformers，请先 pip install sentence-transformers torch；"
                "或把 EMBED_PROVIDER 改为 dashscope / hash"
            ) from e

        self.model = SentenceTransformer(
            self.cfg.embed_model, device=self.cfg.embed_device
        )
        # sentence-transformers 新版重命名了该方法，兼容两种命名
        getter = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self.dim = getter() or 1024

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.embed_batch_size,
            normalize_embeddings=self.cfg.embed_normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vecs]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 文档侧：不加指令前缀
        return self._encode([t.replace("\n", " ").strip() for t in texts])

    def embed_query(self, text: str) -> list[float]:
        # 查询侧：加 BGE 官方指令前缀
        return self._encode([BGE_QUERY_INSTRUCTION + text.strip()])[0]


# --------------------------------------------------------------------------- #
# 2. DashScope 嵌入
# --------------------------------------------------------------------------- #
class DashScopeEmbeddings(Embeddings):
    """通义千问 text-embedding-v3（OpenAI 兼容模式）。"""

    def __init__(self, cfg: Settings | None = None, model: str = "text-embedding-v3"):
        from finrag.config import get_settings

        self.cfg = cfg or get_settings()
        if not self.cfg.has_api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法使用 DashScope 嵌入")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.cfg.dashscope_api_key, base_url=self.cfg.llm_base_url
        )
        self.model = model
        self.dim = 1024

    def _encode(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.model, input=[t[:8000] for t in texts]
        )
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 10):  # 接口限制 batch
            out.extend(self._encode(texts[i : i + 10]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


# --------------------------------------------------------------------------- #
# 3. 零依赖哈希嵌入（离线兜底）
# --------------------------------------------------------------------------- #
class HashNGramEmbeddings(Embeddings):
    """字符 n-gram 哈希嵌入。

    把文本切成 2/3-gram，经 MD5 映射到固定维度并做 L2 归一化。
    虽无深层语义，但对中文「词面重叠」型查询（如"营业收入是多少"）具备
    稳定召回能力，且完全确定性、可复现，适合作为 CI / 无网络环境的兜底。
    """

    def __init__(self, dim: int = 1024, ngrams: Sequence[int] = (2, 3)):
        self.dim = dim
        self.ngrams = tuple(ngrams)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r"\s+", "", text.lower())
        toks = list(text)
        # 数字单独成 token，避免"2024"/"2025"被完全抹平
        return toks

    def _vec(self, text: str) -> list[float]:
        toks = self._tokenize(text)
        vec = [0.0] * self.dim
        for n in self.ngrams:
            for i in range(len(toks) - n + 1):
                gram = "".join(toks[i : i + n])
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_embeddings(cfg: Settings | None = None, override: str | None = None) -> Embeddings:
    """按配置构建嵌入模型，失败自动降级到下一级。"""
    from finrag.config import get_settings

    cfg = cfg or get_settings()
    order = [override or cfg.embed_provider]
    if "bge" not in order:
        order.append("bge")
    order += ["dashscope", "hash"]

    last_err: Exception | None = None
    for prov in dict.fromkeys(order):
        try:
            if prov == "bge":
                return BGEEmbeddings(cfg)
            if prov == "dashscope":
                return DashScopeEmbeddings(cfg)
            if prov == "hash":
                return HashNGramEmbeddings()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[embeddings] {prov} 初始化失败，降级：{type(e).__name__}: {e}")
    raise RuntimeError(f"所有嵌入后端均不可用：{last_err}")


def embedding_dim(emb: Embeddings) -> int:
    return getattr(emb, "dim", 1024)


__all__ = [
    "BGEEmbeddings",
    "DashScopeEmbeddings",
    "HashNGramEmbeddings",
    "BGE_QUERY_INSTRUCTION",
    "build_embeddings",
    "embedding_dim",
]
