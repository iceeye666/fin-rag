"""基于 Qwen-Plus 的批量摘要生成（多向量检索的第一步）。

为什么要「摘要索引 + 原文召回」：
* 直接把 800 字的原文块拿去做向量，语义被稀释，Top-K 召回常常命中错误的段落；
* 摘要只有 80~150 字，向量表征更聚焦，召回精度显著提升；
* 召回后返回的是**原文块**，LLM 拿到的是完整上下文，不损失细节。
这就是 MultiVectorRetriever 的核心价值：小粒度精准匹配 + 大上下文完整回答。

工程细节：
* 并发调用（ThreadPoolExecutor）+ 失败重试 + 单条降级，任何一条失败不阻塞整体；
* 摘要结果按内容哈希落盘缓存，重复构建索引零成本；
* 无 API Key 时自动切换到**规则摘要**（抽取含数字/指标的关键句），链路照常跑通。
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from finrag.config import Settings
from finrag.models.llms import MockExtractiveChatModel

SUMMARY_SYSTEM = (
    "你是一名资深证券分析师，擅长用最少的文字概括上市公司年报要点。"
)

SUMMARY_PROMPT = """请为下面这段来自上市公司年报的内容生成一段**检索用摘要**。

要求：
1. 长度 60~150 字，一句话概括核心信息；
2. 必须保留：主体（公司/业务/分部名称）、时间（年度/期间）、指标名、金额、单位、同比变动；
3. 若是表格，说明「这是一张什么表、包含哪些科目、覆盖哪些年度」；
4. 不要编造原文中不存在的数字；不要写「本文档介绍了…」这类空话；
5. 只用中文输出摘要正文，不要加引号、标题或解释。

内容：
{content}

摘要："""


class Summarizer:
    """批量摘要生成器。"""

    def __init__(self, cfg: Settings | None = None):
        from finrag.config import get_settings

        self.cfg = cfg or get_settings()
        self._cache_file = self.cfg.cache_path / "summaries.json"
        self._cache: dict[str, str] = {}
        if self._cache_file.exists():
            try:
                self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache = {}

        if self.cfg.has_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model=self.cfg.llm_model,
                    api_key=self.cfg.dashscope_api_key,
                    base_url=self.cfg.llm_base_url,
                    temperature=0.0,
                    max_tokens=300,
                )
                self.mode = "qwen-plus"
            except Exception as e:  # noqa: BLE001
                print(f"[summary] LLM 初始化失败，使用规则摘要：{e}")
                self.llm = None
                self.mode = "rule"
        else:
            self.llm = None
            self.mode = "rule"
            print("[summary] 未检测到 API Key，摘要使用规则模式（离线可用）")

    # ------------------------------------------------------------------ #
    def summarize_one(self, text: str) -> str:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        if key in self._cache:
            return self._cache[key]

        content = text[: self.cfg.summary_max_chars]
        if self.mode == "qwen-plus":
            summary = self._call_llm(content)
        else:
            summary = rule_summary(content)
        summary = (summary or "").strip() or rule_summary(content)
        self._cache[key] = summary
        return summary

    def _call_llm(self, content: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        for attempt in range(3):
            try:
                resp = self.llm.invoke(
                    [
                        SystemMessage(content=SUMMARY_SYSTEM),
                        HumanMessage(content=SUMMARY_PROMPT.format(content=content)),
                    ]
                )
                return re.sub(r"\s+", " ", str(resp.content)).strip()
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"[summary] Qwen 调用失败，回退规则摘要：{e}")
        return ""

    # ------------------------------------------------------------------ #
    def summarize_batch(
        self, texts: list[str], progress: Callable[[int, int], None] | None = None
    ) -> list[str]:
        results: list[str] = [""] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            key = hashlib.md5(t.encode("utf-8")).hexdigest()[:16]
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                todo.append(i)

        if todo:
            workers = max(1, min(self.cfg.summary_concurrency, len(todo)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(self.summarize_one, texts[i]): i for i in todo
                }
                done = 0
                for fut in as_completed(future_map):
                    i = future_map[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[summary] 第 {i} 条摘要失败：{e}")
                        results[i] = rule_summary(texts[i])
                    done += 1
                    if progress:
                        progress(done, len(todo))
        self.flush()
        return results

    def flush(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\d[\d,\.]*\s*(亿元|万元|元|%|个百分点|亿美元|万美元)?")


def table_subject_summary(text: str, max_len: int = 150) -> str:
    """表格专用摘要：列出表内科目名。

    为什么表格不能只用「高信息密度句」：
    一张 30 行的资产负债表，信息密度最高的往往是前几行（货币资金等），
    结果摘要里根本不出现"资产总计"，问"总资产是多少"就永远召不回这一块。
    因此表格摘要改为**枚举科目名**，让每个科目都进入索引。
    """
    rows = [r.strip() for r in text.split("\n") if r.strip()]
    subjects: list[str] = []
    for r in rows:
        cell = r.split("|")[0].strip()
        cell = re.sub(r"^[\s　]*[一二三四五六七八九十]+、", "", cell)
        if not cell or len(cell) > 24:
            continue
        if cell in {"项目", "内容"} or cell in subjects:
            continue
        subjects.append(cell)
    if not subjects:
        return ""

    head = "表格，包含科目："
    picked, length = [], len(head)
    for s in subjects:
        if length + len(s) + 1 > max_len - 1:
            break
        picked.append(s)
        length += len(s) + 1
    return head + "、".join(picked) + "。"


def rule_summary(text: str, max_len: int = 150) -> str:
    """规则摘要：抽取信息密度最高的句子（离线兜底，保证链路可跑）。

    打分启发式：
    * 含数字/单位 → 财报关键信息密度高；
    * 含指标关键词（收入、利润、同比、资产、现金流…）→ 加权；
    * 位置靠前 → 轻微加权（年报通常先给结论）。

    表格（含 `|` 且行数 >= 4）走单独的科目枚举分支，见 table_subject_summary。
    """
    text = re.sub(r"[ \t]+", " ", text.strip())
    if "|" in text and text.count("\n") >= 3:
        s = table_subject_summary(text, max_len)
        if s:
            return s
    sentences = [s for s in re.split(r"(?<=[。；\n])", text) if len(s.strip()) > 4]
    if not sentences:
        return text[:max_len]

    keywords = (
        "营业收入 净利润 毛利 资产 负债 现金流 同比 增长率 净利 归母 扣非 EPS ROE "
        "毛利率 净利率 期间费用 研发 存货 应收 商誉 分红 股东 占比 亿元 万元"
    ).split()

    scored = []
    for i, s in enumerate(sentences):
        s = s.strip()
        score = 0.0
        score += 2.0 * len(_NUM.findall(s))
        score += sum(1.2 for k in keywords if k in s)
        score += 0.3 if i == 0 else 0.0
        score -= 0.05 * i
        scored.append((score, i, s))
    scored.sort(key=lambda x: (-x[0], x[1]))

    picked, length = [], 0
    for _, i, s in scored:
        if length + len(s) > max_len:
            break
        picked.append((i, s))
        length += len(s)
        if length >= max_len * 0.6:
            break
    picked.sort()
    out = "".join(s for _, s in picked).strip()
    return out or text[:max_len]


__all__ = ["Summarizer", "rule_summary", "SUMMARY_PROMPT"]
