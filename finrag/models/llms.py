"""生成模型层：Qwen-Plus（DashScope）为主，Mock 抽取式为兜底。

为什么要 Mock：
    金融年报链路的**瓶颈在检索而非生成**。提供确定性 Mock 生成器后，
    即使没有 API Key 也能完整验证「解析 → 切分 → 摘要 → 检索 → 溯源」，
    便于 CI 与面试现场演示；填入 Key 后无需改任何代码即切换真实模型。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from finrag.config import Settings


class MockExtractiveChatModel(BaseChatModel):
    """确定性抽取式生成器。

    不使用任何外部模型，而是：
    1. 从 messages 中还原「上下文」与「问题」；
    2. 用字符 n-gram 相似度给上下文句子打分；
    3. 取 Top-N 句子拼接成答案，并强制输出引用标记。

    这样即使完全离线，也能端到端验证 Prompt 拼接、引用溯源与链路编排。
    """

    max_sentences: int = 4
    prefix: str = "[MOCK 生成] "

    @property
    def _llm_type(self) -> str:
        return "mock-extractive"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 上下文在 SystemMessage、问题在 HumanMessage，因此需要在**全部**消息里
        # 分别定位，只在 HumanMessage 里找会永远拿不到上下文。
        all_text = "\n".join(str(m.content) for m in messages)
        question = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                mq = re.search(r"<问题>(.*?)</问题>", str(m.content), re.S)
                question = (mq.group(1) if mq else str(m.content)).strip()
                break
        if not question:
            question = all_text.strip()

        mc = re.search(
            r"<<<CONTEXT>>>(.*?)<<<END_CONTEXT>>>", all_text, re.S
        )
        context = mc.group(1).strip() if mc else ""

        answer = self._extract_answer(context, question)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    # ------------------------------------------------------------------ #
    def _extract_answer(self, context: str, question: str) -> str:
        if not context.strip():
            return self.prefix + "未检索到相关内容，根据已知信息无法回答该问题。"

        # 按行/句切分，过滤引用标题行与表格包裹标记
        units: list[str] = []
        for raw in context.split("\n"):
            line = raw.strip()
            if not line or line.startswith("[来源") or line.startswith("---"):
                continue
            if line.startswith("[表格内容"):
                continue
            units.extend([s for s in re.split(r"(?<=[。；;])", line) if len(s.strip()) > 6])
        if not units:
            return self.prefix + "未检索到相关内容，根据已知信息无法回答该问题。"

        q_tokens = self._tokens(question)
        q_nums = _numbers(question)          # 已剔除年份，避免"2025"命中一切

        def score_of(u: str) -> float:
            ut = self._tokens(u)
            if not ut:
                return 0.0
            coverage = len(q_tokens & ut) / (len(q_tokens) + 1e-9)
            u_nums = _numbers(u)
            num_hit = len(q_nums & u_nums) / (len(q_nums) + 1e-9) if q_nums else 0.0
            # 「数字 + 单位」共现是财报答案的强特征
            unit_bonus = 0.20 if re.search(r"\d[^，。]{0,8}(亿元|万元|元|%)", u) else 0.0
            return 2.0 * coverage + 0.8 * num_hit + unit_bonus

        # 表格行优先：财报数值类问题的答案几乎都在表格行里，
        # 与 Prompt 中「表格数据优先」的约束保持一致。
        rows = [u for u in units if "|" in u and re.search(r"\d", u)]
        pool = rows or units

        scored = []
        for i, u in enumerate(pool):
            s = score_of(u)
            if s < 0.20:
                continue
            scored.append((s, i, u))
        if not scored and rows:               # 表格行整体不相关则退回全文
            pool = units
            scored = [
                (s, i, u)
                for i, u in enumerate(pool)
                if (s := score_of(u)) >= 0.20
            ]
        scored.sort(key=lambda x: (-x[0], x[1]))

        if not scored:
            return self.prefix + "未检索到相关内容，根据已知信息无法回答该问题。"
        top = [u for _, _, u in scored[: self.max_sentences]]
        return f"{self.prefix}{''.join(top)}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        t = re.sub(r"[\s，。、；：（）()%“”\"'|]", "", text)
        grams = {t[i : i + 2] for i in range(len(t) - 1)}
        # 数字同样切成 2-gram，但年份会被 _numbers 单独剔除
        return grams | set(re.findall(r"\d+\.?\d*", text))


def _numbers(text: str) -> set[str]:
    """提取「有意义的数值」，剔除年份。

    年报里"2025""2024"几乎出现在每一句，若把它们计入数字命中，
    任何含年份的句子都会拿到满分，把真正的指标行挤下去。
    """
    out = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", text):
        if re.fullmatch(r"(?:19|20)\d{2}", n):
            continue
        out.add(n)
    return out


# --------------------------------------------------------------------------- #
def build_chat_model(cfg: Settings | None = None) -> BaseChatModel:
    """按配置构建聊天模型，失败降级到 Mock。"""
    from finrag.config import get_settings

    cfg = cfg or get_settings()

    if cfg.llm_provider == "mock" or not cfg.has_api_key:
        if cfg.llm_provider != "mock":
            print("[llm] 未检测到 DASHSCOPE_API_KEY，自动降级为 Mock 生成模型")
        return MockExtractiveChatModel()

    try:
        from langchain_openai import ChatOpenAI

        # Qwen3 系列默认开启「思考模式」，推理内容会进入 reasoning_content，
        # 而 content 可能为空——对 RAG 抽取式问答是无意义的开销且会导致空答。
        # 通过 OpenAI SDK 的 extra_body 显式关闭思考，只取最终答案。
        model_kwargs: dict = {"extra_body": {"enable_thinking": False}}
        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.dashscope_api_key,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            model_kwargs=model_kwargs,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[llm] ChatOpenAI 初始化失败，降级 Mock：{e}")
        return MockExtractiveChatModel()


__all__ = ["MockExtractiveChatModel", "build_chat_model"]
