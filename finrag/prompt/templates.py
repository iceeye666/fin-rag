"""金融年报问答的 Prompt 工程。

目标是**把幻觉按死在 Prompt 里**。核心约束（每一条都对应一类真实事故）：

1. 只依据检索上下文回答 —— 对应「模型用预训练知识编造财务数据」；
2. 上下文没有就明确说"无法回答" —— 对应「宁可乱说也不说不知道」；
3. 数字必须与上下文逐字一致，不得换算/估算 —— 对应「单位换算出错、四舍五入造数据」；
4. 每个结论后标注引用编号 —— 对应「无法溯源，业务不敢采信」；
5. 区分"摘要性描述"与"表格精确值"，表格问题优先引用表格 —— 对应「用正文概括值冒充精确值」；
6. 不回答投资建议类问题 —— 对应合规风险。

Prompt 同时要求模型先输出「判断依据」再输出「结论」，利用链式思考提升
对多跳指标（如"毛利率同比变动"）的正确率。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

# 上下文分隔标记。
# 必须用这种「不可能出现在正文里」的唯一标记：早期版本用 <上下文>，
# 结果规则段与正文段边界不清，程序化解析（Mock 生成器 / 引用对齐）会把
# Prompt 规则也当成年报原文抽出来。
CONTEXT_START = "<<<CONTEXT>>>"
CONTEXT_END = "<<<END_CONTEXT>>>"

SYSTEM_TEMPLATE = """你是一名严谨的上市公司年报分析助手，只做"依据给定材料的事实回答"，不做预测、不做投资建议。

# 硬性规则
1. 【只依据上下文】你只能使用 <上下文> 中提供的内容作答。禁止使用任何上下文之外的知识、记忆或推测。
2. 【无据必答"无法回答"】如果 <上下文> 中找不到支撑信息，必须原样回答：
   "根据所提供的年报内容，无法回答该问题。" 然后简要说明缺少哪类信息。禁止勉强编造。
3. 【数字零加工】金额、比例、同比变动等数值必须与上下文**逐字一致**，禁止换算单位、
   四舍五入、心算推导。上下文写"1,234.56 万元"，你就写"1,234.56 万元"。
4. 【强制溯源】每个关键结论后用 [序号] 标注其依据来自哪一个上下文片段，序号与片段的 [来源N] 对应。
   表格数据优先引用表格片段，正文描述优先引用正文片段。
5. 【单位与口径】回答中必须带上单位（元/万元/亿元/%）和时间口径（2024年度/2025年上半年等）。
   若上下文同时存在多个年度，必须明确说明你引用的是哪一个年度。
6. 【合规边界】涉及"是否值得投资""股价预测""买卖建议"的问题，一律拒绝并提示"仅提供客观信息，不构成投资建议"。
7. 【语言与格式】使用简体中文。先给一句结论，再给 2~4 条要点；每段要点不超过 2 行。

# 上下文（年报原文片段，你的全部依据）
<<<CONTEXT>>>
{context}
<<<END_CONTEXT>>>
"""

HUMAN_TEMPLATE = """<问题>{question}</问题>

请按上述规则作答："""

CONDENSE_TEMPLATE = """给定一段对话历史和用户的后续问题，请把后续问题改写成一个**可以独立理解**的检索问题。
要求：保留原问题中的年份、公司、指标名、单位等全部限定条件；不要回答，只输出改写后的问题。

对话历史：
{chat_history}

后续问题：{question}

独立问题："""


def build_qa_prompt() -> ChatPromptTemplate:
    """主问答 Prompt。"""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            ("human", HUMAN_TEMPLATE),
        ]
    )


def build_condense_prompt() -> ChatPromptTemplate:
    """多轮对话的问题改写 Prompt（解决「那毛利率呢」这类指代）。"""
    return ChatPromptTemplate.from_template(CONDENSE_TEMPLATE)


# --------------------------------------------------------------------------- #
def format_context(docs, max_chars_per_doc: int = 1600) -> str:
    """把检索到的文档格式化为带编号来源的上下文块。

    表格片段额外标注 [表格]，提示模型这是结构化数据，需逐字引用。
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        cat = md.get("category", "text")
        tag = "表格" if cat == "table" else "正文"
        cite = md.get("citation", "")
        content = d.page_content
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "…（截断）"
        blocks.append(f"[来源{i} | {tag} | {cite}]\n{content}")
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "SYSTEM_TEMPLATE",
    "HUMAN_TEMPLATE",
    "CONDENSE_TEMPLATE",
    "CONTEXT_START",
    "CONTEXT_END",
    "build_qa_prompt",
    "build_condense_prompt",
    "format_context",
]
