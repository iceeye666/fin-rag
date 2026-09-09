# 高级 RAG 金融年报智能分析助手

面向上市公司年报的**可溯源**智能问答系统。用自然语言提问年报里的文本、表格与财务指标，
返回带页码/章节/表格出处的答案；无依据时明确拒答，而不是编一个看起来合理的数字。

```
问题 ──► 多向量检索（摘要向量 → 原文召回）──► 重排压缩 ──► Qwen-Plus（严格约束）──► 答案 + 引用
              ▲
              │
PDF ──► 结构化解析 ──► 文本/表格双流切分 ──► Qwen-Plus 批量摘要 ──► Chroma 持久化
```

---

## 1. 快速开始

```bash
# 1) 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) 配置（可选，不填也能跑通全流程，会自动降级）
cp .env.example .env
#   把 DASHSCOPE_API_KEY 填进去

# 3) 生成模拟年报（已内置一份，可跳过）
python -m scripts.cli gen-sample

# 4) 构建索引
python -m scripts.cli build --rebuild

# 5) 提问
python -m scripts.cli ask -q "2025 年营业收入是多少？"

# 6) 启动 Web 界面
python -m scripts.cli web        # 或 uvicorn web.app:app --reload
#    浏览器打开 http://127.0.0.1:8000
```

> 首次运行会下载 `BAAI/bge-large-zh-v1.5`（约 1.3 GB）。
> 国内网络可设 `HF_ENDPOINT=https://hf-mirror.com` 加速。

---

## 2. 目录结构

```
fin-rag/
├── finrag/
│   ├── config.py                  # 全局配置（.env / 环境变量 / 代码覆盖）
│   ├── pipeline.py                # 端到端编排：入库 + 问答（主入口）
│   ├── models/
│   │   ├── embeddings.py          # BGE / DashScope / Hash 三级降级嵌入
│   │   └── llms.py                # Qwen-Plus / Mock 抽取式生成器
│   ├── parsing/
│   │   ├── base.py                # 统一中间表示 ParsedElement
│   │   ├── unstructured_parser.py # hi_res 后端（扫描件）
│   │   ├── pdfplumber_parser.py   # 默认后端（有文本层）
│   │   ├── postprocess.py         # 跨页表格合并 / 超大表切片 / 章节挂载
│   │   └── factory.py             # 文档类型探测 + 自动选型 + 缓存
│   ├── chunking/chunker.py        # 文本流 / 表格流双路切分
│   ├── summarize/summarizer.py    # Qwen-Plus 并发摘要 + 规则降级
│   ├── retrieval/
│   │   ├── multivector.py         # MultiVectorRetriever（摘要索引 + 原文召回）
│   │   └── rerank.py              # 关键词 / BGE-Reranker 精排
│   ├── prompt/templates.py        # 反幻觉 Prompt 工程
│   └── chain/qa_chain.py          # 检索 → 重排 → 压缩 → 生成 → 溯源
├── scripts/
│   ├── cli.py                     # 命令行：build / ask / search / web
│   └── gen_sample_report.py       # 生成 11 页模拟年报 PDF（含跨页大表）
├── web/                           # FastAPI + 单页 Web UI（SSE 流式）
├── data/samples/                  # 样例年报
└── storage/                       # chroma 向量库 / docstore / 缓存
```

---

## 3. 核心设计与踩坑

### 3.1 解析：为什么默认不是 unstructured hi_res

项目最初按 hi_res 实现，实测 10 页中文年报后改了默认策略：

| 后端                | 耗时   | 表格结构            | 坐标 | 适用             |
|---------------------|--------|---------------------|------|------------------|
| **pdfplumber**      | ~2 s   | 行列完整、可直接读   | 有   | 有文本层的 PDF   |
| unstructured hi_res | ~180 s | 中文被 OCR 切成单字 | 无   | 扫描件 / 图片版  |

hi_res 会把页面渲染成位图再走「布局检测 + OCR」。对已有文本层的中文 PDF，
OCR 把连续中文切成了单字（"公司中文名称" → `司 中文 名 称`），且慢两个数量级。

因此 `PARSE_BACKEND=auto` 会**先探测再决策**：抽样前 5 页统计字符密度，
低于 120 字符/页判定为扫描件才切 hi_res。两个后端都完整实现，随时可切。

其他解析细节：
* **正文字号基准必须排除表格行**——年报里表格单元格字号最小且数量最多，
  计入中位数后会把全部正文误判成标题（实测 55 个"标题" vs 实际 22 个）。
* **页眉页脚要单独标记**：它们会夹在两个续表中间，阻断跨页合并，
  混入索引后还会产生大量无关噪声块。

### 3.2 跨页表格合并

一张「合并资产负债表」横跨 3 页时，逐页切分会让后半部分缺表头，
向量检索时语义残缺——这是财报 RAG 最典型的召回失败原因。

判据按可信度排序（都需满足「页码相邻 + 列数相同」）：

1. **位置证据（最强）**：前表贴页底（`y1_rel ≥ 0.80`）且后表顶页顶（`y0_rel ≤ 0.20`），物理事实几乎不误判；
2. **重复表头**：排版引擎跨页时会自动复写表头，后表首行 == 前表首行即为续表；
3. **未完结**：前表末行不是「合计/总计/小计」，说明表格还没收尾。

> 踩坑：只看「末行是否含合计」会把 `流动资产合计 → 非流动资产合计` 这类分页点
> 误判成两张独立表，所以位置证据必须优先。

### 3.3 多向量检索：摘要建索引，原文做召回

```
摘要（60~150 字，语义密度高）──embed──► Chroma（metadata.doc_id）
                                              │ Top-K 相似
                                              ▼
                                     DocStore(JSON) ──► 原文块 / 完整表格 ──► LLM
```

* 直接把 800 字原文块向量化，语义被稀释，"同比增速是多少"这类指标型问题召回很漂；
* 摘要只有百来字，向量表征聚焦，命中率显著提升；
* 召回后返回的是**原文**，LLM 仍能看到完整表格与数字，不损失细节。

**BGE 的查询指令不能省**：`BAAI/bge-large-zh-v1.5` 要求查询侧加前缀
`为这个句子生成表示以用于检索相关文章：`，文档侧不加。两侧都加或都不加，
召回质量会明显下降——这是很多人用 BGE 效果差的首要原因，本项目在 `embed_query` 中严格区分。

### 3.4 Prompt：把幻觉按死在提示词里

每一条约束都对应一类真实事故：

| 约束 | 防的是什么 |
|------|-----------|
| 只依据 `<上下文>` 作答 | 用预训练知识编造财务数据 |
| 无据必须原样输出"无法回答" | 宁可乱说也不说不知道 |
| 数字逐字一致，禁止换算/四舍五入 | 单位换算出错、心算造数据 |
| 每个结论标注 `[序号]` 引用 | 无法溯源，业务不敢采信 |
| 区分正文描述与表格精确值 | 用正文概括值冒充精确值 |
| 拒绝投资建议类问题 | 合规风险 |

### 3.5 三级降级：任何环境都能跑通

| 组件 | 首选 | 降级 1 | 降级 2 |
|------|------|--------|--------|
| 嵌入 | BAAI/bge-large-zh-v1.5 | DashScope text-embedding-v3 | 字符 n-gram 哈希（零依赖） |
| 生成 | Qwen-Plus | — | Mock 抽取式（确定性） |
| 摘要 | Qwen-Plus 并发 | — | 规则摘要（信息密度打分） |
| 解析 | pdfplumber | unstructured hi_res | — |

没有 API Key、没有网络也能完整验证「解析 → 切分 → 摘要 → 检索 → 溯源」，
填入 Key 后无需改任何代码即切换真实模型。

---

## 4. 实测数据（内置样例年报）

```
解析：11 页 → 42 个元素（标题 22 / 正文 11 / 表格 9）
跨页：P8 + P9 合并为 1 张 30 行资产负债表
切分：24 个 chunk（正文 11 / 表格 13），平均 330 字
入库：24 条摘要向量 + 24 条原文块，约 6 s（不含模型加载）
```

真实问答命中情况（引用列省略，接口实际返回含页码与表格 HTML）：

| 问题 | 命中内容 |
|------|----------|
| 2025 年末总资产是多少？ | 合并资产负债表 P8：`资产总计 \| 21,376,540,118.92 \| 18,204,996,315.70` |
| 归属于上市公司股东的净利润是多少？ | 主要会计数据 P3：`1,295,826,441.05`，同比 31.10% |
| 工业机器人本体的收入和毛利率？ | 分产品表 P4：`3,846,405,362.23`、毛利率 `23.34%` |
| 经营活动现金流量净额是多少？ | 现金流量表 P10：`1,540,772,889.31` |
| 研发投入占营业收入比例？ | 研发表 P5：`11.86%`，同比增加 1.44 个百分点 |
| 综合毛利率是多少？ | 正文 P4：`30.46%`，较上年增加 0.51 个百分点 |

> 上表为 **Mock 抽取式生成**（无 API Key 时的降级模式）输出，用于验证
> 「检索是否命中正确的块」。填入 DashScope Key 后由 Qwen-Plus 生成，
> 会输出带 `[1]` 引用编号的完整自然语言答案。

---

## 5. 命令行

```bash
python -m scripts.cli gen-sample                    # 生成模拟年报
python -m scripts.cli build --rebuild               # 重建索引
python -m scripts.cli build --pdf a.pdf b.pdf       # 指定 PDF
python -m scripts.cli ask                           # 交互式
python -m scripts.cli ask -q "毛利率是多少？"        # 单条
python -m scripts.cli ask -q "..." --show-context   # 打印召回原文
python -m scripts.cli search -q "现金流" -k 5       # 只检索，调试召回
python -m scripts.cli info                          # 配置与索引状态
```

## 6. Web API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/info` | 实际生效的模型与索引状态 |
| POST | `/api/ingest` | 上传 PDF 入库（multipart） |
| POST | `/api/ingest/local` | 入库 data 目录下的 PDF |
| POST | `/api/ask` | 问答，返回答案 + 引用 + 置信度 |
| POST | `/api/ask/stream` | SSE 流式：`citations → delta* → done` |
| DELETE | `/api/index` | 清空索引 |

---

## 7. 面试讲解要点（30 秒版）

> "这个项目解决的是**金融年报问不准**的问题。三个关键设计：
> 一是解析层做了双后端自动选型，并且专门处理了跨页表格合并——一张资产负债表
> 横跨三页时，逐页切分会让后半部分丢表头，检索必然失败；
> 二是检索用多向量架构，摘要建索引保证匹配精度，原文做召回保证上下文完整，
> 小粒度精准匹配和大上下文完整回答两者兼得；
> 三是 Prompt 层把幻觉按死——只依据上下文、无据必答无法回答、数字逐字一致、
> 每个结论带引用编号，所以输出的每个数字都能点开溯源到具体页码和表格。"

**追问预案**：
- 为什么不用更大的 chunk？→ 语义稀释，且表格会被切断。
- BGE 还有什么坑？→ 查询侧指令前缀；向量必须归一化配合 cosine。
- 怎么评测？→ `scripts/cli.py search` 看召回命中；可外接 RAGAS 做 faithfulness / answer relevancy。
- 表格怎么不让 LLM 读串？→ 表格不参与文本切分，独立成块并打 `[表格内容开始/结束]` 标记。

---

## 8. 后续可优化

- [ ] 接入 `BAAI/bge-reranker-large` 做交叉编码精排（`ENABLE_RERANK=true`）
- [ ] 多轮对话的问题改写（模板已备好 `build_condense_prompt`）
- [ ] 父文档检索（small-to-big）：chunk 召回后返回所在章节全文
- [ ] 指标抽取结构化：先抽「科目-年度-数值-单位」三元组，再基于结构化数据做计算类问答
- [ ] 接入 RAGAS 做 faithfulness / context precision 自动化评测

---

## 9. 用真实大模型运行（已验证）

填入 `.env` 的 `DASHSCOPE_API_KEY` 后，`llm_provider` 自动切到 DashScope，无需改代码。
当前验证组合：`LLM_MODEL=qwen3.7-flash` + `EMBED_MODEL=BAAI/bge-large-zh-v1.5`。

**两个已踩坑并固化在代码里的问题（避坑）：**
1. **必须装 `langchain-openai`**——`build_chat_model` 通过 OpenAI 兼容模式调用 DashScope，
   漏装会导致 ImportError 被兜底成 Mock，表现成"配置都对却一直走规则抽取"。
2. **Qwen3 必须关思考模式**：默认开启的思考会把推理塞进 `reasoning_content`，
   `content` 偶发为空（个别问题空答）。已在 `models/llms.py` 用
   `model_kwargs={"extra_body": {"enable_thinking": False}}` 关闭。

**检索参数（针对财报数值类问题调优，索引无需重建）：**
`TOP_K_SUMMARY=8` / `TOP_K_FINAL=6` / `MMR_LAMBDA=0.7`——
`top_k_final=4` 会把含"营业收入"等数字的表块挤到第 5~6 位而被截断，真实 LLM 在上下文里
看不到数字就会如实拒答。放大候选池并让 MMR 偏向相关性后，营收/总资产/归母净利/工业机器人/
研发占比/分红/风险等问题均精确命中（置信度 0.82–0.90，带 `[来源]` 引用）。

调试接口：`POST /api/retrieve`（仅看召回块，不调 LLM）、`POST /api/ask/stream`（SSE 流式）。

