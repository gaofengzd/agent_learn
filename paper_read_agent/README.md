# 研究生论文精读 Agent

这是一个面向学术论文精读的本地 RAG 应用：导入一篇或多篇 PDF 后，可以按指定论文或全库提问、生成不同层级的总结、提取方法，并区分作者明示贡献与 Agent 推断的潜在创新。所有可回答结论都应落到原文证据；证据不足时明确回答不知道或部分回答。

## 已实现功能与业务规则

- PDF 上传、去重、处理状态、失败重试、强制重处理和级联删除。
- 原生文字解析，疑似扫描页按页调用免费本地 RapidOCR；混合 PDF 不整篇 OCR。
- 单论文和多论文问答、多轮会话、会话级检索范围及范围变更记录。
- 简要、标准、详细三档总结；方法角色与缺失字段；作者贡献和 Agent 推断分开呈现。
- 回答状态分为充分、部分回答和证据不足。冲突证据并列展示，不静默裁决。
- 每条事实性结论绑定内部 `evidence_id`，可展开论文、页码、章节、原文片段和来源类型。
- GLM-4.7 只接收检索后证据；不得使用模型常识补齐原文没有的信息，不得伪造页码或引用。

本项目不生成复习题，不处理非 PDF，不做开放域聊天、论文写作、创新评分、联网学术搜索、多人协作、自动文件夹同步或公网部署。

## 技术架构

```text
FastAPI + Jinja UI
        │
        ▼
应用服务：论文处理 / 会话 / QA / 总结 / 方法 / 创新 / 删除与恢复
        │
        ├── 文档流水线：预检 → Docling → 按页 RapidOCR → 规范化 → 质量评估 → 父子 chunk
        ├── 检索流水线：查询规划 → Chroma 向量召回 + SQLite FTS5 → RRF → 本地重排 → 上下文
        ├── 生成与校验：GLM-4.7 → 结构化输出 → 充分性/引用校验 → 可信响应
        └── 持久化：原 PDF + 解析快照 + SQLite 元数据/关键词索引 + Chroma 向量
```

模块边界位于 `src/paper_read_agent/`：`document_pipeline` 负责文件内容，`retrieval` 负责召回与上下文，`llm` 是模型边界，`application` 编排用例，`persistence` 管理 SQLite/Chroma 一致性，`ui` 只负责本地页面和 façade 接口。

页面通过 `UIFacade` 与业务服务解耦。默认 ASGI 入口从项目 `.env` 读取配置并创建 `LocalUIFacade`，初始化本地目录、SQLite、上传服务和后台处理队列；新 PDF 会依次进入 Docling、按页 RapidOCR、质量评估、父子 chunk、FTS5 与 Chroma。问答、总结、方法提取和创新分析共用该 façade 中按需加载的本地检索、证据注册与 GLM 服务图。

## 数据流

1. 上传时检查扩展名、MIME、PDF 魔数、文件大小、页数、加密和损坏；用内容哈希去重，并先保存原 PDF。
2. Docling 提取原生文字和结构。低文本密度的疑似扫描页交给本地 RapidOCR；每页保留 `native_pdf` 或 `ocr` 来源。
3. 规范化为 paper/version/page/block，保留章节路径、表格、公式、页码与解析质量；解析快照可复现。
4. 按结构切分父子 chunk，写入 SQLite 元数据/FTS5 和 Chroma。只有索引全部成功的论文才进入 `ready`。
5. 问题先判定范围和意图，再生成检索查询。向量和关键词两路召回经 RRF 融合、本地 reranker 精排。
6. 上下文构建器优先放直接证据，再补父块和相邻上下文，并保留约 40%–60% 的 token 预算给证据。
7. GLM-4.7 返回固定 JSON；校验器检查结构、证据 ID、范围、冲突和充分性，然后生成带引用或明确拒答的响应。

删除顺序按 Chroma、FTS、业务记录、文件执行，并记录补偿信息；中途失败不得把残留数据宣称为已清除。

## Chunk 策略

- 子 chunk 默认 300–500 tokens，用于精确召回。
- 父 chunk 默认 1000–1500 tokens，用于补充完整语境。
- 优先按章节、段落、页、表格、公式和参考文献边界切分；表格、公式不与普通段落混合。
- 仅当单个结构块超过上限而被强制切分时使用 15% overlap，普通相邻块不机械重叠。
- chunk 保存 paper/version/page range/section/content type/parent/adjacency/index version，确保引用和重建稳定。

## 检索方式

默认候选上限 50，精排保留 12：

1. Chroma 使用本地 `bge-large-zh-v1.5` 做语义召回。
2. SQLite FTS5/BM25 做关键词召回，补足术语、缩写、公式名和专有名词。
3. Reciprocal Rank Fusion 合并两路排名；某一路不可用时标记降级，不伪装成完整检索。
4. 本地 `bge-reranker-v2-m3` 对候选重排；按论文范围、版本和质量门槛过滤。
5. 返回子块证据，并按预算扩展父块/邻块。多论文问题保留来源归属，版本和结论冲突不合并消失。

CPU 环境可通过 `PAPER_AGENT_RERANK_INPUT_LIMIT` 独立限制送入精排的融合候选数；默认 24，且必须满足“精排输出数 ≤ 精排输入数 ≤ 混合召回候选数”。该限制只控制计算预算，不改变 HybridRetriever 的融合候选集合。

## Prompt 与可信回答设计

系统 Prompt 的核心约束是：只使用 `Evidence`，只引用输入中的 `evidence_id`，不编造事实/页码/引用，证据不足则保留未回答项。任务 Prompt 携带问题、实际论文范围和序列化证据。模型输出必须包含回答状态、简短答案、逐条 claim 及证据、直接/推断/冲突/不支持标记、不确定性、冲突、未回答项和拒答理由。

生成后还会执行确定性校验：引用必须存在且属于实际范围；直接结论需要直接证据；推断必须明确标注；不支持的 claim 被移除或触发拒答。没有有效证据时回答“根据当前原文证据，我不知道”，而不是尝试补全。

## 安装与配置

要求 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```powershell
uv sync --extra dev
```

默认本地模型路径：

```text
E:\00project\02agent\models\bge-large-zh-v1.5
E:\00project\02agent\models\bge-reranker-v2-m3
```

可通过环境变量覆盖。不要把密钥写进仓库：

```powershell
$env:PAPER_AGENT_GLM_API_KEY = "你的密钥"
$env:PAPER_AGENT_GLM_MODEL = "glm-4.7"
$env:PAPER_AGENT_EMBEDDING_MODEL_PATH = "D:\models\bge-large-zh-v1.5"
$env:PAPER_AGENT_RERANKER_MODEL_PATH = "D:\models\bge-reranker-v2-m3"
$env:PAPER_AGENT_DATA_DIR = "D:\paper-agent-data"
```

默认数据目录为项目下 `data/`：SQLite 在 `data/paper_agent.sqlite3`，原 PDF 在 `data/pdfs/`，解析快照在 `data/parsed/`，Chroma 在 `data/chroma/`，日志在 `data/logs/`。`.env`、`data/` 和日志均被 Git 忽略。

## 启动与测试

启动本地 Web 应用：

```powershell
uv run uvicorn paper_read_agent.ui.app:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。三个页面名称为“论文库”“问答”“阅读分析”。默认入口可以真实上传论文并在后台解析、OCR 和建立索引；问答与阅读分析会调用真实检索和 GLM 服务。首次加载 Docling、OCR、embedding 或 reranker 模型可能较慢。

运行稳定测试集：

```powershell
uv run --frozen --no-sync pytest -m "not real_glm"
```

显式验证真实 GLM（会联网并产生 API 用量）：

```powershell
$env:ZHIPUAI_API_KEY = "你的密钥"
uv run --frozen --no-sync pytest -m real_glm tests/test_real_glm.py
```

验收清单位于 `tests/e2e/acceptance_cases.json`，性能烟雾门槛位于 `tests/e2e/PERFORMANCE_BASELINE.md`。它们用于回归，不是对所有硬件的 SLA。

## 隐私与安全边界

PDF、解析文本、向量、会话和日志默认留在本地数据目录；embedding、reranker 与 OCR 都在本地运行。只有经过检索和预算裁剪的证据片段会发送给 GLM API，因此这不是完全离线系统。使用者必须确认论文许可和机构数据政策。日志会做密钥脱敏，但仍不应记录整篇论文、完整 Prompt 或 Authorization header。删除操作覆盖应用管理的存储，不等同于安全擦除磁盘或删除上游 API 日志。

## 已知失败案例

- 模糊、旋转、手写或复杂扫描页会让 RapidOCR 漏字；低质量页会告警，严重时拒答。
- 极端双栏、跨页表格、嵌套脚注和复杂公式可能被 Docling 错排，引用页码正确也不代表公式语义完整。
- 中文 FTS 分词可能产生过宽候选；依赖向量召回和重排缓解，但专有缩写仍可能漏召回。
- 小型论文库不足以支持“领域首创”等全局判断；系统只报告当前比较范围。
- CPU 首次加载 embedding/reranker 较慢且占内存；资源门槛触发时任务应排队或失败，不静默换用在线模型。
- GLM 超时、限流、空响应或无效 JSON 会重试后明确失败；不会退回无证据自由回答。
- 跨 Chroma、FTS、SQLite 和文件的删除不是原子事务，中断时需要根据补偿记录重试。

## 后续改进方向

- 增加真实、可再分发的复杂论文小样本和人工标注的检索/引用指标。
- 改善中文关键词切词、跨页表格重建、公式表示与版面阅读顺序。
- 提供后台任务进度推送、取消任务和更细粒度资源监控。
- 增加模型量化、批量 embedding、索引迁移和跨存储一致性修复工具。
- 在不改变可信边界的前提下，评估本地生成模型以支持完全离线模式。
