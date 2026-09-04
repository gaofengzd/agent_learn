# 项目问题与解决方法记录

本文记录项目开发和真实 PDF 端到端测试中遇到的问题、影响范围、处理方式及对应代码位置。记录以实际出现的问题为准；尚未完成的事项会明确标记为“遗留”。

## 1. 默认 `LocalUIFacade` 未接入阅读服务

### 问题概述

QA、分层总结、方法提取和创新分析的应用服务及页面已经存在，但默认运行入口中的 `LocalUIFacade` 只实现了论文上传和处理。`qa_view()`、`analysis_view()` 仍返回占位数据，也没有实现页面调用的问答与分析操作。

### 对哪些功能有影响

- 默认 Web 应用无法执行真实多轮问答。
- 总结、方法提取和创新分析页面只能展示测试 façade 数据。
- 会话创建、删除、范围切换和引用展示无法在默认运行时形成完整闭环。

### 解决方法

- 新增 `LocalReadingServices`，统一装配本地 embedding、Chroma、SQLite FTS5、RRF、reranker、Evidence、GLM 和生成后校验服务。
- 在 `LocalUIFacade` 中按首次阅读请求懒加载服务，避免仅打开论文库就加载大型模型。
- 接入会话生命周期、QA、总结、方法、创新及引用展示。
- 更新 README，移除“默认 façade 尚未接线”的过期说明。

### 修改的代码位置

- `src/paper_read_agent/application/local_reading.py`
- `src/paper_read_agent/ui/local_facade.py`
- `README.md`
- `tests/test_readme.py`

## 2. 失败论文的状态、质量和提示互相矛盾

### 问题概述

论文处理失败后，页面可能同时显示“处理失败”“质量：ready”和“处理中，暂不可阅读”。原因是 UI 直接显示此前保存的质量等级，并把所有不可读状态统一渲染成“处理中”。

### 对哪些功能有影响

- 论文库状态可信度。
- 用户判断应该等待、重试还是检查错误的能力。
- 失败恢复流程的可理解性。

### 解决方法

- 失败状态优先于历史质量值，失败论文统一显示 `quality=failed`。
- 为不可读论文提供状态相关提示：失败时显示“处理失败，暂不可阅读”，排队或解析时才显示“处理中，暂不可阅读”。
- 保留旧测试 façade 的默认提示兼容性。

### 修改的代码位置

- `src/paper_read_agent/ui/local_facade.py`：`list_papers()`
- `src/paper_read_agent/ui/templates/library.html`
- `tests/test_local_facade.py`
- `tests/test_library_ui.py`

## 3. 页面点击删除后无法删除论文

### 问题概述

Web 删除路由会调用 `facade.delete_paper()`，但默认 `LocalUIFacade` 没有实现该方法，因此点击删除会进入异常分支，论文和相关数据不会被清除。

### 对哪些功能有影响

- 论文库删除按钮。
- 失败论文清理和重新上传。
- PDF、SQLite、FTS5 与 Chroma 之间的数据一致性。

### 解决方法

- 将已有 `PaperDeletionService` 接入默认 façade。
- 删除顺序覆盖 Chroma、FTS5、PDF 文件、历史引用标记和 SQLite 业务记录。
- 新增轻量 Chroma 删除适配器；当论文尚未建立向量集合时按空索引处理，避免失败论文无法删除。
- 保留处理中任务禁止删除的安全检查。

### 修改的代码位置

- `src/paper_read_agent/ui/local_facade.py`：`delete_paper()`、`_ChromaDeleteAdapter`
- `src/paper_read_agent/application/deletion.py`
- `src/paper_read_agent/ui/web.py`：删除路由
- `tests/test_deletion.py`
- `tests/test_local_facade.py`

## 4. Docling 首次运行找不到本地模型快照

### 问题概述

首次处理 `KG2RAG.pdf` 时，Docling 需要的模型不在 Hugging Face 本地缓存；网络受限环境下抛出 `LocalEntryNotFoundError`，论文进入失败状态。

### 对哪些功能有影响

- PDF 主解析。
- 后续 OCR 合并、标准化、chunk、索引及全部阅读功能。
- 新环境首次启动体验。

### 解决方法

- 首次运行时允许 Docling 联网下载缺失的官方模型快照。
- 下载完成后继续使用本地缓存；后续处理不再依赖重复下载。
- 将该情况视为明确的外部依赖失败，不降级为无结构自由解析。

### 修改的代码位置

- `src/paper_read_agent/document_pipeline/docling_parser.py`
- `src/paper_read_agent/application/ingestion.py`：`LocalDocumentIngestionProcessor`
- 模型缓存位于用户 Hugging Face cache，不写入 Git 仓库。

## 5. 同一文档中的重复 provenance 生成相同 `block_id`

### 问题概述

Docling 的一个内容项可能包含同页多个 provenance。原 ID 由版本、页码、Docling order、类型、文本和来源生成；当同一项的文本和 order 相同，仅 bbox 不同时，会产生相同 `block_id`，写入数据库时报错：

```text
IntegrityError: UNIQUE constraint failed: content_blocks.block_id
```

### 对哪些功能有影响

- 文档标准化结果持久化。
- 失败重试和重新上传。
- chunk、FTS5、Chroma 以及所有下游阅读功能。

### 解决方法

- 在稳定 ID 输入中加入候选排序后的 `position`。
- 同一输入重复运行仍产生相同 ID，但同页重复文本或多 provenance 获得不同 ID。
- 增加包含两个同页 provenance 的稳定性和唯一性回归测试。

### 修改的代码位置

- `src/paper_read_agent/document_pipeline/normalizer.py`：`DocumentNormalizer.normalize()`
- `tests/test_normalizer.py`：`test_repeated_provenance_blocks_have_stable_unique_ids`

## 6. Embedding 输入超过 BGE 的 512-token 上限

### 问题概述

真实论文生成的部分 chunk 经 tokenizer 编码后达到 1331 tokens。`LocalBGEEmbedder` 虽设置了 `truncation=True`，但没有给出确定的 `max_length`，导致模型位置张量与输入长度不一致：

```text
RuntimeError: The size of tensor a (1331) must match the size of tensor b (512)
```

### 对哪些功能有影响

- Chroma 向量索引建立。
- 论文无法从解析完成进入最终 `ready`。
- 语义召回、QA、总结、方法和创新分析。

### 解决方法

- 为 `LocalBGEEmbedder` 增加默认 `max_length=512`。
- tokenizer 调用显式使用 `truncation=True, max_length=512`。
- 对非法的非正数长度配置提前报错。

### 修改的代码位置

- `src/paper_read_agent/retrieval/vector_index.py`：`LocalBGEEmbedder`
- `tests/test_vector_index.py`

## 7. 首次 CPU 向量化耗时较长但容易被误判为挂死

### 问题概述

`KG2RAG.pdf` 产生 248 个 chunk。本地 `bge-large-zh-v1.5` 在 CPU 上首次加载和批量向量化期间长时间没有新日志，但进程持续消耗 CPU，最终任务正常完成。

### 对哪些功能有影响

- 用户对处理进度的判断。
- 大型论文首次导入体验。
- 若用户误操作重启，可能造成额外恢复成本。

### 解决方法

- 当前通过任务状态、进程响应和数据库产物数量区分“运行中”与“挂死”。
- 处理期间保持 `running`，完成后切换为 `succeeded`；本次真实测试最终得到 13 页、367 blocks、248 chunks。
- 后续建议增加阶段级进度、批次进度推送和任务取消能力。

### 修改的代码位置

- `src/paper_read_agent/application/processing_tasks.py`
- `src/paper_read_agent/application/ingestion.py`
- `src/paper_read_agent/ui/local_facade.py`：任务进度映射

## 8. RapidOCR 健康检查误报

### 问题概述

健康检查通过模块名 `rapidocr_onnxruntime` 判断 OCR 是否安装，但实际项目使用的 `rapidocr` 能成功加载 ONNXRuntime 引擎和三个本地模型。因此页面报告“RapidOCR engine dependency is not installed”，实际 OCR 初始化却成功。

### 对哪些功能有影响

- 系统健康状态的准确性。
- 用户可能误以为扫描 PDF 一定无法处理。
- 不影响本次 `KG2RAG.pdf` 的 RapidOCR 模型加载。

### 解决方法

- 将健康检查依赖名从 `rapidocr_onnxruntime` 改为实际适配器导入的 `rapidocr`。
- 增加回归测试，确认健康报告包含 `rapidocr`，且不再查询旧模块名。
- 浏览器验证健康状态现显示 `rapidocr` 可用。OCR 模型加载与逐页执行仍由真实处理流程验证。

### 修改的代码位置

- `src/paper_read_agent/application/system_health.py`：`REQUIRED_DEPENDENCIES`
- `src/paper_read_agent/document_pipeline/ocr.py`
- `tests/test_system_health.py`

## 10. CPU reranker 的候选预算与请求取消（遗留）

### 问题概述

本地 `bge-reranker-v2-m3` 在 CPU 上对 12 个真实 chunk 执行单查询重排约需 30.6 秒。复合问题会按原问题与拆分子问题多次评分；此前多个请求排队时，一次三查询问答超过了 180 秒，客户端取消后服务器计算仍继续。

### 已完成的改进

- 保留 `_LimitedReranker` 的候选限流，并将输入上限与最终输出上限拆成独立配置 `PAPER_AGENT_RERANK_INPUT_LIMIT`。
- 按当前验收要求，默认输入上限为 24，最终输出上限仍为 12。
- 配置校验要求 `rerank_result_limit <= rerank_input_limit <= candidate_limit`，并补充截断行为和边界测试。

### 当前结论与后续建议

输入 24 能让 RRF 排名 13–24 的候选参与语义重排，召回机会优于只重排前 12 个，但 CPU 耗时预计增加。当前以任务最终完成和证据正确为验收目标，不设置问答时长上限。同步请求取消后仍继续计算的问题尚未解决；后续可改为可取消的后台任务，或在带人工相关性标注的检索集上评估量化、ONNX 和更小的 reranker。

### 修改的代码位置

- `src/paper_read_agent/application/local_reading.py`
- `src/paper_read_agent/config.py`
- `tests/test_local_reading.py`
- `tests/test_config.py`

## 9. Windows 测试环境依赖安装不完整

### 问题概述

执行 `uv sync --extra dev` 时，Windows 在安装 `accelerate` 可执行包装器阶段出现 PE resource 错误，导致 `.venv` 中 pytest 的传递依赖不完整，先后缺少 `pluggy`、`httpx/httpx2`、`attrs` 等包。

### 对哪些功能有影响

- pytest 收集阶段。
- FastAPI/Starlette `TestClient`。
- 不代表业务测试断言失败，但会阻止回归测试运行。

### 解决方法

- 使用 `uv pip install --python .venv/Scripts/python.exe ...` 补齐缺失的测试依赖。
- 依赖补齐后运行完整非真实 GLM 回归测试并通过。
- 若再次出现，优先修复虚拟环境或重新创建 `.venv`，不要把系统 Python 的结果误认为项目环境结果。

### 修改的代码位置

- `pyproject.toml`：`[project.optional-dependencies].dev`
- `uv.lock`
- `.venv/` 为本地环境，不提交到 Git。

## 最终真实测试结果

使用文件：`E:\07文献\大模型\KG2RAG.pdf`

- 删除旧失败论文：成功。
- 重新上传：成功，重复数为 0。
- Docling、RapidOCR、标准化、质量评估、chunk、FTS5 和 Chroma：完成。
- 最终任务状态：`succeeded`。
- 最终论文状态：`ready`，质量：`ready`。
- 处理产物：13 页、367 个内容块、248 个 chunk。
- Web 页面显示“可用”并提供“开始阅读”入口。

## 11. 分析任务与问答任务的 JSON 结构约束冲突

### 问题概述

总结、方法提取和创新分析曾复用问答专用的系统提示。系统提示强制返回
`answer_status`、`claims` 等问答字段，而任务提示又要求返回
`sections`、`methods` 或 `author_claims`，形成互相冲突的输出要求。
真实测试中表现为方法字段结构异常、空响应或结果退化为大量
`not_stated`。

### 解决方法

- 将“只能使用给定 Evidence、不得虚构 evidence_id”的通用约束拆为
  `EVIDENCE_SYSTEM`。
- 问答继续使用带完整问答 JSON schema 的 `TRUSTED_SYSTEM`。
- 总结、方法和创新改用 `evidence_messages()`，只继承证据边界，再由各自
  的任务提示规定结构。
- 增加回归测试，确认非问答提示不会再强制包含 `answer_status`。

### 修改的代码位置

- `src/paper_read_agent/llm/glm_client.py`
- `src/paper_read_agent/application/local_reading.py`
- `tests/test_glm_client.py`

## 12. 泛化分析检索无法保证论文各章节的证据覆盖

### 问题概述

总结、方法和创新原先分别只用 `summary`、`method`、`innovation`
进行一次泛化检索。真实 KG2RAG 测试中，方法提取第一次虽返回 HTTP 200，
却主要引用实验设置块，KG2RAG 核心方法的目标、模块和工作流仍被标记为
未说明；标准总结也曾错误声称摘要不可用。

### 解决方法

- 为三类分析任务定义任务专属的检索维度。
- 将 24 个 reranker 输入预算分配到不同维度，避免每个维度都重排 24 个
  候选而成倍增加 CPU 开销。
- 从 SQLite 按论文版本读取 chunk，并依据 Abstract、Introduction、
  Methodology、Experiments、Conclusion、Limitations 等章节路径进行
  确定性回填。
- 合并时按 chunk 去重，优先保留每个分析维度的最佳证据，最终上下文仍
  遵守 12 个结果上限。

### 真实验证

修复后 KG2RAG 核心方法能够提取目标、输入输出、假设、模块、工作流、
数据、参数和评估，并引用 Methodology、2.1、2.2、2.3 与 Experiment
Setup。论文未明确提供的训练目标仍保持 `not_stated`。

### 修改的代码位置

- `src/paper_read_agent/application/local_reading.py`
- `src/paper_read_agent/persistence/repositories.py`
- `tests/test_persistence.py`

## 13. 长篇详细总结空响应且缺少可诊断信息

### 问题概述

KG2RAG 详细总结此前连续三次得到空响应。取消客户端超时只能允许请求继续
运行，不能降低一次性生成长 JSON 的失败概率。空响应日志也没有包含模型
结束原因和 token 使用情况。

### 解决方法

- 按必需章节分别生成摘要、引言、方法、实验、结论和局限，再合并为一个
  `SummaryResult`，避免单次输出过长。
- 对标准总结同样按章节生成，保证每个必需部分都有独立结果。
- 空响应错误加入安全的 `finish_reason` 和 `usage` 信息，不记录请求
  内容或 API Key。
- 保留最多三次的空响应、格式和连接异常重试，分析请求不设置主动超时。

### 真实验证

- KG2RAG 标准总结成功生成摘要、方法、结果和局限，均带 evidence_id。
- KG2RAG 详细总结成功生成六个必需章节，HTTP 200，页面刷新后结果仍可见。
- KG2RAG 创新分析成功输出六条作者明示贡献并绑定证据；单论文没有本地
  可比对象，因此未生成 Agent 比较性推断。

### 修改的代码位置

- `src/paper_read_agent/application/local_reading.py`
- `tests/test_local_reading.py`

## 14. QA 默认 60 秒客户端超时会中断复杂问答

### 问题概述

分析类任务已经允许无限等待，但问答使用的 `GLMClient` 仍曾默认设置
60 秒超时。复杂问题需要较长的本地 CPU 重排和远程生成时间，因此可能在
模型能够完成前被客户端中止。

### 解决方法

- `GLMClient` 的默认超时改为 `None`。
- 增加 `PAPER_AGENT_QA_TIMEOUT` 独立配置；未配置以及配置为
  `none`、`null` 或 `off` 时均表示不设置客户端超时。
- 如部署环境需要限制等待时间，可配置正数秒数；零和负数会在启动配置
  校验时被拒绝。
- 默认本地服务将该配置明确传入 QA 客户端，且测试确认 transport 收到
  的 timeout 为 `None`。

### 修改的代码位置

- `src/paper_read_agent/config.py`
- `src/paper_read_agent/llm/glm_client.py`
- `src/paper_read_agent/application/local_reading.py`
- `tests/test_config.py`
- `tests/test_glm_client.py`
- `README.md`

## 15. 长分析占用同步 HTTP 请求导致页面假死

### 问题概述

总结、方法提取和创新分析原先直接在页面 POST 请求中执行。本地 reranker
和远程生成可能运行数分钟，浏览器或自动化控制层会在服务器完成之前回收
连接，用户也无法区分任务仍在运行、已经失败或页面失去响应。

### 解决方法

- 增加进程内单工作线程分析队列，POST 只负责校验范围和提交任务，随后
  立即返回分析页面。
- 任务记录 `queued`、`running`、`succeeded`、`failed` 状态，
  并展示排队中、运行中、已完成或具体失败信息。
- 有活动任务时页面每三秒自动刷新；完成后停止自动刷新并展示原有分析
  结果和证据。
- 对任务状态、分析结果和引用分别增加线程同步保护，避免后台写入与页面
  刷新同时发生时读取到不一致状态。
- 当前队列只使用一个 worker，避免多个分析同时争用本地 CPU reranker。

### 当前边界

任务和结果的服务重启恢复已在后续的 SQLite 持久化修复中完成，任务取消
和相同请求去重见第 17 节。

### 修改的代码位置

- `src/paper_read_agent/ui/local_facade.py`
- `src/paper_read_agent/ui/web.py`
- `src/paper_read_agent/ui/templates/base.html`
- `src/paper_read_agent/ui/templates/analysis.html`
- `tests/test_analysis_tasks.py`
- `tests/test_analysis_ui.py`

## 18. 已运行 migration 4 的数据库缺少取消字段

### 问题概述

任务取消功能初版直接修改了 migration 4 的建表语句。全新数据库可以正常
工作，但已经执行过 migration 4 的本地数据库不会重新运行同一版本；代码
读取 `cancel_requested` 时会报字段不存在，且旧表状态约束不允许写入
`cancelled`。

### 解决方法

- 保持 migration 4 为最初发布的后台任务表结构，避免改变已发布迁移含义。
- 新增 migration 5，在事务中重建分析任务表，加入取消标记和
  `cancelled` 状态约束。
- 将旧任务及其结果 JSON、时间和状态完整复制到新表，旧记录的取消标记
  初始化为 0，然后重建索引。

### 测试覆盖

回归测试先构造一个已执行 migration 1–4 且包含成功分析结果的数据库，再
运行正常初始化；验证 schema 升级到 v5，任务状态和结果未丢失，并获得
`cancel_requested=0`。

### 修改的代码位置

- `src/paper_read_agent/persistence/database.py`
- `tests/test_persistence.py`

## 16. 服务重启后分析任务、结果和引用丢失

### 问题概述

后台化初版将任务状态、总结、方法、创新结果及引用保存在
`LocalUIFacade` 内存中。刷新页面可以恢复显示，但服务重启后所有分析
状态和结果都会消失；重启前处于运行中的任务还可能被用户误认为仍会继续。

### 解决方法

- 增加 SQLite migration 4 和 `analysis_tasks` 表，保存任务类型、论文
  ID、活动版本 ID、总结层级、状态、消息、结果 JSON 和生命周期时间。
- 后台任务提交、开始、成功和失败状态均写入 SQLite；成功时将分析结果及
  引用作为一个结果快照保存。
- 服务启动时按时间恢复成功结果，并合并总结、方法、作者贡献和 Agent
  推断结果。
- 恢复前校验论文仍存在且活动版本与任务记录完全一致；论文重新处理后不
  展示旧版本分析。
- 上次服务遗留的 `queued` 或 `running` 任务统一标记为失败，并显示
  “服务重启，未完成任务已中断”，不伪装为仍在执行。

### 测试覆盖

- 队列对象销毁并重新创建后仍可读取成功结果。
- 重启恢复时能够识别并结束遗留运行任务。
- 完整重建 `LocalUIFacade` 后，同一论文版本的总结和 evidence 引用仍
  能恢复。
- 数据库迁移可重复执行，最新 schema 版本自动更新。

### 修改的代码位置

- `src/paper_read_agent/persistence/database.py`
- `src/paper_read_agent/ui/local_facade.py`
- `tests/test_analysis_tasks.py`
- `tests/test_local_facade.py`
- `README.md`

## 17. 重复分析挤占 CPU，页面离开后无法取消

### 问题概述

用户重复点击同一种分析会创建多个内容、论文版本和总结层级完全相同的
任务，依次占用本地 CPU reranker。页面请求虽然已经后台化，但用户无法
取消误提交或不再需要的任务，运行中的任务仍会继续后续检索和生成步骤。

### 解决方法

- 提交前按分析类型、论文 ID、活动版本 ID 和总结层级查找排队中或运行中
  的相同任务；命中时复用原任务，并在页面明确提示无需重复提交。
- `analysis_tasks` 增加持久化取消标记和 `cancelled` 状态，页面为活动任务
  提供“取消任务”操作。
- 尚未开始的任务直接从执行器队列撤销；已经开始的任务在检索、每个分析
  分段以及模型调用前后检查取消标记，停止剩余工作。
- 取消状态写入 SQLite，刷新页面或服务重启后仍可正确显示。

### 当前边界

取消采用协作式检查。它不能安全地强制终止正在执行的单个 PyTorch
reranker 批次，也不能中断已经发出且未设置客户端超时的 GLM HTTP 调用；
当前步骤返回后会立即停止，不再开始后续步骤。若需要即时中断，后续应将
模型推理隔离到可终止的工作进程，并为传输层增加独立取消能力。

### 测试覆盖

- 相同活动请求只创建一个任务并返回原任务 ID。
- 排队任务可在执行前取消，不会调用分析服务。
- 运行任务在收到取消信号后以 `cancelled` 结束，不保存成功结果。
- 页面显示取消按钮、重复任务提示，并将任务 ID 传给取消接口。

### 修改的代码位置

- `src/paper_read_agent/application/local_reading.py`
- `src/paper_read_agent/persistence/database.py`
- `src/paper_read_agent/ui/local_facade.py`
- `src/paper_read_agent/ui/web.py`
- `src/paper_read_agent/ui/templates/analysis.html`
- `tests/test_analysis_tasks.py`
- `tests/test_analysis_ui.py`
