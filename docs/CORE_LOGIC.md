## 你接下来要产出的 Markdown 文件（建议路径与文件名）

- `docs/CORE_LOGIC.md`

---

# PandaWiki Core Logic（可记忆版）

## 0. 一句话总览（记忆锚点）

- **PandaWiki = 权限 + 内容 + 发布版本 + 编排**
- **RAG 服务 = embedding/索引 + 检索 +（可能的）解析/切分 +（可能的）重排**
- **引用溯源 = RAG 返回 doc/chunk → PandaWiki 用 `doc_id ↔ node_release` 映射回页面 URL**

---

## 1. 核心对象与不变量（建议先背这 5 个）

- **KB（KnowledgeBase）**
  - PandaWiki 侧的“站点/知识库”元数据与访问配置（host/port/cert/trust proxy…）
  - **关键字段：`DatasetID`**（RAG 服务侧的数据集 ID）
- **Node**
  - PandaWiki 的“目录/文档”树节点（草稿态内容 + 权限）
- **NodeRelease**
  - Node 的“发布快照”（发布后的版本，才会进向量库/检索）
  - **关键字段：`DocID`**（RAG 服务侧 document id）
- **DocID（RAG Document ID）**
  - 用于把 RAG chunk 命中“精确溯源”到某个发布文档
- **Permissions（可见/可访问/可问答）**
  - PandaWiki 决定用户能不能看到、打开、以及能不能被问答命中（通常会影响检索过滤或结果后置过滤）

---

## 2. 调用链路：创建知识库（KB/站点）

### 2.1 从 HTTP 入口开始

- **入口：** `backend/handler/v1/knowledge_base.go` 的 `CreateKnowledgeBase`
  - 会把 `MaxKb` 从 license 限制注入到请求：`req.MaxKB = domain.GetBaseEditionLimitation(ctx).MaxKb`
  - 错误映射（已增强过）：
    - `ErrMaxKnowledgeBaseLimitReached` → “知识库数量已达上限”
    - `ErrRAGServiceUnavailable` → “RAG 服务不可用…”

### 2.2 Usecase 负责编排（PandaWiki 的边界）

- **编排：** `backend/usecase/knowledge_base.go` 的 `CreateKnowledgeBase`
  - 先调用 RAG：`u.rag.CreateKnowledgeBase(ctx)` 得到 **`datasetID`**
  - 再写 DB：`u.repo.CreateKnowledgeBase(ctx, req.MaxKB, kb)`
  - DB 失败会回滚：`_ = u.rag.DeleteKnowledgeBase(ctx, datasetID)`（避免 dataset 泄漏）

### 2.3 Repository 负责落库与 Caddy 同步

- `backend/repo/pg/knowledge_base.go`
  - 初始化时会 `SyncKBAccessSettingsToCaddy`，把已有 KB 的 host/port 路由规则同步到 Caddy
  - 关键点：**Caddy 给 `/share/v1/chat/message` 注入 `X-KB-ID` 头**（后面问答链路会用到）

### 2.4 记忆总结

- **PandaWiki 创建 KB = DB 一条 KB + Caddy 路由 + RAG 一个 dataset**
- **RAG 不可用会直接导致创建失败**（因为必须先拿到 `DatasetID` 才能保证后续可索引）

---

## 3. 调用链路：文档/目录（Node）与“发布（Release）→ 入库（Index）”

### 3.1 Node 的日常编辑：只影响 PandaWiki，不进向量库

- Node 的创建/更新发生在 DB（草稿态）
- Node 典型字段：
  - `Status: Draft`
  - `RagInfo: Pending/Running/Succeeded/...`（用于管理后台展示“学习状态”）
  - `Permissions: Answerable/Visitable/Visible`

### 3.2 发布（Release）才会触发向量库更新

心智模型：

- **草稿（Node）**：可随便改，不影响线上检索
- **发布（NodeRelease）**：生成“对外可引用”的稳定版本，并进入 RAG dataset 形成可检索索引

实现层面通常是：

- 发布时创建/更新一条 `node_releases` 记录
- 把发布内容通过 MQ 异步投递给 consumer
- consumer 调用 RAG upsert / parse / reindex

### 3.3 异步更新（MQ）是为了把“写内容”和“向量化”解耦

- 前端“重新学习”入口：`/api/v1/node/restudy`
- 后端入口：`backend/handler/v1/node.go` 的 `NodeRestudy`
- MQ：`backend/repo/mq/rag.go`（producer）+ `backend/handler/mq/rag*.go`（consumer）

---

## 4. 调用链路：问答（Chat/Q&A）与检索（Search）

### 4.1 Share Chat：对外站点的问答入口

- `backend/handler/share/chat.go` 的 `ChatMessage`
  - `req.KBID = Header("X-KB-ID")`（来自 Caddy 注入）
  - SSE 输出（`text/event-stream`），用于流式返回

关键点：对外站点并不靠前端传 KBID，而是靠网关（Caddy）在不同 host/port 路由时注入。

### 4.2 LLMUsecase：把“对话历史 + 检索结果”拼成最终 Prompt

- 检索得到 `RankedNodeChunks`
- 用 `backend/domain/llm.go` 的 `FormatNodeChunks(nodeChunks, baseURL)` 格式化为 `<documents>` 块
- 其中会把内容里的 `/static-file/...` 补成完整 URL，确保图片可访问

### 4.3 Prompt 约束：引用格式与引用列表

- `backend/domain/llm.go` 的 `SystemDefaultPrompt`
  - 要求回答中插入内联引用：`[[序号](URL)]`
  - 并在末尾输出“引用列表”

---

## 5. 引用溯源（Source tracing）到底是怎么“对齐到页面”的？

### 5.1 两条溯源线

- **内容溯源线：**
  - `node_release.content`（发布内容） → 发送给 RAG → RAG 生成 chunks → 检索返回 chunks
- **链接溯源线：**
  - `node_release.doc_id`（RAG doc） ↔ PandaWiki 的 node/release
  - 通过 `GetURL(baseURL)` 还原到站点页面 URL

### 5.2 结论

- **RAG 只认识 doc/chunk**
- **PandaWiki 才认识页面/树结构/权限/发布版本**

---

## 6. PandaWiki vs RAG：明确边界（避免职责混乱）

### 6.1 PandaWiki 负责（权威）

- 权限模型（谁能看、谁能问、导航是否可见）
- 内容与版本（草稿、发布、回滚）
- 站点访问与路由（host/port/cert → Caddy）
- 工作流编排（发布触发学习、失败重试、状态展示）
- 对外 API/SSE（聊天、搜索、文档浏览）

### 6.2 RAG 服务负责（权威）

- dataset/document 的生命周期（create/delete/update）
- 文档解析（取决于具体 RAG 实现）
- chunking、embedding、向量索引、相似度检索
- （可选）rerank、query rewrite

---

## 7. `pageIndex` 与 RAG 的关系

### 7.1 `pageIndex` 实际做的事

`pageIndex` 本质上是一个 **“内容结构化 + 元信息生成”** 的预处理器，例如：

- 从 Markdown/文档提取标题层级（TOC）并构建树
- 为节点生成 `node_id`
- 可选：
  - thinning（按 token 阈值剪枝/聚合）
  - 为每个节点生成 summary
  - 生成 doc_description

### 7.2 把 `pageIndex` 用到 PandaWiki 知识库构建：方案 A vs 方案 B

下面这段是你关心的关键区别。

#### 方案 A（低侵入）：作为“导入前预处理”（推荐起步）

- **做法**
  - 你有一份外部文档（例如 PDF/MD/HTML）
  - 先跑 `pageIndex`：把它拆成结构化的章节树（可带 summary/description）
  - 再把 `pageIndex` 的输出 *转换成 PandaWiki 的 Node 树*（创建 Node、填充内容、设置父子关系）
  - 最后走 PandaWiki 现有“发布 → MQ → RAG 学习”的链路

- **对现有链路的影响**
  - 基本不改 RAG 写入逻辑，只是“你往 PandaWiki 里塞什么内容”变得更结构化

- **收益**
  - 快速见效：能显著改善“文档目录树质量”和“章节粒度”，间接提升检索质量
  - 失败面小：RAG 服务、consumer、索引接口都不需要你重做

- **局限**
  - `pageIndex` 产生的结构主要用于“页面组织/展示”
  - RAG 的 chunk 边界仍由 RAG（或当前写入方式）决定，`pageIndex` 结构对 chunk 的影响是“间接的”

#### 方案 B（中侵入）：增强发布时的“chunk 组织策略”（让结构直接影响索引）

- **做法**
  - 仍然在 PandaWiki 侧管理 Node/Release
  - 但在“发布→写入 RAG”这一步，把 `pageIndex` 的结构作为 **索引输入的第一公民**：
    - 以“章节/子章节”为 chunk 边界
    - 把 `summary`、`heading path`、`line range` 等作为 metadata 写入 RAG

- **对现有链路的影响**
  - 你需要改“写入 RAG 的数据形态”或“写入前的预处理”
  - MQ consumer / RAG upsert 的 payload 很可能要扩展（例如增加 metadata 字段、chunk 列表等）

- **收益**
  - 检索质量提升更直接：
    - chunk 语义边界更稳定（按标题层级而不是纯长度切分）
    - citation 更精确（可以指向章节级、甚至标题路径）

- **局限/风险**
  - 需要你们定义清楚：
    - chunk 的“权威切分规则”到底在 PandaWiki 还是 RAG
    - `DocID` 与 NodeRelease 的映射、以及章节级 chunk 的可追踪性
  - 侵入性更高：要更谨慎地兼容历史数据与重建索引

#### 一句话对比（便于记忆）

- **方案 A：先把内容“变成更像 Wiki 的树”，再按原流程让 RAG 学**（结构主要服务于 PandaWiki）
- **方案 B：让结构“直接进入索引层”，决定 chunk 怎么切、metadata 怎么写**（结构主要服务于 RAG 质量）

---

## 8. 常见故障的定位路线

- **创建 KB 失败**
  - 先看是否 `ErrRAGServiceUnavailable`（RAG base_url/端口是否可达）
  - 再看是否端口/域名冲突（Caddy sync）
  - 再看 license 的 `MaxKb`
- **问答没引用/引用点不开**
  - 检查 Prompt 是否被覆盖（System prompt）
  - 检查 `FormatNodeChunks` 的 `baseURL` 是否正确
  - 检查静态资源路径是否以 `/static-file/` 开头
- **文档学习失败**
  - 看 Node 的 `rag_info.status/message`
  - 通过“重新学习（restudy）”重新入队
  - 看 MQ consumer 日志 & RAG 服务日志

---

## 9. 代码索引（从这些文件开始跳）

- **KB：**
  - `backend/handler/v1/knowledge_base.go`
  - `backend/usecase/knowledge_base.go`
  - `backend/repo/pg/knowledge_base.go`
- **Chat：**
  - `backend/handler/share/chat.go`
  - `backend/usecase/chat.go`
  - `backend/usecase/llm.go`
  - `backend/domain/llm.go`
- **Node / Release / 学习：**
  - `backend/repo/pg/node.go`
  - `backend/handler/v1/node.go`（`NodeRestudy`）
  - `backend/repo/mq/rag.go`、`backend/handler/mq/rag*.go`
- **RAG SDK：**
  - `sdk/rag/dataset.go`
  - `sdk/rag/document.go`
