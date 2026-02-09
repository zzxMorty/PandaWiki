# PandaWiki 本地开发环境搭建指引

本文档旨在指导开发者快速搭建 PandaWiki 的本地开发环境，实现前端、后端服务的本地运行，并使用 Docker 管理依赖服务。

## 1. 环境要求

在开始之前，请确保你的开发环境满足以下要求：

-   **Git**: 代码版本管理
-   **Go**: 1.24.3 及以上版本
-   **Node.js**: 建议使用 `mise` 或 `nvm` 管理版本
-   **pnpm**: 10.x 及以上版本 (前端包管理工具)
-   **Docker**: 20.x 及以上版本 (用于运行依赖服务)
-   **Text Editor**: 推荐使用 Cursor 或 VS Code

**依赖安装偏好**:
-   **系统级包**: `brew` (macOS/Linux)
-   **Go/Node 版本**: `mise` (多语言版本管理)
-   **Python 包**: `uv` (如果有 Python 相关开发需求)
-   **前端包**: `pnpm` (本项目前端强制使用)

## 2. 代码获取

克隆 PandaWiki 仓库到本地：

```bash
git clone https://github.com/chaitin/PandaWiki.git
cd PandaWiki
```

## 3. 启动依赖服务 (Docker)

PandaWiki 后端依赖 PostgreSQL、NATS、Redis、MinIO 等服务。这些服务建议通过 Docker 容器运行。

### 3.1 首次启动或重建

如果你是首次启动，或者想重建依赖服务，可以使用项目提供的 `docker-compose.deps.yml`：

```bash
docker compose -f docker-compose.deps.yml up -d
```

这将会在后台启动所有依赖容器。

### 3.2 已有依赖容器时启动

如果你本机已有 `panda-wiki-postgres-dev` 等容器（例如之前通过其他方式部署），可以直接启动它们：

```bash
docker start panda-wiki-postgres-dev panda-wiki-nats-dev panda-wiki-redis-dev panda-wiki-minio-dev
```

> **注意**: 请确保这些容器的配置（尤其是密码）与后端 `backend/config.yml` 或环境变量一致。本项目已将默认密码统一为 `123456`。

## 4. 后端服务 (Go)

后端服务在宿主机本地运行，通过 `localhost` 连接 Docker 中的依赖服务。

### 4.1 配置密码

项目已将 `backend/config.yml` 中的所有密码配置（包括 `admin_password`、`pg.dsn`、`mq.nats.password`、`redis.password`、`auth.jwt.secret`、`s3.secret_key`）统一设置为 `123456`。

如果你在本机执行了之前的步骤，该文件应该已更新。如果未更新，请手动修改 `backend/config.yml` 中的密码为 `123456`。

### 4.2 编译与运行

进入 `backend` 目录，编译并运行数据库迁移和 API 服务。

```bash
cd backend

# （可选）生成 wire_gen.go 和 Swagger 文档。首次运行或代码结构变更后需要执行。
# 注意：这需要安装 swag 工具。
# make generate

# 编译并运行数据库迁移工具
go build -o /tmp/panda-wiki-migrate ./cmd/migrate
/tmp/panda-wiki-migrate

# 编译并启动 API 服务
go build -o /tmp/panda-wiki-api ./cmd/api
/tmp/panda-wiki-api & # 后台运行，API 监听 8000 端口

cd .. # 返回项目根目录
```

> **注意**：
> - 首次 `go build` 或 `go run` 会下载大量 Go 模块依赖，可能需要几分钟。
> - 如果 `go build` 失败，可能是网络问题导致依赖下载不完全。可尝试重新运行 `go mod download`。
> - 如果 API 启动时提示端口 `8000` 占用，请先查找并结束旧进程。

### 4.3 验证后端

访问以下地址验证后端是否正常运行：

-   **健康检查**: `http://localhost:8000/health` → 应返回 `{"status":"ok"}`
-   **根路径**：`http://localhost:8000/` → 应返回 `{"status":"ok", "service":"panda-wiki-api"}`
-   **登录接口** (POST): `http://localhost:8000/api/v1/user/login` (发送空 JSON 请求体应返回 `{"message":"invalid request ...","success":false}`，状态码 200)
-   **Swagger 文档** (仅当 `ENV=local` 环境变量设置时): `http://localhost:8000/swagger/index.html`

## 5. 前端服务 (React/Next.js)

前端服务也将在宿主机本地运行。

### 5.1 环境变量

仓库提供了示例文件 `web/app/.env.local.example` 与 `web/admin/.env.local.example`。

在启动前端前，请先复制生成本地环境变量文件：

```bash
cp web/app/.env.local.example web/app/.env.local
cp web/admin/.env.local.example web/admin/.env.local
```

### 5.2 安装依赖与运行

进入 `web` 目录，安装 pnpm 依赖并启动前端开发服务器：

```bash
cd web
pnpm install
pnpm dev # 启动 admin (Vite, 默认 5173 端口) 和 app (Next.js, 默认 3010 端口)
```

> **注意**：首次 `pnpm install` 会下载大量 npm 包，可能需要几分钟。

### 5.3 访问前端页面

前端服务启动成功后，可以通过以下地址访问：

-   **管理端 (Admin Console)**: `http://localhost:5173/`
    -   登录页面：`http://localhost:5173/login`
-   **用户端 (Wiki Website)**: `http://localhost:3010/`

## 6. RAG 服务 (AI 大模型依赖)

AI 能力（AI 创作、问答、搜索）依赖于 RAG 服务。

-   **默认配置**: 后端 `config.yml` 中 `rag.ct_rag.base_url` 指向 `http://localhost:5050`。
-   **部署**: RAG 服务通常需要单独部署。如果未配置或未启动，AI 相关功能将不可用。
-   **密钥**: `rag.ct_rag.api_key` 为 `sk-1234567890`，在实际部署时应替换为有效密钥。

---

**一键启动脚本**：
项目根目录下提供了 `./scripts/local-dev.sh` 脚本，可以帮助你一键检查依赖容器、执行后端迁移、编译并启动后端 API。
