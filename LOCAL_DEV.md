# 本地开发指南

前端、后端在宿主机本地运行，其他组件（PostgreSQL、NATS、Redis、MinIO）用 Docker 运行。

## 1. 启动依赖服务

已有容器时直接启动：

```bash
docker start panda-wiki-postgres-dev panda-wiki-nats-dev panda-wiki-redis-dev panda-wiki-minio-dev
```

或使用项目提供的 compose（会创建新容器）：

```bash
docker compose -f docker-compose.deps.yml up -d
```

## 2. 后端

```bash
cd backend
cp config.example.yml config.yml   # 已配置 localhost，可按需修改
# 可选：make generate   # 需要安装 swag，用于生成 wire_gen.go、swag 文档
go run ./cmd/migrate   # 执行数据库迁移（首次 go run 会下载依赖，较慢）
go run ./cmd/api       # 启动 API，监听 8000
```

或使用脚本（检查依赖、迁移、编译并启动 API）：

```bash
./scripts/local-dev.sh
```

## 3. 前端

已为本地开发准备好 `web/app/.env.local` 和 `web/admin/.env.local`（TARGET=http://localhost:8000）。

```bash
cd web
pnpm install
pnpm dev
```

## 4. 检查是否成功

- **后端**：`curl http://localhost:8000/` 或 `curl http://localhost:8000/health` 应返回 `{"status":"ok", ...}`（200）。API 文档（仅当 `ENV=local` 时）：http://localhost:8000/swagger/index.html
- **管理端 / 用户端**：按 `pnpm dev` 输出的地址访问

## 5. RAG 服务

AI 能力依赖 RAG 服务（ct_rag），需单独部署或配置 `RAG_CT_RAG_BASE_URL` 指向已有服务。未配置时，AI 相关功能不可用。

## 依赖安装偏好

- 系统包：brew
- 语言版本：mise
- Python：uv
- 前端：pnpm
