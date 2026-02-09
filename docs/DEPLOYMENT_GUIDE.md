# PandaWiki 部署指引

本文档提供 PandaWiki 的部署指南，主要基于 Docker Compose 进行多服务部署。

## 1. 部署架构

本项目推荐使用 Docker Compose 部署所有 PandaWiki 组件和依赖服务，包括：

-   **PandaWiki 后端 API 服务**
-   **PandaWiki 后端 Consumer 服务** (处理异步任务)
-   **PostgreSQL** (数据库)
-   **NATS** (消息队列)
-   **Redis** (缓存)
-   **MinIO** (对象存储)
-   **RAG 服务** (AI 大模型服务，可选或外部集成)
-   **Caddy** (反向代理，用于管理前端流量和 SSL)

## 2. 部署方式

### 2.1 Docker Compose 部署

PandaWiki 提供了完整的 Docker 化部署方案。通常，部署脚本会包括构建后端 Docker 镜像和启动所有服务的 Compose 文件。

1.  **构建后端镜像**:
    -   进入 `backend` 目录。
    -   根据 `Dockerfile.api` 和 `Dockerfile.consumer` 构建 API 和 Consumer 服务的 Docker 镜像。
    -   示例 (在 `backend` 目录下执行):
        ```bash
        # 构建 API 镜像
        docker build -f Dockerfile.api -t panda-wiki-api:latest .
        # 构建 Consumer 镜像
        docker build -f Dockerfile.consumer -t panda-wiki-consumer:latest .
        ```

2.  **准备 Docker Compose 文件**:
    -   项目中可能存在专门的 `deploy/docker-compose.yml` 文件。如果不存在，你需要根据 `docker-compose.deps.yml` 和后端 Dockerfile 自行构建。
    -   Compose 文件应包含所有依赖服务（PostgreSQL、NATS、Redis、MinIO）以及 PandaWiki 的 API 和 Consumer 服务。
    -   **注意配置**：
        -   **数据库连接**: 确保 API 和 Consumer 服务能正确连接到 PostgreSQL 容器。
        -   **环境变量**: 配置好各种密码、密钥和外部服务的连接信息（如 RAG 服务的 URL 和 API Key）。
        -   **持久化存储**: 为 PostgreSQL、Redis 和 MinIO 配置数据卷，确保数据持久化。
        -   **网络**: 配置 Docker 内部网络，使各服务能够互相发现和通信。

3.  **启动服务**:
    -   在包含 `docker-compose.yml` 文件的目录下执行：
        ```bash
        docker compose up -d
        ```
    -   这将启动所有在 Compose 文件中定义的服务。

### 2.2 生产环境配置考量

在生产环境中部署时，除了上述 Docker Compose 配置，还需要考虑：

-   **安全性**:
    -   所有服务密码、密钥必须使用强密码，并通过环境变量安全注入。
    -   MinIO 访问密钥和 secret_key 应该妥善管理。
    -   启用 HTTPS，为前端和 API 接口配置 SSL 证书（可通过 Caddy 自动管理）。
-   **高可用性**:
    -   考虑数据库集群 (如 PostgreSQL Replication)、Redis Sentinel/Cluster 等高可用方案。
    -   多实例部署 PandaWiki API 和 Consumer 服务，并通过负载均衡器进行分发。
-   **可观测性**:
    -   集成 Sentry 进行错误监控。
    -   集成 OpenTelemetry 收集追踪和指标，配合 Jaeger/Prometheus 进行监控。
    -   配置日志系统（如 ELK Stack）集中收集和分析日志。
-   **备份与恢复**:
    -   制定详细的数据库和 MinIO 数据备份策略。
    -   测试数据恢复流程。
-   **性能优化**:
    -   根据实际负载调整容器资源（CPU, 内存）限制。
    -   优化数据库查询和索引。
    -   前端部署 CDN 加速。
-   **外部服务集成**:
    -   确保 RAG 服务稳定可靠，并配置正确的连接信息。
    -   如果使用 Caddy 作为反向代理，配置好域名、SSL 证书和路由规则。

## 3. RAG 服务部署

RAG 服务是 AI 能力的核心，通常需要独立部署，可能涉及：

-   **部署 LLM**: 部署或接入大规模语言模型。
-   **向量数据库**: 部署向量数据库 (如 Milvus, Weaviate) 存储文档向量。
-   **RAG 逻辑服务**: 实际执行 RAG 流程的服务。

PandaWiki 的后端通过 HTTP 接口调用 RAG 服务，因此确保 RAG 服务的 `base_url` 和 `api_key` 配置正确，并且服务可达。
