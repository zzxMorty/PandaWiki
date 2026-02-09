#!/usr/bin/env bash
# 本地开发一键启动：依赖已用 Docker 启动后，执行本脚本完成迁移、后端 API、前端 dev
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/4] 检查依赖容器..."
for c in panda-wiki-postgres-dev panda-wiki-nats-dev panda-wiki-redis-dev panda-wiki-minio-dev; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
    echo "请先启动依赖: docker start $c (或 docker compose -f docker-compose.deps.yml up -d)"
    exit 1
  fi
done
echo "  依赖容器已运行"

echo "[2/4] 后端 config..."
if [[ ! -f backend/config.yml ]]; then
  cp backend/config.example.yml backend/config.yml
  echo "  已从 config.example.yml 复制"
fi

echo "[3/4] 后端迁移 + API..."
cd backend
if ! go build -o /tmp/panda-wiki-migrate ./cmd/migrate 2>/dev/null; then
  echo "  首次编译迁移工具（可能需几分钟）..."
  go build -o /tmp/panda-wiki-migrate ./cmd/migrate
fi
/tmp/panda-wiki-migrate
echo "  迁移完成"
if ! go build -o /tmp/panda-wiki-api ./cmd/api 2>/dev/null; then
  echo "  首次编译 API（可能需几分钟）..."
  go build -o /tmp/panda-wiki-api ./cmd/api
fi
echo "  启动 API (端口 8000)..."
/tmp/panda-wiki-api &
API_PID=$!
cd "$ROOT"

echo "[4/4] 等待 API 就绪并检查..."
for i in $(seq 1 15); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null | grep -qE '^[0-9]+$'; then
    break
  fi
  sleep 1
done
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null; then
  echo "  后端 API 可访问: http://localhost:8000"
else
  echo "  警告: 无法确认 API，请手动检查 http://localhost:8000"
fi

echo ""
echo "前端: 在另一终端执行: cd $ROOT/web && pnpm install && pnpm dev"
echo "停止 API: kill $API_PID"
echo "完成"
