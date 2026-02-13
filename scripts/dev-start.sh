#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/.panda-dev"
LOG_DIR="$PID_DIR/logs"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/dev-start.sh [--backend] [--admin] [--app] [--all] [--no-check-deps]

默认行为：启动全部（backend + admin + app）。

说明：
- 本脚本不会自动执行 docker compose up，仅检查依赖容器是否已运行（可用 --no-check-deps 跳过）。
- 前端会在缺少 .env.local 时，从同目录的 .env.local.example 复制生成。
- 进程 PID / 日志写入：.panda-dev/
EOF
}

CHECK_DEPS=1
START_BACKEND=0
START_ADMIN=0
START_APP=0

if [[ $# -eq 0 ]]; then
  START_BACKEND=1
  START_ADMIN=1
  START_APP=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      START_BACKEND=1
      START_ADMIN=1
      START_APP=1
      ;;
    --backend)
      START_BACKEND=1
      ;;
    --admin)
      START_ADMIN=1
      ;;
    --app)
      START_APP=1
      ;;
    --no-check-deps)
      CHECK_DEPS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_seconds="$3"

  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if command -v nc >/dev/null 2>&1; then
      if nc -z "$host" "$port" >/dev/null 2>&1; then
        return 0
      fi
    else
      if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
        return 0
      fi
    fi

    if (( $(date +%s) - start_ts >= timeout_seconds )); then
      return 1
    fi
    sleep 0.2
  done
}

start_cmd() {
  local name="$1"; shift
  local pid_file="$PID_DIR/${name}.pid"
  local log_file="$LOG_DIR/${name}.log"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[$name] already running (pid=$old_pid)"
      return 0
    fi
    rm -f "$pid_file"
  fi

  echo "[$name] starting..."

  : >"$log_file"

  # macOS 默认可能没有 setsid；这里使用 nohup 后台运行并记录 PID。
  # 对于 bash -lc 场景，请在命令中用 exec 替换 shell 进程，以便 pid 指向真实服务进程。
  if command -v nohup >/dev/null 2>&1; then
    nohup "$@" >>"$log_file" 2>&1 &
  else
    "$@" >>"$log_file" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "[$name] started (pid=$pid), log=$log_file"
}

if [[ "$CHECK_DEPS" -eq 1 && "$START_BACKEND" -eq 1 ]]; then
  echo "[deps] checking docker containers..."
  missing=0
  for c in panda-wiki-postgres-dev panda-wiki-nats-dev panda-wiki-redis-dev panda-wiki-minio-dev; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${c}$"; then
      echo "[deps] missing: $c" >&2
      missing=1
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "[deps] 请先启动依赖：docker start panda-wiki-postgres-dev panda-wiki-nats-dev panda-wiki-redis-dev panda-wiki-minio-dev" >&2
    echo "[deps] 或：docker compose -f docker-compose.deps.yml up -d" >&2
    exit 1
  fi
  echo "[deps] ok"
fi

if [[ "$START_BACKEND" -eq 1 ]]; then
  if [[ ! -f "$ROOT/backend/config.yml" ]]; then
    cp "$ROOT/backend/config.example.yml" "$ROOT/backend/config.yml"
    echo "[backend] copied backend/config.example.yml -> backend/config.yml"
  fi

  echo "[backend] waiting for postgres on localhost:5432 ..."
  if ! wait_for_port 127.0.0.1 5432 15; then
    echo "[backend] postgres not ready on localhost:5432" >&2
    echo "[backend] 请确认依赖已启动：docker compose -f docker-compose.deps.yml up -d" >&2
    exit 1
  fi

  echo "[backend] build migrate + api..."
  (
    cd "$ROOT/backend"
    if ! go build -o /tmp/panda-wiki-migrate ./cmd/migrate 2>/dev/null; then
      go build -o /tmp/panda-wiki-migrate ./cmd/migrate
    fi
    if ! go build -o /tmp/panda-wiki-api ./cmd/api 2>/dev/null; then
      go build -o /tmp/panda-wiki-api ./cmd/api
    fi
    if ! go build -o /tmp/panda-wiki-consumer ./cmd/consumer 2>/dev/null; then
      go build -o /tmp/panda-wiki-consumer ./cmd/consumer
    fi
  )

  echo "[backend] migrate..."
  (
    cd "$ROOT/backend"
    /tmp/panda-wiki-migrate
  )

  start_cmd "backend" bash -lc "cd '$ROOT/backend' && exec /tmp/panda-wiki-api"
  start_cmd "consumer" bash -lc "cd '$ROOT/backend' && exec /tmp/panda-wiki-consumer"
fi

ensure_env_local() {
  local dir="$1"
  if [[ ! -f "$dir/.env.local" ]]; then
    if [[ -f "$dir/.env.local.example" ]]; then
      cp "$dir/.env.local.example" "$dir/.env.local"
      echo "[env] copied $dir/.env.local.example -> $dir/.env.local"
    else
      echo "[env] missing $dir/.env.local and $dir/.env.local.example" >&2
      return 1
    fi
  fi
}

if [[ "$START_ADMIN" -eq 1 || "$START_APP" -eq 1 ]]; then
  if [[ ! -f "$ROOT/web/pnpm-lock.yaml" ]]; then
    echo "[web] pnpm workspace not found? expected web/pnpm-lock.yaml" >&2
    exit 1
  fi

  # 提前检查是否安装过依赖（不强制自动安装，避免脚本隐式做耗时操作）
  if [[ ! -d "$ROOT/web/node_modules" ]]; then
    echo "[web] 未检测到 web/node_modules。请先执行：cd web && pnpm install" >&2
  fi
fi

if [[ "$START_ADMIN" -eq 1 ]]; then
  ensure_env_local "$ROOT/web/admin"
  start_cmd "admin" bash -lc "cd '$ROOT/web/admin' && exec pnpm dev"
fi

if [[ "$START_APP" -eq 1 ]]; then
  ensure_env_local "$ROOT/web/app"
  start_cmd "app" bash -lc "cd '$ROOT/web/app' && exec pnpm dev"
fi

echo ""
echo "PIDs saved in: $PID_DIR"
echo "Logs saved in: $LOG_DIR"
