#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$ROOT/.panda-dev"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/dev-stop.sh [--backend] [--admin] [--app] [--all]

默认行为：停止全部（backend + admin + app）。

说明：
- 优先使用 .panda-dev/*.pid
- 若 pidfile 不存在或已失效，会尝试通过端口查找并停止进程（backend:8000, admin:5173, app:3010）
EOF
}

STOP_BACKEND=0
STOP_ADMIN=0
STOP_APP=0

if [[ $# -eq 0 ]]; then
  STOP_BACKEND=1
  STOP_ADMIN=1
  STOP_APP=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      STOP_BACKEND=1
      STOP_ADMIN=1
      STOP_APP=1
      ;;
    --backend)
      STOP_BACKEND=1
      ;;
    --admin)
      STOP_ADMIN=1
      ;;
    --app)
      STOP_APP=1
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

stop_by_pidfile() {
  local name="$1"
  local pid_file="$PID_DIR/${name}.pid"

  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "[$name] empty pidfile removed"
    return 1
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "[$name] not running (stale pidfile removed)"
    return 1
  fi

  echo "[$name] stopping (pid=$pid)..."

  # 优先杀进程组（start 用 setsid 启动，pid 即 pgid）
  if kill -TERM "-$pid" 2>/dev/null; then
    :
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi

  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "[$name] stopped"
      return 0
    fi
    sleep 0.2
  done

  echo "[$name] force killing..."
  if kill -KILL "-$pid" 2>/dev/null; then
    :
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi

  rm -f "$pid_file"
  echo "[$name] killed"
}

pids_listen_on_port() {
  local port="$1"

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  # macOS: -nP 禁止 DNS/服务名解析，-sTCP:LISTEN 只看监听
  # 输出可能多行，这里只取 PID 列并去重
  lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $2}' | sort -u
}

stop_by_port() {
  local name="$1"
  local port="$2"

  local pids
  pids="$(pids_listen_on_port "$port" || true)"
  if [[ -z "$pids" ]]; then
    echo "[$name] pidfile missing and no process is listening on port $port"
    return 0
  fi

  local pid
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    echo "[$name] stopping by port $port (pid=$pid)..."

    # 这里不能假设有进程组（可能不是脚本启动的），优先 kill pid
    kill -TERM "$pid" 2>/dev/null || true

    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$name] stopped (pid=$pid)"
        pid=""
        break
      fi
      sleep 0.2
    done

    if [[ -n "${pid:-}" ]]; then
      echo "[$name] force killing (pid=$pid)..."
      kill -KILL "$pid" 2>/dev/null || true
      echo "[$name] killed (pid=$pid)"
    fi
  done <<< "$pids"
}

if [[ "$STOP_APP" -eq 1 ]]; then
  if ! stop_by_pidfile "app"; then
    stop_by_port "app" 3010
  fi
fi

if [[ "$STOP_ADMIN" -eq 1 ]]; then
  if ! stop_by_pidfile "admin"; then
    stop_by_port "admin" 5173
  fi
fi

if [[ "$STOP_BACKEND" -eq 1 ]]; then
  if ! stop_by_pidfile "backend"; then
    stop_by_port "backend" 8000
  fi
fi
