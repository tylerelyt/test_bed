#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--background" ]; then
  mkdir -p logs/console
  if [ -f logs/console/launcher.pid ] && kill -0 "$(cat logs/console/launcher.pid)" 2>/dev/null; then
    echo '工作台后台进程已运行：http://127.0.0.1:3100'
    exit 0
  fi
  python3 - <<'BACKGROUND'
import subprocess
from pathlib import Path
with open('logs/console/launcher.log', 'ab') as log:
    process = subprocess.Popen(['/bin/bash', str(Path('start_console.sh').resolve())],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
Path('logs/console/launcher.pid').write_text(str(process.pid) + '\n')
BACKGROUND
  echo '工作台正在后台启动：http://127.0.0.1:3100'
  echo "日志：$PWD/logs/console/launcher.log"
  exit 0
fi
if [ -n "${WORKBENCH_PYTHON:-}" ]; then
  workbench_python="$WORKBENCH_PYTHON"
elif [ -x .venv-full/bin/python ]; then
  workbench_python="$PWD/.venv-full/bin/python"
else
  workbench_python=python3
fi
"$workbench_python" -c 'import fastapi, fastmcp, sqlalchemy, uvicorn, httpx, pypdf' || {
  echo '请用 Python 3.11 安装 requirements-context.txt，并设置 WORKBENCH_PYTHON；原业务工具另装 requirements-platform.txt。' >&2
  exit 1
}
[ -d console/node_modules ] || { echo '请先运行 npm --prefix console ci（Node.js 22+）。' >&2; exit 1; }
workbench_state_dir="$($workbench_python -c 'import os;from pathlib import Path;print(Path(os.environ.get("WORKBENCH_STATE_DIR", ".workbench")).resolve())')"
export WORKBENCH_STATE_DIR="$workbench_state_dir"
owned_pids=()
cleanup(){
  trap - EXIT INT TERM
  for pid in "${owned_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${owned_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
listening(){ "$workbench_python" - "$1" <<'PY'
import socket,sys
try:
    with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=.3):pass
except OSError:sys.exit(1)
PY
}
if [ -f "$workbench_state_dir/database-url.txt" ] && [ -d "$workbench_state_dir/postgres" ]; then
  "$workbench_python" tools/start_platform_db.py
fi
"$workbench_python" - <<'PY'
import os
from pathlib import Path
from src.search_engine.workbench_access import private_write
directory=Path(os.environ['WORKBENCH_STATE_DIR'])
directory.mkdir(parents=True,exist_ok=True,mode=0o700)
if not os.environ.get('WORKBENCH_DATABASE_URL') and not (directory/'database-url.txt').exists():
    private_write(directory/'database-url.txt','sqlite:///'+str(directory/'agent-platform-v1.sqlite3'))
    print('未配置 PostgreSQL，使用统一 SQLite 开发库；部署配置见 README。')
PY
if ! listening 3001; then
  "$workbench_python" -m src.search_engine.mcp.dynamic_mcp_server &
  mcp_pid=$!;owned_pids+=("$mcp_pid")
  for attempt in $(seq 1 60); do listening 3001 && break; kill -0 "$mcp_pid"; sleep .5; done
  listening 3001 || { echo 'MCP 服务启动超时。' >&2;exit 1; }
fi
if ! listening 5001 && "$workbench_python" -c 'import importlib.util;raise SystemExit(0 if importlib.util.find_spec("flask") else 1)'; then
  "$workbench_python" -m flask --app image_generation_service run --host 127.0.0.1 --port 5001 &
  owned_pids+=("$!")
fi
"$workbench_python" -m src.search_engine.platform_worker &
worker_pid=$!;owned_pids+=("$worker_pid")
worker_ready=0
for attempt in $(seq 1 120); do
  kill -0 "$worker_pid" || { echo '执行进程启动失败；请检查旧进程及日志，未启动控制台。' >&2;exit 1; }
  if "$workbench_python" - "$workbench_state_dir/agent-worker.json" "$worker_pid" <<'PY'
import json,sys
from pathlib import Path
try:ready=json.loads(Path(sys.argv[1]).read_text())['pid']==int(sys.argv[2])
except (OSError,ValueError,KeyError):ready=False
raise SystemExit(0 if ready else 1)
PY
  then worker_ready=1;break;fi
  sleep .25
done
[ "$worker_ready" = 1 ] || { echo '执行进程未写入本次 PID，启动超时。' >&2;exit 1; }
kill -0 "$worker_pid"
"$workbench_python" -m src.search_engine.platform_api &
api_pid=$!;owned_pids+=("$api_pid")
for attempt in $(seq 1 40); do listening 8766 && break;kill -0 "$api_pid";sleep .25;done
listening 8766 || { echo 'Python 执行服务启动超时。' >&2;exit 1; }
(
  cd console
  node scripts/copy-vis.mjs
  if [ -f .next/BUILD_ID ]; then
    exec node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port 3100
  else
    exec node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 --port 3100
  fi
) &
web_pid=$!;owned_pids+=("$web_pid")
wait "$web_pid"
