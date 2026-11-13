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
"$workbench_python" -c 'import gradio, torch, fastmcp, gremlin_python, sqlalchemy, uvicorn' || {
  echo '请用 Python 3.11 安装 requirements-platform.txt，或设置 WORKBENCH_PYTHON 指向原完整环境。' >&2
  exit 1
}
[ -d console/node_modules ] || { echo '请先运行 npm --prefix console ci（Node.js 22+）。' >&2; exit 1; }
workbench_state_dir="$($workbench_python -c 'import os;from pathlib import Path;print(Path(os.environ.get("WORKBENCH_STATE_DIR", ".workbench")).resolve())')"
export WORKBENCH_STATE_DIR="$workbench_state_dir"
"$workbench_python" - <<'PY'
import os,socket,urllib.parse
url=os.environ.get('JANUSGRAPH_URL') or os.environ.get('GREMLIN_SERVER_URL') or 'ws://localhost:8182/gremlin'
parsed=urllib.parse.urlparse(url)
try:
    with socket.create_connection((parsed.hostname,parsed.port or 8182),timeout=3):pass
except OSError:
    print('原图谱服务未就绪：岗位Agent仍可使用，图谱操作需另行启动原服务。')
PY
owned_pids=()
cleanup(){ for pid in "${owned_pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
listening(){ "$workbench_python" - "$1" <<'PY'
import socket,sys
try:
    with socket.create_connection(('127.0.0.1',int(sys.argv[1])),timeout=.3):pass
except OSError:sys.exit(1)
PY
}
if ! listening 3001; then
  "$workbench_python" -m src.search_engine.mcp.dynamic_mcp_server &
  mcp_pid=$!;owned_pids+=("$mcp_pid")
  for attempt in $(seq 1 60); do listening 3001 && break; kill -0 "$mcp_pid"; sleep .5; done
  listening 3001 || { echo 'MCP 服务启动超时。' >&2;exit 1; }
fi
if ! listening 5001; then
  "$workbench_python" -m flask --app image_generation_service run --host 127.0.0.1 --port 5001 &
  owned_pids+=("$!")
fi
if [ -f "$workbench_state_dir/database-url.txt" ] && [ -d "$workbench_state_dir/postgres" ]; then
  "$workbench_python" tools/start_platform_db.py
fi
"$workbench_python" -m src.search_engine.platform_worker &
worker_pid=$!;owned_pids+=("$worker_pid")
for attempt in $(seq 1 120); do
  [ -f "$workbench_state_dir/agent-worker.json" ] && break
  kill -0 "$worker_pid";sleep .25
done
kill -0 "$worker_pid"
"$workbench_python" -m src.search_engine.platform_api &
api_pid=$!;owned_pids+=("$api_pid")
for attempt in $(seq 1 40); do listening 8766 && break;kill -0 "$api_pid";sleep .25;done
listening 8766 || { echo 'Python 执行服务启动超时。' >&2;exit 1; }
if [ -f console/.next/BUILD_ID ]; then
  npm --prefix console start &
else
  npm --prefix console run dev &
fi
web_pid=$!;owned_pids+=("$web_pid")
wait "$web_pid"
