"""One local job dispatcher, deliberately without distributed scheduling."""
import fcntl
import json
import os
from pathlib import Path
import signal

from .agent_platform.service import AgentPlatform
from .workbench_access import AccessPolicy, private_write
from .workbench_context import ContextStore


def main():
    directory = Path(os.environ.get('WORKBENCH_STATE_DIR') or Path(__file__).resolve().parents[2] / '.workbench')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (directory / 'agent-worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('本机岗位 Agent 执行进程已经运行')
    platform = AgentPlatform(directory, ContextStore(directory), AccessPolicy(directory), background=False)
    platform.runs.recover()
    marker = directory / 'agent-worker.json'
    private_write(marker, json.dumps({'pid': os.getpid()}))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: platform.runs.stop.set())
    try:
        platform.runs.loop()
    finally:
        platform.close()
        marker.unlink(missing_ok=True)
        lock.close()


if __name__ == '__main__':
    main()
