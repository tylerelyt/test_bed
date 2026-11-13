"""Cheap capability checks; training and inference can use separate Python environments."""
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def python_for(kind):
    return os.environ.get('WORKBENCH_TRAIN_PYTHON' if kind in ('train', 'merge') else 'WORKBENCH_INFER_PYTHON') or sys.executable


@lru_cache(maxsize=8)
def probe(python, package, bucket):
    try:
        result = subprocess.run([python, '-c', 'import importlib.metadata,sys;print(importlib.metadata.version(sys.argv[1]))', package],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capabilities(directory, storage):
    bucket = int(time.time() // 30)
    marker = Path(directory)/'agent-worker.json'
    worker = False
    try:
        pid = json.loads(marker.read_text())['pid']
        os.kill(pid, 0)
        worker = True
    except (OSError, ValueError, KeyError):
        pass
    return {'storage': storage, 'context_storage': storage,
            'worker_running': worker, 'training_python': python_for('train'),
            'inference_python': python_for('deploy'),
            'llamafactory': probe(python_for('train'), 'llamafactory', bucket),
            'vllm': probe(python_for('deploy'), 'vllm', bucket),
            'pypdf': probe(sys.executable, 'pypdf', bucket),
            'document_formats': ['txt', 'md', 'csv', 'json', 'docx', 'pdf'],
            'job_slots': 2, 'agent_slots': 1,
            'notice': '安装状态不代表硬件满足模型要求；训练、合并和推理在独立原生进程中执行。'}
