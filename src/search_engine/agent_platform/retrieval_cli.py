"""Trusted Pi tool transport. Input names a published index, never a caller-provided path."""
import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.search_engine.agent_platform.knowledge import search_artifact


def main():
    request=json.loads(sys.stdin.read(10000))
    manifest=json.loads(Path(os.environ['WORKBENCH_CONTEXT_FILE']).read_text())
    allowed={item['id']:item for item in manifest.get('retrievals',[])}
    item=allowed.get(request.get('index_id'))
    if item is None: raise ValueError('索引不属于本次岗位发布包')
    with contextlib.redirect_stdout(sys.stderr):
        result=search_artifact(item,request.get('query'),request.get('limit',5))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False));sys.exit(2)
