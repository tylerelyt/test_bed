import re
from pathlib import Path
from urllib.parse import urlparse
from ..workbench_access import require_business, private_write
from .common import get, owned, public, text
from .database import ModelService, Job


class ModelServices:
    def __init__(self, session, directory):
        self.session, self.directory = session, Path(directory)

    def model(self, data, deployment_job=None):
        require_business()
        base = text(data.get('base_url'), '模型服务地址', 2000).rstrip('/')
        parsed = urlparse(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('模型服务地址须为不含凭证或查询参数的 HTTP(S) 地址')
        model = text(data.get('model'), '模型标识', 200)
        if not re.fullmatch(r'[A-Za-z0-9_.:/-]+', model):
            raise ValueError('模型标识格式无效')
        window, maximum = data.get('context_window', 32000), data.get('max_tokens', 2048)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (window, maximum)) or not 1024 <= window <= 2000000 or not 1 <= maximum <= min(32000, window):
            raise ValueError('上下文长度或输出上限无效')
        key = data.get('api_key') or ''
        if not isinstance(key, str) or len(key) > 4096 or '\n' in key or '\r' in key:
            raise ValueError('模型密钥格式无效')
        with self.session.begin() as db:
            job = get(db, Job, deployment_job) if deployment_job else None
            if job:
                db.refresh(job, with_for_update=True)
                if job.status != 'running' or job.kind != 'deploy':
                    raise ValueError('部署作业已被停止，未登记模型服务')
            item = ModelService(**owned(), name=text(data.get('name'), '模型服务名称'), model=model,
                                base_url=base, context_window=window, max_tokens=maximum, has_credential=bool(key))
            if key:
                folder = self.directory / 'model-credentials'
                folder.mkdir(mode=0o700, exist_ok=True)
                private_write(folder / (item.id + '.txt'), key)
            db.add(item)
            db.flush()
            if job:
                job.status = 'serving'
                job.result = {'model_id': item.id, 'artifact_id': job.parameters['artifact']['id']}
            return public(item)
