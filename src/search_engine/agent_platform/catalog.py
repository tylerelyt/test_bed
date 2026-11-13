import hashlib
import json
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import func, select
from ..workbench_access import identity, require_business, require_operator, private_write
from .database import Agent, ModelService, Position, Release, Run
from .pi_runtime import PiRuntime


def text(value, label, maximum=120):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(label + '为空或长度超限')
    return value.strip()


def owned():
    who = identity()
    return {'id': uuid.uuid4().hex, 'project': who.project, 'actor': who.id}


def get(db, cls, item_id):
    item = db.get(cls, item_id)
    if item is None or item.project != identity().project:
        raise ValueError('记录不存在或不属于当前项目')
    return item


def public(item):
    return {column.name: getattr(item, column.name) for column in item.__table__.columns
            if column.name != 'execution_identity'}


class Catalog:
    def __init__(self, session, directory, context):
        self.session, self.directory, self.context = session, Path(directory), context

    def listing(self):
        with self.session() as db:
            return {name: [public(x) for x in db.scalars(select(cls).where(cls.project == identity().project)
                    .order_by(cls.created_at.desc()).limit(200))]
                    for name, cls in [('positions', Position), ('agents', Agent), ('models', ModelService),
                                      ('releases', Release), ('runs', Run)]}

    def position(self, name, duties):
        require_operator()
        with self.session.begin() as db:
            item = Position(**owned(), name=text(name, '岗位名称'), duties=text(duties, '岗位职责', 12000))
            db.add(item)
            db.flush()
            return public(item)

    def agent(self, position_id, name):
        require_operator()
        with self.session.begin() as db:
            position = get(db, Position, position_id)
            item = Agent(**owned(), position_id=position.id, name=text(name, 'Agent 名称'))
            db.add(item)
            db.flush()
            return public(item)

    def model(self, data):
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
            item = ModelService(**owned(), name=text(data.get('name'), '模型服务名称'), model=model,
                                base_url=base, context_window=window, max_tokens=maximum, has_credential=bool(key))
            if key:
                folder = self.directory / 'model-credentials'
                folder.mkdir(mode=0o700, exist_ok=True)
                private_write(folder / (item.id + '.txt'), key)
            db.add(item)
            db.flush()
            return public(item)

    def release(self, agent_id, model_id, bundle_id, bundle_version=None):
        require_business()
        # The existing context service owns authorization and version resolution.
        context = self.context.assemble(bundle_id, bundle_version)
        with self.session.begin() as db:
            agent = get(db, Agent, agent_id)
            position = get(db, Position, agent.position_id)
            model = get(db, ModelService, model_id)
            manifest = {'schema': 1, 'harness': PiRuntime.descriptor(),
                        'position': {'id': position.id, 'name': position.name, 'duties': position.duties},
                        'agent': {'id': agent.id, 'name': agent.name},
                        'model': public(model), 'context': context,
                        'tools': ['read_enterprise_context'],
                        'notice': '本版仅开放读取预配置上下文的工具，不开放 shell 或任意文件访问'}
            digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            version = (db.scalar(select(func.max(Release.version)).where(Release.agent_id == agent.id)) or 0) + 1
            item = Release(**owned(), agent_id=agent.id, version=version, status='candidate', digest=digest,
                           manifest=manifest)
            db.add(item)
            db.flush()
            return public(item)

    def publish(self, release_id, run_id):
        require_business()
        with self.session.begin() as db:
            release, run = get(db, Release, release_id), get(db, Run, run_id)
            if run.release_id != release.id or run.status != 'completed':
                raise ValueError('请先完成该候选版本的试运行并核对结果')
            agent = get(db, Agent, release.agent_id)
            release.status, release.adopted_run, agent.current_release = 'published', run.id, release.id
            return public(release)
