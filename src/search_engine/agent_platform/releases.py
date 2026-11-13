import hashlib
import json
from sqlalchemy import select, func
from ..workbench_access import identity, require_business
from .common import get, owned, public
from .database import Agent, ModelService, Position, Release, Run, Job
from .pi_runtime import PiRuntime


class Releases:
    def __init__(self, session, context):
        self.session, self.context = session, context

    def check_model(self, db, model_id):
        deployment = db.scalar(select(Job).where(Job.kind == 'deploy', Job.project == identity().project,
            Job.result['model_id'].as_string() == model_id))
        if deployment and deployment.status != 'serving':
            raise ValueError('此模型的受管部署已停止或中断，请先重新部署再生成候选版本')

    def release(self, agent_id, model_id, bundle_id, bundle_version=None):
        require_business()
        # The existing context service owns authorization and version resolution.
        context = self.context.assemble(bundle_id, bundle_version)
        with self.session.begin() as db:
            agent = get(db, Agent, agent_id)
            position = get(db, Position, agent.position_id)
            model = get(db, ModelService, model_id)
            self.check_model(db, model.id)
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
            self.check_model(db, release.manifest['model']['id'])
            agent = get(db, Agent, release.agent_id)
            release.status, release.adopted_run, agent.current_release = 'published', run.id, release.id
            return public(release)
