"""Single-host durable supply jobs, with bounded native-process ownership."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time

from sqlalchemy import select, update

from ..workbench_access import identity, require_business, require_operator
from .common import get, owned, public, text
from .database import Agent, Job, Release

ACTIVE = ('queued', 'running', 'serving', 'cancelling')


class Cancelled(Exception):
    pass


class Jobs:
    def __init__(self, session, directory, access, knowledge, models):
        self.session, self.directory, self.access = session, Path(directory), access
        self.knowledge, self.models = knowledge, models
        self.stop = threading.Event()
        self.shutdown_requested = False
        self.lock = threading.RLock()
        self.processes = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='supply-job')
        self.futures = []
        self.thread = None
        self.cancellation_error = Cancelled

    def listing(self):
        with self.session() as db:
            return [public(x) for x in db.scalars(select(Job).where(Job.project == identity().project)
                    .order_by(Job.created_at.desc()).limit(200))]

    def detail(self, job_id):
        with self.session() as db:
            item = public(get(db, Job, job_id))
        log = self.directory/'jobs'/job_id/'process.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-32000))
                item['log'] = stream.read(32000).decode('utf-8', errors='replace')
        else:
            item['log'] = ''
        return item

    def start(self,data):
        # Compatibility endpoint; new callers enter through the owning supply service.
        if data.get('kind') in ('train','merge','deploy'):
            return self.models.start_operation(data['kind'],data,self)
        if data.get("kind")=="import":return self.knowledge.import_file(data,self)
        return self.enqueue(data)

    def enqueue(self, data, prepare=None):
        require_operator() if data.get('kind') == 'import' else require_business()
        request_id = data.get('request_id')
        if not isinstance(request_id, str) or not re.fullmatch('[a-f0-9]{32}', request_id):
            raise ValueError('作业请求标识须为 32 位十六进制值')
        kind, name = data.get('kind'), text(data.get('name'), '作业名称')
        if kind not in ('import', 'train', 'merge', 'deploy'):
            raise ValueError('作业类型无效')
        job_id = hashlib.sha256((identity().project+':'+request_id).encode()).hexdigest()[:32]
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.session() as db:
            old = db.get(Job, job_id)
            if old:
                get(db, Job, job_id)
                if old.parameters.get('request_sha256') != fingerprint:
                    raise ValueError('同一请求标识不能用于不同作业')
                return public(old)
            if len(list(db.scalars(select(Job).where(Job.status.in_(ACTIVE))))) >= 20:
                raise ValueError('作业队列已满')
        folder = self.directory/'jobs'/job_id
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        params = {'request_sha256': fingerprint}
        if prepare is None:raise ValueError('作业必须由所属供应服务准备')
        params.update(prepare(folder))
        with self.session.begin() as db:
            fields = owned(); fields['id'] = job_id
            job = Job(**fields, kind=kind, name=name, parameters=params, execution_identity=identity().claim())
            db.add(job); db.flush()
            return public(job)

    def cancel(self, job_id):
        require_operator()
        with self.session.begin() as db:
            item = get(db, Job, job_id)
            if item.kind != 'import':
                require_business()
            # Do not silently break a released Agent by stopping its model.
            model_id = item.result.get('model_id')
            if item.status == 'serving' and model_id:
                defaults = set(db.scalars(select(Agent.current_release).where(Agent.project == identity().project)))
                referenced = any(r.id in defaults and r.manifest['model']['id'] == model_id for r in db.scalars(
                    select(Release).where(Release.project == identity().project)))
                if referenced:
                    raise ValueError('此部署正被岗位 Agent 的采用版本使用，请先为相关 Agent 采用其他模型版本')
            if item.status in ACTIVE:
                status = 'cancelled' if item.status == 'queued' else 'cancelling'
                db.execute(update(Job).where(Job.id == job_id, Job.status.in_(ACTIVE)).values(status=status))
        return self.detail(job_id)

    def cancelled(self, job_id):
        with self.session() as db:
            return self.shutdown_requested or self.stop.is_set() or db.get(Job, job_id).status in ('cancelling', 'cancelled')

    def recover(self):
        with self.session.begin() as db:
            for item in db.scalars(select(Job).where(Job.status.in_(ACTIVE))):
                item.status, item.error, item.finished_at = 'interrupted', '执行进程已重启；请核对模型进程和产物，未自动重跑', time.time()

    def execute_next(self, asynchronous=False):
        with self.session.begin() as db:
            item = db.scalar(select(Job).where(Job.status == 'queued').order_by(Job.created_at).limit(1))
            if not item:
                return False
            changed = db.execute(update(Job).where(Job.id == item.id, Job.status == 'queued').values(status='running'))
            if changed.rowcount != 1:
                return True
            job_id, claims = item.id, item.execution_identity
        if asynchronous:
            self.futures.append(self.pool.submit(self.execute, job_id, claims))
        else:
            self.execute(job_id, claims)
        return True

    def execute(self, job_id, claims):
        try:
            with self.access.resume(claims):
                with self.session() as db:
                    item = public(db.get(Job, job_id))
                require_operator() if item['kind'] == 'import' else require_business()
                if self.cancelled(job_id):
                    raise Cancelled()
                p = item['parameters']
                result = (self.knowledge.execute_import(item,self) if item['kind']=='import'
                          else self.models.operations.execute(item,self))
                with self.session.begin() as db:
                    current = db.get(Job, job_id)
                    current.result, current.status, current.finished_at = result, 'completed', time.time()
        except Exception as exc:
            with self.session.begin() as db:
                current = db.get(Job, job_id)
                current.status = 'cancelled' if isinstance(exc, Cancelled) or self.cancelled(job_id) else 'failed'
                current.error = '' if current.status == 'cancelled' else str(exc)[:2000]
                current.finished_at = time.time()

    def loop(self):
        while not self.shutdown_requested and not self.stop.wait(.3):
            self.futures = [x for x in self.futures if not x.done()]
            if len(self.futures) < 2:
                try:
                    self.execute_next(asynchronous=True)
                except Exception:
                    self.stop.wait(1)

    def start_worker(self):
        self.recover()
        self.thread = threading.Thread(target=self.loop, daemon=True, name='supply-dispatcher')
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.pool.shutdown(wait=True)
