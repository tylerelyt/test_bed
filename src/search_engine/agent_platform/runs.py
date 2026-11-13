import hashlib
import json
import re
import threading
import time

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from ..workbench_access import identity, require_business
from .catalog import Agent, Release, Run, get, owned, public, text
from .database import Event


class Runs:
    def __init__(self, session, runtime, access, background=True):
        self.session, self.runtime, self.access = session, runtime, access
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.thread = None
        self.last_error = None
        if background:
            self.recover()
            self.thread = threading.Thread(target=self.loop, daemon=True, name='position-agent-worker')
            self.thread.start()

    def recover(self):
        # Called by the sole worker, never by a restarting management API.
        with self.session.begin() as db:
            for run in db.scalars(select(Run).where(Run.status.in_(['queued', 'running', 'cancelling']))):
                run.status, run.error, run.finished_at = 'interrupted', '服务已重启，请核对原生会话；未自动重跑', time.time()

    def start(self, agent_id, release_id, task, request_id):
        require_business()
        if not isinstance(request_id, str) or not re.fullmatch(r'[a-f0-9]{32}', request_id):
            raise ValueError('运行请求标识须为 32 位十六进制值')
        task = text(task, '任务内容', 12000)
        with self.lock, self.session.begin() as db:
            agent = get(db, Agent, agent_id)
            release = get(db, Release, release_id or agent.current_release)
            if release.agent_id != agent.id:
                raise ValueError('发布版本不属于该 Agent')
            existing = db.scalar(select(Run).where(Run.project == identity().project, Run.request_id == request_id))
            if existing:
                if (existing.agent_id, existing.release_id, existing.task) != (agent.id, release.id, task):
                    raise ValueError('同一请求标识不能用于不同任务')
                return public(existing)
            if not self.runtime.available():
                raise ValueError('Pi 尚未安装，请执行 npm --prefix runtimes/pi ci --ignore-scripts')
            pending = db.scalar(select(func.count()).select_from(Run).where(Run.status.in_(['queued', 'running'])))
            if pending >= 20:
                raise ValueError('岗位 Agent 队列已满，请等待或取消任务')
            run = Run(**owned(), agent_id=agent.id, release_id=release.id, request_id=request_id,
                      execution_identity=identity().claim(), task=task, status='queued')
            db.add(run)
            db.flush()
            return public(run)

    def detail(self, run_id, after=0):
        with self.session() as db:
            run = get(db, Run, run_id)
            events = db.scalars(select(Event).where(Event.run_id == run.id, Event.seq > after)
                               .order_by(Event.seq).limit(100)).all()
            return {**public(run), 'events': [{'seq': x.seq, **x.body} for x in events]}

    def cancel(self, run_id):
        require_business()
        with self.lock, self.session.begin() as db:
            run = get(db, Run, run_id)
            if run.status == 'queued':
                changed = db.execute(update(Run).where(Run.id == run_id, Run.status == 'queued')
                                     .values(status='cancelled', finished_at=time.time()))
                if not changed.rowcount:
                    db.refresh(run)
            if run.status == 'running':
                db.execute(update(Run).where(Run.id == run_id, Run.status == 'running').values(status='cancelling'))
        self.runtime.stop(run_id)
        return self.detail(run_id)

    def event(self, run_id, body):
        with self.lock, self.session.begin() as db:
            seq = (db.scalar(select(func.max(Event.seq)).where(Event.run_id == run_id)) or 0) + 1
            db.add(Event(run_id=run_id, seq=seq, body=body))

    def execute_next(self):
        with self.lock, self.session.begin() as db:
            run = db.scalar(select(Run).where(Run.status == 'queued').order_by(Run.created_at).limit(1))
            if run is None:
                return False
            release = db.get(Release, run.release_id)
            claims, manifest, run_id, task = run.execution_identity, release.manifest, run.id, run.task
            expected_digest = release.digest
            claimed = db.execute(update(Run).where(Run.id == run_id, Run.status == 'queued').values(status='running'))
            if claimed.rowcount != 1:
                return True
        try:
            with self.access.resume(claims):
                require_business()
                digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if digest != expected_digest:
                    raise ValueError('发布包校验失败，未启动 Harness')
                self.event(run_id, {'type': 'platform_started', 'harness': manifest['harness'], 'digest': digest})
                def cancelled():
                    with self.session() as db:
                        return self.stop.is_set() or db.get(Run, run_id).status == 'cancelling'
                result = self.runtime.execute(run_id, manifest, task, lambda event:self.event(run_id, event), cancelled)
                with self.session.begin() as db:
                    current = db.get(Run, run_id)
                    current.result, current.native_session = result['result'], result['native_session']
                    current.status = 'cancelled' if cancelled() else 'completed'
                    current.finished_at = time.time()
        except Exception as exc:
            # Internal terminal bookkeeping must survive credential revocation.
            with self.session.begin() as db:
                current = db.get(Run, run_id)
                current.status = 'cancelled' if current.status == 'cancelling' else 'failed'
                current.error, current.finished_at = str(exc)[:2000], time.time()
        with self.session() as db:
            current = db.get(Run, run_id)
            self.event(run_id, {'type': 'platform_settled', 'status': current.status, 'error': current.error})
        return True

    def loop(self):
        while not self.stop.wait(.3):
            try:
                self.execute_next()
                self.last_error = None
            except Exception:
                self.last_error = '任务存储暂不可用，请检查执行进程日志和数据库'
                # A store outage does not spawn a second worker or replay a task.
                self.stop.wait(1)

    def close(self):
        self.stop.set()
        with self.runtime.lock:
            for run_id in list(self.runtime.processes):
                self.runtime.stop(run_id)
        if self.thread:
            self.thread.join(timeout=8)
