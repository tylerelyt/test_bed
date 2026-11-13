"""Shared persistence for all business modules; old SQLite is a migration source only."""
import json
from pathlib import Path
import sqlite3
from sqlalchemy import select, func, text
from .database import BusinessRecord, BusinessEvent, ContextAudit, connect
from ..workbench_access import check, identity

KINDS = ('runs', 'tasks', 'notes', 'suites', 'batches', 'schedules', 'artifacts')
ACTIVE = ('queued', 'running', 'awaiting_review')


class BusinessRecords:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.engine, self.session = connect(directory)
        self.migrate()

    def migrate(self):
        with self.session.begin() as db:
            if self.engine.dialect.name == 'postgresql':
                db.execute(text('SELECT pg_advisory_xact_lock(79432202)'))
            marker = 'migration:sqlite-business-v1'
            if db.get(ContextAudit, marker): return
            path = self.directory/'native-runs-v1.sqlite3'
            if path.exists():
                with sqlite3.connect(path) as old:
                    backup = path.with_suffix('.sqlite3.pre-platform.bak')
                    if not backup.exists():
                        with sqlite3.connect(backup) as target: old.backup(target)
                        backup.chmod(0o600)
                    tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for kind in KINDS:
                        if kind not in tables: continue
                        for raw, in old.execute('SELECT body FROM '+kind+' ORDER BY rowid'):
                            self.write(db, kind, json.loads(raw), authorize=False)
                    if 'events' in tables:
                        for run_id, seq, raw in old.execute('SELECT run_id,seq,body FROM events'):
                            db.add(BusinessEvent(run_id=run_id, seq=seq, body=json.loads(raw)))
            db.add(ContextAudit(id=marker, body={'kind':'migration','source':'native-runs-v1.sqlite3','source_preserved':True}))

    def row(self, db, kind, item_id, lock=False):
        if kind not in KINDS: raise ValueError('记录类型无效')
        query = select(BusinessRecord).where(BusinessRecord.kind==kind, BusinessRecord.id==item_id)
        return db.scalar(query.with_for_update() if lock else query)

    def write(self, db, kind, item, authorize=True):
        old = self.row(db, kind, item['id'], lock=True)
        if authorize:
            if old: check(old.body)
            item.setdefault('project', old.project if old else identity().project)
            item.setdefault('actor', old.body.get('actor', {}) if old else identity().public())
            check(item)
        values = dict(project=item.get('project','local'), module=item.get('module',''),
                      status=item.get('status',''), body=dict(item))
        if old:
            for key,value in values.items(): setattr(old,key,value)
        else: db.add(BusinessRecord(kind=kind, id=item['id'], **values))

    def get(self, kind, item_id):
        with self.session() as db:
            row = self.row(db, kind, item_id)
            return dict(row.body) if row else None

    def put(self, kind, item, authorize=True):
        with self.session.begin() as db: self.write(db, kind, item, authorize)

    def all(self, kind, project=None, module=None, limit=None):
        if kind not in KINDS: raise ValueError('记录类型无效')
        query = select(BusinessRecord).where(BusinessRecord.kind==kind)
        if project is not None: query = query.where(BusinessRecord.project==project)
        if module is not None: query = query.where(BusinessRecord.module==module)
        query=query.order_by(BusinessRecord.sequence.desc())
        if limit: query=query.limit(limit)
        with self.session() as db: return [dict(x.body) for x in db.scalars(query)]

    def pending(self):
        with self.session() as db:
            return db.scalar(select(func.count()).select_from(BusinessRecord).where(
                BusinessRecord.kind=='runs', BusinessRecord.status.in_(ACTIVE)))

    def trace(self, run_id, after=0, limit=None):
        query=select(BusinessEvent).where(BusinessEvent.run_id==run_id, BusinessEvent.seq>after).order_by(BusinessEvent.seq)
        if limit: query=query.limit(limit)
        with self.session() as db: return [(x.seq,json.dumps(x.body,ensure_ascii=False)) for x in db.scalars(query)]

    def event(self, run, body):
        with self.session.begin() as db:
            self.row(db, 'runs', run['id'], lock=True)
            seq=(db.scalar(select(func.max(BusinessEvent.seq)).where(BusinessEvent.run_id==run['id'])) or 0)+1
            db.add(BusinessEvent(run_id=run['id'],seq=seq,body=body))
            self.write(db,'runs',run)
            return seq
