"""Shared SQL context repository; legacy rules and MCP contracts are retained."""
from contextlib import nullcontext
import json
from pathlib import Path
import sqlite3
import time

from sqlalchemy import select, text
from .database import ContextResource, ContextBundle, ContextAudit
from ..workbench_context import LegacyContextStore
from ..workbench_access import identity, visible, check

TABLES = {'resources': ContextResource, 'bundles': ContextBundle}


class ContextRepository(LegacyContextStore):
    def __init__(self, session, directory):
        self.session, self.directory = session, Path(directory)
        self.migrate_legacy()

    def migrate_legacy(self):
        marker = 'migration:sqlite-context-v1'
        with self.session.begin() as db:
            if db.bind.dialect.name == 'postgresql':
                db.execute(text('SELECT pg_advisory_xact_lock(79432201)'))
            if db.get(ContextAudit, marker): return
            path = self.directory/'context-supply-v1.sqlite3'
            if path.exists():
                with sqlite3.connect(path) as old:
                    backup = path.with_suffix('.sqlite3.pre-platform.bak')
                    if not backup.exists():
                        with sqlite3.connect(backup) as snapshot:
                            old.backup(snapshot)
                        backup.chmod(0o600)
                    for table, cls in TABLES.items():
                        for item_id, version, raw in old.execute('SELECT id,version,body FROM '+table):
                            if db.get(cls,(item_id,version)) is None:
                                db.add(cls(id=item_id,version=version,body=json.loads(raw)))
                    for item_id, raw in old.execute('SELECT id,body FROM audit'):
                        if db.get(ContextAudit,item_id) is None:
                            db.add(ContextAudit(id=item_id,body=json.loads(raw)))
            db.add(ContextAudit(id=marker,body={'kind':'migration','at':time.time(),'source_preserved':True}))

    def _read(self, table, item_id, version=None):
        if table not in TABLES or not isinstance(item_id,str): raise ValueError('上下文标识无效')
        with self.session() as db:
            query=select(TABLES[table]).where(TABLES[table].id==item_id)
            if version is not None: query=query.where(TABLES[table].version==version)
            row=db.scalar(query.order_by(TABLES[table].version.desc()).limit(1))
            if row is None: raise ValueError('上下文资源不存在')
            return check(dict(row.body))

    def catalog(self, query=''):
        result={}
        for table,cls in TABLES.items():
            with self.session() as db:
                rows=list(db.scalars(select(cls).order_by(cls.version.desc())))
            latest={}
            for row in rows:
                if row.id not in latest:latest[row.id]=dict(row.body)
            items=sorted(latest.values(),key=lambda x:x.get('updated_at',0),reverse=True)
            result[table]=[{k:v for k,v in x.items() if k!='content'} for x in items if visible(x)
                and query.lower() in json.dumps({k:v for k,v in x.items() if k!='content'},ensure_ascii=False).lower()
                and (identity().role=='operator' or (x['id'] in identity().bundles if table=='bundles'
                     else self.agent_resource_allowed(x['id'])))][:100]
        return result

    def save(self,table,item,expected_version=None,*,connection=None,audit=None):
        if table not in TABLES:raise ValueError('上下文类型无效')
        item={**item,'project':item.get('project',identity().project),'actor':identity().public()}
        check(item);cls=TABLES[table]
        with (self.session.begin() if connection is None else nullcontext(connection)) as db:
            previous=db.scalar(select(cls).where(cls.id==item['id']).order_by(cls.version.desc()).limit(1).with_for_update())
            if previous:check(previous.body)
            current=previous.version if previous else None
            if current!=expected_version:raise ValueError('上下文版本已变化，请重新读取后修改')
            item.update(version=(current or 0)+1,updated_at=time.time())
            db.add(cls(id=item['id'],version=item['version'],body=item))
            if audit is not None:db.add(ContextAudit(id=audit['id'],body={**audit,'version':item['version']}))
        return item

    def write_notebook(self,bundle_id,resource_id,expected_version,content,session,bundle_version=None):
        if not isinstance(session,str) or not 0<len(session)<=120:raise ValueError('写入需要会话来源')
        import uuid
        with self.session.begin() as db:
            # Lock the current grant for the whole write transaction.
            db.scalar(select(ContextBundle).where(ContextBundle.id==bundle_id).order_by(ContextBundle.version.desc()).limit(1).with_for_update())
            old=self.get('bundles',bundle_id,bundle_version);latest=self.get('bundles',bundle_id)
            refs=[next((r for r in b['resources'] if r['id']==resource_id),None) for b in (old,latest)]
            if any(not r or r['permission']!='read_write' for r in refs):raise ValueError('当前配置未授予该笔记写入权限')
            resource=self.get('resources',resource_id)
            if resource['kind']!='notebook' or resource['permission']!='read_write':raise ValueError('资源不可写')
            audit={'id':uuid.uuid4().hex,'bundle_id':bundle_id,'resource_id':resource_id,'session':session,
                   'at':time.time(),'project':identity().project,'actor':identity().public()}
            return self.resource(resource['name'],'notebook',content,resource['source'],'read_write',
                resource['tags'],resource_id,expected_version,connection=db,audit=audit)

    def record_supply(self,assembled,run_id,session):
        import uuid,hashlib
        item={'id':uuid.uuid4().hex,'kind':'callback_supply','bundle_id':assembled['bundle_id'],
              'bundle_version':assembled['version'],'project':identity().project,'actor':identity().public(),
              'run_id':run_id,'session':session,'at':time.time(),'used_chars':assembled['used_chars'],
              'max_chars':assembled['max_chars'],
              'resources':[{'id':r['id'],'version':r['resolved_version'],'name':r['name']} for r in assembled['resources']],
              'content_sha256':hashlib.sha256(json.dumps(assembled['resources'],sort_keys=True).encode()).hexdigest()}
        with self.session.begin() as db:db.add(ContextAudit(id=item['id'],body=item))
        return item
