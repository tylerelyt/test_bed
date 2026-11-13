"""Supply boundaries for existing algorithms: shared jobs, inputs and publishable outputs."""
import json
from sqlalchemy import select
from .database import BusinessRecord
from ..workbench_access import check,identity,require_business

DOMAINS=[('离线索引','context'),('在线召回','context'),('RAG','context'),('OAG','context'),
         ('上下文工程','context'),('多模态','context'),('数据回收','model'),('系统监控','operations')]

def domain(module):
    return next((owner for name,owner in DOMAINS if name in module),'operations')

class BusinessOperations:
    def __init__(self,session,context,models):
        self.session,self.context,self.models=session,context,models

    def listing(self):
        with self.session() as db:
            rows=db.scalars(select(BusinessRecord).where(BusinessRecord.kind=='runs',
                BusinessRecord.project==identity().project).order_by(BusinessRecord.sequence.desc()).limit(200))
            result=[]
            for row in rows:
                r=row.body
                result.append({'id':'business:'+row.id,'name':r['action'],'kind':'business','module':r['module'],
                    'domain':domain(r['module']),'status':r['status'],'error':r.get('error',''),
                    'created_at':r['started_at'],'result':{'run_id':row.id,'artifact_count':len(r.get('artifacts',[]))},
                    'source':'business','run_id':row.id})
            return result

    def publish(self,run_id,name,kind='knowledge'):
        require_business()
        with self.session() as db:
            row=db.scalar(select(BusinessRecord).where(BusinessRecord.kind=='runs',BusinessRecord.id==run_id))
            if not row:raise ValueError('运行不存在')
            run=check(row.body)
        if run['status']!='completed':raise ValueError('只有成功完成的运行可以发布产物')
        outputs=run.get('outputs',{})
        if not outputs:raise ValueError('本次运行没有可发布的输出')
        if kind not in ('knowledge','instructions','experience'):raise ValueError('资源类型无效')
        content=json.dumps(outputs,ensure_ascii=False,indent=2)
        return self.context.resource(name,kind,content,source='business-run:'+run_id+'; '+run['module']+'; '+run['action'])
