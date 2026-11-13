"""Context supply owns source snapshots, index artifacts and their retrieval contract."""
import hashlib
import base64
import json
from pathlib import Path
import shutil
from sqlalchemy import select
from .common import get, owned, public, text
from .database import ContextIndex
from .index_adapter import LexicalIndex
from .files import extract,digest
from .graph_source import GraphSource
from ..workbench_access import identity, require_business, require_operator


class KnowledgeSupply:
    def __init__(self, session, directory, context, adapter=None, legacy_index=None):
        self.session, self.directory, self.context = session, Path(directory), context
        self.adapter = adapter or LexicalIndex()
        self.legacy_index = Path(legacy_index or Path(__file__).resolve().parents[3]/'models/index_data.json')

    def import_file(self,data,queue):
        require_operator()
        return queue.enqueue({**data,'kind':'import'},prepare=lambda folder:self.prepare_import(data,folder))

    def prepare_import(self,data,folder):
        filename=text(data.get('filename'),'文件名',200);suffix=Path(filename).suffix.lower()
        if suffix not in ('.txt','.md','.csv','.json','.pdf','.docx'):raise ValueError('不支持此文件格式')
        try:raw=base64.b64decode(data.get('content_base64',''),validate=True)
        except (ValueError,TypeError):raise ValueError('文件编码无效')
        if not 0<len(raw)<=5_000_000:raise ValueError('导入文件须为 1 字节—5 MB')
        path=folder/('source'+suffix);path.write_bytes(raw);path.chmod(0o600)
        return {'filename':filename,'source':str(path),'sha256':digest(path),'suffix':suffix}

    def execute_import(self,item,execution):
        p=item['parameters'];source=Path(p['source'])
        if digest(source)!=p['sha256']:raise ValueError('原文件摘要校验失败')
        content=extract(source,p['suffix'])
        if execution.cancelled(item['id']):raise execution.cancellation_error()
        resource=self.context.resource(item['name'],'knowledge',content,
            '导入文件：'+p['filename']+'；SHA256：'+p['sha256'])
        return {'resource_id':resource['id'],'version':resource['version'],'chars':len(content)}

    def listing(self):
        with self.session() as db:
            return [public(item) for item in db.scalars(select(ContextIndex).where(
                ContextIndex.project == identity().project).order_by(ContextIndex.created_at.desc()).limit(100))]

    def legacy_documents(self):
        require_business()
        documents = self.adapter.documents(self.legacy_index)
        return [{'id': key, 'chars': len(value), 'preview': value[:100]} for key,value in list(documents.items())[:1000]]

    def snapshot_legacy(self, doc_id, name):
        require_business()
        doc_id=text(doc_id,'原文档标识',500)
        name=text(name,'资料快照名称')
        content = self.adapter.documents(self.legacy_index).get(doc_id)
        if content is None: raise ValueError('原索引中不存在该文档')
        digest = hashlib.sha256(content.encode()).hexdigest()
        return self.context.resource(name, 'knowledge', content,
            source='legacy-index/document/'+doc_id+'; sha256='+digest)

    def snapshot_graph(self,entity,name,source=None):
        require_business()
        name=text(name,'图谱快照名称')
        facts=(source or GraphSource()).read(entity)
        content=json.dumps(facts,ensure_ascii=False,indent=2)
        digest=hashlib.sha256(content.encode()).hexdigest()
        return self.context.resource(name,'knowledge',content,source='legacy-graph/entity/'+facts['entity']+'; sha256='+digest)

    def build(self, name, refs):
        require_operator()
        name = text(name, '索引名称')
        if not isinstance(refs,list) or not 1 <= len(refs) <= 30:
            raise ValueError('请选择 1—30 个资料版本')
        documents, seen = [], set()
        for ref in refs:
            if not isinstance(ref,dict) or ref.get('id') in seen: raise ValueError('资料选择无效或重复')
            resource = self.context.get('resources', ref.get('id'), ref.get('version'))
            if resource['kind'] != 'knowledge': raise ValueError('检索索引只接受资料资源，不接受工具定义或界面输出')
            seen.add(resource['id'])
            documents.append({'id': resource['id'], 'version': resource['version'], 'name':resource['name'],
                              'source':resource['source'], 'content':resource['content']})
        if sum(len(d['content']) for d in documents)>1_000_000:
            raise ValueError('单个索引最多一百万字符，请拆成多个索引')
        fields = owned(); folder = self.directory/'context-indexes'/fields['id']
        folder.mkdir(mode=0o700,parents=True)
        path = folder/'index.json'
        try:
            self.adapter.build(documents,path)
            path.chmod(0o600)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {'schema':1,'engine':'existing-inverted-index-v1','engine_digest':self.adapter.fingerprint(),
                        'sources':[{k:v for k,v in d.items() if k!='content'} for d in documents]}
            with self.session.begin() as db:
                item = ContextIndex(**fields, name=name, digest=digest, path=str(path), manifest=manifest)
                db.add(item);db.flush();return public(item)
        except BaseException:
            shutil.rmtree(folder);raise

    def get(self, index_id):
        with self.session() as db: return public(get(db,ContextIndex,index_id))

    def search(self, index_id, query, limit=5):
        require_operator()
        item = self.get(index_id)
        return search_artifact(item,query,limit,self.adapter)

    def reference(self,index_id,name):
        require_operator()
        item = self.get(index_id)
        return self.context.resource(name,'retrieval',json.dumps({'index_id':item['id'],'digest':item['digest']}),
                                     source='context-index:'+item['id'])

    def freeze(self, assembled):
        """Resolve only explicit retrieval resources; a UI field never selects an index."""
        artifacts = []
        for resource in assembled['resources']:
            if resource['kind'] != 'retrieval': continue
            reference = json.loads(resource['content'])
            item = self.get(reference['index_id'])
            if item['digest'] != reference['digest']: raise ValueError('检索资源与索引摘要不一致')
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['digest']:
                raise ValueError('索引文件已变化，不能发布')
            artifacts.append(item)
        return {**assembled,'retrievals':artifacts}


def search_artifact(item, query, limit=5, adapter=None):
    query = text(query,'检索问题',1000)
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=10: raise ValueError('返回条数须为 1—10')
    if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['digest']:
        raise ValueError('索引摘要校验失败')
    adapter=adapter or LexicalIndex()
    if adapter.fingerprint()!=item['manifest']['engine_digest']:
        raise ValueError('检索实现已变化，请用当前版本重建索引')
    return {'index_id':item['id'],'digest':item['digest'],'query':query,
            'documents':adapter.search(item['path'],item['manifest']['sources'],query,limit)}
