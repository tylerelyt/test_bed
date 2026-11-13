import json,os,tempfile,threading,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import gradio as gr
from sqlalchemy import select
from src.search_engine.native_runtime import NativeRuntime
from src.search_engine.workbench_runs import RunStore
from src.search_engine.agent_platform.service import AgentPlatform
from src.search_engine.agent_platform.database import BusinessRecord
from src.search_engine.workbench_access import AccessPolicy,Identity
import test_workbench_runs as old_runs
import test_workbench_evaluations as old_evaluations

class UnifiedRuns(old_runs.RunStoreTests):
    def setUp(self):
        super().setUp();self.store.close()
        Path(self.directory.name,'database-url.txt').write_text('sqlite:///'+str(Path(self.directory.name,'platform.sqlite3')))
        self.store=RunStore(self.runtime,self.directory.name);self.addCleanup(self.store.close)

class UnifiedEvaluations(old_evaluations.EvaluationTests):
    def setUp(self):
        super().setUp();self.store.close()
        (self.store.directory/'database-url.txt').write_text('sqlite:///'+str(self.store.directory/'platform.sqlite3'))
        self.store=RunStore(self.runtime,self.store.directory);self.addCleanup(self.store.close);self.eval=self.store.evaluations

class BusinessSupplyTests(unittest.TestCase):
    def test_all_eight_domains_share_jobs_model_context_and_output_publication(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(data)
                body=json.dumps({'id':'response','choices':[{'index':0,'message':{'role':'assistant','content':data['messages'][0]['content']},'finish_reason':'stop'}]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);(folder/'database-url.txt').write_text('sqlite:///'+str(folder/'platform.sqlite3'))
                with gr.Blocks() as blocks:
                    with gr.Tabs():
                        for name in ['离线索引构建','在线召回排序','RAG','OAG 图谱推理','上下文工程','多模态系统','数据回收训练','系统监控']:
                            with gr.Tab(name):
                                question=gr.Textbox(value='问题',label='问题');result=gr.JSON()
                                def work(query):
                                    from src.search_engine.business_execution import OpenAI
                                    client=OpenAI(api_key='unused',base_url='http://127.0.0.1:9/v1')
                                    return {'answer':client.chat.completions.create(model='must-be-overridden',messages=[{'role':'user','content':query}]).choices[0].message.content,'echo':'selected-private-key'}
                                gr.Button('执行').click(work,question,result)
                runtime=NativeRuntime(lambda:blocks);store=RunStore(runtime,folder);platform=AgentPlatform(folder,store.context,store.access,background=False)
                try:
                    runtime.layout('session')
                    resource=store.context.resource('制度','knowledge','金额超过5000元需审批')
                    bundle=store.context.bundle('共享上下文',[{'id':resource['id'],'required':True}])
                    model=platform.catalog.model({'name':'业务模型','api_key':'selected-private-key','model':'selected-model','base_url':f'http://127.0.0.1:{server.server_port}/v1'})
                    for fn in blocks.fns.values():
                        fn_id=fn._id;cid=str(fn.inputs[0]._id)
                        run=store.start('session',fn_id,{cid:'如何处理'},metadata={'supply':{'model_id':model['id'],'context_binding':{'bundle_id':bundle['id'],'version':1,'component_id':cid}}})
                        list(store.events(run['run_id']));record=store.get('runs',run['run_id'])
                        self.assertEqual(record['status'],'completed',record.get('error'))
                        self.assertEqual(record['model_service']['id'],model['id'])
                        self.assertNotIn('selected-private-key',json.dumps(record))
                        self.assertIn('5000',json.dumps(record['outputs'],ensure_ascii=False))
                    self.assertEqual(len(calls),8);self.assertTrue(all(x['model']=='selected-model' for x in calls))
                    self.assertEqual(len(platform.business.listing()),8)
                    published=platform.business.publish(run['run_id'],'业务结果')
                    self.assertIn('5000',published['content'])
                    with store.access.bind(Identity('other','other','operator',business_access=True)):
                        self.assertEqual(platform.business.listing(),[])
                        with self.assertRaises(PermissionError):platform.business.publish(run['run_id'],'越权')
                    with platform.session() as db:
                        self.assertEqual(len(list(db.scalars(select(BusinessRecord).where(BusinessRecord.kind=='runs')))),8)
                finally:store.close();platform.close();runtime.close()
        finally:server.shutdown();server.server_close()

class BusinessHistoryMigration(unittest.TestCase):
    def test_all_record_kinds_and_events_migrate_once_without_new_writes_to_old_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runtime=NativeRuntime();old=RunStore(runtime,root)
            kinds=('runs','tasks','notes','suites','batches','schedules','artifacts')
            ids={kind:__import__('uuid').uuid4().hex for kind in kinds}
            try:
                for kind in kinds:old.put(kind,{'id':ids[kind],'module':'RAG','status':'completed','version':2,'revisions':[{'version':1}]})
                run=old.get('runs',ids['runs']);old.append_event(run,{'type':'completed','updates':{'answer':'historical result'}})
            finally:old.close()
            before=hashlib.sha256(old.path.read_bytes()).hexdigest()
            (root/'database-url.txt').write_text('sqlite:///'+str(root/'platform.sqlite3'))
            current=RunStore(runtime,root)
            try:
                for kind in kinds:self.assertEqual(current.get(kind,ids[kind])['version'],2)
                self.assertEqual(len(current.trace(ids['runs'])),1)
                item=current.get('tasks',ids['tasks']);item['version']=3;current.put('tasks',item)
                self.assertEqual(hashlib.sha256(old.path.read_bytes()).hexdigest(),before)
                self.assertTrue(old.path.with_suffix('.sqlite3.pre-platform.bak').exists())
            finally:current.close();runtime.close()
