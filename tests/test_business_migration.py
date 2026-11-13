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
    def test_job_api_only_manages_submitted_supply_jobs_and_preserves_callback_history(self):
        from fastapi.testclient import TestClient
        from src.search_engine.platform_api import create_app
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'database-url.txt').write_text('sqlite:///'+str(root/'platform.sqlite3'))
            with gr.Blocks() as blocks:
                with gr.Tab('检索'):
                    output=gr.JSON()
                    gr.Button('查看状态').click(lambda:{'ready':True},[],output)
            runtime=NativeRuntime(lambda:blocks);store=RunStore(runtime,root)
            try:
                runtime.layout('session');fn=next(iter(blocks.fns))
                run=store.start('session',fn,{})
                list(store.events(run['run_id']))
                token=(root/'bootstrap-access.txt').read_text().strip()
                with TestClient(create_app(runtime,store)) as client:
                    client.headers['Authorization']='Bearer '+token
                    self.assertEqual(client.get('/api/platform/jobs').json(),[])
                    job=client.post('/api/platform/context-supply/import',json={
                        'name':'真实导入作业','request_id':__import__('uuid').uuid4().hex,
                        'filename':'policy.md','content_base64':'YQ=='}).json()
                    rows=client.get('/api/platform/jobs').json()
                    self.assertEqual([row['id'] for row in rows],[job['id']])
                    self.assertEqual(rows[0]['kind'],'import')
                    pseudo='/api/platform/jobs/business:'+run['run_id']
                    self.assertEqual(client.get(pseudo).status_code,400)
                    self.assertEqual(client.post(pseudo+'/cancel',json={}).status_code,400)
                    self.assertEqual(client.get('/api/native/runs/'+run['run_id']).json()['status'],'completed')
                    self.assertEqual(client.post('/api/platform/jobs/'+job['id']+'/cancel',json={}).json()['status'],'cancelled')
                self.assertEqual(store.get('runs',run['run_id'])['outputs'][str(output._id)]['value'],{'ready':True})
            finally:store.close();runtime.close()

    def test_legacy_callback_cannot_override_model_or_be_published_as_knowledge(self):
        from fastapi.testclient import TestClient
        from src.search_engine.platform_api import create_app
        with tempfile.TemporaryDirectory() as directory:
            with gr.Blocks() as blocks:
                with gr.Tab('RAG'):
                    field=gr.Textbox(value='原输入',label='问题');output=gr.JSON()
                    gr.Button('执行').click(lambda query:{'answer':query},field,output)
            runtime=NativeRuntime(lambda:blocks);store=RunStore(runtime,directory)
            try:
                runtime.layout('session');fn=next(iter(blocks.fns))
                with self.assertRaisesRegex(ValueError,'不接受供应覆盖'):
                    store.start('session',fn,{str(field._id):'用户输入'},metadata={'supply':{'model_id':'arbitrary'}})
                run=store.start('session',fn,{str(field._id):'用户输入'});list(store.events(run['run_id']))
                record=store.get('runs',run['run_id'])
                self.assertEqual(record['inputs'][str(field._id)],'用户输入')
                self.assertNotIn('model_service',record)
                token=(Path(directory)/'bootstrap-access.txt').read_text().strip()
                with TestClient(create_app(runtime,store)) as client:
                    response=client.post('/api/platform/business-runs/'+run['run_id']+'/publish',
                        headers={'Authorization':'Bearer '+token},json={'name':'界面输出'})
                    self.assertEqual(response.status_code,410)
                    self.assertEqual(store.context.catalog()['resources'],[])
            finally:store.close();runtime.close()

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
