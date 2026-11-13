import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from harness_test_model import server
from src.search_engine.agent_platform.database import Position, Release
from src.search_engine.agent_platform.service import AgentPlatform
from src.search_engine.workbench_access import AccessPolicy, Identity, private_write
from src.search_engine.workbench_context import ContextStore


class AgentPlatformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.access = AccessPolicy(self.directory)
        self.token = (self.directory/'bootstrap-access.txt').read_text().strip()
        self.scope = self.access.bind(self.access.authenticate('Bearer '+self.token))
        self.scope.__enter__()
        self.context = ContextStore(self.directory)
        self.platform = AgentPlatform(self.directory, self.context, self.access, background=False)
        self.model = server()
        threading.Thread(target=self.model.serve_forever, daemon=True).start()
        self.position = self.platform.catalog.position('财务审核', '按照企业规定审核报销并引用来源')
        self.agent = self.platform.catalog.agent(self.position['id'], '财务助理')
        self.resource = self.context.resource('报销规定', 'knowledge', '超过5000元的报销需要经理审批', '制度第3条')
        self.bundle = self.context.bundle('财务上下文', [{'id':self.resource['id'], 'required':True}])
        self.config = self.platform.catalog.model({'name':'协议测试模型','model':'test-fixture',
            'base_url':f'http://127.0.0.1:{self.model.server_port}/v1','api_key':'fixture-private-model-token'})
        self.release = self.platform.catalog.release(self.agent['id'], self.config['id'], self.bundle['id'])

    def tearDown(self):
        self.model.release.set()
        self.platform.close()
        self.model.shutdown()
        self.model.server_close()
        self.scope.__exit__(None,None,None)
        self.tmp.cleanup()

    def start(self, task='6800元报销应怎么处理？', request_id=None):
        return self.platform.runs.start(self.agent['id'],self.release['id'],task,request_id or uuid.uuid4().hex)

    def test_candidate_freezes_context_and_position(self):
        self.context.resource('报销规定','knowledge','新版规定',resource_id=self.resource['id'],expected_version=1)
        with self.platform.session.begin() as db:
            db.get(Position,self.position['id']).duties='新职责'
        manifest=self.platform.catalog.listing()['releases'][0]['manifest']
        self.assertIn('5000',manifest['context']['resources'][0]['content'])
        self.assertEqual(manifest['context']['resources'][0]['resolved_version'],1)
        self.assertEqual(manifest['position']['duties'],'按照企业规定审核报销并引用来源')

    def test_real_pi_uses_tool_model_and_persists_native_session(self):
        self.assertTrue(self.platform.runtime.available(),'请先 npm --prefix runtimes/pi ci --ignore-scripts')
        run=self.start()
        self.platform.runs.execute_next()
        detail=self.platform.runs.detail(run['id'])
        self.assertEqual(detail['status'],'completed',detail['error'])
        self.assertIn('5000',detail['result'])
        self.assertEqual(len(self.model.requests),2)
        self.assertIn('read_enterprise_context',[e.get('toolName') for e in detail['events']])
        self.assertTrue((self.directory/detail['native_session']).is_file())
        self.assertNotIn('fixture-private-model-token',json.dumps(detail))
        config=json.loads((self.directory/'agent-workspaces'/run['id']/'pi/models.json').read_text())
        self.assertEqual(config['providers']['workbench']['apiKey'],'${WORKBENCH_MODEL_KEY}')
        adopted=self.platform.catalog.publish(self.release['id'],run['id'])
        self.assertEqual(adopted['status'],'published')
        self.assertEqual(self.platform.catalog.listing()['agents'][0]['current_release'],self.release['id'])

    def test_request_id_deduplicates_and_cannot_change_task(self):
        key=uuid.uuid4().hex
        first=self.start(request_id=key)
        self.assertEqual(self.start(request_id=key)['id'],first['id'])
        with self.assertRaises(ValueError):self.start('另一个任务',key)
        self.assertEqual(len(self.platform.catalog.listing()['runs']),1)

    def test_cancel_queued_never_executes(self):
        run=self.start()
        self.assertEqual(self.platform.runs.cancel(run['id'])['status'],'cancelled')
        self.assertFalse(self.platform.runs.execute_next())
        self.assertEqual(len(self.model.requests),0)

    def test_cross_project_records_not_accessible(self):
        with self.access.bind(Identity('other','another-project','operator',business_access=True)):
            self.assertEqual(self.platform.catalog.listing()['agents'],[])
            with self.assertRaises(ValueError):self.platform.catalog.agent(self.position['id'],'越权')
            with self.assertRaises(ValueError):self.start()

    def test_cannot_publish_without_completed_trial(self):
        run=self.start()
        with self.assertRaises(ValueError):self.platform.catalog.publish(self.release['id'],run['id'])

    def test_revocation_prevents_pending_harness_execution(self):
        run=self.start()
        config=json.loads(self.access.path.read_text());config['principals'][0]['enabled']=False
        private_write(self.access.path,json.dumps(config))
        self.platform.runs.execute_next()
        detail=self.platform.runs.detail(run['id'])
        self.assertEqual(detail['status'],'failed')
        self.assertEqual(self.model.requests,[])

    def test_tampered_release_never_reaches_provider(self):
        run=self.start()
        with self.platform.session.begin() as db:
            release=db.get(Release,self.release['id'])
            release.manifest={**release.manifest,'tools':['bash']}
        self.platform.runs.execute_next()
        self.assertEqual(self.platform.runs.detail(run['id'])['status'],'failed')
        self.assertEqual(self.model.requests,[])

    def test_interrupt_terminates_real_pi_process(self):
        self.model.pause=True
        run=self.start()
        worker=threading.Thread(target=self.platform.runs.execute_next)
        worker.start()
        try:
            self.assertTrue(self.model.received.wait(20),'Pi 未向测试模型提交请求')
            self.platform.runs.cancel(run['id'])
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.platform.runs.detail(run['id'])['status'],'cancelled')
            self.assertEqual(self.platform.runtime.processes,{})
        finally:
            self.model.release.set()
            worker.join(timeout=10)

    def test_management_api_authentication_and_legacy_adapter(self):
        from fastapi.testclient import TestClient
        from src.search_engine.native_runtime import NativeRuntime
        from src.search_engine.workbench_runs import RunStore
        from src.search_engine.platform_api import create_app
        executor=NativeRuntime()
        records=RunStore(executor,self.directory,background=False)
        try:
            with TestClient(create_app(executor,records)) as client:
                self.assertEqual(client.get('/api/platform/overview').status_code,401)
                headers={'Authorization':'Bearer '+self.token}
                response=client.get('/api/platform/overview',headers=headers)
                self.assertEqual(response.status_code,200,response.text)
                self.assertNotIn('fixture-private-model-token',response.text)
                self.assertEqual(client.get('/api/health').status_code,200)
                self.assertFalse(client.get('/api/health').json()['modules_initialized'])
        finally:
            records.close();executor.close()


if __name__=='__main__':unittest.main()
