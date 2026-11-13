import base64
from io import BytesIO
import json
import socket
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid
import zipfile

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from src.search_engine.agent_platform.context import ContextRepository
from src.search_engine.agent_platform.database import connect, ContextAudit, ModelArtifact, Job
from src.search_engine.agent_platform.service import AgentPlatform
from src.search_engine.workbench_access import AccessPolicy, Identity
from src.search_engine.workbench_context import LegacyContextStore


class SupplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.access = AccessPolicy(self.directory)
        token = (self.directory/'bootstrap-access.txt').read_text().strip()
        self.scope = self.access.bind(self.access.authenticate('Bearer '+token)); self.scope.__enter__()
        self.engine, self.session = connect(self.directory)
        self.context = ContextRepository(self.session, self.directory)
        self.platform = AgentPlatform(self.directory, self.context, self.access, background=False)

    def tearDown(self):
        self.platform.close(); self.engine.dispose()
        self.scope.__exit__(None, None, None); self.tmp.cleanup()

    def job(self, **data):
        return self.platform.jobs.start({'name':'测试作业', 'request_id':uuid.uuid4().hex, **data})

    def test_import_native_docx_and_shared_context_versions(self):
        stream = BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>超过5000元需审批</w:t></w:r></w:p></w:body></w:document>')
        job = self.job(kind='import', filename='制度.docx', content_base64=base64.b64encode(stream.getvalue()).decode())
        self.platform.jobs.execute_next()
        result = self.platform.jobs.detail(job['id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        resource = self.context.get('resources', result['result']['resource_id'])
        self.assertIn('5000', resource['content'])
        self.assertIn('SHA256', resource['source'])
        bundle = self.context.bundle('岗位资料', [{'id':resource['id']}])
        self.context.resource(resource['name'], 'knowledge', '新制度', resource_id=resource['id'], expected_version=1)
        self.assertIn('5000', self.context.assemble(bundle['id'])['resources'][0]['content'])

    def test_damaged_document_and_deduplication(self):
        data = {'kind':'import', 'name':'坏文档', 'request_id':uuid.uuid4().hex, 'filename':'bad.docx', 'content_base64':'YQ=='}
        first = self.platform.jobs.start(data)
        self.assertEqual(first['id'], self.platform.jobs.start(data)['id'])
        with self.assertRaises(ValueError): self.platform.jobs.start({**data, 'name':'其他'})
        self.platform.jobs.execute_next()
        self.assertEqual(self.platform.jobs.detail(first['id'])['status'], 'failed')
        self.assertEqual(self.context.catalog()['resources'], [])

    def test_pdf_import_extracts_actual_page_text(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter(); page = writer.add_blank_page(width=300, height=200)
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),
            NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({
            NameObject('/F1'):writer._add_object(font)})})
        stream = DecodedStreamObject(); stream.set_data(b'BT /F1 12 Tf 20 100 Td (Manager approval required) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = BytesIO(); writer.write(output)
        job = self.job(kind='import', filename='policy.pdf', content_base64=base64.b64encode(output.getvalue()).decode())
        self.platform.jobs.execute_next(); result = self.platform.jobs.detail(job['id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertIn('Manager approval required', self.context.get('resources',result['result']['resource_id'])['content'])

    def test_cancel_queued_and_project_boundary(self):
        job = self.job(kind='import', filename='a.md', content_base64='YQ==')
        with self.access.bind(Identity('another','another-project','operator',business_access=True)):
            self.assertEqual(self.platform.jobs.listing(), [])
            with self.assertRaises(ValueError): self.platform.jobs.cancel(job['id'])
            self.assertEqual(self.context.catalog()['resources'], [])
        self.assertEqual(self.platform.jobs.cancel(job['id'])['status'], 'cancelled')
        self.assertFalse(self.platform.jobs.execute_next())
        self.assertEqual(self.context.catalog()['resources'], [])

    def test_dataset_validation_and_immutable_snapshot(self):
        with self.assertRaises(ValueError): self.platform.models.dataset('坏数据', [{'instruction':'问题'}])
        first = self.platform.models.dataset('岗位样本', [{'instruction':'问题', 'output':'答案'}])
        second = self.platform.models.dataset('岗位样本新版', [{'instruction':'问题', 'output':'新答案'}])
        self.assertNotEqual(first['id'], second['id'])
        self.assertNotEqual(first['digest'], second['digest'])
        self.assertEqual(json.loads(Path(first['path']).read_text())[0]['output'], '答案')

    def test_artifact_requires_actual_files(self):
        with self.assertRaises(ValueError): self.platform.models.artifact({'name':'空目录','path':str(self.directory)})

    def fixture(self, mode='success'):
        """A real owned child process, not a model-quality or native-engine test."""
        path = self.directory/'fixture-engine'
        path.write_text('#!'+sys.executable+'\n'+'''import json,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv[-1]).read_text())
print('fixture native-process log',flush=True)
'''+ ('time.sleep(30)\n' if mode=='wait' else 'raise SystemExit(3)\n' if mode=='fail' else '''out=Path(config.get('output_dir') or config['export_dir']);out.mkdir()
(out/'adapter_config.json').write_text('{}')
(out/'adapter_model.safetensors').write_text('fixture only')
'''))
        path.chmod(0o700)
        model = self.directory/'base'; model.mkdir()
        (model/'config.json').write_text('{}'); (model/'model.safetensors').write_text('fixture only')
        artifact = self.platform.models.artifact({'name':'基础模型','path':str(model)})
        dataset = self.platform.models.dataset('样本', [{'instruction':'问题', 'output':'答案'}])
        return path, artifact, dataset

    def test_training_adapter_records_process_output_and_lineage(self):
        executable, artifact, dataset = self.fixture()
        with patch('src.search_engine.agent_platform.model_operations.probe',return_value='fixture'), patch('src.search_engine.agent_platform.model_operations.python_for',return_value=str(executable)):
            job = self.job(kind='train', artifact_id=artifact['id'], dataset_id=dataset['id'])
            self.platform.jobs.execute_next()
        result = self.platform.jobs.detail(job['id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertIn('fixture native-process log', result['log'])
        with self.session() as db:
            output = db.get(ModelArtifact, result['result']['artifact_id'])
            self.assertEqual(output.source_job, job['id'])
            self.assertEqual(output.format, 'adapter')
        invocation = json.loads((self.directory/'jobs'/job['id']/'invocation.json').read_text())
        self.assertIn('train', invocation['argv'])

    def test_native_failure_never_registers_output(self):
        executable, artifact, dataset = self.fixture('fail')
        with patch('src.search_engine.agent_platform.model_operations.probe',return_value='fixture'), patch('src.search_engine.agent_platform.model_operations.python_for',return_value=str(executable)):
            job = self.job(kind='train', artifact_id=artifact['id'], dataset_id=dataset['id'])
            self.platform.jobs.execute_next()
        self.assertEqual(self.platform.jobs.detail(job['id'])['status'], 'failed')
        self.assertEqual(len(self.platform.models.listing()['artifacts']), 1)

    def test_deployment_waits_for_actual_readiness_and_stops_owned_service(self):
        executable, artifact, _ = self.fixture()
        executable.write_text('#!'+sys.executable+'\n'+'''import sys,time,json
from http.server import BaseHTTPRequestHandler,HTTPServer
port=int(sys.argv[sys.argv.index('--port')+1]);model=sys.argv[sys.argv.index('--served-model-name')+1]
time.sleep(1)
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  body=json.dumps({'data':[{'id':model}]}).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
HTTPServer(('127.0.0.1',port),Handler).serve_forever()
''')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0)); port=listener.getsockname()[1]
        with patch('src.search_engine.agent_platform.model_operations.probe',return_value='fixture'), patch('src.search_engine.agent_platform.model_operations.python_for',return_value=str(executable)):
            job=self.job(kind='deploy',artifact_id=artifact['id'],port=port)
            worker=threading.Thread(target=self.platform.jobs.execute_next);worker.start()
            try:
                time.sleep(.4)
                self.assertEqual(self.platform.catalog.listing()['models'],[])
                for _ in range(60):
                    detail=self.platform.jobs.detail(job['id'])
                    if detail['status']=='serving': break
                    time.sleep(.1)
                self.assertEqual(detail['status'],'serving',detail['error'])
                self.assertEqual(len(self.platform.catalog.listing()['models']),1)
                model_id=detail['result']['model_id']
                self.assertEqual(self.platform.catalog.listing()['models'][0]['id'],model_id)
                self.platform.jobs.cancel(job['id']);worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(self.platform.jobs.detail(job['id'])['status'],'cancelled')
                with socket.socket() as connection:
                    self.assertNotEqual(connection.connect_ex(('127.0.0.1',port)),0)
                resource=self.context.resource('制度','knowledge','原文')
                bundle=self.context.bundle('资料',[{'id':resource['id']}])
                position=self.platform.catalog.position('岗位','职责')
                agent=self.platform.catalog.agent(position['id'],'助理')
                with self.assertRaises(ValueError):self.platform.catalog.release(agent['id'],model_id,bundle['id'])
            finally:
                self.platform.jobs.stop.set();worker.join(timeout=10)

    def test_cancellation_stops_actual_owned_process(self):
        executable, artifact, dataset = self.fixture('wait')
        with patch('src.search_engine.agent_platform.model_operations.probe',return_value='fixture'), patch('src.search_engine.agent_platform.model_operations.python_for',return_value=str(executable)):
            job = self.job(kind='train', artifact_id=artifact['id'], dataset_id=dataset['id'])
            worker = threading.Thread(target=self.platform.jobs.execute_next); worker.start()
            try:
                for _ in range(50):
                    if self.platform.jobs.processes: break
                    time.sleep(.1)
                self.assertTrue(self.platform.jobs.processes)
                self.platform.jobs.cancel(job['id']); worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(self.platform.jobs.detail(job['id'])['status'], 'cancelled')
                self.assertEqual(self.platform.jobs.processes, {})
            finally:
                self.platform.jobs.stop.set(); worker.join(timeout=10)

    def test_migration_preserves_ids_versions_and_does_not_duplicate(self):
        other = self.directory/'migration'; other.mkdir()
        legacy = LegacyContextStore(other)
        resource = legacy.resource('制度','knowledge','旧制度')
        legacy.resource('制度','knowledge','新制度',resource_id=resource['id'],expected_version=1)
        engine, session = connect(other)
        try:
            repository = ContextRepository(session, other)
            self.assertEqual(repository.get('resources', resource['id'],1)['content'],'旧制度')
            self.assertEqual(repository.get('resources', resource['id'])['version'],2)
            ContextRepository(session, other)
            self.assertEqual(len(repository.catalog()['resources']),1)
            self.assertTrue((other/'context-supply-v1.sqlite3.pre-platform.bak').exists())
        finally: engine.dispose()

    def test_shared_notebook_write_is_atomic_on_audit_failure(self):
        resource = self.context.resource('笔记','notebook','初始',permission='read_write')
        bundle = self.context.bundle('授权',[{'id':resource['id'],'permission':'read_write'}])
        with self.assertRaises(ValueError):
            self.context.write_notebook(bundle['id'], resource['id'], 0, '不应写入', 'session')
        self.assertEqual(self.context.get('resources',resource['id'])['version'],1)
        saved = self.context.write_notebook(bundle['id'], resource['id'], 1, '正常写入', 'session')
        self.assertEqual(saved['version'],2)
        with self.session() as db:
            audits = [x.body for x in db.scalars(select(ContextAudit)) if x.body.get('resource_id')==resource['id']]
            self.assertEqual(len(audits),1)
        with patch('uuid.uuid4') as generated:
            generated.return_value.hex = audits[0]['id']
            with self.assertRaises(IntegrityError):
                self.context.write_notebook(bundle['id'], resource['id'], 2, '审计冲突，不应写入', 'session')
        self.assertEqual(self.context.get('resources', resource['id'])['version'], 2)


if __name__ == '__main__': unittest.main()
