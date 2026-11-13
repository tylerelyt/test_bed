"""Domain acceptance: original services -> versioned products -> native Harness tools."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from src.search_engine.agent_platform.service import AgentPlatform
from src.search_engine.agent_platform.knowledge import KnowledgeSupply
from src.search_engine.agent_platform.index_adapter import LexicalIndex
from src.search_engine.agent_platform.model_training import LlamaFactoryTraining
from src.search_engine.workbench_access import AccessPolicy,Identity
from src.search_engine.workbench_context import ContextStore
from tools.harness_test_model import server

class SupplyContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.access=AccessPolicy(self.root)
        token=(self.root/'bootstrap-access.txt').read_text().strip()
        self.scope=self.access.bind(self.access.authenticate('Bearer '+token));self.scope.__enter__()
        self.context=ContextStore(self.root)
        self.platform=AgentPlatform(self.root,self.context,self.access,background=False)
        self.resource=self.context.resource('报销制度','knowledge','报销金额超过5000元需要经理审批。','制度第3条')

    def tearDown(self):
        self.platform.close();self.scope.__exit__(None,None,None);self.tmp.cleanup()

    def build(self):
        return self.platform.knowledge.build('财务索引',[{'id':self.resource['id'],'version':1}])

    def test_source_is_read_from_existing_document_service_not_component_outputs(self):
        path=self.root/'legacy.json'
        LexicalIndex().build([{'id':'real-document','content':'采购金额超过9000元需审批'}],path)
        service=KnowledgeSupply(self.platform.session,self.root,self.context,legacy_index=path)
        self.assertEqual(service.legacy_documents()[0]['id'],'real-document')
        saved=service.snapshot_legacy('real-document','采购制度')
        self.assertIn('9000',saved['content']);self.assertIn('legacy-index/document/real-document',saved['source'])
        self.assertNotIn('updates',saved['content'])

    def test_graph_source_reads_typed_facts_without_writing_graph_or_ui_outputs(self):
        from src.search_engine.agent_platform.graph_source import GraphSource
        class Service:
            closed=False
            def get_entity_relations(self,entity):return {'outgoing':[{'target':'经理','predicate':'审批人','doc_id':'制度42'}],'incoming':[]}
            def close(self):self.closed=True
        service=Service()
        saved=self.platform.knowledge.snapshot_graph('报销','审批关系',GraphSource(lambda:service))
        body=json.loads(saved['content'])
        self.assertEqual(body['facts'][0],{'subject':'报销','predicate':'审批人','object':'经理','document_id':'制度42'})
        self.assertTrue(service.closed);self.assertIn('legacy-graph/entity/报销',saved['source'])

    def test_index_freezes_original_document_versions_and_checks_scope_and_tampering(self):
        index=self.build()
        self.context.resource('报销制度','knowledge','新规定10000元',resource_id=self.resource['id'],expected_version=1)
        result=self.platform.knowledge.search(index['id'],'报销审批')
        self.assertEqual(result['documents'][0]['version'],1)
        self.assertIn('5000',result['documents'][0]['content'])
        self.assertNotIn('10000',result['documents'][0]['content'])
        with self.access.bind(Identity('other','other','operator',business_access=True)):
            self.assertEqual(self.platform.knowledge.listing(),[])
            with self.assertRaises(ValueError):self.platform.knowledge.search(index['id'],'审批')
        Path(index['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'摘要'):self.platform.knowledge.search(index['id'],'审批')

    def test_release_resolves_only_explicit_authorized_index_references(self):
        index=self.build();ref=self.platform.knowledge.reference(index['id'],'岗位财务检索')
        bundle=self.context.bundle('财务供应',[{'id':ref['id'],'required':True}])
        position=self.platform.catalog.position('财务','按制度审批');agent=self.platform.catalog.agent(position['id'],'财务Agent')
        model=self.platform.catalog.model({'name':'只登记不调用','model':'test','base_url':'http://127.0.0.1:1/v1'})
        release=self.platform.catalog.release(agent['id'],model['id'],bundle['id'])
        self.assertEqual(release['manifest']['context']['retrievals'][0]['digest'],index['digest'])
        self.assertIn('search_enterprise_knowledge',release['manifest']['tools'])
        self.assertNotIn(self.resource['id'],[r['id'] for r in release['manifest']['context']['resources']])

    def test_training_plan_uses_existing_business_mapping_without_callback_state(self):
        folder=self.root/'model';folder.mkdir();(folder/'config.json').write_text('{}');(folder/'model.safetensors').write_bytes(b'fixture')
        artifact=self.platform.models.artifact({'name':'基础权重','path':str(folder)})
        for stage,rows in [('pt',[{'text':'训练语料'}]),('sft',[{'instruction':'问题','output':'回答'}]),('dpo',[{'instruction':'问题','chosen':'正确','rejected':'错误'}])]:
            dataset=self.platform.models.dataset(stage,rows,stage)
            plan=LlamaFactoryTraining().prepare({'artifact_id':artifact['id'],'dataset_id':dataset['id'],'stage':stage},self.platform.models)
            config=LlamaFactoryTraining().configuration(plan,self.root/'output')
            self.assertEqual(config['stage'],stage)
            self.assertEqual(config['dataset_dir'],str(Path(dataset['path']).parent))
            self.assertNotIn('runtime_id',config);self.assertNotIn('trust_remote_code',config)
            wrong='sft' if stage=='pt' else 'pt'
            with self.assertRaises(ValueError):LlamaFactoryTraining().prepare({'artifact_id':artifact['id'],'dataset_id':dataset['id'],'stage':wrong},self.platform.models)

    def test_rag_validation_uses_explicit_services_and_does_not_publish_results(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(request)
                body=json.dumps({'choices':[{'message':{'content':'需经理审批，依据5000元制度'}}]},ensure_ascii=False).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            index=self.build();config=self.platform.catalog.model({'name':'显式模型','model':'explicit-validation-model','base_url':f'http://127.0.0.1:{server.server_port}/v1'})
            before=len(self.context.catalog()['resources'])
            result=self.platform.validation.rag(index['id'],config['id'],'报销审批')
            self.assertEqual(result['model']['id'],config['id']);self.assertIn('5000',result['prompt_sent'])
            self.assertEqual(calls[0]['model'],'explicit-validation-model')
            self.assertEqual(len(self.context.catalog()['resources']),before)
            self.assertEqual(result['documents'][0]['version'],1)
        finally:server.shutdown();server.server_close()

    def test_runtime_tool_cannot_choose_a_path_or_an_unpublished_index(self):
        import subprocess,sys,os
        index=self.build();manifest=self.root/'runtime-context.json'
        manifest.write_text(json.dumps({'retrievals':[index]}))
        cli=Path(__file__).resolve().parents[1]/'src/search_engine/agent_platform/retrieval_cli.py'
        result=subprocess.run([sys.executable,str(cli)],input=json.dumps({'index_id':'unpublished','query':'审批','path':index['path']}),
            capture_output=True,text=True,env={**os.environ,'WORKBENCH_CONTEXT_FILE':str(manifest)})
        self.assertEqual(result.returncode,2)
        self.assertIn('不属于',json.loads(result.stdout)['error'])

    def test_real_pi_calls_the_published_retrieval_service(self):
        model=server();model.retrieval_query='报销审批'
        threading.Thread(target=model.serve_forever,daemon=True).start()
        try:
            index=self.build();ref=self.platform.knowledge.reference(index['id'],'财务检索')
            bundle=self.context.bundle('财务供应',[{'id':ref['id'],'required':True}])
            position=self.platform.catalog.position('财务','审核报销并引用来源');agent=self.platform.catalog.agent(position['id'],'财务助理')
            config=self.platform.catalog.model({'name':'协议模型','model':'test','base_url':f'http://127.0.0.1:{model.server_port}/v1'})
            release=self.platform.catalog.release(agent['id'],config['id'],bundle['id'])
            run=self.platform.runs.start(agent['id'],release['id'],'6800元报销如何审批？',uuid.uuid4().hex)
            self.platform.runs.execute_next();detail=self.platform.runs.detail(run['id'])
            self.assertEqual(detail['status'],'completed',detail['error'])
            self.assertIn('5000',detail['result']);self.assertIn('制度第3条',detail['result'])
            self.assertTrue(any(e.get('toolName')=='search_enterprise_knowledge' for e in detail['events']))
        finally:model.shutdown();model.server_close()
