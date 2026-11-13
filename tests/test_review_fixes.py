import asyncio
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from unittest.mock import patch
import test_workbench_evaluations as fixtures
from src.search_engine.workbench_access import AccessPolicy, AccessDenied, history_session, private_write
from src.search_engine.workbench_context import ContextStore, register_context_mcp
from src.search_engine.workbench_history import HistoryStore
from src.search_engine.workbench_api import make_handler
from src.search_engine.workbench_telemetry import observe_usage, record_usage


class ExecutionFixTests(unittest.TestCase):
    setUp = fixtures.EvaluationTests.setUp
    run_answer = fixtures.EvaluationTests.run_answer
    task = fixtures.EvaluationTests.task
    finish_batch = fixtures.EvaluationTests.finish_batch

    def test_invalid_later_variant_submits_nothing(self):
        task = self.task()
        resource = self.store.context.resource('操作说明', 'instructions', '核对依据')
        bundle = self.store.context.bundle('配置', [{'id': resource['id']}])
        task = self.store.update_task(task['id'], task['rules'], expected_version=1,
            context_binding={'bundle_id': bundle['id'], 'version': 1, 'component_id': str(self.query._id)})
        suite = self.eval.save_suite('在线召回排序', '预检', [task['id']],
            [{'name': '有效', 'values': {}}, {'name': '无效', 'values': {str(self.query._id): None}}], 1)
        calls = self.calls
        with self.assertRaises(ValueError): self.eval.start(suite['id'])
        self.assertEqual(self.calls, calls)
        self.assertEqual(self.store.listing('batches', '在线召回排序'), [])
        self.assertFalse(any(key.startswith('eval-') for key in self.runtime.sessions))

    def test_admission_failure_is_explicit_and_not_a_baseline(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '提交故障', [task['id']], [{'name': '默认', 'values': {}}], 2)
        original = self.store.start
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise OSError('模拟提交故障')
            return original(*args, **kwargs)
        calls = self.calls
        with patch.object(self.store, 'start', side_effect=fail_second): batch = self.eval.start(suite['id'])
        self.assertFalse(batch['complete'])
        self.assertTrue(batch['terminal'])
        self.assertEqual(batch['status'], 'submission_failed')
        self.assertEqual(len(batch['rows']), batch['expected_count'])
        self.assertEqual(batch['rows'][0]['status'], 'cancelled')
        self.assertEqual(self.calls, calls)
        self.assertIsNone(batch['recommendation'])
        with self.assertRaises(ValueError): self.eval.baseline(suite['id'], batch['id'])

    def test_event_http_waits_in_same_queue_and_is_recorded(self):
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def hold(): entered.set(); release.wait(5)
        self.store.enqueue(('module', '在线召回排序'), hold)
        self.assertTrue(entered.wait(2))
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.runtime, self.store))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        token = (self.store.directory / 'bootstrap-access.txt').read_text().strip()
        def invoke():
            request = Request(f'http://127.0.0.1:{server.server_port}/api/native/event',
                data=json.dumps({'session': 'test-http', 'fn_id': self.ids['answer'], 'values': {}}).encode(),
                headers={'Authorization': 'Bearer ' + token})
            with urlopen(request, timeout=5) as response: list(response)
            completed.set()
        client = threading.Thread(target=invoke)
        try:
            client.start()
            self.assertFalse(completed.wait(.15))
            self.assertEqual(self.calls, 0)
            release.set();client.join(5)
            self.assertTrue(completed.is_set())
            self.assertEqual(self.calls, 1)
            self.assertEqual(len(self.store.listing('runs', '在线召回排序')), 1)
        finally:
            release.set();server.shutdown();server.server_close();client.join(5)

    def test_finished_evaluations_release_sessions(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '回收', [task['id']],
            [{'name': str(i), 'values': {}} for i in range(4)], 5)
        batch = self.finish_batch(self.eval.start(suite['id']))
        self.assertTrue(batch['complete'])
        self.assertEqual(len(self.runtime.sessions), 1)
        self.assertEqual(self.runtime.session_pins, {})

    def test_upload_quota_expiry_and_active_lease(self):
        with patch.dict(os.environ, {'WORKBENCH_UPLOAD_MAX_MB': '1', 'WORKBENCH_UPLOAD_QUOTA_MB': '1'}):
            with self.assertRaises(ValueError): self.runtime.store_upload('large', io.BytesIO(), 1024*1024+1)
            file = self.runtime.store_upload('ok', io.BytesIO(b'x'*(1024*1024)), 1024*1024)
            with self.assertRaises(ValueError): self.runtime.store_upload('other', io.BytesIO(b'x'), 1)
            lease = self.runtime.pin('test', {'file': file})
            self.runtime.cleanup_resources(now=time.monotonic()+7200)
            self.assertIn(file['url'].rsplit('/', 1)[-1], self.runtime.assets)
            self.runtime.unpin(lease)
            self.runtime.release_session('test')
            self.runtime.cleanup_resources(now=time.monotonic()+7200)
            self.assertEqual(self.runtime.upload_records, {})
            self.assertEqual(self.runtime.reserved_upload_bytes, 0)

    def test_partial_usage_never_enters_cost_ranking(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '缺失用量', [task['id']], [{'name': '默认', 'values': {}}], 2)
        batch = self.finish_batch(self.eval.start(suite['id']))
        row = self.store.get('runs', batch['cells'][1]['run_id']);row['usage'] = [];self.store.put('runs', row)
        group = self.eval.summary(batch['id'])['groups'][0]
        self.assertFalse(group['usage_complete'])
        self.assertIsNone(group['cost']);self.assertIsNone(group['input_tokens'])

    def test_stream_and_missing_usage_are_explicit(self):
        with observe_usage() as calls:
            record_usage(SimpleNamespace(model='m', usage=None))
            record_usage(SimpleNamespace(model='m', usage=None), stream=True, call_id='a')
            record_usage(SimpleNamespace(model='m', usage=None), stream=True, call_id='a')
            record_usage(SimpleNamespace(model='m', usage={'prompt_tokens': 2, 'completion_tokens': 3}), stream=True, call_id='a')
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]['usage_known']);self.assertTrue(calls[1]['usage_known'])


class AccessAndHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.policy = AccessPolicy(self.directory.name)
        self.tokens = {'alice': 'a'*32, 'bob': 'b'*32, 'agent': 'c'*32}
        self.config = {'version': 1, 'principals': [
            {'id': 'alice', 'project': 'a', 'role': 'operator', 'business_access': True},
            {'id': 'bob', 'project': 'b', 'role': 'operator'},
            {'id': 'agent', 'project': 'a', 'role': 'agent', 'bundles': []}]}
        for entry in self.config['principals']:
            entry['token_sha256'] = hashlib.sha256(self.tokens[entry['id']].encode()).hexdigest()
        self.save()
        self.store = ContextStore(self.directory.name)

    def save(self): private_write(self.policy.path, json.dumps(self.config))
    def who(self, name): return self.policy.authenticate('Bearer ' + self.tokens[name])

    def test_project_scope_and_agent_version_grant(self):
        with self.policy.bind(self.who('alice')):
            resource = self.store.resource('A资料', 'knowledge', 'v1')
            bundle = self.store.bundle('A配置', [{'id': resource['id']}])
            self.store.resource('A资料', 'knowledge', 'v2', resource_id=resource['id'], expected_version=1)
        with self.policy.bind(self.who('bob')):
            self.assertEqual(self.store.catalog()['resources'], [])
            with self.assertRaises(AccessDenied): self.store.get('resources', resource['id'])
            own = self.store.resource('B资料', 'knowledge', 'b')
            with self.assertRaises(AccessDenied):
                self.store.bundle('劫持', [{'id': own['id']}], bundle_id=bundle['id'], expected_version=1)
        self.config['principals'][2]['bundles'] = [bundle['id']];self.save()
        with self.policy.bind(self.who('agent')):
            self.assertEqual(self.store.get('resources', resource['id'], 1)['content'], 'v1')
            with self.assertRaises(AccessDenied): self.store.get('resources', resource['id'], 2)
            with self.assertRaises(AccessDenied): self.store.resource('禁止', 'knowledge', 'x')

    def test_revocation_rechecked_and_delegate_cannot_login(self):
        who = self.who('alice')
        with self.policy.bind(who): headers = self.policy.delegated_headers()
        self.assertEqual(self.policy.authenticate(headers['Authorization'], delegation=True).id, 'alice')
        with self.assertRaises(AccessDenied): self.policy.authenticate(headers['Authorization'])
        self.config['principals'][0]['enabled'] = False;self.save()
        with self.assertRaises(AccessDenied): self.policy.revalidate(who)
        with self.assertRaises(AccessDenied): self.policy.authenticate(headers['Authorization'], delegation=True)

    def test_history_read_and_clear_are_identity_and_session_scoped(self):
        history = HistoryStore(self.directory.name)
        with self.policy.bind(self.who('alice')), history_session('same'):
            history.append({'user': 'A私有'});self.assertEqual(len(history.read()), 1)
        with self.policy.bind(self.who('bob')), history_session('same'):
            self.assertEqual(history.read(), []);history.append({'user': 'B私有'});history.clear()
        with self.policy.bind(self.who('alice')), history_session('same'):
            self.assertEqual(history.read()[0]['user'], 'A私有')
        with self.policy.bind(self.who('alice')), history_session('other'):
            self.assertEqual(history.read(), [])

    def test_project_records_and_upload_references_are_not_transferable(self):
        import gradio as gr
        from src.search_engine.native_runtime import NativeRuntime
        from src.search_engine.workbench_runs import RunStore
        with gr.Blocks() as blocks:
            with gr.Tab('操作'):
                out=gr.JSON();gr.Button('执行').click(lambda:{'ok':True},[],out)
        runtime=NativeRuntime(lambda:blocks);self.addCleanup(runtime.close)
        store=RunStore(runtime,self.directory.name);self.addCleanup(store.close)
        with self.policy.bind(self.who('alice')):
            file=runtime.store_upload('private.txt',io.BytesIO(b'private'),7)
            actual=runtime.prepare(file)
            run=store.start('same',0,{});list(store.events(run['run_id']))
            self.assertEqual(len(store.catalog('操作')['runs']),1)
        with self.policy.bind(self.who('bob')):
            self.assertEqual(store.catalog('操作')['runs'],[])
            with self.assertRaises(AccessDenied): store.get('runs',run['run_id'])
            with self.assertRaises(AccessDenied):
                store.put('runs',{'id':run['run_id'],'project':'b'})
            with self.assertRaises(AccessDenied): runtime.prepare(file)
            with self.assertRaises(AccessDenied): runtime.prepare(actual)

    def test_notebook_and_audit_rollback_together_and_old_grant_is_revoked(self):
        with self.policy.bind(self.who('alice')):
            resource = self.store.resource('笔记', 'notebook', 'v1', permission='read_write')
            bundle = self.store.bundle('配置', [{'id': resource['id'], 'permission': 'read_write'}])
            with self.store.db() as db:
                db.execute("CREATE TRIGGER audit_failure BEFORE INSERT ON audit BEGIN SELECT RAISE(ABORT, 'test failure'); END")
            with self.assertRaises(sqlite3.IntegrityError): self.store.write_notebook(bundle['id'], resource['id'], 1, 'v2', 's')
            self.assertEqual(self.store.get('resources', resource['id'])['version'], 1)
            with self.store.db() as db: db.execute('DROP TRIGGER audit_failure')
            self.store.bundle('配置', [{'id': resource['id'], 'permission': 'read_only'}], bundle_id=bundle['id'], expected_version=1)
            with self.assertRaises(ValueError): self.store.write_notebook(bundle['id'], resource['id'], 1, '越权', 's', bundle_version=1)

    def test_mcp_actual_http_authentication_and_project_scope(self):
        import socket
        import uvicorn
        from fastmcp import FastMCP, Client
        from fastmcp.client.transports import StreamableHttpTransport
        from src.search_engine.workbench_access import configure_mcp, mcp_auth
        with self.policy.bind(self.who('alice')):
            resource = self.store.resource('A资料', 'knowledge', 'A内容')
            bundle = self.store.bundle('A配置', [{'id': resource['id']}])
        self.config['principals'][2]['bundles'] = [bundle['id']];self.save()
        mcp = FastMCP('access-regression', auth=mcp_auth(self.policy), stateless_http=True)
        register_context_mcp(mcp, self.store)
        @mcp.tool(name='retrieve')
        def retrieve(): return {'shared_business': True}
        configure_mcp(mcp, self.policy)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0));port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(mcp.http_app(), host='127.0.0.1', port=port, log_level='error'))
        thread = threading.Thread(target=server.run, daemon=True);thread.start()
        for _ in range(100):
            if server.started: break
            time.sleep(.02)
        self.assertTrue(server.started)
        async def check():
            url = f'http://127.0.0.1:{port}/mcp'
            # Initialization can be public in FastMCP; application requests must fail closed.
            denied = False
            try:
                async with Client(url) as client:
                    await client.read_resource('context://catalog')
            except Exception:
                denied = True
            self.assertTrue(denied)
            async with Client(StreamableHttpTransport(url, headers={'Authorization': 'Bearer ' + self.tokens['bob']})) as client:
                catalog = await client.read_resource('context://catalog')
                self.assertNotIn(resource['id'], catalog[0].text)
                with self.assertRaises(Exception):
                    await client.read_resource(f"context://resources/{resource['id']}/1")
                with self.assertRaises(Exception): await client.call_tool('retrieve', {})
            async with Client(StreamableHttpTransport(url, headers={'Authorization': 'Bearer ' + self.tokens['agent']})) as client:
                material = await client.call_tool('load_context', {'bundle_id': bundle['id']})
                self.assertEqual(material.data['resources'][0]['content'], 'A内容')
                with self.assertRaises(Exception): await client.call_tool('retrieve', {})
        try: asyncio.run(check())
        finally: server.should_exit = True;thread.join(5)

    def test_queued_identity_is_rechecked_after_revocation(self):
        import gradio as gr
        from src.search_engine.native_runtime import NativeRuntime
        from src.search_engine.workbench_runs import RunStore
        calls = []
        with gr.Blocks() as blocks:
            with gr.Tab('操作'):
                out = gr.JSON()
                gr.Button('执行').click(lambda: calls.append(True) or {'ok': True}, [], out)
        runtime = NativeRuntime(lambda: blocks);self.addCleanup(runtime.close)
        store = RunStore(runtime, self.directory.name);self.addCleanup(store.close)
        entered, release = threading.Event(), threading.Event()
        def hold(): entered.set();release.wait(5)
        store.enqueue(('module', '操作'), hold);self.assertTrue(entered.wait(2))
        with self.policy.bind(self.who('alice')):
            run = store.start('same', 0, {})
        self.config['principals'][0]['enabled'] = False;self.save()
        release.set()
        # Inspect as trusted internal bookkeeping, retaining the original project scope.
        with self.policy.bind(self.policy.from_entry(self.config['principals'][0])):
            events = list(store.events(run['run_id']))
        self.assertEqual(events[-1]['status'], 'failed')
        self.assertEqual(calls, [])

    def test_another_operator_approves_without_taking_execution_identity(self):
        import gradio as gr
        from src.search_engine.workbench_access import identity
        from src.search_engine.native_runtime import NativeRuntime
        from src.search_engine.workbench_runs import RunStore
        self.tokens['charlie'] = 'd'*32
        self.config['principals'].append({'id':'charlie','project':'a','role':'operator','business_access':True,
            'token_sha256':hashlib.sha256(self.tokens['charlie'].encode()).hexdigest()});self.save()
        with gr.Blocks() as blocks:
            with gr.Tab('操作'):
                out=gr.JSON();gr.Button('执行').click(lambda:{'who':identity().id},[],out)
        runtime=NativeRuntime(lambda:blocks);self.addCleanup(runtime.close)
        store=RunStore(runtime,self.directory.name);self.addCleanup(store.close)
        with self.policy.bind(self.who('alice')):
            first=store.start('same',0,{});list(store.events(first['run_id']))
            task=store.save_task(first['run_id'],'放行任务','核对执行身份',requires_review=True)
            held=store.start('same',0,{},task_id=task['id'])
        with self.policy.bind(self.who('charlie')):
            store.decide(held['run_id'],'approve');list(store.events(held['run_id']))
            result=store.get('runs',held['run_id'])
        self.assertEqual(result['outputs'][str(out._id)]['value']['who'],'alice')
        self.assertEqual(result['reviews'][0]['actor'],'charlie')

    def test_bootstrap_installation_is_atomic_under_concurrent_start(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=4) as pool:
                policies=list(pool.map(lambda _:AccessPolicy(directory),range(4)))
            token=(Path(directory)/'bootstrap-access.txt').read_text().strip()
            self.assertTrue(all(policy.authenticate('Bearer '+token).id=='local-owner' for policy in policies))
            self.assertEqual(len({policy.key for policy in policies}),1)


if __name__ == '__main__': unittest.main()
