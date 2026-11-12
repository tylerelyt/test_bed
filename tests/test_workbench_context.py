import asyncio
import json
import tempfile
import threading
import unittest
from fastmcp import FastMCP, Client
from src.search_engine.workbench_context import ContextStore, register_context_mcp


class ContextTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = ContextStore(directory.name)
        self.directory = directory.name
        self.guide = self.store.resource('报销规则', 'instructions', '先核对票据日期，再核对审批记录。', '企业手册', tags=['报销'])
        self.knowledge = self.store.resource('材料', 'knowledge', '发票属于待核对资料。', '内部资料', tags=['票据'])
        self.notebook = self.store.resource('Agent 笔记', 'notebook', '第一次记录', permission='read_write')
        self.bundle = self.store.bundle('报销上下文', [{'id': self.guide['id'], 'required': True},
            {'id': self.knowledge['id']}, {'id': self.notebook['id'], 'permission': 'read_write'}])

    def test_discovery_does_not_load_all_content_and_assembly_is_budgeted(self):
        catalog = self.store.catalog('报销')
        self.assertNotIn('content', catalog['resources'][0])
        found = self.store.discover(self.bundle['id'], query='票据')
        self.assertEqual(len(found['resources']), 1)
        assembled = self.store.assemble(self.bundle['id'], resource_ids=[self.knowledge['id']])
        self.assertEqual({item['id'] for item in assembled['resources']}, {self.guide['id'], self.knowledge['id']})
        self.assertEqual(assembled['used_chars'], sum(len(item['content']) for item in assembled['resources']))
        with self.assertRaises(ValueError): self.store.assemble(self.bundle['id'], resource_ids=['outside'])
        long = self.store.resource('长资料', 'knowledge', 'x' * 120)
        small = self.store.bundle('小预算', [{'id': self.guide['id'], 'required': True}, {'id': long['id']}], 100)
        self.assertEqual(self.store.assemble(small['id'])['omitted'][0]['id'], long['id'])
        required = self.store.bundle('必选超限', [{'id': long['id'], 'required': True}], 100)
        with self.assertRaises(ValueError): self.store.assemble(required['id'])

    def test_versions_are_immutable_and_notebook_is_shared_between_sessions(self):
        updated = self.store.resource('报销规则', 'instructions', '新版规则', '企业手册', resource_id=self.guide['id'], expected_version=1)
        self.assertEqual(updated['version'], 2)
        assembled = self.store.assemble(self.bundle['id'])
        self.assertIn('先核对票据日期', assembled['resources'][0]['content'])
        self.store.write_notebook(self.bundle['id'], self.notebook['id'], 1, '会话 A 的核对结论', 'session-a')
        reopened = ContextStore(self.directory)
        from_b = reopened.assemble(self.bundle['id'])
        note = next(item for item in from_b['resources'] if item['id'] == self.notebook['id'])
        self.assertEqual(note['content'], '会话 A 的核对结论')
        self.assertEqual(note['resolved_version'], 2)
        self.assertEqual(reopened.get('resources', self.notebook['id'], 1)['content'], '第一次记录')

    def test_read_only_permissions_and_write_conflicts_are_enforced(self):
        with self.assertRaises(ValueError): self.store.write_notebook(self.bundle['id'], self.guide['id'], 1, '越权', 'a')
        with self.assertRaises(ValueError): self.store.resource('文档', 'knowledge', '内容', permission='read_write')
        with self.assertRaises(ValueError): self.store.bundle('权限提升', [{'id': self.guide['id'], 'permission': 'read_write'}])
        results = []
        barrier = threading.Barrier(2)
        def write(session):
            barrier.wait()
            try:
                self.store.write_notebook(self.bundle['id'], self.notebook['id'], 1, session, session)
                results.append('written')
            except ValueError:
                results.append('conflict')
        threads = [threading.Thread(target=write, args=(session,)) for session in ('a', 'b')]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertCountEqual(results, ['written', 'conflict'])

    def test_tool_schema_requires_explicit_contract_and_never_executes_code(self):
        with self.assertRaises(ValueError): self.store.resource('坏工具', 'tool', '{}')
        tool = self.store.resource('检索工具', 'tool', json.dumps({'name': 'retrieve', 'description': '原有检索工具', 'input_schema': {'type': 'object'}}))
        bundle = self.store.bundle('工具配置', [{'id': tool['id']}])
        assembled = self.store.assemble(bundle['id'])
        self.assertEqual(json.loads(assembled['resources'][0]['content'])['name'], 'retrieve')

    def test_actual_mcp_resource_discovery_loading_and_notebook_write(self):
        server = FastMCP('上下文供应验收')
        register_context_mcp(server, self.store)
        async def exercise():
            async with Client(server) as client:
                resources = await client.list_resources()
                self.assertIn('context://catalog', [str(item.uri) for item in resources])
                result = await client.call_tool('discover_context', {'bundle_id': self.bundle['id'], 'query': '票据'})
                self.assertEqual(len(result.data['resources']), 1)
                loaded = await client.call_tool('load_context', {'bundle_id': self.bundle['id'], 'resource_ids': [self.guide['id']]})
                self.assertEqual(loaded.data['resources'][0]['content'], self.guide['content'])
                await client.call_tool('write_context_notebook', {'bundle_id': self.bundle['id'], 'resource_id': self.notebook['id'],
                    'expected_version': 1, 'content': '通过 MCP 写入的经验', 'session': 'mcp-session'})
                contents = await client.read_resource(f"context://resources/{self.notebook['id']}/2")
                self.assertEqual(json.loads(contents[0].text)['content'], '通过 MCP 写入的经验')
        asyncio.run(exercise())
