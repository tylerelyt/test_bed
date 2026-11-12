"""Console controls, separate from original callback transport and MCP business tools."""
import importlib.metadata
import json
import os
import socket
from urllib.parse import urlparse
from .workbench_context import ContextStore


class WorkbenchControls:
    def __init__(self, records):
        self.records = records
        self.context = records.context

    def get(self, path, query):
        for table in ('tasks', 'notes'):
            prefix = '/api/native/' + table + '/'
            if path.startswith(prefix):
                record = self.records.get(table, path[len(prefix):])
                version = int(query['version'][0]) if query.get('version') else record.get('version', 1)
                if version == record.get('version', 1):
                    return record
                previous = next((item for item in record.get('revisions', []) if item.get('version', 1) == version), None)
                if previous is None:
                    raise ValueError('该版本不存在')
                return previous
        if path == '/api/native/context':
            return self.context.catalog(query.get('q', [''])[0])
        for table in ('resources', 'bundles'):
            prefix = '/api/native/context/' + table + '/'
            if path.startswith(prefix):
                version = query.get('version', [None])[0]
                return self.context.get(table, path[len(prefix):], int(version) if version else None)
        if path == '/api/native/evaluations':
            with self.records.db() as db:
                tasks = [json.loads(row[0]) for row in db.execute('SELECT body FROM tasks ORDER BY rowid DESC LIMIT 200')]
            return {'tasks': tasks}
        if path.startswith('/api/native/evaluations/'):
            return self.records.evaluations.summary(path.rsplit('/', 1)[-1])
        if path == '/api/native/diagnostics':
            services = []
            graph = urlparse(os.environ.get('JANUSGRAPH_URL') or os.environ.get('GREMLIN_SERVER_URL') or 'ws://127.0.0.1:8182/gremlin')
            for name, address, port in [('MCP', '127.0.0.1', 3001), ('图像服务', '127.0.0.1', 5001),
                                         ('图服务', graph.hostname, graph.port or 8182)]:
                try:
                    with socket.create_connection((address, port), timeout=.4):
                        reachable = True
                except OSError:
                    reachable = False
                services.append({'name': name, 'reachable': reachable, 'check': '端口连通性；不代表模型或业务健康'})
            packages = {}
            for name in ('gradio', 'fastmcp', 'torch'):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            return {'services': services, 'packages': packages, 'model_key_configured': bool(os.environ.get('DASHSCOPE_API_KEY')),
                    'judge_configured': bool(os.environ.get('WORKBENCH_JUDGE_URL') and os.environ.get('WORKBENCH_JUDGE_MODEL')),
                    'context_bundles': len(self.context.catalog()['bundles']),
                    'scheduler': {'last_tick': self.records.evaluations.last_tick, 'error': self.records.evaluations.last_error},
                    'note': '不加载模型、不下载权重、不发起付费调用。'}
        return None

    def post(self, path, data):
        action = data.get('action')
        records, evaluations = self.records, self.records.evaluations
        if path == '/api/native/runs/control':
            return records.decide(data.get('run_id'), action)
        if path == '/api/native/tasks':
            if action == 'save':
                return records.save_task(data.get('run_id'), data.get('name'), data.get('expectation'), data.get('rules'), data.get('requires_review', False))
            if action == 'rules':
                return records.update_task(data.get('task_id'), data.get('rules'), data.get('requires_review', False), data.get('expected_version'), data.get('secret_env'), data.get('context_binding'), data.get('case_kind'))
            if action == 'evaluate':
                return evaluations.evaluate(data.get('task_id'), data.get('run_id'))
            if action == 'review':
                return evaluations.review(data.get('run_id'), data.get('passed'), data.get('reason'))
        if path == '/api/native/notes' and action == 'update':
            return records.update_note(data.get('note_id'), data.get('expected_version'), data.get('body'), data.get('status'))
        if path == '/api/native/notes' and action == 'consolidate':
            return records.consolidate(data.get('title'), data.get('run_ids'), data.get('use_model', False))
        if path == '/api/native/evaluations':
            if action == 'audit':
                return evaluations.audit(data.get('suite_id'))
            if action == 'save':
                return evaluations.save_suite(data.get('module'), data.get('name'), data.get('task_ids'), data.get('variants'), data.get('repeats', 1), data.get('suite_id'), data.get('expected_version'))
            if action == 'start':
                return evaluations.start(data.get('suite_id'), data.get('mode', 'full'), data.get('request_id'))
            if action == 'baseline':
                return evaluations.baseline(data.get('suite_id'), data.get('batch_id'))
            if action == 'compare':
                return evaluations.compare(data.get('suite_id'), data.get('batch_id'))
            if action == 'cancel':
                batch = evaluations.summary(data.get('batch_id'))
                for row in batch['rows']:
                    if row['status'] in ('queued', 'running', 'awaiting_review'):
                        records.decide(row['run_id'], 'cancel')
                return evaluations.summary(batch['id'])
            raise ValueError('评测操作无效')
        if path == '/api/native/schedules':
            return evaluations.schedule(data.get('suite_id'), data.get('interval_minutes'), data.get('enabled', True), data.get('schedule_id'))
        if path == '/api/native/context':
            if action == 'resource':
                return self.context.resource(data.get('name'), data.get('kind'), data.get('content'), data.get('source', ''),
                                             data.get('permission', 'read_only'), data.get('tags'), data.get('resource_id'), data.get('expected_version'))
            if action == 'bundle':
                return self.context.bundle(data.get('name'), data.get('resources'), data.get('max_chars', 12000), data.get('bundle_id'), data.get('expected_version'))
            if action == 'assemble':
                return self.context.assemble(data.get('bundle_id'), data.get('version'), data.get('resource_ids'), data.get('query', ''))
            if action == 'notebook':
                return self.context.write_notebook(data.get('bundle_id'), data.get('resource_id'), data.get('expected_version'),
                                                   data.get('content'), data.get('session'), data.get('bundle_version'))
            if action == 'from_note':
                note = records.get('notes', data.get('note_id'))
                if note.get('status') != 'approved':
                    raise ValueError('请先审阅并采用经验，再加入上下文')
                return self.context.resource(note['title'], 'experience', note['body'],
                    f"run:{note['source_run']} note:{note['id']} version:{note.get('version', 1)}", tags=[note['module']])
            raise ValueError('上下文操作无效')
        return None
