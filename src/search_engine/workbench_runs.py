"""Local, durable evidence for native callbacks; business execution stays in NativeRuntime."""
import hashlib
import inspect
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()[:12]


SECRET = re.compile(r'password|secret|credential|api[_ -]?key|access[_ -]?token|密码|密钥|令牌', re.I)
HIDDEN = '【已隐藏】'


class RunStore:
    def __init__(self, executor, directory=None):
        self.executor = executor
        self.directory = Path(directory or os.environ.get('WORKBENCH_STATE_DIR') or
                              Path(__file__).resolve().parents[2] / '.workbench')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / 'native-runs-v1.sqlite3'
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.live = {}
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (run_id TEXT, seq INTEGER, body TEXT NOT NULL,
                    PRIMARY KEY(run_id, seq));
                CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            ''')
            for row in db.execute('SELECT id, body FROM runs').fetchall():
                run = json.loads(row[1])
                if run['status'] == 'running':
                    run.update(status='interrupted', finished_at=time.time(),
                               error='执行服务已重启；操作未自动重跑，请核对实际业务状态。')
                    db.execute('UPDATE runs SET body=? WHERE id=?', (encoded(run), row[0]))
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, table, item_id):
        if table not in ('runs', 'tasks', 'notes'):
            raise ValueError('记录类型无效')
        if not isinstance(item_id, str) or not item_id or len(item_id) > 120:
            raise ValueError('记录标识无效')
        with self.db() as db:
            row = db.execute(f'SELECT body FROM {table} WHERE id=?', (item_id,)).fetchone()
        if not row:
            raise ValueError('记录不存在')
        return json.loads(row[0])

    def put(self, table, item):
        if table not in ('runs', 'tasks', 'notes'):
            raise ValueError('记录类型无效')
        with self.db() as db:
            db.execute(f'INSERT OR REPLACE INTO {table} VALUES (?, ?)', (item['id'], encoded(item)))

    def listing(self, table, module):
        with self.db() as db:
            rows = db.execute(f'SELECT body FROM {table} ORDER BY rowid DESC').fetchall()
        return [item for row in rows if (item := json.loads(row[0]))['module'] == module][:100]

    def catalog(self, module):
        runs = self.listing('runs', module)
        return {'runs': [{k: v for k, v in r.items() if k not in ('inputs', 'outputs', 'event')}
                         for r in runs], 'tasks': self.listing('tasks', module),
                'notes': self.listing('notes', module)}

    def descriptor(self, fn_id):
        blocks = self.executor.initialize()
        config = blocks.config
        dependency = next((d for d in config['dependencies'] if d['id'] == fn_id), None)
        if dependency is None or fn_id not in blocks.fns:
            raise ValueError('操作不存在')
        components = {c['id']: c for c in config['components']}
        target = next((t[0] for t in dependency['targets'] if t[0] is not None), None)
        def ancestors(node, trail):
            if node['id'] == target:
                return trail + [node['id']]
            for child in node.get('children', []):
                found = ancestors(child, trail + [node['id']])
                if found:
                    return found
            return []
        tabs = [components[c]['props'].get('label', '') for c in ancestors(config['layout'], [])
                if c in components and components[c]['type'] in ('tab', 'tabitem')]
        module = tabs[0] if tabs else '工作台'
        target_props = components.get(target, {}).get('props', {})
        label = target_props.get('value') or target_props.get('label') or dependency.get('api_name') or '执行'
        fn = blocks.fns[fn_id].fn
        try:
            source = inspect.getsource(fn)
        except (TypeError, OSError):
            source = repr(getattr(fn, '__code__', None))
        schema = digest({'components': [(c['id'], c['type'], c['props'].get('label'),
                                         c['props'].get('type')) for c in config['components']],
                         'inputs': dependency['inputs'], 'outputs': dependency['outputs'],
                         'source': source, 'fn_id': fn_id})
        return {'module': module, 'action': str(label), 'fn_id': fn_id, 'schema': schema,
                'input_ids': dependency['inputs']}

    def sanitizer(self, session, submitted):
        components = self.executor.blocks.config['components']
        sensitive = {str(c['id']) for c in components if c['props'].get('type') == 'password' or
                     SECRET.search(str(c['props'].get('label', '')))}
        values = {**self.executor.session(session)['values'], **submitted}
        secrets = [str(v) for k, v in values.items() if k in sensitive and isinstance(v, str) and v]
        def clean(value, key=''):
            if str(key) in sensitive or SECRET.search(str(key)):
                return HIDDEN
            if isinstance(value, dict):
                if value.get('meta', {}).get('_type') == 'gradio.FileData' or any(
                        isinstance(value.get(k), str) and '/api/native/assets/' in value[k] for k in ('path', 'url')):
                    return {'file': value.get('orig_name', '文件'), 'unavailable': True}
                return {k: clean(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, str):
                for secret in secrets:
                    value = value.replace(secret, HIDDEN)
                value = re.sub(r'/api/native/assets/[a-f0-9]+', '【临时文件未归档】', value)
            return value
        return clean

    def start(self, session, fn_id, submitted, event=None, task_id=None, request_id=None):
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch('[a-f0-9]{32}', request_id)):
            raise ValueError('运行请求标识无效')
        descriptor = self.descriptor(fn_id)
        task = self.get('tasks', task_id) if task_id else None
        if task and (task['schema'] != descriptor['schema'] or task['fn_id'] != fn_id):
            raise ValueError('任务的操作版本已变化，请重新保存任务')
        clean = self.sanitizer(session, submitted)
        actual = self.executor.session(session)['values']
        inputs = {str(cid): submitted.get(str(cid), actual.get(str(cid))) for cid in descriptor.pop('input_ids')
                  if not getattr(self.executor.blocks.blocks[cid], 'stateful', False)}
        snapshot = clean(self.executor.export(inputs))
        run = {**descriptor, 'id': request_id or uuid.uuid4().hex, 'session': session, 'status': 'running',
               'started_at': time.time(), 'finished_at': None, 'inputs': snapshot,
               'event': clean(event), 'outputs': {}, 'task_id': task_id,
               'config_version': digest({'schema': descriptor['schema'], 'inputs': snapshot})}
        with self.lock:
            with self.db() as db:
                existing = db.execute('SELECT body FROM runs WHERE id=?', (run['id'],)).fetchone()
            if existing:
                existing = json.loads(existing[0])
                if existing['session'] != session or existing['fn_id'] != fn_id:
                    raise ValueError('运行请求标识已被其他操作使用')
                return {'run_id': existing['id'], 'config_version': existing['config_version']}
            self.put('runs', run)
            self.live[run['id']] = {}
        # Capture values now: later edits in the browser must not change a queued callback's inputs.
        execution_values = dict(inputs)
        def work():
            def emit(item):
                live = self.executor.export(item)
                safe = clean(live)
                with self.condition:
                    run['outputs'].update(safe.get('updates', {}))
                    with self.db() as db:
                        seq = db.execute('SELECT COALESCE(MAX(seq), 0)+1 FROM events WHERE run_id=?',
                                         (run['id'],)).fetchone()[0]
                        db.execute('INSERT INTO events VALUES (?, ?, ?)', (run['id'], seq, encoded(safe)))
                        db.execute('UPDATE runs SET body=? WHERE id=?', (encoded(run), run['id']))
                    self.live[run['id']][seq] = live
                    # The database holds the complete trace; retain only recent live previews.
                    for old_seq in list(self.live[run['id']])[:-8]:
                        del self.live[run['id']][old_seq]
                    self.condition.notify_all()
            try:
                # Live events use the original values, while durable evidence is redacted.
                self.executor.execute(session, fn_id, execution_values, event, emit=emit)
                run['status'] = 'completed'
            except Exception as exc:
                run.update(status='failed', error=clean(str(exc)))
                emit({'type': 'failed', 'error': str(exc)})
            finally:
                run['finished_at'] = time.time()
                with self.condition:
                    self.put('runs', run)
                    # Only previews need temporary URLs; durable records never contain them.
                    finished = [key for key in self.live if self.get('runs', key)['status'] != 'running']
                    for key in finished[:-20]:
                        del self.live[key]
                    self.condition.notify_all()
        threading.Thread(target=work, daemon=True, name='workbench-' + run['id'][:8]).start()
        return {'run_id': run['id'], 'config_version': run['config_version']}

    def events(self, run_id, after=0):
        self.get('runs', run_id)
        while True:
            with self.condition:
                with self.db() as db:
                    rows = db.execute('SELECT seq, body FROM events WHERE run_id=? AND seq>? ORDER BY seq',
                                      (run_id, after)).fetchall()
                run = self.get('runs', run_id)
                if not rows and run['status'] == 'running':
                    self.condition.wait(10)
            for seq, body in rows:
                after = seq
                yield {**self.live.get(run_id, {}).get(seq, json.loads(body)), 'seq': seq}
            if run['status'] != 'running':
                yield {'type': 'settled', 'status': run['status'], 'error': run.get('error'), 'seq': after}
                return
            if not rows:
                yield {'type': 'heartbeat', 'seq': after}

    def save_task(self, run_id, name, expectation):
        run = self.get('runs', run_id)
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError('任务名称须为 1—120 个字')
        if not isinstance(expectation, str) or not expectation.strip() or len(expectation) > 2000:
            raise ValueError('请填写任务验收要点（2000 字以内）')
        task = {k: run[k] for k in ('module', 'action', 'fn_id', 'schema', 'inputs', 'event', 'config_version')}
        task.update(id=uuid.uuid4().hex, name=name.strip(), expectation=expectation.strip(), source_run=run_id,
                    baseline=run_id if run['status'] == 'completed' else None, created_at=time.time())
        self.put('tasks', task)
        return task

    def restore(self, task_id):
        task = self.get('tasks', task_id)
        if task['schema'] != self.descriptor(task['fn_id'])['schema']:
            raise ValueError('操作版本已变化，请重新保存任务')
        def unavailable(value):
            return value == HIDDEN or (isinstance(value, str) and HIDDEN in value) or (
                isinstance(value, dict) and (value.get('unavailable') or any(unavailable(v) for v in value.values()))) or (
                isinstance(value, list) and any(unavailable(v) for v in value))
        missing = [k for k, v in task['inputs'].items() if unavailable(v)]
        return {'values': {k: v for k, v in task['inputs'].items() if k not in missing},
                'missing': missing, 'task': task}

    def baseline(self, task_id, run_id):
        with self.lock:
            task, run = self.get('tasks', task_id), self.get('runs', run_id)
            if run['status'] != 'completed' or run['schema'] != task['schema'] or run['fn_id'] != task['fn_id']:
                raise ValueError('基线须是相同操作版本的一次已完成运行')
            task['baseline'] = run_id
            self.put('tasks', task)
        return task

    def compare(self, task_id, run_id):
        task, run = self.get('tasks', task_id), self.get('runs', run_id)
        if not task.get('baseline'):
            raise ValueError('任务尚未设置基线')
        baseline = self.get('runs', task['baseline'])
        if baseline['schema'] != run['schema'] or baseline['fn_id'] != run['fn_id']:
            raise ValueError('只能比较相同操作版本的运行')
        def changes(before, after):
            return [{'component': key, 'before': before.get(key), 'after': after.get(key)}
                    for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
        return {'baseline': baseline['id'], 'run': run_id, 'status': run['status'],
                'duration_seconds': [(r['finished_at'] - r['started_at']) if r['finished_at'] else None
                                     for r in (baseline, run)],
                'inputs': changes(baseline['inputs'], run['inputs']),
                'outputs': changes(baseline['outputs'], run['outputs']),
                'event_changed': baseline['event'] != run['event'],
                'notice': '仅比较参数、结果与耗时；运行完成不代表业务成功，结果差异不等于质量评分。'}

    def save_note(self, run_id, title, body):
        run = self.get('runs', run_id)
        if not isinstance(title, str) or not 0 < len(title.strip()) <= 120 or not isinstance(body, str) or not 0 < len(body.strip()) <= 8000:
            raise ValueError('请填写经验标题（120 字以内）与适用条件、做法或结论（8000 字以内）')
        note = {'id': uuid.uuid4().hex, 'module': run['module'], 'source_run': run_id,
                'config_version': run['config_version'], 'title': title.strip(), 'body': body.strip(),
                'created_at': time.time()}
        self.put('notes', note)
        return note
