"""Execute the original workbench callbacks in-process; expose its layout to Next.js."""
import asyncio
import atexit
import base64
import io
import copy
import json
import math
import os
from .workbench_secrets import private_field, public_value
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import threading
import uuid
from pathlib import Path
import tempfile
import time
from .workbench_access import session_key, history_session, identity, check, require_business


class NativeRuntime:
    def __init__(self, loader=None):
        self.loader = loader
        self.blocks = None
        self.lock = threading.RLock()
        self.sessions = {}
        self.fn_locks = {}
        self.assets = {}
        self.uploads = tempfile.TemporaryDirectory(prefix='test-bed-native-')
        self.asset_tokens = {}
        self.asset_projects = {}
        self.credential_values = set()
        self.allowed_output_files = set()
        self.shared_files = set()
        self.session_pins = {}
        self.resource_pins = {}
        self.upload_records = {}
        self.reserved_upload_bytes = 0
        self.reserved_upload_count = 0
        self.next_cleanup_at = 0
        self.instance_id = uuid.uuid4().hex

    def close(self):
        self.uploads.cleanup()
        self.sessions.clear()
        self.assets.clear()
        self.asset_tokens.clear()
        self.asset_projects.clear()
        self.credential_values.clear()
        self.allowed_output_files.clear()

    def release_session(self, session_id):
        key = session_key(session_id)
        with self.lock:
            if not self.session_pins.get(key):
                self.sessions.pop(key, None)

    def tokens_in(self, value):
        found = set()
        def walk(item):
            if isinstance(item, dict):
                for child in item.values(): walk(child)
            elif isinstance(item, (list, tuple)):
                for child in item: walk(child)
            elif isinstance(item, str):
                if item in self.asset_tokens: found.add(self.asset_tokens[item])
                if item.startswith('/api/native/assets/'): found.add(item.rsplit('/', 1)[-1])
        walk(value)
        return found

    def pin(self, session_id, values):
        self.session(session_id)
        key = session_key(session_id)
        tokens = self.tokens_in(values)
        with self.lock:
            self.session_pins[key] = self.session_pins.get(key, 0) + 1
            for token in tokens: self.resource_pins[token] = self.resource_pins.get(token, 0) + 1
        return key, tokens

    def unpin(self, lease, discard=False):
        key, tokens = lease
        with self.lock:
            self.session_pins[key] = max(0, self.session_pins.get(key, 0) - 1)
            if not self.session_pins[key]:
                self.session_pins.pop(key, None)
                if discard: self.sessions.pop(key, None)
            for token in tokens:
                self.resource_pins[token] = max(0, self.resource_pins.get(token, 0) - 1)
                if not self.resource_pins[token]: self.resource_pins.pop(token, None)

    def cleanup_resources(self, now=None):
        explicit = now is not None
        now = time.monotonic() if now is None else now
        if not explicit and now < self.next_cleanup_at: return
        self.next_cleanup_at = now + 5
        ttl = max(.01, float(os.environ.get('WORKBENCH_SESSION_TTL_SECONDS', '3600')))
        file_ttl = max(.01, float(os.environ.get('WORKBENCH_UPLOAD_TTL_SECONDS', '3600')))
        with self.lock:
            for key, session in list(self.sessions.items()):
                if not self.session_pins.get(key) and now - session.get('last_used', now) > ttl:
                    self.sessions.pop(key)
            referenced = set().union(*(self.tokens_in(s['values']) for s in self.sessions.values())) if self.sessions else set()
            for token, item in list(self.upload_records.items()):
                if token not in referenced and not self.resource_pins.get(token) and now - item['at'] > file_ttl:
                    path = item['path']
                    path.unlink(missing_ok=True)
                    try: path.parent.rmdir()
                    except OSError: pass
                    self.upload_records.pop(token, None)
                    self.assets.pop(token, None)
                    self.asset_projects.pop(token, None)
                    self.asset_tokens.pop(str(path), None)

    def initialize(self):
        with self.lock:
            if self.blocks is None:
                if self.loader:
                    self.blocks = self.loader()
                else:
                    from .portal import SearchUI
                    self.blocks = SearchUI().interface
            return self.blocks

    def export(self, value):
        if hasattr(value, 'model_dump'):
            return self.export(value.model_dump())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, 'columns') and hasattr(value, 'to_numpy'):
            return self.export({'headers': list(value.columns), 'data': value.astype(object).where(value.notna(), None).to_numpy().tolist()})
        if isinstance(value, dict):
            result = {key: self.export(v) for key, v in value.items()}
            path = value.get('path')
            if isinstance(path, str) and (path in self.allowed_output_files or path in self.asset_tokens) and Path(path).is_file():
                scope = 'shared-business' if path in self.shared_files else identity().project
                token = self.asset_tokens.get(path)
                if token is None or self.asset_projects.get(token, 'local') != scope:
                    token = uuid.uuid4().hex
                    self.asset_tokens[path] = token
                self.assets[token] = Path(path)
                self.asset_projects[token] = scope
                result['url'] = '/api/native/assets/' + token
                result['path'] = result['url']
            return result
        if isinstance(value, (list, tuple)):
            return [self.export(v) for v in value]
        if callable(value):
            return None
        if hasattr(value, 'tolist'):
            return self.export(value.tolist())
        return value

    def session(self, session_id):
        from gradio.state_holder import SessionState
        blocks = self.initialize()
        session_id = session_key(session_id)
        self.cleanup_resources()
        with self.lock:
            if session_id not in self.sessions:
                maximum = max(1, int(os.environ.get('WORKBENCH_MAX_SESSIONS', '256')))
                if len(self.sessions) >= maximum:
                    candidates = [key for key in self.sessions if not self.session_pins.get(key)]
                    if not candidates: raise ValueError('活动会话已达容量上限，请稍后重试')
                    self.sessions.pop(min(candidates, key=lambda key:self.sessions[key]['last_used']))
                config = blocks.config
                for component in config['components']:
                    if component['type'] in ('file', 'image', 'audio', 'video', 'gallery'):
                        self.allow_files(component.get('props', {}).get('value'), shared=True)
                self.sessions[session_id] = {'state': SessionState(blocks),
                    'values': {str(c['id']): self.export(c.get('props', {}).get('value')) for c in config['components'] if c['type'] != 'state'},
                    'props': {}, 'lock': threading.RLock(), 'cancelled': set()}
            self.sessions[session_id]['last_used'] = time.monotonic()
            return self.sessions[session_id]

    def allow_files(self, value, shared=False):
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and Path(value['path']).is_file():
                self.allowed_output_files.add(value['path'])
                if shared: self.shared_files.add(value['path'])
            for item in value.values():
                self.allow_files(item, shared=shared)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.allow_files(item, shared=shared)

    def layout(self, session_id):
        session = self.session(session_id)
        config = self.blocks.config
        secrets = self.capture_credentials(session)
        components = []
        for component in config['components']:
            item = copy.deepcopy(component)
            cid = str(item['id'])
            if item['type'] == 'state':
                item['props'] = {}
            else:
                item['props'].update(session['props'].get(cid, {}))
                private = private_field(item['props'])
                item['props']['value'] = None if private else public_value(self.export(session['values'].get(cid)), secrets)
            components.append(item)
        return self.export({'components': components, 'layout': config['layout'],
                            'dependencies': config['dependencies'], 'session': session_id, 'runtime_id': self.instance_id})

    def capture_credentials(self, session):
        with self.lock:
            self.credential_values.update(value for component in self.blocks.config['components']
                if private_field({**component['props'], **session['props'].get(str(component['id']), {})})
                and isinstance(value := session['values'].get(str(component['id'])), str) and value)
            return tuple(self.credential_values)

    def public(self, value, session):
        secrets = self.capture_credentials(session)
        result = self.export(value)
        components = {str(component['id']): component for component in self.blocks.config['components']}
        for cid, entry in result.get('updates', {}).items():
            if not isinstance(entry, dict):
                continue
            props = {**components.get(cid, {}).get('props', {}), **session['props'].get(cid, {})}
            if 'value' in entry:
                entry['value'] = None if private_field(props) else public_value(entry['value'], secrets)
            if 'props' in entry:
                for key in ('value', 'choices', 'samples', 'info'):
                    if key in entry['props']:
                        entry['props'][key] = None if key == 'value' and private_field(props) else public_value(entry['props'][key], secrets)
        if 'error' in result:
            result['error'] = public_value(result['error'], secrets)
        return result

    def store_upload(self, name, stream, size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('文件长度无效')
        limit = max(1, int(os.environ.get('WORKBENCH_UPLOAD_MAX_MB', '64'))) * 1024 * 1024
        quota = max(1, int(os.environ.get('WORKBENCH_UPLOAD_QUOTA_MB', '512'))) * 1024 * 1024
        self.cleanup_resources(now=time.monotonic())
        with self.lock:
            if len(self.upload_records) + self.reserved_upload_count >= int(os.environ.get('WORKBENCH_MAX_UPLOADS', '1000')):
                raise ValueError('临时文件数量已达上限')
            used = sum(item['size'] for item in self.upload_records.values())
            if size > limit: raise ValueError('文件超过当前单文件上传限制')
            if used + self.reserved_upload_bytes + size > quota: raise ValueError('临时上传空间不足，请等待过期文件清理')
            self.reserved_upload_bytes += size
            self.reserved_upload_count += 1
        folder = Path(self.uploads.name) / uuid.uuid4().hex
        path = folder / (Path(name).name or 'upload')
        try:
            folder.mkdir(mode=0o700)
            with path.open('wb') as target:
                path.chmod(0o600)
                remaining = size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk: raise ValueError('文件上传中断，请重新上传')
                    target.write(chunk)
                    remaining -= len(chunk)
            token = uuid.uuid4().hex
            with self.lock:
                self.assets[token] = path
                self.asset_projects[token] = identity().project
                self.asset_tokens[str(path)] = token
                self.upload_records[token] = {'path': path, 'size': size, 'at': time.monotonic()}
            return {'path': '/api/native/assets/' + token, 'url': '/api/native/assets/' + token,
                    'orig_name': name, 'meta': {'_type': 'gradio.FileData'}}
        except BaseException:
            path.unlink(missing_ok=True)
            if folder.exists(): folder.rmdir()
            raise
        finally:
            with self.lock:
                self.reserved_upload_bytes -= size
                self.reserved_upload_count -= 1

    def authorize_asset(self, token):
        scope = self.asset_projects.get(token, 'local')
        if scope == 'shared-business': require_business()
        else: check({'project': scope})

    def prepare(self, value):
        if isinstance(value, dict):
            if '__upload' in value:
                upload = value['__upload']
                payload = base64.b64decode(upload['base64'], validate=True)
                return self.prepare(self.store_upload(upload['name'], io.BytesIO(payload), len(payload)))
            if isinstance(value.get('path'), str):
                if value['path'] in self.asset_tokens:
                    self.authorize_asset(self.asset_tokens[value['path']])
                    return value
                token = value['path'].rsplit('/', 1)[-1]
                if token in self.assets:
                    self.authorize_asset(token)
                    return {**value, 'path': str(self.assets[token]), 'url': None}
                if value.get('meta', {}).get('_type') == 'gradio.FileData':
                    raise ValueError('文件引用已过期，请重新上传')
            return {k: self.prepare(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.prepare(v) for v in value]
        return value

    def execute(self, session_id, fn_id, values, event=None, emit=None):
        session = self.session(session_id)
        config = self.blocks.config
        dependencies = {d['id']: d for d in config['dependencies']}
        if fn_id not in dependencies:
            raise ValueError('操作不存在')
        dep = dependencies[fn_id]
        for cancelled in dep.get('cancels', []):
            session['cancelled'].add(cancelled)
        def send(item):
            if emit:
                emit(self.public(item, session))
        async def run():
            import gradio as gr
            updates = {}
            trace = []
            async def invoke(index, selection=None, visited=None):
                visited = set() if visited is None else visited
                if index in visited:
                    return
                visited.add(index)
                dependency = dependencies[index]
                if not dependency.get('backend_fn', True):
                    return
                fn = self.blocks.fns[index]
                session['cancelled'].discard(index)
                inputs = []
                for component in fn.inputs:
                    value = None if getattr(component, 'stateful', False) else self.prepare(session['values'].get(str(component._id)))
                    if component.get_block_name() == 'textbox' and value is None:
                        value = ''
                    inputs.append(value)
                event_object = None
                if selection is not None:
                    kind = next((target[1] for target in dependency.get('targets', []) if target[1] == 'select'), None)
                    event_object = gr.SelectData(None, selection) if kind else gr.EventData(None, selection)
                iterator = None
                send({'type': 'started', 'fn_id': index})
                while True:
                    response = await self.blocks.process_api(index, inputs, state=session['state'],
                        iterator=iterator, session_hash=session_id, event_data=event_object,
                        explicit_call=True, simple_format=True)
                    delta = {}
                    for component, output in zip(fn.outputs, response['data']):
                        if getattr(component, 'stateful', False):
                            continue
                        cid = str(component._id)
                        if component.get_block_name() in ('file', 'image', 'audio', 'video', 'gallery'):
                            self.allow_files(output)
                        if isinstance(output, dict) and output.get('__type__') == 'update':
                            props = {k: v for k, v in output.items() if k != '__type__'}
                            session['props'].setdefault(cid, {}).update(props)
                            if 'value' in props:
                                session['values'][cid] = props['value']
                            delta[cid] = {'props': props}
                        else:
                            session['values'][cid] = output
                            delta[cid] = {'value': output}
                    updates.update(delta)
                    send({'type': 'update', 'fn_id': index, 'updates': delta})
                    iterator = response.get('iterator')
                    if not response.get('is_generating') or index in session['cancelled']:
                        if iterator and hasattr(iterator, 'aclose'):
                            await iterator.aclose()
                        break
                trace.append(index)
                for changed in response.get('changed_state_ids', []):
                    for observer in config['dependencies']:
                        if any(t[0] == changed and t[1] == 'change' for t in observer.get('targets', [])):
                            await invoke(observer['id'], visited=visited)
                for child in config['dependencies']:
                    if child.get('trigger_after') == index:
                        await invoke(child['id'], visited=visited)
            for cid, value in values.items():
                component = self.blocks.blocks.get(int(cid))
                if component is None or getattr(component, 'stateful', False):
                    continue
                prepared = self.prepare(value)
                if component.get_block_name() == 'dataframe' and isinstance(prepared, list):
                    prepared = {'headers': component.headers, 'data': prepared}
                session['values'][str(cid)] = prepared
            await invoke(fn_id, event)
            result = {'type': 'completed', 'updates': updates, 'trace': trace}
            send(result)
            return self.public(result, session)
        with self.lock:
            fn_lock = self.fn_locks.setdefault(fn_id, threading.RLock())
        with fn_lock, history_session(session_id):
            return asyncio.run(run())


runtime = NativeRuntime()
atexit.register(runtime.close)
