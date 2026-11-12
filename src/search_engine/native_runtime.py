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
        self.credential_values = set()
        self.allowed_output_files = set()
        self.instance_id = uuid.uuid4().hex

    def close(self):
        self.uploads.cleanup()
        self.sessions.clear()
        self.assets.clear()
        self.asset_tokens.clear()
        self.credential_values.clear()
        self.allowed_output_files.clear()

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
                token = self.asset_tokens.setdefault(path, uuid.uuid4().hex)
                self.assets[token] = Path(path)
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
        with self.lock:
            if session_id not in self.sessions:
                config = blocks.config
                for component in config['components']:
                    if component['type'] in ('file', 'image', 'audio', 'video', 'gallery'):
                        self.allow_files(component.get('props', {}).get('value'))
                self.sessions[session_id] = {'state': SessionState(blocks),
                    'values': {str(c['id']): self.export(c.get('props', {}).get('value')) for c in config['components'] if c['type'] != 'state'},
                    'props': {}, 'lock': threading.RLock(), 'cancelled': set()}
            return self.sessions[session_id]

    def allow_files(self, value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and Path(value['path']).is_file():
                self.allowed_output_files.add(value['path'])
            for item in value.values():
                self.allow_files(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.allow_files(item)

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
        folder = Path(self.uploads.name) / uuid.uuid4().hex
        folder.mkdir()
        path = folder / (Path(name).name or 'upload')
        try:
            with path.open('wb') as target:
                remaining = size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('文件上传中断，请重新上传')
                    target.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            folder.rmdir()
            raise
        token = uuid.uuid4().hex
        self.assets[token] = path
        self.asset_tokens[str(path)] = token
        return {'path': '/api/native/assets/' + token, 'url': '/api/native/assets/' + token,
                'orig_name': name, 'meta': {'_type': 'gradio.FileData'}}

    def prepare(self, value):
        if isinstance(value, dict):
            if '__upload' in value:
                upload = value['__upload']
                payload = base64.b64decode(upload['base64'], validate=True)
                return self.prepare(self.store_upload(upload['name'], io.BytesIO(payload), len(payload)))
            if isinstance(value.get('path'), str):
                if value['path'] in self.asset_tokens:
                    return value
                token = value['path'].rsplit('/', 1)[-1]
                if token in self.assets:
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
        with fn_lock:
            return asyncio.run(run())


runtime = NativeRuntime()
atexit.register(runtime.close)
