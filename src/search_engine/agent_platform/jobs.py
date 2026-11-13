"""Single-host durable supply jobs, with bounded native-process ownership."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

import httpx
from sqlalchemy import select, update

from ..workbench_access import identity, require_business, require_operator, private_write
from .common import get, owned, public, text
from .database import Agent, Job, ModelArtifact, Release
from .environment import python_for, probe
from .files import extract, save_json, digest

ACTIVE = ('queued', 'running', 'serving', 'cancelling')


class Cancelled(Exception):
    pass


class Jobs:
    def __init__(self, session, directory, access, context, models, catalog):
        self.session, self.directory, self.access = session, Path(directory), access
        self.context, self.models, self.catalog = context, models, catalog
        self.stop = threading.Event()
        self.shutdown_requested = False
        self.lock = threading.RLock()
        self.processes = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='supply-job')
        self.futures = []
        self.thread = None

    def listing(self):
        with self.session() as db:
            return [public(x) for x in db.scalars(select(Job).where(Job.project == identity().project)
                    .order_by(Job.created_at.desc()).limit(200))]

    def detail(self, job_id):
        with self.session() as db:
            item = public(get(db, Job, job_id))
        log = self.directory/'jobs'/job_id/'process.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size-32000))
                item['log'] = stream.read(32000).decode('utf-8', errors='replace')
        else:
            item['log'] = ''
        return item

    def start(self, data):
        require_operator() if data.get('kind') == 'import' else require_business()
        request_id = data.get('request_id')
        if not isinstance(request_id, str) or not re.fullmatch('[a-f0-9]{32}', request_id):
            raise ValueError('作业请求标识须为 32 位十六进制值')
        kind, name = data.get('kind'), text(data.get('name'), '作业名称')
        if kind not in ('import', 'train', 'merge', 'deploy'):
            raise ValueError('作业类型无效')
        job_id = hashlib.sha256((identity().project+':'+request_id).encode()).hexdigest()[:32]
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.session() as db:
            old = db.get(Job, job_id)
            if old:
                get(db, Job, job_id)
                if old.parameters.get('request_sha256') != fingerprint:
                    raise ValueError('同一请求标识不能用于不同作业')
                return public(old)
            if len(list(db.scalars(select(Job).where(Job.status.in_(ACTIVE))))) >= 20:
                raise ValueError('作业队列已满')
        folder = self.directory/'jobs'/job_id
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        params = {'request_sha256': fingerprint}
        if kind == 'import':
            filename = text(data.get('filename'), '文件名', 200)
            suffix = Path(filename).suffix.lower()
            if suffix not in ('.txt', '.md', '.csv', '.json', '.pdf', '.docx'):
                raise ValueError('不支持此文件格式')
            try:
                raw = base64.b64decode(data.get('content_base64', ''), validate=True)
            except (ValueError, TypeError):
                raise ValueError('文件编码无效')
            if not 0 < len(raw) <= 5_000_000:
                raise ValueError('导入文件须为 1 字节—5 MB')
            path = folder/('source'+suffix)
            path.write_bytes(raw); path.chmod(0o600)
            params.update(filename=filename, source=str(path), sha256=digest(path), suffix=suffix)
        else:
            package = 'llamafactory' if kind in ('train', 'merge') else 'vllm'
            version = probe(python_for(kind), package, int(time.time()//30))
            if not version:
                raise ValueError(f'{package} 未安装；请在运行环境页查看对应 Python 环境')
            params.update(engine=package, engine_version=version)
            if kind == 'train':
                dataset, artifact = self.models.training_inputs(data)
                epochs = data.get('epochs', 1)
                rate = data.get('learning_rate', .0001)
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (epochs, rate)) or not .01 <= epochs <= 100 or not 1e-7 <= rate <= .01:
                    raise ValueError('训练轮次或学习率超出范围')
                template = text(data.get('template', 'default'), '对话模板', 80)
                if not re.fullmatch('[a-zA-Z0-9_-]+', template):
                    raise ValueError('模板名称无效')
                params.update(dataset=dataset, artifact=artifact, epochs=epochs, learning_rate=rate, template=template)
            else:
                with self.session() as db:
                    artifact = public(get(db, ModelArtifact, data.get('artifact_id')))
                if kind == 'merge' and artifact['format'] != 'adapter':
                    raise ValueError('合并导出请选择 LoRA 产物')
                if kind == 'deploy' and artifact['format'] != 'weights':
                    raise ValueError('部署请选择完整模型；LoRA 产物须先合并导出')
                params['artifact'] = artifact
                if kind == 'merge':
                    with self.session() as db:
                        source = db.get(Job, artifact['source_job']) if artifact['source_job'] else None
                        template = source.parameters.get('template', 'default') if source else data.get('template', 'default')
                    template = text(template, '对话模板', 80)
                    if not re.fullmatch('[a-zA-Z0-9_-]+', template):
                        raise ValueError('模板名称无效')
                    params['template'] = template
                if kind == 'deploy':
                    port = data.get('port', 8000)
                    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
                        raise ValueError('端口须为 1024—65535')
                    parser = data.get('tool_parser', '')
                    if not isinstance(parser, str) or len(parser)>80 or (parser and not re.fullmatch('[a-zA-Z0-9_-]+', parser)):
                        raise ValueError('工具调用解析器名称无效')
                    params.update(port=port, tool_parser=parser)
        with self.session.begin() as db:
            fields = owned(); fields['id'] = job_id
            job = Job(**fields, kind=kind, name=name, parameters=params, execution_identity=identity().claim())
            db.add(job); db.flush()
            return public(job)

    def cancel(self, job_id):
        require_operator()
        with self.session.begin() as db:
            item = get(db, Job, job_id)
            if item.kind != 'import':
                require_business()
            # Do not silently break a released Agent by stopping its model.
            model_id = item.result.get('model_id')
            if item.status == 'serving' and model_id:
                defaults = set(db.scalars(select(Agent.current_release).where(Agent.project == identity().project)))
                referenced = any(r.id in defaults and r.manifest['model']['id'] == model_id for r in db.scalars(
                    select(Release).where(Release.project == identity().project)))
                if referenced:
                    raise ValueError('此部署正被岗位 Agent 的采用版本使用，请先为相关 Agent 采用其他模型版本')
            if item.status in ACTIVE:
                status = 'cancelled' if item.status == 'queued' else 'cancelling'
                db.execute(update(Job).where(Job.id == job_id, Job.status.in_(ACTIVE)).values(status=status))
        return self.detail(job_id)

    def cancelled(self, job_id):
        with self.session() as db:
            return self.shutdown_requested or self.stop.is_set() or db.get(Job, job_id).status in ('cancelling', 'cancelled')

    def recover(self):
        with self.session.begin() as db:
            for item in db.scalars(select(Job).where(Job.status.in_(ACTIVE))):
                item.status, item.error, item.finished_at = 'interrupted', '执行进程已重启；请核对模型进程和产物，未自动重跑', time.time()

    def execute_next(self, asynchronous=False):
        with self.session.begin() as db:
            item = db.scalar(select(Job).where(Job.status == 'queued').order_by(Job.created_at).limit(1))
            if not item:
                return False
            changed = db.execute(update(Job).where(Job.id == item.id, Job.status == 'queued').values(status='running'))
            if changed.rowcount != 1:
                return True
            job_id, claims = item.id, item.execution_identity
        if asynchronous:
            self.futures.append(self.pool.submit(self.execute, job_id, claims))
        else:
            self.execute(job_id, claims)
        return True

    def execute(self, job_id, claims):
        try:
            with self.access.resume(claims):
                with self.session() as db:
                    item = public(db.get(Job, job_id))
                require_operator() if item['kind'] == 'import' else require_business()
                if self.cancelled(job_id):
                    raise Cancelled()
                p = item['parameters']
                if item['kind'] == 'import':
                    source = Path(p['source'])
                    if digest(source) != p['sha256']:
                        raise ValueError('原文件摘要校验失败')
                    content = extract(source, p['suffix'])
                    if self.cancelled(job_id):
                        raise Cancelled()
                    resource = self.context.resource(item['name'], 'knowledge', content,
                                                     '导入文件：'+p['filename']+'；SHA256：'+p['sha256'])
                    result = {'resource_id': resource['id'], 'version': resource['version'], 'chars': len(content)}
                else:
                    result = self.native(item)
                with self.session.begin() as db:
                    current = db.get(Job, job_id)
                    current.result, current.status, current.finished_at = result, 'completed', time.time()
        except Exception as exc:
            with self.session.begin() as db:
                current = db.get(Job, job_id)
                current.status = 'cancelled' if isinstance(exc, Cancelled) or self.cancelled(job_id) else 'failed'
                current.error = '' if current.status == 'cancelled' else str(exc)[:2000]
                current.finished_at = time.time()

    def native(self, item):
        job_id, p, kind = item['id'], item['parameters'], item['kind']
        folder = self.directory/'jobs'/job_id
        output = folder/'output'
        artifact = p['artifact']
        if kind in ('train', 'merge'):
            config = {'model_name_or_path': artifact['path'] if kind == 'train' else artifact['base_model'],
                      'finetuning_type': 'lora'}
            if kind == 'train':
                raw = json.loads(Path(p['dataset']['path']).read_text())
                actual = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if actual != p['dataset']['digest']:
                    raise ValueError('训练数据摘要校验失败')
                config.update(stage='sft', do_train=True, dataset='workbench',
                    dataset_dir=str(Path(p['dataset']['path']).parent), template=p['template'],
                    output_dir=str(output), overwrite_output_dir=False, num_train_epochs=p['epochs'],
                    learning_rate=p['learning_rate'], per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    cutoff_len=2048, logging_steps=1, save_steps=100, lora_target='all',
                    preprocessing_num_workers=1, report_to='none', fp16=False, bf16=False)
            else:
                config.update(adapter_name_or_path=artifact['path'], export_dir=str(output),
                              export_size=2, export_device='cpu', export_legacy_format=False, template=p['template'])
            save_json(folder/'config.json', config)
            # LLaMA-Factory's native CLI supports a JSON configuration filename.
            command = [python_for(kind), '-c', 'from llamafactory.cli import main; main()',
                       'train' if kind == 'train' else 'export', str(folder/'config.json')]
        else:
            # Refuse an occupied port before creating the owned inference process.
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', p['port']))
            command = [python_for(kind), '-m', 'vllm.entrypoints.openai.api_server', '--model', artifact['path'],
                       '--host', '127.0.0.1', '--port', str(p['port']), '--served-model-name', 'workbench-'+job_id]
            if p['tool_parser']:
                command += ['--enable-auto-tool-choice', '--tool-call-parser', p['tool_parser']]
        save_json(folder/'invocation.json', {'argv': command, 'engine_version': p['engine_version'], 'artifact_id': artifact['id']})
        with (folder/'process.log').open('ab') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            private_write(folder/'process.json', json.dumps({'pid': process.pid, 'started_at': time.time()}))
            with self.lock:
                self.processes[job_id] = process
            try:
                ready = False
                started = time.monotonic()
                while process.poll() is None:
                    if self.cancelled(job_id):
                        raise Cancelled()
                    if kind == 'deploy' and not ready:
                        if time.monotonic()-started > 600:
                            raise ValueError('推理服务 10 分钟内未就绪，请检查日志和硬件')
                        base = f'http://127.0.0.1:{p["port"]}/v1'
                        try:
                            response = httpx.get(base+'/models', timeout=1)
                            names = [x['id'] for x in response.json().get('data', [])] if response.status_code == 200 else []
                            ready = 'workbench-'+job_id in names and process.poll() is None
                        except (httpx.HTTPError, ValueError, KeyError):
                            ready = False
                        if ready:
                            self.catalog.models.model({'name': item['name'], 'model': 'workbench-'+job_id,
                                                       'base_url': base}, deployment_job=job_id)
                    if (folder/'process.log').stat().st_size > 100_000_000:
                        raise ValueError('作业日志超过 100 MB，已停止进程')
                    self.stop.wait(.3)
                if process.returncode != 0 or kind == 'deploy':
                    raise ValueError(f'原生进程退出（代码 {process.returncode}），请查看作业日志')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                with self.lock:
                    self.processes.pop(job_id, None)
        result = self.models.artifact({'name': item['name'], 'path': str(output),
            'format': 'adapter' if kind == 'train' else 'weights',
            'base_model': artifact['path'] if kind == 'train' else artifact['base_model']}, source_job=job_id)
        return {'artifact_id': result['id'], 'path': result['path']}

    def loop(self):
        while not self.shutdown_requested and not self.stop.wait(.3):
            self.futures = [x for x in self.futures if not x.done()]
            if len(self.futures) < 2:
                try:
                    self.execute_next(asynchronous=True)
                except Exception:
                    self.stop.wait(1)

    def start_worker(self):
        self.recover()
        self.thread = threading.Thread(target=self.loop, daemon=True, name='supply-dispatcher')
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.pool.shutdown(wait=True)
