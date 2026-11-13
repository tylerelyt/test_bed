"""Model-supply operations. The job dispatcher supplies process ownership only."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import httpx
from .common import get,public,text
from .database import Job,ModelArtifact
from .environment import python_for,probe
from .model_training import LlamaFactoryTraining
from .files import save_json
from ..workbench_access import private_write


class ModelOperations:
    def __init__(self,session,directory,models,services):
        self.session,self.directory,self.models,self.services=session,Path(directory),models,services

    def prepare(self,kind,data):
        params={}
        package = 'llamafactory' if kind in ('train', 'merge') else 'vllm'
        version = probe(python_for(kind), package, int(time.time()//30))
        if not version:
            raise ValueError(f'{package} 未安装；请在运行环境页查看对应 Python 环境')
        params.update(engine=package, engine_version=version)
        if kind == 'train':
            params.update(LlamaFactoryTraining().prepare(data,self.models))
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
        return params

    def execute(self, item, execution):
        job_id, p, kind = item['id'], item['parameters'], item['kind']
        folder = self.directory/'jobs'/job_id
        output = folder/'output'
        artifact = p['artifact']
        if probe(python_for(kind),p['engine'],int(time.time()//30))!=p['engine_version']:
            raise ValueError('运行组件版本已变化，请核对环境后重新提交')
        if kind in ('train', 'merge'):
            config = {'model_name_or_path': artifact['path'] if kind == 'train' else artifact['base_model'],
                      'finetuning_type': 'lora'}
            if kind == 'train':
                raw = json.loads(Path(p['dataset']['path']).read_text())
                actual = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if actual != p['dataset']['digest']:
                    raise ValueError('训练数据摘要校验失败')
                config = LlamaFactoryTraining().configuration(p,output)
                dataset_folder=folder/'training-data';dataset_folder.mkdir(mode=0o700)
                save_json(dataset_folder/'data.json',raw)
                save_json(dataset_folder/'dataset_info.json',{'workbench':p['dataset_descriptor']})
                config['dataset_dir']=str(dataset_folder)
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
            with execution.lock:
                execution.processes[job_id] = process
            try:
                ready = False
                started = time.monotonic()
                while process.poll() is None:
                    if execution.cancelled(job_id):
                        raise execution.cancellation_error()
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
                            self.services.model({'name': item['name'], 'model': 'workbench-'+job_id,
                                                       'base_url': base}, deployment_job=job_id)
                    if (folder/'process.log').stat().st_size > 100_000_000:
                        raise ValueError('作业日志超过 100 MB，已停止进程')
                    execution.stop.wait(.3)
                if process.returncode != 0 or kind == 'deploy':
                    raise ValueError(f'原生进程退出（代码 {process.returncode}），请查看作业日志')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                with execution.lock:
                    execution.processes.pop(job_id, None)
        result = self.models.artifact({'name': item['name'], 'path': str(output),
            'format': 'adapter' if kind == 'train' else 'weights',
            'base_model': artifact['path'] if kind == 'train' else artifact['base_model']}, source_job=job_id)
        return {'artifact_id': result['id'], 'path': result['path']}
