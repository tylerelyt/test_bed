"""Model inputs and artifacts. Training/serving is delegated to native tools."""
import hashlib
import json
from pathlib import Path

from sqlalchemy import select
from ..workbench_access import identity, require_business
from .common import get, owned, public, text
from .database import Dataset, ModelArtifact
from .files import save_json


class ModelSupply:
    def __init__(self, session, directory):
        self.session, self.directory = session, Path(directory)

    def train(self,data,queue):
        """Explicit model-supply command; the queue manages execution only."""
        return self.start_operation('train',data,queue)

    def start_operation(self,kind,data,queue):
        if kind not in ('train','merge','deploy'): raise ValueError('模型操作无效')
        return queue.enqueue({**data,'kind':kind},prepare=lambda folder:self.operations.prepare(kind,data))

    def listing(self):
        with self.session() as db:
            return {name: [public(x) for x in db.scalars(select(cls).where(cls.project == identity().project)
                    .order_by(cls.created_at.desc()).limit(200))]
                    for name, cls in [('datasets', Dataset), ('artifacts', ModelArtifact)]}

    def dataset(self, name, rows, stage="sft"):
        require_business()
        name = text(name, '数据集名称')
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
            raise ValueError('数据集须为包含 1—100000 条记录的 JSON 数组')
        if stage not in ('pt','sft','dpo'): raise ValueError('数据阶段无效')
        if stage in ('pt','dpo'):
            clean=[]
            for index,row in enumerate(rows):
                if not isinstance(row,dict):raise ValueError(f'第 {index+1} 条不是对象')
                if stage=='pt':clean.append({'text':text(row.get('text'),f'第 {index+1} 条 text',32000)})
                else:clean.append({'instruction':text(row.get('instruction'),f'第 {index+1} 条 instruction',32000),
                    'chosen':text(row.get('chosen'),f'第 {index+1} 条 chosen',32000),
                    'rejected':text(row.get('rejected'),f'第 {index+1} 条 rejected',32000)})
            descriptor={'columns':{'prompt':'text'}} if stage=='pt' else {'ranking':True,'columns':{'prompt':'instruction','chosen':'chosen','rejected':'rejected'}}
            return self.native_dataset(name,clean,descriptor)
        clean = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f'第 {index+1} 条记录不是对象')
            instruction = text(row.get('instruction'), f'第 {index+1} 条 instruction', 32000)
            output = text(row.get('output'), f'第 {index+1} 条 output', 32000)
            value = row.get('input', '')
            if not isinstance(value, str) or len(value) > 32000:
                raise ValueError(f'第 {index+1} 条 input 无效')
            clean.append({'instruction': instruction, 'input': value, 'output': output})
        raw = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode()
        if len(raw) > 10_000_000:
            raise ValueError('数据集超过 10 MB，请拆分上传')
        item_id = owned()
        folder = self.directory / 'datasets' / item_id['id']
        folder.mkdir(parents=True, mode=0o700)
        save_json(folder/'data.json', clean)
        save_json(folder/'dataset_info.json', {'workbench': {'file_name': 'data.json',
            'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}})
        with self.session.begin() as db:
            item = Dataset(**item_id, name=name, count=len(clean), digest=hashlib.sha256(raw).hexdigest(),
                           path=str(folder/'data.json'))
            db.add(item); db.flush()
            return public(item)

    def artifact(self, data, source_job=None):
        require_business()
        path = Path(text(data.get('path'), '本机模型目录', 2000)).expanduser().resolve()
        fmt = data.get('format', 'weights')
        if fmt not in ('weights', 'adapter') or not path.is_dir():
            raise ValueError('请选择存在的本机模型目录，以及 weights 或 adapter 格式')
        config = 'adapter_config.json' if fmt == 'adapter' else 'config.json'
        patterns = ('adapter_model.safetensors', 'adapter_model.bin') if fmt == 'adapter' else ('*.safetensors', 'pytorch_model*.bin')
        if not (path/config).is_file() or not any(any(path.glob(pattern)) for pattern in patterns):
            raise ValueError('目录缺少模型配置或权重文件，不能登记为可用产物')
        base = text(data.get('base_model') or str(path), '基础模型', 2000)
        if fmt == 'adapter':
            base_path = Path(base).expanduser().resolve()
            if not (base_path/'config.json').is_file():
                raise ValueError('LoRA 产物必须关联实际存在的本机基础模型目录')
            base = str(base_path)
        with self.session.begin() as db:
            item = ModelArtifact(**owned(), name=text(data.get('name'), '模型产物名称'), path=str(path),
                                 base_model=base, format=fmt, source_job=source_job)
            db.add(item); db.flush()
            return public(item)

    def training_inputs(self, data):
        with self.session() as db:
            dataset = get(db, Dataset, data.get('dataset_id'))
            artifact = get(db, ModelArtifact, data.get('artifact_id'))
            if artifact.format != 'weights':
                raise ValueError('训练需要完整基础模型；LoRA 产物请先合并导出')
            return public(dataset), public(artifact)

    def native_dataset(self,name,rows,descriptor):
        """Freeze native LLaMA-Factory data without changing CPT/DPO column semantics."""
        require_business()
        name=text(name,'数据集名称')
        if not isinstance(rows,list) or not rows or len(rows)>100000:
            raise ValueError('原训练数据须为非空 JSON 数组，最多十万条')
        raw=json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()
        fields=owned();folder=self.directory/'datasets'/fields['id'];folder.mkdir(parents=True,mode=0o700)
        save_json(folder/'data.json',rows)
        save_json(folder/'dataset_info.json',{'workbench':{**descriptor,'file_name':'data.json'}})
        with self.session.begin() as db:
            item=Dataset(**fields,name=name,count=len(rows),digest=hashlib.sha256(raw).hexdigest(),path=str(folder/'data.json'))
            db.add(item);db.flush();return public(item)
