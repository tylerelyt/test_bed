"""Model-supply training adapter. Native configuration comes from existing business logic."""
import json
import re
from pathlib import Path
from ..training_config import build_train_args
from .common import text


class LlamaFactoryTraining:
    def prepare(self, data, models):
        dataset, artifact = models.training_inputs(data)
        stage = data.get('stage','sft')
        if stage not in ('pt','sft','dpo'): raise ValueError('训练阶段须为 pt、sft 或 dpo')
        template = text(data.get('template','default'),'对话模板',80)
        if not re.fullmatch('[a-zA-Z0-9_-]+',template): raise ValueError('模板名称无效')
        epochs,rate = data.get('epochs',1),data.get('learning_rate',.0001)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in (epochs,rate)) or not .01<=epochs<=100 or not 1e-7<=rate<=.01:
            raise ValueError('训练轮次或学习率超出范围')
        descriptor = json.loads((Path(dataset['path']).parent/'dataset_info.json').read_text())['workbench']
        if descriptor.get('file_name')!='data.json':raise ValueError('数据集文件描述已变化，请重新登记')
        ranking = descriptor.get('ranking',False)
        if stage=='dpo' and not ranking: raise ValueError('DPO 需要偏好数据集，须包含 chosen 和 rejected')
        if stage!='dpo' and ranking: raise ValueError('偏好数据集只能用于 DPO')
        is_text=descriptor.get('columns',{}).get('prompt')=='text'
        if (stage=='pt') != is_text: raise ValueError('CPT 需要 text 语料，SFT 需要指令数据')
        cutoff=data.get('cutoff_len',512)
        if isinstance(cutoff,bool) or not isinstance(cutoff,int) or not 32<=cutoff<=8192:raise ValueError('训练截断长度须为 32—8192')
        return {'dataset_descriptor':descriptor,'cutoff_len':cutoff,'dataset':dataset,'artifact':artifact,'stage':stage,'epochs':epochs,'learning_rate':rate,'template':template}

    def configuration(self, parameters, output):
        p=parameters
        config = build_train_args({'stage':p.get('stage','sft'),'model_name_or_path':p['artifact']['path'],
            'dataset':'workbench','dataset_dir':str(Path(p['dataset']['path']).parent),
            'template':p['template'],'output_dir':str(output),'num_train_epochs':p['epochs'],
            'learning_rate':p['learning_rate'],'per_device_train_batch_size':1,'gradient_accumulation_steps':8,
            'cutoff_len':p.get('cutoff_len',512),'logging_steps':1,'overwrite_output_dir':False})
        config.pop('trust_remote_code',None)  # LLaMA-Factory 0.9.1 does not accept this argument.
        config.update(overwrite_output_dir=False,report_to='none',plot_loss=False)
        return config
