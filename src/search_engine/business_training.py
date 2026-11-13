"""Attach native CPT/SFT/DPO processes to the worker's durable business run."""
import json
from pathlib import Path
import shutil
from .business_execution import current_execution
from .agent_platform.models import ModelSupply
from .agent_platform.database import connect
from .workbench_access import private_write


def prepare_training(config, args):
    scope=current_execution()
    if not scope:return None
    folder=scope['directory']/'business-processes'/scope['run']['id']
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    engine,session=connect(scope['directory'])
    models=ModelSupply(session,scope['directory'])
    try:
        source=Path(config.get('dataset_dir','data/llmops'))
        info=json.loads((source/'dataset_info.json').read_text())
        dataset_names=[x.strip() for x in config['dataset'].split(',')]
        ids=[]
        for name in dataset_names:
            descriptor=info[name]
            if not descriptor.get('file_name'):raise ValueError('训练需要已下载到本机的数据文件，请先导入模型供应')
            rows=json.loads((source/descriptor['file_name']).read_text())
            saved=models.native_dataset(name,rows,descriptor)
            ids.append(saved['id'])
            # Freeze the exact native descriptor too, so custom column mappings survive.
            shutil.copy2(source/descriptor['file_name'],folder/(saved['id']+'.json'))
            info[name]={**descriptor,'file_name':saved['id']+'.json'}
        private_write(folder/'dataset_info.json',json.dumps({n:info[n] for n in dataset_names},ensure_ascii=False))
        args['dataset_dir']=str(folder.resolve())
        args.pop('trust_remote_code',None)  # not accepted by LLaMA-Factory 0.9.1
        private_write(folder/'train-config.json',json.dumps(args,ensure_ascii=False,indent=2))
        scope['run']['training']={'stage':config.get('stage','sft'),'dataset_ids':ids,
            'base_model':config['model_name_or_path'],'output_dir':str(Path(args['output_dir']).resolve()),
            'config':str(folder/'train-config.json'),'log':str(folder/'process.log')}
        scope['records'].put('runs',scope['run'])
        return {'scope':scope,'models':models,'engine':engine,'folder':folder}
    except BaseException:
        engine.dispose();raise


def training_finished(managed,return_code):
    if not managed:return
    scope=managed['scope'];training=scope['run']['training']
    if return_code==0:
        try:
            artifact=managed['models'].artifact({'name':scope['run']['action'],'path':training['output_dir'],
                'format':'adapter','base_model':training['base_model']},source_job=scope['run']['id'])
            training['artifact_id']=artifact['id']
        except ValueError as exc:training['artifact_error']=str(exc)
    training['return_code']=return_code
    scope['records'].system_put('runs',scope['run']);managed['engine'].dispose()
