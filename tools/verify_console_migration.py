import argparse,json,urllib.request,time
from pathlib import Path
parser=argparse.ArgumentParser(description="原模块只读和输入校验回归；不执行桌面动作或在线模型任务")
parser.add_argument("--url",default="http://127.0.0.1:8766")
parser.add_argument("--output",default="logs/console-migration-acceptance.json")
args=parser.parse_args()
base=args.url.rstrip('/')+'/api/native/'
def post(path,data):
 request=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=120) as response:
  return json.load(response) if path=='layout' else [json.loads(line) for line in response]
layout=post('layout',{'session':'original-acceptance'})
components={c['id']:c for c in layout['components']}
by_name={d['api_name']:d for d in layout['dependencies']}
checks=[]
# Read-only operations and input guards; no desktop actions, data deletion or online model calls.
cases=['lambda','lambda_1','lambda_2','lambda_3','refresh_kg_stats_with_line','build_kg_workspace','search_entities','query_entity_relations','on_index_tab_load','update_results','show_search_stats','refresh_local_models','unload_local_model','refresh_local_models_1','unload_local_model_1','get_status_with_tab','view_history_with_tab','lambda_9','lambda_10','get_generation_history_wrapper','get_vm_status','refresh_cpt_models','refresh_dpo_models','refresh_inference_model_pair','unload_infer_model','view_preferences','show_data_stats','refresh_samples_1','run_w2v_viz','run_clm_viz','visualize_clip_data','show_system_status','check_data_quality','show_performance','show_model_status','process_rag_query','process_rag_query_1','lambda_6','lambda_7','lambda_8','generate_ab_responses']
for name in cases:
 dep=by_name[name];values={}
 for cid in dep['inputs']:
  c=components[cid]
  if c['type']!='state':values[str(cid)]=c['props'].get('value')
 if name=='update_results':values.update({'116':'机器学习','109':'tfidf'})
 if name in ('process_rag_query','process_rag_query_1','generate_ab_responses','lambda_8'):
  values[str(dep['inputs'][0])]=''
 if name in ('lambda_6','lambda_7'):
  values[str(dep['inputs'][0])]=None
 if name=='search_entities':values[str(dep['inputs'][0])]='中国'
 start=time.monotonic()
 try:
  events=post('event',{'session':'original-acceptance','fn_id':dep['id'],'values':values})
  failed=[e.get('error') for e in events if e['type']=='failed']
  checks.append({'name':name,'fn_id':dep['id'],'passed':not failed and events[-1]['type']=='completed','errors':failed,'duration':round(time.monotonic()-start,2)})
 except Exception as e:checks.append({'name':name,'passed':False,'errors':[str(e)]})
 print(name,checks[-1]['passed'],checks[-1].get('errors'),flush=True)
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({'scope':'只读操作与输入校验；不代表在线模型质量或桌面动作验收','component_count':len(layout['components']),'event_count':len(layout['dependencies']),'checks':checks},ensure_ascii=False,indent=2)+'\n')
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
