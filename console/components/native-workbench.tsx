'use client';
import {useCallback,useEffect,useRef,useState} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {Loader2,RefreshCw} from 'lucide-react';
import {displayLabel,displayMarkdown} from '../lib/presentation';
import {resultDocument} from '../lib/result-document';
import RunHistory,{request,SavedTask} from './run-history';
import {runStatus} from '../lib/workbench-api';
import ServiceDiagnostics from './service-diagnostics';
type Props=Record<string,unknown>;
type Component={id:number;type:string;props:Props};
type Node={id:number;children?:Node[]};
type Dependency={id:number;targets:[number|null,string][];inputs:number[];outputs:number[];trigger_after:number|null;cancels:number[];backend_fn:boolean};
type Layout={runtime_id:string;components:Component[];layout:Node;dependencies:Dependency[]};
type Delta=Record<string,{props?:Props;value?:unknown}>;
const text=(v:unknown)=>v==null?'':typeof v==='string'?v:JSON.stringify(v);
const fileUrl=(v:unknown):string=>typeof v==='string'?v:((v as {url?:string;path?:string})?.url||(v as {path?:string})?.path||'');
export default function NativeWorkbench({moduleIndex=0}:{moduleIndex?:number}){
 const [layout,setLayout]=useState<Layout|null>(null),[values,setValues]=useState<Record<string,unknown>>({}),[overrides,setOverrides]=useState<Record<string,Props>>({}),[active,setActive]=useState<Record<string,number>>({}),[busy,setBusy]=useState(0),[error,setError]=useState(''),[status,setStatus]=useState('');
 const runtimeId=useRef('');
 const session=useRef(''),current=useRef<Record<string,unknown>>({}),chain=useRef<Promise<void>>(Promise.resolve());
 const [loading,setLoading]=useState(false),[uploading,setUploading]=useState<number|null>(null);
 const [revision,setRevision]=useState(0),[savedTask,setSavedTask]=useState<SavedTask|null>(null);
 const attached=useRef(new Map<string,AbortController>());
 function remember(id:string,remove=false){const ids:string[]=JSON.parse(sessionStorage.getItem('workbench-active-runs')||'[]');sessionStorage.setItem('workbench-active-runs',JSON.stringify(remove?ids.filter(v=>v!==id):Array.from(new Set([...ids,id]))))}
 async function connect(){setLoading(true);setError('');try{
  if(!session.current){session.current=sessionStorage.getItem('workbench-session')||crypto.randomUUID();sessionStorage.setItem('workbench-session',session.current)}
  const r=await fetch('/api/native/layout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session:session.current})});const data=await r.json();if(!r.ok)throw Error(data.error||'模块初始化失败');
  runtimeId.current=data.runtime_id;const initial=Object.fromEntries(data.components.map((c:Component)=>[String(c.id),c.props.value]));current.current=initial;setValues(initial);setOverrides({});setLayout(data);setStatus('工作台已就绪');
 }catch(e){setError((e as Error).message)}finally{setLoading(false)}}
 useEffect(()=>{void connect()},[]);
 const components=new Map(layout?.components.map(c=>[c.id,c])||[]);
 const rootTabs=layout?.components.find(c=>c.type==='tabs');
 const findNode=(node:Node,id:number):Node|undefined=>node.id===id?node:node.children?.map(n=>findNode(n,id)).find(Boolean);
 const rootTabNodes=layout&&rootTabs?findNode(layout.layout,rootTabs.id)?.children||[]:[];
 const rootTabIds=new Set(rootTabNodes.map(n=>n.id));
 const firstMarkdown=(node:Node):number|undefined=>components.get(node.id)?.type==='markdown'?node.id:node.children?.map(firstMarkdown).find(id=>id!==undefined);
 const introIds=new Set(rootTabNodes.map(firstMarkdown));
 const moduleLabel=text(components.get(rootTabNodes[moduleIndex]?.id)?.props.label);
 const apply=useCallback((delta:Delta)=>{
  const next={...current.current};
  for(const [id,entry]of Object.entries(delta)){if(!entry||typeof entry!=='object')continue;if('value'in entry)next[id]=entry.value;if(entry.props&&'value'in entry.props)next[id]=entry.props.value;}
  current.current=next;setValues(next);
  setOverrides(old=>{const merged={...old};for(const[id,entry]of Object.entries(delta))if(entry.props)merged[id]={...merged[id],...entry.props};return merged});
 },[]);
 async function followRun(id:string){
  if(attached.current.has(id))return;
  const controller=new AbortController();attached.current.set(id,controller);setBusy(n=>n+1);
  let cursor=0,settled=false;
  try{for(let attempt=0;attempt<3&&!settled;attempt++){
   try{
    const record=await request('runs/'+id);if(record.runtime_id!==runtimeId.current||record.current_runtime_id!==runtimeId.current){remember(id,true);setStatus('执行环境已变化，历史结果保留在运行记录中');return}
    const response=await fetch(`/api/native/runs/${id}/events?after=${cursor}`,{signal:controller.signal,cache:'no-store'});
    if(!response.ok){const result=await response.json();if(result.error==='记录不存在')remember(id,true);throw Error(result.error||'运行输出连接失败')}
    const reader=response.body!.getReader(),decoder=new TextDecoder();let buffer='';
    const receive=(line:string)=>{if(!line.trim())return;const item=JSON.parse(line);cursor=item.seq??cursor;if(item.type==='started')setStatus('正在执行…');if(item.updates)apply(item.updates);if(item.type==='failed')setError(item.error);if(item.type==='settled'){settled=true;remember(id,true);setStatus(runStatus[item.status]||'运行已结束');if(item.error)setError(item.error)}};
    while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop()||'';lines.forEach(receive)}receive(buffer+decoder.decode());
    if(!settled)throw Error('运行输出连接中断');
   }catch(e){if(controller.signal.aborted)return;if(attempt===2)throw e;setStatus('正在重新连接运行输出…');await new Promise(resolve=>setTimeout(resolve,700))}
  }}catch(e){setStatus('输出连接中断，可在记录中重连');setError((e as Error).message+'。可在运行记录中重新连接，操作仍由服务端执行。')}finally{attached.current.delete(id);setBusy(n=>Math.max(0,n-1));setRevision(n=>n+1)}
 }
 async function run(dependency:Dependency,event?:unknown){
  setBusy(n=>n+1);setError('');setStatus('正在执行…');let requestId:string|undefined;
  try{
   const input=Object.fromEntries(dependency.inputs.filter(id=>components.get(id)?.type!=='state').map(id=>[String(id),current.current[id]??null]));
   if(dependency.targets.some(t=>['click','submit'].includes(t[1]))){
    requestId=crypto.randomUUID().replaceAll('-','');remember(requestId);
    const result=await request('runs',{session:session.current,fn_id:dependency.id,values:input,event,task_id:savedTask?.fn_id===dependency.id?savedTask.id:undefined,task_version:savedTask?.fn_id===dependency.id?(savedTask.version||1):undefined,request_id:requestId});setStatus(savedTask?.requires_review?'等待人工放行，请在运行记录中处理':'操作已入队，正在连接输出…');setRevision(n=>n+1);await followRun(result.run_id);return;
   }
   const r=await fetch('/api/native/event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session:session.current,fn_id:dependency.id,values:input,event})});
   if(!r.ok){const d=await r.json();throw Error(d.error||'请求失败')}
   const reader=r.body!.getReader(),decoder=new TextDecoder();let buffer='';
   const receive=(line:string)=>{if(!line.trim())return;const item=JSON.parse(line);if(item.type==='failed')throw Error(item.error);if(item.updates)apply(item.updates);if(item.type==='completed')setStatus('执行完成')};
   while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop()||'';lines.forEach(receive)}receive(buffer+decoder.decode());
  }catch(e){const code=(e as Error&{status?:number}).status;if(requestId&&code&&code>=400&&code<500)remember(requestId,true);setError((e as Error).message);setStatus('执行失败')}finally{setBusy(n=>n-1)}
 }
 useEffect(()=>{if(layout){for(const d of layout.dependencies.filter(d=>d.targets.some(t=>t[1]==='load')))chain.current=chain.current.then(()=>run(d));chain.current=chain.current.then(()=>{const ids:string[]=JSON.parse(sessionStorage.getItem('workbench-active-runs')||'[]');ids.forEach(id=>void followRun(id))})}},[layout]);
 useEffect(()=>()=>{attached.current.forEach(controller=>controller.abort())},[]);
 function restoreTask(task:SavedTask,restored:Record<string,unknown>,missing:string[]){
  const next={...current.current,...restored};missing.forEach(id=>{next[id]=null});current.current=next;setValues(next);setSavedTask(task);
  const target=layout?.dependencies.find(d=>d.id===task.fn_id)?.targets[0]?.[0];
  const trail=(node:Node):number[]|null=>node.id===target?[node.id]:(node.children||[]).map(child=>{const found=trail(child);return found?[node.id,...found]:null}).find(Boolean)||null;
  const path=layout?trail(layout.layout):null;
  if(path){const tabs:Record<string,number>={};for(let i=0;i<path.length-1;i++)if(components.get(path[i])?.type==='tabs')tabs[path[i]]=path[i+1];setActive(old=>({...old,...tabs}));setOverrides(old=>{const next={...old};for(const[id,tab]of Object.entries(tabs))next[id]={...next[id],selected:components.get(tab)?.props.id??components.get(tab)?.props.label};return next})}
  setStatus('任务参数已恢复；核对后点击原操作按钮执行');
 }
 function trigger(id:number,kind:string,event?:unknown){
  const dependencies=layout?.dependencies.filter(d=>d.targets.some(t=>t[0]===id&&t[1]===kind))||[];
  for(const dependency of dependencies){
   void run(dependency,event);
  }
 }
 function set(id:number,value:unknown){current.current={...current.current,[id]:value};setValues({...current.current})}
 function changed(id:number,value:unknown){set(id,value);trigger(id,'change')}
 async function upload(id:number,list:FileList|null,multiple:boolean,kind:string){
  if(!list||list.length===0)return;setError('');setUploading(id);setBusy(n=>n+1);setStatus('正在上传文件…');
  try{
   const prepared=await Promise.all(Array.from(list).map(async file=>{
    const response=await fetch('/api/native/upload',{method:'POST',headers:{'X-File-Name':encodeURIComponent(file.name),'X-File-Size':String(file.size)},body:file});
    const value=await response.json();if(!response.ok)throw Error(value.error||'文件上传失败');return value;
   }));
   changed(id,multiple?prepared:prepared[0]);trigger(id,'upload');setStatus('文件上传完成');
  }catch(e){setError((e as Error).message);setStatus('文件上传失败')}finally{setUploading(null);setBusy(n=>n-1)}
 }

 function render(node:Node):React.ReactNode{
  const c=components.get(node.id);if(!c)return <div>{node.children?.map(n=><div key={n.id}>{render(n)}</div>)}</div>;
  const p={...c.props,...overrides[c.id]},value=values[c.id],id=`native-${c.id}`;
  if(p.visible===false||c.type==='state')return null;
  const children=()=>node.children?.map(n=><div key={n.id} className="native-node" data-component-id={n.id}>{render(n)}</div>);
  const isInput=layout?.dependencies.some(d=>d.inputs.includes(c.id));
  const isOutput=layout?.dependencies.some(d=>d.outputs.includes(c.id));
  const label=displayLabel(text(p.label)),interactive=p.interactive==null?(!isOutput||Boolean(isInput)):p.interactive!==false;
  if(['row','column','group','form','blocks'].includes(c.type))return <div className={`native-${c.type}`}>{children()}</div>;
  if(c.type==='tabs'){
   const tabs=(node.children||[]).filter(n=>components.get(n.id)?.type==='tabitem'||components.get(n.id)?.type==='tab');
   const visible=tabs.filter(t=>({...components.get(t.id)?.props,...overrides[t.id]}).visible!==false);
   const requested=overrides[c.id]?.selected??p.selected;
   const requestedTab=visible.find(t=>components.get(t.id)?.props.id===requested||components.get(t.id)?.props.label===requested);
   const selected=c.id===rootTabs?.id?(visible[moduleIndex]?.id??visible[0]?.id):(requestedTab?.id??active[c.id]??visible[0]?.id);
   return <div className="native-tabs"><div role="tablist" className="native-tablist" hidden={c.id===rootTabs?.id}>{visible.map(tab=><button key={tab.id} role="tab" aria-selected={selected===tab.id} onClick={()=>{setActive({...active,[c.id]:tab.id});setOverrides(old=>({...old,[c.id]:{...old[c.id],selected:components.get(tab.id)?.props.id??components.get(tab.id)?.props.label}}));trigger(tab.id,'select')}}>{displayLabel(text(components.get(tab.id)?.props.label))}</button>)}</div>{visible.map(tab=><div key={tab.id} role="tabpanel" hidden={selected!==tab.id}>{render(tab)}</div>)}</div>
  }
  if(c.type==='tabitem'||c.type==='tab')return <div className={`native-tab-content ${rootTabIds.has(c.id)?'native-module':''}`}>{children()}</div>;
  if(c.type==='accordion')return <details open={p.open!==false} className="native-accordion"><summary>{label}</summary>{children()}</details>;
  if(c.type==='markdown'){
   // The shell already names the module; keep its original supporting description.
   const content=displayMarkdown(text(value),introIds.has(c.id));
   if(!content||(!introIds.has(c.id)&&node.id===layout?.layout.children?.[0]?.id))return null;
   return <div className={`native-markdown ${introIds.has(c.id)?'native-intro':''}`}><Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown></div>;
  }
  if(c.type==='html')return <Html value={text(value)}/>;
  if(c.type==='button')return <button className={p.variant==='primary'?'primary':'secondary'} disabled={p.interactive===false||(uploading!=null&&Boolean(layout?.dependencies.some(d=>d.targets.some(t=>t[0]===c.id)&&d.inputs.includes(uploading))))} onClick={()=>trigger(c.id,'click')}>{displayLabel(text(value))||label}</button>;
  if(c.type==='textbox')return <div className="native-field"><label htmlFor={id}>{label}</label>{p.type==='password'?<input type="password" id={id} autoComplete="off" value={text(value)} disabled={!interactive} onChange={e=>set(c.id,e.target.value)} onBlur={()=>trigger(c.id,'change')}/>:<textarea id={id} rows={Number(p.lines)||2} value={text(value)} readOnly={!interactive} placeholder={text(p.placeholder)} onChange={e=>set(c.id,e.target.value)} onBlur={()=>trigger(c.id,'change')} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&(Number(p.lines)<=1||e.ctrlKey||e.metaKey)){e.preventDefault();trigger(c.id,'submit')}}}/>}</div>;
  if(c.type==='number'||c.type==='slider')return <div className="native-field"><label htmlFor={id}>{label}{c.type==='slider'&&<span>{text(value)}</span>}</label><input id={id} type={c.type==='slider'?'range':'number'} style={c.type==='slider'?{padding:0}:undefined} value={value==null?'':Number(value)} min={p.minimum==null?undefined:Number(p.minimum)} max={p.maximum==null?undefined:Number(p.maximum)} step={Number(p.step)||'any'} disabled={!interactive} onChange={e=>set(c.id,e.target.value===''?null:Number(e.target.value))} onBlur={()=>{if(c.type==='number')trigger(c.id,'change')}} onPointerUp={()=>{if(c.type==='slider'){trigger(c.id,'change');trigger(c.id,'release')}}} onKeyUp={e=>{if(c.type==='slider'&&e.key.startsWith('Arrow'))trigger(c.id,'change')}}/></div>;
  if(c.type==='checkbox')return <label className="native-check"><input type="checkbox" checked={Boolean(value)} disabled={!interactive} onChange={e=>changed(c.id,e.target.checked)}/>{label}</label>;
  if(['dropdown','radio','checkboxgroup'].includes(c.type)){
   const choices=((p.choices||[])as unknown[]).map(v=>Array.isArray(v)?{label:text(v[0]),value:v[1]}:{label:text(v),value:v});
   const multiple=Boolean(p.multiselect)||c.type==='checkboxgroup';
   const resolve=(s:string)=>choices.find(v=>String(v.value)===s)?.value??s;
   return <div className="native-field"><label htmlFor={id}>{label}</label>{c.type==='radio'?<div className="native-radio">{choices.map(v=><label key={text(v.value)}><input type="radio" name={id} disabled={!interactive} checked={value===v.value} onChange={()=>changed(c.id,v.value)}/>{v.label}</label>)}</div>:<select id={id} multiple={multiple} disabled={!interactive} value={multiple?(Array.isArray(value)?value.map(String):[]):String(value??'')} onChange={e=>changed(c.id,multiple?Array.from(e.target.selectedOptions).map(o=>resolve(o.value)):resolve(e.target.value))}>{!multiple&&<option value="">请选择</option>}{choices.map(v=><option key={text(v.value)} value={String(v.value)}>{v.label}</option>)}</select>}</div>;
  }
  if(['dataframe','dataset'].includes(c.type)){
   const table=value as {headers?:unknown[];data?:unknown[][]}|null;
   const rows=(c.type==='dataset'?p.samples:table?.data||value||[]) as unknown[][];
   const headers=(c.type==='dataset'?p.headers:table?.headers||p.headers||[])as unknown[];
   return <div className="native-field"><label>{label}</label><div className="native-table-scroll"><table><thead><tr>{headers.map((h,i)=><th key={i}>{text(h)}</th>)}</tr></thead><tbody>{Array.isArray(rows)&&rows.map((row,r)=><tr key={r}>{Array.isArray(row)&&row.map((v,col)=><td key={col} onClick={()=>{if(c.type==='dataset'){set(c.id,r);trigger(c.id,'click',{index:r,value:row,selected:true})}else trigger(c.id,'select',{index:[r,col],value:v,selected:true})}}>{interactive&&c.type==='dataframe'?<input aria-label={`${label} ${r+1}行${col+1}列`} value={text(v)} onChange={e=>{const datatype=Array.isArray(p.datatype)?p.datatype[col]:p.datatype;const edited=datatype==='number'||typeof v==='number'?(e.target.value===''?null:Number(e.target.value)):e.target.value;const next=rows.map((line,index)=>index===r?line.map((item,j)=>j===col?edited:item):line);changed(c.id,{headers,data:next})}}/>:text(v)}</td>)}</tr>)}</tbody></table></div></div>;
  }
  if(['file','image','audio','video'].includes(c.type))return <div className="native-field"><label htmlFor={id}>{label}</label>{fileUrl(value)&&!(value as {__upload?:unknown})?.__upload&&<Media value={value} kind={c.type}/>}{interactive&&<input id={id} type="file" disabled={uploading!=null} multiple={p.file_count==='multiple'} accept={c.type==='image'?'image/*':undefined} onChange={e=>upload(c.id,e.target.files,p.file_count==='multiple',c.type)}/>}{interactive&&value!=null&&<button className="text-button" disabled={uploading!=null} onClick={()=>{changed(c.id,null);trigger(c.id,'clear');const input=document.getElementById(id) as HTMLInputElement|null;if(input)input.value=''}}>清除文件</button>}</div>;
  if(c.type==='gallery')return <div className="native-field"><label>{label}</label><div className="native-gallery">{(Array.isArray(value)?value:[]).map((item,i)=>{const entry=item as {image?:unknown;caption?:string};return <button key={i} onClick={()=>trigger(c.id,'select',{index:i,value:item,selected:true})}><Media value={entry.image||item} kind="image"/><span>{entry.caption}</span></button>})}</div></div>;
  if(c.type==='json')return <div className="native-field"><label>{label}</label><pre className="native-json">{JSON.stringify(value,null,2)}</pre></div>;
  if(node.children?.length)return <div>{children()}</div>;
  return <p role="alert" className="inline-error">尚未支持的原控件：{c.type}（{label}）</p>;
 }
 return <section className="native-workbench" data-verify-unit="NativeWorkbench" data-verify-module={moduleIndex} data-verify-state={loading?'loading':busy>0?'running':layout?'ready':'unavailable'} data-verify-busy={busy} data-verify-components={layout?.components.length||0} data-verify-events={layout?.dependencies.length||0}><div className="native-toolbar"><p>{status||'正在载入原工作台…'}</p><button className="text-button" disabled={loading} onClick={connect}>{loading?<Loader2 className="spin" size={15}/>:<RefreshCw size={15}/>}重新载入</button>{busy>0&&<span><Loader2 className="spin" size={15}/>执行中</span>}{savedTask?.module===moduleLabel&&<span className="preset-badge">预设：{savedTask.name} · v{savedTask.version||1}<button className="text-button" onClick={()=>setSavedTask(null)}>退出任务预设</button></span>}</div>{error&&<p role="alert" className="inline-error">{error}</p>}{(error||moduleIndex===7)&&<ServiceDiagnostics/>}{layout&&<RunHistory activeTask={savedTask} onExitPreset={()=>setSavedTask(null)} actions={Object.fromEntries(layout.dependencies.map(d=>[String(d.id),displayLabel(text(components.get(d.targets[0]?.[0]??-1)?.props.value||components.get(d.targets[0]?.[0]??-1)?.props.label||'操作'))]))} module={moduleLabel} revision={revision} labels={Object.fromEntries([...components.values()].map(c=>[String(c.id),displayLabel(text(c.props.label||c.props.value))]))} onRestore={restoreTask} onReconnect={record=>{if(record.session!==session.current){setError('此运行来自其他会话。请在记录中查看结果；当前页面不会接管其会话状态。');return}remember(record.id);void followRun(record.id)}}/>}{loading&&<div className="empty-state"><Loader2 className="spin"/><p>首次载入正在初始化原模块与数据服务。</p></div>}{layout&&render(layout.layout)}</section>
}
function Media({value,kind}:{value:unknown;kind:string}){if(Array.isArray(value))return <>{value.map((v,i)=><Media key={i} value={v} kind={kind}/>)}</>;const url=fileUrl(value);if(!url)return null;return <div>{kind==='image'?<img src={url} alt={(value as {orig_name?:string})?.orig_name||'模块图片'} style={{maxWidth:'100%'}}/>:kind==='audio'?<audio controls src={url}/>:kind==='video'?<video controls src={url} style={{maxWidth:'100%'}}/>:<a href={url} download>下载 {(value as {orig_name?:string})?.orig_name||'文件'}</a>}</div>}
function Html({value}:{value:string}){
 const ref=useRef<HTMLIFrameElement>(null),[height,setHeight]=useState(250);
 useEffect(()=>{
  const listener=(e:MessageEvent)=>{if(e.source===ref.current?.contentWindow&&typeof e.data?.height==='number')setHeight(Math.min(2400,Math.max(40,e.data.height)))};
  window.addEventListener('message',listener);
  ref.current?.contentWindow?.postMessage({type:'native-measure'},'*');
  return()=>window.removeEventListener('message',listener);
 },[]);
 if(!value)return null;
 const srcdoc=resultDocument(value);
 return <iframe ref={ref} className="native-html-result" title="模块结果" sandbox="allow-scripts" onLoad={()=>ref.current?.contentWindow?.postMessage({type:'native-measure'},'*')} srcDoc={srcdoc} style={{border:0,width:'100%',height}}/>;
}
