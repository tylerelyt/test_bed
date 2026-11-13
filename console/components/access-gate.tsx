'use client';
import {createContext,useContext,useEffect,useState,ReactNode} from 'react';
type Identity={id:string;project:string;role:string;business_access:boolean};
const Access=createContext<{identity:Identity;logout:()=>void}|null>(null);
export const useWorkbenchAccess=()=>useContext(Access);
export function AccessIdentity(){
  const access=useContext(Access);
  return access?<span className="access-identity"><span>{access.identity.id} · {access.identity.project}</span><button className="text-button" onClick={access.logout}>退出</button></span>:null;
}
export default function AccessGate({children}:{children:ReactNode}){
  const [who,setWho]=useState<Identity|null>(null),[ready,setReady]=useState(false),[token,setToken]=useState(''),[error,setError]=useState(''),[pending,setPending]=useState(false);
  useEffect(()=>{
    let alive=true;
    const check=()=>fetch('/api/access',{cache:'no-store'}).then(r=>r.json()).then(data=>{if(alive){setWho(data.authenticated&&data.identity.role==='operator'?data.identity:null);setError(data.error||(data.authenticated&&data.identity.role!=='operator'?'此凭证仅用于 Agent 接入':''));setReady(true)}}).catch(()=>{if(alive){setError('身份服务未连接，请检查工作台服务');setReady(true)}});
    void check();const timer=setInterval(check,60000);
    const revoked=()=>{setWho(null);setError('访问已失效，请重新登录')};window.addEventListener('workbench-auth-required',revoked);
    return()=>{alive=false;clearInterval(timer);window.removeEventListener('workbench-auth-required',revoked)};
  },[]);
  async function login(event:React.FormEvent){
    event.preventDefault();setPending(true);setError('');
    try{const response=await fetch('/api/access',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});const data=await response.json();if(!response.ok)throw Error(data.error);setToken('');setWho(data.identity)}catch(e){setError((e as Error).message)}finally{setPending(false)}
  }
  async function logout(){await fetch('/api/access',{method:'DELETE'});setWho(null);setToken('');sessionStorage.removeItem('workbench-session');sessionStorage.removeItem('workbench-active-runs')}
  if(!ready)return <main className="access-gate"><p>正在核对访问身份…</p></main>;
  if(!who)return <main className="access-gate"><form onSubmit={login}><p className="eyebrow">企业上下文供应后台</p><h1>进入工作台</h1><p>使用分配给你的操作员凭证。项目范围和操作权限由服务端核对。</p><label htmlFor="access-token">访问凭证</label><input id="access-token" type="password" autoComplete="off" required value={token} onChange={e=>setToken(e.target.value)}/>{error&&<p role="alert" className="inline-error">{error}</p>}<button className="primary" disabled={pending||!token.trim()}>{pending?'正在核对…':'进入工作台'}</button><p className="run-help">首次启动的本机管理员凭证保存在 .workbench/bootstrap-access.txt；团队和 Agent 凭证由部署维护者分配。</p></form></main>;
  return <Access.Provider value={{identity:who,logout:()=>void logout()}}><div key={who.id+':'+who.project+':'+who.role+':'+who.business_access}>{children}</div></Access.Provider>;
}
