'use client';
import {useEffect,useState} from 'react';
import {installVerification} from '../lib/verification';
import {Database,Search,Workflow,Network,Blocks,Image,ChartNoAxesCombined,Activity,PanelLeftClose,BookOpen,ArrowRight} from 'lucide-react';
import NativeWorkbench from '../components/native-workbench';
const modules=[
 {title:'离线索引构建',description:'文档、倒排索引与知识图谱',icon:Database},
 {title:'在线召回排序',description:'召回、排序与点击反馈',icon:Search},
 {title:'RAG',description:'检索增强与回答依据',icon:Workflow},
 {title:'OAG 图谱推理',description:'实体、关系与多步游走',icon:Network},
 {title:'上下文工程',description:'MCP 工具与上下文装配',icon:Blocks},
 {title:'多模态系统',description:'图片检索、生成与 GUI Agent',icon:Image},
 {title:'数据回收训练',description:'模型训练、评估与优化',icon:ChartNoAxesCombined},
 {title:'系统监控',description:'运行状态与样本指标',icon:Activity},
];
export default function Page(){
 const [selected,setSelected]=useState(0),[rail,setRail]=useState(true);
 useEffect(installVerification,[]);
 return <div className={`shell ${rail?'':'compact'}`}><aside className="sidebar"><a className="brand" href="/"><span className="brand-icon"><Workflow size={22}/></span><span>test bed<small>企业上下文供应</small></span></a><div className="workspace-label">企业上下文工程 <span>本地</span></div><nav>{modules.map(({title,icon:Icon},i)=><button key={title} data-module-index={i} aria-pressed={i===selected} className={i===selected?'active':''} onClick={()=>setSelected(i)}><Icon size={18}/><span>{title}</span></button>)}</nav><div className="sidebar-note"><span className="eyebrow">面向业务任务</span><p>配置与验证<br/>Agent 所需的企业上下文。</p><a href="https://github.com/tylerelyt/test_bed/blob/main/README.md" target="_blank" rel="noreferrer"><BookOpen size={15}/>项目文档<ArrowRight size={14}/></a></div></aside><main><header><div className="breadcrumbs"><button className="icon-button collapse" aria-label="切换侧栏" onClick={()=>setRail(!rail)}><PanelLeftClose size={18}/></button><span>FDE 工作台</span><span>/</span><strong>{modules[selected].title}</strong></div></header><div className="page-content"><div className="page-heading"><div><div className="eyebrow">企业上下文供应后台</div><h1>{modules[selected].title}</h1><p>{modules[selected].description}</p></div></div><NativeWorkbench moduleIndex={selected}/></div><footer>TEST BED <span>企业上下文工程工作台</span><a href="https://github.com/tylerelyt/test_bed" target="_blank" rel="noreferrer">GitHub ↗</a></footer></main></div>
}
