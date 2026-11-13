import {readFile} from 'node:fs/promises';
import {spawn} from 'node:child_process';

// One explicit tool, with no caller-selected path, URL or command.
export default function enterpriseContext(pi) {
  pi.registerTool({
    name: 'read_enterprise_context',
    label: '读取岗位上下文',
    description: '读取后台为本次岗位 Agent 固定的企业资料与来源；回答业务问题前先读取。',
    parameters: {type: 'object', properties: {}, additionalProperties: false},
    async execute() {
      const body = JSON.parse(await readFile(process.env.WORKBENCH_CONTEXT_FILE, 'utf8'));
      if(body.retrievals) body.retrievals=body.retrievals.map(({id,name,digest,manifest})=>({id,name,digest,sources:manifest.sources}));
      return {content: [{type: 'text', text: JSON.stringify(body)}], details: {source: 'published-context-package'}};
    },
  });
  if (process.env.WORKBENCH_RETRIEVAL_CLI) pi.registerTool({
    name:'search_enterprise_knowledge', label:'检索岗位资料',
    description:'只检索本次发布包中明确配置的索引版本，返回原文、资料版本和来源。索引标识见 read_enterprise_context 的 retrievals。',
    parameters:{type:'object',properties:{index_id:{type:'string'},query:{type:'string',maxLength:1000},limit:{type:'integer',minimum:1,maximum:10}},required:['index_id','query'],additionalProperties:false},
    async execute(_id,parameters){
      const result=await new Promise((resolve,reject)=>{
        const child=spawn(process.env.WORKBENCH_CONTEXT_PYTHON,[process.env.WORKBENCH_RETRIEVAL_CLI],{stdio:['pipe','pipe','pipe']});
        let output='',error='';
        const timer=setTimeout(()=>{child.kill();reject(Error('检索服务超时'));},30000);
        child.stdout.on('data',part=>{output+=part;if(output.length>2000000){child.kill();reject(Error('检索结果过大'));}});
        child.stderr.on('data',part=>{error=(error+part).slice(-2000);});
        child.on('error',e=>{clearTimeout(timer);reject(e);});
        child.on('close',code=>{clearTimeout(timer);try{const body=JSON.parse(output);if(code||body.error)reject(Error(body.error||error));else resolve(body);}catch(e){reject(e);}});
        child.stdin.end(JSON.stringify(parameters));
      });
      return {content:[{type:'text',text:JSON.stringify(result)}],details:{source:'published-retrieval-index',index_id:result.index_id,digest:result.digest}};
    },
  });

}
