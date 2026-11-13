"""Explicit single-answer RAG validation; never patches clients or runs an Agent loop."""
from pathlib import Path
import httpx
from .common import get,public,text
from .database import ModelService
from ..workbench_access import require_business

class SupplyValidation:
    def __init__(self,session,directory,knowledge):
        self.session,self.directory,self.knowledge=session,Path(directory),knowledge

    def rag(self,index_id,model_id,query):
        require_business()
        query=text(query,'验证问题',1000)
        with self.session() as db:model=public(get(db,ModelService,model_id))
        evidence=self.knowledge.search(index_id,query,5)
        if not evidence['documents']:raise ValueError('没有检索到资料，请先修正资料或索引；未调用模型')
        from ..rag_tab.rag_service import RAGService
        class Reader:
            def search(self,*args,**kwargs):
                return [(d['id'],d['score'],d['content']) for d in evidence['documents']]
        def complete(messages,selected_model):
            if selected_model!=model['model']:raise ValueError('验证模型与选定服务不一致')
            credential=''
            if model['has_credential']:
                key=self.directory/'model-credentials'/(model['id']+'.txt')
                if not key.is_file():raise ValueError('模型凭证文件缺失')
                credential=key.read_text().strip()
            try:
                response=httpx.post(model['base_url']+'/chat/completions',
                    headers={'Authorization':'Bearer '+credential} if credential else {},
                    json={'model':model['model'],'messages':messages,'max_tokens':model['max_tokens'],'stream':False},timeout=30)
                response.raise_for_status()
                answer=response.json()['choices'][0]['message']['content']
                if not isinstance(answer,str) or not answer.strip():raise ValueError('模型未返回文本')
                return answer.replace(credential,'【密钥已隐藏】') if credential else answer
            except (httpx.HTTPError,KeyError,IndexError,TypeError,ValueError):
                raise ValueError('模型验证调用失败，请核对模型服务、协议和凭证') from None
        result=RAGService(Reader(),llm=complete).rag_query(query,model=model['model'],multi_step=False)
        return {'kind':'rag-validation','index':{'id':index_id,'digest':evidence['digest']},
                'model':model,'documents':evidence['documents'],'answer':result['answer'],
                'prompt_sent':result['prompt_sent'],'notice':'这是资料与模型组合的验证结果，不会自动发布为上下文或 Agent。'}
