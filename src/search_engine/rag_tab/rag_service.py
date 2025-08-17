#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG服务模块
基于现有的倒排索引和TF-IDF实现检索增强生成
"""

import json
import requests
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

class RAGService:
    """RAG服务：基于倒排索引的检索增强生成"""
    
    def __init__(self, index_service, ollama_url: str = "http://localhost:11434"):
        """
        初始化RAG服务
        
        Args:
            index_service: 索引服务实例
            ollama_url: Ollama服务URL
        """
        self.index_service = index_service
        self.ollama_url = ollama_url
        self.default_model = "llama3.1:8b"
        
    def check_ollama_connection(self) -> Tuple[bool, str]:
        """检查Ollama连接状态"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                return True, f"✅ Ollama连接成功！\n可用模型: {', '.join(model_names)}"
            else:
                return False, f"❌ Ollama连接失败，状态码: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"❌ Ollama连接失败: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return [self.default_model]
        except:
            return [self.default_model]
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        使用倒排索引检索相关文档
        
        Args:
            query: 查询字符串
            top_k: 返回top_k个文档
            
        Returns:
            List[Tuple[str, float, str]]: (doc_id, score, content)
        """
        try:
            # 使用现有的索引服务进行检索
            results = self.index_service.search(query, top_k)
            print(f"📖 检索到 {len(results)} 个相关文档")
            return results
        except Exception as e:
            print(f"❌ 文档检索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context: str, model: Optional[str] = None) -> str:
        """
        使用Ollama生成回答
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            model: 使用的模型名称
            
        Returns:
            str: 生成的回答
        """
        if model is None:
            model = self.default_model
            
        # 构建提示词
        prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。

上下文信息：
{context}

用户问题：{query}

请用中文回答："""
        
        try:
            # 调用Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "生成回答失败")
            else:
                return f"❌ 生成回答失败，状态码: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ 调用Ollama失败: {str(e)}"
    
    def generate_answer_with_prompt(self, prompt: str, model: Optional[str] = None) -> str:
        """
        直接使用提示词生成回答
        
        Args:
            prompt: 完整的提示词
            model: 使用的模型名称
            
        Returns:
            str: 生成的回答
        """
        if model is None:
            model = self.default_model
            
        try:
            # 调用Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "生成回答失败")
            else:
                return f"❌ 生成回答失败，状态码: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"❌ 调用Ollama失败: {str(e)}"
    
    def rag_query(self, query: str, top_k: int = 5, model: Optional[str] = None) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            model: 使用的模型
            
        Returns:
            Dict: 包含检索结果和生成答案的字典
        """
        start_time = datetime.now()
        
        # 1. 检索相关文档
        retrieved_docs = self.retrieve_documents(query, top_k)
        
        if not retrieved_docs:
            return {
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "answer": "❌ 没有找到相关文档",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        # 2. 构建上下文
        context_parts = []
        for i, (doc_id, score, content) in enumerate(retrieved_docs, 1):
            context_parts.append(f"文档{i} (ID: {doc_id}, 相关度: {score:.4f}):\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. 构建提示词
        prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。

上下文信息：
{context}

用户问题：{query}

请用中文回答："""
        
        # 4. 生成回答
        answer = self.generate_answer_with_prompt(prompt, model)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "answer": answer,
            "processing_time": processing_time,
            "model_used": model or self.default_model,
            "prompt_sent": prompt
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取RAG服务统计信息"""
        index_stats = self.index_service.get_stats()
        ollama_connected, ollama_status = self.check_ollama_connection()
        
        return {
            "ollama_connected": ollama_connected,
            "ollama_status": ollama_status,
            "ollama_url": self.ollama_url,
            "available_models": self.get_available_models(),
            "index_stats": index_stats
        } 