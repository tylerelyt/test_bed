#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG服务模块
基于现有的倒排索引和TF-IDF实现检索增强生成
"""

import json
import re
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
    
    def _react_reasoning(self, query: str, model: Optional[str], retrieval_enabled: bool, top_k: int = 5, max_steps: int = 5) -> Tuple[str, str]:
        """
        ReAct风格多步推理：Thought -> Action(SEARCH/FINISH) -> Observation，循环直到FINISH或步数上限。
        返回 (final_answer, trace_text)
        """
        if model is None:
            model = self.default_model
        
        trace_lines: List[str] = []
        observations: List[str] = []

        tool_desc = (
            "你可以使用一个工具：SEARCH(\"查询词\")，它会返回与查询词最相关的文档片段列表。"
        )
        format_instructions = (
            "每轮请严格输出以下格式中的一行Action，便于解析：\n"
            "Thought: <你的简短思考>\n"
            "Action: SEARCH(\"<查询词>\") 或 Action: FINISH(\"<最终答案>\")\n"
            "不要输出其他多余内容。"
        )

        search_pattern = re.compile(r"Action:\s*SEARCH\(\"([\s\S]*?)\"\)")
        finish_pattern = re.compile(r"Action:\s*FINISH\(\"([\s\S]*?)\"\)")

        scratchpad = ""
        for step in range(1, max_steps + 1):
            prompt = (
                f"你是一个会逐步思考并合理使用工具的助手。\n"
                f"用户问题：{query}\n\n"
                f"工具说明：{tool_desc}\n"
                f"注意：{'当前禁止使用SEARCH工具。' if not retrieval_enabled else '可以使用SEARCH工具。'}\n\n"
                f"历史推理：\n{scratchpad}\n\n"
                f"请开始第{step}步。\n{format_instructions}"
            )
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=60
                )
                if resp.status_code != 200:
                    trace_lines.append(f"系统: 模型调用失败，状态码 {resp.status_code}")
                    break
                text = resp.json().get("response", "").strip()
            except requests.exceptions.RequestException as e:
                trace_lines.append(f"系统: 模型调用异常 {str(e)}")
                break

            # 记录模型输出
            trace_lines.append(f"Step {step} 模型输出:\n{text}")

            # 解析动作
            finish_match = finish_pattern.search(text)
            if finish_match:
                final_answer = finish_match.group(1)
                trace_lines.append("Action: FINISH")
                return final_answer, "\n\n".join(trace_lines)

            search_match = search_pattern.search(text)
            if search_match:
                search_query = search_match.group(1).strip()
                if retrieval_enabled:
                    # 执行检索
                    docs = self.retrieve_documents(search_query, top_k=top_k)
                    if not docs:
                        observation = "未检索到相关文档。"
                    else:
                        # 只取前3条，避免上下文过长
                        obs_parts = []
                        for i, (doc_id, score, content) in enumerate(docs[:3], 1):
                            snippet = content[:400]
                            obs_parts.append(f"[{i}] id={doc_id} score={score:.4f} snippet={snippet}")
                        observation = "\n".join(obs_parts)
                    observations.append(observation)
                    trace_lines.append(f"Observation:\n{observation}")
                    scratchpad += f"Thought/Action(SEARCH): {search_query}\nObservation: {observation}\n\n"
                    continue
                else:
                    observation = "SEARCH工具被禁用。请直接FINISH。"
                    observations.append(observation)
                    trace_lines.append(f"Observation:\n{observation}")
                    scratchpad += f"Action(SEARCH被拒): {search_query}\nObservation: {observation}\n\n"
                    continue

            # 若无法解析动作，提示并继续下一步
            notice = "未解析到有效的Action，请按格式输出。"
            trace_lines.append(f"系统: {notice}")
            scratchpad += f"系统提示: {notice}\n\n"

        # 未显式FINISH时，尝试让模型基于观察做最终总结
        summary_context = "\n\n".join(observations[-3:]) if observations else ""
        final_prompt = (
            f"请基于以下观察与你已有的推理，给出问题的最终中文答案。若观察为空，请直接根据常识作答。\n\n"
            f"问题：{query}\n\n"
            f"观察：\n{summary_context}\n\n"
            f"请直接输出答案，不要再输出思维过程。"
        )
        try:
            final_resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": final_prompt, "stream": False},
                timeout=60
            )
            if final_resp.status_code != 200:
                answer = f"❌ 多步推理总结失败，状态码: {final_resp.status_code}"
            else:
                answer = final_resp.json().get("response", "生成回答失败")
        except requests.exceptions.RequestException as e:
            answer = f"❌ 调用Ollama失败: {str(e)}"
        trace_lines.append("系统: 未检测到FINISH，已进行自动总结。")
        return answer, "\n\n".join(trace_lines)

    def rag_query(self, query: str, top_k: int = 5, model: Optional[str] = None, retrieval_enabled: bool = True, multi_step: bool = False) -> Dict[str, Any]:
        """
        执行RAG查询
        
        Args:
            query: 用户查询
            top_k: 检索文档数量
            model: 使用的模型
            retrieval_enabled: 是否开启检索增强
            multi_step: 是否开启多步推理
            
        Returns:
            Dict: 包含检索结果和生成答案的字典
        """
        start_time = datetime.now()
        
        # 如果关闭检索与多步推理，则直接问 LLM（无上下文直连）
        if not retrieval_enabled and not multi_step:
            direct_prompt = f"请用中文回答用户问题：\n\n问题：{query}"
            answer = self.generate_answer_with_prompt(direct_prompt, model)
            return {
                "query": query,
                "retrieved_docs": [],
                "context": "",
                "answer": answer,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "model_used": model or self.default_model,
                "prompt_sent": direct_prompt
            }

        # 1) 若开启检索，先检索并构建上下文；否则上下文为空
        retrieved_docs = []
        context = ""
        if retrieval_enabled:
            retrieved_docs = self.retrieve_documents(query, top_k)
            # 即使未检索到文档，也继续，让模型直接回答或多步推理
            if retrieved_docs:
                context_parts = []
                for i, (doc_id, score, content) in enumerate(retrieved_docs, 1):
                    context_parts.append(f"文档{i} (ID: {doc_id}, 相关度: {score:.4f}):\n{content}")
                context = "\n\n".join(context_parts)

        # 2) 生成回答：多步推理优先，否则普通单步回答
        if multi_step:
            answer, trace_text = self._react_reasoning(
                query=query,
                model=model,
                retrieval_enabled=retrieval_enabled,
                top_k=top_k
            )
            prompt_used = trace_text  # 将完整推理轨迹回显
        else:
            # 构建标准提示
            prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。
            
上下文信息：
{context}
            
用户问题：{query}
            
请用中文回答："""
            answer = self.generate_answer_with_prompt(prompt, model)
            prompt_used = prompt
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "answer": answer,
            "processing_time": processing_time,
            "model_used": model or self.default_model,
            "prompt_sent": prompt_used if prompt_used is not None else "多步推理（内部多提示）"
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