#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 LLM 的命名实体识别服务：从文档中提取实体和关系，构建知识图谱。
仅支持 OpenAI 兼容 API（如 DashScope）。
"""

import json
import os
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import re

# 尝试导入 OpenAI 客户端
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class NERService:
    """基于 LLM 的命名实体识别服务（仅 OpenAI 兼容 API）。"""

    @classmethod
    def try_create(
        cls,
        api_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> Optional["NERService"]:
        """在缺少 API 密钥等配置时返回 ``None``，避免调用方用裸 ``try/except`` 分支。"""
        try:
            return cls(
                api_type=api_type,
                api_key=api_key,
                base_url=base_url,
                default_model=default_model,
            )
        except ValueError:
            return None

    def __init__(self,
                 api_type: str = "openai",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: Optional[str] = None):
        """
        初始化 NER 服务。

        Args:
            api_type: 固定为 "openai"（保留参数兼容）
            api_key: API 密钥
            base_url: API 基础 URL
            default_model: 默认模型名称
        """
        if not HAS_OPENAI:
            raise ValueError("需要安装 openai 包: pip install openai")

        self.api_type = "openai"
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not self.api_key:
            raise ValueError("需要设置 api_key 或环境变量 DASHSCOPE_API_KEY/OPENAI_API_KEY")

        self.openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_model = default_model or os.environ.get("LLM_MODEL", "qwen-plus")
        
    def extract_entities_and_relations(
        self,
        text: str,
        model: Optional[str] = None,
        ontology_predicates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        使用LLM提取实体和关系
        
        Args:
            text: 待分析的文本
            model: 使用的模型
            
        Returns:
            Dict: 包含实体和关系的字典
        """
        if model is None:
            model = self.default_model
            
        ontology_predicates = [p.strip() for p in (ontology_predicates or []) if str(p).strip()]
        if ontology_predicates:
            allowed_predicates_text = "、".join(sorted(set(ontology_predicates)))
            predicate_instruction = (
                f"关系谓词必须只从以下集合中选择：{allowed_predicates_text}。\n"
                "如果文本表达与集合不完全一致，请映射到最接近的同义谓词；无法映射则不要输出该关系。"
            )
        else:
            predicate_instruction = "关系类型包括：属于、位于、开发、使用、相关、影响等。"

        # 构建NER提示词
        prompt = f"""请从以下文本中提取实体和关系，返回JSON格式的结果。

文本：{text}

请按照以下格式返回：
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "description": "实体描述"
        }}
    ],
    "relations": [
        {{
            "subject": "主体实体",
            "predicate": "关系类型",
            "object": "客体实体",
            "description": "关系描述"
        }}
    ]
}}

实体类型包括：人物、地点、组织、概念、技术、产品、事件等。
{predicate_instruction}

请确保返回的是有效的JSON格式。"""
        
        try:
            print(f"🔍 [NER] 开始处理文本，长度: {len(text)}")
            print(f"🤖 [NER] 使用模型: {model}")
            print(f"🔧 [NER] API类型: {self.api_type}")
            
            llm_response = self._call_openai_api(prompt, model)
            
            # 检查API调用是否出错
            if llm_response.startswith("ERROR:"):
                return {"error": llm_response}
            
            print(f"✅ [NER] LLM响应成功，响应长度: {len(llm_response)}")
            print(f"📝 [NER] LLM原始响应: {llm_response[:500]}...")
            
            # 解析JSON响应
            parsed_result = self._parse_ner_response(llm_response)
            
            if "error" in parsed_result:
                print(f"❌ [NER] 解析错误: {parsed_result['error']}")
            else:
                entities_count = len(parsed_result.get("entities", []))
                relations_count = len(parsed_result.get("relations", []))
                print(f"✅ [NER] 解析成功: {entities_count}个实体, {relations_count}个关系")
            
            return parsed_result
                
        except Exception as e:
            error_msg = f"NER提取失败: {str(e)}"
            print(f"❌ [NER] 异常: {error_msg}")
            import traceback
            print(f"📝 [NER] 详细错误: {traceback.format_exc()}")
            return {"error": error_msg}
    
    def _call_openai_api(self, prompt: str, model: str) -> str:
        """调用OpenAI兼容API"""
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            error_msg = f"OpenAI API调用异常: {str(e)}"
            print(f"❌ [NER] {error_msg}")
            return f"ERROR: {error_msg}"
    
    def _parse_ner_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的NER响应
        
        Args:
            response: LLM的响应文本
            
        Returns:
            Dict: 解析后的实体和关系
        """
        try:
            print(f"🔍 [NER-Parse] 开始解析响应，长度: {len(response)}")
            
            # 尝试直接解析JSON
            if response.strip().startswith('{'):
                print(f"✅ [NER-Parse] 检测到JSON格式，尝试直接解析")
                result = json.loads(response)
                print(f"✅ [NER-Parse] 直接解析成功")
                return result
            
            # 查找JSON部分
            print(f"🔍 [NER-Parse] 在响应中查找JSON部分")
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"✅ [NER-Parse] 找到JSON部分: {json_str[:200]}...")
                result = json.loads(json_str)
                print(f"✅ [NER-Parse] JSON解析成功")
                return result
            
            # 如果没有找到JSON，返回空结果
            print(f"⚠️  [NER-Parse] 未找到JSON格式，返回空结果")
            return {"entities": [], "relations": []}
            
        except json.JSONDecodeError as e:
            print(f"❌ [NER-Parse] JSON解析失败: {str(e)}")
            print(f"📝 [NER-Parse] 尝试备用解析方法")
            # JSON解析失败，尝试简单的文本解析
            return self._fallback_parse(response)
        except Exception as e:
            print(f"❌ [NER-Parse] 解析过程异常: {str(e)}")
            return {"error": f"解析失败: {str(e)}"}
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """
        备用解析方法，当JSON解析失败时使用
        
        Args:
            response: LLM的响应文本
            
        Returns:
            Dict: 解析后的实体和关系
        """
        entities = []
        relations = []
        
        # 简单的文本解析逻辑
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if '实体' in line or 'Entity' in line:
                # 尝试提取实体信息
                if ':' in line:
                    entity_info = line.split(':', 1)[1].strip()
                    entities.append({
                        "name": entity_info,
                        "type": "未分类",
                        "description": ""
                    })
            elif '关系' in line or 'Relation' in line:
                # 尝试提取关系信息
                if ':' in line:
                    relation_info = line.split(':', 1)[1].strip()
                    relations.append({
                        "subject": relation_info,
                        "predicate": "相关",
                        "object": "",
                        "description": ""
                    })
        
        return {"entities": entities, "relations": relations}
    
    def extract_from_document(self, doc_id: str, content: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        从单个文档提取实体和关系
        
        Args:
            doc_id: 文档ID
            content: 文档内容
            model: 使用的模型
            
        Returns:
            Dict: 包含文档ID和提取结果的字典
        """
        # 清理文档内容，去除多余的空白字符
        cleaned_content = content.strip()
        if not cleaned_content:
            return {"doc_id": doc_id, "error": "文档内容为空"}
        
        # 如果文档过长，进行分段处理
        max_length = 2000
        if len(cleaned_content) > max_length:
            # 分段处理
            chunks = [cleaned_content[i:i+max_length] for i in range(0, len(cleaned_content), max_length)]
            
            all_entities = []
            all_relations = []
            
            for i, chunk in enumerate(chunks):
                print(f"处理文档 {doc_id} 的第 {i+1}/{len(chunks)} 段")
                chunk_result = self.extract_entities_and_relations(chunk, model)
                
                if "error" not in chunk_result:
                    all_entities.extend(chunk_result.get("entities", []))
                    all_relations.extend(chunk_result.get("relations", []))
            
            # 去重和合并
            entities = self._deduplicate_entities(all_entities)
            relations = self._deduplicate_relations(all_relations)
            
        else:
            # 直接处理
            result = self.extract_entities_and_relations(cleaned_content, model)
            if "error" in result:
                return {"doc_id": doc_id, "error": result["error"]}
            
            entities = result.get("entities", [])
            relations = result.get("relations", [])
        
        return {
            "doc_id": doc_id,
            "entities": entities,
            "relations": relations,
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重实体"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.get("name", "").lower(), entity.get("type", ""))
            if key not in seen and entity.get("name"):
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            key = (
                relation.get("subject", "").lower(),
                relation.get("predicate", "").lower(),
                relation.get("object", "").lower()
            )
            if key not in seen and all(relation.get(k) for k in ["subject", "predicate", "object"]):
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def batch_extract_from_documents(self, documents: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        """
        批量从文档提取实体和关系
        
        Args:
            documents: 文档字典 {doc_id: content}
            model: 使用的模型
            
        Returns:
            Dict: 批量提取结果
        """
        results = {}
        total_docs = len(documents)
        
        print(f"开始批量NER提取，共 {total_docs} 个文档")
        
        for i, (doc_id, content) in enumerate(documents.items(), 1):
            print(f"📄 [Batch-NER] 处理文档 {i}/{total_docs}: {doc_id}")
            print(f"📄 [Batch-NER] 文档内容长度: {len(content)}")
            print(f"📄 [Batch-NER] 文档内容预览: {content[:200]}...")
            
            result = self.extract_from_document(doc_id, content, model)
            
            if "error" in result:
                print(f"❌ [Batch-NER] 文档 {doc_id} 处理失败: {result['error']}")
            else:
                entities_count = len(result.get("entities", []))
                relations_count = len(result.get("relations", []))
                print(f"✅ [Batch-NER] 文档 {doc_id} 处理成功: {entities_count}个实体, {relations_count}个关系")
                
            results[doc_id] = result
        
        print(f"批量NER提取完成")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取NER服务统计信息"""
        stats = {
            "service_name": "NER Service",
            "api_type": self.api_type,
            "default_model": self.default_model,
            "supported_entity_types": ["人物", "地点", "组织", "概念", "技术", "产品", "事件"],
            "supported_relation_types": ["属于", "位于", "开发", "使用", "相关", "影响"]
        }
        
        stats["base_url"] = self.base_url
        stats["has_api_key"] = bool(self.api_key)
        
        return stats 