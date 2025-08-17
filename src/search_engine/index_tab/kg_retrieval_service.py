#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱检索服务
简化版本：只支持查询实体及其相关实体和关系
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from .ner_service import NERService
from .knowledge_graph import KnowledgeGraph

class KGRetrievalService:
    """知识图谱检索服务 - 简化版本"""
    
    def __init__(self, 
                 graph_file: str = "models/knowledge_graph.pkl",
                 api_type: str = "ollama",
                 ollama_url: str = "http://localhost:11434",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: Optional[str] = None):
        """
        初始化知识图谱检索服务
        
        Args:
            graph_file: 知识图谱文件路径
            api_type: API类型 ("ollama" 或 "openai")
            ollama_url: Ollama服务URL
            api_key: API密钥 (当api_type为openai时使用)
            base_url: API基础URL (当api_type为openai时使用)
            default_model: 默认模型名称
        """
        self.ner_service = NERService(
            api_type=api_type,
            ollama_url=ollama_url,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model
        )
        self.knowledge_graph = KnowledgeGraph(graph_file)
        self.graph_file = graph_file
        self.is_graph_built = self._check_graph_exists()
    
    def _check_graph_exists(self) -> bool:
        """检查知识图谱是否存在"""
        return (os.path.exists(self.graph_file) and 
                self.knowledge_graph.graph.number_of_nodes() > 0)
    
    def build_knowledge_graph(self, documents: Dict[str, str], 
                            model: Optional[str] = None) -> Dict[str, Any]:
        """
        构建知识图谱
        
        Args:
            documents: 文档字典 {doc_id: content}
            model: 使用的LLM模型
            
        Returns:
            Dict: 构建结果
        """
        start_time = datetime.now()
        
        print(f"开始构建知识图谱，共 {len(documents)} 个文档")
        
        # 1. 批量NER提取
        print("步骤1: 批量NER提取...")
        ner_results = self.ner_service.batch_extract_from_documents(documents, model)
        
        # 2. 构建知识图谱
        print("步骤2: 构建知识图谱...")
        self.knowledge_graph.build_from_ner_results(ner_results)
        
        # 3. 保存图谱
        print("步骤3: 保存知识图谱...")
        self.knowledge_graph.save_graph()
        
        # 4. 更新状态
        self.is_graph_built = True
        
        build_time = (datetime.now() - start_time).total_seconds()
        
        stats = self.knowledge_graph.get_stats()
        
        return {
            "success": True,
            "build_time": build_time,
            "processed_documents": len(documents),
            "stats": stats,
            "message": f"知识图谱构建完成！共构建 {stats['entity_count']} 个实体，{stats['relation_count']} 条关系"
        }
    
    def rebuild_knowledge_graph(self, documents: Dict[str, str], 
                               model: Optional[str] = None) -> Dict[str, Any]:
        """
        重新构建知识图谱
        
        Args:
            documents: 文档字典 {doc_id: content}
            model: 使用的LLM模型
            
        Returns:
            Dict: 构建结果
        """
        # 清空现有图谱
        self.knowledge_graph.clear_graph()
        self.is_graph_built = False
        
        # 重新构建
        return self.build_knowledge_graph(documents, model)
    
    def query_entity_relations(self, entity_name: str) -> Dict[str, Any]:
        """
        查询实体的相关实体和关系（核心功能）
        
        Args:
            entity_name: 实体名称
            
        Returns:
            Dict: 实体关系信息
        """
        if not self.is_graph_built:
            return {
                "error": "知识图谱未构建",
                "entity": entity_name,
                "exists": False
            }
        
        if not self.knowledge_graph.graph.has_node(entity_name):
            return {
                "error": "实体不存在",
                "entity": entity_name,
                "exists": False
            }
        
        # 获取实体基本信息
        node_data = self.knowledge_graph.graph.nodes[entity_name]
        
        # 获取直接关系
        relations = self.knowledge_graph.get_entity_relations(entity_name)
        
        # 获取相关实体（距离≤2）
        related_entities = self.knowledge_graph.get_related_entities(entity_name, max_distance=2)
        
        # 获取包含该实体的文档
        documents = self.knowledge_graph.get_entity_documents(entity_name)
        
        return {
            "entity": entity_name,
            "exists": True,
            "type": node_data.get("entity_type", "未分类"),
            "description": node_data.get("description", ""),
            "doc_count": node_data.get("doc_count", 0),
            "documents": documents,
            "relations": relations,
            "related_entities": related_entities,
            "created_at": node_data.get("created_at", "")
        }
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索实体（用于找到要查询的实体）
        
        Args:
            query: 搜索查询
            limit: 返回数量限制
            
        Returns:
            List[Dict]: 实体列表
        """
        if not self.is_graph_built:
            return []
        
        return self.knowledge_graph.search_entities(query, limit)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图谱统计信息
        
        Returns:
            Dict: 统计信息
        """
        base_stats = self.knowledge_graph.get_stats()
        
        return {
            **base_stats,
            "is_graph_built": self.is_graph_built,
            "graph_file": self.graph_file,
            "ner_service": self.ner_service.get_stats()
        }
    
    def export_graph(self, format: str = "json") -> Tuple[Optional[str], str]:
        """
        导出知识图谱
        
        Args:
            format: 导出格式 (json)
            
        Returns:
            Tuple[Optional[str], str]: (文件路径, 状态消息)
        """
        if not self.is_graph_built:
            return None, "知识图谱未构建"
        
        try:
            graph_data = self.knowledge_graph.export_graph_data()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"knowledge_graph_export_{timestamp}.json"
            filepath = os.path.join("data", filename)
            
            os.makedirs("data", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            return filepath, f"知识图谱导出成功！文件: {filename}"
            
        except Exception as e:
            return None, f"导出失败: {str(e)}"
    
    def clear_graph(self) -> str:
        """
        清空知识图谱
        
        Returns:
            str: 状态消息
        """
        self.knowledge_graph.clear_graph()
        self.is_graph_built = False
        
        # 删除图谱文件
        if os.path.exists(self.graph_file):
            try:
                os.remove(self.graph_file)
                return "知识图谱已清空，文件已删除"
            except Exception as e:
                return f"知识图谱已清空，但删除文件失败: {str(e)}"
        
        return "知识图谱已清空"
    
    def get_graph_visualization_data(self) -> Dict[str, Any]:
        """
        获取图谱可视化数据
        
        Returns:
            Dict: 可视化数据
        """
        if not self.is_graph_built:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        # 获取节点数据
        for node in self.knowledge_graph.graph.nodes():
            node_data = self.knowledge_graph.graph.nodes[node]
            nodes.append({
                "id": node,
                "label": node,
                "type": node_data.get("entity_type", "未分类"),
                "description": node_data.get("description", ""),
                "doc_count": node_data.get("doc_count", 0)
            })
        
        # 获取边数据
        for source, target, edge_data in self.knowledge_graph.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "predicate": edge_data.get("predicate", ""),
                "description": edge_data.get("description", "")
            })
        
        return {"nodes": nodes, "edges": edges} 