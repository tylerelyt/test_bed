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
        # 强制只使用预置知识图谱：忽略磁盘上的pickle，清空后尝试加载预置文件
        self.knowledge_graph.clear_graph()
        loaded = False
        # 1) 优先尝试 OpenKG 三元组数据
        openkg_path = os.path.join("data", "openkg_triples.tsv")
        if os.path.exists(openkg_path):
            loaded = self.knowledge_graph.load_from_openkg_triples(openkg_path, max_triples=3000)
            if loaded:
                print(f"✅ 已加载OpenKG三元组: {openkg_path}")
        # 2) 回退到项目内置 JSON 预置图谱
        if not loaded:
            preloaded_kg_path = os.path.join("data", "preloaded_knowledge_graph.json")
            if os.path.exists(preloaded_kg_path):
                loaded = self.knowledge_graph.load_from_json_file(preloaded_kg_path)
                if loaded:
                    print(f"✅ 已加载预置知识图谱: {preloaded_kg_path}")
        if not loaded:
            print("⚠️ 未找到可用的预置知识图谱文件，图谱为空")
        self.graph_file = graph_file
        # 仅根据预置文件加载结果设置状态
        self.is_graph_built = bool(loaded)
    
    def _check_graph_exists(self) -> bool:
        """检查知识图谱是否存在"""
        return (os.path.exists(self.graph_file) and 
                self.knowledge_graph.graph.number_of_nodes() > 0)
    
    def build_knowledge_graph(self, documents: Dict[str, str], 
                            model: Optional[str] = None) -> Dict[str, Any]:
        """当前版本暂不支持在线自建图谱，请使用预置知识图谱"""
        return {"error": "KG building is disabled in this version. A preloaded KG is used if available."}
    
    def rebuild_knowledge_graph(self, documents: Dict[str, str], 
                               model: Optional[str] = None) -> Dict[str, Any]:
        """当前版本暂不提供重建操作"""
        return {"error": "Rebuild is disabled. Use the preloaded KG or clear and reload the file."}
    
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
        return "Operation disabled: dynamic modifications are not allowed. Preloaded KG is read-only."
    
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