#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索引擎实现 - 使用索引服务接口
负责召回+排序的核心功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .search_interface import SearchInterface
from ..index_tab import get_index_service
from ..training_tab import CTRModel
from typing import List, Dict, Any, Tuple
import math

class SearchEngine(SearchInterface):
    """搜索引擎实现类 - 使用索引服务接口"""
    
    def __init__(self):
        self.index_service = get_index_service()  # 使用索引服务
        self.current_results = []  # 当前搜索结果
        self.ctr_model = CTRModel()  # CTR模型
        self.load_ctr_model()
    
    def load_ctr_model(self):
        """加载CTR模型"""
        try:
            if self.ctr_model.load_model():
                print("CTR模型加载成功")
            else:
                print("未找到CTR模型文件，将使用默认排序")
        except Exception as e:
            print(f"加载CTR模型失败: {e}")
    
    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """召回阶段：返回初步相关的文档ID列表（按TF-IDF分数粗排）"""
        if not query.strip():
            return []
        # 使用索引服务进行召回
        return self.index_service.search_doc_ids(query.strip(), top_k=top_k)
    
    def rank(self, query: str, doc_ids: List[str], top_k: int = 10) -> List[Tuple[str, float, str]]:
        """排序阶段：对召回的文档ID进行精排，使用CTR模型重新排序"""
        if not query.strip() or not doc_ids:
            return []
        
        # 第一步：使用索引服务获取TF-IDF分数和摘要
        full_results = self.index_service.search(query.strip(), top_k=len(doc_ids))
        
        # 过滤出在doc_ids中的结果
        filtered_results = []
        for doc_id, score, summary in full_results:
            if doc_id in doc_ids:
                filtered_results.append((doc_id, score, summary))
        
        if not filtered_results:
            return []
        
        # 第二步：使用CTR模型重新排序
        if self.ctr_model.is_trained:
            # 有CTR模型时，计算CTR分数并重新排序
            ctr_scores = {}
            for position, (doc_id, tfidf_score, summary) in enumerate(filtered_results, 1):
                ctr_score = self.ctr_model.predict_ctr(query, doc_id, position, tfidf_score, summary)
                ctr_scores[doc_id] = (ctr_score, tfidf_score, summary)
            
            # 按CTR分数排序
            sorted_results = sorted(ctr_scores.items(), key=lambda x: x[1][0], reverse=True)
            
            # 构建最终结果（同时保存TF-IDF和CTR分数）
            results = []
            for doc_id, (ctr_score, tfidf_score, summary) in sorted_results[:top_k]:
                # 返回元组：(doc_id, tfidf_score, ctr_score, summary)
                results.append((doc_id, tfidf_score, ctr_score, summary))
        else:
            # 没有CTR模型时，使用TF-IDF分数排序
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
            
            # 构建最终结果（只返回TF-IDF分数）
            results = []
            for doc_id, tfidf_score, summary in sorted_results[:top_k]:
                # 返回元组：(doc_id, tfidf_score, None, summary)
                results.append((doc_id, tfidf_score, None, summary))
        
        self.current_results = results
        return results
    
    def get_document(self, doc_id: str) -> str:
        """获取文档内容"""
        return self.index_service.get_document(doc_id) or ""
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return self.index_service.get_stats()
    
    def get_current_results(self) -> List[Tuple[str, float, float, str]]:
        """获取当前搜索结果"""
        return self.current_results
    
    # 新增：索引管理方法
    def add_document(self, doc_id: str, content: str) -> bool:
        """添加文档到索引"""
        return self.index_service.add_document(doc_id, content)
    
    def batch_add_documents(self, documents: Dict[str, str]) -> int:
        """批量添加文档"""
        return self.index_service.batch_add_documents(documents)
    
    def save_index(self, filepath: str = "") -> bool:
        """保存索引"""
        return self.index_service.save_index(filepath)
    
    def clear_index(self) -> bool:
        """清空索引"""
        return self.index_service.clear_index()
    
    def get_all_documents(self) -> Dict[str, str]:
        """获取所有文档"""
        return self.index_service.get_all_documents()
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        return self.index_service.delete_document(doc_id) 