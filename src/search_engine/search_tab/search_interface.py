#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索接口 - 定义核心功能
使用接口解耦，不依赖具体的服务调用
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class SearchInterface(ABC):
    """搜索接口抽象类"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> List[str]:
        """
        召回阶段：返回初步相关的文档ID列表
        Args:
            query: 搜索查询
            top_k: 返回文档ID数量
        Returns:
            List of doc_id
        """
        pass

    @abstractmethod
    def rank(self, query: str, doc_ids: List[str], top_k: int = 10) -> List[Tuple[str, float, float, str]]:
        """
        排序阶段：对召回的文档ID进行精排，返回最终结果
        Args:
            query: 搜索查询
            doc_ids: 召回的文档ID列表
            top_k: 返回结果数量
        Returns:
            List of (doc_id, tfidf_score, ctr_score, summary)
        """
        pass

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, float, str]]:
        """
        可选：一体化检索（默认实现为retrieve+rank）
        """
        doc_ids = self.retrieve(query, top_k=top_k*2)
        return self.rank(query, doc_ids, top_k=top_k)

    @abstractmethod
    def get_document(self, doc_id: str) -> str:
        """
        获取文档内容
        Args:
            doc_id: 文档ID
        Returns:
            文档内容
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        Returns:
            统计信息字典
        """
        pass

class CTRInterface(ABC):
    """CTR数据收集接口抽象类"""
    
    @abstractmethod
    def record_impression(self, query: str, doc_id: str, position: int, score: float, summary: str):
        """
        记录曝光
        Args:
            query: 搜索查询
            doc_id: 文档ID
            position: 位置
            score: 分数
            summary: 摘要
        """
        pass
    
    @abstractmethod
    def record_click(self, query: str, doc_id: str, position: int):
        """
        记录点击
        Args:
            query: 搜索查询
            doc_id: 文档ID
            position: 位置
        """
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取历史记录
        Returns:
            历史记录列表
        """
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """
        导出CTR数据
        Returns:
            CTR数据字典
        """
        pass 