#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据访问层工具函数
提供简化的数据访问接口，减少业务层对数据服务的直接依赖
"""

from typing import List, Dict, Any, Optional
import pandas as pd
# 延迟导入，避免循环导入
def get_data_service():
    from .service_manager import service_manager
    return service_manager.data_service


def record_search_impression(query: str, doc_id: str, position: int, 
                           score: float, summary: str, request_id: str) -> Dict[str, Any]:
    """记录搜索展示事件"""
    return get_data_service().record_impression(query, doc_id, position, score, summary, request_id)


def record_document_click(doc_id: str, request_id: str) -> bool:
    """记录文档点击事件"""
    return get_data_service().record_click(doc_id, request_id)


def get_ctr_samples(request_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """获取CTR样本数据"""
    if request_id:
        return get_data_service().get_samples_by_request(request_id)
    return get_data_service().get_all_samples()


def get_ctr_dataframe(request_id: Optional[str] = None) -> pd.DataFrame:
    """获取CTR样本DataFrame"""
    return get_data_service().get_samples_dataframe(request_id)


def get_data_statistics() -> Dict[str, Any]:
    """获取数据统计信息"""
    return get_data_service().get_stats()


def clear_all_data():
    """清空所有CTR数据"""
    get_data_service().clear_data()


def export_ctr_data(filepath: str) -> bool:
    """导出CTR数据"""
    return get_data_service().export_data(filepath)


def import_ctr_data(filepath: str) -> bool:
    """导入CTR数据"""
    return get_data_service().import_data(filepath)


def save_data():
    """保存数据到文件"""
    get_data_service()._save_data()


# 数据验证工具
def validate_search_params(query: str, doc_id: str, position: int, score: float) -> List[str]:
    """验证搜索参数"""
    errors = []
    
    if not query or not query.strip():
        errors.append("查询不能为空")
    
    if not doc_id or not doc_id.strip():
        errors.append("文档ID不能为空")
    
    if position < 1:
        errors.append("位置必须大于0")
    
    if score < 0:
        errors.append("分数不能为负数")
    
    return errors


def validate_click_params(doc_id: str, request_id: str) -> List[str]:
    """验证点击参数"""
    errors = []
    
    if not doc_id or not doc_id.strip():
        errors.append("文档ID不能为空")
    
    if not request_id or not request_id.strip():
        errors.append("请求ID不能为空")
    
    return errors


# 数据分析工具
def analyze_click_patterns() -> Dict[str, Any]:
    """分析点击模式"""
    samples = get_ctr_samples()
    if not samples:
        return {'error': '没有数据可分析'}
    
    df = pd.DataFrame(samples)
    
    analysis = {
        'total_impressions': len(df),
        'total_clicks': df['clicked'].sum() if 'clicked' in df.columns else 0,
        'overall_ctr': 0.0,
        'position_analysis': {},
        'query_analysis': {},
        'doc_analysis': {}
    }
    
    if analysis['total_impressions'] > 0:
        analysis['overall_ctr'] = analysis['total_clicks'] / analysis['total_impressions']
    
    # 位置分析
    if 'position' in df.columns and 'clicked' in df.columns:
        position_stats = df.groupby('position').agg({
            'clicked': ['count', 'sum', 'mean']
        }).round(4)
        analysis['position_analysis'] = position_stats.to_dict()
    
    # 查询分析
    if 'query' in df.columns and 'clicked' in df.columns:
        query_stats = df.groupby('query').agg({
            'clicked': ['count', 'sum', 'mean']
        }).round(4)
        analysis['query_analysis'] = query_stats.head(10).to_dict()
    
    # 文档分析
    if 'doc_id' in df.columns and 'clicked' in df.columns:
        doc_stats = df.groupby('doc_id').agg({
            'clicked': ['count', 'sum', 'mean']
        }).round(4)
        analysis['doc_analysis'] = doc_stats.head(10).to_dict()
    
    return analysis 