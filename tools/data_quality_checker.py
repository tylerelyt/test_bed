#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量检查工具
用于检查搜索引擎测试床的数据质量和潜在问题
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from offline.index_service import get_index_service
from online.search_engine import SearchEngine

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.index_service = get_index_service()
        self.search_engine = SearchEngine()
        self.quality_issues = []
    
    def check_document_quality(self) -> Dict[str, Any]:
        """检查文档质量"""
        print("检查文档质量...")
        
        # 获取所有文档
        all_docs = self.index_service.get_all_documents()
        
        if not all_docs:
            return {"error": "没有找到文档"}
        
        issues = []
        stats = {
            'total_documents': len(all_docs),
            'empty_documents': 0,
            'short_documents': 0,
            'long_documents': 0,
            'duplicate_documents': 0,
            'documents_with_special_chars': 0,
            'average_length': 0,
            'length_distribution': {}
        }
        
        # 检查每个文档
        doc_contents = []
        doc_lengths = []
        
        for doc_id, doc_content in all_docs.items():
            doc_length = len(doc_content)
            doc_lengths.append(doc_length)
            doc_contents.append(doc_content)
            
            # 检查空文档
            if not doc_content.strip():
                issues.append(f"文档 {doc_id}: 内容为空")
                stats['empty_documents'] += 1
            
            # 检查过短文档
            elif doc_length < 10:
                issues.append(f"文档 {doc_id}: 内容过短 ({doc_length} 字符)")
                stats['short_documents'] += 1
            
            # 检查过长文档
            elif doc_length > 10000:
                issues.append(f"文档 {doc_id}: 内容过长 ({doc_length} 字符)")
                stats['long_documents'] += 1
            
            # 检查特殊字符
            if re.search(r'[^\w\s\u4e00-\u9fff]', doc_content):
                stats['documents_with_special_chars'] += 1
        
        # 检查重复文档
        content_counter = Counter(doc_contents)
        for content, count in content_counter.items():
            if count > 1:
                stats['duplicate_documents'] += count - 1
                issues.append(f"发现 {count} 个重复文档")
        
        # 计算统计信息
        if doc_lengths:
            stats['average_length'] = np.mean(doc_lengths)
            stats['length_distribution'] = {
                'min': min(doc_lengths),
                'max': max(doc_lengths),
                'std': np.std(doc_lengths)
            }
        
        self.quality_issues.extend(issues)
        
        return {
            'stats': stats,
            'issues': issues,
            'quality_score': self._calculate_quality_score(stats)
        }
    
    def check_index_quality(self) -> Dict[str, Any]:
        """检查索引质量"""
        print("检查索引质量...")
        
        index_stats = self.index_service.get_stats()
        
        issues = []
        stats = {
            'total_documents': index_stats.get('total_documents', 0),
            'total_terms': index_stats.get('total_terms', 0),
            'average_doc_length': index_stats.get('average_doc_length', 0),
            'term_frequency_distribution': {},
            'document_frequency_distribution': {}
        }
        
        # 检查索引完整性
        if stats['total_documents'] == 0:
            issues.append("索引中没有文档")
        
        if stats['total_terms'] == 0:
            issues.append("索引中没有词汇")
        
        # 检查词汇分布
        if stats['total_terms'] > 0 and stats['total_documents'] > 0:
            avg_terms_per_doc = stats['total_terms'] / stats['total_documents']
            if avg_terms_per_doc < 2:
                issues.append(f"平均每文档词汇数过少: {avg_terms_per_doc:.2f}")
        
        self.quality_issues.extend(issues)
        
        return {
            'stats': stats,
            'issues': issues,
            'quality_score': self._calculate_index_quality_score(stats)
        }
    
    def check_ctr_data_quality(self) -> Dict[str, Any]:
        """检查CTR数据质量"""
        print("检查CTR数据质量...")
        
        ctr_file = "data/ctr_data.json"
        
        if not os.path.exists(ctr_file):
            return {"error": f"CTR数据文件不存在: {ctr_file}"}
        
        try:
            with open(ctr_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 处理不同的数据格式
                if isinstance(data, dict) and 'records' in data:
                    ctr_data = data['records']
                elif isinstance(data, list):
                    ctr_data = data
                else:
                    ctr_data = []
        except Exception as e:
            return {"error": f"读取CTR数据失败: {e}"}
        
        issues = []
        stats = {
            'total_records': len(ctr_data),
            'positive_samples': 0,
            'negative_samples': 0,
            'missing_features': 0,
            'invalid_scores': 0,
            'average_ctr': 0,
            'ctr_distribution': {}
        }
        
        if not ctr_data:
            issues.append("CTR数据为空")
            return {'stats': stats, 'issues': issues, 'quality_score': 0}
        
        # 分析CTR数据
        ctr_values = []
        feature_counts = []
        
        for record in ctr_data:
            # 检查点击率（clicked字段）
            clicked = record.get('clicked', 0)
            ctr_values.append(clicked)
            
            if clicked > 0:
                stats['positive_samples'] += 1
            else:
                stats['negative_samples'] += 1
            
            # 检查特征数量（使用score作为特征）
            score = record.get('score', 0)
            feature_count = 1 if score is not None else 0
            feature_counts.append(feature_count)
            
            if feature_count == 0:
                stats['missing_features'] += 1
            
            # 检查无效分数
            if score < 0 or score > 1:
                stats['invalid_scores'] += 1
                issues.append(f"发现无效分数: {score}")
        
        # 计算统计信息
        if ctr_values:
            stats['average_ctr'] = np.mean(ctr_values)
            stats['ctr_distribution'] = {
                'min': min(ctr_values),
                'max': max(ctr_values),
                'std': np.std(ctr_values)
            }
        
        # 检查数据不平衡
        if stats['positive_samples'] > 0 and stats['negative_samples'] > 0:
            imbalance_ratio = stats['negative_samples'] / stats['positive_samples']
            if imbalance_ratio > 10:
                issues.append(f"数据严重不平衡，负样本/正样本比例: {imbalance_ratio:.2f}")
        
        # 检查特征一致性
        if feature_counts:
            avg_features = np.mean(feature_counts)
            if avg_features < 5:
                issues.append(f"平均特征数量过少: {avg_features:.2f}")
        
        self.quality_issues.extend(issues)
        
        return {
            'stats': stats,
            'issues': issues,
            'quality_score': self._calculate_ctr_quality_score(stats)
        }
    
    def check_search_quality(self, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """检查搜索质量"""
        print("检查搜索质量...")
        
        if test_queries is None:
            test_queries = [
                "人工智能",
                "机器学习",
                "深度学习",
                "自然语言处理",
                "计算机视觉"
            ]
        
        issues = []
        stats = {
            'total_queries': len(test_queries),
            'successful_queries': 0,
            'failed_queries': 0,
            'empty_results': 0,
            'average_results_per_query': 0,
            'response_times': []
        }
        
        results_per_query = []
        
        for query in test_queries:
            try:
                # 执行搜索
                doc_ids = self.search_engine.retrieve(query, top_k=10)
                results = self.search_engine.rank(query, doc_ids, top_k=5)
                
                stats['successful_queries'] += 1
                results_per_query.append(len(results))
                
                if len(results) == 0:
                    stats['empty_results'] += 1
                    issues.append(f"查询 '{query}' 没有返回结果")
                
            except Exception as e:
                stats['failed_queries'] += 1
                issues.append(f"查询 '{query}' 失败: {e}")
        
        # 计算统计信息
        if results_per_query:
            stats['average_results_per_query'] = np.mean(results_per_query)
        
        # 检查搜索覆盖率
        coverage_rate = stats['successful_queries'] / stats['total_queries']
        if coverage_rate < 0.8:
            issues.append(f"搜索覆盖率过低: {coverage_rate:.2%}")
        
        # 检查结果多样性
        if results_per_query:
            avg_results = np.mean(results_per_query)
            if avg_results < 2:
                issues.append(f"平均结果数量过少: {avg_results:.2f}")
        
        self.quality_issues.extend(issues)
        
        return {
            'stats': stats,
            'issues': issues,
            'quality_score': self._calculate_search_quality_score(stats)
        }
    
    def _calculate_quality_score(self, stats: Dict[str, Any]) -> float:
        """计算文档质量分数"""
        score = 100.0
        
        # 空文档扣分
        if stats['total_documents'] > 0:
            empty_ratio = stats['empty_documents'] / stats['total_documents']
            score -= empty_ratio * 30
        
        # 短文档扣分
        if stats['total_documents'] > 0:
            short_ratio = stats['short_documents'] / stats['total_documents']
            score -= short_ratio * 20
        
        # 重复文档扣分
        if stats['total_documents'] > 0:
            duplicate_ratio = stats['duplicate_documents'] / stats['total_documents']
            score -= duplicate_ratio * 25
        
        return max(0, score)
    
    def _calculate_index_quality_score(self, stats: Dict[str, Any]) -> float:
        """计算索引质量分数"""
        score = 100.0
        
        # 文档数量检查
        if stats['total_documents'] == 0:
            score = 0
        elif stats['total_documents'] < 10:
            score -= 20
        
        # 词汇数量检查
        if stats['total_terms'] == 0:
            score = 0
        elif stats['total_terms'] < 50:
            score -= 15
        
        # 平均文档长度检查
        if stats['average_doc_length'] < 50:
            score -= 10
        
        return max(0, score)
    
    def _calculate_ctr_quality_score(self, stats: Dict[str, Any]) -> float:
        """计算CTR数据质量分数"""
        score = 100.0
        
        # 记录数量检查
        if stats['total_records'] == 0:
            score = 0
        elif stats['total_records'] < 100:
            score -= 20
        
        # 数据不平衡检查
        if stats['positive_samples'] > 0 and stats['negative_samples'] > 0:
            imbalance_ratio = stats['negative_samples'] / stats['positive_samples']
            if imbalance_ratio > 10:
                score -= 15
        
        # 无效分数检查
        if stats['total_records'] > 0:
            invalid_ratio = stats['invalid_scores'] / stats['total_records']
            score -= invalid_ratio * 30
        
        return max(0, score)
    
    def _calculate_search_quality_score(self, stats: Dict[str, Any]) -> float:
        """计算搜索质量分数"""
        score = 100.0
        
        # 成功率检查
        if stats['total_queries'] > 0:
            success_rate = stats['successful_queries'] / stats['total_queries']
            score -= (1 - success_rate) * 40
        
        # 结果覆盖率检查
        if stats['successful_queries'] > 0:
            empty_ratio = stats['empty_results'] / stats['successful_queries']
            score -= empty_ratio * 30
        
        return max(0, score)
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """生成完整质量报告"""
        print("生成数据质量报告...")
        
        # 执行各项检查
        doc_quality = self.check_document_quality()
        index_quality = self.check_index_quality()
        ctr_quality = self.check_ctr_data_quality()
        search_quality = self.check_search_quality()
        
        # 计算总体质量分数
        scores = []
        if 'quality_score' in doc_quality:
            scores.append(doc_quality['quality_score'])
        if 'quality_score' in index_quality:
            scores.append(index_quality['quality_score'])
        if 'quality_score' in ctr_quality:
            scores.append(ctr_quality['quality_score'])
        if 'quality_score' in search_quality:
            scores.append(search_quality['quality_score'])
        
        overall_score = np.mean(scores) if scores else 0
        
        # 汇总所有问题
        all_issues = []
        all_issues.extend(doc_quality.get('issues', []))
        all_issues.extend(index_quality.get('issues', []))
        all_issues.extend(ctr_quality.get('issues', []))
        all_issues.extend(search_quality.get('issues', []))
        
        return {
            'overall_quality_score': overall_score,
            'component_scores': {
                'document_quality': doc_quality.get('quality_score', 0),
                'index_quality': index_quality.get('quality_score', 0),
                'ctr_quality': ctr_quality.get('quality_score', 0),
                'search_quality': search_quality.get('quality_score', 0)
            },
            'detailed_reports': {
                'document_quality': doc_quality,
                'index_quality': index_quality,
                'ctr_quality': ctr_quality,
                'search_quality': search_quality
            },
            'all_issues': all_issues,
            'issue_count': len(all_issues),
            'recommendations': self._generate_recommendations(all_issues)
        }
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """根据问题生成建议"""
        recommendations = []
        
        if any('空文档' in issue for issue in issues):
            recommendations.append("清理空文档，确保所有文档都有有效内容")
        
        if any('重复文档' in issue for issue in issues):
            recommendations.append("检测并移除重复文档，提高索引质量")
        
        if any('内容过短' in issue for issue in issues):
            recommendations.append("增加文档内容长度，提供更丰富的信息")
        
        if any('数据不平衡' in issue for issue in issues):
            recommendations.append("平衡CTR数据，增加正样本或使用采样技术")
        
        if any('覆盖率过低' in issue for issue in issues):
            recommendations.append("优化搜索算法，提高查询覆盖率")
        
        if any('没有返回结果' in issue for issue in issues):
            recommendations.append("检查索引完整性，确保所有相关文档都被索引")
        
        return recommendations

def main():
    """主函数"""
    print("搜索引擎测试床数据质量检查工具")
    print("=" * 50)
    
    checker = DataQualityChecker()
    
    try:
        # 生成质量报告
        report = checker.generate_quality_report()
        
        # 输出总体质量分数
        print(f"\n总体质量分数: {report['overall_quality_score']:.1f}/100")
        
        # 输出各组件分数
        print("\n各组件质量分数:")
        for component, score in report['component_scores'].items():
            print(f"  {component}: {score:.1f}/100")
        
        # 输出问题统计
        print(f"\n发现 {report['issue_count']} 个问题:")
        for i, issue in enumerate(report['all_issues'][:10], 1):  # 只显示前10个问题
            print(f"  {i}. {issue}")
        
        if len(report['all_issues']) > 10:
            print(f"  ... 还有 {len(report['all_issues']) - 10} 个问题")
        
        # 输出建议
        if report['recommendations']:
            print(f"\n改进建议:")
            for i, recommendation in enumerate(report['recommendations'], 1):
                print(f"  {i}. {recommendation}")
        
        # 质量评估
        print(f"\n质量评估:")
        if report['overall_quality_score'] >= 90:
            print("  优秀 - 数据质量很好，可以放心使用")
        elif report['overall_quality_score'] >= 70:
            print("  良好 - 数据质量较好，建议关注发现的问题")
        elif report['overall_quality_score'] >= 50:
            print("  一般 - 数据质量需要改进，建议优先处理关键问题")
        else:
            print("  较差 - 数据质量需要大幅改进，建议全面检查")
        
        # 保存详细报告
        report_file = "logs/data_quality_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"检查过程中出现错误: {e}")

if __name__ == "__main__":
    main() 