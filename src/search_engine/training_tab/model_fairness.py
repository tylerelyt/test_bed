#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型公平性分析模块 - 评估模型在不同群体上的性能差异
用于教学：初步了解如何评估模型的公平性
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class ModelFairnessAnalyzer:
    """模型公平性分析器 - 评估不同群体的性能差异"""
    
    def __init__(self):
        self.group_metrics = {}
    
    def define_groups(
        self,
        ctr_data: List[Dict[str, Any]],
        group_by: str = 'query'
    ) -> Dict[str, List[int]]:
        """
        定义不同的群体
        
        Args:
            ctr_data: CTR数据列表
            group_by: 分组依据，可选: 'query', 'doc_id', 'position', 'custom'
        
        Returns:
            群体字典，键是群体名称，值是该群体的数据索引列表
        """
        df = pd.DataFrame(ctr_data)
        groups = {}
        
        if group_by == 'query':
            # 按查询分组
            for query in df['query'].unique():
                indices = df[df['query'] == query].index.tolist()
                if len(indices) >= 3:  # 至少3个样本才作为一个群体
                    groups[f"查询: {query[:20]}"] = indices
        
        elif group_by == 'doc_id':
            # 按文档ID分组
            for doc_id in df['doc_id'].unique():
                indices = df[df['doc_id'] == doc_id].index.tolist()
                if len(indices) >= 3:
                    groups[f"文档: {doc_id[:20]}"] = indices
        
        elif group_by == 'position':
            # 按位置分组
            for position in sorted(df['position'].unique()):
                indices = df[df['position'] == position].index.tolist()
                if len(indices) >= 3:
                    groups[f"位置: {position}"] = indices
        
        elif group_by == 'position_range':
            # 按位置范围分组
            df['position_range'] = pd.cut(
                df['position'],
                bins=[0, 3, 6, 10, float('inf')],
                labels=['顶部(1-3)', '中部(4-6)', '下部(7-10)', '底部(>10)']
            )
            for range_name in df['position_range'].unique():
                if pd.notna(range_name):
                    indices = df[df['position_range'] == range_name].index.tolist()
                    if len(indices) >= 3:
                        groups[str(range_name)] = indices
        
        elif group_by == 'score_range':
            # 按相似度分数范围分组
            score_quantiles = df['score'].quantile([0, 0.25, 0.5, 0.75, 1.0])
            df['score_range'] = pd.cut(
                df['score'],
                bins=score_quantiles.values,
                labels=['低分', '中低分', '中高分', '高分'],
                include_lowest=True
            )
            for range_name in df['score_range'].unique():
                if pd.notna(range_name):
                    indices = df[df['score_range'] == range_name].index.tolist()
                    if len(indices) >= 3:
                        groups[str(range_name)] = indices
        
        return groups
    
    def evaluate_group_performance(
        self,
        model_instance,
        ctr_data: List[Dict[str, Any]],
        group_indices: List[int],
        model_instance_extract_features
    ) -> Dict[str, Any]:
        """
        评估特定群体的模型性能
        
        Args:
            model_instance: 训练好的模型
            ctr_data: 完整的CTR数据
            group_indices: 该群体的数据索引
            model_instance_extract_features: 特征提取函数
        
        Returns:
            该群体的性能指标
        """
        try:
            # 提取该群体的数据
            group_data = [ctr_data[i] for i in group_indices if i < len(ctr_data)]
            
            if len(group_data) < 2:
                return {'error': '群体数据量不足'}
            
            # 提取特征和标签
            features, labels = model_instance_extract_features(group_data)
            
            if len(features) == 0 or len(labels) == 0:
                return {'error': '特征提取失败'}
            
            # 标准化特征（使用模型的scaler）
            if hasattr(model_instance, 'scaler') and model_instance.scaler:
                features_scaled = model_instance.scaler.transform(features)
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
            
            # 预测
            y_pred = model_instance.predict(features_scaled)
            
            # 计算基础指标
            accuracy = accuracy_score(labels, y_pred)
            precision = precision_score(labels, y_pred, zero_division=0)
            recall = recall_score(labels, y_pred, zero_division=0)
            f1 = f1_score(labels, y_pred, zero_division=0)
            
            # 计算AUC（如果有predict_proba）
            auc = None
            if hasattr(model_instance, 'predict_proba'):
                try:
                    y_pred_proba = model_instance.predict_proba(features_scaled)[:, 1]
                    if len(np.unique(labels)) == 2:
                        auc = roc_auc_score(labels, y_pred_proba)
                except:
                    pass
            
            # 计算混淆矩阵
            cm = confusion_matrix(labels, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # 计算点击率
            click_rate = float(np.mean(labels)) if len(labels) > 0 else 0.0
            
            return {
                'n_samples': len(group_data),
                'click_rate': click_rate,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc) if auc is not None else None,
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                }
            }
            
        except Exception as e:
            return {'error': f'评估失败: {str(e)}'}
    
    def analyze_fairness(
        self,
        model_instance,
        ctr_data: List[Dict[str, Any]],
        group_by: str = 'position_range',
        model_instance_extract_features = None
    ) -> Dict[str, Any]:
        """
        分析模型在不同群体上的公平性
        
        Args:
            model_instance: 训练好的模型
            ctr_data: CTR数据列表
            group_by: 分组依据
            model_instance_extract_features: 特征提取函数（如果模型有extract_features方法，会自动使用）
        
        Returns:
            公平性分析结果
        """
        try:
            # 获取特征提取函数
            if model_instance_extract_features is None:
                if hasattr(model_instance, 'extract_features'):
                    def extract_fn(data):
                        return model_instance.extract_features(data)
                else:
                    return {'error': '需要提供特征提取函数'}
            else:
                extract_fn = model_instance_extract_features
            
            # 定义群体
            groups = self.define_groups(ctr_data, group_by)
            
            if len(groups) < 2:
                return {
                    'error': f'无法定义足够的群体（至少需要2个），当前只有{len(groups)}个',
                    'groups': list(groups.keys())
                }
            
            # 评估每个群体的性能
            group_results = {}
            for group_name, group_indices in groups.items():
                result = self.evaluate_group_performance(
                    model_instance,
                    ctr_data,
                    group_indices,
                    extract_fn
                )
                if 'error' not in result:
                    group_results[group_name] = result
            
            if len(group_results) < 2:
                return {
                    'error': '成功评估的群体数量不足',
                    'groups': list(groups.keys())
                }
            
            # 计算公平性指标
            fairness_metrics = self._calculate_fairness_metrics(group_results)
            
            return {
                'groups': group_results,
                'fairness_metrics': fairness_metrics,
                'group_by': group_by,
                'n_groups': len(group_results)
            }
            
        except Exception as e:
            return {'error': f'公平性分析失败: {str(e)}'}
    
    def _calculate_fairness_metrics(self, group_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算公平性指标
        
        Args:
            group_results: 各群体的性能结果
        
        Returns:
            公平性指标字典
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        fairness = {}
        
        for metric in metrics:
            values = [r[metric] for r in group_results.values() if metric in r]
            if values:
                fairness[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0  # 变异系数
                }
        
        # AUC公平性（如果有）
        auc_values = [r['auc'] for r in group_results.values() if r.get('auc') is not None]
        if auc_values:
            fairness['auc'] = {
                'mean': float(np.mean(auc_values)),
                'std': float(np.std(auc_values)),
                'min': float(np.min(auc_values)),
                'max': float(np.max(auc_values)),
                'range': float(np.max(auc_values) - np.min(auc_values)),
                'cv': float(np.std(auc_values) / np.mean(auc_values)) if np.mean(auc_values) > 0 else 0.0
            }
        
        # 计算性能差异（最大差异）
        if 'accuracy' in fairness:
            fairness['max_accuracy_gap'] = fairness['accuracy']['range']
        if 'f1' in fairness:
            fairness['max_f1_gap'] = fairness['f1']['range']
        
        # 计算点击率差异
        click_rates = [r['click_rate'] for r in group_results.values() if 'click_rate' in r]
        if click_rates:
            fairness['click_rate'] = {
                'mean': float(np.mean(click_rates)),
                'std': float(np.std(click_rates)),
                'min': float(np.min(click_rates)),
                'max': float(np.max(click_rates)),
                'range': float(np.max(click_rates) - np.min(click_rates))
            }
        
        return fairness
    
    def generate_fairness_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        生成公平性分析报告（HTML格式）
        
        Args:
            analysis_result: 公平性分析结果
        
        Returns:
            HTML格式的报告
        """
        if 'error' in analysis_result:
            return f"<h4>❌ 公平性分析失败</h4><p>{analysis_result['error']}</p>"
        
        report = "<h4>📊 模型公平性分析报告</h4>"
        report += f"<p><strong>分组依据:</strong> {analysis_result.get('group_by', 'N/A')}</p>"
        report += f"<p><strong>分析群体数:</strong> {analysis_result.get('n_groups', 0)}</p>"
        
        # 各群体性能
        report += "<h5>各群体性能</h5>"
        report += "<table border='1' style='border-collapse: collapse; width: 100%; margin: 10px 0;'>"
        report += "<thead><tr style='background-color: #e9ecef;'>"
        report += "<th style='padding: 8px;'>群体</th>"
        report += "<th style='padding: 8px;'>样本数</th>"
        report += "<th style='padding: 8px;'>点击率</th>"
        report += "<th style='padding: 8px;'>准确率</th>"
        report += "<th style='padding: 8px;'>精确率</th>"
        report += "<th style='padding: 8px;'>召回率</th>"
        report += "<th style='padding: 8px;'>F1</th>"
        report += "<th style='padding: 8px;'>AUC</th>"
        report += "</tr></thead><tbody>"
        
        for group_name, metrics in analysis_result.get('groups', {}).items():
            report += f"<tr><td style='padding: 8px;'>{group_name}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('n_samples', 0)}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('click_rate', 0):.3f}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('accuracy', 0):.3f}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('precision', 0):.3f}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('recall', 0):.3f}</td>"
            report += f"<td style='padding: 8px; text-align: center;'>{metrics.get('f1', 0):.3f}</td>"
            auc = metrics.get('auc', 'N/A')
            if isinstance(auc, (int, float)):
                report += f"<td style='padding: 8px; text-align: center;'>{auc:.3f}</td>"
            else:
                report += f"<td style='padding: 8px; text-align: center;'>{auc}</td>"
            report += "</tr>"
        
        report += "</tbody></table>"
        
        # 公平性指标
        fairness = analysis_result.get('fairness_metrics', {})
        if fairness:
            report += "<h5>公平性指标</h5>"
            report += "<div style='margin: 10px 0;'>"
            
            for metric_name, metric_stats in fairness.items():
                if isinstance(metric_stats, dict) and 'mean' in metric_stats:
                    report += f"<div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>"
                    report += f"<strong>{metric_name}</strong>"
                    report += "<ul style='margin: 5px 0;'>"
                    report += f"<li>平均值: {metric_stats['mean']:.3f}</li>"
                    report += f"<li>标准差: {metric_stats['std']:.3f}</li>"
                    report += f"<li>范围: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]</li>"
                    report += f"<li>最大差异: {metric_stats.get('range', 0):.3f}</li>"
                    if 'cv' in metric_stats:
                        report += f"<li>变异系数: {metric_stats['cv']:.3f}</li>"
                    report += "</ul></div>"
            
            report += "</div>"
        
        # 公平性结论
        report += "<h5>公平性评估</h5><div>"
        
        if 'accuracy' in fairness:
            acc_range = fairness['accuracy'].get('range', 0)
            if acc_range < 0.05:
                report += "<p style='color: green;'>✅ <strong>准确率差异较小</strong>，模型在不同群体上表现相对公平</p>"
            elif acc_range < 0.15:
                report += "<p style='color: orange;'>⚠️ <strong>准确率存在一定差异</strong>，建议进一步分析原因</p>"
            else:
                report += "<p style='color: red;'>❌ <strong>准确率差异较大</strong>，模型可能存在公平性问题</p>"
        
        if 'f1' in fairness:
            f1_range = fairness['f1'].get('range', 0)
            if f1_range < 0.1:
                report += "<p style='color: green;'>✅ <strong>F1分数差异较小</strong></p>"
            elif f1_range < 0.2:
                report += "<p style='color: orange;'>⚠️ <strong>F1分数存在一定差异</strong></p>"
            else:
                report += "<p style='color: red;'>❌ <strong>F1分数差异较大</strong></p>"
        
        report += "</div>"
        
        return report

