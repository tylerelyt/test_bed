#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块 - 交叉验证和性能评估
用于教学：评估模型在不同数据子集上的性能，了解模型的泛化能力
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold, 
    cross_val_score,
    cross_validate,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器 - 支持交叉验证和性能分析"""
    
    def __init__(self):
        self.scaler = None
        self.cv_results = None
    
    def prepare_data(self, ctr_data: List[Dict[str, Any]], model_instance) -> Tuple[np.ndarray, np.ndarray]:
        """准备评估数据"""
        if not ctr_data:
            raise ValueError("数据为空")
        
        # 使用模型的特征提取方法
        if hasattr(model_instance, 'extract_features'):
            features, labels = model_instance.extract_features(ctr_data)
        else:
            # 如果没有extract_features方法，尝试使用训练方法
            # 这里假设模型有类似的特征提取逻辑
            df = pd.DataFrame(ctr_data)
            features = np.array([[r.get('position', 0), r.get('score', 0)] for r in ctr_data])
            labels = np.array([r.get('clicked', 0) for r in ctr_data])
        
        # 标准化特征
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        return features_scaled, labels
    
    def cross_validate_model(
        self,
        model_instance,
        ctr_data: List[Dict[str, Any]],
        cv_folds: int = 5,
        scoring_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行K折交叉验证
        
        Args:
            model_instance: 模型实例（需要支持fit和predict方法）
            ctr_data: CTR数据列表
            cv_folds: 交叉验证折数
            scoring_metrics: 评估指标列表
        
        Returns:
            包含交叉验证结果的字典
        """
        try:
            # 准备数据
            X, y = self.prepare_data(ctr_data, model_instance)
            
            if len(X) == 0:
                return {
                    'error': '数据准备失败',
                    'cv_scores': {},
                    'cv_mean': {},
                    'cv_std': {}
                }
            
            # 检查数据是否足够进行交叉验证
            if len(X) < cv_folds:
                return {
                    'error': f'数据量不足，需要至少{cv_folds}条记录，当前只有{len(X)}条',
                    'cv_scores': {},
                    'cv_mean': {},
                    'cv_std': {}
                }
            
            # 默认评估指标
            if scoring_metrics is None:
                scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            # 检查标签分布
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                return {
                    'error': '标签分布不平衡，无法进行交叉验证',
                    'cv_scores': {},
                    'cv_mean': {},
                    'cv_std': {}
                }
            
            # 选择交叉验证策略
            # 如果数据不平衡，使用分层K折
            if len(y) >= cv_folds * 2:
                try:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=None)
                except:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=None)
            else:
                cv = KFold(n_splits=min(cv_folds, len(X) // 2), shuffle=True, random_state=None)
            
            # 执行交叉验证
            cv_results = {}
            cv_scores = {}
            cv_mean = {}
            cv_std = {}
            
            # 对每个指标进行交叉验证
            for metric in scoring_metrics:
                try:
                    # 对于roc_auc，需要predict_proba
                    if metric == 'roc_auc':
                        if hasattr(model_instance, 'predict_proba'):
                            scores = cross_val_score(
                                model_instance, X, y, 
                                cv=cv, 
                                scoring='roc_auc',
                                n_jobs=1
                            )
                        else:
                            # 如果没有predict_proba，跳过roc_auc
                            continue
                    else:
                        scores = cross_val_score(
                            model_instance, X, y,
                            cv=cv,
                            scoring=metric,
                            n_jobs=1
                        )
                    
                    cv_scores[metric] = scores.tolist()
                    cv_mean[metric] = float(np.mean(scores))
                    cv_std[metric] = float(np.std(scores))
                    
                except Exception as e:
                    print(f"指标 {metric} 的交叉验证失败: {e}")
                    continue
            
            # 执行详细的交叉验证（包含多个指标）
            try:
                scoring_dict = {
                    'accuracy': 'accuracy',
                    'precision': 'precision',
                    'recall': 'recall',
                    'f1': 'f1'
                }
                
                # 如果支持roc_auc，添加它
                if hasattr(model_instance, 'predict_proba'):
                    scoring_dict['roc_auc'] = 'roc_auc'
                
                detailed_cv = cross_validate(
                    model_instance, X, y,
                    cv=cv,
                    scoring=scoring_dict,
                    return_train_score=True,
                    n_jobs=1
                )
                
                cv_results['detailed'] = {
                    'test_accuracy': detailed_cv['test_accuracy'].tolist(),
                    'test_precision': detailed_cv['test_precision'].tolist(),
                    'test_recall': detailed_cv['test_recall'].tolist(),
                    'test_f1': detailed_cv['test_f1'].tolist(),
                    'train_accuracy': detailed_cv['train_accuracy'].tolist() if 'train_accuracy' in detailed_cv else [],
                }
                
                if 'test_roc_auc' in detailed_cv:
                    cv_results['detailed']['test_roc_auc'] = detailed_cv['test_roc_auc'].tolist()
                
            except Exception as e:
                print(f"详细交叉验证失败: {e}")
            
            self.cv_results = {
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_folds': cv_folds,
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'detailed': cv_results.get('detailed', {})
            }
            
            return self.cv_results
            
        except Exception as e:
            return {
                'error': f'交叉验证失败: {str(e)}',
                'cv_scores': {},
                'cv_mean': {},
                'cv_std': {}
            }
    
    def evaluate_on_splits(
        self,
        model_instance,
        ctr_data: List[Dict[str, Any]],
        test_size: float = 0.2,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        在不同数据子集上评估模型性能
        
        Args:
            model_instance: 模型实例
            ctr_data: CTR数据列表
            test_size: 测试集比例
            n_splits: 划分次数
        
        Returns:
            包含各次划分结果的字典
        """
        try:
            X, y = self.prepare_data(ctr_data, model_instance)
            
            if len(X) == 0:
                return {'error': '数据准备失败'}
            
            results = []
            
            for i in range(n_splits):
                try:
                    # 每次使用不同的随机种子
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size, 
                        random_state=42 + i,
                        stratify=y if len(np.unique(y)) == 2 else None
                    )
                    
                    # 训练模型
                    model_instance.fit(X_train, y_train)
                    
                    # 预测
                    y_pred = model_instance.predict(X_test)
                    
                    # 计算指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # 如果有predict_proba，计算AUC
                    auc = None
                    if hasattr(model_instance, 'predict_proba'):
                        try:
                            y_pred_proba = model_instance.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, y_pred_proba)
                        except:
                            pass
                    
                    results.append({
                        'split': i + 1,
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'auc': float(auc) if auc is not None else None
                    })
                    
                except Exception as e:
                    print(f"第 {i+1} 次划分评估失败: {e}")
                    continue
            
            if not results:
                return {'error': '所有划分评估都失败'}
            
            # 计算统计信息
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            summary = {}
            for metric in metrics:
                values = [r[metric] for r in results if r[metric] is not None]
                if values:
                    summary[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            # AUC统计（如果有）
            auc_values = [r['auc'] for r in results if r['auc'] is not None]
            if auc_values:
                summary['auc'] = {
                    'mean': float(np.mean(auc_values)),
                    'std': float(np.std(auc_values)),
                    'min': float(np.min(auc_values)),
                    'max': float(np.max(auc_values))
                }
            
            return {
                'splits': results,
                'summary': summary,
                'n_splits': len(results)
            }
            
        except Exception as e:
            return {'error': f'评估失败: {str(e)}'}

