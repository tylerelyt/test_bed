#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型可解释性分析模块 - 使用LIME和SHAP进行模型解释
用于教学：理解模型预测的原因和特征重要性
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入LIME和SHAP
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME未安装，可解释性分析功能将受限。安装: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP未安装，SHAP分析功能将不可用。安装: pip install shap")


class ModelInterpretability:
    """模型可解释性分析器"""
    
    def __init__(self):
        self.lime_explainer = None
        self.feature_names = None
        self.training_data = None
    
    def prepare_lime_explainer(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None
    ):
        """准备LIME解释器"""
        if not LIME_AVAILABLE:
            return False, "LIME未安装"
        
        try:
            self.training_data = training_data
            self.feature_names = feature_names
            
            # 创建LIME解释器
            self.lime_explainer = LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=class_names or ['未点击', '点击'],
                mode='classification',
                discretize_continuous=True
            )
            return True, "LIME解释器创建成功"
        except Exception as e:
            return False, f"LIME解释器创建失败: {str(e)}"
    
    def explain_with_lime(
        self,
        model_instance,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        使用LIME解释单个预测
        
        Args:
            model_instance: 训练好的模型
            instance: 要解释的实例（单个样本）
            num_features: 显示的特征数量
            num_samples: LIME采样数量
        
        Returns:
            解释结果字典
        """
        if not LIME_AVAILABLE:
            return {'error': 'LIME未安装，请运行: pip install lime'}
        
        if self.lime_explainer is None:
            return {'error': 'LIME解释器未初始化，请先调用prepare_lime_explainer'}
        
        try:
            # 确保instance是2D数组
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
            
            # 创建预测函数
            def predict_fn(x):
                if hasattr(model_instance, 'predict_proba'):
                    return model_instance.predict_proba(x)
                else:
                    # 如果没有predict_proba，使用predict
                    pred = model_instance.predict(x)
                    # 转换为概率格式
                    prob = np.zeros((len(pred), 2))
                    prob[:, 1] = pred
                    prob[:, 0] = 1 - pred
                    return prob
            
            # 生成解释
            explanation = self.lime_explainer.explain_instance(
                instance[0],
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # 提取解释结果
            explanation_list = explanation.as_list()
            
            # 转换为字典格式
            result = {
                'prediction': float(explanation.predict_proba[1]) if len(explanation.predict_proba) > 1 else 0.0,
                'features': []
            }
            
            for feature, weight in explanation_list:
                result['features'].append({
                    'feature': feature,
                    'weight': float(weight)
                })
            
            # 按权重绝对值排序
            result['features'].sort(key=lambda x: abs(x['weight']), reverse=True)
            
            return result
            
        except Exception as e:
            return {'error': f'LIME解释失败: {str(e)}'}
    
    def explain_batch_with_lime(
        self,
        model_instance,
        instances: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        max_instances: int = 10
    ) -> Dict[str, Any]:
        """
        批量使用LIME解释多个预测
        
        Args:
            model_instance: 训练好的模型
            instances: 要解释的实例数组
            num_features: 显示的特征数量
            num_samples: LIME采样数量
            max_instances: 最大解释实例数（避免计算时间过长）
        
        Returns:
            批量解释结果
        """
        if not LIME_AVAILABLE:
            return {'error': 'LIME未安装'}
        
        if len(instances) > max_instances:
            instances = instances[:max_instances]
            print(f"⚠️ 限制解释实例数为 {max_instances}")
        
        results = []
        for i, instance in enumerate(instances):
            try:
                explanation = self.explain_with_lime(
                    model_instance, instance, num_features, num_samples
                )
                explanation['instance_id'] = i
                results.append(explanation)
            except Exception as e:
                results.append({
                    'instance_id': i,
                    'error': f'解释失败: {str(e)}'
                })
        
        return {
            'explanations': results,
            'n_explained': len(results)
        }
    
    def explain_with_shap(
        self,
        model_instance,
        training_data: np.ndarray,
        instance: Optional[np.ndarray] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        使用SHAP解释模型预测
        
        Args:
            model_instance: 训练好的模型
            training_data: 训练数据（用于SHAP背景数据）
            instance: 要解释的实例（可选，如果为None则解释所有训练数据）
            max_samples: 最大背景样本数（用于加速计算）
        
        Returns:
            SHAP解释结果
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP未安装，请运行: pip install shap'}
        
        try:
            # 限制背景数据大小以加速计算
            if len(training_data) > max_samples:
                indices = np.random.choice(len(training_data), max_samples, replace=False)
                background_data = training_data[indices]
            else:
                background_data = training_data
            
            # 创建预测函数（返回正类概率）
            def predict_fn(x):
                if hasattr(model_instance, 'predict_proba'):
                    proba = model_instance.predict_proba(x)
                    # 只返回正类概率（列向量）
                    return proba[:, 1]
                else:
                    return model_instance.predict(x)
            
            # 创建SHAP解释器
            # 对于树模型，使用TreeExplainer；对于其他模型，使用KernelExplainer
            if hasattr(model_instance, 'tree_') or hasattr(model_instance, 'estimators_'):
                # 树模型
                try:
                    explainer = shap.TreeExplainer(model_instance)
                    shap_values = explainer.shap_values(instance if instance is not None else background_data)
                except:
                    # 如果TreeExplainer失败，使用KernelExplainer
                    explainer = shap.KernelExplainer(predict_fn, background_data)
                    if instance is not None:
                        shap_values = explainer.shap_values(instance)
                    else:
                        shap_values = explainer.shap_values(background_data)
            else:
                # 对于线性模型（如LogisticRegression），优先使用LinearExplainer
                if hasattr(model_instance, 'coef_'):
                    try:
                        # LinearExplainer 适用于线性模型，速度更快
                        explainer = shap.LinearExplainer(model_instance, background_data)
                        if instance is not None:
                            shap_values = explainer.shap_values(instance)
                        else:
                            shap_values = explainer.shap_values(background_data)
                    except Exception as e:
                        # 如果LinearExplainer失败，回退到KernelExplainer
                        print(f"LinearExplainer失败，使用KernelExplainer: {e}")
                        explainer = shap.KernelExplainer(predict_fn, background_data)
                        if instance is not None:
                            shap_values = explainer.shap_values(instance)
                        else:
                            shap_values = explainer.shap_values(background_data)
                else:
                    # 非线性模型，使用KernelExplainer
                    explainer = shap.KernelExplainer(predict_fn, background_data)
                    if instance is not None:
                        shap_values = explainer.shap_values(instance)
                    else:
                        shap_values = explainer.shap_values(background_data)
            
            # 处理SHAP值（可能是列表，取正类）
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # 计算特征重要性（平均绝对SHAP值）
            if len(shap_values.shape) == 2:
                feature_importance = np.abs(shap_values).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values)
            
            result = {
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'feature_importance': feature_importance.tolist() if isinstance(feature_importance, np.ndarray) else feature_importance,
                'n_samples': len(background_data)
            }
            
            if self.feature_names:
                result['feature_names'] = self.feature_names
                # 创建特征重要性字典
                importance_dict = {}
                for i, name in enumerate(self.feature_names):
                    if i < len(feature_importance):
                        importance_dict[name] = float(feature_importance[i])
                result['feature_importance_dict'] = importance_dict
            
            return result
            
        except Exception as e:
            return {'error': f'SHAP解释失败: {str(e)}'}
    
    def get_feature_importance_from_model(self, model_instance) -> Dict[str, Any]:
        """
        从模型本身获取特征重要性（如果支持）
        
        Args:
            model_instance: 模型实例
        
        Returns:
            特征重要性字典
        """
        importance = {}
        
        # 逻辑回归：使用系数绝对值
        if hasattr(model_instance, 'coef_'):
            coef = model_instance.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # 取第一个类的系数
            importance['coefficients'] = np.abs(coef).tolist()
            importance['coefficients_raw'] = coef.tolist()
        
        # 树模型：使用feature_importances_
        if hasattr(model_instance, 'feature_importances_'):
            importance['tree_importance'] = model_instance.feature_importances_.tolist()
        
        # 如果有特征名称，创建字典
        if self.feature_names and importance:
            importance_dict = {}
            for key, values in importance.items():
                if isinstance(values, list):
                    importance_dict[key] = {
                        name: float(val) 
                        for name, val in zip(self.feature_names, values)
                        if name and val is not None
                    }
            importance['by_feature'] = importance_dict
        
        return importance

