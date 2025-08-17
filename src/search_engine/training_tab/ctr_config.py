#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR配置类 - 定义CTR训练样本的字段和配置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime

@dataclass
class CTRSampleConfig:
    """CTR样本配置类"""
    
    # 基础字段
    timestamp: str = field(default="", metadata={"description": "时间戳", "type": "str"})
    query: str = field(default="", metadata={"description": "搜索查询", "type": "str"})
    doc_id: str = field(default="", metadata={"description": "文档ID", "type": "str"})
    position: int = field(default=0, metadata={"description": "展示位置", "type": "int"})
    score: float = field(default=0.0, metadata={"description": "相似度分数", "type": "float"})
    clicked: int = field(default=0, metadata={"description": "是否点击", "type": "int"})
    
    # 内容字段
    summary: str = field(default="", metadata={"description": "文档摘要", "type": "str"})
    doc_length: int = field(default=0, metadata={"description": "文档长度", "type": "int"})
    query_length: int = field(default=0, metadata={"description": "查询长度", "type": "int"})
    summary_length: int = field(default=0, metadata={"description": "摘要长度", "type": "int"})
    
    # 请求字段
    request_id: str = field(default="", metadata={"description": "请求ID", "type": "str"})
    request_time: str = field(default="", metadata={"description": "请求时间", "type": "str"})
    
    # 特征字段
    match_score: float = field(default=0.0, metadata={"description": "查询匹配度", "type": "float"})
    query_ctr: float = field(default=0.0, metadata={"description": "查询历史CTR", "type": "float"})
    doc_ctr: float = field(default=0.0, metadata={"description": "文档历史CTR", "type": "float"})
    position_decay: float = field(default=0.0, metadata={"description": "位置衰减", "type": "float"})
    
    # 点击详情字段
    click_count: int = field(default=0, metadata={"description": "点击次数", "type": "int"})
    click_time: str = field(default="", metadata={"description": "首次点击时间", "type": "str"})
    last_click_time: str = field(default="", metadata={"description": "最后点击时间", "type": "str"})
    
    @classmethod
    def get_field_names(cls) -> List[str]:
        """获取所有字段名"""
        return list(cls.__dataclass_fields__.keys())
    
    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """获取字段描述"""
        return {name: field.metadata.get("description", "") 
                for name, field in cls.__dataclass_fields__.items()}
    
    @classmethod
    def get_field_types(cls) -> Dict[str, str]:
        """获取字段类型"""
        return {name: field.metadata.get("type", "unknown") 
                for name, field in cls.__dataclass_fields__.items()}
    
    @classmethod
    def create_empty_sample(cls) -> Dict[str, Any]:
        """创建空的CTR样本"""
        return {field_name: field.default 
                for field_name, field in cls.__dataclass_fields__.items()}
    
    @classmethod
    def validate_sample(cls, sample: Dict[str, Any]) -> List[str]:
        """验证CTR样本的完整性"""
        errors = []
        field_names = cls.get_field_names()
        
        # 检查必需字段
        for field_name in field_names:
            if field_name not in sample:
                errors.append(f"缺失字段: {field_name}")
        
        # 检查字段类型
        for field_name, field in cls.__dataclass_fields__.items():
            if field_name in sample:
                expected_type = field.metadata.get("type", "unknown")
                actual_value = sample[field_name]
                
                if expected_type == "int" and not isinstance(actual_value, int):
                    errors.append(f"字段 {field_name} 类型错误，期望 int，实际 {type(actual_value)}")
                elif expected_type == "float" and not isinstance(actual_value, (int, float)):
                    errors.append(f"字段 {field_name} 类型错误，期望 float，实际 {type(actual_value)}")
                elif expected_type == "str" and not isinstance(actual_value, str):
                    errors.append(f"字段 {field_name} 类型错误，期望 str，实际 {type(actual_value)}")
        
        return errors

class CTRFeatureConfig:
    """CTR特征配置类"""
    
    # 特征权重配置
    FEATURE_WEIGHTS = {
        'position': 1.0,
        'position_decay': 1.0,
        'score': 0.8,
        'match_score': 0.6,
        'doc_length': 0.3,
        'query_length': 0.2,
        'summary_length': 0.2,
        'query_ctr': 0.5,
        'doc_ctr': 0.5
    }
    
    # 特征归一化配置
    FEATURE_SCALING = {
        'position': {'min': 1, 'max': 10},
        'score': {'min': 0, 'max': 1},
        'match_score': {'min': 0, 'max': 1},
        'doc_length': {'min': 0, 'max': 1000},
        'query_length': {'min': 0, 'max': 100},
        'summary_length': {'min': 0, 'max': 500},
        'query_ctr': {'min': 0, 'max': 1},
        'doc_ctr': {'min': 0, 'max': 1},
        'position_decay': {'min': 0.1, 'max': 1.0}
    }
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """获取特征名称列表"""
        return list(cls.FEATURE_WEIGHTS.keys())
    
    @classmethod
    def get_feature_weights(cls) -> Dict[str, float]:
        """获取特征权重"""
        return cls.FEATURE_WEIGHTS.copy()
    
    @classmethod
    def get_scaling_config(cls) -> Dict[str, Dict[str, float]]:
        """获取特征归一化配置"""
        return cls.FEATURE_SCALING.copy()

class CTRModelConfig:
    """CTR模型配置类"""
    
    # 支持的模型类型
    SUPPORTED_MODELS = {
        'logistic_regression': {
            'name': '逻辑回归 (LR)',
            'description': '经典线性模型，训练快速，解释性强',
            'class': 'CTRModel',
            'module': '.ctr_model',
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        'wide_and_deep': {
            'name': 'Wide & Deep',
            'description': '结合线性模型和深度神经网络，效果更好',
            'class': 'WideAndDeepCTRModel',
            'module': '.ctr_wide_deep_model',
            'params': {
                'epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            }
        }
    }
    
    # 默认模型
    DEFAULT_MODEL = 'logistic_regression'
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, Dict[str, Any]]:
        """获取支持的模型列表"""
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        return cls.SUPPORTED_MODELS.get(model_type, {})
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """获取模型名称列表"""
        return [(k, v['name']) for k, v in cls.SUPPORTED_MODELS.items()]

class CTRTrainingConfig:
    """CTR训练配置类"""
    
    # 训练参数
    MIN_SAMPLES = 10  # 最小训练样本数
    TEST_SIZE = 0.2   # 测试集比例
    RANDOM_STATE = 42 # 随机种子
    
    # 评估指标
    EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    @classmethod
    def get_evaluation_metrics(cls) -> List[str]:
        """获取评估指标"""
        return cls.EVALUATION_METRICS.copy()

# 导出配置实例
ctr_sample_config = CTRSampleConfig()
ctr_feature_config = CTRFeatureConfig()
ctr_model_config = CTRModelConfig()
ctr_training_config = CTRTrainingConfig() 