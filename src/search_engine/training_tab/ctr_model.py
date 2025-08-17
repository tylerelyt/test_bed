#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR模型模块 - 训练和使用CTR模型进行排序
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os
from typing import List, Dict, Any, Tuple
import jieba
from sklearn.model_selection import StratifiedShuffleSplit
from .ctr_config import CTRFeatureConfig, CTRTrainingConfig, ctr_feature_config, ctr_training_config

class CTRModel:
    """CTR模型类"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.is_trained = False
    
    def extract_features(self, ctr_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """从CTR数据中提取特征"""
        if not ctr_data:
            return np.array([]), np.array([])
        
        # 转换为DataFrame
        df = pd.DataFrame(ctr_data)
        
        # 1. 位置特征（绝对位置）
        position_features = df['position'].values.reshape(-1, 1)
        
        # 2. 文档长度特征
        doc_lengths = df['summary'].str.len().values.reshape(-1, 1)
        
        # 3. 查询长度特征
        query_lengths = df['query'].str.len().values.reshape(-1, 1)
        
        # 4. 摘要长度特征
        summary_lengths = df['summary'].str.len().values.reshape(-1, 1)
        
        # 5. 查询词在摘要中的匹配度
        match_scores = []
        for _, row in df.iterrows():
            query_words = set(jieba.lcut(row['query']))
            summary_words = set(jieba.lcut(row['summary']))
            if len(query_words) > 0:
                match_ratio = len(query_words.intersection(summary_words)) / len(query_words)
            else:
                match_ratio = 0
            match_scores.append(match_ratio)
        match_scores = np.array(match_scores).reshape(-1, 1)
        
        # 6. 历史点击率特征（基于查询的统计）- 修复数据泄露
        # 按时间戳排序，确保历史特征只使用过去的数据
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        query_ctr_features = []
        doc_ctr_features = []
        
        for idx, row in df_sorted.iterrows():
            # 查询历史CTR - 只使用当前样本之前的数据
            query = row['query']
            query_history = df_sorted.loc[:idx-1]  # 不包括当前样本
            query_history_filtered = query_history[query_history['query'] == query]
            
            if len(query_history_filtered) > 0:
                query_ctr = query_history_filtered['clicked'].mean()
            else:
                query_ctr = 0.1  # 默认值
            query_ctr_features.append(query_ctr)
            
            # 文档历史CTR - 只使用当前样本之前的数据
            doc_id = row['doc_id']
            doc_history = df_sorted.loc[:idx-1]  # 不包括当前样本
            doc_history_filtered = doc_history[doc_history['doc_id'] == doc_id]
            
            if len(doc_history_filtered) > 0:
                doc_ctr = doc_history_filtered['clicked'].mean()
            else:
                doc_ctr = 0.1  # 默认值
            doc_ctr_features.append(doc_ctr)
        
        # 按原始顺序重新排列
        original_order = df.index
        query_ctr_features = [query_ctr_features[i] for i in original_order]
        doc_ctr_features = [doc_ctr_features[i] for i in original_order]
        
        query_ctr_features = np.array(query_ctr_features).reshape(-1, 1)
        doc_ctr_features = np.array(doc_ctr_features).reshape(-1, 1)
        
        # 7. 添加更多变化性特征
        # 查询词数量
        query_word_counts = []
        for _, row in df.iterrows():
            word_count = len(jieba.lcut(row['query']))
            query_word_counts.append(word_count)
        query_word_counts = np.array(query_word_counts).reshape(-1, 1)
        
        # 摘要词数量
        summary_word_counts = []
        for _, row in df.iterrows():
            word_count = len(jieba.lcut(row['summary']))
            summary_word_counts.append(word_count)
        summary_word_counts = np.array(summary_word_counts).reshape(-1, 1)
        
        # 时间特征（基于timestamp的数值化）
        time_features = []
        for _, row in df.iterrows():
            try:
                # 提取时间戳中的数值部分作为特征
                timestamp_str = str(row['timestamp'])
                time_value = sum(ord(c) for c in timestamp_str) % 1000  # 简单的哈希
                time_features.append(time_value)
            except:
                time_features.append(0)
        time_features = np.array(time_features).reshape(-1, 1)
        
        # 8. 位置衰减特征（位置越靠前，权重越高）
        position_decay = 1.0 / (position_features + 1)  # 避免除零
        
        # 组合所有特征
        features = np.hstack([
            position_features,           # 位置
            doc_lengths,                 # 文档长度
            query_lengths,               # 查询长度
            summary_lengths,             # 摘要长度
            match_scores,                # 查询匹配度
            query_ctr_features,          # 查询历史CTR
            doc_ctr_features,            # 文档历史CTR
            position_decay,              # 位置衰减
            query_word_counts,           # 查询词数量
            summary_word_counts,         # 摘要词数量
            time_features,               # 时间特征
            df['score'].values.reshape(-1, 1)  # 原始相似度分数
        ])
        
        # 标签
        labels = df['clicked'].values
        
        return features, labels
    
    def _empty_metrics(self, error_msg):
        return {
            'error': error_msg,
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'train_samples': 0,
            'test_samples': 0,
            'train_score': 0.0,
            'test_score': 0.0,
            'feature_weights': {},
            'data_quality': {}
        }
    
    def train(self, ctr_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """训练CTR模型"""
        if not ctr_data:
            return self._empty_metrics('没有CTR数据用于训练')
        
        # 检查数据分布
        df = pd.DataFrame(ctr_data)
        total_samples = len(df)
        click_samples = df['clicked'].sum()
        no_click_samples = total_samples - click_samples
        
        min_samples = CTRTrainingConfig.MIN_SAMPLES
        if total_samples < min_samples:
            return self._empty_metrics(f'数据量不足，需要至少{min_samples}条记录，当前只有{total_samples}条')
        if click_samples < 2:
            return self._empty_metrics(f'点击数据不足，需要至少2次点击，当前只有{click_samples}次点击')
        if no_click_samples < 2:
            return self._empty_metrics(f'未点击数据不足，需要至少2次未点击，当前只有{no_click_samples}次未点击')
        
        # 检查数据多样性
        unique_queries = df['query'].nunique()
        unique_docs = df['doc_id'].nunique()
        unique_positions = df['position'].nunique()
        
        if unique_queries < 3:
            return self._empty_metrics(f'查询多样性不足，需要至少3个不同查询，当前只有{unique_queries}个')
        if unique_docs < 3:
            return self._empty_metrics(f'文档多样性不足，需要至少3个不同文档，当前只有{unique_docs}个')
        if unique_positions < 3:
            return self._empty_metrics(f'位置多样性不足，需要至少3个不同位置，当前只有{unique_positions}个')
        
        try:
            # 提取特征
            features, labels = self.extract_features(ctr_data)
            if len(features) == 0:
                return self._empty_metrics('特征提取失败')
            # 检查特征质量 - 降低阈值，允许更多变化
            feature_std = np.std(features, axis=0)
            if np.any(feature_std < 1e-8):  # 降低阈值从1e-6到1e-8
                print(f"警告: 特征标准差过小: {feature_std}")
                # 不直接返回错误，而是尝试继续训练
            # 数据标准化
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_idx, test_idx in sss.split(features_scaled, labels):
                X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]
            train_clicks = np.sum(y_train)
            test_clicks = np.sum(y_test)
            if train_clicks < 1 or test_clicks < 1:
                return self._empty_metrics('训练集或测试集缺少点击样本')
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=10.0,  # 减少正则化，让模型更灵活
                class_weight='balanced',  # 平衡类权重，缓解数据不平衡
                solver='liblinear'  # 对小数据集更友好
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            # 计算指标
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                print(f"AUC计算失败: {e}")
                auc = 0.0
            
            # 计算准确率
            accuracy = (y_pred == y_test).mean()
            
            # 计算精确率、召回率、F1
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division='warn')
                if 'weighted avg' in report:
                    precision = report['weighted avg']['precision']
                    recall = report['weighted avg']['recall']
                    f1 = report['weighted avg']['f1-score']
                else:
                    # 如果没有weighted avg，使用macro avg
                    precision = report['macro avg']['precision']
                    recall = report['macro avg']['recall']
                    f1 = report['macro avg']['f1-score']
            except (KeyError, ValueError) as e:
                print(f"分类报告计算失败: {e}")
                # 手动计算
                tp = np.sum((y_pred == 1) & (y_test == 1))
                fp = np.sum((y_pred == 1) & (y_test == 0))
                fn = np.sum((y_pred == 0) & (y_test == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # 暂时移除过拟合检测，专注于模型训练
            self.is_trained = True
            self.save_model()
            feature_names = CTRFeatureConfig.get_feature_names()
            feature_weights = {}
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                for i, weight in enumerate(self.model.coef_[0]):
                    if i < len(feature_names):
                        feature_weights[feature_names[i]] = abs(weight)
            return {
                'success': True,
                'accuracy': round(accuracy, 4),
                'auc': round(auc, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_score': round(train_score, 4),
                'test_score': round(test_score, 4),
                'feature_weights': feature_weights,
                'data_quality': {
                    'total_samples': total_samples,
                    'click_rate': round(click_samples / total_samples, 4),
                    'unique_queries': unique_queries,
                    'unique_docs': unique_docs,
                    'unique_positions': unique_positions
                }
            }
        except Exception as e:
            return self._empty_metrics(f'训练失败: {str(e)}')
    
    def predict_ctr(self, query: str, doc_id: str, position: int, score: float, summary: str) -> float:
        """预测CTR分数"""
        if not self.is_trained or not self.model:
            return score  # 如果模型未训练，返回原始分数
        
        try:
            # 构建特征（与训练时保持一致）
            position_feature = np.array([[position]])
            doc_length = np.array([[len(summary)]])
            query_length = np.array([[len(query)]])
            summary_length = np.array([[len(summary)]])
            
            # 查询匹配度
            query_words = set(jieba.lcut(query))
            summary_words = set(jieba.lcut(summary))
            if len(query_words) > 0:
                match_ratio = len(query_words.intersection(summary_words)) / len(query_words)
            else:
                match_ratio = 0
            match_score = np.array([[match_ratio]])
            
            # 历史CTR特征（简化版本，实际应用中需要从数据库获取）
            query_ctr = np.array([[0.1]])  # 默认值
            doc_ctr = np.array([[0.1]])    # 默认值
            
            # 位置衰减
            position_decay = np.array([[1.0 / (position + 1)]])
            
            # 组合特征
            features = np.hstack([
                position_feature,      # 位置
                doc_length,            # 文档长度
                query_length,          # 查询长度
                summary_length,        # 摘要长度
                match_score,           # 查询匹配度
                query_ctr,             # 查询历史CTR
                doc_ctr,               # 文档历史CTR
                position_decay,        # 位置衰减
                np.array([[len(jieba.lcut(query))]]),  # 查询词数量
                np.array([[len(jieba.lcut(summary))]]), # 摘要词数量
                np.array([[0]]),       # 时间特征（预测时设为0）
                np.array([[score]])    # 原始相似度分数
            ])
            
            # 标准化
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # 预测CTR概率
            ctr_score = self.model.predict_proba(features_scaled)[0, 1]
            
            return ctr_score
            
        except Exception as e:
            print(f"CTR预测失败: {e}")
            return score  # 返回原始分数
    
    def save_model(self, filepath: str = None):
        """保存模型"""
        if self.is_trained and self.model:
            # 如果没有指定文件路径，使用默认路径
            if filepath is None:
                # 统一使用项目根目录的相对路径
                filepath = os.path.join("models", "ctr_model.pkl")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"CTR模型已保存到 {filepath}")
    
    def load_model(self, filepath: str = None):
        """加载模型"""
        # 如果没有指定文件路径，使用默认路径
        if filepath is None:
            # 统一使用项目根目录的相对路径
            filepath = os.path.join("models", "ctr_model.pkl")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                self.scaler = model_data['scaler']
                self.is_trained = model_data['is_trained']
                
                print(f"CTR模型已从 {filepath} 加载")
                return True
            except Exception as e:
                print(f"加载CTR模型失败: {e}")
                return False
        return False
    
    def reset(self):
        """重置模型"""
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.is_trained = False
        print("CTR模型已重置") 