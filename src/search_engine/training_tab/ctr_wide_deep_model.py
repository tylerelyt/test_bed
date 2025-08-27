#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wide & Deep CTR模型模块 - 基于TensorFlow的Wide & Deep Learning
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import jieba
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .ctr_config import CTRFeatureConfig, CTRTrainingConfig

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow未安装，Wide & Deep模型将不可用")

class WideAndDeepCTRModel:
    """Wide & Deep CTR模型类"""
    
    def __init__(self):
        self.model = None
        self.wide_scaler = None
        self.deep_scaler = None
        self.categorical_encoders = {}
        self.is_trained = False
        self.feature_columns = None
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow未安装，无法使用Wide & Deep模型")
    
    def _check_tensorflow(self):
        """检查TensorFlow是否可用"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow")
    
    def extract_features(self, ctr_data: List[Dict[str, Any]], is_training: bool = True, train_indices: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        从CTR数据中提取Wide和Deep特征（修复数据泄露问题）
        
        Args:
            ctr_data: CTR数据列表
            is_training: 是否为训练模式
            train_indices: 训练集索引（用于避免数据泄露）
        
        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: (特征字典, 标签)
        """
        if not ctr_data:
            return {}, np.array([])
        
        # 转换为DataFrame
        df = pd.DataFrame(ctr_data)
        
        # === Wide特征 (线性特征) ===
        wide_features = []
        
        # 1. 位置特征
        position_features = df['position'].values.reshape(-1, 1)
        wide_features.append(position_features)
        
        # 2. 位置衰减特征
        position_decay = 1.0 / (position_features.flatten() + 1)
        wide_features.append(position_decay.reshape(-1, 1))
        
        # 3. 原始相似度分数
        score_features = df['score'].values.reshape(-1, 1)
        wide_features.append(score_features)
        
        # 4. 查询匹配度
        match_scores = []
        for _, row in df.iterrows():
            query_words = set(jieba.lcut(row['query']))
            summary_words = set(jieba.lcut(row['summary']))
            if len(query_words) > 0:
                match_ratio = len(query_words.intersection(summary_words)) / len(query_words)
            else:
                match_ratio = 0
            match_scores.append(match_ratio)
        wide_features.append(np.array(match_scores).reshape(-1, 1))
        
        # 5. 历史CTR特征（修复数据泄露）
        query_ctr_features = []
        doc_ctr_features = []
        
        # 保持原始索引到排序索引的映射
        df_with_orig_idx = df.reset_index()
        df_with_orig_idx['orig_idx'] = df_with_orig_idx['index']
        df_sorted = df_with_orig_idx.sort_values('timestamp').reset_index(drop=True)
        
        for idx, row in df_sorted.iterrows():
            # 查询历史CTR - 只使用当前样本之前的数据
            query = row['query']
            query_history = df_sorted.loc[:idx-1]
            
            # 如果是训练模式且有训练集索引，只使用训练集数据计算历史CTR
            if is_training and train_indices is not None:
                query_history = query_history[query_history['orig_idx'].isin(train_indices)]
            
            query_history_filtered = query_history[query_history['query'] == query]
            
            if len(query_history_filtered) > 0:
                query_ctr = query_history_filtered['clicked'].mean()
            else:
                query_ctr = 0.1
            query_ctr_features.append(query_ctr)
            
            # 文档历史CTR - 只使用当前样本之前的数据
            doc_id = row['doc_id']
            doc_history = df_sorted.loc[:idx-1]
            
            # 如果是训练模式且有训练集索引，只使用训练集数据计算历史CTR
            if is_training and train_indices is not None:
                doc_history = doc_history[doc_history['orig_idx'].isin(train_indices)]
            
            doc_history_filtered = doc_history[doc_history['doc_id'] == doc_id]
            
            if len(doc_history_filtered) > 0:
                doc_ctr = doc_history_filtered['clicked'].mean()
            else:
                doc_ctr = 0.1
            doc_ctr_features.append(doc_ctr)
        
        # 创建排序索引到原始索引的映射
        sorted_to_orig = {i: row['orig_idx'] for i, row in df_sorted.iterrows()}
        
        # 按原始顺序重新排列特征（保持时间因果关系）
        query_ctr_reordered = [0.1] * len(df)
        doc_ctr_reordered = [0.1] * len(df)
        
        for sorted_idx, orig_idx in sorted_to_orig.items():
            query_ctr_reordered[orig_idx] = query_ctr_features[sorted_idx]
            doc_ctr_reordered[orig_idx] = doc_ctr_features[sorted_idx]
        
        wide_features.append(np.array(query_ctr_reordered).reshape(-1, 1))
        wide_features.append(np.array(doc_ctr_reordered).reshape(-1, 1))
        
        # 合并Wide特征
        wide_features_combined = np.hstack(wide_features)
        
        # === Deep特征 (非线性特征) ===
        deep_features = []
        
        # 1. 长度特征
        doc_lengths = df['summary'].str.len().values.reshape(-1, 1)
        query_lengths = df['query'].str.len().values.reshape(-1, 1)
        summary_lengths = df['summary'].str.len().values.reshape(-1, 1)
        deep_features.extend([doc_lengths, query_lengths, summary_lengths])
        
        # 2. 词数量特征
        query_word_counts = []
        summary_word_counts = []
        for _, row in df.iterrows():
            query_word_count = len(jieba.lcut(row['query']))
            summary_word_count = len(jieba.lcut(row['summary']))
            query_word_counts.append(query_word_count)
            summary_word_counts.append(summary_word_count)
        
        deep_features.append(np.array(query_word_counts).reshape(-1, 1))
        deep_features.append(np.array(summary_word_counts).reshape(-1, 1))
        
        # 3. 时间特征
        time_features = []
        for _, row in df.iterrows():
            try:
                timestamp_str = str(row['timestamp'])
                time_value = sum(ord(c) for c in timestamp_str) % 1000
                time_features.append(time_value)
            except:
                time_features.append(0)
        deep_features.append(np.array(time_features).reshape(-1, 1))
        
        # 4. 交叉特征 (位置×分数，查询长度×匹配度等)
        position_score_cross = position_features.flatten() * score_features.flatten()
        query_len_match_cross = np.array(query_lengths).flatten() * np.array(match_scores)
        
        deep_features.append(position_score_cross.reshape(-1, 1))
        deep_features.append(query_len_match_cross.reshape(-1, 1))
        
        # 合并Deep特征
        deep_features_combined = np.hstack(deep_features)
        
        # === 分类特征 (嵌入特征) ===
        # 查询哈希特征 - 使用abs确保非负数
        query_hashes = []
        for query in df['query']:
            query_hash = abs(hash(query)) % 1000  # 简单哈希到1000个桶，确保非负
            query_hashes.append(query_hash)
        
        # 文档ID哈希特征 - 使用abs确保非负数
        doc_hashes = []
        for doc_id in df['doc_id']:
            doc_hash = abs(hash(doc_id)) % 1000  # 简单哈希到1000个桶，确保非负
            doc_hashes.append(doc_hash)
        
        # 位置分组特征
        position_groups = []
        for pos in df['position']:
            if pos <= 3:
                group = 0  # 头部
            elif pos <= 10:
                group = 1  # 中部
            else:
                group = 2  # 尾部
            position_groups.append(group)
        
        # 标签
        labels = df['clicked'].values
        
        # 返回特征字典
        features = {
            'wide': wide_features_combined,
            'deep': deep_features_combined,
            'query_hash': np.array(query_hashes),
            'doc_hash': np.array(doc_hashes),
            'position_group': np.array(position_groups)
        }
        
        return features, labels
    
    def _build_model(self, wide_dim: int, deep_dim: int, vocab_sizes: Dict[str, int]):
        """构建Wide & Deep模型"""
        self._check_tensorflow()
        
        # Wide部分输入
        wide_input = keras.Input(shape=(wide_dim,), name='wide')
        
        # Deep部分输入
        deep_input = keras.Input(shape=(deep_dim,), name='deep')
        
        # 分类特征输入
        query_input = keras.Input(shape=(), name='query_hash', dtype='int32')
        doc_input = keras.Input(shape=(), name='doc_hash', dtype='int32')
        position_input = keras.Input(shape=(), name='position_group', dtype='int32')
        
        # 嵌入层
        query_embedding = layers.Embedding(vocab_sizes['query_hash'], 8, name='query_embedding')(query_input)
        doc_embedding = layers.Embedding(vocab_sizes['doc_hash'], 8, name='doc_embedding')(doc_input)
        position_embedding = layers.Embedding(vocab_sizes['position_group'], 4, name='position_embedding')(position_input)
        
        # 展平嵌入向量
        query_flat = layers.Flatten()(query_embedding)
        doc_flat = layers.Flatten()(doc_embedding)
        position_flat = layers.Flatten()(position_embedding)
        
        # Deep部分：连接数值特征和嵌入特征
        deep_concat = layers.Concatenate()([deep_input, query_flat, doc_flat, position_flat])
        
        # Deep部分的全连接层
        deep_hidden1 = layers.Dense(128, activation='relu', name='deep_hidden1')(deep_concat)
        deep_dropout1 = layers.Dropout(0.2)(deep_hidden1)
        deep_hidden2 = layers.Dense(64, activation='relu', name='deep_hidden2')(deep_dropout1)
        deep_dropout2 = layers.Dropout(0.2)(deep_hidden2)
        deep_hidden3 = layers.Dense(32, activation='relu', name='deep_hidden3')(deep_dropout2)
        
        # Wide & Deep合并
        wide_deep_concat = layers.Concatenate()([wide_input, deep_hidden3])
        
        # 输出层
        output = layers.Dense(1, activation='sigmoid', name='output')(wide_deep_concat)
        
        # 创建模型
        model = keras.Model(
            inputs=[wide_input, deep_input, query_input, doc_input, position_input],
            outputs=output,
            name='wide_and_deep_ctr'
        )
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def _empty_metrics(self, error_msg):
        """返回空的评估指标"""
        return {
            'error': error_msg,
            'auc': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'train_samples': 0,
            'test_samples': 0,
            'train_score': 0.0,
            'test_score': 0.0,
            'model_type': 'wide_deep',
            'feature_importance': {},
            'data_quality': {}
        }
    
    def train(self, ctr_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """训练Wide & Deep模型"""
        if not TF_AVAILABLE:
            return self._empty_metrics('TensorFlow未安装，无法训练Wide & Deep模型')
        
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
        
        # 检查数据多样性（与LR模型保持一致）
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
            # 首先进行数据分割（在特征提取之前）
            df = pd.DataFrame(ctr_data)
            labels_temp = df['clicked'].values
            
            # 数据分割
            indices = np.arange(len(labels_temp))
            train_indices, test_indices = train_test_split(
                indices, test_size=0.3, random_state=42, stratify=labels_temp
            )
            
            # 提取特征（传入训练集索引以避免数据泄露）
            features, labels = self.extract_features(ctr_data, is_training=True, train_indices=train_indices)
            if len(features) == 0:
                return self._empty_metrics('特征提取失败')
            
            # 数据标准化（只在训练集上fit）
            self.wide_scaler = StandardScaler()
            self.deep_scaler = StandardScaler()
            
            # 先分割再标准化
            train_wide = features['wide'][train_indices]
            test_wide = features['wide'][test_indices]
            train_deep = features['deep'][train_indices]
            test_deep = features['deep'][test_indices]
            
            # 在训练集上fit，在训练集和测试集上transform
            train_wide_scaled = self.wide_scaler.fit_transform(train_wide)
            test_wide_scaled = self.wide_scaler.transform(test_wide)
            train_deep_scaled = self.deep_scaler.fit_transform(train_deep)
            test_deep_scaled = self.deep_scaler.transform(test_deep)
            
            # 更新特征
            features['wide'][train_indices] = train_wide_scaled
            features['wide'][test_indices] = test_wide_scaled
            features['deep'][train_indices] = train_deep_scaled
            features['deep'][test_indices] = test_deep_scaled
            
            # 计算词汇表大小 - 使用最大值+1来确保覆盖所有可能的索引
            vocab_sizes = {
                'query_hash': max(1000, np.max(features['query_hash']) + 1),  # 确保至少1000个桶
                'doc_hash': max(1000, np.max(features['doc_hash']) + 1),    # 确保至少1000个桶
                'position_group': max(3, np.max(features['position_group']) + 1)  # 确保至少3个分组
            }
            
            # 获取配置参数
            from .ctr_config import CTRModelConfig
            config = CTRModelConfig.get_model_config('wide_and_deep')
            params = config.get('params', {})
            
            # 构建模型
            wide_dim = features['wide'].shape[1]
            deep_dim = features['deep'].shape[1]
            self.model = self._build_model(wide_dim, deep_dim, vocab_sizes)
            
            # 准备训练数据
            X_train = {
                'wide': features['wide'][train_indices],
                'deep': features['deep'][train_indices],
                'query_hash': features['query_hash'][train_indices],
                'doc_hash': features['doc_hash'][train_indices],
                'position_group': features['position_group'][train_indices]
            }
            
            X_test = {
                'wide': features['wide'][test_indices],
                'deep': features['deep'][test_indices],
                'query_hash': features['query_hash'][test_indices],
                'doc_hash': features['doc_hash'][test_indices],
                'position_group': features['position_group'][test_indices]
            }
            
            y_train = labels[train_indices]
            y_test = labels[test_indices]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                epochs=params.get('epochs', 30),
                batch_size=params.get('batch_size', 64),
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=params.get('early_stopping_patience', 8), 
                        restore_best_weights=True,
                        monitor='val_loss'
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6,
                        monitor='val_loss'
                    )
                ]
            )
            
            # 预测和评估
            y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 计算指标
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except Exception as e:
                print(f"AUC计算失败: {e}")
                auc = 0.0
            
            accuracy = (y_pred == y_test).mean()
            
            # 计算精确率、召回率、F1
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division='warn')
                if 'weighted avg' in report:
                    precision = report['weighted avg']['precision']
                    recall = report['weighted avg']['recall']
                    f1 = report['weighted avg']['f1-score']
                else:
                    precision = report['macro avg']['precision']
                    recall = report['macro avg']['recall']
                    f1 = report['macro avg']['f1-score']
            except (KeyError, ValueError) as e:
                print(f"分类报告计算失败: {e}")
                tp = np.sum((y_pred == 1) & (y_test == 1))
                fp = np.sum((y_pred == 1) & (y_test == 0))
                fn = np.sum((y_pred == 0) & (y_test == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 训练和测试得分
            train_loss, train_acc, train_auc = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_acc, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
            
            self.is_trained = True
            self.save_model()
            
            return {
                'success': True,
                'accuracy': round(accuracy, 4),
                'auc': round(auc, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'train_samples': len(y_train),
                'test_samples': len(y_test),
                'train_score': round(train_acc, 4),
                'test_score': round(test_acc, 4),
                'model_type': 'wide_deep',
                'feature_importance': self._get_feature_importance(),
                'data_quality': {
                    'total_samples': total_samples,
                    'click_rate': round(click_samples / total_samples, 4),
                    'unique_queries': df['query'].nunique(),
                    'unique_docs': df['doc_id'].nunique(),
                    'unique_positions': df['position'].nunique()
                }
            }
            
        except Exception as e:
            return self._empty_metrics(f'训练失败: {str(e)}')
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（简化版本）"""
        if not self.model:
            return {}
        
        # 对于神经网络，特征重要性比较复杂，这里返回一个简化版本
        return {
            'wide_features': 0.6,
            'deep_features': 0.4,
            'query_embedding': 0.15,
            'doc_embedding': 0.15,
            'position_embedding': 0.1
        }
    
    def predict_ctr(self, query: str, doc_id: str, position: int, score: float, summary: str, current_timestamp: str = None) -> float:
        """预测CTR分数"""
        if not self.is_trained or not self.model:
            return score  # 如果模型未训练，返回原始分数
        
        try:
            # 使用当前时间戳或生成一个合理的时间戳
            if current_timestamp is None:
                current_timestamp = datetime.now().isoformat()
            
            # 构建单个样本的特征
            sample_data = [{
                'query': query,
                'doc_id': doc_id,
                'position': position,
                'score': score,
                'summary': summary,
                'clicked': 0,  # 预测时不需要真实标签
                'timestamp': current_timestamp  # 使用真实的时间戳
            }]
            
            # 预测时不使用训练集索引限制
            features, _ = self.extract_features(sample_data, is_training=False)
            if len(features) == 0:
                return score
            
            # 标准化特征
            if self.wide_scaler and self.deep_scaler:
                features['wide'] = self.wide_scaler.transform(features['wide'])
                features['deep'] = self.deep_scaler.transform(features['deep'])
            
            # 准备预测输入
            pred_input = {
                'wide': features['wide'],
                'deep': features['deep'],
                'query_hash': features['query_hash'],
                'doc_hash': features['doc_hash'],
                'position_group': features['position_group']
            }
            
            # 预测CTR
            ctr_prob = self.model.predict(pred_input, verbose=0)[0][0]
            
            # 将CTR概率转换为分数加权
            return float(score * (1 + ctr_prob))
            
        except Exception as e:
            print(f"Wide & Deep预测失败: {e}")
            return score
    
    def save_model(self, model_path: str = "models/wide_deep_ctr_model"):
        """保存模型"""
        if not self.model:
            return False
        
        try:
            os.makedirs("models", exist_ok=True)
            
            # 保存TensorFlow模型
            self.model.save(f"{model_path}.h5")
            
            # 保存预处理器
            with open(f"{model_path}_preprocessors.pkl", 'wb') as f:
                pickle.dump({
                    'wide_scaler': self.wide_scaler,
                    'deep_scaler': self.deep_scaler,
                    'categorical_encoders': self.categorical_encoders,
                    'is_trained': self.is_trained
                }, f)
            
            return True
        except Exception as e:
            print(f"保存Wide & Deep模型失败: {e}")
            return False
    
    def load_model(self, model_path: str = "models/wide_deep_ctr_model"):
        """加载模型"""
        if not TF_AVAILABLE:
            print("TensorFlow未安装，无法加载Wide & Deep模型")
            return False
        
        try:
            if os.path.exists(f"{model_path}.h5"):
                self.model = keras.models.load_model(f"{model_path}.h5")
                
                # 加载预处理器
                if os.path.exists(f"{model_path}_preprocessors.pkl"):
                    with open(f"{model_path}_preprocessors.pkl", 'rb') as f:
                        data = pickle.load(f)
                        self.wide_scaler = data.get('wide_scaler')
                        self.deep_scaler = data.get('deep_scaler')
                        self.categorical_encoders = data.get('categorical_encoders', {})
                        self.is_trained = data.get('is_trained', False)
                
                return True
        except Exception as e:
            print(f"加载Wide & Deep模型失败: {e}")
        
        return False
