#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoML模块 - 自动模型搜索和超参数优化
用于教学：使用AutoML工具进行模型选择和超参数优化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入TPOT
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False
    print("⚠️ TPOT未安装，AutoML功能将受限。安装: pip install tpot")

# 尝试导入Optuna（用于超参数优化）
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna未安装，超参数优化功能将受限。安装: pip install optuna")


class AutoMLOptimizer:
    """AutoML优化器 - 自动模型搜索和超参数优化"""
    
    def __init__(self):
        self.tpot_pipeline = None
        self.best_model = None
        self.optimization_history = []
    
    def optimize_with_tpot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        generations: int = 5,
        population_size: int = 20,
        cv: int = 3,
        scoring: str = 'roc_auc',
        max_time_mins: Optional[int] = None,
        verbosity: int = 2
    ) -> Dict[str, Any]:
        """
        使用TPOT进行自动模型搜索和超参数优化
        
        Args:
            X: 特征矩阵
            y: 标签向量
            generations: 进化代数
            population_size: 种群大小
            cv: 交叉验证折数
            scoring: 评估指标
            max_time_mins: 最大运行时间（分钟）
            verbosity: 详细程度
        
        Returns:
            优化结果字典
        """
        if not TPOT_AVAILABLE:
            return {
                'error': 'TPOT未安装，请运行: pip install tpot',
                'available': False
            }
        
        try:
            # 检查数据
            if len(X) < cv * 2:
                return {
                    'error': f'数据量不足，至少需要{cv * 2}条记录，当前只有{len(X)}条',
                    'available': True
                }
            
            # 创建TPOT分类器
            tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                cv=cv,
                scoring=scoring,
                verbosity=verbosity,
                random_state=42,
                n_jobs=1,  # 避免多进程问题
                max_time_mins=max_time_mins
            )
            
            # 执行优化
            print(f"🚀 开始TPOT优化（代数: {generations}, 种群: {population_size}）...")
            tpot.fit(X, y)
            
            # 获取最佳模型
            self.tpot_pipeline = tpot.fitted_pipeline_
            self.best_model = tpot
            
            # 评估最佳模型
            best_score = tpot.score(X, y)
            
            # 获取最佳管道代码
            pipeline_code = tpot.export()
            
            return {
                'success': True,
                'best_score': float(best_score),
                'best_pipeline': str(self.tpot_pipeline),
                'pipeline_code': pipeline_code,
                'generations': generations,
                'population_size': population_size,
                'scoring': scoring
            }
            
        except Exception as e:
            return {
                'error': f'TPOT优化失败: {str(e)}',
                'available': True
            }
    
    def optimize_hyperparameters_with_optuna(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        n_trials: int = 20,
        cv: int = 3,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        使用Optuna进行超参数优化
        
        Args:
            model_class: 模型类（如LogisticRegression）
            X: 特征矩阵
            y: 标签向量
            param_space: 参数空间定义
            n_trials: 试验次数
            cv: 交叉验证折数
            scoring: 评估指标
        
        Returns:
            优化结果字典
        """
        if not OPTUNA_AVAILABLE:
            return {
                'error': 'Optuna未安装，请运行: pip install optuna',
                'available': False
            }
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            def objective(trial):
                # 从参数空间中采样
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # 创建模型
                model = model_class(**params, random_state=42)
                
                # 交叉验证
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1
                )
                
                return scores.mean()
            
            # 创建研究并优化
            study = optuna.create_study(direction='maximize', study_name='hyperparameter_optimization')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            
            # 训练最佳模型
            best_model = model_class(**best_params, random_state=42)
            best_model.fit(X_scaled, y)
            
            self.best_model = best_model
            
            return {
                'success': True,
                'best_params': best_params,
                'best_score': float(best_score),
                'n_trials': n_trials,
                'study_summary': str(study.trials_dataframe())
            }
            
        except Exception as e:
            return {
                'error': f'Optuna优化失败: {str(e)}',
                'available': True
            }
    
    def simple_grid_search(
        self,
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv: int = 3,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        简单的网格搜索（不依赖外部库）
        
        Args:
            model_class: 模型类
            X: 特征矩阵
            y: 标签向量
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评估指标
        
        Returns:
            优化结果字典
        """
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.preprocessing import StandardScaler
            
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 创建模型
            base_model = model_class(random_state=42)
            
            # 网格搜索
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                verbose=1
            )
            
            grid_search.fit(X_scaled, y)
            
            self.best_model = grid_search.best_estimator_
            
            return {
                'success': True,
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
        except Exception as e:
            return {
                'error': f'网格搜索失败: {str(e)}',
                'available': True
            }
    
    def get_best_model(self):
        """获取优化后的最佳模型"""
        return self.best_model
    
    def predict_with_best_model(self, X: np.ndarray) -> np.ndarray:
        """使用最佳模型进行预测"""
        if self.best_model is None:
            raise ValueError("尚未进行优化，请先调用优化方法")
        
        from sklearn.preprocessing import StandardScaler
        
        # 如果数据需要标准化
        if hasattr(self.best_model, 'scaler'):
            X_scaled = self.best_model.scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        return self.best_model.predict(X_scaled)
    
    def predict_proba_with_best_model(self, X: np.ndarray) -> np.ndarray:
        """使用最佳模型进行概率预测"""
        if self.best_model is None:
            raise ValueError("尚未进行优化，请先调用优化方法")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("最佳模型不支持概率预测")
        
        from sklearn.preprocessing import StandardScaler
        
        # 如果数据需要标准化
        if hasattr(self.best_model, 'scaler'):
            X_scaled = self.best_model.scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        return self.best_model.predict_proba(X_scaled)

