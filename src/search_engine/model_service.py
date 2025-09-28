import os
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from .training_tab.ctr_model import CTRModel
from .training_tab.ctr_config import CTRSampleConfig, CTRModelConfig
from flask import Flask, request, jsonify
import threading
import time


class ModelService:
    """模型服务：负责模型训练、配置管理、模型文件等"""
    
    def __init__(self, model_file: str = None):
        if model_file is None:
            model_file = os.path.join(os.getcwd(), "models", "ctr_model.pkl")
        self.model_file = model_file
        self.ctr_model = CTRModel()  # 默认使用LR模型
        self.current_model_type = "logistic_regression"
        self.model_instances = {}  # 存储不同类型的模型实例
        self._load_model()
        
        # Flask API 服务相关
        self.flask_app = None
        self.api_running = False
    
    def _load_model(self):
        """加载模型"""
        if self.ctr_model.load_model(self.model_file):
            print(f"✅ CTR模型加载成功: {self.model_file}")
        else:
            print(f"⚠️ CTR模型未找到，将使用未训练状态: {self.model_file}")
    
    def create_model_instance(self, model_type: str):
        """创建指定类型的模型实例"""
        try:
            if model_type in self.model_instances:
                return self.model_instances[model_type]
            
            model_config = CTRModelConfig.get_model_config(model_type)
            if not model_config:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            if model_type == 'logistic_regression':
                from .training_tab.ctr_model import CTRModel
                model_instance = CTRModel()
            elif model_type == 'wide_and_deep':
                from .training_tab.ctr_wide_deep_model import WideAndDeepCTRModel
                model_instance = WideAndDeepCTRModel()
            else:
                raise ValueError(f"未实现的模型类型: {model_type}")
            
            # 尝试加载对应的模型文件
            if model_type == 'logistic_regression':
                model_file = os.path.join(os.getcwd(), "models", "ctr_model.pkl")  # LR使用绝对路径
            elif model_type == 'wide_and_deep':
                model_file = os.path.join(os.getcwd(), "models", "wide_deep_ctr_model")
            else:
                model_file = os.path.join(os.getcwd(), "models", f"{model_type}_ctr_model.pkl")
            
            model_instance.load_model(model_file)
            self.model_instances[model_type] = model_instance
            
            return model_instance
            
        except Exception as e:
            print(f"创建模型实例失败: {e}")
            # 回退到默认LR模型
            from .training_tab.ctr_model import CTRModel
            return CTRModel()
    
    def switch_model(self, model_type: str):
        """切换到指定类型的模型"""
        try:
            self.ctr_model = self.create_model_instance(model_type)
            self.current_model_type = model_type
            print(f"✅ 已切换到模型: {CTRModelConfig.get_model_config(model_type).get('name', model_type)}")
            return True
        except Exception as e:
            print(f"切换模型失败: {e}")
            return False
    
    def train_model(self, data_service) -> Dict[str, Any]:
        """训练CTR模型"""
        try:
            print("🚀 开始训练CTR模型...")
            
            # 获取训练数据
            samples = data_service.get_all_samples()
            if not samples:
                return {
                    'success': False,
                    'error': '没有CTR数据用于训练'
                }
            
            # 训练模型
            result = self.ctr_model.train(samples)
            
            if result.get('success', False):
                # 保存模型
                self.save_model()
                print("✅ 模型训练完成并保存")
            else:
                print(f"❌ 模型训练失败: {result.get('error', '未知错误')}")
            
            return result
            
        except Exception as e:
            error_msg = f"训练过程中发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def save_model(self, filepath: Optional[str] = None, model_type: Optional[str] = None) -> bool:
        """保存模型"""
        try:
            model_type = model_type or self.current_model_type
            
            # 根据模型类型确定保存路径
            if filepath:
                save_path = filepath
            else:
                if model_type == 'wide_and_deep':
                    save_path = os.path.join(os.getcwd(), "models", "wide_deep_ctr_model")
                elif model_type == 'logistic_regression':
                    save_path = os.path.join(os.getcwd(), "models", "ctr_model.pkl")  # LR使用标准文件名
                else:
                    save_path = os.path.join(os.getcwd(), "models", f"{model_type}_ctr_model.pkl")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存模型
            self.ctr_model.save_model(save_path)
            
            # 保存模型信息
            info_suffix = '_info.json' if model_type != 'wide_and_deep' else '_info.json'
            info_path = save_path.replace('.pkl', info_suffix).replace('.h5', info_suffix)
            
            model_config = CTRModelConfig.get_model_config(model_type)
            model_info = {
                'model_file': save_path,
                'save_time': datetime.now().isoformat(),
                'model_type': model_config.get('name', model_type),
                'model_class': model_config.get('class', 'Unknown'),
                'feature_count': 0,  # 简化处理
                'training_samples': 0  # 简化处理
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 模型保存成功: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            return False
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """加载模型"""
        try:
            load_path = filepath or self.model_file
            if self.ctr_model.load_model(load_path):
                print(f"✅ 模型加载成功: {load_path}")
                return True
            else:
                print(f"❌ 模型加载失败: {load_path}")
                return False
        except Exception as e:
            print(f"❌ 加载模型时发生错误: {e}")
            return False
    
    def predict_ctr(self, features: Dict[str, Any], model_type: Optional[str] = None) -> float:
        """预测CTR"""
        try:
            # 始终使用指定类型的模型实例，确保使用最新训练的模型
            if model_type:
                model_instance = self.get_model_instance(model_type)
            else:
                # 如果没有指定模型类型，使用当前默认模型类型
                model_instance = self.get_model_instance(self.current_model_type)
            
            if not model_instance.is_trained:
                return 0.1  # 默认CTR
            
            # 使用指定模型的predict_ctr方法
            query = features.get('query', '')
            doc_id = features.get('doc_id', '')
            position = features.get('position', 1)
            score = features.get('score', 0.0)
            summary = features.get('summary', '')
            current_timestamp = features.get('timestamp')  # 获取时间戳参数
            
            # 检查模型是否是Wide & Deep模型，如果是则传递时间戳参数
            if hasattr(model_instance, '__class__') and 'WideAndDeep' in model_instance.__class__.__name__:
                ctr_score = model_instance.predict_ctr(query, doc_id, position, score, summary, current_timestamp)
            else:
                # 对于其他CTR模型（如LR模型），保持原有的调用方式
                ctr_score = model_instance.predict_ctr(query, doc_id, position, score, summary)
            
            return float(ctr_score)
            
        except Exception as e:
            print(f"❌ CTR预测失败: {e}")
            return 0.1
    
    def get_model_instance(self, model_type: str):
        """获取指定类型的模型实例"""
        # 每次都重新创建实例，确保加载最新的模型文件
        # 这解决了训练后模型不同步的问题
        self.model_instances[model_type] = self.create_model_instance(model_type)
        return self.model_instances[model_type]
    
    def _prepare_features(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """准备特征向量"""
        try:
            # 这里需要根据实际的模型特征进行转换
            # 简化版本，实际应该根据训练时的特征工程逻辑
            feature_vector = []
            
            # 基本特征
            feature_vector.append(features.get('position', 1))
            feature_vector.append(features.get('score', 0.0))
            feature_vector.append(features.get('match_score', 0.0))
            feature_vector.append(features.get('query_ctr', 0.1))
            feature_vector.append(features.get('doc_ctr', 0.1))
            
            # 位置衰减
            position = features.get('position', 1)
            feature_vector.append(1.0 / (position + 1))
            
            return feature_vector
            
        except Exception as e:
            print(f"❌ 特征准备失败: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            info_path = self.model_file.replace('.pkl', '_info.json')
            
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
            else:
                model_info = {
                    'model_file': self.model_file,
                    'save_time': None,
                    'model_type': 'CTR_LogisticRegression',
                    'feature_count': 0,
                    'training_samples': 0
                }
            
            # 添加当前状态
            model_info.update({
                'is_trained': self.ctr_model.is_trained,
                'model_exists': os.path.exists(self.model_file),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(self.model_file)).isoformat() if os.path.exists(self.model_file) else None
            })
            
            return model_info
            
        except Exception as e:
            print(f"❌ 获取模型信息失败: {e}")
            return {
                'model_file': self.model_file,
                'is_trained': False,
                'model_exists': False,
                'error': str(e)
            }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        try:
            if not self.ctr_model.is_trained:
                return {
                    'is_trained': False,
                    'accuracy': 0.0,
                    'auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'training_samples': 0,
                    'feature_count': 0
                }
            
            # 获取模型性能指标
            stats = {
                'is_trained': True,
                'accuracy': getattr(self.ctr_model, 'accuracy', 0.0),
                'auc': getattr(self.ctr_model, 'auc', 0.0),
                'precision': getattr(self.ctr_model, 'precision', 0.0),
                'recall': getattr(self.ctr_model, 'recall', 0.0),
                'f1': getattr(self.ctr_model, 'f1', 0.0),
                'training_samples': getattr(self.ctr_model, 'training_samples', 0),
                'feature_count': len(getattr(self.ctr_model, 'feature_names', []))
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取模型统计失败: {e}")
            return {
                'is_trained': False,
                'error': str(e)
            }
    
    def export_model(self, export_path: str) -> bool:
        """导出模型"""
        try:
            if not self.ctr_model.is_trained:
                print("❌ 模型未训练，无法导出")
                return False
            
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # 复制模型文件
            import shutil
            shutil.copy2(self.model_file, export_path)
            
            # 复制模型信息
            info_src = self.model_file.replace('.pkl', '_info.json')
            info_dst = export_path.replace('.pkl', '_info.json')
            if os.path.exists(info_src):
                shutil.copy2(info_src, info_dst)
            
            print(f"✅ 模型导出成功: {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型导出失败: {e}")
            return False
    
    def import_model(self, import_path: str) -> bool:
        """导入模型"""
        try:
            if not os.path.exists(import_path):
                print(f"❌ 模型文件不存在: {import_path}")
                return False
            
            # 复制模型文件
            import shutil
            shutil.copy2(import_path, self.model_file)
            
            # 复制模型信息
            info_src = import_path.replace('.pkl', '_info.json')
            info_dst = self.model_file.replace('.pkl', '_info.json')
            if os.path.exists(info_src):
                shutil.copy2(info_src, info_dst)
            
            # 重新加载模型
            self._load_model()
            
            print(f"✅ 模型导入成功: {import_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型导入失败: {e}")
            return False
    
    def delete_model(self) -> bool:
        """删除模型"""
        try:
            if os.path.exists(self.model_file):
                os.remove(self.model_file)
                print(f"✅ 模型文件删除成功: {self.model_file}")
            
            info_path = self.model_file.replace('.pkl', '_info.json')
            if os.path.exists(info_path):
                os.remove(info_path)
                print(f"✅ 模型信息文件删除成功: {info_path}")
            
            # 重置模型
            self.ctr_model = CTRModel()
            
            return True
            
        except Exception as e:
            print(f"❌ 删除模型失败: {e}")
            return False
    
    def validate_training_data(self, data_service) -> Dict[str, Any]:
        """验证训练数据"""
        try:
            samples = data_service.get_all_samples()
            
            if not samples:
                return {
                    'valid': False,
                    'issues': ['没有CTR数据'],
                    'recommendations': ['进行一些搜索实验生成数据']
                }
            
            df = pd.DataFrame(samples)
            issues = []
            recommendations = []
            
            # 检查数据量
            if len(df) < 10:
                issues.append(f"数据量不足，只有{len(df)}条记录")
                recommendations.append("需要至少10条记录")
            
            # 检查点击数据
            if 'clicked' in df.columns:
                click_count = df['clicked'].sum()
                if click_count < 2:
                    issues.append(f"点击数据不足，只有{click_count}次点击")
                    recommendations.append("需要至少2次点击")
            
            # 检查查询多样性
            if 'query' in df.columns:
                unique_queries = df['query'].nunique()
                if unique_queries < 3:
                    issues.append(f"查询多样性不足，只有{unique_queries}个不同查询")
                    recommendations.append("需要至少3个不同查询")
            
            # 检查文档多样性
            if 'doc_id' in df.columns:
                unique_docs = df['doc_id'].nunique()
                if unique_docs < 3:
                    issues.append(f"文档多样性不足，只有{unique_docs}个不同文档")
                    recommendations.append("需要至少3个不同文档")
            
            return {
                'valid': len(issues) == 0,
                'total_samples': len(df),
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f'验证过程中发生错误: {str(e)}'],
                'recommendations': ['检查数据格式']
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if not self.ctr_model.is_trained:
                return {}
            
            if hasattr(self.ctr_model, 'model') and self.ctr_model.model and hasattr(self.ctr_model.model, 'coef_'):
                feature_names = getattr(self.ctr_model, 'feature_names', [])
                coefficients = self.ctr_model.model.coef_[0]
                
                importance = {}
                for i, name in enumerate(feature_names):
                    if i < len(coefficients):
                        importance[name] = float(abs(coefficients[i]))
                
                # 按重要性排序
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                return sorted_importance
            
            return {}
            
        except Exception as e:
            print(f"❌ 获取特征重要性失败: {e}")
            return {}
    
    def start_api_server(self, host="0.0.0.0", port=8501, debug=False):
        """启动Flask API服务器（独立进程模式）"""
        try:
            if self.api_running:
                print("⚠️ API服务器已在运行")
                return True
            
            self.flask_app = Flask(__name__)
            self._setup_api_routes()
            
            self.api_running = True
            print(f"🚀 Model Serving API启动在 {host}:{port}")
            print("📋 可用接口:")
            print("   - 健康检查: http://localhost:8501/health")
            print("   - 模型列表: http://localhost:8501/v1/models")
            print("   - 预测接口: http://localhost:8501/v1/models/<model_name>/predict")
            print("   - 批量预测: http://localhost:8501/v1/models/<model_name>/batch_predict")
            print("=" * 50)
            
            # 直接运行Flask服务器（独立进程模式）
            self.flask_app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
            
        except Exception as e:
            print(f"❌ 启动API服务器失败: {e}")
            return False
    
    def stop_api_server(self):
        """停止Flask API服务器"""
        self.api_running = False
        print("🛑 API服务器已停止")
    
    def _setup_api_routes(self):
        """设置API路由"""
        
        @self.flask_app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({
                "status": "healthy",
                "model_type": self.current_model_type,
                "model_trained": self.ctr_model.is_trained
            })
        
        @self.flask_app.route('/v1/models', methods=['GET'])
        def list_models():
            """列出所有模型"""
            models = []
            for model_type in ['logistic_regression', 'wide_and_deep']:
                try:
                    model_instance = self.get_model_instance(model_type)
                    models.append({
                        "name": model_type,
                        "status": "loaded" if model_instance.is_trained else "unloaded",
                        "type": "pickle" if model_type == 'logistic_regression' else "tensorflow"
                    })
                except:
                    models.append({
                        "name": model_type,
                        "status": "error",
                        "type": "pickle" if model_type == 'logistic_regression' else "tensorflow"
                    })
            
            return jsonify({"model": models})
        
        @self.flask_app.route('/v1/models/<model_name>', methods=['GET'])
        def get_model_info(model_name):
            """获取特定模型信息"""
            try:
                model_instance = self.get_model_instance(model_name)
                return jsonify({
                    "model": {
                        "name": model_name,
                        "status": "loaded" if model_instance.is_trained else "unloaded",
                        "type": "pickle" if model_name == 'logistic_regression' else "tensorflow"
                    }
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 404
        
        @self.flask_app.route('/v1/models/<model_name>/predict', methods=['POST'])
        def predict(model_name):
            """模型预测"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                # 提取输入数据
                inputs = data.get('inputs', {})
                if not inputs:
                    return jsonify({"error": "No inputs provided"}), 400
                
                # 执行预测
                ctr_score = self.predict_ctr(inputs, model_name)
                
                return jsonify({
                    "outputs": {"ctr_score": ctr_score}
                })
                
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.flask_app.route('/v1/models/<model_name>/batch_predict', methods=['POST'])
        def batch_predict(model_name):
            """批量预测"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                # 提取输入数据
                inputs_list = data.get('inputs', [])
                if not inputs_list:
                    return jsonify({"error": "No inputs provided"}), 400
                
                # 执行批量预测
                results = []
                for inputs in inputs_list:
                    ctr_score = self.predict_ctr(inputs, model_name)
                    results.append({"ctr_score": ctr_score})
                
                return jsonify({
                    "outputs": results
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def is_api_running(self):
        """检查API服务器是否运行"""
        return self.api_running 