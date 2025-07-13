import os
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    algorithms: List[str]
    metrics: List[str]
    duration_days: int
    traffic_split: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: str = "draft"  # draft, running, completed, stopped


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    algorithm: str
    metrics: Dict[str, float]
    sample_count: int
    click_count: int
    click_rate: float
    timestamp: str


class ExperimentService:
    """MLOps实验管理服务"""
    
    def __init__(self, data_file: str = "data/experiments.json"):
        self.data_file = data_file
        self.experiments = {}
        self.results = {}
        self._load_experiments()
    
    def _load_experiments(self):
        """加载实验数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.experiments = data.get('experiments', {})
                    self.results = data.get('results', {})
                print(f"✅ 实验数据加载成功: {len(self.experiments)}个实验")
            else:
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                print("📝 创建新的实验数据文件")
        except Exception as e:
            print(f"❌ 加载实验数据失败: {e}")
    
    def _save_experiments(self):
        """保存实验数据"""
        try:
            data = {
                'experiments': self.experiments,
                'results': self.results,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存实验数据失败: {e}")
    
    def create_experiment(self, config: ExperimentConfig) -> Optional[str]:
        """创建新实验"""
        try:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            experiment_data = {
                'id': experiment_id,
                'config': asdict(config),
                'created_time': datetime.now().isoformat(),
                'status': 'draft'
            }
            
            self.experiments[experiment_id] = experiment_data
            self._save_experiments()
            
            print(f"✅ 实验创建成功: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            print(f"❌ 创建实验失败: {e}")
            return None
    
    def start_experiment(self, experiment_id: str) -> bool:
        """启动实验"""
        try:
            if experiment_id not in self.experiments:
                print(f"❌ 实验不存在: {experiment_id}")
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment['status'] != 'draft':
                print(f"❌ 实验状态不允许启动: {experiment['status']}")
                return False
            
            # 更新实验状态
            experiment['status'] = 'running'
            experiment['start_time'] = datetime.now().isoformat()
            experiment['end_time'] = (
                datetime.now() + timedelta(days=experiment['config']['duration_days'])
            ).isoformat()
            
            self._save_experiments()
            print(f"✅ 实验启动成功: {experiment_id}")
            return True
            
        except Exception as e:
            print(f"❌ 启动实验失败: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """停止实验"""
        try:
            if experiment_id not in self.experiments:
                print(f"❌ 实验不存在: {experiment_id}")
                return False
            
            experiment = self.experiments[experiment_id]
            if experiment['status'] != 'running':
                print(f"❌ 实验状态不允许停止: {experiment['status']}")
                return False
            
            # 更新实验状态
            experiment['status'] = 'completed'
            experiment['end_time'] = datetime.now().isoformat()
            
            self._save_experiments()
            print(f"✅ 实验停止成功: {experiment_id}")
            return True
            
        except Exception as e:
            print(f"❌ 停止实验失败: {e}")
            return False
    
    def record_result(self, experiment_id: str, algorithm: str, metrics: Dict[str, float], 
                     sample_count: int, click_count: int) -> bool:
        """记录实验结果"""
        try:
            if experiment_id not in self.experiments:
                print(f"❌ 实验不存在: {experiment_id}")
                return False
            
            result_id = f"{experiment_id}_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                algorithm=algorithm,
                metrics=metrics,
                sample_count=sample_count,
                click_count=click_count,
                click_rate=click_count / sample_count if sample_count > 0 else 0.0,
                timestamp=datetime.now().isoformat()
            )
            
            self.results[result_id] = asdict(result)
            self._save_experiments()
            
            print(f"✅ 实验结果记录成功: {result_id}")
            return True
            
        except Exception as e:
            print(f"❌ 记录实验结果失败: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> List[Dict[str, Any]]:
        """获取实验结果"""
        try:
            results = []
            for result_id, result_data in self.results.items():
                if result_data['experiment_id'] == experiment_id:
                    results.append(result_data)
            
            # 按时间排序
            results.sort(key=lambda x: x['timestamp'])
            return results
            
        except Exception as e:
            print(f"❌ 获取实验结果失败: {e}")
            return []
    
    def compare_algorithms(self, experiment_id: str) -> Dict[str, Any]:
        """对比算法效果"""
        try:
            results = self.get_experiment_results(experiment_id)
            if not results:
                return {}
            
            # 按算法分组
            algorithm_results = {}
            for result in results:
                algorithm = result['algorithm']
                if algorithm not in algorithm_results:
                    algorithm_results[algorithm] = []
                algorithm_results[algorithm].append(result)
            
            # 计算平均指标
            comparison = {}
            for algorithm, algo_results in algorithm_results.items():
                if not algo_results:
                    continue
                
                # 计算平均值
                avg_metrics = {}
                for metric in algo_results[0]['metrics'].keys():
                    values = [r['metrics'][metric] for r in algo_results if metric in r['metrics']]
                    avg_metrics[metric] = sum(values) / len(values) if values else 0.0
                
                # 计算总体统计
                total_samples = sum(r['sample_count'] for r in algo_results)
                total_clicks = sum(r['click_count'] for r in algo_results)
                avg_click_rate = total_clicks / total_samples if total_samples > 0 else 0.0
                
                comparison[algorithm] = {
                    'avg_metrics': avg_metrics,
                    'total_samples': total_samples,
                    'total_clicks': total_clicks,
                    'avg_click_rate': avg_click_rate,
                    'result_count': len(algo_results)
                }
            
            return comparison
            
        except Exception as e:
            print(f"❌ 对比算法效果失败: {e}")
            return {}
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验摘要"""
        try:
            if experiment_id not in self.experiments:
                return {}
            
            experiment = self.experiments[experiment_id]
            results = self.get_experiment_results(experiment_id)
            comparison = self.compare_algorithms(experiment_id)
            
            # 计算实验统计
            total_results = len(results)
            algorithms_tested = len(comparison)
            
            # 确定最佳算法
            best_algorithm = None
            best_click_rate = 0.0
            for algorithm, stats in comparison.items():
                if stats['avg_click_rate'] > best_click_rate:
                    best_click_rate = stats['avg_click_rate']
                    best_algorithm = algorithm
            
            summary = {
                'experiment_id': experiment_id,
                'name': experiment['config']['name'],
                'description': experiment['config']['description'],
                'status': experiment['status'],
                'start_time': experiment.get('start_time'),
                'end_time': experiment.get('end_time'),
                'total_results': total_results,
                'algorithms_tested': algorithms_tested,
                'best_algorithm': best_algorithm,
                'best_click_rate': best_click_rate,
                'comparison': comparison
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ 获取实验摘要失败: {e}")
            return {}
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出实验"""
        try:
            experiments = []
            for exp_id, exp_data in self.experiments.items():
                if status is None or exp_data['status'] == status:
                    summary = self.get_experiment_summary(exp_id)
                    experiments.append(summary)
            
            # 按创建时间排序
            experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
            return experiments
            
        except Exception as e:
            print(f"❌ 列出实验失败: {e}")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        try:
            if experiment_id not in self.experiments:
                print(f"❌ 实验不存在: {experiment_id}")
                return False
            
            # 删除实验
            del self.experiments[experiment_id]
            
            # 删除相关结果
            result_ids_to_delete = []
            for result_id, result_data in self.results.items():
                if result_data['experiment_id'] == experiment_id:
                    result_ids_to_delete.append(result_id)
            
            for result_id in result_ids_to_delete:
                del self.results[result_id]
            
            self._save_experiments()
            print(f"✅ 实验删除成功: {experiment_id}")
            return True
            
        except Exception as e:
            print(f"❌ 删除实验失败: {e}")
            return False
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """获取实验统计信息"""
        try:
            total_experiments = len(self.experiments)
            total_results = len(self.results)
            
            # 按状态统计
            status_counts = {}
            for exp_data in self.experiments.values():
                status = exp_data['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # 按算法统计
            algorithm_counts = {}
            for result_data in self.results.values():
                algorithm = result_data['algorithm']
                algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
            
            stats = {
                'total_experiments': total_experiments,
                'total_results': total_results,
                'status_distribution': status_counts,
                'algorithm_distribution': algorithm_counts,
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取实验统计失败: {e}")
            return {}
    
    def export_experiment_data(self, experiment_id: str, filepath: str) -> bool:
        """导出实验数据"""
        try:
            if experiment_id not in self.experiments:
                print(f"❌ 实验不存在: {experiment_id}")
                return False
            
            experiment = self.experiments[experiment_id]
            results = self.get_experiment_results(experiment_id)
            comparison = self.compare_algorithms(experiment_id)
            
            export_data = {
                'experiment': experiment,
                'results': results,
                'comparison': comparison,
                'export_time': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 实验数据导出成功: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 导出实验数据失败: {e}")
            return False 