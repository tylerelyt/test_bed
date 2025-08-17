#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRE监控工具
提供系统可靠性、可用性、性能监控，以及告警和报告功能
"""

import time
import psutil
import threading
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from offline.index_service import get_index_service
from online.search_engine import SearchEngine

class SREMonitor:
    """SRE监控器"""
    
    def __init__(self, log_file: str = "logs/sre_monitor.log"):
        self.log_file = log_file
        self.monitoring = False
        self.metrics_history = []
        self.alerts = []
        self.index_service = get_index_service()
        self.search_engine = SearchEngine()
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # SRE指标阈值
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'load_avg_1min': 5.0,
            'load_avg_5min': 3.0,
            'load_avg_15min': 2.0,
            'net_connections': 1000,
            'process_count': 200,
            'search_response_time_ms': 1000,
            'search_error_rate': 0.05,
            'data_quality_score': 70
        }
    
    def start_monitoring(self, interval: int = 30):
        """开始SRE监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"SRE监控已启动，监控间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止SRE监控"""
        self.monitoring = False
        print("SRE监控已停止")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_sre_metrics()
                self.metrics_history.append(metrics)
                self._check_alerts(metrics)
                self._log_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"SRE监控过程中出现错误: {e}")
                time.sleep(interval)
    
    def _collect_sre_metrics(self) -> Dict[str, Any]:
        """收集SRE指标"""
        timestamp = datetime.now().isoformat()
        
        # 系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 系统负载
        try:
            load_avg = psutil.getloadavg()
        except:
            load_avg = (0, 0, 0)
        
        # 网络和进程
        try:
            net_connections = len(psutil.net_connections())
        except:
            net_connections = 0
        try:
            process_count = len(psutil.pids())
        except:
            process_count = 0
        
        # 索引服务指标
        index_stats = self.index_service.get_stats()
        
        # 搜索性能测试
        search_metrics = self._test_search_performance()
        
        # 数据质量检查
        data_quality = self._check_data_quality()
        
        return {
            'timestamp': timestamp,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2],
                'net_connections': net_connections,
                'process_count': process_count
            },
            'index': {
                'total_documents': index_stats.get('total_documents', 0),
                'total_terms': index_stats.get('total_terms', 0),
                'average_doc_length': index_stats.get('average_doc_length', 0)
            },
            'search': search_metrics,
            'data_quality': data_quality
        }
    
    def _test_search_performance(self) -> Dict[str, Any]:
        """测试搜索性能"""
        test_queries = ["人工智能", "机器学习", "深度学习"]
        results = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                doc_ids = self.search_engine.retrieve(query, top_k=10)
                search_results = self.search_engine.rank(query, doc_ids, top_k=5)
                response_time = (time.time() - start_time) * 1000
                
                results.append({
                    'query': query,
                    'response_time_ms': response_time,
                    'results_count': len(search_results),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'query': query,
                    'response_time_ms': 0,
                    'results_count': 0,
                    'success': False,
                    'error': str(e)
                })
        
        successful_results = [r for r in results if r['success']]
        response_times = [r['response_time_ms'] for r in successful_results]
        
        return {
            'total_queries': len(results),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'error_rate': (len(results) - len(successful_results)) / len(results) if results else 0
        }
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """检查数据质量"""
        try:
            # 运行数据质量检查
            result = subprocess.run([
                "python3", "tools/data_quality_checker.py"
            ], capture_output=True, text=True, timeout=60)
            
            output = result.stdout
            if "总体质量分数:" in output:
                score_line = [line for line in output.split('\n') if "总体质量分数:" in line][0]
                score = float(score_line.split(':')[1].split('/')[0].strip())
            else:
                score = 0
            
            return {
                'overall_score': score,
                'check_success': result.returncode == 0
            }
        except Exception as e:
            return {
                'overall_score': 0,
                'check_success': False,
                'error': str(e)
            }
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警"""
        alerts = []
        timestamp = metrics['timestamp']
        
        # CPU告警
        if metrics['system']['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'critical',
                'message': f"CPU使用率过高: {metrics['system']['cpu_percent']:.1f}%"
            })
        elif metrics['system']['cpu_percent'] > self.thresholds['cpu_percent'] * 0.8:
            alerts.append({
                'timestamp': timestamp,
                'level': 'warning',
                'message': f"CPU使用率偏高: {metrics['system']['cpu_percent']:.1f}%"
            })
        
        # 内存告警
        if metrics['system']['memory_percent'] > self.thresholds['memory_percent']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'critical',
                'message': f"内存使用率过高: {metrics['system']['memory_percent']:.1f}%"
            })
        
        # 磁盘告警
        if metrics['system']['disk_percent'] > self.thresholds['disk_percent']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'critical',
                'message': f"磁盘使用率过高: {metrics['system']['disk_percent']:.1f}%"
            })
        
        # 负载告警
        if metrics['system']['load_avg_1min'] > self.thresholds['load_avg_1min']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'critical',
                'message': f"系统负载过高: {metrics['system']['load_avg_1min']:.2f}"
            })
        
        # 搜索性能告警
        if metrics['search']['avg_response_time_ms'] > self.thresholds['search_response_time_ms']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'warning',
                'message': f"搜索响应时间过长: {metrics['search']['avg_response_time_ms']:.2f}ms"
            })
        
        if metrics['search']['error_rate'] > self.thresholds['search_error_rate']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'critical',
                'message': f"搜索错误率过高: {metrics['search']['error_rate']:.2%}"
            })
        
        # 数据质量告警
        if metrics['data_quality']['overall_score'] < self.thresholds['data_quality_score']:
            alerts.append({
                'timestamp': timestamp,
                'level': 'warning',
                'message': f"数据质量分数过低: {metrics['data_quality']['overall_score']:.1f}/100"
            })
        
        # 添加新告警
        self.alerts.extend(alerts)
        
        # 保留最近100条告警
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """记录指标到日志"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"记录SRE指标失败: {e}")
    
    def get_sre_report(self) -> Dict[str, Any]:
        """生成SRE报告"""
        if not self.metrics_history:
            return {"error": "没有监控数据"}
        
        latest_metrics = self.metrics_history[-1]
        
        # 计算健康度评分
        health_score = 100
        
        # CPU评分
        cpu_percent = latest_metrics['system']['cpu_percent']
        if cpu_percent > 80:
            health_score -= 20
        elif cpu_percent > 60:
            health_score -= 10
        
        # 内存评分
        memory_percent = latest_metrics['system']['memory_percent']
        if memory_percent > 85:
            health_score -= 20
        elif memory_percent > 70:
            health_score -= 10
        
        # 磁盘评分
        disk_percent = latest_metrics['system']['disk_percent']
        if disk_percent > 90:
            health_score -= 15
        
        # 搜索性能评分
        search_response_time = latest_metrics['search']['avg_response_time_ms']
        if search_response_time > 1000:
            health_score -= 15
        elif search_response_time > 500:
            health_score -= 10
        
        search_error_rate = latest_metrics['search']['error_rate']
        if search_error_rate > 0.05:
            health_score -= 20
        
        # 数据质量评分
        data_quality_score = latest_metrics['data_quality']['overall_score']
        if data_quality_score < 70:
            health_score -= 15
        elif data_quality_score < 80:
            health_score -= 10
        
        health_score = max(0, health_score)
        
        # 健康度等级
        if health_score >= 90:
            health_level = "优秀"
        elif health_score >= 70:
            health_level = "良好"
        elif health_score >= 50:
            health_level = "一般"
        else:
            health_level = "较差"
        
        return {
            'health_score': health_score,
            'health_level': health_level,
            'latest_metrics': latest_metrics,
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'alert_count': len(self.alerts),
            'monitoring_duration': len(self.metrics_history)
        }
    
    def get_sla_metrics(self) -> Dict[str, Any]:
        """获取SLA指标"""
        if not self.metrics_history:
            return {"error": "没有监控数据"}
        
        # 计算可用性
        total_checks = len(self.metrics_history)
        successful_checks = sum(1 for m in self.metrics_history 
                              if m['search']['error_rate'] < 0.05)
        availability = successful_checks / total_checks if total_checks > 0 else 0
        
        # 计算平均响应时间
        response_times = [m['search']['avg_response_time_ms'] for m in self.metrics_history]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 计算错误率
        error_rates = [m['search']['error_rate'] for m in self.metrics_history]
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
        
        return {
            'availability': availability,
            'avg_response_time_ms': avg_response_time,
            'avg_error_rate': avg_error_rate,
            'total_checks': total_checks,
            'successful_checks': successful_checks
        }

def main():
    """主函数"""
    print("SRE监控工具")
    print("=" * 50)
    
    monitor = SREMonitor()
    
    try:
        # 启动监控
        print("\n1. 启动SRE监控...")
        monitor.start_monitoring(interval=10)
        
        # 等待收集数据
        print("\n2. 收集监控数据...")
        time.sleep(30)
        
        # 生成报告
        print("\n3. 生成SRE报告...")
        report = monitor.get_sre_report()
        
        if 'error' in report:
            print(f"   错误: {report['error']}")
        else:
            print(f"   健康度评分: {report['health_score']}/100")
            print(f"   健康度等级: {report['health_level']}")
            print(f"   监控时长: {report['monitoring_duration']} 个数据点")
            print(f"   告警数量: {report['alert_count']}")
            
            if report['recent_alerts']:
                print("\n   最近告警:")
                for alert in report['recent_alerts'][-5:]:
                    print(f"   - [{alert['level']}] {alert['message']}")
        
        # 获取SLA指标
        print("\n4. SLA指标:")
        sla_metrics = monitor.get_sla_metrics()
        
        if 'error' not in sla_metrics:
            print(f"   可用性: {sla_metrics['availability']:.2%}")
            print(f"   平均响应时间: {sla_metrics['avg_response_time_ms']:.2f}ms")
            print(f"   平均错误率: {sla_metrics['avg_error_rate']:.2%}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断监控")
    except Exception as e:
        print(f"\n监控过程中出现错误: {e}")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("\nSRE监控已停止")

if __name__ == "__main__":
    main() 