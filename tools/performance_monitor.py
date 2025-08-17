#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控工具
用于监控搜索引擎测试床的性能指标和潜在风险
"""

import time
import psutil
import threading
import json
import os
from typing import Dict, List, Any
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from offline.index_service import get_index_service
from online.search_engine import SearchEngine

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.log_file = log_file
        self.monitoring = False
        self.metrics = []
        self.index_service = get_index_service()
        self.search_engine = SearchEngine()
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def start_monitoring(self, interval: int = 5):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"性能监控已启动，监控间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        print("性能监控已停止")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                self._log_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"监控过程中出现错误: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        # 系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 索引服务指标
        index_stats = self.index_service.get_stats()
        
        # 时间戳
        timestamp = datetime.now().isoformat()
        
        return {
            'timestamp': timestamp,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            },
            'index': {
                'total_documents': index_stats.get('total_documents', 0),
                'total_terms': index_stats.get('total_terms', 0),
                'average_doc_length': index_stats.get('average_doc_length', 0)
            }
        }
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """记录指标到日志"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"记录指标失败: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.metrics:
            return {"error": "没有监控数据"}
        
        # 计算统计信息
        cpu_values = [m['system']['cpu_percent'] for m in self.metrics]
        memory_values = [m['system']['memory_percent'] for m in self.metrics]
        
        return {
            'monitoring_duration': len(self.metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'latest_metrics': self.metrics[-1] if self.metrics else None
        }

class RiskDetector:
    """风险检测器"""
    
    def __init__(self):
        self.risk_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'response_time_ms': 1000,
            'error_rate': 0.05
        }
    
    def detect_risks(self, metrics: Dict[str, Any]) -> List[str]:
        """检测风险"""
        risks = []
        
        # CPU 使用率风险
        if metrics['system']['cpu_percent'] > self.risk_thresholds['cpu_percent']:
            risks.append(f"CPU使用率过高: {metrics['system']['cpu_percent']}%")
        
        # 内存使用率风险
        if metrics['system']['memory_percent'] > self.risk_thresholds['memory_percent']:
            risks.append(f"内存使用率过高: {metrics['system']['memory_percent']}%")
        
        # 磁盘使用率风险
        if metrics['system']['disk_percent'] > self.risk_thresholds['disk_percent']:
            risks.append(f"磁盘使用率过高: {metrics['system']['disk_percent']}%")
        
        return risks

class LoadTester:
    """负载测试器"""
    
    def __init__(self):
        self.index_service = get_index_service()
        self.search_engine = SearchEngine()
    
    def test_search_performance(self, queries: List[str], iterations: int = 100) -> Dict[str, Any]:
        """测试搜索性能"""
        results = []
        start_time = time.time()
        
        for i in range(iterations):
            query = queries[i % len(queries)]
            query_start = time.time()
            
            try:
                # 执行搜索
                doc_ids = self.search_engine.retrieve(query, top_k=10)
                results_ranked = self.search_engine.rank(query, doc_ids, top_k=5)
                
                query_time = (time.time() - query_start) * 1000  # 转换为毫秒
                results.append({
                    'query': query,
                    'response_time_ms': query_time,
                    'results_count': len(results_ranked),
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
        
        total_time = time.time() - start_time
        
        # 计算统计信息
        successful_results = [r for r in results if r['success']]
        response_times = [r['response_time_ms'] for r in successful_results]
        
        return {
            'total_queries': len(results),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
            'total_time_seconds': total_time,
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'queries_per_second': len(results) / total_time,
            'error_rate': (len(results) - len(successful_results)) / len(results)
        }
    
    def test_concurrent_search(self, queries: List[str], concurrent_users: int = 10) -> Dict[str, Any]:
        """测试并发搜索"""
        import concurrent.futures
        
        def search_worker(query):
            start_time = time.time()
            try:
                doc_ids = self.search_engine.retrieve(query, top_k=10)
                results = self.search_engine.rank(query, doc_ids, top_k=5)
                response_time = (time.time() - start_time) * 1000
                return {'success': True, 'response_time_ms': response_time, 'results_count': len(results)}
            except Exception as e:
                return {'success': False, 'error': str(e), 'response_time_ms': 0}
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(search_worker, queries[i % len(queries)]) 
                      for i in range(concurrent_users * 10)]  # 每个用户10次搜索
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r['success']]
        response_times = [r['response_time_ms'] for r in successful_results]
        
        return {
            'concurrent_users': concurrent_users,
            'total_queries': len(results),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
            'total_time_seconds': total_time,
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'queries_per_second': len(results) / total_time,
            'error_rate': (len(results) - len(successful_results)) / len(results)
        }

def main():
    """主函数"""
    print("搜索引擎测试床性能监控工具")
    print("=" * 50)
    
    # 创建监控器
    monitor = PerformanceMonitor()
    detector = RiskDetector()
    load_tester = LoadTester()
    
    # 测试查询
    test_queries = [
        "人工智能",
        "机器学习",
        "深度学习",
        "自然语言处理",
        "计算机视觉",
        "强化学习",
        "神经网络",
        "知识图谱"
    ]
    
    try:
        # 1. 启动性能监控
        print("\n1. 启动性能监控...")
        monitor.start_monitoring(interval=2)
        
        # 2. 运行负载测试
        print("\n2. 运行搜索性能测试...")
        performance_result = load_tester.test_search_performance(test_queries, iterations=50)
        
        print(f"   总查询数: {performance_result['total_queries']}")
        print(f"   成功查询: {performance_result['successful_queries']}")
        print(f"   失败查询: {performance_result['failed_queries']}")
        print(f"   平均响应时间: {performance_result['avg_response_time_ms']:.2f}ms")
        print(f"   最大响应时间: {performance_result['max_response_time_ms']:.2f}ms")
        print(f"   查询吞吐量: {performance_result['queries_per_second']:.2f} QPS")
        print(f"   错误率: {performance_result['error_rate']:.2%}")
        
        # 3. 运行并发测试
        print("\n3. 运行并发搜索测试...")
        concurrent_result = load_tester.test_concurrent_search(test_queries, concurrent_users=5)
        
        print(f"   并发用户数: {concurrent_result['concurrent_users']}")
        print(f"   总查询数: {concurrent_result['total_queries']}")
        print(f"   平均响应时间: {concurrent_result['avg_response_time_ms']:.2f}ms")
        print(f"   查询吞吐量: {concurrent_result['queries_per_second']:.2f} QPS")
        print(f"   错误率: {concurrent_result['error_rate']:.2%}")
        
        # 4. 等待一段时间收集监控数据
        print("\n4. 收集监控数据...")
        time.sleep(10)
        
        # 5. 生成性能报告
        print("\n5. 生成性能报告...")
        report = monitor.get_performance_report()
        
        print(f"   监控时长: {report['monitoring_duration']} 个数据点")
        print(f"   CPU使用率 - 平均: {report['cpu']['avg']:.1f}%, 最大: {report['cpu']['max']:.1f}%")
        print(f"   内存使用率 - 平均: {report['memory']['avg']:.1f}%, 最大: {report['memory']['max']:.1f}%")
        
        # 6. 风险检测
        if report['latest_metrics']:
            risks = detector.detect_risks(report['latest_metrics'])
            if risks:
                print("\n6. 检测到的风险:")
                for risk in risks:
                    print(f"   - {risk}")
            else:
                print("\n6. 未检测到明显风险")
        
        # 7. 性能评估
        print("\n7. 性能评估:")
        if performance_result['avg_response_time_ms'] < 100:
            print("   - 响应时间: 优秀 (< 100ms)")
        elif performance_result['avg_response_time_ms'] < 500:
            print("   - 响应时间: 良好 (< 500ms)")
        else:
            print("   - 响应时间: 需要优化 (> 500ms)")
        
        if performance_result['error_rate'] < 0.01:
            print("   - 错误率: 优秀 (< 1%)")
        elif performance_result['error_rate'] < 0.05:
            print("   - 错误率: 良好 (< 5%)")
        else:
            print("   - 错误率: 需要关注 (> 5%)")
        
        if report['cpu']['avg'] < 50:
            print("   - CPU使用率: 优秀 (< 50%)")
        elif report['cpu']['avg'] < 80:
            print("   - CPU使用率: 良好 (< 80%)")
        else:
            print("   - CPU使用率: 需要优化 (> 80%)")
        
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("\n性能监控已停止")

if __name__ == "__main__":
    main() 