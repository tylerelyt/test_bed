#!/usr/bin/env python3
"""
模型服务进程管理器
用于管理模型服务的独立进程
"""

import os
import sys
import signal
import subprocess
import time
import psutil
from typing import Optional, List
import requests

class ModelServiceManager:
    """模型服务进程管理器"""
    
    def __init__(self, port: int = 8501):
        self.port = port
        self.service_url = f"http://localhost:{port}"
        self.script_path = os.path.join(os.path.dirname(__file__), '..', 'start_model_serving.py')
        self.process: Optional[subprocess.Popen] = None
    
    def is_running(self) -> bool:
        """检查模型服务是否正在运行"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_process_info(self) -> Optional[dict]:
        """获取模型服务进程信息"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'start_model_serving.py' in ' '.join(proc.info['cmdline']):
                    return {
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'status': proc.status(),
                        'memory': proc.memory_info().rss / 1024 / 1024,  # MB
                        'cpu_percent': proc.cpu_percent()
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def start(self) -> bool:
        """启动模型服务"""
        if self.is_running():
            print("✅ 模型服务已在运行")
            return True
        
        try:
            print("🚀 启动模型服务独立进程...")
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(self.script_path)
            )
            
            # 等待服务启动
            print("⏳ 等待服务启动...")
            for i in range(10):  # 最多等待10秒
                time.sleep(1)
                if self.is_running():
                    print("✅ 模型服务启动成功")
                    return True
                print(f"   等待中... ({i+1}/10)")
            
            print("❌ 模型服务启动超时")
            return False
            
        except Exception as e:
            print(f"❌ 启动模型服务失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止模型服务"""
        if not self.is_running():
            print("⚠️ 模型服务未运行")
            return True
        
        try:
            # 查找并终止进程
            process_info = self.get_process_info()
            if process_info:
                pid = process_info['pid']
                print(f"🛑 停止模型服务进程 (PID: {pid})...")
                
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(2)
                    
                    # 检查是否已停止
                    if not self.is_running():
                        print("✅ 模型服务已停止")
                        return True
                    else:
                        # 强制终止
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(1)
                        print("✅ 模型服务已强制停止")
                        return True
                        
                except ProcessLookupError:
                    print("✅ 模型服务进程已不存在")
                    return True
            else:
                print("⚠️ 未找到模型服务进程")
                return True
                
        except Exception as e:
            print(f"❌ 停止模型服务失败: {e}")
            return False
    
    def restart(self) -> bool:
        """重启模型服务"""
        print("🔄 重启模型服务...")
        if self.stop():
            time.sleep(1)
            return self.start()
        return False
    
    def status(self) -> dict:
        """获取模型服务状态"""
        is_running = self.is_running()
        process_info = self.get_process_info()
        
        status = {
            'running': is_running,
            'port': self.port,
            'url': self.service_url
        }
        
        if process_info:
            status.update(process_info)
        
        return status
    
    def health_check(self) -> dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response': response.json()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型服务进程管理器')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'health'], 
                       help='操作类型')
    parser.add_argument('--port', type=int, default=8501, help='服务端口')
    
    args = parser.parse_args()
    
    manager = ModelServiceManager(port=args.port)
    
    if args.action == 'start':
        manager.start()
    elif args.action == 'stop':
        manager.stop()
    elif args.action == 'restart':
        manager.restart()
    elif args.action == 'status':
        status = manager.status()
        print("📊 模型服务状态:")
        print(f"   运行状态: {'✅ 运行中' if status['running'] else '❌ 未运行'}")
        print(f"   端口: {status['port']}")
        print(f"   地址: {status['url']}")
        if 'pid' in status:
            print(f"   进程ID: {status['pid']}")
            print(f"   内存使用: {status['memory']:.1f} MB")
            print(f"   CPU使用: {status['cpu_percent']:.1f}%")
    elif args.action == 'health':
        health = manager.health_check()
        print("🏥 健康检查:")
        print(f"   状态: {health['status']}")
        if health['status'] == 'healthy':
            print(f"   响应: {health['response']}")
        else:
            print(f"   错误: {health['error']}")

if __name__ == "__main__":
    main()
