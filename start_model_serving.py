#!/usr/bin/env python3
"""
启动Model Serving API服务 - 独立进程模式
"""

import sys
import os
import signal
import time

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from search_engine.model_service import ModelService

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号，正在关闭模型服务...")
    sys.exit(0)

def main():
    """主函数"""
    print("🚀 启动Model Serving API服务（独立进程）...")
    print("=" * 60)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建并启动服务
        model_service = ModelService()
        
        print("📋 服务信息:")
        print(f"   进程ID: {os.getpid()}")
        print(f"   地址: http://0.0.0.0:8501")
        print(f"   健康检查: http://localhost:8501/health")
        print(f"   模型列表: http://localhost:8501/v1/models")
        print("   按 Ctrl+C 停止服务")
        print("=" * 60)
        
        # 启动服务（这会阻塞进程）
        model_service.start_api_server(port=8501)
        
    except KeyboardInterrupt:
        print("\n🛑 模型服务已停止")
    except Exception as e:
        print(f"❌ 启动模型服务失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
