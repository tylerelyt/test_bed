#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLOps搜索引擎测试床 - 启动脚本
功能：启动完整的搜索引擎系统，包括MCP服务器、数据服务、索引服务、模型服务和UI界面
"""

import subprocess
import os
import sys
import signal
import time
import importlib.util
import asyncio
from typing import List, Optional
from urllib import request, error

def load_env_file():
    """加载 .env 文件中的环境变量"""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📄 加载环境变量文件: {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"✅ 加载环境变量: {key}")
    else:
        print(f"⚠️  环境变量文件不存在: {env_file}")

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("🎯 MLOps搜索引擎测试床 - 启动脚本")
    print("=" * 60)
    print("📖 功能: 启动完整的搜索引擎系统")
    print("🔧 包含: MCP服务器、数据服务、索引服务、模型服务、UI界面")
    print("🌐 访问: http://localhost:7861 (主系统)")
    print("🔗 MCP: http://localhost:3001/mcp (统一服务器)")
    print("🛑 停止: 按 Ctrl+C 或关闭终端")
    print("=" * 60)

def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("\n🔍 步骤1: 检查系统依赖")
    print("-" * 30)
    
    required_packages = [
        ('gradio', 'gradio>=4.0.0'),
        ('pandas', 'pandas>=1.5.0'),
        ('numpy', 'numpy>=1.21.0'),
        ('sklearn', 'scikit-learn>=1.2.0'),
        ('jieba', 'jieba>=0.42.1')
    ]
    
    missing_packages = []
    for package, requirement in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(requirement)
                print(f"❌ 缺少依赖: {requirement}")
            else:
                print(f"✅ 已安装: {package}")
        except ImportError:
            missing_packages.append(requirement)
            print(f"❌ 缺少依赖: {requirement}")
    
    if missing_packages:
        print(f"\n❌ 发现 {len(missing_packages)} 个缺少的依赖包")
        print("🔧 请运行以下命令安装依赖:")
        print("   pip install -r requirements.txt")
        print("\n或者安装单个包:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_api_keys():
    """检查API密钥配置"""
    print("\n🔑 步骤2: 检查API密钥配置")
    print("-" * 30)
    
    # 检查DashScope API密钥
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        print(f"✅ API密钥已加载: {dashscope_key[:10]}...")
        
        # 测试API密钥是否有效
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=dashscope_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            # 简单测试调用
            response = client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10
            )
            print("✅ DashScope API密钥验证成功")
            return True
            
        except Exception as e:
            print(f"❌ DashScope API密钥验证失败: {str(e)}")
            print("🔧 请检查API密钥是否正确配置")
            return False
    else:
        print("❌ DASHSCOPE_API_KEY 环境变量未设置")
        print("🔧 请在 .env 文件中设置 DASHSCOPE_API_KEY=your_api_key")
        return False

def check_project_structure():
    """检查项目结构是否完整"""
    print("\n📁 步骤3: 检查项目结构")
    print("-" * 30)
    
    required_files = [
        'src/search_engine/portal.py',
        'src/search_engine/data_service.py',
        'src/search_engine/index_service.py',
        'src/search_engine/model_service.py',
        'requirements.txt'
    ]
    
    required_dirs = [
        'src/search_engine',
        'models',
        'data',
        'tools'
    ]
    
    missing_items = []
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(file_path)
            print(f"❌ 缺少文件: {file_path}")
        else:
            print(f"✅ 文件存在: {file_path}")
    
    # 检查目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(dir_path)
            print(f"❌ 缺少目录: {dir_path}")
        else:
            print(f"✅ 目录存在: {dir_path}")
    
    if missing_items:
        print(f"\n❌ 发现 {len(missing_items)} 个缺少的文件/目录")
        print("请检查项目结构是否完整")
        return False
    
    print("✅ 项目结构检查通过")
    return True

def kill_processes_on_ports(ports):
    """清理指定端口的进程"""
    print("\n🔧 步骤4: 清理端口占用")
    print("-" * 30)
    
    for port in ports:
        try:
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"🔄 终止进程 {pid} (端口 {port})")
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(1)
                        except ProcessLookupError:
                            pass
                        except Exception as e:
                            print(f"⚠️  终止进程失败: {e}")
            else:
                print(f"✅ 端口 {port} 未被占用")
        except Exception as e:
            print(f"⚠️  检查端口 {port} 失败: {e}")

def build_index_if_needed(current_dir, env):
    """如果需要，构建索引"""
    print("\n📦 步骤5: 检查索引文件")
    print("-" * 30)
    
    # 若存在预置文档文件，则优先使用服务层自动加载，无需强制离线构建
    preloaded_path = os.path.join('data', 'preloaded_documents.json')
    if not os.path.exists('models/index_data.json'):
        if os.path.exists(preloaded_path):
            print("📄 检测到预置文档文件，将由服务层在首次初始化时自动加载: data/preloaded_documents.json")
            print("✅ 跳过离线构建，等待服务层创建索引")
        else:
            print("📄 索引文件不存在，开始构建...")
            print("⏳ 这可能需要几分钟时间，请耐心等待...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "search_engine.index_tab.offline_index"], 
                    check=True, 
                    cwd=current_dir,
                    env=env
                )
                print("✅ 离线索引构建完成")
            except subprocess.CalledProcessError as e:
                print(f"❌ 离线索引构建失败: {e}")
                print("💡 建议: 检查数据文件是否存在，或运行 python -m search_engine.index_tab.offline_index")
                return False
    else:
        print("✅ 索引文件已存在，跳过构建")
    
    return True

def start_mcp_server():
    """启动统一MCP服务器"""
    print("\n🚀 步骤6: 启动统一MCP服务器")
    print("-" * 30)
    
    # 若已运行则直接使用现有实例
    mcp_url = "http://localhost:3001/mcp"
    try:
        req = request.Request(mcp_url, method="GET")
        with request.urlopen(req, timeout=2) as resp:
            if 200 <= resp.status < 300:
                print("✅ 检测到已运行的统一MCP服务器，直接复用: http://localhost:3001/mcp")
                # 返回一个非 None 的占位对象表示成功
                return {"status": "already_running", "url": mcp_url}
    except Exception:
        pass

    # 检查MCP服务器文件是否存在
    mcp_server_file = "src/search_engine/mcp/dynamic_mcp_server.py"
    if not os.path.exists(mcp_server_file):
        print(f"❌ MCP服务器文件不存在: {mcp_server_file}")
        return None
    
    # 启动动态MCP服务器
    print("🔄 正在启动动态MCP服务器...")
    mcp_process = subprocess.Popen([
        sys.executable, 
        mcp_server_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待服务器启动
    print("⏳ 等待MCP服务器启动...")
    time.sleep(2)
    
    # 启动后再次探测HTTP健康
    try:
        with request.urlopen(request.Request(mcp_url, method="GET"), timeout=3) as resp:
            if 200 <= resp.status < 300:
                print("✅ 统一MCP服务器启动成功")
                print("📍 MCP服务器地址: http://localhost:3001/mcp")
                return mcp_process
    except Exception:
        pass
    
    # 兜底：检查进程状态
    if mcp_process.poll() is None:
        print("✅ 统一MCP服务器进程已启动（等待就绪）")
        print("📍 MCP服务器地址: http://localhost:3001/mcp")
        return mcp_process
    
    stdout, stderr = mcp_process.communicate()
    print(f"❌ 统一MCP服务器启动失败")
    if stderr:
        try:
            print(f"错误输出: {stderr.decode()}")
        except Exception:
            print("错误输出: <无法解码>")
    return None

def start_system(current_dir, env):
    """启动系统"""
    print("\n🚀 步骤6: 启动MLOps系统")
    print("-" * 30)
    print("🔄 正在启动以下服务:")
    print("   📊 数据服务 (DataOps)")
    print("   📄 索引服务 (DevOps)")
    print("   🤖 模型服务 (ModelOps)")
    print("   🧪 实验服务 (ExperimentService)")
    print("   🖥️  UI界面 (Portal)")
    print("   🔗 MCP集成 (MCP Tab)")
    
    try:
        print("\n🌐 启动Web界面...")
        print("⏳ 正在加载，请稍等...")
        print("💡 系统启动完成后，浏览器将自动打开或显示访问地址")
        
        subprocess.run(
            [sys.executable, "-m", "search_engine.portal"], 
            cwd=current_dir,
            env=env
        )
    except KeyboardInterrupt:
        print("\n")
        print("🛑 用户中断，正在优雅关闭...")
        print("✅ 系统已停止")
    except Exception as e:
        print(f"\n❌ 启动系统失败: {e}")
        print("💡 建议:")
        print("   1. 检查依赖是否完整: pip install -r requirements.txt")
        print("   2. 检查端口是否被占用: lsof -i :7861")
        print("   3. 查看详细错误信息并检查日志文件")

def main():
    """主函数"""
    print_banner()
    
    # 加载环境变量
    load_env_file()
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置Python路径
    src_path = os.path.join(current_dir, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = src_path + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = src_path
    
    # 确保 API 密钥环境变量被传递
    if 'DASHSCOPE_API_KEY' in os.environ:
        env['DASHSCOPE_API_KEY'] = os.environ['DASHSCOPE_API_KEY']
        print(f"✅ API密钥已加载: {os.environ['DASHSCOPE_API_KEY'][:15]}...")
    else:
        print("⚠️ 未找到 DASHSCOPE_API_KEY 环境变量")
    
    # 执行启动流程
    try:
        # 1. 检查依赖
        if not check_dependencies():
            return 1
        
        # 2. 检查API密钥
        if not check_api_keys():
            return 1
        
        # 3. 检查项目结构
        if not check_project_structure():
            return 1
        
        # 4. 清理端口
        kill_processes_on_ports([7860, 7861, 7862, 7863, 7864, 7865])
        
        # 5. 构建索引
        if not build_index_if_needed(current_dir, env):
            return 1
        
        # 6. 启动MCP服务器
        mcp_process = start_mcp_server()
        if mcp_process is None:
            print("❌ 统一MCP服务器启动失败，无法继续启动主系统。")
            return 1
        
        # 7. 启动系统
        start_system(current_dir, env)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 启动过程被中断")
        return 1
    except Exception as e:
        print(f"\n❌ 启动过程发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 