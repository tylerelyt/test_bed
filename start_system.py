#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLOps搜索引擎测试床 - 启动脚本
功能：启动完整的搜索引擎系统，包括数据服务、索引服务、模型服务和UI界面
"""

import subprocess
import os
import sys
import signal
import time
import importlib.util

def print_banner():
    """打印启动横幅"""
    print("=" * 60)
    print("🎯 MLOps搜索引擎测试床 - 启动脚本")
    print("=" * 60)
    print("📖 功能: 启动完整的搜索引擎系统")
    print("🔧 包含: 数据服务、索引服务、模型服务、UI界面")
    print("🌐 访问: http://localhost:7861 (或自动分配端口)")
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

def check_project_structure():
    """检查项目结构是否完整"""
    print("\n📁 步骤2: 检查项目结构")
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
    print("\n🔧 步骤3: 清理端口占用")
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
    print("\n📦 步骤4: 检查索引文件")
    print("-" * 30)
    
    if not os.path.exists('models/index_data.json'):
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

def start_system(current_dir, env):
    """启动系统"""
    print("\n🚀 步骤5: 启动MLOps系统")
    print("-" * 30)
    print("🔄 正在启动以下服务:")
    print("   📊 数据服务 (DataOps)")
    print("   📄 索引服务 (DevOps)")
    print("   🤖 模型服务 (ModelOps)")
    print("   🧪 实验服务 (ExperimentService)")
    print("   🖥️  UI界面 (Portal)")
    
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
    
    # 执行启动流程
    try:
        # 1. 检查依赖
        if not check_dependencies():
            return 1
        
        # 2. 检查项目结构
        if not check_project_structure():
            return 1
        
        # 3. 清理端口
        kill_processes_on_ports([7860, 7861, 7862, 7863, 7864, 7865])
        
        # 4. 构建索引
        if not build_index_if_needed(current_dir, env):
            return 1
        
        # 5. 启动系统
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