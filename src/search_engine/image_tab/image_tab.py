#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片检索页面 - 基于CLIP的图搜图和文搜图界面
"""

import gradio as gr
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import threading
import subprocess
import platform


# ==================== 全局任务状态管理 ====================

# 任务执行状态
_task_running = False
_task_stop_flag = False
_task_lock = threading.Lock()
_keyboard_listener = None  # 键盘监听器


def set_task_running(running: bool):
    """设置任务运行状态"""
    global _task_running
    with _task_lock:
        _task_running = running


def is_task_running() -> bool:
    """检查任务是否正在运行"""
    global _task_running
    with _task_lock:
        return _task_running


def set_task_stop_flag(stop: bool):
    """设置任务停止标志"""
    global _task_stop_flag
    with _task_lock:
        _task_stop_flag = stop


def should_stop_task() -> bool:
    """检查是否应该停止任务"""
    global _task_stop_flag
    with _task_lock:
        return _task_stop_flag


def _on_esc_pressed():
    """ESC 键按下时的回调函数"""
    if is_task_running():
        print("\n⚠️  检测到 ESC 键，正在中断任务...")
        set_task_stop_flag(True)
        
        # 尝试显示通知（如果函数已定义）
        try:
            _show_autopilot_notification("⚠️ ESC 键中断\n\n任务正在停止...")
        except:
            pass


def start_keyboard_listener():
    """启动键盘监听（监听 ESC 键）"""
    global _keyboard_listener
    
    # 如果已经有监听器在运行，先停止
    if _keyboard_listener is not None:
        try:
            _keyboard_listener.stop()
        except:
            pass
    
    try:
        from pynput import keyboard
        
        def on_press(key):
            try:
                # 检测 ESC 键
                if key == keyboard.Key.esc:
                    _on_esc_pressed()
            except Exception as e:
                pass
        
        # 创建并启动监听器
        _keyboard_listener = keyboard.Listener(on_press=on_press)
        _keyboard_listener.daemon = True  # 设置为守护线程
        _keyboard_listener.start()
        
        # 检查监听器是否正常工作
        import time
        time.sleep(0.1)
        if _keyboard_listener.is_alive():
            print("⌨️  键盘监听已启动（按 ESC 可中断任务）")
            return True
        else:
            print("⚠️  键盘监听启动失败")
            _show_permission_guide()
            return False
        
    except ImportError:
        print("⚠️  pynput 未安装，无法启用 ESC 键中断功能")
        print("💡 安装方法: pip install pynput")
        return False
    except Exception as e:
        print(f"⚠️  启动键盘监听失败: {e}")
        if "Accessibility" in str(e) or "permission" in str(e).lower():
            _show_permission_guide()
        return False


def _show_permission_guide():
    """显示权限设置指南"""
    if platform.system() == "Darwin":  # macOS
        guide = """
╔══════════════════════════════════════════════════════════════╗
║           ⚠️  需要辅助功能权限才能使用 ESC 键中断            ║
╚══════════════════════════════════════════════════════════════╝

📋 macOS 权限设置步骤：

1. 打开 "系统设置" (System Settings)
2. 进入 "隐私与安全性" → "辅助功能" (Privacy & Security → Accessibility)
3. 找到您的终端应用（Terminal、iTerm2 或 Python）
4. 确保已勾选授予权限
5. 如果没有看到应用，点击 "+" 添加
6. 重启此程序

💡 提示: 
   - 授予权限后需要重启应用才能生效
   - 如果不授予权限，可以使用 Gradio 界面停止任务
   - 本地模式需要该权限才能监听 ESC 键

🔗 详细说明: https://support.apple.com/zh-cn/guide/mac-help/mh43185/mac
"""
        print(guide)
        
        # 尝试打开系统设置（需要用户手动导航）
        try:
            import subprocess
            subprocess.Popen(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'])
            print("✅ 已尝试打开系统设置，请按照上述步骤授予权限")
        except:
            pass
    else:
        print("💡 Linux/Windows 通常不需要额外权限，如有问题请检查系统安全设置")


def stop_keyboard_listener():
    """停止键盘监听"""
    global _keyboard_listener
    
    if _keyboard_listener is not None:
        try:
            _keyboard_listener.stop()
            _keyboard_listener = None
            print("⌨️  键盘监听已停止")
        except Exception as e:
            print(f"⚠️  停止键盘监听失败: {e}")


def _show_autopilot_notification(message: str):
    """
    在本地执行时通过 OS 原生方式显示 Autopilot 状态。

    - macOS: 使用 osascript display dialog，1 秒后自动关闭
    - Linux: 使用 notify-send 系统通知
    - Windows: 使用 msg 命令
    - 失败时静默忽略，不影响主流程
    """
    try:
        if not message:
            return
        
        import platform
        import subprocess
        
        system = platform.system()
        safe_message = str(message)[:100]  # 限制长度
        
        if system == "Darwin":  # macOS
            # 转义特殊字符
            safe_message = safe_message.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$')
            # 使用 osascript display dialog，2 秒后自动关闭
            script = f'''
display dialog "🤖 Autopilot\\n\\n{safe_message}" ¬
    with title "Autopilot 正在执行" ¬
    buttons {{"执行中..."}} ¬
    default button 1 ¬
    giving up after 2
'''
            # 使用 Popen 非阻塞执行，让对话框自动显示和关闭
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL
            )
        
        elif system == "Linux":
            # Linux 使用 notify-send
            try:
                subprocess.Popen(
                    ["notify-send", "-t", "1500", "🤖 Autopilot", safe_message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except:
                pass
        
        elif system == "Windows":
            # Windows 使用 msg 命令（需要管理员权限，可能不可用）
            try:
                subprocess.Popen(
                    ["msg", "*", f"Autopilot: {safe_message}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except:
                pass
        
    except Exception:
        # 通知失败不影响任务执行
        pass


# ==================== GUI-Agent 辅助函数 ====================

def get_osworld_container_port():
    """获取运行中的 OSWorld 容器端口"""
    try:
        import subprocess
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}\t{{.Ports}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        containers = result.stdout.strip().split('\n')
        
        for line in containers:
            if any(keyword in line.lower() for keyword in ['osworld', 'gifted', 'happysixd']):
                # 提取端口映射 5000->5000/tcp 或 0.0.0.0:55000->5000/tcp
                if '5000' in line and '->' in line:
                    parts = line.split()
                    for part in parts:
                        if '5000' in part and '->' in part:
                            # 格式: 0.0.0.0:55000->5000/tcp 或 55000->5000/tcp
                            host_part = part.split('->')[0]
                            if ':' in host_part:
                                host_port = host_part.split(':')[-1]
                            else:
                                host_port = host_part
                            return int(host_port)
        return None
    except Exception:
        return None


def find_existing_container():
    """查找已存在的 OSWorld 容器（包括已停止的）"""
    try:
        import subprocess
        import docker
        
        # 先尝试使用 docker 库
        try:
            client = docker.from_env()
            # 查找所有容器（包括已停止的）
            all_containers = client.containers.list(all=True)
            
            for container in all_containers:
                container_name = container.name.lower()
                if any(keyword in container_name for keyword in ['osworld', 'gifted', 'happysixd']):
                    return container
        except:
            pass
        
        # 备用方案：使用命令行
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.Status}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        containers = result.stdout.strip().split('\n')
        
        for line in containers:
            if line.strip():
                name = line.split('\t')[0]
                if any(keyword in name.lower() for keyword in ['osworld', 'gifted', 'happysixd']):
                    try:
                        client = docker.from_env()
                        return client.containers.get(name)
                    except:
                        pass
        
        return None
    except Exception:
        return None


def start_vm_container():
    """启动 OSWorld Docker 容器"""
    try:
        try:
            import docker
        except ImportError:
            return "❌ Docker 库未安装\n💡 请安装: pip install docker"
        
        import time
        from pathlib import Path
        
        try:
            client = docker.from_env()
        except docker.errors.DockerException as e:
            return f"❌ Docker 连接失败: {str(e)}\n💡 请确保 Docker Desktop 已启动"
        
        # 查找已存在的容器
        existing_container = find_existing_container()
        
        if existing_container:
            # 如果容器已存在，检查状态
            existing_container.reload()
            if existing_container.status == 'running':
                # 获取端口
                port = get_osworld_container_port()
                if port:
                    return f"✅ 容器已在运行中\n🌐 API 端口: {port}\n💡 无需重复启动"
            
            # 容器存在但已停止，尝试启动
            try:
                existing_container.start()
                time.sleep(3)  # 等待容器启动
                
                # 获取端口
                port = get_osworld_container_port()
                if port:
                    # 尝试等待 API 就绪
                    import requests
                    api_ready = False
                    for i in range(10):
                        try:
                            response = requests.get(f'http://localhost:{port}/screenshot', timeout=2)
                            if response.status_code == 200:
                                api_ready = True
                                break
                        except:
                            pass
                        time.sleep(1)
                    
                    status_msg = f"✅ 容器已启动（重启）\n🌐 API 端口: {port}"
                    if api_ready:
                        status_msg += "\n✅ API 服务器已就绪"
                    else:
                        status_msg += "\n⏳ 等待服务就绪（约 1-2 分钟）..."
                    status_msg += f"\n💡 桌面环境可能需要额外时间初始化，如动作执行失败请等待 5-10 分钟后重试"
                    return status_msg
                else:
                    return f"✅ 容器已启动，但端口检测失败\n💡 请稍后刷新状态查看"
            except Exception as e:
                return f"❌ 启动现有容器失败: {str(e)}"
        
        # 容器不存在，创建新容器
        container_name = "osworld-vm-test"
        image_name = "happysixd/osworld-docker"
        
        # 检查镜像是否存在
        try:
            client.images.get(image_name)
        except docker.errors.ImageNotFound:
            return f"❌ Docker 镜像不存在: {image_name}\n💡 请先拉取镜像: docker pull {image_name}"
        
        # 检查虚拟机镜像文件
        vm_image_path = Path("data/osworld_vm/Ubuntu.qcow2")
        if not vm_image_path.exists():
            return f"❌ 虚拟机镜像文件不存在: {vm_image_path}\n💡 请先运行: python test_osworld_vm_screenshot.py 下载镜像"
        
        # 端口配置
        vnc_port = 58006
        server_port = 55000
        chrome_port = 59222
        vlc_port = 58080
        
        # 环境变量
        environment = {
            "DISK_SIZE": "8G",
            "RAM_SIZE": "2G",
            "CPU_CORES": "2",
            "KVM": "N"  # macOS 不支持 KVM
        }
        
        try:
            container = client.containers.run(
                image_name,
                name=container_name,
                environment=environment,
                cap_add=["NET_ADMIN"],
                volumes={
                    str(vm_image_path.absolute()): {
                        "bind": "/System.qcow2",
                        "mode": "ro"
                    }
                },
                ports={
                    8006: vnc_port,
                    5000: server_port,
                    9222: chrome_port,
                    8080: vlc_port
                },
                detach=True
            )
            
            time.sleep(2)  # 等待容器启动
            
            # 尝试等待并初始化桌面环境
            init_message = ""
            try:
                # 等待 API 服务器启动
                import requests
                for i in range(30):  # 最多等待 30 秒
                    try:
                        response = requests.get(f'http://localhost:{server_port}/screenshot', timeout=2)
                        if response.status_code == 200:
                            init_message = "\n✅ API 服务器已就绪"
                            break
                    except:
                        pass
                    time.sleep(1)
            except:
                pass
            
            return f"""✅ 容器已创建并启动

📦 容器 ID: {container.short_id}
🌐 API 端口: {server_port}
🖥️  VNC 端口: {vnc_port}
{init_message}

⏳ 虚拟机正在启动中，这可能需要 2-5 分钟
💡 提示：
   - 桌面环境需要额外时间初始化
   - 如果动作执行失败，请等待 5-10 分钟后重试
   - 可通过 VNC 查看桌面状态: http://localhost:{vnc_port}"""
            
        except docker.errors.APIError as e:
            if "port is already allocated" in str(e).lower():
                return f"❌ 端口已被占用\n💡 请检查是否有其他容器在使用端口 {server_port}"
            return f"❌ 容器创建失败: {str(e)}"
        except Exception as e:
            import traceback
            return f"❌ 启动失败: {str(e)}\n详情: {traceback.format_exc()[:200]}"
            
    except Exception as e:
        import traceback
        return f"❌ 启动失败: {str(e)}\n详情: {traceback.format_exc()[:200]}"


def initialize_gui_agent(provider_name, os_type, model_name, api_key, base_url):
    """初始化 GUI-Agent 环境和代理"""
    try:
        # 导入 GUI-Agent 服务
        from ..gui_agent_service import gui_agent_service
        
        # 使用提供的配置初始化
        result = gui_agent_service.initialize(
            provider_name=provider_name,
            os_type=os_type,
            model=model_name,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None
        )
        
        if result['status'] == 'success':
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
            
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"


def get_vm_status():
    """获取虚拟机状态（通过 HTTP API）"""
    try:
        import requests
        
        # 获取容器端口
        port = get_osworld_container_port()
        
        if not port:
            # 检查是否有已停止的容器
            existing_container = find_existing_container()
            if existing_container:
                existing_container.reload()
                container_status = existing_container.status
                container_name = existing_container.name
                
                if container_status == 'exited' or container_status == 'stopped':
                    return f"""
                    <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px;">
                        <h4>🖥️ 虚拟机状态</h4>
                        <ul style="list-style: none; padding-left: 0;">
                            <li>🟡 <strong>状态:</strong> 容器已停止</li>
                            <li>📦 <strong>容器名称:</strong> {container_name}</li>
                            <li>💡 <strong>提示:</strong> 点击「启动虚拟机」按钮启动容器</li>
                        </ul>
                    </div>
                    """
            
            return """
            <div style="background-color: #ffebee; padding: 15px; border-radius: 8px;">
                <h4>🖥️ 虚拟机状态</h4>
                <ul style="list-style: none; padding-left: 0;">
                    <li>🔴 <strong>状态:</strong> 未运行</li>
                    <li>💡 <strong>提示:</strong> 点击「启动虚拟机」按钮启动容器</li>
                </ul>
            </div>
            """
        
        # 检查 API 是否可用
        api_available = False
        api_error = None
        try:
            response = requests.get(f'http://localhost:{port}/', timeout=5)
            api_available = response.status_code == 200
        except requests.exceptions.Timeout:
            api_error = "连接超时（可能正在启动）"
        except requests.exceptions.ConnectionError:
            api_error = "无法连接"
        except Exception as e:
            api_error = str(e)[:50]
        
        # 获取容器详细信息
        container_info = ""
        existing_container = find_existing_container()
        if existing_container:
            try:
                existing_container.reload()
                container_info = f"<li>📦 <strong>容器:</strong> {existing_container.name} ({existing_container.status})</li>"
            except:
                pass
        
        status_html = f"""
        <div style="background-color: #{'e8f5e9' if api_available else 'fff3e0'}; padding: 15px; border-radius: 8px;">
            <h4>🖥️ 虚拟机状态</h4>
            <ul style="list-style: none; padding-left: 0;">
                <li>{'🟢' if api_available else '🟡'} <strong>状态:</strong> {'运行中' if api_available else '启动中'}</li>
                <li>🔧 <strong>Provider:</strong> Docker (OSWorld)</li>
                <li>💻 <strong>操作系统:</strong> Ubuntu</li>
                <li>🌐 <strong>API 端口:</strong> {port}</li>
                <li>📡 <strong>API 状态:</strong> {'✅ 可用' if api_available else ('⏳ ' + (api_error or '等待中'))}</li>
                {container_info}
            </ul>
            {f'<p style="color: #666; font-size: 0.9em; margin-top: 10px;">💡 如果 API 不可用，请等待 1-2 分钟让服务启动，或查看容器日志: <code>docker logs osworld-vm-test</code></p>' if not api_available else ''}
        </div>
        """
        
        return status_html
        
    except Exception as e:
        return f"<p style='color: red;'>❌ 获取状态失败: {str(e)}</p>"


def diagnose_vm_connection(port):
    """诊断 VM API 连接问题"""
    try:
        import docker
        import requests
        
        diagnosis = []
        
        # 检查容器状态
        existing_container = find_existing_container()
        if existing_container:
            existing_container.reload()
            container_status = existing_container.status
            container_name = existing_container.name
            
            diagnosis.append(f"📦 容器状态: {container_status}")
            diagnosis.append(f"📦 容器名称: {container_name}")
            
            if container_status != 'running':
                diagnosis.append(f"⚠️  容器未运行，状态: {container_status}")
                if container_status == 'exited':
                    # 尝试获取退出代码
                    try:
                        exit_code = existing_container.attrs['State']['ExitCode']
                        diagnosis.append(f"⚠️  退出代码: {exit_code}")
                    except:
                        pass
        else:
            diagnosis.append("❌ 未找到 OSWorld 容器")
        
        # 如果提供了端口，检查端口和 API
        if port:
            # 检查端口是否被占用
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result == 0:
                    diagnosis.append(f"✅ 端口 {port} 正在监听")
                else:
                    diagnosis.append(f"❌ 端口 {port} 未监听")
            except Exception as e:
                diagnosis.append(f"⚠️  端口检查失败: {str(e)}")
            
            # 尝试连接 API
            try:
                response = requests.get(f'http://localhost:{port}/', timeout=3)
                if response.status_code == 200:
                    diagnosis.append("✅ API 响应正常")
                else:
                    diagnosis.append(f"⚠️  API 响应异常: HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                diagnosis.append("⏳ API 连接超时（可能正在启动）")
            except requests.exceptions.ConnectionError:
                diagnosis.append(f"❌ 无法连接到 API (端口 {port})")
        
        # 检查容器日志（最后几行）
        if existing_container:
            try:
                logs = existing_container.logs(tail=5).decode('utf-8', errors='ignore')
                if logs.strip():
                    diagnosis.append(f"\n📋 容器日志（最后5行）:\n{logs[-200:]}")
            except:
                pass
        
        return "\n".join(diagnosis)
    except Exception as e:
        return f"诊断失败: {str(e)}"


def capture_vm_screenshot():
    """手动截取虚拟机屏幕（通过 HTTP API）"""
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        
        # 获取容器端口
        port = get_osworld_container_port()
        
        if not port:
            # 提供诊断信息
            diagnosis = diagnose_vm_connection(None)
            return f"❌ 未找到运行中的 OSWorld 容器\n\n{diagnosis}\n\n💡 请点击「启动虚拟机」按钮启动容器", None
        
        # 通过 HTTP API 获取截图
        try:
            response = requests.get(f'http://localhost:{port}/screenshot', timeout=15)
            if response.status_code == 200:
                screenshot_bytes = response.content
                
                # 验证是有效的图片
                if len(screenshot_bytes) < 100:
                    return "❌ 截图数据无效（太小）", None
                
                # 转换为 PIL Image
                img = Image.open(BytesIO(screenshot_bytes))
                
                # 保存截图
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_dir = Path("data/gui_screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                
                screenshot_path = screenshot_dir / f"manual_{timestamp}.png"
                img.save(screenshot_path)
                
                # 返回绝对路径，确保 Gradio 可以正确显示
                abs_path = str(screenshot_path.absolute())
                
                return f"✅ 截图成功！\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n尺寸: {img.size[0]}x{img.size[1]}\n大小: {len(screenshot_bytes) / 1024:.2f} KB", abs_path
            else:
                return f"❌ 截图获取失败: HTTP {response.status_code}", None
        except requests.exceptions.Timeout:
            diagnosis = diagnose_vm_connection(port)
            return f"❌ 截图超时：VM 可能还在启动中\n\n{diagnosis}\n\n💡 请等待 1-2 分钟后重试，或检查容器日志", None
        except requests.exceptions.ConnectionError:
            diagnosis = diagnose_vm_connection(port)
            return f"❌ 无法连接到 VM API (端口 {port})\n\n{diagnosis}\n\n💡 建议：\n1. 检查容器是否正在运行\n2. 等待 1-2 分钟让服务启动\n3. 查看容器日志: docker logs osworld-vm-test", None
        
    except Exception as e:
        import traceback
        return f"❌ 截图失败: {str(e)}", None


def send_local_action(action_type, action_params):
    """发送动作到本地系统（直接控制）"""
    try:
        import pyautogui
        import json
        from PIL import ImageGrab
        from datetime import datetime
        from pathlib import Path
        
        # 解析动作参数
        try:
            params = json.loads(action_params) if action_params else {}
        except:
            return "❌ 动作参数格式错误（需要 JSON 格式）", None
        
        # 执行动作
        try:
            if action_type == "click":
                x = params.get('x', 100)
                y = params.get('y', 100)
                pyautogui.click(x, y)
                action_str = f"pyautogui.click({x}, {y})"
            elif action_type == "type":
                text = params.get('text', '')
                # 使用 interval 参数减慢输入速度，避免第一个字符丢失
                # 特别是在 hotkey 之后，输入框可能还没完全准备好
                import time
                # 先等待一小段时间，确保输入框已获得焦点
                time.sleep(0.3)
                # 使用较慢的输入速度（每个字符间隔 0.1 秒）
                pyautogui.typewrite(text, interval=0.1)
                action_str = f"pyautogui.typewrite('{text}', interval=0.1)"
            elif action_type == "press":
                key = params.get('key', 'enter')
                pyautogui.press(key)
                action_str = f"pyautogui.press('{key}')"
            elif action_type == "moveTo":
                x = params.get('x', 500)
                y = params.get('y', 500)
                pyautogui.moveTo(x, y)
                action_str = f"pyautogui.moveTo({x}, {y})"
            elif action_type == "custom":
                action_str = params.get('command', '')
                # 安全执行：只允许 pyautogui 命令
                if not action_str.strip().startswith('pyautogui.'):
                    return "❌ 自定义命令必须以 'pyautogui.' 开头", None
                exec(action_str, {'pyautogui': pyautogui})
            else:
                return f"❌ 不支持的动作类型: {action_type}", None
            
            # 等待一下让动作生效
            import time
            time.sleep(0.5)
            
            # 获取执行后的截图
            screenshot_path = None
            try:
                import platform
                system = platform.system()
                
                if system == "Darwin":  # macOS
                    # macOS 上优先使用 PyAutoGUI 截图
                    import pyautogui
                    screenshot = pyautogui.screenshot()
                else:
                    # Linux/Windows 使用 ImageGrab
                    from PIL import ImageGrab
                    screenshot = ImageGrab.grab()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_dir = Path("data/gui_screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"local_action_{timestamp}.png"
                screenshot.save(screenshot_path)
                screenshot_path = str(screenshot_path.absolute())
            except Exception as e:
                screenshot_path = None
            
            result_msg = f"""✅ 本地动作执行成功！

🎯 动作类型: {action_type}
📝 命令: {action_str}
🖥️  执行位置: 本地系统
"""
            
            return result_msg, screenshot_path
            
        except Exception as e:
            return f"❌ 动作执行失败: {str(e)}", None
            
    except ImportError:
        return "❌ PyAutoGUI 未安装\n💡 请安装: pip install pyautogui", None
    except Exception as e:
        import traceback
        return f"❌ 本地动作执行失败: {str(e)}", None


def capture_local_screenshot():
    """截取本地系统屏幕"""
    try:
        import platform
        import time
        from datetime import datetime
        from pathlib import Path
        
        # 根据操作系统选择截图方法
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # macOS 上优先使用 PyAutoGUI 截图（会捕获所有窗口，包括活动窗口）
            try:
                import pyautogui
                # 等待一小段时间，确保窗口状态稳定
                time.sleep(0.2)
                # 使用 PyAutoGUI 截图（会捕获整个屏幕，包括所有窗口）
                screenshot = pyautogui.screenshot()
            except Exception as e:
                # 备用方案：使用系统命令 screencapture
                import subprocess
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_dir = Path("data/gui_screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"local_manual_{timestamp}.png"
                
                # 使用 screencapture 命令（macOS 原生，-x 禁用声音，-C 捕获光标）
                result = subprocess.run(
                    ['screencapture', '-x', '-C', str(screenshot_path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and screenshot_path.exists():
                    from PIL import Image
                    img = Image.open(screenshot_path)
                    abs_path = str(screenshot_path.absolute())
                    return f"✅ 本地截图成功！\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n尺寸: {img.size[0]}x{img.size[1]}\n保存位置: {abs_path}", abs_path
                else:
                    return f"❌ 截图失败: {result.stderr or '未知错误'}", None
        else:
            # Linux/Windows 使用 ImageGrab
            from PIL import ImageGrab
            time.sleep(0.2)  # 等待窗口状态稳定
            screenshot = ImageGrab.grab()
        
        # 保存截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = Path("data/gui_screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        screenshot_path = screenshot_dir / f"local_manual_{timestamp}.png"
        screenshot.save(screenshot_path)
        
        abs_path = str(screenshot_path.absolute())
        
        return f"✅ 本地截图成功！\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n尺寸: {screenshot.size[0]}x{screenshot.size[1]}\n保存位置: {abs_path}", abs_path
        
    except ImportError:
        return "❌ PyAutoGUI/PIL 未安装\n💡 请安装: pip install pyautogui pillow", None
    except Exception as e:
        import traceback
        return f"❌ 本地截图失败: {str(e)}\n详情: {traceback.format_exc()[:200]}", None


def send_vm_action(action_type, action_params):
    """发送动作到虚拟机（通过 HTTP API）"""
    try:
        import requests
        import json
        from PIL import Image
        from io import BytesIO
        
        # 获取容器端口
        port = get_osworld_container_port()
        
        if not port:
            return "❌ 未找到运行中的 OSWorld 容器\n💡 请先启动容器: python test_osworld_vm_screenshot.py", None
        
        # 解析动作参数
        try:
            params = json.loads(action_params) if action_params else {}
        except:
            return "❌ 动作参数格式错误（需要 JSON 格式）", None
        
        # 构造动作命令
        if action_type == "click":
            x = params.get('x', 100)
            y = params.get('y', 100)
            action_str = f"pyautogui.click({x}, {y})"
        elif action_type == "type":
            text = params.get('text', '')
            action_str = f"pyautogui.typewrite('{text}')"
        elif action_type == "press":
            key = params.get('key', 'enter')
            action_str = f"pyautogui.press('{key}')"
        elif action_type == "moveTo":
            x = params.get('x', 500)
            y = params.get('y', 500)
            action_str = f"pyautogui.moveTo({x}, {y})"
        elif action_type == "custom":
            action_str = params.get('command', '')
        else:
            return f"❌ 不支持的动作类型: {action_type}", None
        
        # 验证坐标范围（对于需要坐标的动作）
        if action_type in ["click", "moveTo"]:
            x = params.get('x', 0)
            y = params.get('y', 0)
            if x < 0 or y < 0:
                return f"❌ 坐标无效: ({x}, {y})\n💡 坐标必须为非负数", None
        
        # 通过 HTTP API 执行动作（带重试机制）
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f'http://localhost:{port}/execute',
                    json={'action': action_str},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 检查响应中的错误
                    if result.get('status') == 'error':
                        error_msg = result.get('message', '未知错误')
                        
                        # 针对常见错误提供解决方案
                        if 'list index out of range' in error_msg.lower():
                            diagnosis = diagnose_vm_connection(port)
                            return f"""❌ 动作执行失败: {error_msg}

🔍 可能原因：
1. 虚拟机屏幕环境尚未完全初始化
2. 屏幕截图功能暂时不可用
3. 坐标超出屏幕范围

{diagnosis}

💡 建议：
1. 等待 10-20 秒后重试
2. 先尝试「截取屏幕」确认屏幕可用
3. 检查容器日志: docker logs osworld-vm-test
4. 如果持续失败，尝试重启容器""", None
                        else:
                            return f"❌ 动作执行失败: {error_msg}\n\n💡 请检查动作参数是否正确", None
                    
                # 获取执行后的截图
                screenshot_path = None
                try:
                    screenshot_response = requests.get(f'http://localhost:{port}/screenshot', timeout=10)
                    if screenshot_response.status_code == 200:
                        screenshot_bytes = screenshot_response.content
                        img = Image.open(BytesIO(screenshot_bytes))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_dir = Path("data/gui_screenshots")
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path = screenshot_dir / f"action_{timestamp}.png"
                        img.save(screenshot_path)
                        # 转换为绝对路径
                        screenshot_path = str(screenshot_path.absolute())
                except:
                    screenshot_path = None  # 截图获取失败不影响动作执行结果
                
                    result_msg = f"""✅ 动作执行成功！

🎯 动作类型: {action_type}
📝 命令: {action_str}
📊 结果: {result.get('status', 'unknown')}
"""
                    
                    return result_msg, screenshot_path
                else:
                    # 尝试解析错误响应
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', response.text[:200])
                    except:
                        error_msg = response.text[:200]
                    
                    last_error = f"HTTP {response.status_code}: {error_msg}"
                    
                    # 如果是 500 错误且是第一次尝试，等待后重试
                    if response.status_code == 500 and attempt < max_retries - 1:
                        import time
                        time.sleep(2)  # 等待 2 秒后重试
                        continue
                    
                    # 解析详细错误信息
                    if 'list index out of range' in error_msg.lower():
                        diagnosis = diagnose_vm_connection(port)
                        return f"""❌ 动作执行失败: {error_msg}

🔍 可能原因：
1. 虚拟机屏幕环境尚未完全初始化
2. 屏幕截图功能暂时不可用
3. 坐标超出屏幕范围

{diagnosis}

💡 建议：
1. 等待 10-20 秒后重试
2. 先尝试「截取屏幕」确认屏幕可用
3. 检查容器日志: docker logs osworld-vm-test
4. 如果持续失败，尝试重启容器""", None
                    else:
                        return f"❌ 动作执行失败: {last_error}\n\n💡 请检查动作参数或等待 VM 完全启动", None
                        
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                diagnosis = diagnose_vm_connection(port)
                return f"❌ 动作执行超时：VM 可能还在启动中\n\n{diagnosis}\n\n💡 请等待 1-2 分钟后重试", None
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                diagnosis = diagnose_vm_connection(port)
                return f"❌ 无法连接到 VM API (端口 {port})\n\n{diagnosis}\n\n💡 建议：\n1. 检查容器是否正在运行\n2. 等待 1-2 分钟让服务启动\n3. 查看容器日志: docker logs osworld-vm-test", None
        
        # 如果所有重试都失败
        if last_error:
            return f"❌ 动作执行失败（已重试 {max_retries} 次）: {last_error}\n\n💡 请检查容器状态或等待 VM 完全启动", None
        
        # 如果没有任何错误但也没有成功（不应该到达这里）
        return "❌ 动作执行失败：未知错误", None
        
    except Exception as e:
        import traceback
        return f"❌ 动作执行失败: {str(e)}", None


def execute_model_parsed_actions(parsed_actions_text, control_target):
    """执行模型返回的解析动作（OSWorld 风格）"""
    try:
        import json
        import re
        import time
        
        if not parsed_actions_text or "无动作" in parsed_actions_text:
            return "❌ 没有可执行的动作\n💡 请先发送截图和指令给模型，获取动作建议", None
        
        # 从文本中提取 JSON 格式的动作列表
        actions = []
        json_match = re.search(r'\[动作列表JSON\]:\s*(\[.*?\])', parsed_actions_text)
        if json_match:
            try:
                actions = json.loads(json_match.group(1))
            except:
                pass
        
        # 如果没有找到 JSON，尝试从文本中解析动作
        if not actions:
            # 从文本中提取 pyautogui 命令
            lines = parsed_actions_text.split('\n')
            for line in lines:
                line = line.strip()
                # 跳过编号和空行
                if not line or line.startswith('[') or '动作列表JSON' in line:
                    continue
                # 提取动作（去掉编号）
                action = re.sub(r'^\d+\.\s*', '', line)
                if action.startswith('pyautogui.') or action in ['DONE', 'FAIL', 'WAIT']:
                    actions.append(action)
        
        if not actions:
            return "❌ 无法从文本中解析出动作\n💡 请确保模型返回了有效的 PyAutoGUI 动作", None
        
        # 使用 OSWorld 风格的执行方式
        # 仅本地模式支持直接执行 pyautogui 命令，VM 模式仍需通过 API
        is_local = "本地" in control_target or "Local" in control_target
        
        # 初始化 pyautogui（本地模式）
        controller = None
        if is_local:
            try:
                import pyautogui
                controller = pyautogui
                # 设置安全延迟，避免执行过快
                pyautogui.PAUSE = 0.5
            except ImportError:
                return "❌ PyAutoGUI 未安装\n💡 请安装: pip install pyautogui", None
        
        # 执行每个动作
        results = []
        success_count = 0
        
        for i, action in enumerate(actions):
            # 处理控制符
            if action in ['DONE', 'FAIL', 'WAIT']:
                results.append(f"步骤 {i+1}: {action}")
                if action == 'DONE':
                    success_count += 1
                continue
            
            # 执行 PyAutoGUI 动作
            try:
                if is_local:
                    # 本地模式：直接使用 exec 执行（OSWorld 风格）
                    if not action.strip().startswith('pyautogui.'):
                        results.append(f"步骤 {i+1}: ❌ 不安全的命令 - {action}")
                        continue
                    
                    # 在安全的命名空间中执行
                    namespace = {'pyautogui': controller}
                    exec(action, namespace)
                    
                    results.append(f"步骤 {i+1} ({action}): ✅ 执行成功")
                    success_count += 1
                    
                    # 等待界面响应
                    time.sleep(1.0)
                    
                else:
                    # VM 模式：通过 API 执行
                    result_msg, screenshot = send_vm_action("custom", json.dumps({"command": action}))
                    
                    # 提取第一行作为简短状态
                    first_line = result_msg.split(chr(10))[0] if result_msg else '执行中...'
                    results.append(f"步骤 {i+1} ({action}): {first_line}")
                    
                    if result_msg and "✅" in result_msg:
                        success_count += 1
                    elif result_msg and "❌" in result_msg:
                        # 如果执行失败，停止后续动作
                        results.append(f"\n💡 VM 模式提示：\n- VM 环境可能尚未完全初始化\n- 等待 10-20 秒后重试\n- 先尝试「截取屏幕」确认 VM 可用")
                        break
                    
                    # 等待界面响应
                    time.sleep(1.0)
                    
            except Exception as e:
                error_msg = f"❌ 执行失败: {str(e)}"
                results.append(f"步骤 {i+1} ({action}): {error_msg}")
                
                # 本地模式的错误提示
                if is_local:
                    import traceback
                    error_detail = traceback.format_exc()
                    results.append(f"\n💡 错误详情:\n{error_detail[:200]}")
                
                # 执行失败后停止后续动作
                break
        
        result_text = "\n".join(results)
        return f"✅ 成功执行 {success_count}/{len(actions)} 个动作\n\n{result_text}", None
        
    except Exception as e:
        import traceback
        return f"❌ 执行模型动作失败: {str(e)}\n\n{traceback.format_exc()[:300]}", None


def send_to_model_interaction(
    screenshot_source,
    instruction,
    model_name,
    api_key,
    base_url,
    enable_thinking,
    manual_screenshot_path,
    screenshot_target,
    require_a11y_tree=False,
    a11y_focused_only=True
):
    """将截图和任务指令发送给模型（OSWorld 风格）"""
    try:
        from ..gui_agent_service import SimplePromptAgent
        from PIL import Image
        from io import BytesIO
        import base64
        
        # 1. 获取截图
        screenshot_bytes = None
        screenshot_path = None
        
        if "使用上方截图" in screenshot_source:
            # 使用已有的截图
            # manual_screenshot_path 可能是：
            # 1. 字符串路径（绝对路径或相对路径）
            # 2. PIL Image 对象
            # 3. None 或空值
            # 4. 字典格式（Gradio Image 组件可能返回 {"image": path, ...}）
            
            has_screenshot = False
            actual_path = None
            
            if manual_screenshot_path is not None:
                # 处理字典格式（Gradio Image 组件可能返回的格式）
                if isinstance(manual_screenshot_path, dict):
                    # 尝试从字典中提取路径
                    actual_path = manual_screenshot_path.get('image') or manual_screenshot_path.get('path') or manual_screenshot_path.get('name')
                    if actual_path and isinstance(actual_path, str) and actual_path.strip():
                        has_screenshot = True
                # 处理字符串路径
                elif isinstance(manual_screenshot_path, str) and manual_screenshot_path.strip():
                    actual_path = manual_screenshot_path
                    has_screenshot = True
                # 处理 PIL Image 对象
                elif hasattr(manual_screenshot_path, 'save'):  # PIL Image 对象
                    has_screenshot = True
                # 处理 numpy 数组（Gradio Image 组件可能返回）
                elif hasattr(manual_screenshot_path, 'shape'):  # numpy array
                    from PIL import Image
                    try:
                        manual_screenshot_path = Image.fromarray(manual_screenshot_path)
                        has_screenshot = True
                    except:
                        pass
            
            if has_screenshot:
                try:
                    # 如果是字符串路径，读取文件
                    if actual_path or (isinstance(manual_screenshot_path, str) and manual_screenshot_path.strip()):
                        file_path = actual_path if actual_path else manual_screenshot_path
                        # 确保路径存在
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                screenshot_bytes = f.read()
                            screenshot_path = file_path
                        else:
                            return f"❌ 截图文件不存在: {file_path}\n\n💡 请重新截取屏幕", "", None
                    # 如果是 PIL Image 对象或 numpy 数组
                    elif hasattr(manual_screenshot_path, 'save') or hasattr(manual_screenshot_path, 'shape'):
                        # 确保是 PIL Image
                        if hasattr(manual_screenshot_path, 'shape'):
                            from PIL import Image
                            manual_screenshot_path = Image.fromarray(manual_screenshot_path)
                        
                        # 转换为 bytes
                        buffer = BytesIO()
                        manual_screenshot_path.save(buffer, format='PNG')
                        screenshot_bytes = buffer.getvalue()
                        
                        # 保存用于预览
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_dir = Path("data/gui_screenshots")
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path = screenshot_dir / f"model_interaction_{timestamp}.png"
                        manual_screenshot_path.save(screenshot_path)
                        screenshot_path = str(screenshot_path.absolute())
                    else:
                        return "❌ 无法识别截图格式\n\n💡 请重新截取屏幕", "", None
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()[:300]
                    return f"❌ 读取已有截图失败: {str(e)}\n\n详情:\n{error_detail}", "", None
            else:
                return "❌ 没有可用的截图\n\n💡 请执行以下操作：\n1. 在上方「手动截图」中点击「截取屏幕」按钮\n2. 等待截图成功后再选择「使用上方截图」", "", None
        else:
            # 自动截图
            if "本地" in screenshot_target or "Local" in screenshot_target:
                # 本地截图
                try:
                    import platform
                    if platform.system() == "Darwin":
                        import pyautogui
                        screenshot = pyautogui.screenshot()
                    else:
                        from PIL import ImageGrab
                        screenshot = ImageGrab.grab()
                    
                    # 转换为 bytes
                    buffer = BytesIO()
                    screenshot.save(buffer, format='PNG')
                    screenshot_bytes = buffer.getvalue()
                    
                    # 保存截图用于预览
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_dir = Path("data/gui_screenshots")
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = screenshot_dir / f"model_interaction_{timestamp}.png"
                    screenshot.save(screenshot_path)
                    screenshot_path = str(screenshot_path.absolute())
                except Exception as e:
                    return f"❌ 本地截图失败: {str(e)}", "", None
            else:
                # VM 截图
                port = get_osworld_container_port()
                if not port:
                    return "❌ 未找到运行中的 OSWorld 容器\n💡 请先启动容器", "", None
                
                try:
                    import requests
                    response = requests.get(f'http://localhost:{port}/screenshot', timeout=15)
                    if response.status_code == 200:
                        screenshot_bytes = response.content
                        
                        # 保存截图用于预览
                        img = Image.open(BytesIO(screenshot_bytes))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_dir = Path("data/gui_screenshots")
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path = screenshot_dir / f"model_interaction_{timestamp}.png"
                        img.save(screenshot_path)
                        screenshot_path = str(screenshot_path.absolute())
                    else:
                        return f"❌ 截图获取失败: HTTP {response.status_code}", "", None
                except Exception as e:
                    return f"❌ VM 截图失败: {str(e)}", "", None
        
        if not screenshot_bytes:
            return "❌ 无法获取截图", "", None
        
        # 2. 确定 API Key 和 Base URL
        # 如果是 Qwen 模型，使用阿里云百炼
        if model_name.startswith("qwen") or model_name.startswith("qvq"):
            final_api_key = api_key.strip() if api_key and api_key.strip() else (os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'))
            final_base_url = base_url.strip() if base_url and base_url.strip() else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            # GPT 模型使用 OpenAI
            final_api_key = api_key.strip() if api_key and api_key.strip() else os.getenv('OPENAI_API_KEY')
            final_base_url = base_url.strip() if base_url and base_url.strip() else "https://api.openai.com/v1"
        
        # 验证 API Key
        if not final_api_key:
            if model_name.startswith("qwen") or model_name.startswith("qvq"):
                return "❌ 未配置 API Key\n\n💡 请执行以下操作之一：\n1. 在「模型配置」中输入 DASHSCOPE_API_KEY\n2. 或设置环境变量：export DASHSCOPE_API_KEY='your_key'\n\n📖 获取 API Key：https://help.aliyun.com/zh/model-studio/visual-reasoning", "", None
            else:
                return "❌ 未配置 API Key\n\n💡 请执行以下操作之一：\n1. 在「模型配置」中输入 OPENAI_API_KEY\n2. 或设置环境变量：export OPENAI_API_KEY='your_key'", "", None
        
        # 3. 验证任务指令
        if not instruction or not instruction.strip():
            return "❌ 请输入任务指令\n\n💡 请在「任务指令」文本框中输入要执行的任务描述", "", screenshot_path if screenshot_path else None
        
        # 4. 初始化代理（传递 enable_thinking 参数）
        try:
            agent = SimplePromptAgent(
                model=model_name,
                api_key=final_api_key,
                base_url=final_base_url,
                enable_thinking=enable_thinking if (model_name.startswith("qwen") or model_name.startswith("qvq")) else False
            )
            
            # 验证客户端是否初始化成功
            if not agent.client:
                return "❌ 模型客户端初始化失败\n\n💡 请检查：\n1. API Key 是否正确\n2. Base URL 是否正确\n3. 是否安装了 openai 库：pip install openai", "", screenshot_path if screenshot_path else None
        except Exception as e:
            return f"❌ 代理初始化失败: {str(e)}\n\n💡 请检查模型配置是否正确", "", screenshot_path if screenshot_path else None
        
        # 5. 获取 Accessibility Tree（如果启用）
        accessibility_tree = None
        if require_a11y_tree:
            try:
                from ..accessibility_tree import get_accessibility_tree, is_accessibility_available
                if is_accessibility_available():
                    mode_str = "仅焦点窗口" if a11y_focused_only else "所有前台窗口"
                    print(f"🌲 正在获取 Accessibility Tree ({mode_str}, 使用 OSWorld 标准深度 MAX_DEPTH=50)...")
                    accessibility_tree = get_accessibility_tree(include_dock=False, focused_window_only=a11y_focused_only)
                    if accessibility_tree:
                        print(f"✅ Accessibility Tree 已获取 ({len(accessibility_tree)} 字符)")
                    else:
                        print("⚠️  Accessibility Tree 为空")
                else:
                    print("ℹ️  Accessibility Tree 不可用（当前平台不支持）")
            except Exception as e:
                print(f"⚠️  获取 Accessibility Tree 失败: {e}")
        
        # 6. 构造观察对象
        observation = {
            'screenshot': screenshot_bytes,
            'screenshot_path': screenshot_path,
            'accessibility_tree': accessibility_tree,
            'timestamp': datetime.now().isoformat()
        }
        
        # 7. 调用模型预测
        try:
            response_text, actions = agent.predict(instruction, observation)
            
            # 8. 格式化响应
            actions_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)]) if actions else "无动作"
            
            # 9. 返回结果（包含 actions 列表供后续执行使用）
            # 将 actions 作为 JSON 字符串附加到 actions_text，以便 execute_model_parsed_actions 可以解析
            import json
            if actions:
                actions_json = json.dumps(actions, ensure_ascii=False)
                actions_text_with_json = f"{actions_text}\n\n[动作列表JSON]: {actions_json}"
            else:
                actions_text_with_json = actions_text
            
            return response_text, actions_text_with_json, screenshot_path
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()[:500]
            return f"❌ 模型调用失败: {str(e)}\n\n详情:\n{error_detail}", "", screenshot_path
        
    except Exception as e:
        import traceback
        return f"❌ 模型交互失败: {str(e)}\n\n{traceback.format_exc()[:300]}", "", None


def run_gui_agent_task(
    instruction,
    max_steps,
    sleep_time,
    model_name,
    api_key,
    base_url,
    enable_thinking,
    use_history,
    control_target,
    enable_grid=True,
    show_notifications=True,
    require_a11y_tree=False,
    a11y_focused_only=True
):
    """执行 GUI-Agent 任务并返回结果和截图（OSWorld 完整循环）"""
    try:
        import time
        import json
        from ..gui_agent_service import SimplePromptAgent
        from PIL import Image
        from io import BytesIO
        
        if not instruction or not instruction.strip():
            return "❌ 请输入任务指令", [], []
        
        # 1. 检查是否有任务正在执行，如果有则中断
        if is_task_running():
            print("⚠️  检测到正在执行的任务，正在中断...")
            set_task_stop_flag(True)  # 设置停止标志
            
            # 等待旧任务停止（最多等待 5 秒）
            wait_count = 0
            while is_task_running() and wait_count < 50:  # 50 * 0.1 = 5秒
                time.sleep(0.1)
                wait_count += 1
            
            if is_task_running():
                print("⚠️  旧任务未能及时停止，强制开始新任务")
            else:
                print("✅ 旧任务已成功中断")
        
        # 2. 重置环境和标志
        set_task_stop_flag(False)  # 清除停止标志
        set_task_running(True)     # 设置为运行状态
        
        # 3. 启动键盘监听（监听 ESC 键）
        keyboard_listener_started = start_keyboard_listener()
        if keyboard_listener_started:
            print("💡 提示: 按 ESC 键可随时中断任务")
        else:
            print("💡 提示: ESC 键监听未启用，可以通过 Gradio 界面或重新执行新任务来中断")
        
        print(f"🔄 环境已重置，开始新任务: {instruction}")
        
        # 确定执行目标
        is_local = "本地" in control_target or "Local" in control_target
        
        # 为本次任务创建独立的截图文件夹
        task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"task_{task_timestamp}"
        task_screenshot_dir = Path("data/gui_screenshots") / task_id
        task_screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 任务截图目录: {task_screenshot_dir}")
        
        # 初始化 Agent
        from ..gui_agent_service import SimplePromptAgent
        
        # 确定 API Key 和 Base URL
        if model_name.startswith("qwen") or model_name.startswith("qvq"):
            final_api_key = api_key.strip() if api_key and api_key.strip() else (os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'))
            final_base_url = base_url.strip() if base_url and base_url.strip() else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            final_api_key = api_key.strip() if api_key and api_key.strip() else os.getenv('OPENAI_API_KEY')
            final_base_url = base_url.strip() if base_url and base_url.strip() else "https://api.openai.com/v1"
        
        # 验证 API Key
        if not final_api_key:
            set_task_running(False)  # 重置状态
            return "❌ 未配置 API Key\n\n💡 请在「模型配置」中输入 API Key 或设置环境变量", [], []
        
        # 初始化代理
        try:
            agent = SimplePromptAgent(
                model=model_name,
                api_key=final_api_key,
                base_url=final_base_url,
                enable_thinking=enable_thinking if (model_name.startswith("qwen") or model_name.startswith("qvq")) else False,
                use_trajectory=use_history  # 根据用户选择决定是否使用历史轨迹
            )
            
            if not agent.client:
                set_task_running(False)  # 重置状态
                return "❌ 模型客户端初始化失败\n\n💡 请检查 API Key 和 Base URL 是否正确", [], []
            
            # 重置 Agent 的历史轨迹
            agent.reset()
            if use_history:
                print("🔄 Agent 历史轨迹已重置（启用历史记录）")
            else:
                print("🔄 Agent 历史轨迹已重置（禁用历史记录）")
            
        except Exception as e:
            set_task_running(False)  # 重置状态
            return f"❌ 代理初始化失败: {str(e)}", [], []
        
        # 初始化 pyautogui（本地模式）
        controller = None
        if is_local:
            try:
                import pyautogui
                controller = pyautogui
                pyautogui.PAUSE = 0.5
            except ImportError:
                set_task_running(False)  # 重置状态
                return "❌ PyAutoGUI 未安装\n💡 请安装: pip install pyautogui", [], []
        
        # 主循环：截图 -> 模型 -> 执行动作（使用上面创建的 task_screenshot_dir）
        step_count = 0
        steps_data = []
        screenshot_paths = []
        done = False
        final_status = "running"
        
        summary = f"🔄 开始执行任务...\n\n📋 任务指令: {instruction}\n📊 最大步数: {max_steps}\n🎯 执行目标: {control_target}\n🤖 模型: {model_name}\n\n"
        
        # 本地模式：在任务开始前自动打开新标签页，切换到空白界面
        if is_local and controller:
            try:
                print("🔄 正在切换到空白界面...")
                summary += "🔄 切换到空白界面...\n"
                
                # macOS：使用 Command+T 打开新标签页（假设在浏览器中）
                # 如果不在浏览器，这个命令在大多数应用中是无害的
                if platform.system() == "Darwin":
                    controller.hotkey('command', 't')
                else:
                    # Linux/Windows：使用 Ctrl+T
                    controller.hotkey('ctrl', 't')
                
                # 等待页面切换完成
                time.sleep(2.0)
                
                print("✅ 已切换到新标签页")
                summary += "✅ 已切换到新标签页\n\n"
            except Exception as e:
                print(f"⚠️  切换标签页失败: {e}")
                summary += f"⚠️  切换标签页失败: {e}\n\n"
        
        while not done and step_count < max_steps:
            # 检查是否应该停止任务
            if should_stop_task():
                summary += "\n⚠️  任务已被用户中断\n"
                final_status = "interrupted"
                break
            
            step_count += 1
            print(f"\n{'='*50}")
            print(f"🔄 步骤 {step_count}/{max_steps}")
            print(f"{'='*50}")
            summary += f"\n{'='*50}\n步骤 {step_count}\n{'='*50}\n"
            
            # 显示步骤开始（非阻塞，不影响后续执行）
            if is_local and show_notifications:
                _show_autopilot_notification(f"📍 步骤 {step_count}/{max_steps}\n正在截取屏幕...")
                # ⚠️ 关键：必须等待弹窗消失后再截图，否则截图会包含弹窗
                # macOS 弹窗设置为 2 秒自动关闭，所以等待 2.5 秒确保弹窗完全消失
                # 在等待过程中检查停止标志
                for _ in range(25):  # 25 * 0.1 = 2.5秒
                    if should_stop_task():
                        summary += "\n⚠️  任务已被用户中断\n"
                        final_status = "interrupted"
                        break
                    time.sleep(0.1)
            elif is_local:
                # 即使不显示通知，也要给一个短暂延迟，确保界面稳定
                time.sleep(0.3)
                
                if should_stop_task():
                    break
            else:
                # VM 模式不需要等待弹窗
                # 但仍然检查停止标志
                for _ in range(5):  # 5 * 0.1 = 0.5秒
                    if should_stop_task():
                        summary += "\n⚠️  任务已被用户中断\n"
                        final_status = "interrupted"
                        break
                    time.sleep(0.1)
                
                if should_stop_task():
                    break
            
            # 1. 截图
            try:
                if is_local:
                    # 本地截图
                    print("📸 正在截取本地屏幕...")
                    if platform.system() == "Darwin":
                        import pyautogui
                        screenshot = pyautogui.screenshot()
                    else:
                        from PIL import ImageGrab
                        screenshot = ImageGrab.grab()
                    
                    # 转换为 bytes
                    buffer = BytesIO()
                    screenshot.save(buffer, format='PNG')
                    screenshot_bytes = buffer.getvalue()
                    
                    # 对截图进行标注（添加坐标基准点和网格）
                    try:
                        from ..gui_agent_service import annotate_screenshot_with_coordinates
                        import pyautogui
                        logical_size = pyautogui.size()
                        annotated_bytes = annotate_screenshot_with_coordinates(
                            screenshot_bytes,
                            logical_size.width,
                            logical_size.height,
                            enable_grid=enable_grid
                        )
                        # 使用标注后的截图
                        screenshot_bytes = annotated_bytes
                        screenshot = Image.open(BytesIO(annotated_bytes))
                        print(f"🎯 截图已标注坐标基准点和网格")
                    except Exception as e:
                        print(f"⚠️  截图标注失败，使用原始截图: {e}")
                    
                    # 保存截图到任务专属目录
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = task_screenshot_dir / f"step_{step_count}_{timestamp}.png"
                    screenshot.save(screenshot_path)
                    screenshot_path = str(screenshot_path.absolute())
                    screenshot_paths.append(screenshot_path)
        
                    print(f"✅ 截图成功: {os.path.basename(screenshot_path)}")
                    summary += f"📸 截图成功: {os.path.basename(screenshot_path)}\n"
                else:
                    # VM 截图
                    port = get_osworld_container_port()
                    if not port:
                        summary += "❌ 未找到运行中的 OSWorld 容器\n"
                        final_status = "failed"
                        break
                    
                    import requests
                    response = requests.get(f'http://localhost:{port}/screenshot', timeout=15)
                    if response.status_code == 200:
                        screenshot_bytes = response.content
                        img = Image.open(BytesIO(screenshot_bytes))
                        
                        # 对截图进行标注（添加坐标基准点和网格）
                        try:
                            from ..gui_agent_service import annotate_screenshot_with_coordinates
                            # VM 使用固定的分辨率 (通常是 1920x1080)
                            screen_width, screen_height = img.size
                            annotated_bytes = annotate_screenshot_with_coordinates(
                                screenshot_bytes,
                                screen_width,
                                screen_height,
                                enable_grid=enable_grid
                            )
                            # 使用标注后的截图
                            screenshot_bytes = annotated_bytes
                            img = Image.open(BytesIO(annotated_bytes))
                            print(f"🎯 VM截图已标注坐标基准点和网格")
                        except Exception as e:
                            print(f"⚠️  VM截图标注失败，使用原始截图: {e}")
                        
                        # 保存截图到任务专属目录
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = task_screenshot_dir / f"step_{step_count}_{timestamp}.png"
                        img.save(screenshot_path)
                        screenshot_path = str(screenshot_path.absolute())
                        screenshot_paths.append(screenshot_path)
                        
                        summary += f"📸 截图成功: {os.path.basename(screenshot_path)}\n"
                    else:
                        summary += f"❌ 截图失败: HTTP {response.status_code}\n"
                        final_status = "failed"
                        break
            except Exception as e:
                summary += f"❌ 截图失败: {str(e)}\n"
                final_status = "failed"
                break
            
            # 2. 调用模型
            try:
                # 获取 Accessibility Tree（如果启用）
                accessibility_tree = None
                if require_a11y_tree:
                    try:
                        from ..accessibility_tree import get_accessibility_tree, is_accessibility_available
                        if is_accessibility_available():
                            # 任务执行时根据配置决定是否只获取焦点窗口
                            mode_text = "仅焦点窗口" if a11y_focused_only else "所有前台窗口"
                            print(f"🌲 正在获取 Accessibility Tree ({mode_text}, 使用 OSWorld 标准深度 MAX_DEPTH=50)...")
                            accessibility_tree = get_accessibility_tree(include_dock=False, focused_window_only=a11y_focused_only)
                            if accessibility_tree:
                                print(f"✅ Accessibility Tree 已获取 ({len(accessibility_tree)} 字符)")
                            else:
                                print("⚠️  Accessibility Tree 为空")
                        else:
                            print("ℹ️  Accessibility Tree 不可用（当前平台不支持）")
                    except Exception as e:
                        print(f"⚠️  获取 Accessibility Tree 失败: {e}")
                
                observation = {
                    'screenshot': screenshot_bytes,
                    'screenshot_path': screenshot_path,
                    'accessibility_tree': accessibility_tree,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 显示正在调用模型（非阻塞）
                if is_local and show_notifications:
                    _show_autopilot_notification(f"🧠 步骤 {step_count}/{max_steps}\n正在等待模型思考...")
                
                print("🧠 正在调用模型...")
                summary += "🧠 调用模型中...\n"
                
                # 在调用模型前检查停止标志
                if should_stop_task():
                    summary += "\n⚠️  任务已被用户中断（调用模型前）\n"
                    final_status = "interrupted"
                    break
                
                # 调用模型（这里会花费较长时间，是同步的，但这是必要的）
                response_text, actions = agent.predict(instruction, observation)
                
                # 调用模型后再次检查停止标志
                if should_stop_task():
                    summary += "\n⚠️  任务已被用户中断（调用模型后）\n"
                    final_status = "interrupted"
                    break
                
                print(f"✅ 模型返回 {len(actions)} 个动作: {actions}")
                summary += f"🤖 模型返回 {len(actions)} 个动作: {actions}\n"
                
                # 显示模型返回的动作
                if is_local and actions:
                    # 提取思考过程（如果有）
                    thinking_preview = ""
                    if enable_thinking and response_text:
                        # 提取思考过程的前50个字符
                        lines = response_text.split('\n')
                        for line in lines:
                            if line.strip() and not line.strip().startswith('pyautogui'):
                                thinking_preview = line.strip()[:50]
                                break
                    
                    action_preview = ', '.join([a[:30] for a in actions[:2]])
                    if len(actions) > 2:
                        action_preview += f"... (共{len(actions)}个)"
                    
                    if show_notifications:
                        notification_msg = f"🤖 步骤 {step_count}/{max_steps}\n"
                        if thinking_preview:
                            notification_msg += f"💭 {thinking_preview}...\n"
                        notification_msg += f"📋 动作: {action_preview}"
                        
                        _show_autopilot_notification(notification_msg)
                    
                    # ⚠️ 关键：等待弹窗消失后再执行动作，避免动作执行时截图包含弹窗
                    # 弹窗显示 2 秒自动关闭，等待 2.5 秒确保完全消失
                    # 在等待过程中检查停止标志
                    for _ in range(25):  # 25 * 0.1 = 2.5秒
                        if should_stop_task():
                            summary += "\n⚠️  任务已被用户中断\n"
                            final_status = "interrupted"
                            break
                        time.sleep(0.1)
                    
                    if should_stop_task():
                        break
            except Exception as e:
                summary += f"❌ 模型调用失败: {str(e)}\n"
                final_status = "failed"
                break
            
            # 3. 执行动作
            for i, action in enumerate(actions):
                # 在每个动作前检查停止标志
                if should_stop_task():
                    summary += "\n⚠️  任务已被用户中断（执行动作前）\n"
                    final_status = "interrupted"
                    break
                
                # 处理控制符
                if action in ['DONE', 'FAIL', 'WAIT']:
                    summary += f"  {i+1}. {action}\n"
                    steps_data.append([step_count, action, '🎯' if action == 'DONE' else '❌' if action == 'FAIL' else '⏸️', os.path.basename(screenshot_path)])
                    
                    if action == 'DONE':
                        done = True
                        final_status = "completed"
                        summary += "\n✅ 任务完成！\n"
                    elif action == 'FAIL':
                        done = True
                        final_status = "failed"
                        summary += "\n❌ 任务失败！\n"
                    elif action == 'WAIT':
                        # WAIT 时也检查停止标志
                        wait_seconds = float(sleep_time)
                        wait_intervals = int(wait_seconds / 0.1)
                        for _ in range(wait_intervals):
                            if should_stop_task():
                                summary += "\n⚠️  任务已被用户中断（WAIT 期间）\n"
                                final_status = "interrupted"
                                break
                            time.sleep(0.1)
                    break
                
                # 执行 PyAutoGUI 动作
                try:
                    if is_local:
                        # 本地模式：直接执行
                        if not action.strip().startswith('pyautogui.'):
                            print(f"  ❌ 动作 {i+1}: 不安全的命令 - {action}")
                            summary += f"  {i+1}. ❌ 不安全的命令: {action}\n"
                            steps_data.append([step_count, action, '❌', os.path.basename(screenshot_path)])
                            continue
                        
                        print(f"  ▶️  动作 {i+1}: {action}")
                        
                        # 执行动作（动作执行不显示弹窗，避免截图包含弹窗）
                        namespace = {'pyautogui': controller}
                        exec(action, namespace)
                        
                        print(f"  ✅ 动作 {i+1} 执行成功")
                        summary += f"  {i+1}. ✅ {action}\n"
                        steps_data.append([step_count, action[:80], '✅', os.path.basename(screenshot_path)])
                        
                        # 动作执行后等待，让界面更新
                        time.sleep(0.8)
                    else:
                        # VM 模式：通过 API 执行
                        result_msg, _ = send_vm_action("custom", json.dumps({"command": action}))
                        
                        if result_msg and "✅" in result_msg:
                            summary += f"  {i+1}. ✅ {action}\n"
                            steps_data.append([step_count, action[:80], '✅', os.path.basename(screenshot_path)])
                        else:
                            summary += f"  {i+1}. ❌ {action}: {result_msg.split(chr(10))[0] if result_msg else '执行失败'}\n"
                            steps_data.append([step_count, action[:80], '❌', os.path.basename(screenshot_path)])
                            # VM 执行失败不中断，继续执行
                    
                    # 等待界面完全响应后再进行下一个动作
                    time.sleep(float(sleep_time))
                    
                except Exception as e:
                    summary += f"  {i+1}. ❌ {action}: {str(e)}\n"
                    steps_data.append([step_count, action[:80], '❌', os.path.basename(screenshot_path)])
                    # 执行失败不中断，继续执行
        
        # 任务结束
        if step_count >= max_steps and not done:
            final_status = "max_steps_reached"
            summary += f"\n⏱️ 达到最大步数 {max_steps}，任务停止\n"
        
        # 最终统计
        status_emoji = {
            'completed': '✅',
            'failed': '❌',
            'max_steps_reached': '⏱️',
            'running': '🔄',
            'interrupted': '⚠️'
        }
        
        final_summary = f"""{status_emoji.get(final_status, '❓')} 任务状态: {final_status}

📋 任务指令: {instruction}
🆔 任务 ID: {task_id}
📁 截图目录: {task_screenshot_dir}
📊 执行步数: {step_count} / {max_steps}
📸 生成截图: {len(screenshot_paths)} 张
🎯 执行目标: {control_target}
🤖 模型: {model_name}

{summary}
"""
        
        # 本地模式下，任务结束时给出 Autopilot 完成通知
        if is_local:
            if final_status == "completed":
                print(f"\n✅ 任务完成！共执行 {step_count} 步")
                if show_notifications:
                    _show_autopilot_notification(f"🎉 任务完成！\n\n📋 指令: {instruction[:40]}...\n📊 共执行 {step_count} 步")
            elif final_status == "max_steps_reached":
                print(f"\n⏱️ 达到最大步数 {max_steps}，任务停止")
                if show_notifications:
                    _show_autopilot_notification(f"⏱️ 达到最大步数\n\n已执行 {step_count}/{max_steps} 步\n任务未完成，已停止")
            elif final_status == "interrupted":
                print("\n⚠️  任务已被中断")
                if show_notifications:
                    _show_autopilot_notification(f"⚠️ 任务已中断\n\n已执行 {step_count} 步\n用户手动中断")
            elif final_status == "failed":
                print("\n❌ 任务执行失败")
                if show_notifications:
                    _show_autopilot_notification(f"❌ 任务失败\n\n在步骤 {step_count} 处失败\n请查看日志了解详情")
        
        # 重置任务状态
        set_task_running(False)
        set_task_stop_flag(False)
        print("\n🔄 任务执行完成，状态已重置")
        
        return final_summary, steps_data, screenshot_paths
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        
        # 确保重置任务状态
        set_task_running(False)
        set_task_stop_flag(False)
        
        return f"❌ 任务执行失败: {str(e)}\n\n详情:\n{error_detail[:500]}", [], []
    
    finally:
        # 无论如何都要重置任务状态
        set_task_running(False)
        set_task_stop_flag(False)
        
        # 停止键盘监听
        stop_keyboard_listener()


# ==================== 图片检索辅助函数 ====================


def upload_and_add_image(image_service, image_file, description="", tags=""):
    """上传并添加图片到索引"""
    try:
        if image_file is None:
            return "❌ 请选择要上传的图片", None, []
        
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # 添加图片到索引
        image_id = image_service.add_image(
            image_path=image_file.name,
            description=description,
            tags=tag_list
        )
        
        # 获取图片信息用于预览
        image_info = image_service.get_image_info(image_id)
        
        # 刷新图片列表
        all_images = get_all_images_list(image_service)
        
        return f"✅ 图片上传成功！\nID: {image_id}\n描述: {description}\n标签: {', '.join(tag_list)}", image_file, all_images
        
    except Exception as e:
        return f"❌ 上传图片失败: {str(e)}", None, []

def search_images_by_image(image_service, query_image, top_k=10):
    """图搜图功能"""
    try:
        if query_image is None:
            return [], "❌ 请选择要搜索的图片", []
        
        # 执行图搜图
        results = image_service.search_by_image(query_image.name, top_k=top_k)
        
        if not results:
            return [], "🔍 没有找到相似的图片", []
        
        # 格式化结果
        formatted_results = []
        gallery_images = []
        
        for result in results:
            similarity_score = f"{result['similarity']:.4f}"
            formatted_results.append([
                result['original_name'],
                result['description'] or "无描述",
                ', '.join(result['tags']) or "无标签",
                f"{result['width']}x{result['height']}",
                similarity_score,
                result['id']
            ])
            
            # 添加到图片画廊
            if os.path.exists(result['stored_path']):
                gallery_images.append(result['stored_path'])
        
        status_msg = f"🎯 找到 {len(results)} 张相似图片，相似度分数范围: {results[-1]['similarity']:.4f} - {results[0]['similarity']:.4f}"
        
        return formatted_results, status_msg, gallery_images
        
    except Exception as e:
        return [], f"❌ 图搜图失败: {str(e)}", []

def search_images_by_text(image_service, query_text, top_k=10):
    """文搜图功能"""
    try:
        if not (query_text or '').strip():
            return [], "❌ 请输入搜索文本", []
        
        # 执行文搜图
        results = image_service.search_by_text(query_text, top_k=top_k)
        
        if not results:
            return [], "🔍 没有找到匹配的图片", []
        
        # 格式化结果
        formatted_results = []
        gallery_images = []
        
        for result in results:
            similarity_score = f"{result['similarity']:.4f}"
            formatted_results.append([
                result['original_name'],
                result['description'] or "无描述",
                ', '.join(result['tags']) or "无标签",
                f"{result['width']}x{result['height']}",
                similarity_score,
                result['id']
            ])
            
            # 添加到图片画廊
            if os.path.exists(result['stored_path']):
                gallery_images.append(result['stored_path'])
        
        status_msg = f"🎯 找到 {len(results)} 张匹配图片，相似度分数范围: {results[-1]['similarity']:.4f} - {results[0]['similarity']:.4f}"
        
        return formatted_results, status_msg, gallery_images
        
    except Exception as e:
        return [], f"❌ 文搜图失败: {str(e)}", []

def get_all_images_list(image_service):
    """获取所有图片列表"""
    try:
        all_images = image_service.get_all_images()
        
        if not all_images:
            return []
        
        # 按创建时间排序
        all_images.sort(key=lambda x: x['created_at'], reverse=True)
        
        formatted_list = []
        for image_info in all_images:
            file_size_mb = round(image_info['file_size'] / (1024 * 1024), 2)
            formatted_list.append([
                image_info['original_name'],
                image_info['description'] or "无描述",
                ', '.join(image_info['tags']) or "无标签",
                f"{image_info['width']}x{image_info['height']}",
                f"{file_size_mb} MB",
                image_info['created_at'][:16].replace('T', ' '),
                image_info['id']
            ])
        
        return formatted_list
        
    except Exception as e:
        print(f"❌ 获取图片列表失败: {e}")
        return []

def get_image_stats(image_service):
    """获取图片统计信息"""
    try:
        stats = image_service.get_stats()
        
        formats_str = ", ".join([f"{fmt}({count})" for fmt, count in stats['formats'].items()]) if stats['formats'] else "无"
        
        html_content = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4>📊 图片库统计信息</h4>
            <ul>
                <li><strong>图片总数:</strong> {stats['total_images']} 张</li>
                <li><strong>存储大小:</strong> {stats['total_size_mb']} MB</li>
                <li><strong>图片格式:</strong> {formats_str}</li>
                <li><strong>嵌入维度:</strong> {stats['embedding_dimension']}</li>
                <li><strong>计算设备:</strong> {stats['model_device']}</li>
                <li><strong>存储目录:</strong> {stats['storage_dir']}</li>
            </ul>
            <p style="color: #6c757d; font-size: 0.9em;">统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        return html_content
        
    except Exception as e:
        return f"<p style='color: red;'>获取统计信息失败: {str(e)}</p>"

def delete_selected_image(image_service, selected_data):
    """删除选中的图片"""
    try:
        if not selected_data:
            return "❌ 请在图片列表中选择要删除的图片", []
        
        # 获取选中的图片ID（最后一列）
        image_id = selected_data[-1]
        
        # 删除图片
        success = image_service.delete_image(image_id)
        
        if success:
            # 刷新图片列表
            updated_list = get_all_images_list(image_service)
            return f"✅ 图片删除成功: {image_id}", updated_list
        else:
            return f"❌ 图片删除失败: {image_id}", []
            
    except Exception as e:
        return f"❌ 删除图片失败: {str(e)}", []

def clear_all_images(image_service):
    """清空所有图片"""
    try:
        image_service.clear_index()
        return "✅ 所有图片已清空", []
    except Exception as e:
        return f"❌ 清空失败: {str(e)}", []

def _check_accessibility_available():
    """检查 Accessibility Tree 是否可用"""
    try:
        from ..accessibility_tree import is_accessibility_available
        return is_accessibility_available()
    except ImportError:
        return False
    except Exception:
        return False

def build_image_tab(image_service):
    """构建多模态系统页面（包含图片检索和GUI-Agent）"""
    
    # 检查 Accessibility Tree 是否可用
    a11y_available = _check_accessibility_available()
    if a11y_available:
        print("✅ Accessibility Tree 可用 - UI 中将显示 Access tree 选项")
    else:
        print("⚠️  Accessibility Tree 不可用 - UI 中将隐藏 Access tree 选项")
        print("💡 提示：如果应该可用但未显示，请检查：")
        print("   1. 是否在 testbed conda 环境中运行系统")
        print("   2. 是否已安装依赖：pip install pyobjc-framework-Quartz pyobjc-framework-ApplicationServices lxml")
    
    with gr.Blocks() as image_tab:
        gr.Markdown("""
        ### 🖼️ 多模态系统 - 跨模态理解与交互
        
        **图片检索**：基于 CLIP 模型的图搜图、文搜图功能  
        **GUI-Agent**：基于 OSWorld 架构的桌面自动化代理
        """)
        
        with gr.Tabs():
            # 图片上传标签页
            with gr.Tab("📤 图片上传"):
                gr.Markdown("#### 上传图片到图片库")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.File(
                            label="选择图片文件",
                            file_types=["image"],
                            file_count="single"
                        )
                        
                        image_description = gr.Textbox(
                            label="图片描述",
                            placeholder="请输入图片的描述信息...",
                            lines=3
                        )
                        
                        image_tags = gr.Textbox(
                            label="图片标签",
                            placeholder="输入标签，用逗号分隔，如：动物,猫,宠物",
                            lines=1
                        )
                        
                        upload_btn = gr.Button("📤 上传图片", variant="primary")
                        upload_status = gr.Textbox(
                            label="上传状态",
                            lines=4,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 图片预览")
                        image_preview = gr.Image(
                            label="图片预览",
                            height=300
                        )
            
            # 图搜图标签页
            with gr.Tab("🔍 图搜图"):
                gr.Markdown("#### 使用图片搜索相似图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.File(
                            label="选择查询图片",
                            file_types=["image"],
                            file_count="single"
                        )
                        
                        image_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="返回结果数量"
                        )
                        
                        image_search_btn = gr.Button("🔍 图搜图", variant="primary")
                        
                        image_search_status = gr.Textbox(
                            label="搜索状态",
                            lines=2,
                            interactive=False
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### 搜索结果")
                        image_search_results = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "相似度", "ID"],
                            label="相似图片列表",
                            interactive=False
                        )
                        
                # 结果图片画廊
                image_gallery = gr.Gallery(
                    label="相似图片画廊",
                    show_label=True,
                    elem_id="image_gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
            
            # 文搜图标签页
            with gr.Tab("💬 文搜图"):
                gr.Markdown("#### 使用文本描述搜索图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_query = gr.Textbox(
                            label="搜索文本",
                            placeholder="输入描述性文本，如：一只橙色的猫在睡觉",
                            lines=3
                        )
                        
                        text_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="返回结果数量"
                        )
                        
                        text_search_btn = gr.Button("💬 文搜图", variant="primary")
                        
                        text_search_status = gr.Textbox(
                            label="搜索状态",
                            lines=2,
                            interactive=False
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### 搜索结果")
                        text_search_results = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "相似度", "ID"],
                            label="匹配图片列表",
                            interactive=False
                        )
                
                # 结果图片画廊
                text_gallery = gr.Gallery(
                    label="匹配图片画廊",
                    show_label=True,
                    elem_id="text_gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
            
            # 图片管理标签页
            with gr.Tab("📋 图片管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 图片库统计")
                        stats_btn = gr.Button("📊 刷新统计", variant="secondary")
                        stats_display = gr.HTML(value="<p>点击按钮查看统计信息...</p>")
                        
                        gr.Markdown("#### 图片库列表")
                        refresh_list_btn = gr.Button("🔄 刷新列表", variant="secondary")
                        
                        images_list = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "大小", "创建时间", "ID"],
                            label="所有图片",
                            interactive=False
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### 图片操作")
                        
                        delete_btn = gr.Button("🗑️ 删除选中图片", variant="stop")
                        clear_all_btn = gr.Button("🗑️ 清空所有图片", variant="stop")
                        
                        operation_status = gr.Textbox(
                            label="操作状态",
                            lines=3,
                            interactive=False
                        )
            
            # 图像生成标签页
            with gr.Tab("🎨 图像生成"):
                gr.Markdown("""
                #### AI 图像生成 - 文本到图像（Text-to-Image）
                
                **功能：** 使用 Stable Diffusion v1.5 从文本描述生成图像
                
                **使用说明：**
                1. 使用 `./quick_start.sh` 启动系统（图像服务自动运行）
                2. 点击下方"加载模型"按钮加载 SD 1.5 模型
                3. 输入提示词，调整参数，生成图像
                
                **提示：** 首次使用需要下载模型（约4GB），请耐心等待
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # 服务配置
                        gr.Markdown("##### 1️⃣ 服务配置")
                        
                        load_model_btn = gr.Button("📥 加载 SD 1.5 模型", variant="primary", size="lg")
                        model_status = gr.Textbox(
                            label="服务状态",
                            value="未检查服务状态",
                            lines=4,
                            interactive=False
                        )
                        
                        gr.Markdown("##### 2️⃣ 生成参数")
                        
                        # 提示词
                        gen_prompt = gr.Textbox(
                            label="正向提示词 (Prompt)",
                            placeholder="例如: a beautiful landscape with mountains and a lake at sunset, highly detailed, 4k",
                            lines=3,
                            value="a cute cat playing with a ball, high quality, detailed"
                        )
                        
                        gen_negative_prompt = gr.Textbox(
                            label="负向提示词 (Negative Prompt)",
                            placeholder="不想看到的内容，例如: blurry, low quality, distorted",
                            lines=2,
                            value="blurry, low quality, watermark"
                        )
                        
                        # 生成参数
                        with gr.Row():
                            gen_steps = gr.Slider(
                                minimum=20,
                                maximum=100,
                                value=50,
                                step=5,
                                label="推理步数 (Steps)",
                                info="步数越多质量越高但越慢（推荐50）"
                            )
                            
                            gen_guidance = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="引导强度 (CFG Scale)",
                                info="值越高越贴近提示词（推荐7.5）"
                            )
                        
                        with gr.Row():
                            gen_width = gr.Slider(
                                minimum=256,
                                maximum=768,
                                value=512,
                                step=64,
                                label="宽度 (Width)",
                                info="SD 1.5 推荐 512"
                            )
                            
                            gen_height = gr.Slider(
                                minimum=256,
                                maximum=768,
                                value=512,
                                step=64,
                                label="高度 (Height)",
                                info="SD 1.5 推荐 512"
                            )
                        
                        with gr.Row():
                            gen_seed = gr.Number(
                                label="随机种子 (Seed)",
                                value=-1,
                                precision=0,
                                info="-1 表示随机"
                            )
                            
                            gen_num_images = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="生成数量"
                            )
                        
                        # 生成按钮
                        generate_btn = gr.Button("🎨 生成图像", variant="primary", size="lg")
                        
                        generation_status = gr.Textbox(
                            label="生成状态",
                            lines=3,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("##### 3️⃣ 生成结果")
                        
                        # 生成的图像
                        generated_images = gr.Gallery(
                            label="生成的图像",
                            show_label=True,
                            elem_id="generated_gallery",
                            columns=2,
                            rows=2,
                            height="auto"
                        )
                        
                        # 图像元数据
                        generation_info = gr.JSON(
                            label="生成信息",
                            visible=True
                        )
                
                # 历史记录
                with gr.Accordion("📜 生成历史", open=False):
                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 刷新历史", variant="secondary")
                        clear_history_btn = gr.Button("🗑️ 清空历史", variant="secondary")
                    
                    history_gallery = gr.Gallery(
                        label="历史图片",
                        columns=4,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )
                    
                    history_info = gr.Markdown(
                        value="点击刷新历史查看生成记录",
                        label="生成信息"
                    )
            
            # GUI-Agent 标签页
            with gr.Tab("🤖 GUI-Agent"):
                gr.Markdown("""
                #### 桌面自动化代理 - 基于 OSWorld 架构（虚拟机隔离）
                
                **核心能力：**
                - 👀 **观察**：自动截取虚拟机屏幕
                - 🧠 **思考**：基于视觉语言模型理解任务并决策
                - 🖱️ **行动**：在虚拟机中执行鼠标、键盘操作
                - 🔄 **循环**：持续执行直到任务完成
                
                **🛡️ 安全设计**：所有操作在隔离的虚拟环境中执行，不会影响主机系统
                
                **参考：** [OSWorld GitHub](https://github.com/xlang-ai/OSWorld)
                """)
                
                # 虚拟机状态监控和手动截图
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖥️ 虚拟机状态")
                        with gr.Row():
                            vm_status_btn = gr.Button("🔄 刷新状态", variant="secondary")
                            vm_start_btn = gr.Button("🚀 启动虚拟机", variant="primary")
                        
                        vm_status_display = gr.HTML(value="<p>点击按钮查看虚拟机状态...</p>")
                        
                        vm_start_status = gr.Textbox(
                            label="启动状态",
                            lines=4,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📸 手动截图")
                        screenshot_target = gr.Radio(
                            choices=["本地系统 (Local)", "虚拟机 (VM)"],
                            value="本地系统 (Local)",
                            label="截图目标",
                            info="选择要截取的目标（本地模式在 macOS 上需要屏幕录制权限）"
                        )
                        screenshot_btn = gr.Button("📷 截取屏幕", variant="primary", size="lg")
                        screenshot_status = gr.Textbox(label="截图状态", lines=3, interactive=False)
                        manual_screenshot = gr.Image(label="当前屏幕", height=300)
                
                # 模型交互部分
                with gr.Accordion("🤖 模型交互（OSWorld 风格）", open=True):
                    gr.Markdown("""
                    #### 将截图和任务指令发送给视觉语言模型
                    
                    **功能说明：**
                    - 📸 自动截取当前屏幕（或使用已有截图）
                    - 📝 输入任务指令
                    - 🤖 模型分析截图并返回思考过程和动作建议
                    - 🔍 查看模型的完整响应内容
                    
                    **参考：** [OSWorld](https://github.com/xlang-ai/OSWorld) 的实现方式
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # 截图选择
                            model_screenshot_source = gr.Radio(
                                choices=["自动截图（当前屏幕）", "使用上方截图"],
                                value="自动截图（当前屏幕）",
                                label="截图来源",
                                info="选择要发送给模型的截图"
                            )
                            
                            # 任务指令
                            model_instruction = gr.Textbox(
                                label="任务指令",
                                placeholder="例如：点击屏幕中心的按钮，或者：打开浏览器并搜索 Python",
                                lines=4,
                                value="请分析当前屏幕截图，描述你看到的内容，并建议下一步可以执行的动作。"
                            )
                            
                            # 模型配置
                            with gr.Accordion("⚙️ 模型配置", open=False):
                                model_interaction_model = gr.Dropdown(
                                    choices=[
                                        "qwen3-vl-plus",
                                        "qwen3-vl-flash",
                                        "qvq-max",
                                        "qvq-plus",
                                        "gpt-4o",
                                        "gpt-4-vision-preview"
                                    ],
                                    value="qwen3-vl-plus",
                                    label="视觉语言模型",
                                    info="推荐使用 Qwen3-VL 系列（阿里云百炼）"
                                )
                                
                                model_interaction_api_key = gr.Textbox(
                                    label="API Key（可选）",
                                    placeholder="留空则使用环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY",
                                    type="password"
                                )
                                
                                model_interaction_base_url = gr.Textbox(
                                    label="API Base URL（可选）",
                                    placeholder="留空则自动选择（Qwen 使用阿里云，GPT 使用 OpenAI）",
                                    value=""
                                )
                            
                            # 高级选项（放在 Accordion 外面，更显眼）
                            with gr.Row():
                                model_enable_thinking = gr.Checkbox(
                                    label="启用思考过程（仅 Qwen3-VL 系列）",
                                    value=False,
                                    info="开启后模型会先输出思考过程，再输出最终回复"
                                )
                                
                                model_enable_a11y_tree = gr.Checkbox(
                                    label="启用 Accessibility Tree",
                                    value=False,
                                    info="获取系统 UI 元素树结构，提供更精确的元素定位（需要系统权限，仅 macOS/Linux）",
                                    visible=a11y_available
                                )
                            
                            # Accessibility Tree 子选项（只在启用时显示）
                            with gr.Row(visible=a11y_available) as a11y_options_row:
                                model_a11y_focused_only = gr.Checkbox(
                                    label="只获取焦点窗口",
                                    value=True,
                                    info="过滤被遮挡的窗口，减少 76% 数据量，提高准确性（推荐开启）"
                                )
                            
                            send_to_model_btn = gr.Button("🚀 发送给模型", variant="primary", size="lg")
                            
                        with gr.Column(scale=1):
                            # 模型响应显示
                            model_response = gr.Textbox(
                                label="模型响应",
                                lines=15,
                                interactive=False,
                                placeholder="模型的分析结果和动作建议将显示在这里..."
                            )
                            
                            # 解析出的动作
                            model_parsed_actions = gr.Textbox(
                                label="解析出的动作",
                                lines=5,
                                interactive=False,
                                placeholder="从模型响应中解析出的 PyAutoGUI 动作将显示在这里..."
                            )
                            
                            # 执行模型返回的动作
                            with gr.Row():
                                execute_model_actions_btn = gr.Button("▶️ 执行模型返回的动作", variant="primary", size="lg")
                            
                            model_action_result = gr.Textbox(
                                label="动作执行结果",
                                lines=5,
                                interactive=False,
                                placeholder="执行模型返回的动作后的结果将显示在这里..."
                            )
                            
                            # 模型使用的截图预览
                            model_screenshot_preview = gr.Image(
                                label="发送给模型的截图",
                                height=200
                            )
                
                # 手动操作控制（移到模型交互下面）
                with gr.Accordion("🎮 手动操作控制", open=True):
                    gr.Markdown("""
                    #### 直接发送动作指令
                    
                    **⚠️ 安全提示：**
                    - **虚拟机模式**：所有操作在隔离的 Docker 容器中执行，安全可靠（推荐）
                    - **本地模式**：直接控制当前系统，请谨慎使用！在 macOS 上需要授予辅助功能权限
                    """)
                    
                    # 控制目标选择
                    control_target = gr.Radio(
                        choices=["本地系统 (Local)", "虚拟机 (VM)"],
                        value="本地系统 (Local)",
                        label="控制目标",
                        info="选择要控制的目标：虚拟机（安全隔离）或本地系统（直接控制）"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            action_type = gr.Dropdown(
                                choices=["click", "type", "press", "moveTo", "custom"],
                                value="click",
                                label="动作类型",
                                info="选择要执行的动作类型"
                            )
                            
                            action_params = gr.Textbox(
                                label="动作参数 (JSON格式)",
                                placeholder='例如：{"x": 500, "y": 300} 或 {"text": "Hello"} 或 {"key": "enter"}',
                                lines=3,
                                value='{"x": 500, "y": 300}'
                            )
                            
                            send_action_btn = gr.Button("🚀 发送动作", variant="primary")
                            
                        with gr.Column(scale=1):
                            action_result = gr.Textbox(
                                label="执行结果",
                                lines=8,
                                interactive=False
                            )
                            action_screenshot = gr.Image(label="执行后截图", height=250)
                    
                    # 常用动作快捷按钮
                    with gr.Row():
                        gr.Markdown("##### 🔥 快捷操作")
                    with gr.Row():
                        quick_click_center = gr.Button("点击屏幕中心", size="sm")
                        quick_press_enter = gr.Button("按 Enter 键", size="sm")
                        quick_open_terminal = gr.Button("打开终端", size="sm")
                        quick_take_screenshot = gr.Button("截图", size="sm")
                
                # 初始化部分
                with gr.Accordion("⚙️ 环境配置（高级）", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("##### 虚拟机配置")
                            
                            gui_provider_name = gr.Dropdown(
                                choices=["docker", "vmware", "aws", "local"],
                                value="docker",
                                label="Provider 类型",
                                info="推荐 docker（简单安全）或 vmware（完整虚拟机）"
                            )
                            
                            gui_os_type = gr.Dropdown(
                                choices=["Ubuntu", "macOS", "Windows"],
                                value="Ubuntu",
                                label="虚拟机操作系统",
                                info="Docker 推荐 Ubuntu，VMware 可选其他系统"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("##### 模型配置")
                            
                            gui_model_name = gr.Dropdown(
                                choices=[
                                    "qwen3-vl-plus",
                                    "qwen3-vl-flash",
                                    "qvq-max",
                                    "qvq-plus",
                                    "gpt-4o",
                                    "gpt-4-vision-preview"
                                ],
                                value="qwen3-vl-plus",
                                label="视觉语言模型",
                                info="推荐使用 Qwen3-VL 系列（阿里云百炼）"
                            )
                            
                            gui_api_key = gr.Textbox(
                                label="API Key",
                                placeholder="留空则使用环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY",
                                type="password"
                            )
                            
                            gui_base_url = gr.Textbox(
                                label="API Base URL",
                                placeholder="留空则使用默认",
                                value=""
                            )
                    
                    gui_init_btn = gr.Button("🚀 初始化环境和代理", variant="primary")
                    gui_init_status = gr.Textbox(label="初始化状态", lines=2, interactive=False)
                
                # 任务执行部分
                gr.Markdown("""
                ### 🚀 任务执行
                
                💡 **提示**：
                - 首次使用会自动启动 Docker 虚拟机（需安装 Docker Desktop）
                - 所有操作在虚拟机中执行，主机系统完全安全
                - 也可以在上方「环境配置」中选择其他虚拟化方案（VMware、AWS等）
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### 任务配置")
                        
                        gui_task_instruction = gr.Textbox(
                            label="任务指令",
                            placeholder="例如：打开浏览器并搜索 OSWorld 项目",
                            lines=3
                        )
                        
                        with gr.Row():
                            gui_max_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=15,
                                step=1,
                                label="最大步数"
                            )
                            
                            gui_sleep_time = gr.Slider(
                                minimum=0.5,
                                maximum=5.0,
                                value=1.5,
                                step=0.5,
                                label="每步等待时间（秒）"
                            )
                        
                        gr.Markdown("##### 模型配置")
                        
                        with gr.Row():
                            gui_task_model = gr.Dropdown(
                                choices=[
                                    "qwen3-vl-plus",
                                    "qwen3-vl-flash",
                                    "qvq-max",
                                    "qvq-plus",
                                    "gpt-4o",
                                    "gpt-4-vision-preview"
                                ],
                                value="qwen3-vl-plus",
                                label="视觉语言模型",
                                info="推荐使用 Qwen3-VL 系列"
                            )
                            
                            gui_task_enable_thinking = gr.Checkbox(
                                value=False,
                                label="启用思考过程",
                                info="仅 Qwen/QVQ 模型支持"
                            )
                        
                        with gr.Row():
                            gui_task_use_history = gr.Checkbox(
                                value=True,
                                label="使用历史轨迹",
                                info="传递前几步的截图和动作给模型，提供更多上下文（复杂任务推荐开启）"
                            )
                            
                            gui_enable_grid = gr.Checkbox(
                                value=True,
                                label="启用网格标注",
                                info="在截图上添加坐标网格和参考点，帮助模型精确定位"
                            )
                        
                        with gr.Row():
                            gui_show_notifications = gr.Checkbox(
                                value=True,
                                label="显示过程提示",
                                info="在任务执行过程中显示系统通知（仅本地模式有效）"
                            )
                            
                            gui_enable_a11y_tree = gr.Checkbox(
                                value=False,
                                label="启用 Accessibility Tree",
                                info="获取系统 UI 元素树结构，提供更精确的元素定位（需要系统权限，仅 macOS/Linux）",
                                visible=a11y_available
                            )
                        
                        with gr.Row():
                            gui_task_api_key = gr.Textbox(
                                label="API Key",
                                placeholder="留空则使用环境变量",
                                type="password",
                                scale=2
                            )
                            
                            gui_task_base_url = gr.Textbox(
                                label="Base URL",
                                placeholder="留空则使用默认",
                                value="",
                                scale=1
                            )
                        
                        gui_run_btn = gr.Button("▶️ 执行任务", variant="primary", size="lg")
                        
                        gr.Markdown("##### 任务状态")
                        gui_task_summary = gr.Textbox(
                            label="执行摘要",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("##### 执行记录")
                        
                        gui_steps_table = gr.Dataframe(
                            headers=["步骤", "动作", "状态", "截图"],
                            label="动作执行历史",
                            interactive=False,
                            wrap=True
                        )
                
                # 截图展示部分
                gr.Markdown("### 📸 执行过程截图")
                
                gui_screenshot_gallery = gr.Gallery(
                    label="所有步骤截图",
                    show_label=True,
                    elem_id="gui_screenshot_gallery",
                    columns=3,
                    rows=3,
                    height="auto",
                    object_fit="contain"
                )
                
                # 示例任务
                with gr.Accordion("📚 示例任务", open=False):
                    gr.Examples(
                        examples=[
                            ["打开终端并输入 'echo Hello OSWorld'", 10, 1.5],
                            ["移动鼠标到屏幕中心 (960, 540) 并点击", 8, 1.0],
                            ["打开文件管理器并创建新文件夹", 15, 2.0],
                            ["在桌面上右键点击并查看菜单", 5, 1.5],
                        ],
                        inputs=[gui_task_instruction, gui_max_steps, gui_sleep_time],
                        label="点击示例自动填充（适合 Ubuntu 虚拟机）"
                        )
        
        # 绑定事件处理函数
        
        # 图片上传
        upload_btn.click(
            fn=lambda img, desc, tags: upload_and_add_image(image_service, img, desc, tags),
            inputs=[upload_image, image_description, image_tags],
            outputs=[upload_status, image_preview, images_list]
        )
        
        # 图搜图
        image_search_btn.click(
            fn=lambda img, k: search_images_by_image(image_service, img, k),
            inputs=[query_image, image_top_k],
            outputs=[image_search_results, image_search_status, image_gallery]
        )
        
        # 文搜图
        text_search_btn.click(
            fn=lambda text, k: search_images_by_text(image_service, text, k),
            inputs=[text_query, text_top_k],
            outputs=[text_search_results, text_search_status, text_gallery]
        )
        
        # 统计信息
        stats_btn.click(
            fn=lambda: get_image_stats(image_service),
            outputs=stats_display
        )
        
        # 刷新图片列表
        refresh_list_btn.click(
            fn=lambda: get_all_images_list(image_service),
            outputs=images_list
        )
        
        # 删除图片
        delete_btn.click(
            fn=lambda data: delete_selected_image(image_service, data),
            inputs=images_list,
            outputs=[operation_status, images_list]
        )
        
        # 清空所有图片
        clear_all_btn.click(
            fn=lambda: clear_all_images(image_service),
            outputs=[operation_status, images_list]
        )
        
        # 图像生成事件绑定
        
        # 创建扩散模型服务实例（延迟初始化）
        from ..diffusion_service import DiffusionService
        diffusion_service = DiffusionService()
        
        def load_diffusion_model():
            """加载扩散模型"""
            success, message = diffusion_service.load_model()
            return message
        
        def generate_images_wrapper(prompt, negative_prompt, steps, guidance, width, height, seed, num_images):
            """生成图像的包装函数"""
            result = diffusion_service.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                seed=int(seed),
                num_images=int(num_images)
            )
            
            if result['success']:
                # 返回图像列表和元数据
                return result['images'], result['message'], result['metadata']
            else:
                return [], result['message'], {}
        
        def get_generation_history_wrapper():
            """获取生成历史"""
            history = diffusion_service.get_generation_history(limit=20)
            
            print(f"[DEBUG] 历史记录数量: {len(history)}")
            
            if not history:
                return [], "暂无生成历史\n\n💡 生成图片后，历史记录会显示在这里"
            
            # 收集所有图片
            images = []
            info_text = "### 📜 生成历史\n\n"
            
            for idx, entry in enumerate(reversed(history), 1):  # 最新的在前
                # 添加图片（通过服务端 URL 获取）
                paths = entry.get('paths', [])
                print(f"[DEBUG] 记录 {idx} 路径: {paths}")
                
                for path in paths:
                    # 从路径中提取文件名，构造服务端 URL
                    filename = os.path.basename(path)
                    image_url = f"{diffusion_service.service_url}/image/{filename}"
                    
                    # 尝试下载图片
                    try:
                        import requests
                        response = requests.get(image_url, timeout=5)
                        if response.status_code == 200:
                            from io import BytesIO
                            from PIL import Image
                            img = Image.open(BytesIO(response.content))
                            # 保存到临时文件供 Gallery 显示
                            import tempfile
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            img.save(temp_file.name)
                            images.append(temp_file.name)
                            print(f"[DEBUG] 成功获取图片: {filename}")
                        else:
                            print(f"[DEBUG] 无法获取图片 {filename}: HTTP {response.status_code}")
                    except Exception as e:
                        print(f"[DEBUG] 获取图片失败 {filename}: {e}")
                
                # 添加信息
                time_str = entry['timestamp'].split('T')[1].split('.')[0]
                info_text += f"**{idx}. {time_str}**\n"
                info_text += f"- **提示词**: {entry['prompt']}\n"
                info_text += f"- **模型**: {entry.get('model', 'SD 1.5')}\n"
                info_text += f"- **参数**: 步数={entry['steps']}, 种子={entry['seed']}, 尺寸={entry['size']}\n"
                info_text += f"- **耗时**: {entry['generation_time']:.2f}秒\n"
                info_text += f"- **图片数**: {entry.get('num_images', 1)}张\n\n"
            
            print(f"[DEBUG] 找到的图片数: {len(images)}")
            
            if not images:
                info_text += "\n⚠️ 未找到图片文件（可能已被删除）"
            
            return images, info_text
        
        def clear_generation_history():
            """清空生成历史"""
            diffusion_service.clear_history()
            return [], "✅ 历史记录已清空"
        
        # 加载模型
        load_model_btn.click(
            fn=load_diffusion_model,
            outputs=[model_status]
        )
        
        # 生成图像
        generate_btn.click(
            fn=generate_images_wrapper,
            inputs=[
                gen_prompt,
                gen_negative_prompt,
                gen_steps,
                gen_guidance,
                gen_width,
                gen_height,
                gen_seed,
                gen_num_images
            ],
            outputs=[generated_images, generation_status, generation_info]
        )
        
        # 刷新历史
        refresh_history_btn.click(
            fn=get_generation_history_wrapper,
            outputs=[history_gallery, history_info]
        )
        
        # 清空历史
        clear_history_btn.click(
            fn=clear_generation_history,
            outputs=[history_gallery, history_info]
        )
        
        # GUI-Agent 事件绑定
        
        # 虚拟机状态监控
        vm_status_btn.click(
            fn=get_vm_status,
            outputs=[vm_status_display]
        )
        
        # 启动虚拟机
        def start_vm_and_refresh_status():
            """启动虚拟机并刷新状态"""
            start_result = start_vm_container()
            # 等待一下让容器启动
            import time
            time.sleep(2)
            status_html = get_vm_status()
            return start_result, status_html
        
        vm_start_btn.click(
            fn=start_vm_and_refresh_status,
            outputs=[vm_start_status, vm_status_display]
        )
        
        # 手动截图（根据目标选择）
        def capture_screenshot_by_target(target):
            """根据目标选择截图函数"""
            if "本地" in target or "Local" in target:
                return capture_local_screenshot()
            else:
                return capture_vm_screenshot()
        
        screenshot_btn.click(
            fn=capture_screenshot_by_target,
            inputs=[screenshot_target],
            outputs=[screenshot_status, manual_screenshot]
        )
        
        # 发送动作（根据目标选择）
        def send_action_by_target(target, action_type, action_params):
            """根据目标选择动作函数"""
            if "本地" in target or "Local" in target:
                return send_local_action(action_type, action_params)
            else:
                return send_vm_action(action_type, action_params)
        
        send_action_btn.click(
            fn=send_action_by_target,
            inputs=[control_target, action_type, action_params],
            outputs=[action_result, action_screenshot]
        )
        
        # 快捷操作（根据目标选择）
        def quick_action_click_center(target):
            """快捷点击屏幕中心"""
            if "本地" in target or "Local" in target:
                return send_local_action("click", '{"x": 960, "y": 540}')
            else:
                return send_vm_action("click", '{"x": 960, "y": 540}')
        
        def quick_action_press_enter(target):
            """快捷按 Enter"""
            if "本地" in target or "Local" in target:
                return send_local_action("press", '{"key": "enter"}')
            else:
                return send_vm_action("press", '{"key": "enter"}')
        
        def quick_action_open_terminal(target):
            """快捷打开终端"""
            if "本地" in target or "Local" in target:
                # macOS 使用 Command+Space 打开 Spotlight，然后输入 Terminal
                import platform
                if platform.system() == "Darwin":
                    return send_local_action("press", '{"key": "command+space"}')
                else:
                    return send_local_action("press", '{"key": "ctrl+alt+t"}')
            else:
                return send_vm_action("press", '{"key": "ctrl+alt+t"}')
        
        def quick_action_screenshot(target):
            """快捷截图"""
            if "本地" in target or "Local" in target:
                return capture_local_screenshot()
            else:
                return capture_vm_screenshot()
        
        quick_click_center.click(
            fn=quick_action_click_center,
            inputs=[control_target],
            outputs=[action_result, action_screenshot]
        )
        
        quick_press_enter.click(
            fn=quick_action_press_enter,
            inputs=[control_target],
            outputs=[action_result, action_screenshot]
        )
        
        quick_open_terminal.click(
            fn=quick_action_open_terminal,
            inputs=[control_target],
            outputs=[action_result, action_screenshot]
        )
        
        quick_take_screenshot.click(
            fn=quick_action_screenshot,
            inputs=[control_target],
            outputs=[screenshot_status, manual_screenshot]
        )
        
        # 模型交互
        send_to_model_btn.click(
            fn=send_to_model_interaction,
            inputs=[
                model_screenshot_source,
                model_instruction,
                model_interaction_model,
                model_interaction_api_key,
                model_interaction_base_url,
                model_enable_thinking,
                manual_screenshot,
                screenshot_target,
                model_enable_a11y_tree,      # 是否启用 Accessibility Tree
                model_a11y_focused_only      # 是否只获取焦点窗口
            ],
            outputs=[model_response, model_parsed_actions, model_screenshot_preview]
        )
        
        # 执行模型返回的动作
        execute_model_actions_btn.click(
            fn=execute_model_parsed_actions,
            inputs=[model_parsed_actions, control_target],
            outputs=[model_action_result, action_screenshot]
        )
        
        # 环境初始化
        gui_init_btn.click(
            fn=initialize_gui_agent,
            inputs=[gui_provider_name, gui_os_type, gui_model_name, gui_api_key, gui_base_url],
            outputs=[gui_init_status]
        )
        
        # 任务执行
        gui_run_btn.click(
            fn=run_gui_agent_task,
            inputs=[
                gui_task_instruction,
                gui_max_steps,
                gui_sleep_time,
                gui_task_model,           # 使用任务执行的模型选择
                gui_task_api_key,         # 使用任务执行的 API Key
                gui_task_base_url,        # 使用任务执行的 Base URL
                gui_task_enable_thinking, # 使用任务执行的思考选项
                gui_task_use_history,     # 使用历史轨迹选项
                control_target,           # 使用手动控制的目标选择
                gui_enable_grid,          # 是否启用网格标注
                gui_show_notifications,   # 是否显示过程提示
                gui_enable_a11y_tree      # 是否启用 Accessibility Tree
            ],
            outputs=[gui_task_summary, gui_steps_table, gui_screenshot_gallery]
        )
        
        # 页面加载时自动刷新统计和列表
        image_tab.load(
            fn=lambda: get_image_stats(image_service),
            outputs=stats_display
        )
        
        image_tab.load(
            fn=lambda: get_all_images_list(image_service),
            outputs=images_list
        )
    
    return image_tab
