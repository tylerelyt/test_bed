# OSWorld 风格环境与适配

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

本仓库的桌面智能体沿用观察、动作、反馈的环境交互思路，并保留多种控制环境适配。判断集成程度应查看实际导入、控制器与任务结果，不能只凭页面名称。

环境初始化后，任务服务重置环境，逐步获取观察和执行动作；运行记录用于核对每次调用的实际效果。

## 怎样验证

1. 确认控制器初始化成功，并核对虚拟机连接。
2. 检查一次 reset、观察读取与简单 step 的结果。
3. 使用明确的终态条件验证任务，再扩大任务复杂度。

## 使用边界

工作台迁移没有执行 OSWorld 基准测试，也没有验证所有虚拟化提供方。不存在可直接引用的通用成功率。

## 实现入口

- [src/search_engine/gui_agent_service.py](../../../../src/search_engine/gui_agent_service.py)
- [src/search_engine/gui_agent_tab/gui_agent_tab.py](../../../../src/search_engine/gui_agent_tab/gui_agent_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```bash
# Download pre-configured Ubuntu image
wget https://drive.usercontent.google.com/download?id=1y9FocOC0y5R78sqhD24j0P5a7KXSLFJ4 \
    -O Ubuntu.qcow2.zip

# Extract
unzip Ubuntu.qcow2.zip -d data/osworld_vm/
```

### 示例 2

```bash
qemu-system-x86_64 \
    -enable-kvm \
    -m 4096 \
    -smp 2 \
    -drive file=data/osworld_vm/Ubuntu.qcow2,format=qcow2 \
    -vnc :0 \
    -device e1000,netdev=net0 \
    -netdev user,id=net0
```

### 示例 3

```python
from PIL import Image
import subprocess

def capture_vm_screenshot():
    """Capture screenshot via VNC"""
    # Connect to VNC display
    subprocess.run([
        "vncsnapshot",
        "localhost:5900",
        "temp_screenshot.png"
    ])
    
    return Image.open("temp_screenshot.png")
```

### 示例 4

```python
import pyautogui

def execute_action(action):
    """Execute action in VM"""
    if action['type'] == 'CLICK':
        pyautogui.click(action['x'], action['y'])
    
    elif action['type'] == 'TYPE':
        pyautogui.write(action['text'])
    
    elif action['type'] == 'SCROLL':
        pyautogui.scroll(-3 if action['direction'] == 'down' else 3)
```

### 示例 5

```bash
# Create snapshot
qemu-img snapshot -c clean_state Ubuntu.qcow2

# Restore snapshot
qemu-img snapshot -a clean_state Ubuntu.qcow2
```

### 示例 6

```python
import paramiko

def setup_vm_automation():
    """Enable SSH access to VM"""
    ssh = paramiko.SSHClient()
    ssh.connect('localhost', port=2222, username='user', password='password')
    
    # Install dependencies
    ssh.exec_command('sudo apt-get update')
    ssh.exec_command('sudo apt-get install -y python3-tk')
    
    return ssh
```

</details>
