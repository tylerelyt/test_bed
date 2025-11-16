---
layout: default
title: OSWorld Integration
parent: GUI Automation Agent
nav_order: 2
---

# OSWorld Integration

OSWorld provides a virtual Ubuntu desktop environment for safe GUI agent experimentation.

---

## Setup

### Download VM Image

```bash
# Download pre-configured Ubuntu image
wget https://drive.usercontent.google.com/download?id=1y9FocOC0y5R78sqhD24j0P5a7KXSLFJ4 \
    -O Ubuntu.qcow2.zip

# Extract
unzip Ubuntu.qcow2.zip -d data/osworld_vm/
```

### Start VM

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

---

## Screenshot Capture

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

---

## Action Execution

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

---

## VM Management

### Snapshot & Restore

```bash
# Create snapshot
qemu-img snapshot -c clean_state Ubuntu.qcow2

# Restore snapshot
qemu-img snapshot -a clean_state Ubuntu.qcow2
```

### Automation

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

---

## Best Practices

- ✅ Use snapshots for reset between tasks
- ✅ Set reasonable timeouts (5 min per task)
- ✅ Monitor VM resource usage
- ❌ Don't run untrusted code without isolation
- ❌ Don't exceed VM memory limits

