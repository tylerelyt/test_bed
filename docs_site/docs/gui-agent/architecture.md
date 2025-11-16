---
layout: default
title: Architecture
parent: GUI Automation Agent
nav_order: 1
---

# GUI Agent Architecture

OSWorld-inspired architecture for vision-language model powered desktop automation.

---

## System Overview

```mermaid
graph TB
    A[Task Instruction] --> B[GUIAgent]
    B --> C[Observation]
    C --> D[VLM Reasoning<br/>Qwen-VL]
    D --> E[Action Decision]
    E --> F{Action Type}
    
    F -->|Mouse| G[PyAutoGUI]
    F -->|Keyboard| H[PyAutoGUI]
    F -->|Wait| I[Sleep]
    
    G --> J[Execute Action]
    H --> J
    I --> J
    
    J --> K[Screenshot]
    K --> C
    
    E -->|DONE| L[Task Complete]
```

---

## Core Components

### 1. GUIAgent
- Manages observation-action loop
- Coordinates VLM and execution engine
- Tracks task progress

### 2. VLM Reasoning
- **Model**: Qwen-VL-Chat (7B)
- **Input**: Screenshot + task description
- **Output**: Structured action

### 3. Execution Engine
- **PyAutoGUI**: Mouse/keyboard control
- **Platform**: Ubuntu desktop (VM or native)
- **Safety**: Sandboxed execution

---

## Observation-Action Loop

```python
while not task_complete:
    # 1. Observe
    screenshot = capture_screen()
    
    # 2. Reason
    action = vlm.decide_action(screenshot, task, history)
    
    # 3. Act
    execute_action(action)
    
    # 4. Check completion
    if action['type'] == 'DONE':
        break
```

---

## Action Space

| Action | Parameters | Example |
|:-------|:-----------|:--------|
| **CLICK** | x, y | `{"type": "CLICK", "x": 500, "y": 300}` |
| **TYPE** | text | `{"type": "TYPE", "text": "hello"}` |
| **SCROLL** | direction | `{"type": "SCROLL", "direction": "down"}` |
| **WAIT** | seconds | `{"type": "WAIT", "seconds": 2}` |
| **DONE** | - | `{"type": "DONE"}` |

---

## Deployment Options

### Local Mode
- Run on host machine
- Fast, no VM overhead
- ⚠️ Lower isolation

### VM Mode (OSWorld)
- Ubuntu 22.04 in QEMU
- Complete isolation
- Screenshot via VNC

---

## Resources

- [OSWorld Paper](https://arxiv.org/abs/2404.07972)
- [Qwen-VL Model](https://github.com/QwenLM/Qwen-VL)

