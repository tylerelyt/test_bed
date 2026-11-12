# 桌面自动化任务的设计

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

把任务写成有初始状态、操作对象和可观察终态的步骤，比使用“帮我处理一下”更容易验证。先确保单个动作可执行，再组合为多步任务。

工作台已有手动截图、动作输入和任务运行入口。可以利用这些入口区分环境故障、模型判断错误和动作解析错误，而不必直接运行复杂工作流。

## 怎样验证

1. 记录初始截图、具体任务和预期终态。
2. 逐步扩大动作序列，检查每次状态变化。
3. 失败时保留原始响应、动作与截图，先定位失败层再重试。

## 使用边界

旧示例中的网页、窗口名称与坐标仅用于说明。实际网页变化和系统布局会改变动作结果，不能将示例序列当作稳定自动化脚本。

## 实现入口

- [src/search_engine/image_tab/image_tab.py](../../../../src/search_engine/image_tab/image_tab.py)
- [src/search_engine/gui_agent_service.py](../../../../src/search_engine/gui_agent_service.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```python
task = "Open the calculator application"

# VLM decides actions
actions = [
    {"type": "CLICK", "x": 50, "y": 900},  # Click Activities
    {"type": "WAIT", "seconds": 1},
    {"type": "TYPE", "text": "calculator"},
    {"type": "WAIT", "seconds": 1},
    {"type": "CLICK", "x": 200, "y": 150},  # Click Calculator icon
    {"type": "DONE"}
]

for action in actions:
    execute_action(action)
```

### 示例 2

```python
task = "Create a file called 'hello.txt' with content 'Hello World'"

# Execution trace
1. CLICK(50, 900)        # Open Files app
2. WAIT(2)
3. CLICK(100, 200)       # New file button
4. TYPE("hello.txt")
5. CLICK(300, 400)       # Confirm
6. TYPE("Hello World")
7. CTRL+S                # Save
8. DONE
```

### 示例 3

```python
task = "Search for 'Python tutorials' on Firefox"

def execute_web_search(query):
    agent = GUIAgent()
    
    # Open Firefox
    agent.execute({"type": "CLICK", "x": firefox_icon_coords})
    agent.wait(3)
    
    # Click address bar
    agent.execute({"type": "CLICK", "x": 400, "y": 100})
    
    # Type search query
    agent.execute({"type": "TYPE", "text": query})
    
    # Press Enter
    agent.execute({"type": "KEY", "key": "enter"})
    
    return agent.capture_screenshot()
```

### 示例 4

```python
tasks = [
    "Open calculator and compute 25 + 37",
    "Take a screenshot and save it as 'desktop.png'",
    "Open Firefox and navigate to github.com"
]

for task in tasks:
    print(f"Executing: {task}")
    agent = GUIAgent()
    success = agent.execute_task(task, max_steps=20)
    print(f"Result: {'Success' if success else 'Failed'}")
```

### 示例 5

```python
def robust_execution(task, max_retries=3):
    """Execute task with automatic retry"""
    
    for attempt in range(max_retries):
        try:
            agent = GUIAgent()
            result = agent.execute_task(task)
            
            if result.success:
                return result
            
            # Reset environment
            reset_desktop_state()
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    return None
```

### 示例 6

```python
# Only capture when needed
if action_type_needs_vision(action):
    screenshot = capture()
else:
    screenshot = last_screenshot
```

### 示例 7

```python
# Group actions that don't need visual feedback
batched_actions = [
    {"type": "TYPE", "text": "filename"},
    {"type": "KEY", "key": "enter"}
]
execute_batch(batched_actions)
```

### 示例 8

```python
from concurrent.futures import ThreadPoolExecutor

def execute_multiple_tasks(tasks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(execute_task, t) for t in tasks]
        results = [f.result() for f in futures]
    return results
```

</details>
