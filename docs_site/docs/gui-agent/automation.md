---
layout: default
title: Automation Examples
parent: GUI Automation Agent
nav_order: 4
---

# Automation Examples

Real-world automation tasks and implementations.

---

## Example 1: Open Calculator

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

---

## Example 2: Create Text File

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

---

## Example 3: Web Search

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

---

## Batch Automation

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

---

## Error Handling

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

---

## Performance Tips

### 1. Reduce Screenshots

```python
# Only capture when needed
if action_type_needs_vision(action):
    screenshot = capture()
else:
    screenshot = last_screenshot
```

### 2. Batch Actions

```python
# Group actions that don't need visual feedback
batched_actions = [
    {"type": "TYPE", "text": "filename"},
    {"type": "KEY", "key": "enter"}
]
execute_batch(batched_actions)
```

### 3. Parallel Execution

```python
from concurrent.futures import ThreadPoolExecutor

def execute_multiple_tasks(tasks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(execute_task, t) for t in tasks]
        results = [f.result() for f in futures]
    return results
```

---

## Use Cases

| Scenario | Complexity | Success Rate |
|:---------|:-----------|:-------------|
| **Open Applications** | Low | 95% |
| **File Operations** | Medium | 85% |
| **Web Navigation** | Medium | 80% |
| **Form Filling** | High | 70% |
| **Multi-App Workflows** | High | 65% |

---

## Best Practices

✅ **Do**:
- Start with simple, deterministic tasks
- Add verification steps
- Implement timeouts
- Log all actions for debugging
- Use VM for isolation

❌ **Don't**:
- Assume UI is always in same state
- Skip error handling
- Use hardcoded coordinates (screen resolution varies)
- Execute untrusted tasks without sandboxing

---

## Resources

- [PyAutoGUI Documentation](https://pyautogui.readthedocs.io/)
- [OSWorld Benchmark](https://os-world.github.io/)
- [GUI Automation Patterns](https://arxiv.org/abs/2404.07972)

