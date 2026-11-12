# 从截图解释到动作建议

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

模型对截图的解释、动作建议与动作执行结果是三份不同证据。模型说“已点击”不代表控制器真的执行，更不代表业务目标已经达成。

本仓库构造包含截图与任务的消息，解析模型响应中的动作，再由相应环境执行。阅读示例时应区分“仅分析截图”与“连续执行任务”的入口。

## 怎样验证

1. 选择一个截图中能明确定位的目标，核对模型描述是否准确。
2. 检查动作类型、坐标或按键参数。
3. 执行后重新观察画面，核对目标状态而不是只看文字回答。

## 使用边界

缩放、窗口位置、截图时效和响应解析都会影响动作。不能用单次合理的文字解释代替完整任务验收。

## 实现入口

- [src/search_engine/gui_agent_service.py](../../../../src/search_engine/gui_agent_service.py)
- [src/search_engine/image_tab/image_tab.py](../../../../src/search_engine/image_tab/image_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    trust_remote_code=True
)
```

### 示例 2

```text
你是桌面自动化助手。请根据截图和任务描述，判断下一步动作。

可用动作：
- CLICK(x, y)
- TYPE(text)
- SCROLL(direction)
- WAIT(seconds)
- DONE

仅输出 JSON：
{"type":"ACTION","reasoning":"...","params":{...}}
```

### 示例 3

```python
prompt = f"""任务：{task_description}

历史动作：
{format_history(action_history)}

当前截图：[image]

请给出下一步动作，并严格按 JSON 格式输出。"""
```

### 示例 4

```python
def predict_coordinates(vlm, screenshot, element_description):
    """预测目标元素点击坐标"""
    prompt = f"""请在截图中定位：{element_description}
输出 JSON：{{"x": 123, "y": 456}}"""
    response = vlm.chat(tokenizer, query=prompt, image=screenshot)
    coords = json.loads(response)
    return coords["x"], coords["y"]
```

### 示例 5

```python
def identify_elements(vlm, screenshot):
    """识别截图中的可点击元素"""
    prompt = """列出截图中可操作元素及大致位置。"""
    response = vlm.chat(tokenizer, query=prompt, image=screenshot)
    return parse_elements(response)
```

### 示例 6

```python
def generate_plan(task):
    """生成任务分步计划"""
    prompt = f"""任务：{task}

请在 Ubuntu 桌面环境中给出执行计划：
1.
2.
..."""
    plan = vlm.generate(prompt)
    return parse_plan(plan)
```

### 示例 7

```python
def execute_with_feedback(step, screenshot):
    """执行动作并校验结果"""
    action = decide_action(step, screenshot)
    execute(action)

    new_screenshot = capture()
    verification_prompt = f"""
前一截图：[old]
当前截图：[new]
目标动作：{action}

请判断动作是否成功（yes/no）
"""
    success = vlm.chat(tokenizer, verification_prompt)
    return "yes" in success.lower()
```

### 示例 8

```python
if not verify_action_success(action, before, after):
    adjusted_action = adjust_coordinates(action, offset=10)
    execute(adjusted_action)
```

### 示例 9

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    load_in_4bit=True
)
```

### 示例 10

```python
@lru_cache(maxsize=100)
def cached_vlm_call(screenshot_hash, prompt):
    return vlm.chat(tokenizer, prompt, image=load_image(screenshot_hash))
```

### 示例 11

```python
def evaluate_agent(test_tasks):
    results = []
    for task in test_tasks:
        success, steps, errors = agent.execute(task)
        results.append({
            "task": task,
            "success": success,
            "steps": steps,
            "errors": errors
        })
    return {
        "completion_rate": sum(r["success"] for r in results) / len(results),
        "avg_steps": mean([r["steps"] for r in results])
    }
```

</details>
