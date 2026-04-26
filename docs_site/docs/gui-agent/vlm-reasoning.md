---
layout: default
title: VLM 推理
parent: GUI Automation Agent
nav_order: 3
---

# VLM 推理

使用 Qwen-VL 类模型进行桌面界面理解、动作规划与结果校验的实现说明。

---

## 模型选择

### Qwen-VL-Chat (7B)

**优势**：
- 视觉理解能力稳定
- 支持多轮上下文
- 本地部署链路成熟
- 适合作为可复现基线

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

---

## 提示词设计

### 系统提示词

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

### 任务提示词

```python
prompt = f"""任务：{task_description}

历史动作：
{format_history(action_history)}

当前截图：[image]

请给出下一步动作，并严格按 JSON 格式输出。"""
```

---

## 动作定位

### 坐标预测

```python
def predict_coordinates(vlm, screenshot, element_description):
    """预测目标元素点击坐标"""
    prompt = f"""请在截图中定位：{element_description}
输出 JSON：{{"x": 123, "y": 456}}"""
    response = vlm.chat(tokenizer, query=prompt, image=screenshot)
    coords = json.loads(response)
    return coords["x"], coords["y"]
```

### 可操作元素识别

```python
def identify_elements(vlm, screenshot):
    """识别截图中的可点击元素"""
    prompt = """列出截图中可操作元素及大致位置。"""
    response = vlm.chat(tokenizer, query=prompt, image=screenshot)
    return parse_elements(response)
```

---

## 多步任务规划

### 计划生成

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

### 执行与反馈闭环

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

---

## 错误处理

### 失败检测与重试

```python
if not verify_action_success(action, before, after):
    adjusted_action = adjust_coordinates(action, offset=10)
    execute(adjusted_action)
```

### 恢复策略

1. 重试当前动作（小幅坐标偏移）
2. 切换备选路径（寻找同义控件）
3. 回到已知安全状态后重试
4. 超过阈值后终止并上报

---

## 性能优化

### 推理加速

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    load_in_4bit=True
)
```

### 缓存策略

```python
@lru_cache(maxsize=100)
def cached_vlm_call(screenshot_hash, prompt):
    return vlm.chat(tokenizer, prompt, image=load_image(screenshot_hash))
```

---

## 评估指标

- **任务完成率**：成功任务占比
- **动作效率**：实际步数 / 理想步数
- **恢复成功率**：异常后恢复比例

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
