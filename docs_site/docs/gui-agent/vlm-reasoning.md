---
layout: default
title: VLM Reasoning
parent: GUI Automation Agent
nav_order: 3
---

# Vision-Language Model Reasoning

Using Qwen-VL for visual understanding and action planning.

---

## Model Selection

### Qwen-VL-Chat (7B)

**Advantages**:
- Strong visual understanding
- Multi-turn conversation
- Efficient inference
- Open-source

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

## Prompt Design

### System Prompt

```
You are a desktop automation assistant. Given a screenshot and task description, decide the next action.

Available actions:
- CLICK(x, y): Click at coordinates
- TYPE(text): Type text
- SCROLL(direction): Scroll up/down
- WAIT(seconds): Wait
- DONE: Task completed

Respond ONLY with JSON:
{"type": "ACTION", "reasoning": "...", ...params}
```

### Task Prompt

```python
prompt = f"""Task: {task_description}

Previous actions:
{format_history(action_history)}

Current screenshot shows: [image]

What should I do next? Respond in JSON format."""
```

---

## Action Grounding

### Coordinate Prediction

```python
def predict_coordinates(vlm, screenshot, element_description):
    """Predict click coordinates for UI element"""
    
    prompt = f"""In this screenshot, where is the {element_description}?

Provide coordinates as JSON: {{"x": 123, "y": 456}}"""
    
    response = vlm.chat(
        tokenizer,
        query=prompt,
        image=screenshot
    )
    
    coords = json.loads(response)
    return coords['x'], coords['y']
```

### UI Element Recognition

```python
def identify_elements(vlm, screenshot):
    """Identify clickable elements in screenshot"""
    
    prompt = """List all clickable UI elements visible in this screenshot.

Format:
1. [Element type] at approximately (x, y)
2. ..."""
    
    response = vlm.chat(tokenizer, query=prompt, image=screenshot)
    return parse_elements(response)
```

---

## Multi-Step Planning

### High-Level Plan Generation

```python
def generate_plan(task):
    """Create step-by-step plan"""
    
    prompt = f"""Task: {task}

Create a step-by-step plan to accomplish this task on Ubuntu desktop.

Plan:
1.
2.
..."""
    
    plan = vlm.generate(prompt)
    return parse_plan(plan)
```

### Step Execution with Feedback

```python
def execute_with_feedback(step, screenshot):
    """Execute step and verify"""
    
    # Execute
    action = decide_action(step, screenshot)
    execute(action)
    
    # Verify
    new_screenshot = capture()
    verification_prompt = f"""
Previous screenshot: [old]
Current screenshot: [new]
Intended action: {action}

Was the action successful? (yes/no)"""
    
    success = vlm.chat(tokenizer, verification_prompt)
    
    return 'yes' in success.lower()
```

---

## Error Handling

### Action Failure Detection

```python
if not verify_action_success(action, before, after):
    # Retry with adjusted coordinates
    adjusted_action = adjust_coordinates(action, offset=10)
    execute(adjusted_action)
```

### Recovery Strategies

1. **Retry**: Same action, slight offset
2. **Alternative Path**: Try different UI element
3. **Reset**: Return to known state
4. **Abort**: Report failure

---

## Optimization

### Inference Speedup

```python
# Use quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    load_in_4bit=True  # 4-bit quantization
)
```

### Caching

```python
@lru_cache(maxsize=100)
def cached_vlm_call(screenshot_hash, prompt):
    """Cache VLM responses for identical inputs"""
    return vlm.chat(tokenizer, prompt, image=load_image(screenshot_hash))
```

---

## Evaluation

### Success Metrics

- **Task Completion Rate**: % of tasks completed
- **Action Efficiency**: Steps taken / optimal steps
- **Error Recovery**: % of errors recovered from

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
        "completion_rate": sum(r['success'] for r in results) / len(results),
        "avg_steps": mean([r['steps'] for r in results])
    }
```

