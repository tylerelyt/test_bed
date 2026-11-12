# ReAct 的动作与观察循环

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

ReAct 将模型生成的推理轨迹与任务动作交替组织，动作用于从外部工具或环境获得观察。方法概念见 [ReAct 原论文](https://arxiv.org/abs/2210.03629)。

本仓库保留旧文档检索 ReAct 方法，并另有当前 OAG 的图谱 ReAct 路径。阅读代码时先区分 `_react_reasoning` 与 `_iter_react_oag_reasoning`，再核对页面实际调用，不能把两套动作格式混用。

## 怎样验证

1. 查看页面绑定的服务方法，再定位该方法允许的动作和停止条件。
2. 记录每步工具观察，检查下一步是否利用了观察内容。
3. 检查达到步数上限、空事实和动作无法解析时的处理。

## 使用边界

概念示例中的 SEARCH/FINISH 字样不能替代当前实现的动作协议。推理轨迹是可观察输出，正确性仍要回到事实与任务结果上验证。

## 实现入口

- [src/search_engine/rag_tab/rag_service.py](../../../../src/search_engine/rag_tab/rag_service.py)
- [src/search_engine/rag_tab/rag_tab.py](../../../../src/search_engine/rag_tab/rag_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph TB
    A[问题] --> B[思考]
    B --> C[行动]
    C --> D[观察]
    D --> E{是否可以回答？}
    E -->|否| B
    E -->|是| F[最终回答]
```

### 示例 2

```python
def react_loop(question, max_steps=5):
    """ReAct reasoning loop"""
    context = []
    
    for step in range(1, max_steps + 1):
        # Generate thought and action
        prompt = build_react_prompt(question, context)
        response = llm.generate(prompt)
        
        # Parse thought and action
        thought, action = parse_response(response)
        
        # Execute action
        if action.startswith("Search"):
            query = extract_query(action)
            observation = search(query)
        elif action.startswith("Lookup"):
            term = extract_term(action)
            observation = lookup(term, context)
        elif action.startswith("Finish"):
            return extract_answer(action)
        
        # Update context
        context.append({
            "step": step,
            "thought": thought,
            "action": action,
            "observation": observation
        })
    
    return "Unable to answer after maximum steps"
```

### 示例 3

```python
REACT_PROMPT = """Answer the following question by reasoning and taking actions step-by-step.

You can use these actions:
- Search[query]: Search for information about query
- Lookup[term]: Find specific term in last search result
- Finish[answer]: Provide final answer

Use this format:
Thought: (your reasoning)
Action: (action to take)
Observation: (result of action)
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Action: Finish[answer]

Question: {question}

Thought:"""
```

### 示例 4

```python
REACT_EXAMPLES = """
Example 1:
Question: Who is the spouse of the director of film Nostalgia (1983 Film)?

Thought: I need to find the director of Nostalgia (1983 film), then find their spouse.
Action: Search[Nostalgia 1983 film director]
Observation: Nostalgia is a 1983 Soviet film directed by Andrei Tarkovsky.

Thought: The director is Andrei Tarkovsky. Now I need to find his spouse.
Action: Search[Andrei Tarkovsky spouse]
Observation: Andrei Tarkovsky was married to Larisa Tarkovskaya.

Thought: I now know the final answer.
Action: Finish[Larisa Tarkovskaya]

---

Example 2:
Question: What is the elevation of the highest peak in the mountain range that Mount Fuji is part of?

Thought: First, I need to find what mountain range Mount Fuji belongs to.
Action: Search[Mount Fuji mountain range]
Observation: Mount Fuji is not actually part of a mountain range; it's a stratovolcano.

Thought: Mount Fuji isn't in a mountain range, so the question is based on a false premise.
Action: Finish[Mount Fuji is not part of a mountain range; it's a standalone stratovolcano]
"""
```

### 示例 5

```python
def execute_search(query):
    """Retrieve relevant documents"""
    results = hybrid_search(query, top_k=3)
    
    # Format observation
    observation = "\n".join([
        f"- {doc['title']}: {doc['snippet']}"
        for doc in results
    ])
    
    return observation
```

### 示例 6

```python
def execute_lookup(term, last_search_results):
    """Find specific term in previous search results"""
    for doc in last_search_results:
        if term.lower() in doc['content'].lower():
            # Extract sentence containing term
            sentences = doc['content'].split('.')
            relevant = [s for s in sentences if term.lower() in s.lower()]
            return relevant[0] if relevant else "Term not found"
    
    return "Term not found in search results"
```

### 示例 7

```python
def execute_calculate(expression):
    """Perform mathematical calculation"""
    try:
        result = eval(expression, {"__builtins__": {}}, safe_math_functions)
        return f"Result: {result}"
    except:
        return "Calculation error"
```

### 示例 8

```python
import re

def parse_react_response(response):
    """Parse LLM response into thought, action, observation"""
    # Extract thought
    thought_match = re.search(r'Thought: (.*?)(?:\n|$)', response)
    thought = thought_match.group(1) if thought_match else ""
    
    # Extract action
    action_match = re.search(r'Action: (.*?)(?:\n|$)', response)
    action = action_match.group(1) if action_match else ""
    
    return thought, action

def extract_action_input(action):
    """Extract query/term from action"""
    # Pattern: Action[input]
    match = re.search(r'\[(.*?)\]', action)
    return match.group(1) if match else ""
```

### 示例 9

```python
def react_with_reflection(question, max_steps=5):
    """ReAct with self-correction"""
    context = []
    
    for step in range(1, max_steps + 1):
        # Generate response
        response = llm.generate(build_prompt(question, context))
        thought, action = parse_response(response)
        
        # Execute action
        observation = execute_action(action)
        
        # Self-reflection
        if step > 1:
            reflection_prompt = f"""Review your previous steps. Are you on the right track?

Previous steps:
{format_context(context)}

Reflection:"""
            reflection = llm.generate(reflection_prompt)
            
            if "wrong" in reflection.lower() or "reconsider" in reflection.lower():
                # Backtrack and try different approach
                context = context[:-1]  # Remove last step
        
        context.append({"thought": thought, "action": action, "observation": observation})
        
        if action.startswith("Finish"):
            return extract_answer(action)
    
    return "Max steps reached"
```

### 示例 10

```python
class MultiAgentReAct:
    def __init__(self):
        self.searcher = SearchAgent()
        self.calculator = CalculatorAgent()
        self.verifier = VerificationAgent()
    
    def solve(self, question):
        """Solve question using multiple specialized agents"""
        plan = self.planner.create_plan(question)
        
        results = []
        for task in plan:
            if task.type == "search":
                result = self.searcher.execute(task)
            elif task.type == "calculate":
                result = self.calculator.execute(task)
            
            # Verify result
            verified = self.verifier.check(result, task)
            results.append(verified)
        
        return self.synthesize_answer(results)
```

### 示例 11

```python
def evaluate_react(test_cases):
    """Evaluate ReAct on test set"""
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        answer = react_loop(case['question'])
        if is_correct(answer, case['ground_truth']):
            correct += 1
    
    return {
        "accuracy": correct / total,
        "avg_steps": mean([len(trace) for trace in traces])
    }
```

### 示例 12

```python
def analyze_trajectory(context):
    """Analyze reasoning trajectory quality"""
    metrics = {
        "num_steps": len(context),
        "unique_searches": len(set(c['action'] for c in context if 'Search' in c['action'])),
        "backtracking": sum(1 for i, c in enumerate(context[1:]) 
                           if similar(c['thought'], context[i]['thought'])),
        "answer_consistency": check_consistency([c['observation'] for c in context])
    }
    return metrics
```

</details>
