# 不启用检索的直接对话

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

直接对话把用户问题交给模型，不添加文档检索结果。它可用于观察模型自身能力，或作为检索增强实验的对照。

当前在“RAG”页关闭“开启检索增强 (RAG)”即可走无检索路径。推理模式仍按页面选择使用 DashScope API 或已加载的本地模型，不存在独立的“Direct Chat”导航页。

## 怎样验证

1. 在同一问题、同一模型下比较开启与关闭检索的回答。
2. 确认关闭检索时没有把旧文档结果误当作本次回答依据。
3. 对企业私有事实，检查模型能否获得对应材料，而不是只比较表达流畅度。

## 使用边界

直接对话不能自动读取本地企业文档。旧示例中的模型速度、星级或通用优劣不是本项目的实测结论。

## 实现入口

- [src/search_engine/rag_tab/rag_tab.py](../../../../src/search_engine/rag_tab/rag_tab.py)
- [src/search_engine/rag_tab/rag_service.py](../../../../src/search_engine/rag_tab/rag_service.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph LR
    A[用户查询] --> B[Ollama API]
    B --> C[大语言模型推理]
    C --> D[生成回答]
    D --> E[展示给用户]
```

### 示例 2

```python
def direct_chat(query: str, model: str = "llama3.1:8b"):
    """Direct LLM chat without retrieval"""

    # Build simple prompt
    prompt = f"User: {query}\n\nAssistant:"

    # Call Ollama API
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]
```

### 示例 3

```
“用简单的话解释机器学习”
“编写一个反转字符串的 Python 函数”
“运动有哪些益处？”
“比较民主制度与专制制度”
```

### 示例 4

```
“第 42 号文档包含什么内容？” → 使用上下文增强模式
“查找项目 X 的信息” → 使用上下文增强模式
“比较文档 A 和文档 B 中的引用” → 使用多步推理
```

### 示例 5

```python
RECOMMENDED_MODELS = {
    # Fast and lightweight
    "llama3.2": {
        "size": "3B parameters",
        "speed": "Very fast",
        "use_case": "Quick Q&A"
    },

    # Balanced performance
    "llama3.1:8b": {
        "size": "8B parameters",
        "speed": "Fast",
        "use_case": "General purpose"
    },

    # Chinese-optimized
    "qwen2.5": {
        "size": "7B parameters",
        "speed": "Fast",
        "use_case": "Chinese Q&A"
    },

    # Code-specialized
    "deepseek-coder": {
        "size": "6.7B parameters",
        "speed": "Fast",
        "use_case": "Programming help"
    }
}
```

### 示例 6

```python
GENERATION_CONFIG = {
    # Creativity control
    "temperature": 0.7,     # 0.0 = deterministic, 1.0 = creative

    # Sampling strategy
    "top_p": 0.9,           # Nucleus sampling
    "top_k": 40,            # Top-K sampling

    # Quality control
    "repeat_penalty": 1.1,  # Penalize repetition
    "max_tokens": 512,      # Response length limit

    # Stopping
    "stop": ["User:", "\n\n\n"]  # Stop sequences
}
```

### 示例 7

```python
# Low temperature (0.1-0.3): Focused, consistent
temperature = 0.2
# Output: "Machine learning is a subset of artificial intelligence..."

# Medium temperature (0.5-0.7): Balanced
temperature = 0.7
# Output: "Machine learning empowers computers to learn patterns..."

# High temperature (0.8-1.0): Creative, varied
temperature = 1.0
# Output: "Imagine teaching a computer to think like us! That's ML..."
```

### 示例 8

```python
# Conservative (0.7-0.8): More predictable
top_p = 0.7

# Balanced (0.9): Recommended
top_p = 0.9

# Diverse (0.95-1.0): More variety
top_p = 0.95
```

### 示例 9

```python
TYPICAL_LATENCY = {
    "api_call": 0.2,       # HTTP request overhead
    "model_loading": 0.0,  # Already loaded in Ollama
    "inference": 4.5,      # Main compute time
    "streaming": 0.3,      # Response streaming
    "total": 5.0           # seconds
}
```

### 示例 10

```python
# Fast but lower quality
model = "llama3.2"  # 3B params, ~3s response

# Balanced
model = "llama3.1:8b"  # 8B params, ~5s response
```

### 示例 11

```python
# Shorter responses = faster
max_tokens = 256  # Brief answers

max_tokens = 512  # Standard

max_tokens = 1024  # Detailed
```

### 示例 12

```bash
# Ollama automatically uses GPU if available
# Check GPU usage:
nvidia-smi  # Linux/Windows
```

### 示例 13

```python
# Clear and specific
"Explain quantum computing in 3 sentences"

# With constraints
"Write a Python function to sort a list, include comments"

# Structured format
"Compare pros and cons of Python vs Java in a table"
```

### 示例 14

```python
# Too vague
"Tell me about stuff"

# Ambiguous
"How does it work?"  # What is "it"?

# Impossible without context
"What did he say in the meeting?"  # LLM wasn't there!
```

### 示例 15

```python
def direct_chat_with_retry(query, model, max_retries=3):
    """Robust direct chat with retry logic"""

    for attempt in range(max_retries):
        try:
            response = call_ollama(query, model)

            # Validate response
            if len(response) < 10:
                raise ValueError("Response too short")

            return response

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return "Error: Request timed out. Please try again."

        except Exception as e:
            return f"Error: {str(e)}"
```

### 示例 16

```
法国首都是巴黎，位于法国中北部，
以艺术、时尚、美食和文化著称。
```

### 示例 17

```
硅基心智在思考
从无尽数据中学习
未来逐渐成形
```

### 示例 18

```
递归就像两面镜子相互映照。
每次映照都呈现同一图像更小的版本，直到小得看不见。
类似地，递归函数用更小规模的输入调用自身，
直到达到终止条件，结束调用链。
```

### 示例 19

```
我无法访问贵公司的具体第三季度报告。
通常，第三季度报告会包含……
```

### 示例 20

```python
from rag_service import RAGService

# Initialize service
rag_service = RAGService(
    index_service=None,  # Not needed for direct chat
    ollama_url="http://localhost:11434"
)

# Direct chat
response = rag_service.direct_chat(
    query="Explain machine learning",
    model="llama3.1:8b",
    temperature=0.7
)

print(response)
```

### 示例 21

```bash
# Direct call to Ollama API
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.1:8b",
    "prompt": "What is AI?",
    "stream": false
  }'
```

### 示例 22

```python
# 1. Use smaller model
model = "llama3.2"  # Instead of llama3.1:8b

# 2. Reduce max tokens
max_tokens = 256

# 3. Check Ollama is using GPU
# Run: ollama list
# Should show GPU memory usage
```

### 示例 23

```python
# 1. Lower temperature
temperature = 0.3  # More focused

# 2. Add constraints to prompt
prompt = f"Answer in 2-3 sentences: {query}"

# 3. Try different model
model = "qwen2.5"  # Better for certain tasks
```

### 示例 24

```python
# Recognize limitation
if requires_factual_accuracy(query):
    print("Recommendation: Use Context Enhanced mode for factual queries")
    use_rag_mode = True
else:
    use_direct_chat = True
```

</details>
