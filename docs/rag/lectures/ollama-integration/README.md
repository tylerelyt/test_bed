# Ollama 兼容路径说明

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

`RAGService` 保留 `ollama_url`、`check_ollama_connection()` 和旧版 Ollama 生成路径。系统状态也可能显示本机 11434 端口不可达。

当前页面的默认在线推理使用 DashScope API，本地推理使用已加载的训练模块模型。因而“Ollama 未连接”和“当前所选推理方式不可用”不是同一个判断。

## 怎样验证

1. 先核对页面选择的推理方式，再判断需要检查哪项服务。
2. 仅在使用兼容路径时检查 Ollama 的实际地址和模型。
3. 如果 DashScope 或本地模型调用失败，查看对应密钥、端点和加载日志。

## 使用边界

启动 Ollama 不会自动替换当前页面的 DashScope 或本地模型路径。本页不重复维护另一个默认启动方案。

## 实现入口

- [src/search_engine/rag_tab/rag_service.py](../../../../src/search_engine/rag_tab/rag_service.py)
- [src/search_engine/rag_tab/rag_tab.py](../../../../src/search_engine/rag_tab/rag_tab.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve
```

### 示例 2

```bash
# Pull popular models
ollama pull llama3.1:8b
ollama pull qwen2.5
ollama pull deepseek-coder

# List installed models
ollama list
```

### 示例 3

```python
OLLAMA_CONFIG = {
    "url": "http://localhost:11434",
    "default_model": "llama3.1:8b",
    "timeout": 30,
    "generation_options": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }
}
```

### 示例 4

```python
# If Ollama runs on different port/host
ollama_url = "http://192.168.1.100:11434"
```

### 示例 5

```bash
# Test API
curl http://localhost:11434/api/tags

# Expected response:
# {"models": [...]}
```

### 示例 6

```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
killall ollama
ollama serve
```

### 示例 7

```bash
# Pull the model
ollama pull llama3.1:8b

# Verify it's installed
ollama list
```

### 示例 8

```bash
# Check GPU usage (Linux/Windows)
nvidia-smi

# macOS: Uses Metal automatically
```

### 示例 9

```bash
# Limit model memory
export OLLAMA_MAX_LOADED_MODELS=1

# Unload unused models
ollama rm model-name
```

</details>
