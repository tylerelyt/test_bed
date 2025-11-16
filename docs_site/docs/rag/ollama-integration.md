---
layout: default
title: Ollama Integration
parent: Context Engineering
nav_order: 4
---

# Ollama Integration
{: .no_toc }

Setup and configuration guide for local LLM inference using Ollama.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

### macOS / Linux

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve
```

### Windows

Download from [ollama.com](https://ollama.com/download)

---

## Model Management

### Pull Models

```bash
# Pull popular models
ollama pull llama3.1:8b
ollama pull qwen2.5
ollama pull deepseek-coder

# List installed models
ollama list
```

### Recommended Models

| Model | Size | Use Case |
|:------|:-----|:---------|
| **llama3.2** | 3B | Fast general Q&A |
| **llama3.1:8b** | 8B | Balanced performance |
| **qwen2.5** | 7B | Chinese-optimized |
| **deepseek-coder** | 6.7B | Code generation |

---

## Configuration

### Default Settings

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

### Custom Endpoint

```python
# If Ollama runs on different port/host
ollama_url = "http://192.168.1.100:11434"
```

---

## Health Check

### Connection Test

```bash
# Test API
curl http://localhost:11434/api/tags

# Expected response:
# {"models": [...]}
```

### Web UI Test

1. Navigate to RAG tab
2. Click "Check Ollama Connection"
3. Should show: "✅ Connected, models: [...]"

---

## Troubleshooting

### Connection Failed

```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
killall ollama
ollama serve
```

### Model Not Found

```bash
# Pull the model
ollama pull llama3.1:8b

# Verify it's installed
ollama list
```

---

## Performance Tuning

### GPU Acceleration

Ollama automatically uses GPU if available:

```bash
# Check GPU usage (Linux/Windows)
nvidia-smi

# macOS: Uses Metal automatically
```

### Memory Management

```bash
# Limit model memory
export OLLAMA_MAX_LOADED_MODELS=1

# Unload unused models
ollama rm model-name
```

---

## Related Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Model Library](https://ollama.com/library)
- [API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
