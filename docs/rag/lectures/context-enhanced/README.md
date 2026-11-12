# 文档检索增强问答

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

文档检索增强把命中文档组织为上下文，再连同问题发送给模型。它改善模型可获得的信息，但仍需人工或评测检查回答是否忠实于原文。

当前从“RAG”进入，选择推理模式并保持“开启检索增强 (RAG)”打开，设置检索文档数量后执行查询。页面将回答、检索详情、上下文与实际提示词分别呈现。

## 怎样验证

1. 先确认文档索引存在，再用原文中有答案的问题执行一次查询。
2. 对照文档 ID、相关分数和上下文，检查生成内容的依据。
3. 加入一个无依据问题，检查空召回与模型回答，避免把流畅表达当成命中证据。

## 使用边界

增大 Top-K 可能引入噪声并增加上下文长度。提示词要求依据材料回答，不构成模型一定遵循的保证；应保留无命中和错误依据测试。

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
    A[用户查询] --> B[检索 Top-K 篇文档]
    B --> C[构建上下文]
    C --> D[生成提示词]
    D --> E[大语言模型推理]
    E --> F[有依据的回答]
```

### 示例 2

```python
def retrieve_documents(query: str, top_k: int = 5):
    """Retrieve relevant documents using TF-IDF"""
    results = index_service.search(query, top_k)
    return results
```

### 示例 3

```python
def enhance_with_context(query: str, documents: List[Document]):
    """Build enhanced prompt with context"""
    context = "\n\n".join([
        f"Document {i+1}: {doc.content}"
        for i, doc in enumerate(documents)
    ])

    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    return prompt
```

### 示例 4

```python
def generate_answer(prompt: str, model: str):
    """Generate answer using Ollama"""
    response = ollama_client.generate(
        model=model,
        prompt=prompt
    )
    return response
```

### 示例 5

```
根据提供的文档，TF-IDF（词频与逆文档频率）是信息检索中的统计指标，
用于评估某个词语对文档集合中一篇文档的重要性。
它结合词频（词语在文档中出现的频率）和逆文档频率
（词语在全部文档中的稀有程度）两项指标。
```

### 示例 6

```python
TOP_K_GUIDELINES = {
    "simple_questions": 3,       # "What is X?"
    "standard_queries": 5,       # Most queries (default)
    "complex_questions": 10,     # Requires multiple sources
    "exploratory": 15            # Broad research
}
```

### 示例 7

```python
# Avoid context overflow
MAX_CONTEXT_LENGTH = {
    "llama3.2": 4096,      # tokens
    "llama3.1:8b": 8192,
    "qwen2.5": 8192
}

# Truncate if necessary
def truncate_context(context, max_tokens=2000):
    """Keep context within model limits"""
    # Rough estimation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        return context[:max_chars] + "\n...(truncated)"
    return context
```

### 示例 8

```python
PROMPT_TEMPLATES = {
    "basic": """Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:""",

    "with_constraints": """Based on the following context, answer the question.
If the context doesn't contain the answer, say "I cannot answer based on
the provided context."

Context:
{context}

Question: {query}

Answer in Chinese:""",

    "with_citation": """Based on the following context, answer the question
and cite the document numbers you used.

Context:
{context}

Question: {query}

Answer (cite sources):"""
}
```

### 示例 9

```python
# 1. Increase Top-K
top_k = 10  # Retrieve more documents

# 2. Rephrase query
query = "TF-IDF algorithm"  # Instead of "how does search work"

# 3. Check index quality
# Verify documents are properly indexed
```

### 示例 10

```python
# Reduce Top-K
top_k = 3  # Instead of 10

# Truncate documents
def truncate_docs(docs, max_per_doc=500):
    return [doc[:max_per_doc] for doc in docs]

# Use model with larger context
model = "llama3.1:8b"  # 8K context instead of 4K
```

### 示例 11

```python
LATENCY_COMPONENTS = {
    "retrieval": 2.0,      # TF-IDF search
    "context_building": 0.5,  # String concatenation
    "llm_inference": 7.0,  # Main bottleneck
    "total": 9.5           # seconds
}
```

### 示例 12

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def retrieve_documents_cached(query, top_k):
    """Cache retrieval results for identical queries"""
    return index_service.search(query, top_k)
```

### 示例 13

```python
import concurrent.futures

def parallel_rag(queries):
    """Process multiple queries in parallel"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(rag_query, queries)
    return list(results)
```

### 示例 14

```python
def hybrid_retrieval(query, top_k=5):
    """Combine TF-IDF with semantic search"""

    # TF-IDF results
    tfidf_results = index_service.search(query, top_k)

    # Semantic results (if available)
    semantic_results = semantic_search(query, top_k)

    # Merge and rerank
    combined = merge_results(tfidf_results, semantic_results)
    return combined[:top_k]
```

### 示例 15

```python
def compress_context(context, query):
    """Extract only relevant sentences from context"""

    sentences = split_into_sentences(context)

    # Score sentences by relevance to query
    scored = [
        (sentence, relevance_score(sentence, query))
        for sentence in sentences
    ]

    # Keep top 50% most relevant
    sorted_sentences = sorted(scored, key=lambda x: x[1], reverse=True)
    compressed = [s[0] for s in sorted_sentences[:len(sentences)//2]]

    return " ".join(compressed)
```

</details>
