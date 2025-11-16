---
layout: default
title: Architecture
parent: RAG & Context Engineering
nav_order: 1
---

# RAG System Architecture
{: .no_toc }

Layered architecture design for retrieval-augmented generation.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Overview

###Overall Architecture

```mermaid
graph TB
    subgraph ApplicationLayer["Application Layer"]
        A[User Query] --> A1[Query Parser]
        A1 --> A2{Mode Router}
        A2 -->|Direct| B[Direct LLM]
        A2 -->|RAG| C[RAG Mode]
        A2 -->|ReAct| D[ReAct Mode]
    end
    
    subgraph ServiceLayer["Service Layer"]
        C --> E[RAGService]
        D --> E
        E --> E1[Retrieval Orchestration]
        E --> E2[Context Management]
        E --> E3[Reasoning Control]
        
        B --> F[LLMService]
        E2 --> F
        E3 --> F
    end
    
    subgraph DataLayer["Data Layer"]
        E1 --> G[SearchEngine<br/>TF-IDF]
        E1 --> H[VectorStore<br/>Embeddings]
        E1 --> I[KnowledgeGraph<br/>NetworkX]
    end
    
    subgraph InferenceLayer["Inference Layer"]
        F --> J[Ollama API]
        J --> K[Local LLM<br/>Llama/Qwen]
    end
```

---

## Core Components

### 1. RAG Service

**Responsibilities**:
- Orchestrate retrieval from multiple sources
- Manage context window and relevance filtering
- Handle multi-step reasoning loops

**Key Methods**:
```python
class RAGService:
    def retrieve_context(self, query: str, top_k: int = 5)
    def rerank_results(self, query: str, results: List)
    def answer_with_context(self, query: str, mode: str)
```

### 2. Hybrid Retrieval

**Three-Layer Retrieval**:
- **Keyword**: TF-IDF inverted index (fast, exact match)
- **Semantic**: Sentence-BERT embeddings (meaning-based)
- **Structured**: Knowledge graph traversal (entity relationships)

**Result Fusion**:
```python
final_score = α × keyword_score + β × semantic_score + γ × graph_score
```

### 3. Context Management

**Context Window Optimization**:
- Max tokens: 4096 (typical for Llama models)
- Reserve for output: 1024 tokens
- Available for context: ~3000 tokens

**Context Construction**:
```
System: You are a helpful assistant. Use the following context to answer.

Context:
[Top-3 retrieved passages]

User: {query}
