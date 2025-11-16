---
layout: default
title: Context Engineering
parent: RAG & Context Engineering
nav_order: 3
---

# Context Engineering
{: .no_toc }

Optimizing context construction and prompt engineering for RAG systems.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Context Window Management

### Token Budget Allocation

```python
# Typical allocation for 4K context window
CONTEXT_BUDGET = {
    "system_prompt": 100,      # System instructions
    "retrieved_docs": 2500,     # Main context
    "conversation_history": 400, # Recent turns
    "output_reserve": 1000      # Response generation
}
```

### Context Truncation Strategies

**1. Head Truncation**:
```python
# Keep most recent context
context = full_context[-max_tokens:]
```

**2. Tail Truncation**:
```python
# Keep beginning of context
context = full_context[:max_tokens]
```

**3. Sliding Window**:
```python
# Keep relevant parts around entities
context = extract_relevant_windows(full_context, entities, window_size=512)
```

---

## Prompt Engineering

### RAG Prompt Template

```python
RAG_PROMPT = """You are a knowledgeable assistant. Use the following context to answer the user's question accurately.

Context:
{context}

Conversation History:
{history}

User Question: {query}

Instructions:
- Base your answer on the provided context
- If the context doesn't contain the answer, say "I don't have enough information"
- Cite sources when possible
- Be concise but comprehensive

Answer:"""
```

### Multi-Document Context

```python
def format_context(retrieved_docs):
    """Format multiple documents with source attribution"""
    context_parts = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"""
Document {i} (Source: {doc['source']}):
{doc['content']}
---
""")
    
    return "\n".join(context_parts)
```

---

## Context Relevance Filtering

### Relevance Threshold

```python
def filter_by_relevance(results, query, threshold=0.5):
    """Only include sufficiently relevant documents"""
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    pairs = [[query, doc['content']] for doc in results]
    scores = reranker.predict(pairs)
    
    filtered = [
        doc for doc, score in zip(results, scores)
        if score > threshold
    ]
    
    return filtered
```

### Diversity Sampling

```python
def maximal_marginal_relevance(docs, query_embedding, lambda_param=0.5, top_k=5):
    """Select diverse set of documents"""
    selected = []
    remaining = docs.copy()
    
    # Select most relevant document first
    scores = cosine_similarity([query_embedding], [d['embedding'] for d in remaining])[0]
    best_idx = scores.argmax()
    selected.append(remaining.pop(best_idx))
    
    while len(selected) < top_k and remaining:
        # Balance relevance and diversity
        mmr_scores = []
        for doc in remaining:
            relevance = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
            
            # Max similarity to already selected docs
            redundancy = max([
                cosine_similarity([doc['embedding']], [s['embedding']])[0][0]
                for s in selected
            ])
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append(mmr_score)
        
        best_idx = np.argmax(mmr_scores)
        selected.append(remaining.pop(best_idx))
    
    return selected
```

---

## Citation & Attribution

### Inline Citations

```python
def add_citations(answer, sources):
    """Add citation markers to answer"""
    # Pattern: "Machine learning is a subset of AI [1]."
    return f"{answer} [Sources: {', '.join(sources)}]"
```

### Source Tracking

```python
class ContextWithSources:
    def __init__(self):
        self.context = ""
        self.source_map = {}  # sentence_id -> source_id
    
    def add_document(self, content, source):
        sentences = split_sentences(content)
        for sent in sentences:
            sent_id = len(self.source_map)
            self.context += f"[{sent_id}] {sent} "
            self.source_map[sent_id] = source
    
    def get_sources_for_answer(self, answer):
        """Extract which sources were used in answer"""
        used_sources = set()
        for sent_id, source in self.source_map.items():
            if f"[{sent_id}]" in answer:
                used_sources.add(source)
        return list(used_sources)
```

---

## Context Compression

### Extractive Summarization

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def compress_context(long_text, target_sentences=5):
    """Extract key sentences from long context"""
    parser = PlaintextParser.from_string(long_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    
    summary = summarizer(parser.document, target_sentences)
    return ' '.join([str(sentence) for sentence in summary])
```

### LLM-Based Compression

```python
def llm_compress_context(context, query, max_tokens=500):
    """Use LLM to compress context while preserving query-relevant info"""
    compression_prompt = f"""Summarize the following context, keeping only information relevant to answering: "{query}"

Context:
{context}

Compressed context (max {max_tokens} tokens):"""
    
    compressed = llm.generate(
        compression_prompt,
        max_tokens=max_tokens,
        temperature=0.3
    )
    
    return compressed
```

---

## Multi-Turn Context

### Conversation History Management

```python
class ConversationManager:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_turn(self, user_query, assistant_response):
        self.history.append({
            "user": user_query,
            "assistant": assistant_response
        })
        
        # Keep only recent turns
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def format_history(self):
        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"Assistant: {turn['assistant']}")
        return "\n".join(formatted)
```

### Coreference Resolution

```python
def resolve_coreferences(current_query, history):
    """Resolve pronouns using conversation history"""
    # Example: "What about its applications?" → "What about machine learning applications?"
    
    if has_pronouns(current_query):
        resolved = llm.generate(f"""Given the conversation history, rewrite the query to be self-contained.

History:
{format_history(history)}

Current query: {current_query}

Self-contained query:""")
        
        return resolved
    
    return current_query
```

---

## Advanced Techniques

### Contextual Embeddings

```python
def context_aware_embedding(query, conversation_history):
    """Embed query with conversation context"""
    # Prepend recent context to query for embedding
    contextualized_query = f"{conversation_history[-1]} {query}"
    return encode(contextualized_query)
```

### Dynamic Context Routing

```python
def route_context(query, available_contexts):
    """Select best context source based on query type"""
    query_type = classify_query(query)
    
    routing = {
        "factual": ["knowledge_base", "wikipedia"],
        "code": ["code_docs", "stackoverflow"],
        "conversational": ["general_corpus"]
    }
    
    selected_sources = routing.get(query_type, ["knowledge_base"])
    return [ctx for ctx in available_contexts if ctx.source in selected_sources]
```

---

## Best Practices

### Do's

- ✅ Prioritize most relevant documents (top-3 usually sufficient)
- ✅ Use source attribution for transparency
- ✅ Filter low-relevance content to reduce noise
- ✅ Balance context length with generation quality
- ✅ Track and optimize token usage

### Don'ts

- ❌ Don't exceed model's context window
- ❌ Don't include redundant/duplicate information
- ❌ Don't lose important details in compression
- ❌ Don't ignore conversation history in multi-turn
- ❌ Don't forget to handle out-of-context queries

