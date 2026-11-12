# 倒排索引与检索实现

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

`InvertedIndex` 保存文档、词频、文档频率与倒排列表。查询先分词，再取查询词命中的候选文档，只对这些候选计算分数；分数为零的文档不会进入结果。

当前分数累加各查询词的 TF × IDF，其中 TF 为词频除以文档长度，IDF 为 `log(文档总数 / 含词文档数)`。同分结果按文档 ID 稳定排序。因为没有平滑，出现在所有文档中的词，其 IDF 为零；这能解释“查询词存在但没有正分结果”的情况。

## 怎样验证

1. 使用少量已知文档验证分词、命中文档与排序顺序。
2. 分别测试空查询、无命中查询、同分结果。
3. 新增或删除文档后重新检查索引统计与结果。

## 使用边界

本实现属于词法检索。向量检索、混合召回和分布式索引是扩展方向，不能根据概念示例认定已经接入当前查询路径。

## 实现入口

- [src/search_engine/index_tab/offline_index.py](../../../../src/search_engine/index_tab/offline_index.py)
- [src/search_engine/index_tab/index_service.py](../../../../src/search_engine/index_tab/index_service.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```python
# File: src/search_engine/index_tab/offline_index.py
class InvertedIndex:
    """Inverted index implementation - Memory-optimized version"""
    
    def __init__(self):
        # Core data structures
        self.index = defaultdict(set)          # term -> document ID set
        self.doc_lengths = {}                  # document ID -> length
        self.documents = {}                    # document ID -> content
        self.term_freq = defaultdict(dict)     # term -> {doc_id: frequency}
        self.doc_freq = defaultdict(int)       # term -> document frequency
        
        # Chinese stop words
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去'
        }
```

### 示例 2

```python
def preprocess_text(self, text: str) -> List[str]:
    """Text preprocessing - Chinese word segmentation"""
    # Use jieba for Chinese word segmentation
    words = jieba.lcut(text.lower())
    
    # Filter stop words and short words
    words = [word for word in words 
             if len(word) > 1 and word not in self.stop_words]
    
    return words
```

### 示例 3

```python
def add_document(self, doc_id: str, content: str):
    """Add document to index - Inverted index construction"""
    # Save original document
    self.documents[doc_id] = content
    
    # Preprocess text
    words = self.preprocess_text(content)
    
    # Calculate document length
    self.doc_lengths[doc_id] = len(words)
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Update inverted index
    for word, freq in word_freq.items():
        self.index[word].add(doc_id)
        self.term_freq[word][doc_id] = freq
    
    # Update document frequency
    for word in word_freq:
        self.doc_freq[word] = len(self.index[word])
```

### 示例 4

```python
def search(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
    """TF-IDF retrieval core logic"""
    # Step 1: Query preprocessing
    query_terms = self.preprocess_text(query)
    if not query_terms:
        return []
    
    # Step 2: Candidate document recall
    candidates = set()
    for term in query_terms:
        if term in self.index:
            candidates.update(self.index[term])
    
    # Step 3: TF-IDF relevance calculation
    scores = {}
    total_docs = len(self.documents)
    
    for doc_id in candidates:
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        for term in query_terms:
            if term in self.term_freq and doc_id in self.term_freq[term]:
                # Calculate TF (Term Frequency)
                tf = self.term_freq[term][doc_id] / doc_length
                
                # Calculate IDF (Inverse Document Frequency)
                df = self.doc_freq[term]
                idf = math.log(total_docs / df) if df > 0 else 0
                
                # Calculate TF-IDF score
                score += tf * idf
        
        if score > 0:
            scores[doc_id] = score
    
    # Step 4: Top-K sorting and return
    sorted_results = sorted(scores.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_k]
    
    # Add document summaries
    results = []
    for doc_id, score in sorted_results:
        summary = self.documents[doc_id][:200] + "..." \
                  if len(self.documents[doc_id]) > 200 \
                  else self.documents[doc_id]
        results.append((doc_id, score, summary))
    
    return results
```

### 示例 5

```python
# File: src/search_engine/training_tab/ctr_model.py
def extract_features(self, ctr_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 7-dimensional feature vector from CTR data"""
    df = pd.DataFrame(ctr_data)
    
    # Feature 1: Position feature (absolute position)
    position_features = df['position'].values.reshape(-1, 1)
    
    # Feature 2: Document length feature
    doc_lengths = df['summary'].str.len().values.reshape(-1, 1)
    
    # Feature 3: Query length feature
    query_lengths = df['query'].str.len().values.reshape(-1, 1)
    
    # Feature 4: Summary length feature
    summary_lengths = df['summary'].str.len().values.reshape(-1, 1)
    
    # Feature 5: Query-summary match score (core relevance feature)
    match_scores = []
    for _, row in df.iterrows():
        query_words = set(jieba.lcut(row['query']))
        summary_words = set(jieba.lcut(row['summary']))
        if len(query_words) > 0:
            match_ratio = len(query_words.intersection(summary_words)) / len(query_words)
        else:
            match_ratio = 0
        match_scores.append(match_ratio)
    match_scores = np.array(match_scores).reshape(-1, 1)
    
    # 6-7. Historical CTR features (avoid data leakage)
    query_ctr_features, doc_ctr_features = self._extract_historical_ctr(df)
    
    # Combine all features into 7-dimensional vector
    X = np.column_stack([
        position_features,      # Dimension 1: Position
        doc_lengths,           # Dimension 2: Document length
        query_lengths,         # Dimension 3: Query length
        summary_lengths,       # Dimension 4: Summary length
        match_scores,          # Dimension 5: Match score
        query_ctr_features,    # Dimension 6: Query historical CTR
        doc_ctr_features       # Dimension 7: Document historical CTR
    ])
    
    # Extract labels
    y = df['clicked'].values
    
    return X, y
```

### 示例 6

```python
def train(self, training_data: List[Dict]) -> Dict[str, float]:
    """Complete model training pipeline"""
    # Step 1: Feature extraction
    X, y = self.extract_features(training_data)
    if len(X) == 0:
        return {"error": "No training data"}
    
    # Step 2: Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 3: Feature standardization
    self.scaler = StandardScaler()
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)
    
    # Step 4: Model training
    self.model = LogisticRegression(
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000
    )
    self.model.fit(X_train_scaled, y_train)
    
    # Step 5: Model evaluation
    y_pred = self.model.predict(X_test_scaled)
    y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    self.is_trained = True
    return metrics
```

### 示例 7

```python
def predict_ctr(self, query: str, doc_id: str, position: int, 
               tfidf_score: float, summary: str) -> float:
    """Predict CTR probability for single sample"""
    if not self.is_trained:
        return 0.5  # Return neutral probability if untrained
    
    # Construct feature vector
    features = np.array([[
        position,                    # Position feature
        len(summary),               # Document length
        len(query),                 # Query length
        len(summary),               # Summary length
        self._calculate_match_score(query, summary),  # Match score
        0.1,                        # Query historical CTR (simplified)
        0.1                         # Document historical CTR (simplified)
    ]])
    
    # Standardize and predict
    features_scaled = self.scaler.transform(features)
    ctr_prob = self.model.predict_proba(features_scaled)[0][1]
    
    return float(ctr_prob)
```

### 示例 8

```python
# File: src/search_engine/index_service.py
class IndexService:
    """Search service orchestrator"""
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Unified search interface"""
        # Stage 1: Recall - Get candidate documents
        candidates = self.inverted_index.search(query, top_k * 2)
        
        # Stage 2: Ranking - CTR reranking
        if self.ctr_enabled:
            return self.rank_with_ctr(query, candidates, top_k)
        else:
            return candidates[:top_k]
    
    def rank_with_ctr(self, query: str, candidates: List, top_k: int):
        """CTR reranking logic"""
        # Feature construction → Model prediction → Reranking
        features = self._build_features(query, candidates)
        ctr_scores = self.model_service.predict_batch(features)
        return self._rerank_by_ctr(candidates, ctr_scores, top_k)
```

### 示例 9

```python
# Test inverted index
def test_inverted_index():
    index = InvertedIndex()
    index.add_document("doc1", "artificial intelligence machine learning")
    results = index.search("machine learning", top_k=5)
    assert len(results) > 0
    assert results[0][0] == "doc1"

# Test CTR model
def test_ctr_model():
    model = CTRModel()
    training_data = load_test_data()
    metrics = model.train(training_data)
    assert metrics['accuracy'] > 0.5
```

### 示例 10

```python
# Test end-to-end search pipeline
def test_search_pipeline():
    service = IndexService()
    service.load_documents("test_documents.json")
    results = service.search("test query", mode="ctr")
    assert len(results) > 0
    assert all(isinstance(r, tuple) for r in results)
```

</details>
