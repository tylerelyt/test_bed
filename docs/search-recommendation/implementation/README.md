# 索引、检索与排序：代码与图解参考

[当前模块说明](../README.md) · [项目说明](../../../README.md)

原指南重复维护的启动步骤、能力列表和固定性能数字已经移出主说明。使用当前工作台时，先按模块 README 完成操作；需要理解旧设计时再展开下方示例。

## 当前实现入口

- [src/search_engine/index_tab/offline_index.py](../../../src/search_engine/index_tab/offline_index.py)
- [src/search_engine/search_tab/search_tab.py](../../../src/search_engine/search_tab/search_tab.py)
- [src/search_engine/data_service.py](../../../src/search_engine/data_service.py)
- [src/search_engine/model_service.py](../../../src/search_engine/model_service.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph TB
    subgraph "应用层 - Application Layer"
        A[用户查询] --> B[搜索接口 SearchInterface]
        B --> B1[查询预处理]
        B1 --> B2[参数验证]
    end
    
    subgraph "服务层 - Service Layer" 
        B2 --> C[索引服务 IndexService]
        C --> C1[检索编排]
        C1 --> C2[结果聚合]
        
        D[模型服务 ModelService] --> D1[特征工程]
        D1 --> D2[CTR预测]
        
        E[数据服务 DataService] --> E1[点击收集]
        E1 --> E2[样本生成]
    end
    
    subgraph "算法层 - Algorithm Layer"
        C2 --> F[召回算法]
        F --> F1[TF-IDF计算]
        F1 --> F2[候选文档集]
        
        F2 --> G[排序算法]
        G --> G1{排序策略}
        G1 -->|传统| G2[相关度排序]
        G1 -->|智能| D1
        D2 --> G3[CTR重排序]
    end
    
    subgraph "存储层 - Storage Layer"
        H[倒排索引 InvertedIndex] --> F1
        I[文档存储] --> F1
        J[模型存储] --> D
        K[点击日志] --> E
    end
    
    G2 --> L[结果返回]
    G3 --> L
    L --> M[用户反馈]
    M --> E1
    
    subgraph "训练离线流程"
        E2 --> N[模型训练]
        N --> N1[模型评估]
        N1 --> N2[模型发布]
        N2 --> J
    end
```

### 示例 2

```python
# 文件路径: src/search_engine/index_tab/offline_index.py
class InvertedIndex:
    """倒排索引实现 - 内存优化版本"""
    
    def __init__(self):
        # 核心数据结构
        self.index = defaultdict(set)          # 词项 -> 文档ID集合
        self.doc_lengths = {}                  # 文档ID -> 文档长度  
        self.documents = {}                    # 文档ID -> 文档内容
        self.term_freq = defaultdict(dict)     # 词项 -> {文档ID: 词频}
        self.doc_freq = defaultdict(int)       # 词项 -> 文档频率
        
        # 中文停用词表
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理 - 中文分词核心实现"""
        # 使用jieba进行中文分词
        words = jieba.lcut(text.lower())
        
        # 过滤停用词和短词
        words = [word for word in words if len(word) > 1 and word not in self.stop_words]
        
        return words
    
    def add_document(self, doc_id: str, content: str):
        """添加文档到索引 - 倒排索引构建"""
        # 保存原始文档
        self.documents[doc_id] = content
        
        # 预处理文本
        words = self.preprocess_text(content)
        
        # 计算文档长度
        self.doc_lengths[doc_id] = len(words)
        
        # 统计词频
        word_freq = Counter(words)
        
        # 更新倒排索引
        for word, freq in word_freq.items():
            self.index[word].add(doc_id)
            self.term_freq[word][doc_id] = freq
        
        # 更新文档频率
        for word in word_freq:
            self.doc_freq[word] = len(self.index[word])
    
    def search(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
        """TF-IDF检索核心逻辑"""
        # 1. 查询预处理
        query_terms = self.preprocess_text(query)
        if not query_terms:
            return []
        
        # 2. 候选文档召回
        candidates = set()
        for term in query_terms:
            if term in self.index:
                candidates.update(self.index[term])
        
        # 3. TF-IDF相关度计算
        scores = {}
        total_docs = len(self.documents)
        
        for doc_id in candidates:
            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            
            for term in query_terms:
                if term in self.term_freq and doc_id in self.term_freq[term]:
                    # TF计算
                    tf = self.term_freq[term][doc_id] / doc_length
                    
                    # IDF计算
                    df = self.doc_freq[term]
                    idf = math.log(total_docs / df) if df > 0 else 0
                    
                    # TF-IDF分数
                    score += tf * idf
            
            if score > 0:
                scores[doc_id] = score
        
        # 4. Top-K排序返回
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 添加文档摘要
        results = []
        for doc_id, score in sorted_results:
            summary = self.documents[doc_id][:200] + "..." if len(self.documents[doc_id]) > 200 else self.documents[doc_id]
            results.append((doc_id, score, summary))
        
        return results
```

### 示例 3

```python
# 文件路径: src/search_engine/training_tab/ctr_model.py
class CTRModel:
    """CTR预测模型 - 二分类问题"""
    
    def __init__(self):
        self.model = None              # 逻辑回归模型
        self.vectorizer = None         # 文本向量化器
        self.scaler = None             # 特征标准化器
        self.is_trained = False        # 训练状态标志
    
    def extract_features(self, ctr_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """从CTR数据中提取7维特征向量"""
        if not ctr_data:
            return np.array([]), np.array([])
        
        # 转换为DataFrame便于处理
        df = pd.DataFrame(ctr_data)
        
        # 1. 位置特征（绝对位置）- 位置偏差是CTR的重要因子
        position_features = df['position'].values.reshape(-1, 1)
        
        # 2. 文档长度特征 - 长文档可能更权威
        doc_lengths = df['summary'].str.len().values.reshape(-1, 1)
        
        # 3. 查询长度特征 - 长查询通常更具体
        query_lengths = df['query'].str.len().values.reshape(-1, 1)
        
        # 4. 摘要长度特征
        summary_lengths = df['summary'].str.len().values.reshape(-1, 1)
        
        # 5. 查询词在摘要中的匹配度 - 核心相关性特征
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
        
        # 6-7. 历史点击率特征（避免数据泄露）
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        query_ctr_features = []
        doc_ctr_features = []
        
        for idx, row in df_sorted.iterrows():
            # 查询历史CTR - 只使用当前样本之前的数据
            query = row['query']
            query_history = df_sorted.loc[:idx-1]  # 不包括当前样本
            query_history_filtered = query_history[query_history['query'] == query]
            
            if len(query_history_filtered) > 0:
                query_ctr = query_history_filtered['clicked'].mean()
            else:
                query_ctr = 0.1  # 冷启动默认值
            query_ctr_features.append(query_ctr)
            
            # 文档历史CTR
            doc_id = row['doc_id']
            doc_history = df_sorted.loc[:idx-1]
            doc_history_filtered = doc_history[doc_history['doc_id'] == doc_id]
            
            if len(doc_history_filtered) > 0:
                doc_ctr = doc_history_filtered['clicked'].mean()
            else:
                doc_ctr = 0.1  # 冷启动默认值
            doc_ctr_features.append(doc_ctr)
        
        # 按原始顺序重新排列
        original_order = df.index
        query_ctr_features = [query_ctr_features[i] for i in original_order]
        doc_ctr_features = [doc_ctr_features[i] for i in original_order]
        
        query_ctr_features = np.array(query_ctr_features).reshape(-1, 1)
        doc_ctr_features = np.array(doc_ctr_features).reshape(-1, 1)
        
        # 组合所有特征为7维向量
        X = np.column_stack([
            position_features,      # 第1维：位置
            doc_lengths,           # 第2维：文档长度
            query_lengths,         # 第3维：查询长度  
            summary_lengths,       # 第4维：摘要长度
            match_scores,          # 第5维：匹配度
            query_ctr_features,    # 第6维：查询历史CTR
            doc_ctr_features       # 第7维：文档历史CTR
        ])
        
        # 提取标签
        y = df['clicked'].values
        
        return X, y
    
    def train(self, training_data: List[Dict]) -> Dict[str, float]:
        """模型训练完整流程"""
        # 1. 特征提取
        X, y = self.extract_features(training_data)
        if len(X) == 0:
            return {"error": "无训练数据"}
        
        # 2. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3. 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. 模型训练
        self.model = LogisticRegression(
            random_state=42,
            class_weight='balanced',  # 处理类别不平衡
            max_iter=1000
        )
        self.model.fit(X_train_scaled, y_train)
        
        # 5. 模型评估
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
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
    
    def predict_ctr(self, query: str, doc_id: str, position: int, 
                   tfidf_score: float, summary: str) -> float:
        """预测单个样本的CTR概率"""
        if not self.is_trained:
            return 0.5  # 未训练时返回中性概率
        
        # 构造特征向量
        features = np.array([[
            position,                    # 位置特征
            len(summary),               # 文档长度
            len(query),                 # 查询长度
            len(summary),               # 摘要长度
            self._calculate_match_score(query, summary),  # 匹配度
            0.1,                        # 查询历史CTR（简化）
            0.1                         # 文档历史CTR（简化）
        ]])
        
        # 标准化并预测
        features_scaled = self.scaler.transform(features)
        ctr_prob = self.model.predict_proba(features_scaled)[0][1]
        
        return float(ctr_prob)
    
    def _calculate_match_score(self, query: str, summary: str) -> float:
        """计算查询和摘要的匹配度"""
        query_words = set(jieba.lcut(query))
        summary_words = set(jieba.lcut(summary))
        
        if len(query_words) == 0:
            return 0.0
        
        return len(query_words.intersection(summary_words)) / len(query_words)
```

### 示例 4

```python
# 文件: src/search_engine/index_service.py
class IndexService:
    """搜索服务编排器"""
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """统一搜索接口"""
        # 1. 召回阶段 - 获取候选文档
        candidates = self.inverted_index.search(query, top_k * 2)
        
        # 2. 排序阶段 - CTR重排序  
        if self.ctr_enabled:
            return self.rank_with_ctr(query, candidates, top_k)
        else:
            return candidates[:top_k]
    
    def rank_with_ctr(self, query: str, candidates: List, top_k: int):
        """CTR重排序逻辑"""
        # 特征构造 → 模型预测 → 重排序
        features = self._build_features(query, candidates)
        ctr_scores = self.model_service.predict_batch(features)
        return self._rerank_by_ctr(candidates, ctr_scores, top_k)
```

### 示例 5

```mermaid
sequenceDiagram
    participant U as 用户
    participant SI as SearchInterface
    participant IS as IndexService
    participant II as InvertedIndex
    participant MS as ModelService
    
    U->>SI: 搜索请求(query)
    SI->>SI: 参数校验&预处理
    SI->>IS: search(query, top_k)
    
    IS->>II: 召回候选文档
    II->>II: TF-IDF计算
    II-->>IS: 候选文档列表
    
    alt CTR排序模式
        IS->>MS: 特征构造&预测
        MS->>MS: CTR模型推理
        MS-->>IS: CTR分数列表
        IS->>IS: 重排序
    end
    
    IS-->>SI: 最终结果
    SI-->>U: 搜索结果展示
    
    U->>SI: 点击反馈
    SI->>DataService: 记录点击日志
```

### 示例 6

```mermaid
sequenceDiagram
    participant DS as DataService
    participant TS as TrainingService
    participant MS as ModelService
    
    DS->>DS: 收集点击日志
    DS->>TS: 触发模型训练
    
    TS->>TS: 样本构造&特征工程
    TS->>TS: 模型训练&评估
    TS->>MS: 模型发布
    
    MS->>MS: 模型热更新
    Note over MS: 在线服务无中断切换
```

### 示例 7

```python
# 索引分片策略
def get_shard_id(term: str, shard_num: int) -> int:
    return hash(term) % shard_num

# 分布式检索协调
class DistributedIndexService:
    def search(self, query: str) -> List:
        # 1. 查询分发到各分片
        shard_results = self.query_all_shards(query)
        # 2. 结果合并和重排序
        return self.merge_and_rank(shard_results)
```

### 示例 8

```python
# 核心监控指标
class SearchMetrics:
    # 业务指标  
    ctr: float           # 点击率
    mrr: float           # 平均倒数排名
    ndcg_at_10: float    # 归一化折损累积增益
    
    # 系统指标
    avg_latency: float   # 平均延迟
    p95_latency: float   # P95延迟
    qps: float           # 每秒查询数
    
    # 数据指标
    index_coverage: float    # 索引覆盖率
    click_data_quality: float # 点击数据质量
```

### 示例 9

```
src/search_engine/
├── search_tab/                     # 搜索功能模块
│   ├── search_engine.py           # 搜索引擎核心实现
│   ├── search_interface.py        # 搜索接口定义
│   └── search_tab.py              # UI界面和交互逻辑
├── index_tab/                      # 索引功能模块  
│   ├── offline_index.py           # 倒排索引核心实现★
│   ├── index_service.py           # 索引服务编排
│   └── index_tab.py               # 索引管理UI
├── training_tab/                   # 训练功能模块
│   ├── ctr_model.py               # CTR模型实现★
│   ├── ctr_config.py              # 模型配置
│   ├── ctr_wide_deep_model.py     # Wide&Deep架构
│   └── training_tab.py            # 训练界面
├── data_service.py                 # 数据服务★
├── model_service.py                # 模型服务★
└── service_manager.py              # 服务管理器
```

### 示例 10

```
models/                             # 模型存储目录
├── ctr_model.pkl                  # CTR模型文件
├── index_data.json                # 索引元数据
└── training_data/                 # 训练数据目录
    ├── click_samples.json         # 点击样本
    └── feature_cache/             # 特征缓存

data/                              # 原始数据目录
├── preloaded_documents.json      # 预加载文档
└── openkg_triples.tsv            # 知识图谱数据

logs/                              # 日志目录
└── data_quality_report.json      # 数据质量报告
```

### 示例 11

```python
# 主要搜索入口 - src/search_engine/search_tab/search_tab.py
def perform_search(index_service, data_service, query: str, 
                  sort_mode: str = "ctr", model_type: str = "logistic_regression"):
    """
    执行搜索的主入口函数
    
    Args:
        index_service: 索引服务实例
        data_service: 数据服务实例  
        query: 用户查询
        sort_mode: 排序模式 "tfidf" 或 "ctr"
        model_type: 模型类型 "logistic_regression" 或 "wide_and_deep"
    
    Returns:
        搜索结果列表
    """
    
# 索引构建入口 - src/search_engine/index_tab/index_service.py  
def build_index_from_documents(documents: List[Dict]) -> bool:
    """
    从文档列表构建倒排索引
    
    Args:
        documents: 文档列表，格式 [{"id": "doc1", "content": "内容"}]
    
    Returns:
        构建是否成功
    """

# CTR训练入口 - src/search_engine/training_tab/ctr_model.py
def train_ctr_model(click_data: List[Dict]) -> Dict[str, float]:
    """
    训练CTR模型的主入口
    
    Args:
        click_data: 点击数据，包含query, doc_id, clicked等字段
    
    Returns:
        训练指标字典
    """
```

### 示例 12

```python
# CTR模型配置 - src/search_engine/training_tab/ctr_config.py
class CTRFeatureConfig:
    """CTR特征配置"""
    POSITION_WEIGHT = 1.0          # 位置特征权重
    CONTENT_WEIGHT = 0.8           # 内容特征权重  
    HISTORY_WEIGHT = 1.2           # 历史特征权重
    
class CTRTrainingConfig:
    """CTR训练配置"""
    TRAIN_TEST_SPLIT = 0.2         # 测试集比例
    RANDOM_STATE = 42              # 随机种子
    MAX_ITER = 1000                # 最大迭代次数
    CLASS_WEIGHT = 'balanced'      # 类别权重平衡
```

</details>
