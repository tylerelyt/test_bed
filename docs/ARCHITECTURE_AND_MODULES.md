# 🏗️ 搜索引擎测试床 - MLOps架构设计

## 🎯 架构概述

本系统采用**服务解耦的MLOps架构**，将搜索引擎的各个功能模块抽象为独立的服务，通过标准化的接口进行交互，实现高内聚、低耦合的系统设计。

## 🏗️ 整体架构

### 服务分层架构

```mermaid
graph TB
    subgraph "🎨 表现层 - UI界面"
        PORTAL[Portal界面<br/>🚪 统一入口]
        SEARCH_TAB[Search Tab<br/>🔍 检索实验]
        TRAINING_TAB[Training Tab<br/>📊 数据训练]
        INDEX_TAB[Index Tab<br/>🏗️ 索引管理]
        MONITOR_TAB[Monitoring Tab<br/>📈 系统监控]
    end
    
    subgraph "🔧 业务层 - 服务解耦"
        DATA_SERVICE[DataService<br/>📊 数据服务]
        INDEX_SERVICE[IndexService<br/>📚 索引服务]
        MODEL_SERVICE[ModelService<br/>🤖 模型服务]
        EXPERIMENT_SERVICE[ExperimentService<br/>🧪 实验服务]
    end
    
    subgraph "💾 数据层 - 持久化存储"
        CTR_DATA[CTR样本数据<br/>models/ctr_data.json]
        INDEX_DATA[倒排索引数据<br/>models/index_data.json]
        MODEL_DATA[训练模型<br/>models/ctr_model.pkl]
        LOGS[系统日志<br/>logs/]
    end
    
    PORTAL --> DATA_SERVICE
    PORTAL --> INDEX_SERVICE
    PORTAL --> MODEL_SERVICE
    PORTAL --> EXPERIMENT_SERVICE
    
    SEARCH_TAB --> DATA_SERVICE
    SEARCH_TAB --> INDEX_SERVICE
    
    TRAINING_TAB --> DATA_SERVICE
    TRAINING_TAB --> MODEL_SERVICE
    
    INDEX_TAB --> INDEX_SERVICE
    
    MONITOR_TAB --> DATA_SERVICE
    MONITOR_TAB --> INDEX_SERVICE
    MONITOR_TAB --> MODEL_SERVICE
    
    DATA_SERVICE --> CTR_DATA
    INDEX_SERVICE --> INDEX_DATA
    MODEL_SERVICE --> MODEL_DATA
    
    style PORTAL fill:#ff6b6b
    style DATA_SERVICE fill:#4ecdc4
    style INDEX_SERVICE fill:#45b7d1
    style MODEL_SERVICE fill:#96ceb4
    style EXPERIMENT_SERVICE fill:#feca57
```

### MLOps数据流架构

```mermaid
flowchart LR
    subgraph "📊 DataOps - 数据运营"
        A[用户查询] --> B[DataService.record_impression]
        C[用户点击] --> D[DataService.record_click]
        B --> E[CTR样本生成]
        D --> E
        E --> F[数据质量检查]
        F --> G[特征工程]
    end
    
    subgraph "🤖 ModelOps - 模型运营"
        G --> H[ModelService.train_model]
        H --> I[模型评估]
        I --> J[模型部署]
        J --> K[在线预测]
    end
    
    subgraph "🔍 在线服务"
        A --> L[IndexService.retrieve]
        L --> M[IndexService.rank]
        M --> N[ModelService.predict_ctr]
        N --> O[结果排序]
        O --> C
    end
    
    subgraph "🧪 实验管理"
        P[ExperimentService.create_experiment]
        P --> Q[A/B测试]
        Q --> R[效果对比]
        R --> S[模型选择]
    end
    
    K --> N
    S --> J
    
    style A fill:#ff9ff3
    style G fill:#4ecdc4
    style H fill:#45b7d1
    style P fill:#feca57
```

## 🔧 核心服务设计

### 📊 DataService - 数据服务

**职责**: CTR样本的采集、存储、管理和数据质量保证

**核心功能**:
```python
class DataService:
    def record_impression(self, query, doc_id, position, score, summary, request_id)
    def record_click(self, doc_id, request_id)
    def get_samples_dataframe(self, request_id=None)
    def get_all_samples(self)
    def get_stats(self)
    def clear_data(self)
    def import_data(self, data)
    def export_data(self, format='json')
```

**数据流**:
```mermaid
sequenceDiagram
    participant UI as UI界面
    participant DS as DataService
    participant DB as 数据存储
    
    UI->>DS: record_impression(query, doc_id, position)
    DS->>DS: 生成CTR样本
    DS->>DS: 特征工程
    DS->>DB: 保存样本
    DS-->>UI: 确认记录
    
    UI->>DS: record_click(doc_id, request_id)
    DS->>DS: 更新clicked字段
    DS->>DB: 保存更新
    DS-->>UI: 确认点击
```

### 📚 IndexService - 索引服务

**职责**: 倒排索引的构建、查询、管理和文档检索

**核心功能**:
```python
class IndexService:
    def build_index(self, documents)
    def retrieve(self, query, top_k=20)
    def rank(self, query, doc_ids, top_k=10)
    def get_document_page(self, doc_id, request_id, data_service)
    def get_index_stats(self)
    def search_documents(self, query)
```

**索引流程**:
```mermaid
flowchart TD
    A[原始文档] --> B[文档预处理]
    B --> C[分词处理]
    C --> D[倒排索引构建]
    D --> E[TF-IDF计算]
    E --> F[索引持久化]
    F --> G[索引加载]
    G --> H[查询处理]
    H --> I[文档召回]
    I --> J[结果排序]
```

### 🤖 ModelService - 模型服务

**职责**: CTR模型的训练、评估、部署和在线预测

**核心功能**:
```python
class ModelService:
    def train_model(self, samples)
    def predict_ctr(self, features)
    def evaluate_model(self, test_samples)
    def save_model(self, model_path)
    def load_model(self, model_path)
    def get_feature_importance(self)
    def get_model_stats(self)
```

**训练流程**:
```mermaid
flowchart LR
    A[CTR样本] --> B[特征提取]
    B --> C[特征工程]
    C --> D[数据分割]
    D --> E[模型训练]
    E --> F[模型评估]
    F --> G[模型保存]
    G --> H[在线预测]
```

### 🧪 ExperimentService - 实验服务

**职责**: 实验管理、A/B测试、版本控制和效果对比

**核心功能**:
```python
class ExperimentService:
    def create_experiment(self, name, description)
    def run_ab_test(self, experiment_id, variants)
    def compare_results(self, experiment_id)
    def select_best_model(self, experiment_id)
    def get_experiment_history(self)
    def export_experiment_results(self, experiment_id)
```

## 📊 模块依赖关系

### 服务依赖图

```mermaid
graph TD
    subgraph "UI层"
        PORTAL[Portal]
        SEARCH_TAB[Search Tab]
        TRAINING_TAB[Training Tab]
        INDEX_TAB[Index Tab]
        MONITOR_TAB[Monitoring Tab]
    end
    
    subgraph "服务层"
        DATA_SERVICE[DataService]
        INDEX_SERVICE[IndexService]
        MODEL_SERVICE[ModelService]
        EXPERIMENT_SERVICE[ExperimentService]
    end
    
    subgraph "数据层"
        CTR_DATA[CTR数据]
        INDEX_DATA[索引数据]
        MODEL_DATA[模型数据]
    end
    
    PORTAL --> DATA_SERVICE
    PORTAL --> INDEX_SERVICE
    PORTAL --> MODEL_SERVICE
    PORTAL --> EXPERIMENT_SERVICE
    
    SEARCH_TAB --> DATA_SERVICE
    SEARCH_TAB --> INDEX_SERVICE
    
    TRAINING_TAB --> DATA_SERVICE
    TRAINING_TAB --> MODEL_SERVICE
    
    INDEX_TAB --> INDEX_SERVICE
    
    MONITOR_TAB --> DATA_SERVICE
    MONITOR_TAB --> INDEX_SERVICE
    MONITOR_TAB --> MODEL_SERVICE
    
    DATA_SERVICE --> CTR_DATA
    INDEX_SERVICE --> INDEX_DATA
    MODEL_SERVICE --> MODEL_DATA
    
    MODEL_SERVICE -.-> DATA_SERVICE
    INDEX_SERVICE -.-> DATA_SERVICE
```

### 文件结构

```
src/search_engine/
├── portal.py                 # 🚪 统一入口
├── data_service.py           # 📊 数据服务
├── index_service.py          # 📚 索引服务
├── model_service.py          # 🤖 模型服务
├── experiment_service.py     # 🧪 实验服务
├── search_tab/
│   └── search_tab.py        # 🔍 检索实验Tab
├── training_tab/
│   ├── training_tab.py      # 📊 数据训练Tab
│   └── ctr_config.py        # ⚙️ CTR配置
├── index_tab/
│   └── index_tab.py         # 🏗️ 索引管理Tab
└── monitoring_tab/
    └── monitoring_tab.py    # 📈 系统监控Tab
```

## 🔄 数据流设计

### 完整工作流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant P as Portal
    participant DS as DataService
    participant IS as IndexService
    participant MS as ModelService
    participant ES as ExperimentService
    
    U->>P: 输入查询
    P->>IS: retrieve(query)
    IS-->>P: 召回结果
    P->>MS: predict_ctr(results)
    MS-->>P: CTR分数
    P->>P: 排序结果
    P->>DS: record_impression(query, doc_id, position)
    P-->>U: 展示结果
    
    U->>P: 点击文档
    P->>DS: record_click(doc_id, request_id)
    P->>IS: get_document_page(doc_id)
    IS-->>P: 文档内容
    P-->>U: 显示文档
    
    U->>P: 训练模型
    P->>DS: get_all_samples()
    DS-->>P: CTR样本
    P->>MS: train_model(samples)
    MS-->>P: 训练结果
    P->>ES: create_experiment(results)
    P-->>U: 训练完成
```

### CTR样本数据结构

```mermaid
erDiagram
    CTR_SAMPLE {
        string query "查询词"
        string doc_id "文档ID"
        int position "展示位置"
        float score "TF-IDF分数"
        string summary "文档摘要"
        string request_id "请求ID"
        string timestamp "时间戳"
        int clicked "是否点击"
        float match_score "匹配分数"
        float query_ctr "查询CTR"
        float doc_ctr "文档CTR"
        int doc_length "文档长度"
        int query_length "查询长度"
        int summary_length "摘要长度"
        float position_decay "位置衰减"
    }
```

## 🛠️ 扩展设计

### 服务扩展接口

所有服务都遵循标准接口设计：

```python
class BaseService:
    def __init__(self, config=None):
        self.config = config or {}
        self.status = "stopped"
    
    def start(self):
        """启动服务"""
        pass
    
    def stop(self):
        """停止服务"""
        pass
    
    def get_status(self):
        """获取服务状态"""
        return self.status
    
    def health_check(self):
        """健康检查"""
        pass
```

### 新服务添加流程

1. **创建服务类**: 继承 `BaseService` 或实现标准接口
2. **注册服务**: 在 `portal.py` 中注册新服务
3. **UI集成**: 在相应的 Tab 中调用服务方法
4. **配置管理**: 添加服务配置项
5. **监控集成**: 在 Monitoring Tab 中添加监控

### 算法扩展接口

```python
class AlgorithmInterface:
    def train(self, data):
        """训练算法"""
        pass
    
    def predict(self, input_data):
        """预测结果"""
        pass
    
    def evaluate(self, test_data):
        """评估效果"""
        pass
    
    def save(self, path):
        """保存模型"""
        pass
    
    def load(self, path):
        """加载模型"""
        pass
```

## 📈 性能设计

### 性能指标

- **检索延迟**: < 100ms
- **CTR预测**: < 50ms
- **模型训练**: < 30s (1000样本)
- **并发支持**: 多用户同时使用
- **数据一致性**: 实时落盘保证

### 优化策略

1. **索引优化**: 倒排索引预加载，查询缓存
2. **模型优化**: 模型预加载，批量预测
3. **数据优化**: 异步落盘，批量写入
4. **并发优化**: 线程安全，锁机制

## 🔒 安全设计

### 数据安全

- **数据隔离**: 不同用户数据隔离
- **访问控制**: 服务级别权限控制
- **数据加密**: 敏感数据加密存储
- **审计日志**: 完整操作审计

### 系统安全

- **输入验证**: 所有输入参数验证
- **异常处理**: 完善的异常处理机制
- **资源限制**: 防止资源耗尽攻击
- **监控告警**: 异常情况及时告警

## 📋 部署架构

### 单机部署

```mermaid
graph TB
    subgraph "单机环境"
        PORTAL[Portal服务]
        DATA_SERVICE[数据服务]
        INDEX_SERVICE[索引服务]
        MODEL_SERVICE[模型服务]
        EXPERIMENT_SERVICE[实验服务]
        STORAGE[本地存储]
    end
    
    PORTAL --> DATA_SERVICE
    PORTAL --> INDEX_SERVICE
    PORTAL --> MODEL_SERVICE
    PORTAL --> EXPERIMENT_SERVICE
    
    DATA_SERVICE --> STORAGE
    INDEX_SERVICE --> STORAGE
    MODEL_SERVICE --> STORAGE
```

### 分布式部署（未来扩展）

```mermaid
graph TB
    subgraph "负载均衡"
        LB[负载均衡器]
    end
    
    subgraph "应用层"
        PORTAL1[Portal实例1]
        PORTAL2[Portal实例2]
    end
    
    subgraph "服务层"
        DATA_SERVICE[数据服务集群]
        INDEX_SERVICE[索引服务集群]
        MODEL_SERVICE[模型服务集群]
    end
    
    subgraph "数据层"
        DB[分布式数据库]
        CACHE[缓存集群]
    end
    
    LB --> PORTAL1
    LB --> PORTAL2
    
    PORTAL1 --> DATA_SERVICE
    PORTAL1 --> INDEX_SERVICE
    PORTAL1 --> MODEL_SERVICE
    
    PORTAL2 --> DATA_SERVICE
    PORTAL2 --> INDEX_SERVICE
    PORTAL2 --> MODEL_SERVICE
    
    DATA_SERVICE --> DB
    INDEX_SERVICE --> DB
    MODEL_SERVICE --> DB
```

---

**🎯 基于服务解耦的MLOps架构，支持高可扩展、高可维护的搜索引擎算法验证平台！** 