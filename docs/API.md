# API 文档

## 概述

Intelligent Search Engine 提供了完整的Python API，支持程序化访问所有核心功能。

## 核心服务API

### DataService - 数据服务

#### 初始化
```python
from src.search_engine.data_service import DataService

# 使用默认配置
data_service = DataService()

# 自定义配置
data_service = DataService(
    auto_save_interval=30,  # 自动保存间隔（秒）
    batch_size=100         # 批量保存大小
)
```

#### 记录展示事件
```python
sample = data_service.record_impression(
    query="人工智能",
    doc_id="doc1",
    position=1,
    score=0.85,
    summary="人工智能相关文档摘要",
    request_id="req_123"
)
```

#### 记录点击事件
```python
success = data_service.record_click(
    doc_id="doc1",
    request_id="req_123"
)
```

#### 批量操作
```python
# 批量记录展示
impressions = [
    {
        "query": "机器学习",
        "doc_id": "doc1",
        "position": 1,
        "score": 0.9,
        "summary": "机器学习基础",
        "request_id": "req_1"
    },
    # ... 更多记录
]
result = data_service.batch_record_impressions(impressions)

# 批量记录点击
clicks = [
    {"doc_id": "doc1", "request_id": "req_1"},
    {"doc_id": "doc2", "request_id": "req_2"}
]
result = data_service.batch_record_clicks(clicks)
```

#### 数据查询
```python
# 获取所有样本
samples = data_service.get_all_samples()

# 获取指定请求的样本
samples = data_service.get_samples_by_request("req_123")

# 获取DataFrame格式
df = data_service.get_samples_dataframe()

# 获取统计信息
stats = data_service.get_stats()

# 按时间范围查询
samples = data_service.get_samples_by_time_range(
    start_time="2024-01-01T00:00:00",
    end_time="2024-12-31T23:59:59"
)

# 按查询模式查询
samples = data_service.get_samples_by_query_pattern("人工智能.*")
```

#### 数据管理
```python
# 清空数据
data_service.clear_data()

# 导出数据
data_service.export_data("backup.json")

# 导入数据
data_service.import_data("backup.json")

# 强制保存
data_service.force_save()

# 健康检查
health = data_service.get_data_health_check()
```

### IndexService - 索引服务

#### 初始化
```python
from src.search_engine.index_service import IndexService

index_service = IndexService()
```

#### 索引管理
```python
# 构建索引
success = index_service.build_index()

# 添加文档
success = index_service.add_document("doc1", "文档内容")

# 批量添加文档
documents = {
    "doc1": "第一个文档内容",
    "doc2": "第二个文档内容"
}
count = index_service.batch_add_documents(documents)

# 获取文档
content = index_service.get_document("doc1")

# 清空索引
index_service.clear_index()

# 保存索引
index_service.save_index()
```

#### 搜索功能
```python
# 召回文档ID
doc_ids = index_service.retrieve("人工智能", top_k=20)

# 排序结果
ranked_results = index_service.rank("人工智能", doc_ids, top_k=10)

# 完整搜索
results = index_service.search("人工智能", top_k=10)

# 获取文档页面
page = index_service.get_document_page("doc1", "req_123", data_service)
```

#### 索引统计
```python
# 获取索引统计
stats = index_service.get_stats()

# 导入/导出文档
index_service.export_documents("docs.json")
index_service.import_documents("docs.json")
```

### ModelService - 模型服务

#### 初始化
```python
from src.search_engine.model_service import ModelService

model_service = ModelService()
```

#### 模型训练
```python
# 训练模型
result = model_service.train_model(data_service)

# 评估模型
evaluation = model_service.evaluate_model(data_service)

# 验证训练数据
validation = model_service.validate_training_data(data_service)
```

#### 模型预测
```python
# CTR预测
ctr_score = model_service.predict_ctr(features)

# 批量预测
scores = model_service.batch_predict_ctr(features_list)
```

#### 模型管理
```python
# 保存模型
model_service.save_model()

# 加载模型
model_service.load_model("model.pkl")

# 获取模型信息
info = model_service.get_model_info()

# 导出/导入模型
model_service.export_model("model_export.pkl")
model_service.import_model("model_import.pkl")
```

### ServiceManager - 服务管理器

#### 使用单例服务
```python
from src.search_engine.service_manager import (
    get_data_service,
    get_index_service,
    get_model_service,
    service_manager
)

# 获取服务实例
data_service = get_data_service()
index_service = get_index_service()
model_service = get_model_service()

# 获取服务状态
status = service_manager.get_service_status()

# 重置服务
service_manager.reset_services()
```

## 数据工具API

### 便捷函数
```python
from src.search_engine.data_utils import (
    record_search_impression,
    record_document_click,
    get_ctr_samples,
    get_ctr_dataframe,
    get_data_statistics,
    clear_all_data,
    export_ctr_data,
    import_ctr_data,
    analyze_click_patterns
)

# 记录搜索展示
record_search_impression("查询", "doc1", 1, 0.8, "摘要", "req_1")

# 记录点击
record_document_click("doc1", "req_1")

# 获取数据
samples = get_ctr_samples()
df = get_ctr_dataframe()
stats = get_data_statistics()

# 数据分析
patterns = analyze_click_patterns()
```

### 数据验证
```python
from src.search_engine.data_utils import (
    validate_search_params,
    validate_click_params
)

# 验证搜索参数
errors = validate_search_params("查询", "doc1", 1, 0.8)

# 验证点击参数
errors = validate_click_params("doc1", "req_1")
```

## 配置API

### CTR样本配置
```python
from src.search_engine.training_tab.ctr_config import CTRSampleConfig

# 获取字段名
fields = CTRSampleConfig.get_field_names()

# 验证样本
errors = CTRSampleConfig.validate_sample(sample)

# 获取配置
config = CTRSampleConfig.get_config()
```

## 错误处理

### 异常类型
```python
try:
    data_service.record_impression("", "doc1", 1, 0.8, "摘要", "req_1")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"系统错误: {e}")
```

### 常见错误
- `ValueError`: 参数验证失败
- `FileNotFoundError`: 文件不存在
- `PermissionError`: 权限不足
- `MemoryError`: 内存不足

## 性能优化

### 批量操作
```python
# 使用批量操作而不是循环
impressions = [...]
data_service.batch_record_impressions(impressions)

# 而不是
for impression in impressions:
    data_service.record_impression(...)
```

### 缓存利用
```python
# 统计信息会被缓存
stats = data_service.get_stats()  # 第一次计算
stats = data_service.get_stats()  # 使用缓存
```

### 异步保存
```python
# 数据会异步保存，不阻塞主线程
data_service.record_impression(...)  # 立即返回
# 数据在后台保存
```

## 示例代码

### 完整搜索流程
```python
from src.search_engine.service_manager import (
    get_data_service,
    get_index_service,
    get_model_service
)

# 获取服务
data_service = get_data_service()
index_service = get_index_service()
model_service = get_model_service()

# 搜索流程
query = "人工智能"
request_id = "req_123"

# 1. 召回
doc_ids = index_service.retrieve(query, top_k=20)

# 2. 排序
ranked_results = index_service.rank(query, doc_ids, top_k=10)

# 3. 记录展示
for position, result in enumerate(ranked_results, 1):
    doc_id, score, summary = result
    data_service.record_impression(
        query, doc_id, position, score, summary, request_id
    )

# 4. 用户点击
clicked_doc_id = "doc1"
data_service.record_click(clicked_doc_id, request_id)

# 5. 模型训练
result = model_service.train_model(data_service)
```

### 数据分析示例
```python
from src.search_engine.data_utils import analyze_click_patterns

# 分析点击模式
patterns = analyze_click_patterns()

print(f"总体CTR: {patterns['overall_ctr']:.2%}")
print(f"总展示数: {patterns['total_impressions']}")
print(f"总点击数: {patterns['total_clicks']}")

# 位置分析
if patterns['position_analysis']:
    print("位置CTR分析:")
    for pos, stats in patterns['position_analysis'].items():
        print(f"  位置{pos}: CTR={stats['mean']:.2%}")
```

## 扩展开发

### 自定义排序算法
```python
from src.search_engine.index_service import IndexService

class CustomIndexService(IndexService):
    def custom_rank(self, query, doc_ids, top_k=10):
        # 实现自定义排序逻辑
        return ranked_results
```

### 自定义特征
```python
from src.search_engine.data_service import DataService

class CustomDataService(DataService):
    def _create_sample(self, query, doc_id, position, score, summary, request_id):
        sample = super()._create_sample(query, doc_id, position, score, summary, request_id)
        # 添加自定义特征
        sample['custom_feature'] = self._calculate_custom_feature(query, summary)
        return sample
```

更多API详情请参考源码注释和类型提示。 