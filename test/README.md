# 🧪 Test模块

测试模块，包含单元测试和功能验证。

## 📁 文件说明

### `test_units.py`
- **功能**: 核心功能单元测试
- **职责**:
  - 倒排索引测试
  - CTR模型测试
  - 集成测试
  - 端到端测试

### `test_ctr_model.py`
- **功能**: CTR模型专项测试
- **职责**:
  - 特征提取测试
  - 模型训练测试
  - 预测功能测试

### `test_scores.py`
- **功能**: 分数计算测试
- **职责**:
  - TF-IDF分数测试
  - CTR分数测试
  - 排序效果测试

## 🚀 运行测试

```bash
# 运行所有测试
python test/test_units.py

# 运行特定测试
python -m unittest test.test_units.TestInvertedIndex
python -m unittest test.test_units.TestCTRModel

# 运行CTR模型测试
python test/test_ctr_model.py

# 运行分数测试
python test/test_scores.py
```

## 📊 测试覆盖

### 倒排索引测试
- 文本预处理
- 文档添加
- 搜索功能
- 摘要生成

### CTR模型测试
- 特征提取
- 模型训练
- CTR预测
- 数据验证

### 集成测试
- 端到端流程
- 模块协作
- 数据流转
- 错误处理

## 🔧 测试环境

- **Python**: 3.8+
- **依赖**: unittest, numpy, pandas
- **数据**: 自动生成测试数据
- **验证**: 功能正确性和性能指标 