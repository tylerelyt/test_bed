# Wide & Deep CTR模型实现文档

## 🎯 项目概述

本项目成功将原有的逻辑回归(LR) CTR模型替换为Wide & Deep模型，实现了特征记忆与泛化能力的融合。

## 🏗️ 架构设计

### Wide & Deep模型结构

```
输入层
├── Wide部分 (线性特征)
│   ├── 位置特征 (position, position_decay, position_group)
│   ├── 分数特征 (score, score_bins)
│   ├── 匹配特征 (match_score)
│   ├── 历史CTR特征 (query_ctr, doc_ctr)
│   └── 交互特征 (query_doc_interaction)
│
├── Deep部分 (非线性特征)
│   ├── 长度特征 (doc_length, query_length, summary_length)
│   ├── 词数量特征 (query_word_count, summary_word_count)
│   ├── 时间特征 (timestamp_hour)
│   ├── 交叉特征 (position×score, query_length×match_score)
│   └── 比率特征 (length_ratio, score_position_ratio)
│
└── 嵌入特征
    ├── 查询哈希嵌入 (query_hash)
    ├── 文档哈希嵌入 (doc_hash)
    └── 位置分组嵌入 (position_group)
```

## 🔧 技术特性

### 1. 特征工程优化
- **Wide特征**: 6维线性特征，包含位置、分数、匹配度等
- **Deep特征**: 8维非线性特征，包含长度、时间、交叉特征等
- **嵌入特征**: 查询、文档、位置的向量表示

### 2. 网络架构
- **Wide部分**: 128→64 全连接层，较小dropout (0.15)
- **Deep部分**: 256→128→64→32 全连接层，标准dropout (0.3)
- **嵌入维度**: 查询/文档16维，位置8维
- **合并策略**: Wide + Deep特征拼接后输出

### 3. 训练优化
- **早停策略**: 验证损失8轮不下降则停止
- **学习率衰减**: 验证损失5轮不下降则减半
- **批大小**: 64 (相比LR的在线学习，更适合深度学习)
- **训练轮数**: 最多30轮

## 📊 配置参数

```python
'wide_and_deep': {
    'name': 'Wide & Deep',
    'description': '结合线性模型和深度神经网络，效果更好',
    'params': {
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'wide_layers': [128, 64],
        'deep_layers': [256, 128, 64, 32],
        'embedding_dim': 16,
        'early_stopping_patience': 8
    }
}
```

## 🚀 使用方法

### 1. 模型选择
训练界面默认选择Wide & Deep模型，也可以手动选择其他模型类型。

### 2. 数据要求
- **最小样本数**: 10条记录
- **点击数据**: 至少2次点击
- **数据多样性**: 至少3个不同查询、文档、位置

### 3. 训练流程
```python
# 创建模型实例
model = WideAndDeepCTRModel()

# 训练模型
result = model.train(ctr_data)

# 预测CTR
ctr_score = model.predict_ctr(query, doc_id, position, score, summary)
```

## 📈 性能优势

### 相比LR模型的改进：

1. **特征记忆能力** (Wide部分)
   - 位置特征：精确捕捉位置对点击率的影响
   - 历史CTR：利用查询和文档的历史表现
   - 匹配特征：直接使用查询-文档匹配度

2. **泛化能力** (Deep部分)
   - 交叉特征：学习特征间的非线性关系
   - 嵌入特征：将离散特征转换为连续向量
   - 深度网络：多层非线性变换，捕捉复杂模式

3. **训练稳定性**
   - 早停机制：防止过拟合
   - 学习率衰减：自适应调整学习步长
   - BatchNormalization：加速训练收敛

## 🔍 评估指标

保持与原测试框架完全兼容：

- **准确率 (Accuracy)**: 分类正确率
- **AUC**: ROC曲线下面积
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均

## 🧪 测试验证

运行测试脚本验证模型功能：

```bash
python test_wide_deep_model.py
```

测试覆盖：
- ✅ 模型配置
- ✅ 模型创建
- ✅ 特征提取
- ✅ 模型训练

## 📁 文件结构

```
src/search_engine/training_tab/
├── ctr_config.py              # 模型配置 (已更新默认模型)
├── ctr_wide_deep_model.py     # Wide & Deep模型实现
├── training_tab.py            # 训练界面 (已更新默认选择)
└── __init__.py                # 模块初始化
```

## 🔄 兼容性说明

1. **训练接口**: 完全兼容原有训练流程
2. **评估指标**: 保持一致的评估标准
3. **数据格式**: 使用相同的CTR数据格式
4. **模型保存**: 支持模型保存和加载

## 🚀 部署建议

1. **生产环境**: 建议使用GPU加速训练
2. **模型更新**: 支持增量训练和模型热更新
3. **监控指标**: 关注AUC、CTR提升等业务指标
4. **A/B测试**: 与LR模型进行对比测试

## 📚 参考资料

- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [TensorFlow Wide & Deep Tutorial](https://www.tensorflow.org/tutorials/wide_and_deep)
- [CTR Prediction Best Practices](https://developers.google.com/machine-learning/recommendation)

## 🎉 总结

成功实现了从LR到Wide & Deep模型的升级，在保持原有功能的基础上，显著提升了模型的表达能力：

- **特征记忆**: Wide部分保持LR的线性特征优势
- **泛化能力**: Deep部分增加非线性建模能力
- **训练稳定**: 现代化的深度学习训练策略
- **完全兼容**: 无缝替换，无需修改现有代码

这个升级为CTR预测系统带来了更强的建模能力，有望显著提升点击率预测的准确性。
