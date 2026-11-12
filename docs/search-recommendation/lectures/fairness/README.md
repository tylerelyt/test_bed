# 分组效果与公平性分析

[项目说明](../../../../README.md) · [当前模块说明](../../README.md)

当前功能按查询、文档、位置或分数区间分组，比较准确率、精确率、召回率、F1 和可计算时的 AUC。代码还汇总组间均值、标准差、极差和变异系数，帮助发现表现不一致的分组。

位置区间在代码中使用 `pd.cut` 和边界 `[0, 3, 6, 10, inf]`，标签为 1–3、4–6、7–10、>10；位置为 0 的样本需要特别核对，不能沿用旧说明中的 0–3 分组。分数区间来自样本四分位数，不是固定的 0.2、0.5 等阈值。

## 怎样验证

1. 检查每组样本量与正负标签分布；当前分组构造会跳过少于 3 条样本的组。
2. 确认至少有两个组成功完成评估，记录缺失指标与失败原因。
3. 结合位置偏差和查询难度解释组间差异，再复核是否需要进一步调查。

## 使用边界

这是分组效果诊断，不等于已经实现人口统计均等、机会均等或法律合规认证。没有相应属性、定义和检验时，不应把“组间指标接近”表述为模型已经公平。

## 实现入口

- [src/search_engine/training_tab/model_fairness.py](../../../../src/search_engine/training_tab/model_fairness.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph LR
    A[模型预测] --> B[按特征分组]
    B --> C[计算各组指标]
    C --> D[比较效果]
    D --> E[识别差异]
    E --> F[报告结果]
```

### 示例 2

```python
# Example: Compare F1-scores across groups
group_f1_scores = {
    'position_0_3': 0.85,
    'position_4_7': 0.78,
    'position_8+': 0.72
}

# Calculate disparity
max_f1 = max(group_f1_scores.values())
min_f1 = min(group_f1_scores.values())
disparity = max_f1 - min_f1  # 0.13 (13% gap)
```

### 示例 3

```python
def group_by_feature(data, feature_name):
    """Group data by specified feature"""
    groups = {}
    for record in data:
        group_key = get_group_key(record, feature_name)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(record)
    return groups
```

### 示例 4

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_group_metrics(y_true, y_pred, y_pred_proba):
    """Calculate performance metrics for a group"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'size': len(y_true)  # Group size
    }
```

### 示例 5

```python
def analyze_fairness(group_metrics):
    """Analyze fairness across groups"""
    disparities = {}
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        metric_values = [g[metric_name] for g in group_metrics.values()]
        disparities[metric_name] = {
            'max': max(metric_values),
            'min': min(metric_values),
            'gap': max(metric_values) - min(metric_values),
            'std': np.std(metric_values)
        }
    return disparities
```

</details>
