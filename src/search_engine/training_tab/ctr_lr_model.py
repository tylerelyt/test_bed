#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逻辑回归CTR模型训练
基于倒排索引检索系统的点击数据训练点击率预测模型
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_ctr_data():
    """加载最新的CTR数据文件"""
    ctr_files = glob.glob('ctr_data_*.csv')
    
    if not ctr_files:
        print("❌ 未找到CTR数据文件")
        print("请先运行倒排索引检索系统并导出CTR数据")
        return None
    
    # 使用最新的文件
    latest_file = max(ctr_files)
    print(f"📁 使用数据文件: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ 成功加载CTR数据: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

def preprocess_features(df):
    """特征预处理"""
    print("🔧 特征预处理...")
    
    # 基础特征
    features = df[['position', 'score', 'doc_length']].copy()
    
    # 位置特征工程
    features['position_rank'] = 1.0 / features['position']  # 位置倒数
    features['position_log'] = np.log(features['position'])  # 位置对数
    
    # 分数特征工程
    features['score_squared'] = features['score'] ** 2
    features['score_log'] = np.log(features['score'] + 1e-6)
    
    # 文档长度特征工程
    features['doc_length_log'] = np.log(features['doc_length'] + 1)
    features['doc_length_ratio'] = features['doc_length'] / features['doc_length'].max()
    
    # 交互特征
    features['position_score'] = features['position'] * features['score']
    features['position_length'] = features['position'] * features['doc_length_log']
    
    # 查询特征（one-hot编码）
    query_encoded = pd.get_dummies(df['query'], prefix='query')
    features = pd.concat([features, query_encoded], axis=1)
    
    print(f"   特征数量: {features.shape[1]}")
    print(f"   特征列表: {list(features.columns)}")
    
    return features

def train_logistic_regression(X, y):
    """训练逻辑回归模型"""
    print("🤖 训练逻辑回归CTR模型...")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   训练集大小: {len(X_train)}")
    print(f"   测试集大小: {len(X_test)}")
    print(f"   训练集点击率: {np.mean(y_train):.4f}")
    print(f"   测试集点击率: {np.mean(y_test):.4f}")
    
    # 训练逻辑回归模型
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        solver='liblinear'  # 适合小数据集
    )
    
    lr_model.fit(X_train, y_train)
    
    # 预测
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    return lr_model, (X_train, X_test, y_train, y_test), (y_pred, y_prob)

def evaluate_model(y_test, y_pred, y_prob):
    """评估模型性能"""
    print("\n📊 逻辑回归模型评估结果:")
    print("=" * 50)
    
    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    
    print(f"   准确率 (Accuracy): {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
    print(f"   Log Loss: {loss:.4f}")
    
    # 分类报告
    print(f"\n   分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未点击', '点击']))
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': loss
    }

def analyze_feature_importance(model, X):
    """分析特征重要性"""
    print("\n🎯 特征重要性分析:")
    print("=" * 50)
    
    # 获取特征系数
    coefficients = model.coef_[0]
    feature_names = X.columns
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n📈 特征系数 (按重要性排序):")
    print(importance_df.head(15))
    
    # 分析正负影响
    positive_features = importance_df[importance_df['coefficient'] > 0].head(5)
    negative_features = importance_df[importance_df['coefficient'] < 0].head(5)
    
    print(f"\n✅ 正向影响特征 (提高点击率):")
    for _, row in positive_features.iterrows():
        print(f"   {row['feature']}: {row['coefficient']:.4f}")
    
    print(f"\n❌ 负向影响特征 (降低点击率):")
    for _, row in negative_features.iterrows():
        print(f"   {row['feature']}: {row['coefficient']:.4f}")
    
    return importance_df

def visualize_results(importance_df, metrics):
    """可视化结果"""
    print("\n📊 生成可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 特征重要性
    top_features = importance_df.head(10)
    axes[0, 0].barh(range(len(top_features)), top_features['coefficient'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_title('逻辑回归特征系数 (Top 10)')
    axes[0, 0].set_xlabel('Coefficient')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. 模型性能指标
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = axes[0, 1].bar(metric_names, metric_values, color=colors)
    axes[0, 1].set_title('模型性能指标')
    axes[0, 1].set_ylabel('Score')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, metric_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 3. 特征重要性分布
    axes[1, 0].hist(importance_df['coefficient'], bins=20, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('特征系数分布')
    axes[1, 0].set_xlabel('Coefficient')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. 位置与点击率关系
    # 这里需要原始数据，暂时用示例数据
    axes[1, 1].text(0.5, 0.5, '位置-点击率关系图\n(需要更多数据)', 
                   ha='center', va='center', transform=axes[1, 1].transAxes,
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('位置与点击率关系')
    
    plt.tight_layout()
    plt.savefig('ctr_lr_results.png', dpi=300, bbox_inches='tight')
    print("   图表已保存为: ctr_lr_results.png")

def generate_report(model, metrics, importance_df, X):
    """生成CTR分析报告"""
    print("\n📋 生成CTR分析报告...")
    
    report = f"""
# 逻辑回归CTR模型训练报告

## 模型性能总结

### 评估指标
- **准确率 (Accuracy)**: {metrics['accuracy']:.4f}
- **AUC**: {metrics['auc']:.4f}
- **Log Loss**: {metrics['log_loss']:.4f}

### 模型类型
- **算法**: 逻辑回归 (Logistic Regression)
- **求解器**: liblinear
- **正则化**: L2正则化

## 特征工程

### 特征数量
总特征数量: {X.shape[1]}

### 特征类型
- **基础特征**: position, score, doc_length
- **位置特征**: position_rank, position_log
- **分数特征**: score_squared, score_log
- **长度特征**: doc_length_log, doc_length_ratio
- **交互特征**: position_score, position_length
- **查询特征**: 查询词one-hot编码

## 特征重要性分析

### 正向影响特征 (Top 5)
"""
    
    positive_features = importance_df[importance_df['coefficient'] > 0].head(5)
    for _, row in positive_features.iterrows():
        report += f"- **{row['feature']}**: {row['coefficient']:.4f}\n"
    
    report += "\n### 负向影响特征 (Top 5)\n"
    negative_features = importance_df[importance_df['coefficient'] < 0].head(5)
    for _, row in negative_features.iterrows():
        report += f"- **{row['feature']}**: {row['coefficient']:.4f}\n"
    
    report += f"""

## 模型解释

### 逻辑回归优势
1. **可解释性强**: 系数直接表示特征对点击率的影响
2. **训练速度快**: 适合实时更新
3. **内存占用小**: 适合大规模部署
4. **概率输出**: 直接输出点击概率

### 特征影响分析
- **位置特征**: 位置越靠前，点击率越高
- **分数特征**: 相似度分数越高，点击率越高
- **长度特征**: 文档长度适中时点击率较高
- **查询特征**: 不同查询词的点击偏好不同

## 应用建议

### 1. 搜索结果排序优化
- 结合CTR预测分数调整排序
- 公式: 最终分数 = α × 相似度分数 + β × CTR预测分数

### 2. 模型更新策略
- 定期重新训练模型
- 考虑在线学习更新
- 监控模型性能变化

### 3. 特征扩展
- 添加用户特征
- 添加时间特征
- 添加上下文特征

### 4. 评估指标
- 主要关注AUC和Log Loss
- 业务指标: CTR提升、用户满意度
- A/B测试验证效果

## 技术细节

### 训练参数
- 训练集比例: 80%
- 测试集比例: 20%
- 随机种子: 42
- 最大迭代次数: 1000

### 数据质量
- 确保标签平衡
- 处理缺失值
- 特征标准化
"""
    
    with open('ctr_lr_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   报告已保存为: ctr_lr_analysis_report.md")

def save_model(model, X, importance_df):
    """保存模型和特征信息"""
    import pickle
    
    model_info = {
        'model': model,
        'feature_names': list(X.columns),
        'importance_df': importance_df
    }
    
    with open('ctr_lr_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("   模型已保存为: ctr_lr_model.pkl")

def main():
    """主函数"""
    print("🎯 逻辑回归CTR模型训练")
    print("=" * 50)
    
    # 加载数据
    df = load_ctr_data()
    if df is None:
        return
    
    # 数据概览
    print(f"\n📊 数据概览:")
    print(f"   总记录数: {len(df)}")
    print(f"   点击率: {df['clicked'].mean():.4f} ({df['clicked'].mean()*100:.2f}%)")
    print(f"   唯一查询数: {df['query'].nunique()}")
    print(f"   唯一文档数: {df['doc_id'].nunique()}")
    
    # 检查数据质量
    if df['clicked'].sum() < 10:
        print("⚠️  警告: 点击数据较少，可能影响模型训练效果")
    
    # 特征预处理
    X = preprocess_features(df)
    y = df['clicked']
    
    # 训练模型
    model, (X_train, X_test, y_train, y_test), (y_pred, y_prob) = train_logistic_regression(X, y)
    
    # 评估模型
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    # 特征重要性分析
    importance_df = analyze_feature_importance(model, X)
    
    # 可视化结果
    visualize_results(importance_df, metrics)
    
    # 生成报告
    generate_report(model, metrics, importance_df, X)
    
    # 保存模型
    save_model(model, X, importance_df)
    
    print("\n🎉 逻辑回归CTR模型训练完成!")
    print("📁 生成的文件:")
    print("   - ctr_lr_results.png: 可视化图表")
    print("   - ctr_lr_analysis_report.md: 分析报告")
    print("   - ctr_lr_model.pkl: 训练好的模型")
    
    print("\n💡 使用建议:")
    print("   1. 收集更多点击数据以提高模型效果")
    print("   2. 尝试不同的特征组合")
    print("   3. 定期重新训练模型")
    print("   4. 结合业务指标评估模型效果")

if __name__ == "__main__":
    main() 