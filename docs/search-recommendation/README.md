# 索引、检索与排序

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**离线索引构建、在线召回排序；CTR 训练与评估在“数据回收训练”。**

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| 输入 | 文档、查询、点击与未点击样本 |
| 操作 | 查看索引与文档，提交查询，选择排序算法，查看结果详情 |
| 输出 | 命中文档、排序分数、摘要、点击样本和训练结果 |

## 操作与结果核对

1. 在离线索引中核对文档列表与索引统计。
2. 用一个已知问题测试词法召回，再比较 CTR 重排。
3. 查看详情并返回，核对点击样本的查询、文档与 request_id。
4. 需要训练或评估时，进入数据回收训练模块并确认数据与模型。

## 当前能力与边界

词法召回和 CTR 重排已在原业务函数中实现。向量检索、混合召回和分布式服务不能仅凭讲义示例视为当前能力。分组效果与模型解释属于诊断工具。

## 样本与反馈

点击样本由数据服务保存到原位置，用于 CTR 训练与诊断。在线学习开关与新增样本阈值属于 CTR 流程，与大语言模型偏好训练分开管理。

## 按问题阅读

- [实现与评分](lectures/implementation/README.md)
- [CTR 特征与重排](lectures/ctr-prediction/README.md)
- [评估](lectures/model-evaluation/README.md)
- [解释](lectures/interpretability/README.md)
- [分组效果](lectures/fairness/README.md)
- [参数搜索](lectures/automl/README.md)
- [在线学习](lectures/online-learning/README.md)
- [架构](lectures/architecture/README.md)

## 实现入口

- [src/search_engine/index_tab/offline_index.py](../../src/search_engine/index_tab/offline_index.py)
- [src/search_engine/search_tab/search_tab.py](../../src/search_engine/search_tab/search_tab.py)
- [src/search_engine/data_service.py](../../src/search_engine/data_service.py)
- [src/search_engine/model_service.py](../../src/search_engine/model_service.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
