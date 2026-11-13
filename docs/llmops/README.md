# 训练与反馈闭环

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**模型供应 → 训练与评测**，包含 LLMOps 闭环、CTR、表示学习和 CLIP 实验；数据集、模型产物与供应作业在同一入口的“模型与数据”中管理。

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| CPT | 领域文本 → 适配领域语料分布的模型 |
| SFT | 指令或对话数据 → 遵循指令的模型 |
| DPO | 同一问题的偏好回答对 → 偏好对齐模型 |

## 操作与结果核对

1. 核对输入模型、数据文件、数据集注册和输出目录。
2. 先预览配置与样本，确认阶段和格式，再启动训练。
3. 通过日志、进程和输出产物确认训练完成，然后单独加载模型。
4. 使用固定问题做推理对照，采集偏好；独立评估后再决定下一次训练。

## 当前能力与边界

当前指令生成为模拟数据流程，API 生成方法仍返回模拟样本。A/B 使用同一个已加载模型的不同温度生成回答，不是两个生产模型版本的流量实验。采集、训练、评估和发布尚未组成自动编排。

## 按问题阅读

- [数据来源与质量](lectures/data-generation/README.md)
- [CPT](lectures/cpt-training/README.md)
- [SFT](lectures/sft-training/README.md)
- [偏好采集](lectures/online-feedback/README.md)
- [DPO](lectures/dpo-training/README.md)
- [迭代流程](lectures/online-loop/README.md)
- [训练架构](lectures/architecture/README.md)

## 实现入口

- [src/search_engine/training_tab/llmops_tab.py](../../src/search_engine/training_tab/llmops_tab.py)
- [src/search_engine/training_tab/self_instruct_generator.py](../../src/search_engine/training_tab/self_instruct_generator.py)
- [src/search_engine/training_tab/preference_collector.py](../../src/search_engine/training_tab/preference_collector.py)
- [src/search_engine/training_tab/llamafactory_trainer.py](../../src/search_engine/training_tab/llamafactory_trainer.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
