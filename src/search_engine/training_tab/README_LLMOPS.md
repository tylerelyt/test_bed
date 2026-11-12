# 训练模块实现导航

[当前使用指南](../../../docs/llmops/README.md) · [数据与反馈](../../../docs/llmops/lectures/data-generation/README.md)

安装、操作顺序和能力边界统一维护在模块指南。本页帮助定位原实现。

| 文件 | 职责 |
| --- | --- |
| [llmops_tab.py](llmops_tab.py) | 阶段页面、数据操作、推理与偏好事件绑定 |
| [llmops_engine.py](llmops_engine.py) | 组件管理、运行状态与界面更新 |
| [llmops_runner.py](llmops_runner.py) | 参数校验、配置整理、训练启动与停止 |
| [llamafactory_trainer.py](llamafactory_trainer.py) | 生成配置并管理真实训练子进程 |
| [domain_corpus_processor.py](domain_corpus_processor.py) | 领域文本清理、切分、保存与注册 |
| [self_instruct_generator.py](self_instruct_generator.py) | 当前模拟指令数据生成、格式转换与注册 |
| [preference_collector.py](preference_collector.py) | 偏好记录、持久化与训练格式导出 |
| [inference_model.py](inference_model.py) | 按需加载本地模型并生成回答 |

当前 A/B 使用同一模型的不同温度，不是独立模型版本的生产流量实验。训练、评估与部署也没有自动组成闭环；每一步以实际输入、日志和输出产物确认。
