# 验证与运行辅助工具

[项目说明](../README.md) · [运行与开发](../console/README.md)

工具帮助观察状态、诊断数据和验证调用。它们各有检查范围，不能将某个工具的“正常”状态解释为模型质量、全部依赖或业务任务已经通过验收。

## 主要工具

| 工具 | 用途 | 使用时核对 |
| --- | --- | --- |
| [verify_console_migration.py](verify_console_migration.py) | 对完整原运行时执行以读取和输入校验为主的回调验收 | Python 服务与图服务先就绪；会改变测试会话的部分状态 |
| [data_quality_checker.py](data_quality_checker.py) | 检查文档、索引和 CTR 数据 | 对应文件和运行数据是否就是本次任务的数据 |
| [performance_monitor.py](performance_monitor.py) | 资源观测、性能报告和测试工具 | 负载、数据量、环境与采样时间 |
| [sre_monitor.py](sre_monitor.py) | 系统观测、日志和告警逻辑 | 指标定义与观测窗口，不把示例报告当作生产 SLA |
| [model_service_manager.py](model_service_manager.py) | 可选独立 CTR API 的进程与健康管理 | 与工作台 8766 执行服务区分 |
| [reset_system.py](reset_system.py) | 状态、备份与重置辅助 | 重置分支会删除匹配的数据/模型文件，不属于只读检查 |

其他工具用于生成或导入实验数据。先查看相应脚本的参数、输入和输出路径，再决定是否执行；不要把初始化或重置放进日常健康检查。

## 基础检查

在根目录和已激活的工作台环境中运行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
npm --prefix console run build
```

回归测试覆盖原组件参数处理、状态、动态更新、选择事件、文件、流式响应、停止与 HTTP 传输。构建检查 TypeScript 和前端产物。

## 原回调验收

完整业务环境、图服务和 Python 执行服务就绪后：

```bash
python tools/verify_console_migration.py
```

脚本输出逐项结果并写入 JSON；参数与输出位置见脚本本身。它检查原操作的传输和返回，不执行真实训练或桌面任务，也不替代在线模型质量评估。模型卸载等状态操作仅影响相应测试运行状态，执行前应确认使用的是预期环境。

## 解读系统监控

工作台“系统状态”视图主要从数据与索引统计组织结果。绿色状态文案不是对所有模型、MCP、图像服务和桌面环境的独立健康检查，应到对应模块核对实际调用与结果。

对性能或效果给出结论时，同时记录代码版本、数据、模型、环境、参数和测量方法。无法提供这些条件的示例数字不作为本项目的验收依据。
