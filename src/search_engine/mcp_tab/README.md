# 上下文工程模块实现导航

[当前使用指南](../../../docs/context-engineering/README.md) · [MCP 专题](../../../docs/rag/lectures/mcp-architecture/README.md)

本目录负责把用户意图、MCP 模板、工具与资源组织成可执行的上下文流程。操作步骤和能力边界统一维护在当前使用指南，本页只说明代码职责。

| 入口 | 职责 |
| --- | --- |
| [context_pipeline.py](context_pipeline.py) | PipelineState、占位符解析、四阶段执行与历史更新 |
| [smart_agent_demo.py](smart_agent_demo.py) | 原智能体演示与相关调用 |
| [MCP 客户端管理器](../mcp/mcp_client_manager.py) | 模板、工具与资源的协议交互 |
| [MCP 服务](../mcp/dynamic_mcp_server.py) | 当前暴露的模板、工具、资源与历史存储 |

占位符格式为 `${local:...}`、`${mcp:resource:...}` 和 `${mcp:tool:...}`，实际名称与资源 URI 以服务定义为准。

排查时依次核对模板发现、选择结果、替换后的上下文、模型响应及历史更新。状态缓存和调用次数优化不能只靠设计图推断效果，应保留同一任务下的实际请求与耗时记录。
