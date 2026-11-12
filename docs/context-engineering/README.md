# 上下文工程与 MCP

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**上下文工程。**

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| 输入 | 用户意图、MCP 模板、工具定义、资源与历史 |
| 处理 | 模板选择 → 上下文装配 → 模型推理 → 上下文更新 |
| 输出 | 各阶段视图、装配后的上下文、模型结果、状态与历史 |

## 操作与结果核对

1. 通过统一脚本启动或复用 MCP 服务，核对模板、工具和资源实际可用。
2. 输入任务，执行流程并查看选中的模板。
3. 比较替换前模板与装配后的上下文，核对资源和工具占位符。
4. 检查推理结果与历史是否对应本次执行，再进行后续轮次。

## 当前能力与边界

本模块配置和验证原有上下文流程。平台定位不意味着已新增来源管理、Agent 绑定、权限治理或任意企业连接器。

## 按问题阅读

- [模板、资源与工具](../rag/lectures/mcp-architecture/README.md)
- [上下文构建](../rag/lectures/context-engineering/README.md)

## 实现入口

- [src/search_engine/mcp_tab/context_pipeline.py](../../src/search_engine/mcp_tab/context_pipeline.py)
- [src/search_engine/mcp_tab/smart_agent_demo.py](../../src/search_engine/mcp_tab/smart_agent_demo.py)
- [src/search_engine/mcp/dynamic_mcp_server.py](../../src/search_engine/mcp/dynamic_mcp_server.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
