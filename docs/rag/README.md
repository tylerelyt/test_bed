# RAG 与图谱推理

[项目说明](../../README.md) · [文档导航](../README.md)

工作台入口：**RAG、OAG 图谱推理。**

## 输入与输出

| 项目 | 内容 |
| --- | --- |
| RAG 输入 | 问题、检索文档数量、检索增强开关与推理方式 |
| OAG 输入 | 问题、图谱游走深度、每跳关系扩展上限与推理方式 |
| 输出 | 回答、文档或图谱事实、上下文、实际提示词和步骤记录 |

## 操作与结果核对

1. 先在离线索引中确认问题所需的文档或实体关系存在。
2. 选择 DashScope API 或已加载的本地模型。
3. 文档问答进入 RAG；关系路径与多步图谱推理进入 OAG。
4. 核对回答依据、实际提示词和事实来源，不只看生成文本。

## 当前能力与边界

RAG 与 OAG 是两个独立入口。旧服务仍保留兼容方法，但页面不再用“开启多步推理”在三种模式间切换。Ollama 未连接也不自动表示当前 DashScope 或本地推理方式不可用。

## 图谱推理

OAG 以实体、关系与逐步图谱事实为依据，使用游走深度和每跳扩展上限约束查询范围。先确认图数据存在，再核对推理步骤与结果。

## 按问题阅读

- [当前架构](lectures/architecture/README.md)
- [文档问答](lectures/context-enhanced/README.md)
- [直接对话](lectures/direct-chat/README.md)
- [检索依据](lectures/retrieval/README.md)
- [多步推理](lectures/multi-step-reasoning/README.md)
- [ReAct 实现区分](lectures/react-reasoning/README.md)
- [Ollama 兼容路径](lectures/ollama-integration/README.md)
- [上下文构建](lectures/context-engineering/README.md)
- [MCP](lectures/mcp-architecture/README.md)

## 实现入口

- [src/search_engine/rag_tab/rag_tab.py](../../src/search_engine/rag_tab/rag_tab.py)
- [src/search_engine/rag_tab/rag_service.py](../../src/search_engine/rag_tab/rag_service.py)
- [src/search_engine/training_tab/inference_model.py](../../src/search_engine/training_tab/inference_model.py)

[原有代码与图解参考](implementation/README.md)保留示例；当前操作步骤统一以上述说明为准。
