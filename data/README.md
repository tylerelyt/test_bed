# 数据来源与存放位置

[项目说明](../README.md) · [模块导航](../docs/README.md)

本页说明工作台实际使用的数据来源。文档数量、图片数量、样本量和点击率都以当前文件与运行统计为准，不固定写成某次实验的数字。

## 主要位置

| 位置 | 内容与读取方式 |
| --- | --- |
| `data/preloaded_documents.json` | 索引服务优先尝试加载的预置文档 |
| `models/index_data.json` | 默认倒排索引文件；初始化还会核对预置文档是否缺失 |
| `models/ctr_data.json` | 数据服务持久化的展示与点击样本 |
| `models/images/` | 图片、元数据与嵌入索引，由图片服务管理 |
| `data/llmops/cpt/` | 处理后的领域语料及相关数据 |
| `data/llmops/sft/` | 指令或对话数据；当前生成器输出模拟样本 |
| `data/llmops/dpo/` | 偏好记录 `prefs.jsonl` 与导出的训练数据 |
| `checkpoints/` | 原训练产物；具体输出以训练配置为准 |

MCP 历史位置由[原服务实现](../src/search_engine/mcp/dynamic_mcp_server.py)确定。图谱数据由所连接的 JanusGraph/Gremlin 服务保存，持久性取决于该服务配置。

## 导入和核对

1. 确认当前模块实际读取的文件与数据结构。
2. 使用模块已有的导入、处理或索引接口，查看返回状态与统计。
3. 核对样本或文档内容，再执行检索、训练等后续操作。
4. 保存数据来源、处理步骤与训练/评估划分，便于复现。

`documents.txt` 属于早期样例；单独修改它不会自动改变当前已加载的预置文档和索引。模型、图谱和图片索引也不能只通过改一个说明文件完成更新。

## 数据质量与结果解释

- 同一文档 ID、同一查询和同一点击请求的对应关系需要一致。
- 点击样本与偏好样本是不同数据类型，不能混作标签。
- 格式正确不代表内容正确；模拟样本只用于联通流程。
- 训练数据和独立评估数据应能区分，历史统计特征也需检查数据范围。

## 实现入口

- [索引加载](../src/search_engine/index_tab/index_service.py)
- [点击样本](../src/search_engine/data_service.py)
- [图片索引](../src/search_engine/image_service.py)
- [领域语料](../src/search_engine/training_tab/domain_corpus_processor.py)
- [指令生成](../src/search_engine/training_tab/self_instruct_generator.py)
- [偏好记录与导出](../src/search_engine/training_tab/preference_collector.py)
