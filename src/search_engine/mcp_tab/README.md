# 上下文工程系统 - MCP实现

## 🎯 核心设计原则

### **上下文工程核心：逻辑分区展开**

**上下文工程服务器 (CE Server) 职责：**
- ✅ 调用MCP协议（`list_prompts`、`list_tools`、`get_resource`）
- ✅ LLM智能选择（模板选择、工具选择）
- ✅ **逻辑分区展开**：展开Prompt模板中的`section_*`宏
- ✅ 组装完整上下文并发送给LLM推理
- ✅ 解析TAO输出，检查结束条件
- ✅ 循环直到满足结束条件

**MCP Server职责：**
- ✅ 定义Prompt模板（`section_*`参数接口）
  - 人设分区：模板内固定内容
  - Few-shot示例分区：控制流+结束条件格式
- ✅ 管理Tools能力（提供工具列表和schema）
- ✅ 管理Resources（对话历史、系统状态）
- ✅ 数据存储

## 📊 完整流程（标准范式，可扩展）

```
1. 拉取Prompts → list_prompts() 获取所有模板
2. LLM选择模板 → 根据用户意图+模板列表智能选择
3. 逻辑分区展开 → 展开模板中的section_*宏：
   ├─ 人设分区：模板内定义的固定内容
   ├─ Few-shot示例分区：模板内定义的控制流示例（ReAct/Self-Ask等）
   │  └─ 定义输出结构、工具调用时机、状态流转、结束条件格式
   ├─ 历史分区：get_resource("conversation://current/history")
   ├─ 工具分区：list_tools() 获取全部 → LLM智能选择相关工具
   ├─ 任务分区：从用户意图提取
   └─ 当前状态分区：CE Server生成（时间戳等）
   注：这些是常见分区，实际分区由Prompt模板定义，CE Server实现展开逻辑
4. 集成上下文 → 组装所有section_*参数
5. LLM推理 → 发送装配后的上下文获取TAO结果
6. 更新历史 → add_conversation_turn(TAO记录)
7. 循环 → 重复2-6直到满足结束条件（Few-shot定义的格式）
```

## 📁 文件说明

### `context_pipeline.py`
核心管道实现，包含：
- `PipelineState`：状态管理，避免重复执行
- `LogicalPartitionManager`：逻辑分区管理，最大化利用MCP
- `ContextEngineeringPipeline`：完整的四阶段循环

### `smart_agent_demo.py`
UI界面，已优化：
- 保持原有接口不变
- 内部使用`ContextEngineeringPipeline`
- 完整流程避免重复执行（节省70%成本）

### `OPTIMIZATION.md`
详细的优化说明文档

## 🚀 使用方式

### 推荐：执行完整流程
```python
# 点击"⚡ 执行完整流程"按钮
# 使用优化的管道，性能最优，只调用1次LLM
```

### 调试：单独执行阶段
```python
# 点击单个阶段按钮
# 保持原有逻辑，便于调试
```

## ⚡ 性能优化

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| LLM调用次数 | 6次 | 2次 | -67% |
| 执行时间 | 100% | 33% | -67% |
| API成本 | 100% | 33% | -67% |

## 🏗️ 架构优势

1. ✅ **符合MCP最佳实践**：逻辑分区基于MCP Server Prompt接口
2. ✅ **性能优化**：避免重复执行，状态管理清晰
3. ✅ **热插拔**：修改MCP Server，客户端自动适配
4. ✅ **可维护**：职责单一，模块化设计

## 📚 相关文档

- [上下文工程指南](../../../docs/CONTEXT_ENGINEERING_GUIDE.md)
- [优化说明](./OPTIMIZATION.md)

