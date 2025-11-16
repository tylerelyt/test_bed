# 🧠 上下文工程系统技术指南 ([返回README](../README.md))

## 1. 系统概述

上下文工程系统基于Model Context Protocol (MCP)架构，实现符号主义专家系统与连接主义LLM的完美结合。通过动态上下文编排、智能模板选择和工具调用，构建完整的思考-行动-观察循环。

## 2. 核心架构设计

### 2.1 上下文工程架构理念

**核心思想**：将LLM的上下文窗口视为可编程的逻辑空间，通过MCP协议实现符号主义与连接主义的完美结合。

```mermaid
graph TB
    subgraph "符号主义层 - Expert System Layer"
        A[用户意图] --> B[上下文编排引擎]
        B --> C[逻辑分区管理器]
        C --> D[动态资源调度器]
    end
    
    subgraph "连接主义层 - LLM Consumer Layer"
        D --> E[上下文包裹]
        E --> F[LLM推理引擎]
        F --> G[TAO输出解析]
    end
    
    subgraph "MCP协议层 - Protocol Layer"
        H[Prompts服务] --> C
        I[Tools服务] --> C
        J[Resources服务] --> C
    end
    
    subgraph "基础设施层 - Infrastructure Layer"
        K[IndexService] --> I
        L[对话历史] --> J
        M[配置文件] --> J
    end
```

### 2.2 四阶段智能体循环

系统实现了完整的四阶段智能体循环：

```mermaid
graph TB
    A[用户意图] --> B[阶段1: 模板选择]
    B --> C[阶段2: 上下文装配]
    C --> D[阶段3: LLM推理]
    D --> E[阶段4: 上下文更新]
    E --> F[下一轮对话]
```

### 2.3 逻辑分区动态展开流程

**上下文工程的核心**：先获取所有Prompt模板，LLM智能选择，然后替换模板中的占位符

**标准流程**（可扩展）：
1. **拉取全部Prompt模板** → `list_prompts()`
2. **LLM智能选择模板** → 根据用户意图+模板列表，LLM选择最合适的
3. **获取模板内容** → `get_prompt(template)` 返回包含占位符的模板
   - **固定分区**：人设、Few-shot示例直接在模板中定义
   - **占位符**：`${local:xxx}`, `${mcp:resource:xxx}`, `${mcp:tool:xxx}`
4. **占位符替换** → CE Server识别并替换占位符：
   - **本地占位符** `${local:xxx}`：
     - `${local:current_time}` → CE Server生成当前时间
     - `${local:user_intent}` → 用户输入
     - `${local:model_name}` → 模型名称
   - **Resource占位符** `${mcp:resource:xxx}`：
     - `${mcp:resource:conversation://current/history}` → 通过`get_resource()`获取对话历史
   - **Tool占位符** `${mcp:tool:xxx}`：
     - `${mcp:tool:dynamic_tool_selection}` → 特殊宏，调用`list_tools()`获取所有工具 + LLM智能选择相关工具
     - `${mcp:tool:具体工具名}` → 直接获取该工具的定义，无需LLM选择
   - **注：这些只是常见占位符，可以自定义任意占位符并实现对应的解析逻辑**
5. **完成替换** → 生成完整的上下文
6. **LLM推理** → 发送替换后的完整上下文给LLM获取TAO结果
7. **更新历史** → 将TAO记录更新到`MCP Resources`
8. **循环直到结束** → 重复2-7直到满足结束条件

**🔧 可扩展性说明**：
- 上述流程是**标准范式**，不是固定要求
- 占位符格式统一为 `${prefix:key}`，简洁优雅（类似bash变量）
- 可以自定义任意占位符（如 `${local:code_context}`, `${mcp:resource:system://status}`）
- CE Server只需在PlaceholderResolver中添加对应的解析逻辑即可
- 模板保持声明式，易读易维护

```mermaid
sequenceDiagram
    participant User as 用户
    participant CE as 上下文工程服务器
    participant Prompts as MCP Prompts
    participant Tools as MCP Tools
    participant Resources as MCP Resources
    participant LLM as LLM推理引擎
    
    User->>CE: 输入用户意图
    
    Note over CE,Prompts: 阶段1: 模板选择
    CE->>Prompts: list_prompts() 获取所有模板
    Prompts-->>CE: 返回模板列表和描述
    CE->>LLM: 发送用户意图+模板列表
    LLM-->>CE: 智能选择最合适的模板
    
    Note over CE,Resources: 阶段2: 占位符替换
    
    CE->>Prompts: get_prompt(selected_template)
    Prompts-->>CE: 返回包含占位符的模板
    Note over CE: 模板包含：<br/>- 固定分区（人设、Few-shot示例）<br/>- 占位符（${local:xxx}, ${mcp:xxx}）
    
    rect rgb(200, 220, 240)
        Note over CE: 识别本地占位符 ${local:xxx}
        CE->>CE: ${local:current_time} → 生成当前时间
        CE->>CE: ${local:user_intent} → 用户输入
        CE->>CE: ${local:model_name} → 模型名称
    end
    
    rect rgb(220, 240, 200)
        Note over CE,Resources: 识别Resource占位符 ${mcp:resource:xxx}
        CE->>Resources: get_resource("conversation://current/history")
        Resources-->>CE: 返回对话历史数据
        CE->>CE: 替换 ${mcp:resource:conversation://current/history}
    end
    
    rect rgb(240, 220, 200)
        Note over CE,Tools: 识别Tool占位符 ${mcp:tool:xxx}
        alt ${mcp:tool:dynamic_tool_selection}
            CE->>Tools: list_tools() 获取所有工具列表
            Tools-->>CE: 返回工具定义和schema
            CE->>LLM: 发送用户意图+工具列表
            LLM-->>CE: 智能选择相关工具
            CE->>CE: 替换为选中的工具列表
        else ${mcp:tool:具体工具名}
            CE->>Tools: 直接获取该工具定义
            Tools-->>CE: 返回工具schema
            CE->>CE: 替换为该工具定义
        end
    end
    
    CE->>CE: 完成所有占位符替换
    Note over CE: 生成完整上下文
    
    Note over CE,LLM: 阶段3: LLM推理
    CE->>LLM: 发送替换后的完整上下文
    LLM-->>CE: 返回TAO格式推理结果<br/>{reasoning, action, observation}
    
    Note over CE,Resources: 阶段4: 上下文更新
    CE->>CE: 解析TAO输出
    CE->>Resources: add_conversation_turn(TAO记录)
    Resources-->>CE: 更新历史确认
    
    alt 未达到结束条件
        CE->>CE: 进入下一轮循环
        Note over CE: 重复阶段2-4，直到满足结束条件
    else 达到结束条件
        CE->>User: 返回最终结果
    end
```

**📝 流程说明**：
- 上图展示的是**占位符替换的标准流程**
- 占位符格式：`${prefix:key}`（类似bash变量，简洁优雅）
- 三种占位符类型：
  - `${local:xxx}` - CE Server本地生成
  - `${mcp:resource:xxx}` - 调用MCP Resource获取
  - `${mcp:tool:xxx}` - 调用MCP Tools获取
- 可扩展：自定义新占位符只需在PlaceholderResolver中添加解析逻辑
- 这是一个**开放可扩展的架构**，不局限于图中所示的分区

### 2.4 MCP架构组件

- **上下文工程服务器**：`ContextEngineeringPipeline` - 负责编排上下文，执行四阶段循环
- **MCP客户端管理器**：`MCPClientManager` - 管理与MCP服务器的连接和调用
- **动态MCP服务器**：`DynamicMCPServer` - 提供标准化的prompts、tools、resources
- **智能体演示界面**：`smart_agent_demo.py` - 用户交互界面

## 3. 逻辑分区动态展开机制

### 3.1 逻辑分区设计原理

**上下文工程的核心思想**：将复杂的上下文分解为独立的逻辑分区，每个分区通过MCP协议动态获取和装配。

**🔧 设计特点**：
- **标准范式，非固定要求**：图示的五个分区是常见模式
  - **人设分区**：定义AI角色和能力边界
  - **Few-shot示例分区**：控制模型执行范式（ReAct、Self-Ask等）
    - 定义如何组织输出、何时调用工具、如何流转状态
    - **定义结束条件格式**：让模型知道何时任务完成并结束循环
  - **历史分区**：提供对话连续性
  - **工具分区**：提供外部能力
  - **任务分区**：明确当前目标
- **完全可扩展**：可以定义任意的`section_*`参数作为新的逻辑分区
- **职责分离**：
  - MCP Server的Prompt模板定义需要哪些逻辑分区（`section_*`参数）
  - CE Server实现每个分区的展开逻辑（如何生成分区内容）
  - 两者通过MCP协议标准化交互
- **热插拔**：新增分区无需修改框架，只需：
  1. 在Prompt模板中声明新的`section_*`参数
  2. 在CE Server中实现对应的生成方法

**扩展示例**：
```python
# 示例1：在MCP Server中定义新的代码审查模板
@self.mcp.prompt("advanced_code_review")
def advanced_code_review_prompt(
    section_persona: str = "",           # 标准分区
    section_code_context: str = "",      # 新增：代码上下文分区
    section_review_rules: str = "",      # 新增：审查规则分区
    section_similar_issues: str = "",    # 新增：相似问题分区
    section_user_question: str = ""      # 标准分区
) -> str:
    return f"""{section_persona}
{section_code_context}
{section_review_rules}
{section_similar_issues}
{section_user_question}"""

# 示例2：在CE Server中实现新分区的展开逻辑
class LogicalPartitionManager:
    async def generate_section_code_context(self, user_intent: str) -> str:
        """新增：生成代码上下文分区"""
        # 可以通过MCP Resource获取代码库信息
        code_info = await self.mcp_manager.get_resource("code://repository/context")
        return f"[代码上下文] {code_info}"
    
    async def generate_section_review_rules(self, template_name: str) -> str:
        """新增：生成审查规则分区"""
        # 可以通过MCP Resource获取审查规则
        rules = await self.mcp_manager.get_resource("rules://code_review")
        return f"[审查规则] {rules}"
    
    async def generate_section_similar_issues(self, user_intent: str) -> str:
        """新增：生成相似问题分区"""
        # 可以通过MCP Tools调用搜索工具
        similar = await self.mcp_manager.call_tool("search_similar_issues", {
            "query": user_intent
        })
        return f"[相似问题] {similar}"
```

**关键点**：
- ✅ 框架完全不需要修改
- ✅ 只需在两端各自实现新的分区
- ✅ 通过MCP协议保持标准化交互

```mermaid
graph LR
    subgraph "逻辑分区架构"
        A[人设分区] --> E[上下文包裹]
        B[历史分区] --> E
        C[工具分区] --> E
        D[任务分区] --> E
    end
    
    subgraph "MCP服务层"
        F[Prompts服务] --> A
        G[Resources服务] --> B
        H[Tools服务] --> C
        I[动态装配器] --> D
    end
    
    subgraph "数据源层"
        J[模板库] --> F
        K[对话历史] --> G
        L[工具注册表] --> H
        M[用户意图] --> I
    end
```

### 3.2 各逻辑分区的动态展开

#### 3.2.1 人设分区（Persona Partition）

**目标**：动态确定AI助手的角色和风格

**MCP实现**：
```python
# 通过Prompts服务动态获取人设模板
@self.mcp.prompt("simple_chat")
def simple_chat_prompt(user_input: str) -> str:
    return f"""[人设] 你是一个友好的AI助手

[当前状态] 处理时间: {{local:current_time}}
用户意图: {{local:user_intent}}
模型: {{local:model_name}}

[用户问题] {user_input}

[回答] 请保持简洁明了地回答用户问题："""

@self.mcp.prompt("context_engineering")
def context_engineering_prompt(user_input: str = "") -> str:
    return f"""[人设] 你是一个专业的上下文工程专家，擅长动态决策和智能推理

[上下文工程模式] 请严格按照以下格式进行回答：
思考: <详细分析用户问题，评估是否需要外部信息，制定解决方案>
行动: <选择适合的工具，格式：工具名(参数1="值1", 参数2="值2")>
观察: <工具返回的结果或观察到的信息>
"""
```

**动态展开过程**：
1. **模板发现**：通过`mcp_manager.list_prompts()`获取所有可用模板
2. **智能选择**：LLM根据用户意图选择最合适的人设模板
3. **动态装配**：将选择的人设模板与其他分区组合

#### 3.2.2 历史分区（History Partition）

**目标**：提供对话连续性和上下文记忆

**MCP实现**：
```python
@self.mcp.resource(
    uri="conversation://current/history",
    name="当前对话历史",
    description="实时对话历史记录，支持多轮对话上下文管理",
    mime_type="application/json"
)
def get_conversation_history() -> str:
    """获取对话历史资源"""
    return json.dumps(self.conversation_history, ensure_ascii=False, indent=2)

@self.mcp.tool("add_conversation_turn")
def add_conversation_turn(tao_data: str) -> str:
    """添加对话轮次工具"""
    tao_record = json.loads(tao_data)
    simplified_tao = {
        "turn": len(self.conversation_history) + 1,
        "timestamp": tao_record.get("timestamp", "now"),
        "reasoning": tao_record.get("reasoning", ""),
        "action": tao_record.get("action", ""),
        "observation": tao_record.get("observation", "")
    }
    self.conversation_history.append(simplified_tao)
    return json.dumps({"status": "success", "message": f"已添加第{simplified_tao['turn']}轮对话"})
```

**动态展开过程**：
1. **资源订阅**：通过MCP协议订阅对话历史资源
2. **实时获取**：动态获取最新的对话历史数据
3. **结构化存储**：以TAO格式存储每轮对话的思考-行动-观察过程

#### 3.2.3 工具分区（Tools Partition）

**目标**：提供AI可调用的外部能力

**MCP实现**：
```python
@self.mcp.tool(
    name="retrieve",
    description="智能文档检索工具，支持动态决策和思考-行动-观察模式",
    tags={"search", "retrieval", "document", "intelligent"},
    meta={"version": "2.0", "category": "core", "context_engineering": True}
)
def retrieve(
    reasoning: str = "",
    action: str = "search",
    query: str = "", 
    top_k: int = 5, 
    include_metadata: bool = True
) -> Dict[str, Any]:
    """智能文档检索工具
    
    支持思考-行动-观察模式的智能检索工具。模型可以：
    1. 提供推理过程(reasoning)
    2. 决定是否执行检索(action: "search" | "skip")
    3. 指定检索查询(query)
    """
    try:
        # 记录思考-行动-观察过程
        observation = {
            "reasoning": reasoning,
            "action": action,
            "query": query,
            "timestamp": "now",
            "tool": "retrieve"
        }
        
        if action.lower() == "skip":
            observation["result"] = "检索已跳过"
            observation["documents"] = []
            observation["total_found"] = 0
            return {
                "status": "skipped",
                "observation": observation,
                "message": "模型决定跳过检索"
            }
        
        # 执行检索
        results = self.index_service.search(query, top_k)
        documents = []
        
        for doc in results.get("documents", []):
            doc_info = {
                "content": doc.get("content", ""),
                "score": doc.get("score", 0.0)
            }
            if include_metadata:
                doc_info["metadata"] = doc.get("metadata", {})
            documents.append(doc_info)
        
        observation["result"] = "检索完成"
        observation["documents"] = documents
        observation["total_found"] = len(documents)
        
        return {
            "status": "success",
            "observation": observation,
            "query": query,
            "documents": documents,
            "total_found": len(documents),
            "source": "dynamic_mcp_server",
            "search_metadata": {
                "query_time": "real_time",
                "index_size": results.get("total_documents", 0)
            }
        }
    except Exception as e:
        observation = {
            "reasoning": reasoning,
            "action": action,
            "query": query,
            "result": f"检索失败: {str(e)}",
            "timestamp": "now",
            "tool": "retrieve"
        }
        return {
            "status": "error",
            "observation": observation,
            "error": str(e),
            "query": query,
            "documents": [],
            "total_found": 0
        }
```

**动态展开过程**：
1. **工具发现**：通过`mcp_manager.list_tools()`获取所有可用工具
2. **契约验证**：验证工具的输入输出schema
3. **动态调用**：根据LLM的决策动态调用相应工具
4. **结果观察**：将工具执行结果作为观察反馈给LLM

#### 3.2.4 任务分区（Task Partition）

**目标**：明确当前任务目标和执行策略

**动态展开过程**：
1. **意图解析**：分析用户输入，确定任务类型
2. **策略选择**：根据任务类型选择执行策略
3. **目标设定**：明确任务目标和成功标准

### 3.3 上下文装配的完整流程

```mermaid
graph TD
    A[用户意图输入] --> B[阶段1: 模板选择]
    B --> C[获取可用模板列表]
    C --> D[LLM智能选择]
    D --> E[确定人设分区]
    
    E --> F[阶段2: 上下文装配]
    F --> G[获取历史分区]
    G --> H[获取工具分区]
    H --> I[构建任务分区]
    
    I --> J[占位符替换]
    J --> K[逻辑分区组装]
    K --> L[生成上下文包裹]
    
    L --> M[阶段3: LLM推理]
    M --> N[TAO格式输出]
    N --> O[阶段4: 上下文更新]
    O --> P[更新历史分区]
    P --> Q[准备下一轮对话]
```

## 4. 核心实现

### 4.1 阶段1：模板选择（Template Selection）

**目标**：根据用户意图智能选择最合适的提示词模板

**实现逻辑**：
```python
def execute_stage_1_template_selection(user_intent: str) -> str:
    """执行阶段1: 模板选择"""
    # 1. 获取所有可用模板
    prompts = await mcp_manager.list_prompts()
    
    # 2. 构建选择提示词
    selection_prompt = f"""你是一个智能模板选择专家。请根据用户意图，从以下可用模板中选择最合适的模板：
    
    **用户意图**: {user_intent}
    **可用模板**: {prompt_descriptions}
    
    请输出JSON格式的选择结果："""
    
    # 3. LLM智能选择
    response = await llm_client.chat(selection_prompt)
    selection_result = json.loads(response)
    
    return selection_result["selected_template"]
```

**可用模板**：
- `simple_chat`：通用聊天模板
- `rag_answer`：基于检索的上下文回答
- `react_reasoning`：ReAct多步推理
- `code_review`：代码审查专用
- `financial_analysis`：财务分析专用
- `context_engineering`：上下文工程专用

### 4.2 阶段2：上下文装配（Context Assembly）

**目标**：将选择的模板与用户意图结合，生成完整的上下文

**实现逻辑**：
```python
def execute_stage_2_context_assembly(user_intent: str) -> str:
    """执行阶段2: 上下文装配"""
    # 1. 获取选择的模板
    selected_template = get_selected_template()
    
    # 2. 从MCP服务器获取模板内容
    template_content = mcp_manager.get_prompt("unified_server", selected_template, {
        "user_input": user_intent
    })
    
    # 3. 占位符替换
    resolved_content = template_content
    replacements = {
        "{local:current_time}": time.strftime("%Y-%m-%d %H:%M:%S"),
        "{local:user_intent}": user_intent,
        "{local:model_name}": "qwen-max",
        "{mcp:resource:conversation://current/history}": get_conversation_history(),
        "{mcp:tool:dynamic_tool_selection}": get_available_tools()
    }
    
    # 4. 应用替换
    for placeholder, value in replacements.items():
        resolved_content = resolved_content.replace(placeholder, value)
    
    return resolved_content
```

### 4.3 阶段3：LLM推理（LLM Inference）

**目标**：使用装配的上下文调用LLM进行推理

**实现逻辑**：
```python
def execute_stage_3_llm_inference(user_intent: str) -> str:
    """执行阶段3: LLM推理"""
    # 1. 获取装配后的上下文
    context = get_assembled_context()
    
    # 2. 调用LLM
    response = await llm_client.chat(
        model="qwen-max",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": user_intent}
        ]
    )
    
    return response
```

**TAO输出格式**：
```
**Reasoning（思考）**: 用户询问的是MCP的含义。MCP可能指代多个不同的概念，包括但不限于：多芯片封装(Multi-Chip Package)、内存控制处理器(Memory Control Processor)等。为了给出最准确的答案，我需要先确认用户具体指的是哪一个领域或上下文中的MCP。

**Action（行动）**: final_answer: MCP可以代表很多不同的意思，例如在计算机科学中它可以指"多芯片封装"(Multi-Chip Package) 或者是"内存控制处理器"(Memory Control Processor)。请问您是在哪个特定背景下提到MCP的呢？这样我可以提供更精确的信息给您。
```

### 4.4 阶段4：上下文更新（Context Update）

**目标**：更新对话历史，为下一轮对话做准备

**实现逻辑**：
```python
def execute_stage_4_context_update(user_intent: str) -> str:
    """执行阶段4: 上下文更新"""
    # 1. 解析LLM的TAO输出
    reasoning, action, observation = parse_tao_output(llm_response)
    
    # 2. 创建TAO记录
    tao_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reasoning": reasoning,
        "action": action,
        "observation": observation
    }
    
    # 3. 使用工具添加对话轮次
    await mcp_manager.add_conversation_turn(json.dumps(tao_record))
    
    return "上下文更新完成"
```

## 5. MCP服务实现

### 5.1 MCP客户端管理器

```python
class MCPClientManager:
    """MCP客户端管理器 - 正确的FastMCP实现"""
    
    def __init__(self):
        """初始化MCP客户端管理器"""
        self.clients: Dict[str, FastMCP] = {}
        self.server_configs = {
            "unified_server": {
                "url": "http://localhost:3001/mcp",
                "name": "unified-mcp-server",
                "description": "统一MCP服务器"
            }
        }
        self._connection_status = {}
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有提示词"""
        client = self.get_client("unified_server")
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            prompts_dict = await client.get_prompts()
            # FastMCP返回的是dict[str, Prompt]格式，需要转换为列表
            prompts_list = []
            for name, prompt in prompts_dict.items():
                prompts_list.append({
                    "name": name,
                    "description": getattr(prompt, 'description', f'提示词: {name}'),
                    "prompt": prompt
                })
            return prompts_list
        except Exception as e:
            raise Exception(f"获取提示词列表失败: {e}")
    
    def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any] = None) -> str:
        """获取MCP提示词模板 - 使用正确的FastMCP API"""
        try:
            client = self.get_client(server_name)
            
            if client:
                # 使用asyncio运行异步调用
                import asyncio
                
                # 检查当前是否有运行的事件循环
                try:
                    loop = asyncio.get_running_loop()
                    # 如果有运行的事件循环，使用run_in_executor
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._get_prompt_sync, client, prompt_name, arguments)
                        return future.result()
                except RuntimeError:
                    # 没有运行的事件循环，创建新的
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        return loop.run_until_complete(self._get_prompt_async(client, prompt_name, arguments))
                    finally:
                        loop.close()
            else:
                return f"服务器 {server_name} 未连接"
        except Exception as e:
            return f"提示词获取失败: {str(e)}"
    
    async def add_conversation_turn(self, tao_data: str) -> str:
        """添加对话轮次 - 使用工具调用"""
        client = self.get_client("unified_server")
        
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            # 使用工具调用添加对话轮次
            result = await self._run_in_fastmcp_context(client, "add_conversation_turn", {"tao_data": tao_data})
            return result
        except Exception as e:
            raise Exception(f"添加对话轮次失败: {e}")
```

### 5.2 动态MCP服务器

```python
class DynamicMCPServer:
    """动态MCP服务器 - 完全隔离解耦"""
    
    def __init__(self, server_name: str = "dynamic-mcp-server"):
        """初始化动态MCP服务器"""
        self.mcp = FastMCP(server_name)
        self.index_service = get_index_service()
        self.conversation_history = []
        
        # 注册所有功能
        self._register_prompts()
        self._register_tools()
        self._register_resources()
    
    def _register_prompts(self):
        """注册提示词 - 通过MCP协议动态发现"""
        
        @self.mcp.prompt("simple_chat")
        def simple_chat_prompt(user_input: str) -> str:
            """简单对话提示词 - 支持上下文工程"""
            return f"""[人设] 你是一个友好的AI助手

[当前状态] 处理时间: {{local:current_time}}
用户意图: {{local:user_intent}}
模型: {{local:model_name}}

[历史] {{mcp:resource:conversation://current/history}}

[用户问题] {user_input}

[回答] 请保持简洁明了地回答用户问题："""
        
        @self.mcp.prompt("context_engineering")
        def context_engineering_prompt(user_input: str = "") -> str:
            """上下文工程专用提示词 - 完整的思考-行动-观察模式"""
            return f"""[人设] 你是一个专业的上下文工程专家，擅长动态决策和智能推理

[当前状态] 处理时间: {{local:current_time}}
用户意图: {{local:user_intent}}
模型: {{local:model_name}}

[历史] {{mcp:resource:conversation://current/history}}

[可用工具] {{mcp:tool:dynamic_tool_selection}}

[用户问题] {user_input}

[上下文工程模式] 请严格按照以下格式进行回答：

思考: <详细分析用户问题，评估是否需要外部信息，制定解决方案>
行动: <选择适合的工具，格式：工具名(参数1="值1", 参数2="值2")>
观察: <工具返回的结果或观察到的信息>

如果需要多步推理，请重复上述格式。

[最终答案] 基于所有思考、行动和观察，给出完整的答案："""
    
    def _register_tools(self):
        """注册工具 - 遵循FastMCP最佳实践"""
        
        @self.mcp.tool(
            name="retrieve",
            description="智能文档检索工具，支持动态决策和思考-行动-观察模式",
            tags={"search", "retrieval", "document", "intelligent"},
            meta={"version": "2.0", "category": "core", "context_engineering": True}
        )
        def retrieve(
            reasoning: str = "",
            action: str = "search",
            query: str = "", 
            top_k: int = 5, 
            include_metadata: bool = True
        ) -> Dict[str, Any]:
            """智能文档检索工具
            
            支持思考-行动-观察模式的智能检索工具。模型可以：
            1. 提供推理过程(reasoning)
            2. 决定是否执行检索(action: "search" | "skip")
            3. 指定检索查询(query)
            """
            try:
                # 记录思考-行动-观察过程
                observation = {
                    "reasoning": reasoning,
                    "action": action,
                    "query": query,
                    "timestamp": "now",
                    "tool": "retrieve"
                }
                
                if action.lower() == "skip":
                    observation["result"] = "检索已跳过"
                    observation["documents"] = []
                    observation["total_found"] = 0
                    return {
                        "status": "skipped",
                        "observation": observation,
                        "message": "模型决定跳过检索"
                    }
                
                # 执行检索
                results = self.index_service.search(query, top_k)
                documents = []
                
                for doc in results.get("documents", []):
                    doc_info = {
                        "content": doc.get("content", ""),
                        "score": doc.get("score", 0.0)
                    }
                    if include_metadata:
                        doc_info["metadata"] = doc.get("metadata", {})
                    documents.append(doc_info)
                
                observation["result"] = "检索完成"
                observation["documents"] = documents
                observation["total_found"] = len(documents)
                
                return {
                    "status": "success",
                    "observation": observation,
                    "query": query,
                    "documents": documents,
                    "total_found": len(documents),
                    "source": "dynamic_mcp_server",
                    "search_metadata": {
                        "query_time": "real_time",
                        "index_size": results.get("total_documents", 0)
                    }
                }
            except Exception as e:
                observation = {
                    "reasoning": reasoning,
                    "action": action,
                    "query": query,
                    "result": f"检索失败: {str(e)}",
                    "timestamp": "now",
                    "tool": "retrieve"
                }
                return {
                    "status": "error",
                    "observation": observation,
                    "error": str(e),
                    "query": query,
                    "documents": [],
                    "total_found": 0
                }
    
    def _register_resources(self):
        """注册资源 - 遵循FastMCP最佳实践"""
        
        @self.mcp.resource(
            uri="conversation://current/history",
            name="当前对话历史",
            description="实时对话历史记录，支持多轮对话上下文管理",
            mime_type="application/json"
        )
        def get_conversation_history() -> str:
            """获取对话历史资源
            
            返回当前会话的完整对话历史，包括用户输入和AI回复。
            支持多轮对话的上下文管理，为LLM提供对话连续性。
            """
            try:
                return json.dumps(self.conversation_history, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({
                    "error": str(e),
                    "turns": [],
                    "timestamp": "now"
                }, ensure_ascii=False, indent=2)
        
        @self.mcp.tool("add_conversation_turn")
        def add_conversation_turn(tao_data: str) -> str:
            """添加对话轮次工具
            
            Args:
                tao_data: JSON格式的TAO数据，包含reasoning、action、observation
            """
            try:
                tao_record = json.loads(tao_data)
                # 只保存TAO结构
                simplified_tao = {
                    "turn": len(self.conversation_history) + 1,
                    "timestamp": tao_record.get("timestamp", "now"),
                    "reasoning": tao_record.get("reasoning", ""),
                    "action": tao_record.get("action", ""),
                    "observation": tao_record.get("observation", "")
                }
                
                self.conversation_history.append(simplified_tao)
                
                return json.dumps({
                    "status": "success",
                    "message": f"已添加第{simplified_tao['turn']}轮对话",
                    "total_turns": len(self.conversation_history)
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                }, ensure_ascii=False, indent=2)
    
    async def start_server(self, host: str = "localhost", port: int = 3001):
        """启动服务器"""
        await self.mcp.run_http_async(host=host, port=port)
```

## 6. 使用方法

### 6.1 启动系统

```bash
# 启动MCP服务器
python src/search_engine/mcp/dynamic_mcp_server.py

# 启动主系统
python start_system.py
```

### 6.2 访问界面

1. 打开浏览器访问：`http://localhost:7861`
2. 导航到"🧠 第四部分：上下文工程"标签页
3. 输入用户意图，系统将自动执行四阶段循环

### 6.3 功能验证

**测试用例1：简单问答**
- 输入："什么是MCP？"
- 预期：系统选择`simple_chat`模板，直接回答

**测试用例2：专业分析**
- 输入："分析这个项目的架构设计"
- 预期：系统选择`context_engineering`模板，进行结构化分析

**测试用例3：工具调用**
- 输入："搜索关于机器学习的文档"
- 预期：系统调用`retrieve`工具，返回相关文档

## 7. 技术特性

### 7.1 核心功能

1. **动态模板选择**：LLM驱动的智能模板选择
2. **上下文装配**：实时占位符替换和MCP资源集成
3. **TAO推理模式**：完整的思考-行动-观察循环
4. **对话历史管理**：基于MCP资源的持久化对话状态

### 7.2 架构优势

1. **模块解耦**：MCP协议实现组件间标准化交互
2. **可扩展性**：新增工具/资源/模板无需修改核心逻辑
3. **可观测性**：每个阶段都有明确的输入输出，便于调试
4. **生态兼容**：支持所有MCP兼容的LLM客户端

### 7.3 逻辑分区优势

1. **动态性**：每个逻辑分区都可以独立更新和扩展
2. **可组合性**：不同的逻辑分区可以灵活组合
3. **可观测性**：每个分区的状态都可以独立监控
4. **可维护性**：分区化的设计便于独立维护和测试

## 8. 真实示例：完整上下文工程流程

### 8.1 示例场景

**用户输入**："请帮我搜索关于MCP架构的文档，并分析其核心优势"

### 8.2 完整流程演示

#### 阶段1：模板选择

**用户意图分析**：
```
用户意图: "请帮我搜索关于MCP架构的文档，并分析其核心优势"
任务类型: 检索 + 分析
复杂度: 中等（需要工具调用 + 推理分析）
```

**LLM模板选择过程**：
```json
{
    "selected_template": "context_engineering",
    "reasoning": "用户需要搜索文档并进行专业分析，context_engineering模板支持完整的思考-行动-观察模式，能够处理复杂的多步推理任务",
    "confidence": 0.95
}
```

#### 阶段2：上下文装配

**获取选择的模板**：
```python
# 从MCP服务器获取context_engineering模板
template_content = mcp_manager.get_prompt("unified_server", "context_engineering", {
    "user_input": "请帮我搜索关于MCP架构的文档，并分析其核心优势"
})
```

**模板内容**：
```markdown
[人设] 你是一个专业的上下文工程专家，擅长动态决策和智能推理

[当前状态] 处理时间: {local:current_time}
用户意图: {local:user_intent}
模型: {local:model_name}

[历史] {mcp:resource:conversation://current/history}

[可用工具] {mcp:tool:dynamic_tool_selection}

[用户问题] 请帮我搜索关于MCP架构的文档，并分析其核心优势

[上下文工程模式] 请严格按照以下格式进行回答：

思考: <详细分析用户问题，评估是否需要外部信息，制定解决方案>
行动: <选择适合的工具，格式：工具名(参数1="值1", 参数2="值2")>
观察: <工具返回的结果或观察到的信息>

如果需要多步推理，请重复上述格式。

[最终答案] 基于所有思考、行动和观察，给出完整的答案：
```

**占位符替换过程**：
```python
# 从FastMCP获取工具信息
tools = await mcp_manager.list_tools()
tool_info = tools[0]  # retrieve工具

replacements = {
    "{local:current_time}": "2025-08-31 23:45:30",
    "{local:user_intent}": "请帮我搜索关于MCP架构的文档，并分析其核心优势",
    "{local:model_name}": "qwen-max",
    "{mcp:resource:conversation://current/history}": "[]",  # 首次对话，历史为空
    "{mcp:tool:dynamic_tool_selection}": f"- {tool_info['name']}: {tool_info['description']}\n  输入参数: {list(tool_info['inputSchema']['properties'].keys())}\n  输出格式: {list(tool_info['outputSchema']['properties'].keys())}"
}
```

**装配后的完整上下文**：
```markdown
[人设] 你是一个专业的上下文工程专家，擅长动态决策和智能推理

[当前状态] 处理时间: 2025-08-31 23:45:30
用户意图: 请帮我搜索关于MCP架构的文档，并分析其核心优势
模型: qwen-max

[历史] []

[可用工具] - retrieve: 智能文档检索工具，支持动态决策和思考-行动-观察模式
  输入参数: ['reasoning', 'action', 'query', 'top_k', 'include_metadata']
  输出格式: ['status', 'observation', 'query', 'documents', 'total_found']

[用户问题] 请帮我搜索关于MCP架构的文档，并分析其核心优势

[上下文工程模式] 请严格按照以下格式进行回答：

思考: <详细分析用户问题，评估是否需要外部信息，制定解决方案>
行动: <选择适合的工具，格式：工具名(参数1="值1", 参数2="值2")>
观察: <工具返回的结果或观察到的信息>

如果需要多步推理，请重复上述格式。

[最终答案] 基于所有思考、行动和观察，给出完整的答案：
```

#### 阶段3：LLM推理

**LLM输入**：
```python
messages = [
    {
        "role": "system", 
        "content": "装配后的完整上下文（如上所示）"
    },
    {
        "role": "user", 
        "content": "请帮我搜索关于MCP架构的文档，并分析其核心优势"
    }
]
```

**LLM TAO输出**：
```markdown
**Reasoning（思考）**: 用户需要搜索关于MCP架构的文档并分析其核心优势。这是一个需要外部信息支持的分析任务。我需要先使用retrieve工具搜索相关文档，然后基于搜索结果进行分析。

**Action（行动）**: retrieve(reasoning="需要搜索MCP架构相关文档以支持分析", action="search", query="MCP架构 Model Context Protocol", top_k=5, include_metadata=True)

**Observation（观察）**: 检索到3个相关文档：
1. "MCP架构设计原理" - 相关性得分: 0.89
2. "Model Context Protocol技术指南" - 相关性得分: 0.85  
3. "MCP在智能体系统中的应用" - 相关性得分: 0.82

**Reasoning（思考）**: 基于检索到的文档，我可以分析MCP架构的核心优势。从文档内容来看，MCP架构具有标准化、模块化、可扩展性等优势。


```

#### 阶段4：上下文更新

**解析TAO输出**：
```python
tao_record = {
    "timestamp": "2025-08-31 23:45:30",
    "reasoning": "用户需要搜索关于MCP架构的文档并分析其核心优势。这是一个需要外部信息支持的分析任务。我需要先使用retrieve工具搜索相关文档，然后基于搜索结果进行分析。",
    "action": "retrieve(reasoning=\"需要搜索MCP架构相关文档以支持分析\", action=\"search\", query=\"MCP架构 Model Context Protocol\", top_k=5, include_metadata=True)",
    "observation": "检索到3个相关文档，并基于文档内容分析了MCP架构的5个核心优势"
}
```

**更新对话历史**：
```python
# 调用MCP工具添加对话轮次
await mcp_manager.add_conversation_turn(json.dumps(tao_record))
```

**更新后的历史资源**：
```json
[
  {
    "turn": 1,
    "timestamp": "2025-08-31 23:45:30",
    "reasoning": "用户需要搜索关于MCP架构的文档并分析其核心优势。这是一个需要外部信息支持的分析任务。",
    "action": "retrieve(reasoning=\"需要搜索MCP架构相关文档以支持分析\", action=\"search\", query=\"MCP架构 Model Context Protocol\", top_k=5, include_metadata=True)",
    "observation": "检索到3个相关文档，并基于文档内容分析了MCP架构的5个核心优势"
  }
]
```

**LLM TAO输出**：
```markdown
**Action（行动）**: final_answer: 基于检索到的文档，MCP架构具有以下核心优势：

1. **标准化协议**: MCP提供了统一的接口标准，使得不同组件可以无缝集成
2. **模块化设计**: 将复杂的智能体系统分解为独立的prompts、tools、resources模块
3. **可扩展性**: 支持动态添加新的工具和资源，无需修改核心架构
4. **符号主义与连接主义结合**: 通过专家系统管理上下文，LLM专注推理生成
5. **生态兼容性**: 支持所有MCP兼容的客户端和服务

MCP架构通过将上下文工程标准化，实现了智能体系统的模块化和可维护性，是现代AI系统的重要技术基础。
```

### 8.3 逻辑分区动态展开分析

#### 人设分区展开
- **模板选择**: LLM选择了`context_engineering`模板
- **角色设定**: "专业的上下文工程专家，擅长动态决策和智能推理"
- **输出格式**: 严格的TAO格式要求

#### 历史分区展开
- **初始状态**: 空历史`[]`
- **更新后**: 包含第一轮对话的TAO记录
- **持久化**: 通过MCP资源服务存储

#### 工具分区展开
- **工具发现**: 通过`mcp_manager.list_tools()`发现`retrieve`工具
- **工具契约**: 获取完整的输入输出schema
- **工具调用**: 执行文档检索
- **结果观察**: 获取3个相关文档

**FastMCP工具契约示例**：
```json
{
    "name": "retrieve",
    "description": "智能文档检索工具，支持动态决策和思考-行动-观察模式",
    "tags": ["search", "retrieval", "document", "intelligent"],
    "meta": {
        "version": "2.0",
        "category": "core",
        "context_engineering": true
    },
    "inputSchema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "模型的推理过程"},
            "action": {"type": "string", "enum": ["search", "skip"], "description": "行动决策"},
            "query": {"type": "string", "description": "搜索查询字符串"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "include_metadata": {"type": "boolean", "default": true}
        },
        "required": ["query"]
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error", "skipped"]},
            "observation": {"type": "object"},
            "query": {"type": "string"},
            "documents": {"type": "array"},
            "total_found": {"type": "integer"}
        }
    }
}
```

#### 任务分区展开
- **任务类型**: 检索+分析任务
- **执行策略**: 先检索后分析
- **成功标准**: 提供MCP架构的核心优势分析

### 8.4 技术亮点展示

#### 1. 动态模板选择
- LLM根据任务复杂度智能选择`context_engineering`模板
- 避免了简单问答模板的局限性

#### 2. 工具调用集成
- 通过MCP协议无缝调用`retrieve`工具
- 支持思考-行动-观察的完整循环

#### 3. 上下文记忆
- 对话历史以TAO格式结构化存储
- 为后续对话提供上下文连续性

#### 4. 模块化架构
- 每个逻辑分区独立运行
- 通过MCP协议标准化交互

### 8.5 示例总结

这个真实示例展示了上下文工程系统的完整工作流程：

1. **智能感知**: 系统正确识别了复杂任务需求
2. **动态适配**: 选择了最适合的模板和工具
3. **结构化推理**: 通过TAO模式实现了清晰的思考过程
4. **持久化记忆**: 将对话历史结构化存储
5. **可扩展性**: 整个流程都基于MCP协议，便于扩展

通过这个示例，我们可以看到MCP架构如何实现符号主义专家系统与连接主义LLM的完美结合，为智能体系统提供了强大的上下文编排能力。

---

*本文档描述了基于MCP架构的上下文工程系统，实现了符号主义专家系统与连接主义LLM的完美结合，为智能体系统提供了强大的上下文编排能力。*