# 上下文工程与 MCP：代码与图解参考

[当前模块说明](../README.md) · [项目说明](../../../README.md)

原指南重复维护的启动步骤、能力列表和固定性能数字已经移出主说明。使用当前工作台时，先按模块 README 完成操作；需要理解旧设计时再展开下方示例。

## 当前实现入口

- [src/search_engine/mcp_tab/context_pipeline.py](../../../src/search_engine/mcp_tab/context_pipeline.py)
- [src/search_engine/mcp_tab/smart_agent_demo.py](../../../src/search_engine/mcp_tab/smart_agent_demo.py)
- [src/search_engine/mcp/dynamic_mcp_server.py](../../../src/search_engine/mcp/dynamic_mcp_server.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

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

### 示例 2

```mermaid
graph TB
    A[用户意图] --> B[阶段1: 模板选择]
    B --> C[阶段2: 上下文装配]
    C --> D[阶段3: LLM推理]
    D --> E[阶段4: 上下文更新]
    E --> F[下一轮对话]
```

### 示例 3

```mermaid
sequenceDiagram
    participant User as 用户
    participant CE as 上下文工程服务器
    participant Prompts as MCP 提示词
    participant Tools as MCP 工具
    participant Resources as MCP 资源
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

### 示例 4

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

### 示例 5

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

### 示例 6

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

### 示例 7

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

### 示例 8

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

### 示例 9

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

### 示例 10

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

### 示例 11

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

### 示例 12

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

### 示例 13

```
**Reasoning（思考）**: 用户询问的是MCP的含义。MCP可能指代多个不同的概念，包括但不限于：多芯片封装(Multi-Chip Package)、内存控制处理器(Memory Control Processor)等。为了给出最准确的答案，我需要先确认用户具体指的是哪一个领域或上下文中的MCP。

**Action（行动）**: final_answer: MCP可以代表很多不同的意思，例如在计算机科学中它可以指"多芯片封装"(Multi-Chip Package) 或者是"内存控制处理器"(Memory Control Processor)。请问您是在哪个特定背景下提到MCP的呢？这样我可以提供更精确的信息给您。
```

### 示例 14

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

### 示例 15

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

### 示例 16

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

### 示例 17

```bash
# 启动MCP服务器
python src/search_engine/mcp/dynamic_mcp_server.py

# 启动主系统
python start_system.py
```

### 示例 18

```
用户意图: "请帮我搜索关于MCP架构的文档，并分析其核心优势"
任务类型: 检索 + 分析
复杂度: 中等（需要工具调用 + 推理分析）
```

### 示例 19

```json
{
    "selected_template": "context_engineering",
    "reasoning": "用户需要搜索文档并进行专业分析，context_engineering模板支持完整的思考-行动-观察模式，能够处理复杂的多步推理任务",
    "confidence": 0.95
}
```

### 示例 20

```python
# 从MCP服务器获取context_engineering模板
template_content = mcp_manager.get_prompt("unified_server", "context_engineering", {
    "user_input": "请帮我搜索关于MCP架构的文档，并分析其核心优势"
})
```

### 示例 21

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

### 示例 22

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

### 示例 23

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

### 示例 24

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

### 示例 25

```markdown
**Reasoning（思考）**: 用户需要搜索关于MCP架构的文档并分析其核心优势。这是一个需要外部信息支持的分析任务。我需要先使用retrieve工具搜索相关文档，然后基于搜索结果进行分析。

**Action（行动）**: retrieve(reasoning="需要搜索MCP架构相关文档以支持分析", action="search", query="MCP架构 Model Context Protocol", top_k=5, include_metadata=True)

**Observation（观察）**: 检索到3个相关文档：
1. "MCP架构设计原理" - 相关性得分: 0.89
2. "Model Context Protocol技术指南" - 相关性得分: 0.85  
3. "MCP在智能体系统中的应用" - 相关性得分: 0.82

**Reasoning（思考）**: 基于检索到的文档，我可以分析MCP架构的核心优势。从文档内容来看，MCP架构具有标准化、模块化、可扩展性等优势。


```

### 示例 26

```python
tao_record = {
    "timestamp": "2025-08-31 23:45:30",
    "reasoning": "用户需要搜索关于MCP架构的文档并分析其核心优势。这是一个需要外部信息支持的分析任务。我需要先使用retrieve工具搜索相关文档，然后基于搜索结果进行分析。",
    "action": "retrieve(reasoning=\"需要搜索MCP架构相关文档以支持分析\", action=\"search\", query=\"MCP架构 Model Context Protocol\", top_k=5, include_metadata=True)",
    "observation": "检索到3个相关文档，并基于文档内容分析了MCP架构的5个核心优势"
}
```

### 示例 27

```python
# 调用MCP工具添加对话轮次
await mcp_manager.add_conversation_turn(json.dumps(tao_record))
```

### 示例 28

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

### 示例 29

```markdown
**Action（行动）**: final_answer: 基于检索到的文档，MCP架构具有以下核心优势：

1. **标准化协议**: MCP提供了统一的接口标准，使得不同组件可以无缝集成
2. **模块化设计**: 将复杂的智能体系统分解为独立的prompts、tools、resources模块
3. **可扩展性**: 支持动态添加新的工具和资源，无需修改核心架构
4. **符号主义与连接主义结合**: 通过专家系统管理上下文，LLM专注推理生成
5. **生态兼容性**: 支持所有MCP兼容的客户端和服务

MCP架构通过将上下文工程标准化，实现了智能体系统的模块化和可维护性，是现代AI系统的重要技术基础。
```

### 示例 30

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

</details>
