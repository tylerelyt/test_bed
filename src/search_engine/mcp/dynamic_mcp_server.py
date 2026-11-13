#!/usr/bin/env python3
"""
动态MCP服务器

完全隔离解耦的MCP服务器，所有prompts、tools、resources都通过MCP协议动态发现
"""
import asyncio
import json
import sys
import os
from typing import Dict, Any, List
from fastmcp import FastMCP

# 确保能导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

class DynamicMCPServer:
    """动态MCP服务器 - 完全隔离解耦"""
    
    def __init__(self, server_name: str = "dynamic-mcp-server"):
        """初始化动态MCP服务器"""
        from src.search_engine.workbench_access import AccessPolicy, configure_mcp, mcp_auth
        self.access = AccessPolicy()
        self.mcp = FastMCP(server_name, auth=mcp_auth(self.access), stateless_http=True)
        configure_mcp(self.mcp, self.access)
        self._index_service = None
        
        # 注册所有功能
        self._register_prompts()
        self._register_tools()
        self._register_resources()
        from src.search_engine.workbench_context import register_context_mcp
        register_context_mcp(self.mcp)
        
        print(f"🚀 初始化动态MCP服务器: {server_name}")
        print("对话历史按项目、身份和会话隔离保存")
        print("🔒 架构: 完全隔离解耦，所有功能通过MCP协议动态发现")
    
    @property
    def index_service(self):
        if self._index_service is None:
            from src.search_engine.service_manager import get_index_service
            self._index_service = get_index_service()
        return self._index_service

    def _register_prompts(self):
        """注册提示词 - 通过MCP协议动态发现"""
        
        @self.mcp.prompt("simple_chat")
        def simple_chat_prompt(user_input: str = "") -> str:
            """简单对话提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、输出格式
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个友好、专业的AI助手，善于回答各种问题。
你的特点：
1. 回答简洁明了，重点突出
2. 语言流畅自然，易于理解
3. 必要时提供例子或解释
4. 态度友好，乐于助人

[当前状态] 
处理时间: ${{local:current_time}}
用户意图: ${{local:user_intent}}

[对话历史] 
${{mcp:resource:conversation://current/history}}

[用户信息] 
${{local:user_profile}}

[系统概览] 
${{local:system_overview}}

[可用工具] 
${{mcp:tool:dynamic_tool_selection}}

[用户问题] 
{user_input}

[输出格式] 
${{local:tao_example}}

现在请回答用户问题。"""
        
        @self.mcp.prompt("rag_answer")
        def rag_answer_prompt(user_input: str = "") -> str:
            """RAG检索增强提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、RAG流程、输出格式
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个专业的信息检索与分析专家，擅长使用RAG（检索增强生成）技术回答问题。
你的核心能力：
1. 精准检索：从知识库中检索最相关的信息
2. 深度理解：分析检索结果，提取关键信息
3. 综合回答：结合检索内容和背景知识，给出完整答案
4. 来源标注：明确标注信息来源，增强可信度

[当前状态] 
处理时间: ${{local:current_time}}
用户意图: ${{local:user_intent}}

[对话历史] 
${{mcp:resource:conversation://current/history}}

[用户信息] 
${{local:user_profile}}

[系统概览] 
${{local:system_overview}}

[可用工具] 
${{mcp:tool:dynamic_tool_selection}}

[用户问题] 
{user_input}

[RAG工作流程]
1. **理解问题**：分析用户问题的核心意图和关键信息需求
2. **检索信息**：使用retrieve工具从知识库检索相关文档
3. **分析整合**：评估检索结果的相关性和可信度
4. **生成答案**：基于检索内容生成准确、完整的回答
5. **标注来源**：注明信息来源，便于用户验证

[输出格式] 
${{local:tao_example}}

现在请使用RAG流程回答用户问题。"""
        
        @self.mcp.prompt("react_reasoning")
        def react_reasoning_prompt(user_input: str = "") -> str:
            """ReAct推理提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、Few-shot示例
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个专业的AI智能体，擅长使用ReAct范式进行推理和决策。
你的核心能力：
1. 深度思考：分析问题、拆解任务、规划步骤
2. 工具调用：根据需要调用合适的工具获取信息
3. 持续观察：基于观察结果调整策略
4. 最终回答：综合所有信息给出准确答案

[当前状态] 处理时间: ${{local:current_time}}
用户意图: ${{local:user_intent}}
模型: ${{local:model_name}}

[历史] ${{mcp:resource:conversation://current/history}}

[可用工具] ${{mcp:tool:dynamic_tool_selection}}

[用户问题] {user_input}

[执行范式] ReAct (Reasoning + Acting)
你必须严格按照以下格式输出：

**示例1：需要调用工具**
Thought: 我需要搜索相关信息来回答这个问题
Action: retrieve
Action Input: {{"query": "xxx", "top_k": 3}}
Observation: [工具返回的结果]
Thought: 基于搜索结果，我现在可以回答了
Final Answer: 最终答案内容

**示例2：无需调用工具**
Thought: 这是一个简单的问题，我可以直接回答
Final Answer: 最终答案内容

**重要规则**：
1. 每次必须以 Thought: 开始思考
2. 如需工具，输出 Action: 和 Action Input:
3. 观察工具结果后继续思考
4. 确定答案后，以 Final Answer: 输出
5. Final Answer: 标记表示任务完成

请开始执行："""
        
        @self.mcp.prompt("code_review")
        def code_review_prompt(user_input: str = "") -> str:
            """代码审查提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、审查标准、输出格式
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个经验丰富的高级软件工程师和代码审查专家。
你的专长：
1. 代码质量：评估代码的可读性、可维护性和健壮性
2. 安全审计：识别潜在的安全漏洞和风险
3. 性能优化：发现性能瓶颈和优化机会
4. 最佳实践：确保代码符合行业标准和最佳实践

[当前状态] 
处理时间: ${{local:current_time}}
审查任务: ${{local:user_intent}}

[对话历史] 
${{mcp:resource:conversation://current/history}}

[代码/问题] 
{user_input}

[审查维度]
1. **代码质量**：命名规范、代码结构、注释完整性
2. **安全性**：输入验证、权限控制、敏感信息处理
3. **性能**：算法效率、资源使用、潜在瓶颈
4. **可维护性**：模块化、耦合度、测试覆盖
5. **最佳实践**：设计模式、错误处理、日志记录

[输出格式]
请按照以下结构提供审查意见：
1. 总体评价（优点和问题概述）
2. 具体问题列表（按严重程度排序）
3. 改进建议（附代码示例）
4. 最佳实践建议

现在请进行代码审查。"""
        
        @self.mcp.prompt("financial_analysis")
        def financial_analysis_prompt(user_input: str = "") -> str:
            """财务分析提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、分析框架、输出格式
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个资深的财务分析专家和投资顾问。
你的核心能力：
1. 财务报表分析：深入理解资产负债表、利润表、现金流量表
2. 比率分析：计算和解释关键财务比率
3. 趋势分析：识别财务数据的变化趋势和规律
4. 风险评估：评估财务风险和投资价值
5. 战略建议：提供基于数据的决策建议

[当前状态] 
处理时间: ${{local:current_time}}
分析任务: ${{local:user_intent}}

[对话历史] 
${{mcp:resource:conversation://current/history}}

[用户信息] 
${{local:user_profile}}

[系统概览] 
${{local:system_overview}}

[可用工具] 
${{mcp:tool:dynamic_tool_selection}}

[分析需求] 
{user_input}

[分析框架]
1. **数据收集**：确认需要的财务数据和信息来源
2. **比率计算**：计算关键财务比率（流动比率、ROE、ROA等）
3. **趋势分析**：分析历史数据，识别变化趋势
4. **对标分析**：与行业平均水平或竞争对手对比
5. **风险评估**：识别潜在风险和机会
6. **结论建议**：给出明确的结论和行动建议

[输出格式] 
${{local:tao_example}}

现在请进行财务分析。"""
        
        @self.mcp.prompt("context_engineering")
        def context_engineering_prompt(user_input: str = "") -> str:
            """上下文工程专用提示词 - 使用占位符模式
            
            固定分区（模板内定义）：人设、Few-shot示例
            动态分区（占位符）：通过CE Server替换
            
            占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
            """
            return f"""[人设] 你是一个专业的上下文工程专家，擅长动态决策和智能推理

[当前状态] 处理时间: ${{local:current_time}}
用户意图: ${{local:user_intent}}
模型: ${{local:model_name}}

[历史] ${{mcp:resource:conversation://current/history}}

[可用工具] ${{mcp:tool:dynamic_tool_selection}}

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
            
            Args:
                reasoning: 模型的推理过程，说明为什么需要检索
                action: 行动决策，"search"表示执行检索，"skip"表示跳过
                query: 搜索查询字符串
                top_k: 返回的文档数量，默认5个
                include_metadata: 是否包含文档元数据，默认True
                
            Returns:
                包含检索结果和观察信息的字典
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
                
                # index_service.search 返回 List[Tuple[str, float, str]]
                # 格式: (doc_id, score, text)
                if isinstance(results, list):
                    for doc_id, score, text in results:
                        doc_info = {
                            "id": doc_id,
                            "content": text,
                            "score": float(score)
                        }
                        documents.append(doc_info)
                    
                    index_size = len(results)
                elif isinstance(results, dict):
                    # 兼容其他可能的返回格式（字典）
                    for doc in results.get("documents", []):
                        doc_info = {
                            "content": doc.get("content", ""),
                            "score": doc.get("score", 0.0)
                        }
                        if include_metadata:
                            doc_info["metadata"] = doc.get("metadata", {})
                        documents.append(doc_info)
                    
                    index_size = results.get("total_documents", 0)
                else:
                    # 未知格式
                    index_size = 0
                
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
                        "index_size": index_size
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
            """只读取当前项目、身份与会话的历史。"""
            from src.search_engine.workbench_history import HistoryStore
            return json.dumps(HistoryStore().read(), ensure_ascii=False)

    def append_to_history(self, tao_record: dict) -> None:
        from src.search_engine.workbench_history import HistoryStore
        HistoryStore().append(tao_record)

    async def start_server(self, host: str = "localhost", port: int = 3001):
        """启动服务器"""
        print(f"📍 启动动态MCP服务器: http://{host}:{port}/mcp")
        print("🔒 特性: 完全隔离解耦，所有功能通过MCP协议动态发现")
        print("📝 提示词: simple_chat, rag_answer, react_reasoning, code_review, financial_analysis, context_engineering")
        print("🛠️  工具: retrieve (支持思考-行动-观察模式)")
        print("📚 资源: conversation://current/history")
        print("🧠 上下文工程: 支持完整的思考-行动-观察循环")
        
        await self.mcp.run_http_async(host=host, port=port)

def main():
    """Initialize synchronous stores before starting the MCP event loop."""
    server = DynamicMCPServer()
    asyncio.run(server.start_server())

if __name__ == "__main__":
    main()
