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

from src.search_engine.service_manager import get_index_service

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
        
        print(f"🚀 初始化动态MCP服务器: {server_name}")
        print("🔒 架构: 完全隔离解耦，所有功能通过MCP协议动态发现")
    
    def _register_prompts(self):
        """注册提示词 - 通过MCP协议动态发现"""
        
        @self.mcp.prompt("simple_chat")
        def simple_chat_prompt(
            section_persona: str = "",
            section_current_status: str = "",
            section_conversation_history: str = "",
            section_user_profile: str = "",
            section_system_overview: str = "",
            section_available_tools: str = "",
            section_user_question: str = "",
            section_tao_output_example: str = ""
        ) -> str:
            """简单对话提示词 - 统一分区参数装配"""
            return f"""{section_persona}

{section_current_status}

{section_conversation_history}

{section_user_profile}

{section_system_overview}

{section_available_tools}

{section_user_question}

{section_tao_output_example}"""
        
        @self.mcp.prompt("rag_answer")
        def rag_answer_prompt(
            section_persona: str = "",
            section_current_status: str = "",
            section_conversation_history: str = "",
            section_user_profile: str = "",
            section_system_overview: str = "",
            section_available_tools: str = "",
            section_user_question: str = "",
            section_tao_output_example: str = ""
        ) -> str:
            """RAG检索增强提示词 - 统一分区参数装配"""
            return f"""{section_persona}

{section_current_status}

{section_conversation_history}

{section_user_profile}

{section_system_overview}

{section_available_tools}

{section_user_question}

{section_tao_output_example}"""
        
        @self.mcp.prompt("react_reasoning")
        def react_reasoning_prompt(
            section_persona: str = "",
            section_current_status: str = "",
            section_conversation_history: str = "",
            section_user_profile: str = "",
            section_system_overview: str = "",
            section_available_tools: str = "",
            section_user_question: str = "",
            section_tao_output_example: str = ""
        ) -> str:
            """ReAct推理提示词 - 统一分区参数装配"""
            return f"""{section_persona}

{section_current_status}

{section_conversation_history}

{section_user_profile}

{section_system_overview}

{section_available_tools}

{section_user_question}

{section_tao_output_example}"""
        
        @self.mcp.prompt("code_review")
        def code_review_prompt() -> str:
            """代码审查提示词"""
            return """[人设] 你是一个经验丰富的代码审查专家
[代码] {code_content}
[审查要求] {review_requirements}
[审查结果] 请从以下方面进行审查：
1. 代码质量
2. 安全性
3. 性能
4. 可维护性
5. 最佳实践"""
        
        @self.mcp.prompt("financial_analysis")
        def financial_analysis_prompt(
            section_persona: str = "",
            section_current_status: str = "",
            section_conversation_history: str = "",
            section_user_profile: str = "",
            section_system_overview: str = "",
            section_available_tools: str = "",
            section_user_question: str = "",
            section_tao_output_example: str = ""
        ) -> str:
            """财务分析提示词 - 统一分区参数装配"""
            return f"""{section_persona}

{section_current_status}

{section_conversation_history}

{section_user_profile}

{section_system_overview}

{section_available_tools}

{section_user_question}

{section_tao_output_example}"""
        
        @self.mcp.prompt("context_engineering")
        def context_engineering_prompt(
            section_persona: str = "",
            section_current_status: str = "",
            section_conversation_history: str = "",
            section_user_profile: str = "",
            section_system_overview: str = "",
            section_available_tools: str = "",
            section_user_question: str = "",
            section_tao_output_example: str = ""
        ) -> str:
            """上下文工程专用提示词 - 统一分区参数装配"""
            return f"""{section_persona}

{section_current_status}

{section_conversation_history}

{section_user_profile}

{section_system_overview}

{section_available_tools}

{section_user_question}

{section_tao_output_example}"""

    
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
                print(f"🔍 MCP服务器: 获取对话历史，当前长度: {len(self.conversation_history)}")
                print(f"📝 MCP服务器: 历史内容: {self.conversation_history}")
                return json.dumps(self.conversation_history, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"❌ MCP服务器: 获取历史失败: {e}")
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
                print(f"🔄 MCP服务器: 开始添加对话轮次")
                print(f"📥 MCP服务器: 接收到的数据: {tao_data}")
                
                tao_record = json.loads(tao_data)
                # 只保存TAO结构
                simplified_tao = {
                    "turn": len(self.conversation_history) + 1,
                    "timestamp": tao_record.get("timestamp", "now"),
                    "reasoning": tao_record.get("reasoning", ""),
                    "action": tao_record.get("action", ""),
                    "observation": tao_record.get("observation", "")
                }
                
                print(f"📝 MCP服务器: 简化的TAO记录: {simplified_tao}")
                
                self.conversation_history.append(simplified_tao)
                
                print(f"✅ MCP服务器: 历史已更新，当前长度: {len(self.conversation_history)}")
                print(f"📋 MCP服务器: 完整历史: {self.conversation_history}")
                
                return json.dumps({
                    "status": "success",
                    "message": f"已添加第{simplified_tao['turn']}轮对话",
                    "total_turns": len(self.conversation_history)
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"❌ MCP服务器: 添加对话轮次失败: {e}")
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                }, ensure_ascii=False, indent=2)
    
    async def start_server(self, host: str = "localhost", port: int = 3001):
        """启动服务器"""
        print(f"📍 启动动态MCP服务器: http://{host}:{port}/mcp")
        print("🔒 特性: 完全隔离解耦，所有功能通过MCP协议动态发现")
        print("📝 提示词: simple_chat, rag_answer, react_reasoning, code_review, financial_analysis, context_engineering")
        print("🛠️  工具: retrieve (支持思考-行动-观察模式)")
        print("📚 资源: conversation://current/history")
        print("🧠 上下文工程: 支持完整的思考-行动-观察循环")
        
        await self.mcp.run_http_async(host=host, port=port)

async def main():
    """主函数"""
    server = DynamicMCPServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())
