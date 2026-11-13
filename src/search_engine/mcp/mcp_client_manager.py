#!/usr/bin/env python3
"""
MCP客户端管理器 - 基于FastMCP的正确实现

使用FastMCP的依赖注入系统来管理上下文
"""
import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

# 确保能导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

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
    
    async def connect_all_servers(self) -> Dict[str, bool]:
        """连接所有MCP服务器 - 使用标准 FastMCP Client"""
        print("🔗 连接MCP服务器...")
        
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
        
        connection_results = {}
        
        for server_name, config in self.server_configs.items():
            try:
                print(f"   📡 连接 {server_name}: {config['url']}")
                
                # ✅ 使用标准 FastMCP Client 而不是 as_proxy
                # Client 支持完整的 MCP 协议，包括 read_resource()
                from src.search_engine.workbench_access import client_headers
                client = Client(StreamableHttpTransport(config['url'], headers=client_headers()))
                
                # 测试连接（进入上下文）
                await self._test_connection(client, server_name)
                
                # 保存客户端
                self.clients[server_name] = client
                self._connection_status[server_name] = True
                connection_results[server_name] = True
                
                print(f"   ✅ {server_name} 连接成功")
                
            except Exception as e:
                print(f"   ❌ {server_name} 连接失败: {e}")
                self._connection_status[server_name] = False
                connection_results[server_name] = False
        
        return connection_results
    
    async def _test_connection(self, client, server_name: str):
        """测试MCP服务器连接 - C/S架构"""
        try:
            # ✅ 使用 async with 测试连接
            async with client:
            # 获取工具列表测试连接
                tools = await client.list_tools()
                print(f"   📋 {server_name} 可用工具: {len(tools) if tools else 0} 个")
            
            # 获取资源列表测试连接
                resources = await client.list_resources()
                print(f"   📚 {server_name} 可用资源: {len(resources) if resources else 0} 个")
            
            # 获取提示词列表测试连接
                prompts = await client.list_prompts()
                print(f"   📝 {server_name} 可用提示词: {len(prompts) if prompts else 0} 个")
            
        except Exception as e:
            raise Exception(f"连接测试失败: {e}")
    
    def get_client(self, server_name: str) -> Optional[FastMCP]:
        """获取指定服务器的客户端"""
        if server_name not in self.clients:
            return None
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
        from src.search_engine.workbench_access import client_headers
        return Client(StreamableHttpTransport(self.server_configs[server_name]['url'], headers=client_headers()))
    
    def is_connected(self, server_name: str) -> bool:
        """检查服务器是否已连接"""
        return self._connection_status.get(server_name, False)
    
    def connect(self, server_name: str) -> bool:
        """同步连接指定服务器"""
        try:
            import asyncio
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 连接所有服务器
                results = loop.run_until_complete(self.connect_all_servers())
                return results.get(server_name, False)
            finally:
                loop.close()
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用MCP工具 - C/S架构"""
        client = self.get_client("unified_server")
        
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            # ✅ 使用标准 FastMCP Client API
            async with client:
                result = await client.call_tool(
                    name=tool_name,
                    arguments=params
                )
            
                # 处理返回结果
            if hasattr(result, 'content'):
                return {"content": str(result.content), "type": "text"}
            elif isinstance(result, (dict, list, str, int, float, bool)):
                return result
            else:
                return {"content": str(result), "type": "unknown"}
        except Exception as e:
            raise Exception(f"调用工具 {tool_name} 失败: {e}")
    
    async def get_resource(self, resource_uri: str) -> str:
        """
        获取MCP资源
        
        特殊处理：
        - conversation://current/history → 通过MCP Resource读取文件内容
        - 其他资源 → 从MCP Server获取
        
        读通过MCP，写入按当前项目、身份与会话隔离的存储
        """
        # ✅ 对话历史通过MCP Resource读取（MCP Server会读文件）
        client = self.get_client("unified_server")
        
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            # FastMCP Client 需要在 async with 上下文中使用
            async with client:
                content = await client.read_resource(resource_uri)
            
                # 提取文本内容
                if content and len(content) > 0:
                    first_content = content[0]
                    
                    if hasattr(first_content, 'text'):
                        return first_content.text
                    elif isinstance(first_content, str):
                        return first_content
                    elif isinstance(first_content, dict):
                        if 'text' in first_content:
                            return first_content['text']
                        return json.dumps(first_content, ensure_ascii=False)
                    else:
                        return str(first_content)
                
                return "[]"
            
        except Exception as e:
            raise Exception(f"获取资源 {resource_uri} 失败: {e}")
    
    async def add_conversation_turn(self, tao_data: str) -> str:
        """
        添加对话轮次 - 直接 append 到 JSONL 文件
        
        ✅ JSONL 格式优势：
        - O(1) 追加操作，无需读取整个文件
        - 高效：只需在文件末尾写一行
        - 并发安全：追加操作原子性更好
        - 流式处理：支持增量读取
        
        📚 参考：https://modelcontextprotocol.info/docs/concepts/resources/
        
        MCP 标准订阅流程（未来可实现）：
        1. Client: resources/subscribe("conversation://current/history")
        2. Client: 写入当前身份和会话的历史存储
        3. Server: notifications/resources/updated
        4. Client: resources/read（获取最新内容）
        
        Args:
            tao_data: JSON格式的TAO记录
            
        Returns:
            操作结果消息
        """
        try:
            import json
            
            # 解析TAO数据
            tao_record = json.loads(tao_data)
            
            from src.search_engine.workbench_history import HistoryStore
            HistoryStore().append(tao_record)
            return "对话轮次已写入当前项目、身份和会话的历史"

        except Exception as e:
            raise Exception(f"添加对话轮次失败: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具 - C/S架构"""
        client = self.get_client("unified_server")
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            async with client:
                tools = await client.list_tools()
                return tools if isinstance(tools, list) else []
        except Exception as e:
            raise Exception(f"获取工具列表失败: {e}")
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有资源 - C/S架构"""
        client = self.get_client("unified_server")
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            async with client:
                resources = await client.list_resources()
                return resources if isinstance(resources, list) else []
        except Exception as e:
            raise Exception(f"获取资源列表失败: {e}")
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有提示词"""
        client = self.get_client("unified_server")
        if not client:
            raise Exception("统一MCP服务器未连接")
        
        try:
            async with client:
                prompts = await client.list_prompts()
                # FastMCP Client 返回 prompts 列表
                return prompts if isinstance(prompts, list) else []
        except Exception as e:
            raise Exception(f"获取提示词列表失败: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return {
            "servers": self._connection_status,
            "total_servers": len(self.server_configs),
            "connected_servers": sum(self._connection_status.values()),
            "configs": self.server_configs
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "overall": True,
            "servers": {}
        }
        
        for server_name in self.server_configs.keys():
            if self.is_connected(server_name):
                try:
                    client = self.get_client(server_name)
                    # 测试基本功能
                    tools = await client.get_tools()
                    
                    health_status["servers"][server_name] = {
                        "status": "healthy",
                        "tools_count": len(tools)
                    }
                except Exception as e:
                    health_status["servers"][server_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["overall"] = False
            else:
                health_status["servers"][server_name] = {
                    "status": "disconnected"
                }
                health_status["overall"] = False
        
        return health_status
    
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
                        return loop.run_until_complete(
                            self._get_prompt_async(client, prompt_name, arguments)
                        )
                    finally:
                        loop.close()
            else:
                return f"服务器 {server_name} 未连接"
        except Exception as e:
            return f"提示词获取失败: {str(e)}"
    
    def _get_prompt_sync(self, client, prompt_name: str, arguments: Dict[str, Any] = None) -> str:
        """同步获取提示词"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(self._get_prompt_async(client, prompt_name, arguments))
            finally:
                loop.close()
        except Exception as e:
            return f"同步提示词获取失败: {str(e)}"
    
    async def _get_prompt_async(self, client, prompt_name: str, arguments: Dict[str, Any] = None) -> str:
        """异步获取提示词并原生返回纯文本内容 - C/S架构"""
        try:
            # ✅ 使用 async with 确保 C/S 架构
            async with client:
                prompt_result = await client.get_prompt(
                    name=prompt_name,
                    arguments=arguments or {}
                )
                
                # FastMCP Client 的 get_prompt 返回已渲染的提示词
                # 提取纯文本内容
                return self._extract_plain_text(prompt_result)
        except Exception as e:
            return f"异步提示词获取失败: {str(e)}"

    def _extract_plain_text(self, value: Any) -> str:
        """尽最大可能从FastMCP返回结构提取纯文本。

        处理以下情况：
        - 已是字符串
        - 对象包含 messages / content / text 字段
        - 列表/字典嵌套结构
        - FastMCP prompt对象
        """
        try:
            # 1) 字符串
            if isinstance(value, str):
                return value

            # 2) 列表：拼接各项的文本
            if isinstance(value, list):
                parts = []
                for item in value:
                    extracted = self._extract_plain_text(item)
                    if extracted:
                        parts.append(extracted)
                return "\n\n".join(parts)

            # 3) 字典：text > content > messages
            if isinstance(value, dict):
                if 'text' in value and isinstance(value['text'], str):
                    return value['text']
                if 'content' in value:
                    return self._extract_plain_text(value['content'])
                if 'messages' in value:
                    return self._extract_plain_text(value['messages'])
                return str(value)

            # 4) 具备messages属性
            if hasattr(value, 'messages'):
                return self._extract_plain_text(getattr(value, 'messages'))

            # 5) 具备content属性（例如 PromptMessage / TextContent 容器）
            if hasattr(value, 'content'):
                content = getattr(value, 'content')
                # TextContent可能具备text属性
                if hasattr(content, 'text'):
                    return str(getattr(content, 'text'))
                return self._extract_plain_text(content)
            
            # 6) 处理PromptMessage对象（FastMCP特有）
            if hasattr(value, 'role') and hasattr(value, 'content'):
                # 这是一个PromptMessage对象
                role = getattr(value, 'role', '')
                content = getattr(value, 'content', '')
                content_text = self._extract_plain_text(content)
                if role == 'user':
                    return content_text
                elif role == 'system':
                    return f"[系统] {content_text}"
                else:
                    return f"[{role}] {content_text}"
            
            # 7) FastMCP prompt对象 - 尝试获取其内容
            if hasattr(value, 'content'):
                return self._extract_plain_text(getattr(value, 'content'))
            
            # 8) 如果对象有__str__方法，尝试使用
            if hasattr(value, '__str__'):
                str_repr = str(value)
                # 如果字符串表示包含有用的信息，返回它
                if not str_repr.startswith('<') or '>' not in str_repr:
                    return str_repr
                # 否则尝试获取更多属性
                if hasattr(value, 'name'):
                    return f"Prompt: {getattr(value, 'name')}"
            
            # 9) 最后回退到字符串表示
            return str(value)

            # 6) 具备text属性
            if hasattr(value, 'text'):
                return str(getattr(value, 'text'))

            # 7) 其他：字符串化兜底
            return str(value)
        except Exception:
            return str(value)

# 全局单例实例
_mcp_client_manager = None

def get_mcp_client_manager() -> MCPClientManager:
    """获取全局MCP客户端管理器实例"""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager
