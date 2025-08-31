#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 实现模块

提供符合MCP标准的上下文工程服务，包括：
- Prompts: 结构化提示词模板管理
- Tools: 工具能力封装与安全执行  
- Resources: 数据资源管理与实时同步

遵循"符号主义为连接主义服务"的设计哲学。
"""

__version__ = "1.0.0"
__author__ = "Testbed MCP Team"

from .mcp_client_manager import get_mcp_client_manager, MCPClientManager
from .dynamic_mcp_server import DynamicMCPServer

__all__ = [
    "get_mcp_client_manager",  # 客户端管理器工厂函数
    "MCPClientManager",        # MCP客户端管理器
    "DynamicMCPServer"         # 动态MCP服务器
]
