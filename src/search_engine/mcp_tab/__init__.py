#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP上下文工程标签页模块

集成新的MCP架构到现有UI系统中。
"""

from .smart_agent_demo import create_smart_agent_demo

def build_mcp_tab():
    """构建MCP标签页"""
    return create_smart_agent_demo()

__all__ = ["build_mcp_tab"]
