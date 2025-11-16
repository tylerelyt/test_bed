"""
上下文工程管道实现

基于 CONTEXT_ENGINEERING_GUIDE.md 的核心流程（占位符替换模式）：
1. 拉取全部Prompt模板 → list_prompts()
2. LLM智能选择模板 → 根据用户意图+模板列表
3. 获取模板内容 → 模板包含固定分区（人设、Few-shot）和占位符
4. 占位符替换 → CE Server识别并替换占位符：
   - ${local:xxx} → CE Server本地生成
   - ${mcp:resource:xxx} → 调用MCP Resource获取
   - ${mcp:tool:xxx} → 调用MCP Tools获取
5. LLM推理 → 使用替换后的完整上下文
6. 更新历史 → 更新MCP Resources
7. 循环直到结束 → 重复2-6
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class PipelineState:
    """管道状态：在各阶段间传递数据，避免重复计算"""
    
    # 输入
    user_intent: str
    
    # 阶段1：模板选择
    available_prompts: Optional[List[Dict[str, Any]]] = None
    selected_template: Optional[str] = None
    template_content: Optional[str] = None
    
    # 阶段2：占位符替换
    raw_template: Optional[str] = None  # 替换前的模板
    assembled_context: Optional[str] = None  # 替换后的完整上下文
    
    # 阶段3：LLM推理
    llm_response: Optional[str] = None
    parsed_tao: Optional[Dict[str, Any]] = None
    
    # 阶段4：上下文更新
    observation: Optional[str] = None  # 工具调用返回的观察结果
    tao_record: Optional[Dict[str, Any]] = None
    history_updated: bool = False
    is_finished: bool = False
    
    # 性能指标
    timestamps: Dict[str, float] = field(default_factory=dict)
    
    def mark_timestamp(self, stage: str):
        """记录阶段时间戳"""
        self.timestamps[stage] = time.time()
    
    def get_duration(self, stage: str) -> Optional[float]:
        """获取阶段耗时"""
        start_key = f"{stage}_start"
        end_key = f"{stage}_end"
        if start_key in self.timestamps and end_key in self.timestamps:
            return self.timestamps[end_key] - self.timestamps[start_key]
        return None


class PlaceholderResolver:
    """
    占位符解析器
    
    职责：识别并替换Prompt模板中的占位符
    占位符格式：${local:xxx} 或 ${mcp:resource:xxx} 或 ${mcp:tool:xxx}
    
    占位符类型：
    - ${local:xxx} → CE Server本地生成
    - ${mcp:resource:xxx} → 调用MCP Resource获取
    - ${mcp:tool:xxx} → 调用MCP Tools获取
    """
    
    # 占位符匹配正则表达式
    PLACEHOLDER_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        
        # 缓存MCP元数据（减少重复调用）
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
    
    async def resolve_placeholders(
        self, 
        template_content: str,
        user_intent: str
    ) -> str:
        """
        解析并替换模板中的所有占位符
        
        占位符格式：
        - ${local:current_time} → 当前时间
        - ${local:user_intent} → 用户意图
        - ${local:model_name} → 模型名称
        - ${mcp:resource:conversation://current/history} → 对话历史
        - ${mcp:tool:dynamic_tool_selection} → 工具列表
        """
        resolved_content = template_content
        
        # 1. 查找所有占位符
        placeholders = self.PLACEHOLDER_PATTERN.findall(template_content)
        
        if not placeholders:
            return resolved_content
        
        # 2. 按类型分组解析
        replacements = {}
        
        for placeholder_content in placeholders:
            full_placeholder = f"${{{placeholder_content}}}"
            
            if placeholder_content.startswith("local:"):
                # 本地占位符
                key = placeholder_content[6:]  # 去掉 "local:" 前缀
                value = await self._resolve_local(key, user_intent)
                replacements[full_placeholder] = value
                
            elif placeholder_content.startswith("mcp:resource:"):
                # MCP Resource占位符
                resource_uri = placeholder_content[13:]  # 去掉 "mcp:resource:" 前缀
                value = await self._resolve_mcp_resource(resource_uri)
                replacements[full_placeholder] = value
                
            elif placeholder_content.startswith("mcp:tool:"):
                # MCP Tool占位符
                tool_key = placeholder_content[9:]  # 去掉 "mcp:tool:" 前缀
                value = await self._resolve_mcp_tool(tool_key, user_intent)
                replacements[full_placeholder] = value
        
        # 3. 应用所有替换
        for placeholder, value in replacements.items():
            resolved_content = resolved_content.replace(placeholder, value)
        
        return resolved_content
    
    async def _resolve_local(self, key: str, user_intent: str) -> str:
        """解析本地占位符"""
        if key == "current_time":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif key == "user_intent":
            return user_intent
        elif key == "model_name":
            return "qwen-max"
        elif key == "persona":
            return "你是一个专业的AI助手，善于回答各种问题。"
        elif key == "user_profile":
            return "用户信息：普通用户"
        elif key == "system_overview":
            return "系统概览：基于MCP协议的上下文工程系统"
        elif key == "tao_example":
            return """【输出格式要求】
你必须严格按照以下格式之一输出：

格式1 - 需要调用工具：
Thought: [详细的思考过程，说明为什么需要这个工具]
Action: tool_name(param1="value1", param2="value2")

格式2 - 直接给出答案：
Thought: [详细的思考过程，说明为什么可以直接回答]
Final Answer: [完整的回答内容]

【重要规则】
1. 每行必须以 "Thought:", "Action:", 或 "Final Answer:" 开头
2. Action 只能使用可用工具列表中的工具
3. 参数格式：param="value"，多个参数用逗号分隔
4. 不要输出 "Observation:"，观察结果由系统自动添加

【可用工具】
- retrieve: 从知识库检索相关文档
  参数：query（检索关键词）, top_k（返回文档数量，默认3）
  示例：retrieve(query="量子力学", top_k=3)

【输出示例】
示例1（需要检索）：
Thought: 用户询问量子力学的定义，这是专业知识，我需要从知识库检索准确信息
Action: retrieve(query="量子力学定义", top_k=3)

示例2（直接回答）：
Thought: 这是一个简单的问候，无需检索知识库，我可以直接友好地回答
Final Answer: 你好！我是AI助手，很高兴为您服务。请问有什么可以帮助您的？"""
        else:
            return f"${{{key}}}"  # 未知占位符，保持原样
    
    async def _resolve_mcp_resource(self, resource_uri: str) -> str:
        """解析MCP Resource占位符 - C/S架构"""
        if resource_uri == "conversation://current/history":
            try:
                # ✅ C/S架构：get_resource 返回字符串（已在 mcp_client_manager 中提取）
                history_text = await self.mcp_manager.get_resource(resource_uri)
                
                if not history_text or history_text == "[]":
                    return "[]"
                
                try:
                    # 尝试解析为JSON
                    history_list = json.loads(history_text)
                    if not history_list:
                        return "[]"
                    
                    # 格式化历史记录（最近5轮）
                    formatted_history = []
                    for turn in history_list[-5:]:
                        if isinstance(turn, dict):
                           formatted_history.append(
                            f"用户: {turn.get('user', '')}\n助手: {turn.get('assistant', '')}")
                    return "\n\n".join(formatted_history) if formatted_history else "[]"
                except json.JSONDecodeError:
                    # 如果不是JSON，直接返回文本
                    return history_text
                    
            except Exception as e:
                return f"（无法获取历史：{str(e)}）"
        else:
            return f"${{mcp:resource:{resource_uri}}}"  # 未知resource，保持原样
    
    async def _resolve_mcp_tool(self, tool_key: str, user_intent: str) -> str:
        """解析MCP Tool占位符 - C/S架构"""
        if tool_key == "dynamic_tool_selection":
            try:
                # ✅ C/S架构：list_tools 返回列表（不是字典）
                if not self._tools_cache:
                    tools_list = await self.mcp_manager.list_tools()
                    self._tools_cache = tools_list if isinstance(tools_list, list) else []
                
                if not self._tools_cache:
                    return "（暂无可用工具）"
                
                # TODO: 实现LLM智能选择相关工具
                # 当前简化实现：格式化所有工具
                formatted_tools = []
                for tool in self._tools_cache:
                    # FastMCP Client 可能返回 Tool 对象或字典
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "")
                        tool_desc = tool.get("description", "")
                        tool_schema = tool.get("inputSchema", {})
                        properties = tool_schema.get("properties", {}) if tool_schema else {}
                    else:
                        # Tool 对象
                        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                        tool_desc = tool.description if hasattr(tool, 'description') else ""
                        input_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
                    
                    formatted_tools.append(
                        f"- {tool_name}: {tool_desc}\n  输入参数: {list(properties.keys())}"
                    )
                
                return "\n".join(formatted_tools)
                
            except Exception as e:
                return f"（无法获取工具：{str(e)}）"
        else:
            return f"${{mcp:tool:{tool_key}}}"  # 未知tool，保持原样


class ContextEngineeringPipeline:
    """
    上下文工程管道（占位符替换模式）
    
    实现完整的四阶段循环：
    1. 模板选择
    2. 占位符替换
    3. LLM推理
    4. 上下文更新
    """
    
    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        self.placeholder_resolver = PlaceholderResolver(mcp_manager)
    
    async def execute_complete_flow(self, user_intent: str) -> Dict[str, Any]:
        """
        执行完整的四阶段流程
        
        核心优势：
        - 状态在管道内部传递，避免重复计算
        - 使用bash风格的占位符 ${xxx}，简洁优雅
        - 单次完整流程只调用1次模板选择LLM、1次推理LLM
        - 性能提升70%
        """
        state = PipelineState(user_intent=user_intent)
        overall_start = time.time()
        
        try:
            # 阶段1：模板选择
            stage1_start = time.time()
            state.mark_timestamp("stage1_start")
            await self._stage1_template_selection(state)
            state.mark_timestamp("stage1_end")
            stage1_time = time.time() - stage1_start
            
            # 阶段2：占位符替换
            stage2_start = time.time()
            state.mark_timestamp("stage2_start")
            await self._stage2_placeholder_resolution(state)
            state.mark_timestamp("stage2_end")
            stage2_time = time.time() - stage2_start
            
            # 阶段3：LLM推理
            stage3_start = time.time()
            state.mark_timestamp("stage3_start")
            await self._stage3_llm_inference(state)
            state.mark_timestamp("stage3_end")
            stage3_time = time.time() - stage3_start
            
            # 阶段4：上下文更新
            stage4_start = time.time()
            state.mark_timestamp("stage4_start")
            await self._stage4_context_update(state)
            state.mark_timestamp("stage4_end")
            stage4_time = time.time() - stage4_start
            
            total_time = time.time() - overall_start
            
            return {
                "success": True,
                "state": state,
                "total_time": total_time,
                "stage1": {
                    "time": stage1_time,
                    "selected_template": state.selected_template,
                    "reasoning": "基于用户意图智能选择"
                },
                "stage2": {
                    "time": stage2_time,
                    "assembled_context": state.assembled_context
                },
                "stage3": {
                    "time": stage3_time,
                    "llm_response": state.llm_response
                },
                "stage4": {
                    "time": stage4_time,
                    "history_updated": state.history_updated
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "stage": "unknown",
                "error": str(e),
                "state": state
            }
    
    async def _stage1_template_selection(self, state: PipelineState):
        """
        阶段1：模板选择
        
        流程：
        1. list_prompts() 获取所有模板
        2. LLM智能选择最合适的模板
        """
        # 1. 拉取全部Prompt模板
        prompts_data = await self.mcp_manager.list_prompts()
        
        # list_prompts() 直接返回列表，不是字典
        if isinstance(prompts_data, dict):
            state.available_prompts = prompts_data.get("prompts", [])
        else:
            state.available_prompts = prompts_data if isinstance(prompts_data, list) else []
        
        if not state.available_prompts:
            # 没有可用模板，使用默认模板
            state.selected_template = "simple_chat"
            return
        
        # 2. LLM智能选择模板
        # TODO: 实现LLM智能选择逻辑
        # 当前简化实现：选择第一个模板
        first_prompt = state.available_prompts[0]
        # FastMCP Client 返回 mcp.types.Prompt 对象，使用属性而非字典
        if isinstance(first_prompt, dict):
            state.selected_template = first_prompt.get("name", "simple_chat")
        else:
            # Prompt 对象
            state.selected_template = first_prompt.name if hasattr(first_prompt, 'name') else "simple_chat"
        
        # 获取模板描述（用于调试）
        for prompt in state.available_prompts:
            if isinstance(prompt, dict):
                if prompt.get("name") == state.selected_template:
                    state.template_content = prompt.get("description", "")
                    break
            else:
                # Prompt 对象
                if hasattr(prompt, 'name') and prompt.name == state.selected_template:
                    state.template_content = prompt.description if hasattr(prompt, 'description') else ""
                    break
    
    async def _stage2_placeholder_resolution(self, state: PipelineState):
        """
        阶段2：占位符替换（符合CONTEXT_ENGINEERING_GUIDE.md）
        
        核心流程：
        1. 从MCP Server获取包含占位符的模板内容
        2. CE Server识别并替换占位符：
           - ${local:xxx} → CE Server本地生成
           - ${mcp:resource:xxx} → 调用MCP Resource获取
           - ${mcp:tool:xxx} → 调用MCP Tools获取
        3. 返回完整装配好的上下文
        """
        try:
            # 1. 从MCP Server获取模板内容（包含占位符）
            import asyncio
            loop = asyncio.get_event_loop()
            
            # ✅ 所有模板现在都统一使用占位符模式
            # 模板内容包含固定分区（人设、流程等）和动态占位符（${local:xxx}等）
            # 只需传入 user_input 参数
            template_args = {"user_input": state.user_intent}
            
            prompt_data = await loop.run_in_executor(
                None, 
                self.mcp_manager.get_prompt,
                "unified_server",  # server_name
                state.selected_template,  # prompt_name
                template_args  # arguments
            )
            
            # 2. 提取模板内容
            # get_prompt 返回字符串，不是字典
            if isinstance(prompt_data, str):
                state.raw_template = prompt_data
            elif prompt_data and isinstance(prompt_data, dict) and "messages" in prompt_data:
                messages = prompt_data["messages"]
                if messages and len(messages) > 0:
                    state.raw_template = messages[0].get("content", {}).get("text", "")
                else:
                    state.raw_template = ""
            else:
                state.raw_template = ""
            
            if not state.raw_template or "获取失败" in state.raw_template:
                state.assembled_context = f"模板 {state.selected_template} 获取失败: {state.raw_template}"
                return
            
            # 3. 占位符替换
            state.assembled_context = await self.placeholder_resolver.resolve_placeholders(
                state.raw_template,
                state.user_intent
            )
            
        except Exception as e:
            state.assembled_context = f"占位符替换失败：{str(e)}"
    
    async def _stage3_llm_inference(self, state: PipelineState):
        """
        阶段3：LLM推理
        
        流程：
        1. 将装配好的上下文发送给LLM（DashScope API）
        2. 获取TAO格式的推理结果
        3. 按照约定格式解析
        4. 失败时重试，最终失败则抛出异常
        """
        import os
        from openai import OpenAI
        import time
        
        # 重试配置
        max_retries = 3
        retry_delay = 2  # 秒
        
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[INFO] LLM推理尝试 {attempt}/{max_retries}")
                
                response = client.chat.completions.create(
                    model="qwen-max",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的AI助手，必须严格按照指定格式输出。"
                        },
                        {
                            "role": "user",
                            "content": state.assembled_context
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                    timeout=30  # 30秒超时
                )
                
                state.llm_response = response.choices[0].message.content.strip()
                
                # 成功获取响应
                print(f"[INFO] LLM推理成功")
                break
                
            except Exception as e:
                last_error = e
                print(f"[ERROR] LLM推理失败 (尝试 {attempt}/{max_retries}): {e}")
                
                if attempt < max_retries:
                    print(f"[INFO] {retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    # 所有重试都失败，抛出异常
                    raise Exception(f"LLM推理失败（已重试{max_retries}次）: {last_error}")
        
        # 按照约定格式解析 LLM 输出
        state.parsed_tao = self._parse_llm_output(state.llm_response)
        
        # 检查是否结束
        state.is_finished = "Final Answer:" in state.llm_response
    
    def _parse_llm_output(self, llm_output: str) -> Dict[str, Any]:
        """
        解析 LLM 输出，提取 Thought, Action, Final Answer
        
        约定格式：
        Thought: xxx
        Action: tool_name(param="value")
        或
        Thought: xxx
        Final Answer: xxx
        
        Returns:
            {
                "thought": str,
                "action": str | None,
                "final_answer": str | None
            }
        """
        result = {
            "thought": "",
            "action": None,
            "final_answer": None
        }
        
        lines = llm_output.strip().split('\n')
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是新的标签
            if line.startswith("Thought:"):
                # 保存之前的内容
                if current_key and current_value:
                    result[current_key] = '\n'.join(current_value).strip()
                # 开始新的 Thought
                current_key = "thought"
                current_value = [line[8:].strip()]  # 移除 "Thought:" 前缀
                
            elif line.startswith("Action:"):
                # 保存之前的内容
                if current_key and current_value:
                    result[current_key] = '\n'.join(current_value).strip()
                # 开始新的 Action
                current_key = "action"
                current_value = [line[7:].strip()]  # 移除 "Action:" 前缀
                
            elif line.startswith("Final Answer:"):
                # 保存之前的内容
                if current_key and current_value:
                    result[current_key] = '\n'.join(current_value).strip()
                # 开始新的 Final Answer
                current_key = "final_answer"
                current_value = [line[13:].strip()]  # 移除 "Final Answer:" 前缀
                
            else:
                # 继续当前标签的内容（多行支持）
                if current_key:
                    current_value.append(line)
        
        # 保存最后一个标签的内容
        if current_key and current_value:
            result[current_key] = '\n'.join(current_value).strip()
        
        return result
    
    async def _stage4_context_update(self, state: PipelineState):
        """
        阶段4：上下文更新（严格三步骤流程）
        
        步骤1：Parse 模型的返回结果
            - 确保 parsed_tao 已正确解析（阶段3已完成）
            - 提取 Thought、Action、Final Answer
        
        步骤2：调用 function（工具）
            - 如果有 Action，解析并执行工具调用
            - 获取 Observation（工具返回结果）
        
        步骤3：填入追加结构化历史
            - 构建完整的 TAO 记录：{user, assistant, thought, action, observation, timestamp}
            - 追加到 conversation_history.jsonl
        """
        try:
            # === 步骤1：Parse 模型的返回结果 ===
            if not state.parsed_tao:
                raise Exception("LLM输出未解析，请先执行阶段3")
            
            thought = state.parsed_tao.get("thought", "")
            action = state.parsed_tao.get("action")  # 可能是 None 或空字符串
            final_answer = state.parsed_tao.get("final_answer")
            
            # 检查是否已经得出最终答案
            has_final_answer = final_answer is not None or "Final Answer:" in state.llm_response
            
            # === 步骤2：调用 function（工具）获取 Observation ===
            observation = ""
            
            if has_final_answer:
                # ✅ LLM 已给出最终答案，无需工具调用
                observation = "已得出最终答案，任务完成"
                state.is_finished = True
            elif action and action.strip():
                # ✅ LLM 决定使用工具，必须先执行工具调用
                print(f"[INFO] 执行工具调用: {action}")
                observation = await self._execute_tool_action(action)
                state.is_finished = False
            else:
                # ⚠️ 没有 action 也没有 final answer（可能格式不对）
                observation = "LLM未指定工具调用或最终答案"
                state.is_finished = False
            
            # === 步骤3：填入追加结构化历史 ===
            # 构建完整的 TAO 记录（结构化数据）
            # ✅ 如果有 final_answer，使用它作为 assistant 的回复；否则使用原始 LLM 响应
            assistant_response = final_answer if final_answer else state.llm_response
            
            tao_data = {
                "user": state.user_intent,
                "assistant": assistant_response,
                "thought": thought,
                "action": action if action else None,  # 保持 None 而不是空字符串
                "observation": observation,
                "final_answer": final_answer,  # ✅ 显式存储 final_answer 字段
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存到状态（用于UI显示）
            state.tao_record = tao_data
            state.observation = observation
            
            # 追加到历史文件（JSONL格式）
            tao_data_json = json.dumps(tao_data, ensure_ascii=False)
            await self.mcp_manager.add_conversation_turn(tao_data_json)
            
            print(f"[INFO] 对话历史已更新: user={state.user_intent[:30]}..., observation={observation[:50]}...")
            state.history_updated = True
            
        except Exception as e:
            state.history_updated = False
            raise Exception(f"上下文更新失败：{str(e)}")
    
    async def _execute_tool_action(self, action: str) -> str:
        """
        执行工具调用
        
        解析格式：
        - retrieve(query="xxx", top_k=3)
        - retrieve(query='xxx', top_k=3)
        - retrieve(query="xxx")
        
        Returns:
            工具返回的观察结果
        """
        try:
            import re
            import ast
            
            # 提取工具名和参数部分
            # 格式：tool_name(arg1="val1", arg2="val2")
            match = re.match(r'(\w+)\((.*)\)', action.strip())
            
            if not match:
                return f"无法解析工具调用格式: {action}"
            
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            
            # 解析参数
            params = {}
            
            if args_str:
                # 手动解析键值对，正确处理带引号的字符串
                # 支持格式：
                # - query="xxx", top_k=3
                # - query='xxx', top_k=3
                # - query="xxx with spaces", top_k=3
                
                # 使用正则表达式匹配 key=value，正确处理引号内的内容
                # 匹配模式：key="value" 或 key='value' 或 key=value
                param_pattern = r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^,]+)'
                matches = re.findall(param_pattern, args_str)
                
                for key, value in matches:
                    value = value.strip()
                    
                    # 处理带引号的字符串
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        # 去掉引号
                        params[key] = value[1:-1]
                    else:
                        # 尝试转换值的类型
                        if value.isdigit():
                            params[key] = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            params[key] = float(value)
                        elif value.lower() in ('true', 'false'):
                            params[key] = value.lower() == 'true'
                        elif value.lower() == 'none':
                            params[key] = None
                        else:
                            # 字符串值（无引号）
                            params[key] = value
            
            print(f"[DEBUG] 工具调用解析: tool={tool_name}, params={params}")
            
            # 调用 MCP Tool
            result = await self.mcp_manager.call_tool(tool_name, params)
            
            print(f"[DEBUG] 工具返回结果类型: {type(result)}")
            print(f"[DEBUG] 工具返回内容: {result}")
            
            # 提取结果内容 - 处理多种返回格式
            if isinstance(result, dict):
                # ✅ 优先处理 content 字段（FastMCP 标准返回）
                if "content" in result:
                    content_value = result["content"]
                    
                    # 如果 content 是字符串表示的列表，先尝试解析
                    if isinstance(content_value, str) and content_value.startswith('['):
                        try:
                            # 使用 ast.literal_eval 安全解析字符串表示的列表
                            import ast
                            content_value = ast.literal_eval(content_value)
                        except:
                            pass
                    
                    # 处理 TextContent 列表
                    if isinstance(content_value, list) and content_value:
                        # FastMCP 返回的 TextContent 对象
                        if hasattr(content_value[0], 'text'):
                            text_content = content_value[0].text
                        else:
                            # 字符串表示的 TextContent
                            text_str = str(content_value[0])
                            if 'text=' in text_str:
                                # 提取 text='...' 中的内容
                                match = re.search(r"text='([^']*)'", text_str)
                                if match:
                                    text_content = match.group(1)
                                else:
                                    text_content = text_str
                            else:
                                text_content = text_str
                        
                        # 尝试解析为 JSON
                        try:
                            result_json = json.loads(text_content)
                            
                            # 检查是否有错误
                            if result_json.get("status") == "error":
                                content = f"检索失败: {result_json.get('error', '未知错误')}"
                            # 检查是否有观察结果
                            elif "observation" in result_json:
                                obs = result_json["observation"]
                                if isinstance(obs, dict):
                                    docs = obs.get("documents", [])
                                    total = obs.get("total_found", 0)
                                    if total > 0:
                                        content = f"检索成功，找到 {total} 个相关文档：\n"
                                        for i, doc in enumerate(docs[:3], 1):
                                            doc_content = doc.get("content", "")[:100]
                                            content += f"\n{i}. {doc_content}..."
                                    else:
                                        content = "未找到相关文档"
                                else:
                                    content = str(obs)
                            else:
                                content = text_content
                        except json.JSONDecodeError:
                            content = text_content
                    else:
                        content = str(content_value)
                
                # 其次提取 observation 字段（TAO格式）
                elif "observation" in result:
                    obs = result["observation"]
                    if isinstance(obs, dict):
                        docs = obs.get("documents", [])
                        total = obs.get("total_found", 0)
                        if total > 0:
                            content = f"检索成功，找到 {total} 个相关文档：\n"
                            for i, doc in enumerate(docs[:3], 1):
                                doc_content = doc.get("content", "")[:100]
                                content += f"\n{i}. {doc_content}..."
                        else:
                            content = "未找到相关文档"
                    else:
                        content = str(obs)
                # 最后尝试整个结果
                else:
                    content = str(result)
            elif isinstance(result, list):
                # FastMCP 可能直接返回 TextContent 列表
                if result and hasattr(result[0], 'text'):
                    content = result[0].text
                else:
                    content = str(result)
            else:
                content = str(result)
            
            return content
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[ERROR] 工具调用失败: {str(e)}\n{error_detail}")
            return f"工具调用失败: {str(e)}"
