#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI-Agent 页面 - 桌面自动化代理
基于 OSWorld 架构实现的多模态桌面代理
"""

import gradio as gr
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


def initialize_agent(gui_agent_service, provider_name, os_type, model_name, api_key, base_url):
    """初始化 GUI-Agent 环境和代理"""
    try:
        # 使用提供的配置初始化
        result = gui_agent_service.initialize(
            provider_name=provider_name,
            os_type=os_type,
            model=model_name,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None
        )
        
        if result['status'] == 'success':
            return f"✅ {result['message']}", True
        else:
            return f"❌ {result['message']}", False
            
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}", False


def run_task_step_by_step(gui_agent_service, instruction, max_steps, sleep_time):
    """执行任务（逐步）"""
    try:
        if not instruction or not instruction.strip():
            return "❌ 请输入任务指令", [], ""
        
        # 运行任务
        result = gui_agent_service.run_task(
            instruction=instruction,
            max_steps=int(max_steps),
            sleep_after_execution=float(sleep_time)
        )
        
        if result['status'] == 'error':
            return f"❌ {result['message']}", [], ""
        
        # 格式化结果
        task_result = result['results']
        status_emoji = {
            'completed': '✅',
            'failed': '❌',
            'max_steps_reached': '⏱️',
            'running': '🔄'
        }
        
        summary = f"""
        {status_emoji.get(task_result['final_status'], '❓')} 任务状态: {task_result['final_status']}
        
        📋 任务指令: {task_result['instruction']}
        📊 执行步数: {task_result['total_steps']} / {max_steps}
        """
        
        # 构建步骤表格
        steps_data = []
        for step in task_result['steps']:
            for action_result in step['action_results']:
                steps_data.append([
                    step['step'],
                    action_result['action'],
                    '✅' if action_result['reward'] > 0 else ('❌' if action_result['done'] else '⏸️'),
                    action_result.get('screenshot_path', 'N/A')
                ])
        
        # 获取最新的截图路径
        latest_screenshot = ""
        if task_result['steps']:
            last_step = task_result['steps'][-1]
            if last_step['action_results']:
                latest_screenshot = last_step['action_results'][-1].get('screenshot_path', '')
        
        return summary, steps_data, latest_screenshot
        
    except Exception as e:
        return f"❌ 任务执行失败: {str(e)}", [], ""


def get_agent_stats(gui_agent_service):
    """获取代理统计信息"""
    try:
        stats = gui_agent_service.get_stats()
        
        html_content = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4>📊 GUI-Agent 统计信息</h4>
            <ul>
                <li><strong>总任务数:</strong> {stats['total_tasks']}</li>
                <li><strong>成功任务:</strong> {stats['successful_tasks']} ✅</li>
                <li><strong>失败任务:</strong> {stats['failed_tasks']} ❌</li>
                <li><strong>运行状态:</strong> {'🔄 运行中' if stats['is_running'] else '⏸️ 空闲'}</li>
                <li><strong>PyAutoGUI:</strong> {'✅ 可用' if stats['has_pyautogui'] else '❌ 不可用'}</li>
                <li><strong>OpenAI API:</strong> {'✅ 已配置' if stats['has_openai'] else '⚠️ 未配置'}</li>
            </ul>
            <p style="color: #6c757d; font-size: 0.9em;">统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        return html_content
        
    except Exception as e:
        return f"<p style='color: red;'>获取统计信息失败: {str(e)}</p>"


def view_screenshot(screenshot_path):
    """查看截图"""
    try:
        if not screenshot_path or screenshot_path == 'N/A':
            return None
        
        if os.path.exists(screenshot_path):
            return screenshot_path
        else:
            return None
            
    except Exception as e:
        print(f"查看截图失败: {e}")
        return None


def build_gui_agent_tab(gui_agent_service=None):
    """构建 GUI-Agent 页面"""
    
    # 如果没有提供服务实例，创建一个
    if gui_agent_service is None:
        from ..gui_agent_service import gui_agent_service as default_service
        gui_agent_service = default_service
    
    with gr.Blocks() as gui_agent_tab:
        gr.Markdown("""
        ### 🤖 GUI-Agent - 桌面自动化代理
        
        基于 [OSWorld](https://github.com/xlang-ai/OSWorld) 架构实现的多模态桌面代理。
        
        **核心能力：**
        - 👀 **观察**：获取屏幕截图
        - 🧠 **思考**：基于视觉语言模型（VL Model）理解任务并决策
        - 🖱️ **行动**：通过 PyAutoGUI 执行鼠标、键盘操作
        - 🔄 **循环**：持续执行直到任务完成
        
        **工作流程：** 观察（截图） → 决策（LLM推理） → 行动（PyAutoGUI） → 循环
        """)
        
        with gr.Tabs():
            # Tab 1: 环境初始化
            with gr.Tab("⚙️ 步骤0：环境配置"):
                gr.Markdown("#### 初始化 DesktopEnv 和 PromptAgent")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### 环境配置")
                        
                        provider_name = gr.Dropdown(
                            choices=["local", "docker", "vmware", "aws"],
                            value="local",
                            label="Provider 类型",
                            info="local: 本地环境, docker/vmware: 虚拟机, aws: 云端"
                        )
                        
                        os_type = gr.Dropdown(
                            choices=["macOS", "Ubuntu", "Windows"],
                            value="macOS",
                            label="操作系统类型",
                            info="目标系统的操作系统"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("##### 模型配置")
                        
                        model_name = gr.Dropdown(
                            choices=["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo", "claude-3-opus"],
                            value="gpt-4o",
                            label="视觉语言模型",
                            info="用于理解屏幕和规划动作的 VL 模型"
                        )
                        
                        api_key = gr.Textbox(
                            label="API Key",
                            placeholder="留空则使用环境变量 OPENAI_API_KEY",
                            type="password"
                        )
                        
                        base_url = gr.Textbox(
                            label="API Base URL",
                            placeholder="留空则使用 https://api.openai.com/v1",
                            value=""
                        )
                
                init_btn = gr.Button("🚀 初始化环境和代理", variant="primary", size="lg")
                
                init_status = gr.Textbox(
                    label="初始化状态",
                    lines=3,
                    interactive=False
                )
                
                init_success_state = gr.State(value=False)
                
                init_btn.click(
                    fn=lambda p, o, m, k, b: initialize_agent(gui_agent_service, p, o, m, k, b),
                    inputs=[provider_name, os_type, model_name, api_key, base_url],
                    outputs=[init_status, init_success_state]
                )
            
            # Tab 2: 任务执行
            with gr.Tab("🚀 步骤1-6：任务执行"):
                gr.Markdown("""
                #### 输入任务并执行完整的"观察-思考-行动"循环
                
                **执行步骤：**
                1. **重置环境**：加载任务配置
                2. **获取截图**：捕获当前屏幕状态
                3. **VL 模型推理**：分析截图并规划动作
                4. **解析动作**：从模型输出中提取 PyAutoGUI 命令
                5. **执行动作**：调用 `env.step()` 执行命令
                6. **循环**：重复 2-5 直到任务完成或达到最大步数
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### 任务配置")
                        
                        task_instruction = gr.Textbox(
                            label="任务指令",
                            placeholder="例如：打开浏览器并搜索 OSWorld 项目",
                            lines=3
                        )
                        
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=15,
                            step=1,
                            label="最大步数"
                        )
                        
                        sleep_time = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=1.0,
                            step=0.5,
                            label="每步后等待时间（秒）"
                        )
                        
                        run_task_btn = gr.Button("▶️ 执行任务", variant="primary", size="lg")
                        
                        task_summary = gr.Textbox(
                            label="任务执行摘要",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("##### 执行结果")
                        
                        steps_table = gr.Dataframe(
                            headers=["步骤", "动作", "状态", "截图路径"],
                            label="动作执行记录",
                            interactive=False,
                            wrap=True
                        )
                
                gr.Markdown("##### 最新截图")
                latest_screenshot_path = gr.Textbox(visible=False)
                latest_screenshot = gr.Image(
                    label="最新屏幕截图",
                    type="filepath"
                )
                
                run_task_btn.click(
                    fn=lambda i, m, s: run_task_step_by_step(gui_agent_service, i, m, s),
                    inputs=[task_instruction, max_steps, sleep_time],
                    outputs=[task_summary, steps_table, latest_screenshot_path]
                ).then(
                    fn=view_screenshot,
                    inputs=[latest_screenshot_path],
                    outputs=[latest_screenshot]
                )
            
            # Tab 3: 示例任务
            with gr.Tab("📚 示例任务"):
                gr.Markdown("""
                #### 预置任务示例
                
                以下是一些可以尝试的任务示例（需要根据实际环境调整）：
                """)
                
                examples = gr.Examples(
                    examples=[
                        ["移动鼠标到屏幕中心并点击", 10, 1.0],
                        ["打开 Spotlight 搜索（Command+Space）", 5, 1.5],
                        ["截图并保存（Command+Shift+3）", 3, 2.0],
                        ["打开系统偏好设置", 15, 1.0],
                        ["在桌面上右键点击", 5, 1.0],
                    ],
                    inputs=[task_instruction, max_steps, sleep_time],
                    label="点击示例自动填充"
                )
            
            # Tab 4: 监控与统计
            with gr.Tab("📊 监控与统计"):
                gr.Markdown("#### 系统状态监控")
                
                stats_display = gr.HTML(
                    label="统计信息"
                )
                
                refresh_stats_btn = gr.Button("🔄 刷新统计", variant="secondary")
                
                refresh_stats_btn.click(
                    fn=lambda: get_agent_stats(gui_agent_service),
                    inputs=[],
                    outputs=[stats_display]
                )
                
                gr.Markdown("""
                #### 系统要求
                
                **必需依赖：**
                - `pyautogui`: 桌面自动化控制
                - `Pillow`: 图像处理
                - `openai`: OpenAI API 客户端（如果使用 OpenAI 模型）
                
                **可选配置：**
                - 环境变量 `OPENAI_API_KEY`: OpenAI API 密钥
                - 环境变量 `OPENAI_BASE_URL`: 自定义 API 端点
                
                **安装命令：**
                ```bash
                pip install pyautogui Pillow openai
                ```
                
                **参考资料：**
                - [OSWorld GitHub](https://github.com/xlang-ai/OSWorld)
                - [OSWorld 论文](https://arxiv.org/abs/2404.07972)
                - [PyAutoGUI 文档](https://pyautogui.readthedocs.io/)
                """)
                
                # 页面加载时自动显示统计
                gui_agent_tab.load(
                    fn=lambda: get_agent_stats(gui_agent_service),
                    inputs=[],
                    outputs=[stats_display]
                )
    
    return gui_agent_tab


if __name__ == "__main__":
    # 测试界面
    from ..gui_agent_service import gui_agent_service
    
    demo = build_gui_agent_tab(gui_agent_service)
    demo.launch()

