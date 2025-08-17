#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG标签页UI实现
"""

import gradio as gr
import json
from typing import Dict, Any, Tuple, List
from .rag_service import RAGService

def build_rag_tab(index_service):
    """构建RAG标签页"""
    
    # 初始化RAG服务
    rag_service = RAGService(index_service)
    
    with gr.Column():
        gr.Markdown("""
        # 🤖 上下文工程
        
        支持三种模式：直连LLM / 检索增强（RAG）/ 多步推理（ReAct）。
        """)
        
        # 1. 连接状态检查
        with gr.Row():
            check_connection_btn = gr.Button("🔍 检查Ollama连接", variant="secondary")
            connection_status = gr.Textbox(
                label="连接状态",
                value="点击检查连接状态",
                interactive=False
            )
        
        # 2. 查询界面
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="例如：什么是机器学习？",
                    lines=2
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="检索文档数量"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=["llama3.1:8b", "llama3.2:1b", "qwen2.5:7b"],
                        value="llama3.1:8b",
                        label="选择模型"
                    )

                with gr.Row():
                    retrieval_enabled = gr.Checkbox(
                        label="开启检索增强 (RAG)",
                        value=True
                    )
                    multi_step_enabled = gr.Checkbox(
                        label="开启多步推理",
                        value=False
                    )
                
                rag_query_btn = gr.Button("🚀 执行查询", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 系统状态")
                stats_display = gr.JSON(label="上下文工程服务状态")
        
        # 3. 结果展示
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 生成回答")
                answer_output = gr.Textbox(
                    label="回答",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                processing_info = gr.Textbox(
                    label="处理信息",
                    lines=2,
                    interactive=False
                )
        
        # 4. 提示词展示
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 提示词/推理轨迹")
                prompt_display = gr.Textbox(
                    label="完整提示词或推理轨迹",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    placeholder="执行查询后，这里显示发送给LLM的提示词或ReAct推理轨迹",
                    show_copy_button=True,
                    autoscroll=False
                )
        
        # 5. 检索详情
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 检索结果详情")
                retrieved_docs = gr.DataFrame(
                    headers=["文档ID", "相关度分数", "文档内容"],
                    label="检索到的文档",
                    interactive=False
                )
                
                context_output = gr.Textbox(
                    label="构建的上下文",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
    
    # 事件处理函数
    def check_connection():
        """检查Ollama连接"""
        connected, status = rag_service.check_ollama_connection()
        return status
    
    def refresh_model_list():
        """刷新模型列表"""
        models = rag_service.get_available_models()
        return gr.Dropdown(choices=models, value=models[0] if models else "llama3.1:8b")
    
    def get_rag_stats():
        """获取RAG服务统计信息"""
        return rag_service.get_stats()
    
    def process_rag_query(query: str, top_k: int, model: str, retrieval_enabled_flag: bool, multi_step_flag: bool):
        """处理RAG查询"""
        if not query.strip():
            return (
                "请输入您的问题",
                "未处理",
                [],
                "",
                ""
            )
        
        # 执行RAG查询（带开关）
        result = rag_service.rag_query(
            query=query,
            top_k=top_k,
            model=model,
            retrieval_enabled=retrieval_enabled_flag,
            multi_step=multi_step_flag
        )
        
        # 构建检索结果表格
        retrieved_table = []
        for doc_id, score, content in result.get("retrieved_docs", []):
            # 截断内容以适应表格显示
            truncated_content = content[:100] + "..." if len(content) > 100 else content
            retrieved_table.append([doc_id, f"{score:.4f}", truncated_content])
        
        # 构建处理信息
        processing_info = f"""处理时间: {result.get('processing_time', 0):.2f}秒
使用模型: {result.get('model_used', 'unknown')}
检索文档数: {len(result.get('retrieved_docs', []))}"""
        
        return (
            result.get("answer", "生成回答失败"),
            processing_info,
            retrieved_table,
            result.get("context", ""),
            result.get("prompt_sent", "")
        )
    
    # 绑定事件
    check_connection_btn.click(
        fn=check_connection,
        outputs=[connection_status]
    )
    
    rag_query_btn.click(
        fn=process_rag_query,
        inputs=[query_input, top_k_slider, model_dropdown, retrieval_enabled, multi_step_enabled],
        outputs=[answer_output, processing_info, retrieved_docs, context_output, prompt_display]
    )
    
    # 页面加载时获取统计信息
    stats_display.value = get_rag_stats()
    
    # 定期刷新模型列表
    check_connection_btn.click(
        fn=refresh_model_list,
        outputs=[model_dropdown]
    )
    
    return gr.Column() 