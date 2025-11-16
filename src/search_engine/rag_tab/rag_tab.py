#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG标签页UI实现
"""

import gradio as gr
import json
from typing import Dict, Any, Tuple, List
from .rag_service import RAGService

def build_rag_tab(index_service, inference_model=None):
    """构建RAG标签页
    
    Args:
        index_service: 索引服务
        inference_model: 共享的InferenceModel实例（可选）
    """
    
    # 初始化RAG服务
    rag_service = RAGService(index_service)
    
    # 如果没有传入inference_model，创建一个新的
    if inference_model is None:
        from ..training_tab.inference_model import InferenceModel
        inference_model = InferenceModel()
    
    with gr.Column():
        gr.Markdown("""
        # 🤖 上下文工程
        
        支持两种模式：
        - **DashScope API**: 使用阿里云通义千问API（在线）
        - **本地模型**: 使用训练好的SFT/DPO模型（需先加载）
        """)
        
        # 1. 模型选择与加载
        with gr.Row():
            with gr.Column(scale=2):
                inference_mode = gr.Radio(
                    choices=["DashScope API", "本地模型"],
                    value="DashScope API",
                    label="推理模式"
                )
                
                # 本地模型选择（仅在选择"本地模型"时显示）
                with gr.Column(visible=False) as local_model_box:
                    with gr.Row():
                        local_model_dropdown = gr.Dropdown(
                            choices=[],  # 初始为空，通过refresh更新
                            value=None,
                            label="选择本地模型",
                            info="从SFT或DPO训练的模型中选择",
                            scale=4
                        )
                        refresh_local_models_btn = gr.Button("🔄", scale=1)
                    
                    with gr.Row():
                        load_model_btn = gr.Button("▶️ 加载模型", variant="primary")
                        unload_model_btn = gr.Button("⏹️ 卸载模型", variant="secondary")
            
            with gr.Column(scale=1):
                model_status = gr.Textbox(
                    label="模型状态",
                    value="DashScope API 模式（无需加载）",
                    interactive=False,
                    lines=4
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
    def refresh_local_models():
        """刷新本地模型列表"""
        try:
            from ..training_tab.llmops_tab import get_trained_models
            sft_models = get_trained_models("sft")
            dpo_models = get_trained_models("dpo")
            all_models = sft_models + dpo_models
            return gr.update(choices=all_models, value=all_models[0] if all_models else None)
        except Exception as e:
            print(f"❌ 刷新模型列表失败: {e}")
            return gr.update(choices=[], value=None)
    
    def toggle_model_box(mode):
        """切换推理模式时显示/隐藏本地模型选择框"""
        if mode == "本地模型":
            status = "请选择并加载本地模型" if not inference_model.loaded else "模型已加载"
            # 切换到本地模型时，自动刷新模型列表
            return gr.update(visible=True), status, refresh_local_models()
        else:
            return gr.update(visible=False), "DashScope API 模式（无需加载）", gr.update()
    
    def load_local_model(model_path):
        """加载本地模型"""
        if not model_path:
            yield "❌ 请选择模型"
            return
        
        base_model = "Qwen/Qwen2-0.5B"
        for msg in inference_model.load_model(
            base_model=base_model,
            adapter_path=model_path,
            template="qwen"
        ):
            yield msg
    
    def unload_local_model():
        """卸载本地模型"""
        for msg in inference_model.unload_model():
            yield msg
    
    def get_rag_stats():
        """获取RAG服务统计信息"""
        return rag_service.get_stats()
    
    def process_rag_query(query: str, top_k: int, mode: str, retrieval_enabled_flag: bool, multi_step_flag: bool):
        """处理RAG查询（支持DashScope API和本地模型）"""
        if not query.strip():
            return (
                "请输入您的问题",
                "未处理",
                [],
                "",
                ""
            )
        
        # 根据模式选择推理方式
        if mode == "DashScope API":
            # 使用DashScope API
            result = rag_service.rag_query(
            query=query,
            top_k=top_k,
                model="qwen-plus",  # 使用通义千问
            retrieval_enabled=retrieval_enabled_flag,
            multi_step=multi_step_flag
        )
        else:
            # 使用本地模型
            if not inference_model.loaded:
                return (
                    "❌ 请先加载本地模型\n\n点击上方的「▶️ 加载模型」按钮",
                    "未处理",
                    [],
                    "",
                    ""
                )
            
            # 检索文档
            if retrieval_enabled_flag:
                docs = rag_service.index_service.search(query, top_k)
                # docs 是 List[Tuple[str, float, str]] 格式: (doc_id, score, reason/text)
                retrieved_docs = [(doc_id, score, text) for doc_id, score, text in docs]
                context = "\n\n".join([f"文档{i+1}: {text}" for i, (doc_id, score, text) in enumerate(docs)])
                
                # 构建带上下文的提示词
                prompt = f"""基于以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请说明无法根据提供的信息回答。

上下文信息：
{context}

用户问题：{query}

请给出详细的回答："""
            else:
                retrieved_docs = []
                context = ""
                prompt = query
            
            # 使用本地模型生成回答
            import time
            start_time = time.time()
            
            answer = inference_model.generate_once(
                prompt=prompt,
                temperature=0.7,
                max_new_tokens=512
            )
            
            processing_time = time.time() - start_time
            processing_info = f"""处理时间: {processing_time:.2f}秒
推理模式: 本地模型
检索文档数: {len(retrieved_docs)}"""
            
            # 构建检索结果表格
            retrieved_table = []
            for doc_id, score, content in retrieved_docs:
                truncated_content = content[:100] + "..." if len(content) > 100 else content
                retrieved_table.append([doc_id, f"{score:.4f}", truncated_content])
            
            return (
                answer,
                processing_info,
                retrieved_table,
                context,
                prompt
            )
        
        # 构建检索结果表格
        retrieved_table = []
        for doc_id, score, content in result.get("retrieved_docs", []):
            # 截断内容以适应表格显示
            truncated_content = content[:100] + "..." if len(content) > 100 else content
            retrieved_table.append([doc_id, f"{score:.4f}", truncated_content])
        
        # 构建处理信息
        processing_info = f"""处理时间: {result.get('processing_time', 0):.2f}秒
推理模式: {mode}
检索文档数: {len(result.get('retrieved_docs', []))}"""
        
        return (
            result.get("answer", "生成回答失败"),
            processing_info,
            retrieved_table,
            result.get("context", ""),
            result.get("prompt_sent", "")
        )
    
    # 绑定事件
    
    # 推理模式切换事件
    inference_mode.change(
        fn=toggle_model_box,
        inputs=[inference_mode],
        outputs=[local_model_box, model_status, local_model_dropdown]
    )
    
    # 刷新本地模型列表
    refresh_local_models_btn.click(
        fn=refresh_local_models,
        outputs=[local_model_dropdown]
    )
    
    # 本地模型加载/卸载事件
    load_model_btn.click(
        fn=load_local_model,
        inputs=[local_model_dropdown],
        outputs=[model_status]
    )
    
    unload_model_btn.click(
        fn=unload_local_model,
        outputs=[model_status]
    )
    
    # RAG查询事件
    rag_query_btn.click(
        fn=process_rag_query,
        inputs=[query_input, top_k_slider, inference_mode, retrieval_enabled, multi_step_enabled],
        outputs=[answer_output, processing_info, retrieved_docs, context_output, prompt_display]
    )
    
    # 页面加载时获取统计信息
    stats_display.value = get_rag_stats()
    
    return gr.Column() 