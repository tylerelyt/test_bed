#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片检索页面 - 基于CLIP的图搜图和文搜图界面
"""

import gradio as gr
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

def upload_and_add_image(image_service, image_file, description="", tags=""):
    """上传并添加图片到索引"""
    try:
        if image_file is None:
            return "❌ 请选择要上传的图片", None, []
        
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # 添加图片到索引
        image_id = image_service.add_image(
            image_path=image_file.name,
            description=description,
            tags=tag_list
        )
        
        # 获取图片信息用于预览
        image_info = image_service.get_image_info(image_id)
        
        # 刷新图片列表
        all_images = get_all_images_list(image_service)
        
        return f"✅ 图片上传成功！\nID: {image_id}\n描述: {description}\n标签: {', '.join(tag_list)}", image_file, all_images
        
    except Exception as e:
        return f"❌ 上传图片失败: {str(e)}", None, []

def search_images_by_image(image_service, query_image, top_k=10):
    """图搜图功能"""
    try:
        if query_image is None:
            return [], "❌ 请选择要搜索的图片"
        
        # 执行图搜图
        results = image_service.search_by_image(query_image.name, top_k=top_k)
        
        if not results:
            return [], "🔍 没有找到相似的图片"
        
        # 格式化结果
        formatted_results = []
        gallery_images = []
        
        for result in results:
            similarity_score = f"{result['similarity']:.4f}"
            formatted_results.append([
                result['original_name'],
                result['description'] or "无描述",
                ', '.join(result['tags']) or "无标签",
                f"{result['width']}x{result['height']}",
                similarity_score,
                result['id']
            ])
            
            # 添加到图片画廊
            if os.path.exists(result['stored_path']):
                gallery_images.append(result['stored_path'])
        
        status_msg = f"🎯 找到 {len(results)} 张相似图片，相似度分数范围: {results[-1]['similarity']:.4f} - {results[0]['similarity']:.4f}"
        
        return formatted_results, status_msg, gallery_images
        
    except Exception as e:
        return [], f"❌ 图搜图失败: {str(e)}", []

def search_images_by_text(image_service, query_text, top_k=10):
    """文搜图功能"""
    try:
        if not query_text.strip():
            return [], "❌ 请输入搜索文本"
        
        # 执行文搜图
        results = image_service.search_by_text(query_text, top_k=top_k)
        
        if not results:
            return [], "🔍 没有找到匹配的图片"
        
        # 格式化结果
        formatted_results = []
        gallery_images = []
        
        for result in results:
            similarity_score = f"{result['similarity']:.4f}"
            formatted_results.append([
                result['original_name'],
                result['description'] or "无描述",
                ', '.join(result['tags']) or "无标签",
                f"{result['width']}x{result['height']}",
                similarity_score,
                result['id']
            ])
            
            # 添加到图片画廊
            if os.path.exists(result['stored_path']):
                gallery_images.append(result['stored_path'])
        
        status_msg = f"🎯 找到 {len(results)} 张匹配图片，相似度分数范围: {results[-1]['similarity']:.4f} - {results[0]['similarity']:.4f}"
        
        return formatted_results, status_msg, gallery_images
        
    except Exception as e:
        return [], f"❌ 文搜图失败: {str(e)}", []

def get_all_images_list(image_service):
    """获取所有图片列表"""
    try:
        all_images = image_service.get_all_images()
        
        if not all_images:
            return []
        
        # 按创建时间排序
        all_images.sort(key=lambda x: x['created_at'], reverse=True)
        
        formatted_list = []
        for image_info in all_images:
            file_size_mb = round(image_info['file_size'] / (1024 * 1024), 2)
            formatted_list.append([
                image_info['original_name'],
                image_info['description'] or "无描述",
                ', '.join(image_info['tags']) or "无标签",
                f"{image_info['width']}x{image_info['height']}",
                f"{file_size_mb} MB",
                image_info['created_at'][:16].replace('T', ' '),
                image_info['id']
            ])
        
        return formatted_list
        
    except Exception as e:
        print(f"❌ 获取图片列表失败: {e}")
        return []

def get_image_stats(image_service):
    """获取图片统计信息"""
    try:
        stats = image_service.get_stats()
        
        formats_str = ", ".join([f"{fmt}({count})" for fmt, count in stats['formats'].items()]) if stats['formats'] else "无"
        
        html_content = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4>📊 图片库统计信息</h4>
            <ul>
                <li><strong>图片总数:</strong> {stats['total_images']} 张</li>
                <li><strong>存储大小:</strong> {stats['total_size_mb']} MB</li>
                <li><strong>图片格式:</strong> {formats_str}</li>
                <li><strong>嵌入维度:</strong> {stats['embedding_dimension']}</li>
                <li><strong>计算设备:</strong> {stats['model_device']}</li>
                <li><strong>存储目录:</strong> {stats['storage_dir']}</li>
            </ul>
            <p style="color: #6c757d; font-size: 0.9em;">统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        return html_content
        
    except Exception as e:
        return f"<p style='color: red;'>获取统计信息失败: {str(e)}</p>"

def delete_selected_image(image_service, selected_data):
    """删除选中的图片"""
    try:
        if not selected_data:
            return "❌ 请在图片列表中选择要删除的图片", []
        
        # 获取选中的图片ID（最后一列）
        image_id = selected_data[-1]
        
        # 删除图片
        success = image_service.delete_image(image_id)
        
        if success:
            # 刷新图片列表
            updated_list = get_all_images_list(image_service)
            return f"✅ 图片删除成功: {image_id}", updated_list
        else:
            return f"❌ 图片删除失败: {image_id}", []
            
    except Exception as e:
        return f"❌ 删除图片失败: {str(e)}", []

def clear_all_images(image_service):
    """清空所有图片"""
    try:
        image_service.clear_index()
        return "✅ 所有图片已清空", []
    except Exception as e:
        return f"❌ 清空失败: {str(e)}", []

def build_image_tab(image_service):
    """构建图片检索页面"""
    
    with gr.Blocks() as image_tab:
        gr.Markdown("""
        ### 🖼️ 图片检索系统 - 基于CLIP模型
        
        支持图片上传、图搜图、文搜图功能。使用OpenAI CLIP模型进行图片和文本的语义理解。
        """)
        
        with gr.Tabs():
            # 图片上传标签页
            with gr.Tab("📤 图片上传"):
                gr.Markdown("#### 上传图片到图片库")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.File(
                            label="选择图片文件",
                            file_types=["image"],
                            file_count="single"
                        )
                        
                        image_description = gr.Textbox(
                            label="图片描述",
                            placeholder="请输入图片的描述信息...",
                            lines=3
                        )
                        
                        image_tags = gr.Textbox(
                            label="图片标签",
                            placeholder="输入标签，用逗号分隔，如：动物,猫,宠物",
                            lines=1
                        )
                        
                        upload_btn = gr.Button("📤 上传图片", variant="primary")
                        upload_status = gr.Textbox(
                            label="上传状态",
                            lines=4,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 图片预览")
                        image_preview = gr.Image(
                            label="图片预览",
                            height=300
                        )
            
            # 图搜图标签页
            with gr.Tab("🔍 图搜图"):
                gr.Markdown("#### 使用图片搜索相似图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.File(
                            label="选择查询图片",
                            file_types=["image"],
                            file_count="single"
                        )
                        
                        image_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="返回结果数量"
                        )
                        
                        image_search_btn = gr.Button("🔍 图搜图", variant="primary")
                        
                        image_search_status = gr.Textbox(
                            label="搜索状态",
                            lines=2,
                            interactive=False
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### 搜索结果")
                        image_search_results = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "相似度", "ID"],
                            label="相似图片列表",
                            interactive=False
                        )
                        
                # 结果图片画廊
                image_gallery = gr.Gallery(
                    label="相似图片画廊",
                    show_label=True,
                    elem_id="image_gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
            
            # 文搜图标签页
            with gr.Tab("💬 文搜图"):
                gr.Markdown("#### 使用文本描述搜索图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_query = gr.Textbox(
                            label="搜索文本",
                            placeholder="输入描述性文本，如：一只橙色的猫在睡觉",
                            lines=3
                        )
                        
                        text_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="返回结果数量"
                        )
                        
                        text_search_btn = gr.Button("💬 文搜图", variant="primary")
                        
                        text_search_status = gr.Textbox(
                            label="搜索状态",
                            lines=2,
                            interactive=False
                        )
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### 搜索结果")
                        text_search_results = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "相似度", "ID"],
                            label="匹配图片列表",
                            interactive=False
                        )
                
                # 结果图片画廊
                text_gallery = gr.Gallery(
                    label="匹配图片画廊",
                    show_label=True,
                    elem_id="text_gallery",
                    columns=4,
                    rows=2,
                    height="auto"
                )
            
            # 图片管理标签页
            with gr.Tab("📋 图片管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 图片库统计")
                        stats_btn = gr.Button("📊 刷新统计", variant="secondary")
                        stats_display = gr.HTML(value="<p>点击按钮查看统计信息...</p>")
                        
                        gr.Markdown("#### 图片库列表")
                        refresh_list_btn = gr.Button("🔄 刷新列表", variant="secondary")
                        
                        images_list = gr.Dataframe(
                            headers=["图片名称", "描述", "标签", "尺寸", "大小", "创建时间", "ID"],
                            label="所有图片",
                            interactive=False
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### 图片操作")
                        
                        delete_btn = gr.Button("🗑️ 删除选中图片", variant="stop")
                        clear_all_btn = gr.Button("🗑️ 清空所有图片", variant="stop")
                        
                        operation_status = gr.Textbox(
                            label="操作状态",
                            lines=3,
                            interactive=False
                        )
        
        # 绑定事件处理函数
        
        # 图片上传
        upload_btn.click(
            fn=lambda img, desc, tags: upload_and_add_image(image_service, img, desc, tags),
            inputs=[upload_image, image_description, image_tags],
            outputs=[upload_status, image_preview, images_list]
        )
        
        # 图搜图
        image_search_btn.click(
            fn=lambda img, k: search_images_by_image(image_service, img, k),
            inputs=[query_image, image_top_k],
            outputs=[image_search_results, image_search_status, image_gallery]
        )
        
        # 文搜图
        text_search_btn.click(
            fn=lambda text, k: search_images_by_text(image_service, text, k),
            inputs=[text_query, text_top_k],
            outputs=[text_search_results, text_search_status, text_gallery]
        )
        
        # 统计信息
        stats_btn.click(
            fn=lambda: get_image_stats(image_service),
            outputs=stats_display
        )
        
        # 刷新图片列表
        refresh_list_btn.click(
            fn=lambda: get_all_images_list(image_service),
            outputs=images_list
        )
        
        # 删除图片
        delete_btn.click(
            fn=lambda data: delete_selected_image(image_service, data),
            inputs=images_list,
            outputs=[operation_status, images_list]
        )
        
        # 清空所有图片
        clear_all_btn.click(
            fn=lambda: clear_all_images(image_service),
            outputs=[operation_status, images_list]
        )
        
        # 页面加载时自动刷新统计和列表
        image_tab.load(
            fn=lambda: get_image_stats(image_service),
            outputs=stats_display
        )
        
        image_tab.load(
            fn=lambda: get_all_images_list(image_service),
            outputs=images_list
        )
    
    return image_tab
