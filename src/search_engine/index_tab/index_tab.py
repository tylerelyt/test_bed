import gradio as gr
from datetime import datetime
import json
import os
import tempfile

def show_index_stats(search_engine):
    """显示索引统计信息"""
    try:
        stats = search_engine.get_stats()
        html_content = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4>📊 索引统计信息</h4>
            <ul>
                <li><strong>总文档数:</strong> {stats.get('total_documents', 0)}</li>
                <li><strong>总词项数:</strong> {stats.get('total_terms', 0)}</li>
                <li><strong>平均文档长度:</strong> {stats.get('average_doc_length', 0):.2f}</li>
            </ul>
            <p style="color: #6c757d; font-size: 0.9em;">统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        return html_content
    except Exception as e:
        return f"<p style='color: red;'>获取索引统计失败: {str(e)}</p>"

def check_index_quality(search_engine):
    """检查索引质量"""
    try:
        stats = search_engine.get_stats()
        total_docs = stats.get('total_documents', 0)
        total_terms = stats.get('total_terms', 0)
        avg_length = stats.get('average_doc_length', 0)
        
        issues = []
        recommendations = []
        
        if total_docs == 0:
            issues.append("索引中没有文档")
            recommendations.append("添加更多文档到索引")
        
        if total_terms <= 50:
            issues.append("词项数量较少")
            recommendations.append("增加文档多样性")
        
        if avg_length < 10:
            issues.append("文档平均长度过短")
            recommendations.append("增加文档内容长度")
        elif avg_length > 100:
            issues.append("文档平均长度过长")
            recommendations.append("考虑文档分段")
        
        html_content = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4>🔍 索引质量检查报告</h4>
            <h5>📈 质量指标:</h5>
            <ul>
                <li>文档数量: {total_docs} 个</li>
                <li>词项数量: {total_terms} 个</li>
                <li>平均文档长度: {avg_length:.2f} 个词</li>
            </ul>
        """
        
        if issues:
            html_content += f"""
            <h5>⚠️ 发现的问题:</h5>
            <ul style="color: #dc3545;">
                {''.join([f'<li>{issue}</li>' for issue in issues])}
            </ul>
            """
        
        if recommendations:
            html_content += f"""
            <h5>💡 改进建议:</h5>
            <ul style="color: #007bff;">
                {''.join([f'<li>{rec}</li>' for rec in recommendations])}
            </ul>
            """
        
        html_content += f"""
            <p style="color: #6c757d; font-size: 0.9em;">检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        return html_content
    except Exception as e:
        return f"<p style='color: red;'>索引质量检查失败: {str(e)}</p>"

def view_inverted_index(search_engine):
    """查看倒排索引内容"""
    try:
        index_service = search_engine.index_service
        # 直接访问底层InvertedIndex对象
        inverted_index = index_service.index.index
        # 取前20个词项
        items = list(inverted_index.items())[:20]
        data = [[term, ', '.join(list(doc_ids)[:10])] for term, doc_ids in items]
        return data
    except Exception as e:
        return [["错误", str(e)]]

def get_all_documents(search_engine):
    """获取所有文档列表"""
    try:
        documents = search_engine.get_all_documents()
        if not documents:
            return [["暂无文档", "请先导入文档文件"]]
        
        data = []
        for doc_id, content in documents.items():
            # 截取前100个字符作为预览
            preview = content[:100] + "..." if len(content) > 100 else content
            data.append([doc_id, preview])
        
        return data
    except Exception as e:
        return [["错误", str(e)]]

def export_documents(search_engine):
    """导出所有文档到JSON文件"""
    try:
        documents = search_engine.get_all_documents()
        if not documents:
            return None, "❌ 没有文档可导出"
        
        # 生成导出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f"documents_export_{timestamp}.json"
        
        # 导出文档数据
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_documents": len(documents),
            "documents": documents
        }
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
            json.dump(export_data, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        
        return tmp_path, f"✅ 文档导出成功！\n文档数量: {len(documents)}\n点击上方下载按钮获取文件"
    except Exception as e:
        return None, f"❌ 导出文档失败: {str(e)}"

def import_documents_from_file(search_engine, file):
    """从文件导入文档并更新索引"""
    try:
        if file is None:
            return "❌ 请选择要导入的文件"
        
        # 读取文件内容
        with open(file.name, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        # 验证文件格式
        if "documents" not in import_data:
            return "❌ 文件格式错误：缺少 'documents' 字段"
        
        documents = import_data["documents"]
        if not isinstance(documents, dict):
            return "❌ 文件格式错误：'documents' 应该是字典格式"
        
        if not documents:
            return "❌ 文件中没有文档数据"
        
        # 清空现有索引
        search_engine.clear_index()
        
        # 批量添加新文档
        success_count = search_engine.batch_add_documents(documents)
        
        # 保存更新后的索引
        search_engine.save_index()
        
        return f"✅ 文档导入成功！\n导入文档数: {success_count}\n总文档数: {len(documents)}"
    except Exception as e:
        return f"❌ 导入文档失败: {str(e)}"

def build_index_tab(search_engine):
    with gr.Blocks() as index_tab:
        gr.Markdown("""
        ### 🏗️ 第一部分：离线索引构建
        """)
        
        with gr.Tabs():
            # 索引信息标签页
            with gr.Tab("📊 索引信息"):
                with gr.Row():
                    with gr.Column(scale=2):
                        index_stats_btn = gr.Button("📊 查看索引统计", variant="primary")
                        index_stats_output = gr.HTML(value="<p>点击按钮查看索引统计信息...</p>", elem_id="index_stats_output")
                        index_quality_btn = gr.Button("🔍 索引质量检查", variant="secondary")
                        index_quality_output = gr.HTML(value="<p>点击按钮进行索引质量检查...</p>", elem_id="index_quality_output")
                        view_index_btn = gr.Button("📖 查看倒排索引", variant="secondary")
                        view_index_output = gr.Dataframe(headers=["词项", "文档ID列表"], label="倒排索引片段", interactive=False)
                    with gr.Column(scale=3):
                        gr.HTML("<p>索引构建详细信息...</p>")
            
            # 文档管理标签页
            with gr.Tab("📄 文档管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 文档列表")
                        refresh_docs_btn = gr.Button("🔄 刷新文档列表", variant="primary")
                        docs_list = gr.Dataframe(
                            headers=["文档ID", "内容预览"], 
                            label="所有文档", 
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 导出文档")
                        gr.HTML("<p style='color: #6c757d;'>导出所有文档到JSON文件，包含文档ID和内容</p>")
                        export_docs_btn = gr.Button("📤 导出所有文档", variant="primary")
                        export_download = gr.File(label="下载文档文件", interactive=False)
                        export_result = gr.Textbox(label="导出结果", interactive=False)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 导入文档")
                        gr.HTML("<p style='color: #6c757d;'>上传JSON文件导入文档，将替换现有索引</p>")
                        import_file = gr.File(
                            label="选择文档文件", 
                            file_types=[".json"],
                            file_count="single"
                        )
                        import_docs_btn = gr.Button("📥 导入文档并更新索引", variant="primary")
                        import_result = gr.Textbox(label="导入结果", interactive=False)
        
        # 绑定事件
        # 索引信息相关
        index_stats_btn.click(
            fn=lambda: show_index_stats(search_engine), 
            outputs=index_stats_output
        )
        index_quality_btn.click(
            fn=lambda: check_index_quality(search_engine), 
            outputs=index_quality_output
        )
        view_index_btn.click(
            fn=lambda: view_inverted_index(search_engine), 
            outputs=view_index_output
        )
        
        # 文档管理相关
        refresh_docs_btn.click(
            fn=lambda: get_all_documents(search_engine),
            outputs=docs_list
        )
        
        export_docs_btn.click(
            fn=lambda: export_documents(search_engine),
            outputs=[export_download, export_result]
        )
        
        import_docs_btn.click(
            fn=lambda file: import_documents_from_file(search_engine, file),
            inputs=import_file,
            outputs=import_result
        )
        
        # 页面加载时自动刷新文档列表
        index_tab.load(
            fn=lambda: get_all_documents(search_engine),
            outputs=docs_list
        )
    
    return index_tab 