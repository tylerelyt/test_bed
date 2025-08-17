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
        
        return f"✅ 文档导入成功！\n导入文档数: {success_count}\n总文档数: {len(documents)}\n\n💡 提示：现在可以在「🕸️ 知识图谱」标签页中构建知识图谱！"
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
            
            # 知识图谱标签页
            with gr.Tab("🕸️ 知识图谱"):
                gr.Markdown("### 🧠 基于LLM的知识图谱构建")
                
                # 使用说明
                gr.HTML("""
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #155724;">💡 使用指南</h4>
                    <ol style="margin-bottom: 0;">
                        <li><strong>第一步</strong>：在"📄 文档管理"标签页中导入文档</li>
                        <li><strong>第二步</strong>：返回此页面，点击"🔨 构建知识图谱"开始构建</li>
                        <li><strong>第三步</strong>：等待NER处理完成（约2-5分钟），系统会自动保存图谱</li>
                        <li><strong>第四步</strong>：使用"🔍 实体搜索"找到感兴趣的实体</li>
                        <li><strong>第五步</strong>：使用"🔗 实体关系查询"查看实体的相关实体和关系</li>
                    </ol>
                </div>
                """)
                
                # 图谱构建部分
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🏗️ 图谱构建")
                        
                        # API配置（硬编码）
                        gr.Markdown("**配置信息：** 优先使用 qwen-plus (云端)，不可用时自动切换到 qwen2.5-coder:latest (本地Ollama)")
                        
                        with gr.Row():
                            build_kg_btn = gr.Button("🔨 构建知识图谱", variant="primary")
                            rebuild_kg_btn = gr.Button("🔄 重建知识图谱", variant="secondary")
                            clear_kg_btn = gr.Button("🗑️ 清空图谱", variant="stop")
                        
                        # 重建确认对话框
                        gr.Markdown("#### ⚠️ 重建须知")
                        gr.Markdown("""
                        **请注意：**
                        - 知识图谱会自动保存到磁盘 (`models/knowledge_graph.pkl`)
                        - NER处理很耗时，预计需要 **2-5分钟** (取决于文档数量和模型)
                        - 重建将**完全覆盖**现有图谱，无法恢复
                        - 建议在重建前先导出现有图谱作为备份
                        """)
                        
                        # 重建确认输入框
                        rebuild_confirm_input = gr.Textbox(
                            label="确认重建：请输入 'CONFIRM' 来确认重建操作",
                            placeholder="输入 CONFIRM 确认重建",
                            visible=False
                        )
                        
                        # 实际重建按钮
                        rebuild_confirm_btn = gr.Button(
                            "⚠️ 确认重建知识图谱", 
                            variant="stop",
                            visible=False
                        )
                            
                        kg_build_status = gr.Textbox(
                            label="构建状态",
                            value="点击按钮开始构建知识图谱",
                            lines=4,
                            interactive=False
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 图谱统计")
                        kg_stats_display = gr.JSON(label="知识图谱统计")
                        refresh_kg_stats_btn = gr.Button("📊 刷新统计", variant="secondary")
                
                # 实体搜索和查询
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🔍 实体搜索")
                        
                        with gr.Row():
                            entity_search_query = gr.Textbox(
                                label="搜索实体",
                                placeholder="输入实体名称或关键词"
                            )
                            entity_search_btn = gr.Button("🔍 搜索实体", variant="primary")
                            
                        entity_search_results = gr.Dataframe(
                            headers=["实体名称", "类型", "描述", "文档数量", "分数"],
                            label="搜索结果",
                            interactive=False
                        )
                
                # 实体关系查询
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🔗 实体关系查询")
                        
                        with gr.Row():
                            entity_query_input = gr.Textbox(
                                label="查询实体",
                                placeholder="输入要查询的实体名称"
                            )
                            entity_query_btn = gr.Button("🔗 查询关系", variant="primary")
                            
                        entity_query_results = gr.JSON(
                            label="实体关系信息"
                        )
                
                # 图谱导出
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 💾 图谱导出")
                        
                        with gr.Row():
                            export_kg_btn = gr.Button("💾 导出知识图谱", variant="secondary")
                            
                        kg_export_download = gr.File(label="下载知识图谱文件", interactive=False)
                        kg_export_status = gr.Textbox(
                            label="导出状态",
                            value="点击按钮导出知识图谱",
                            lines=2,
                            interactive=False
                        )
        
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
        
        # 知识图谱相关事件
        # 知识图谱构建函数（硬编码模型选择）
        def build_knowledge_graph():
            try:
                # 硬编码配置：优先使用 qwen-plus，失败则使用 ollama
                
                # 获取文档统计信息
                doc_stats = search_engine.get_stats()
                doc_count = doc_stats.get('total_documents', 0)
                
                status_msg = f"🔄 开始构建知识图谱...\n"
                status_msg += f"📊 文档数量: {doc_count}\n"
                status_msg += f"🔧 尝试使用 qwen-plus (云端)，失败时自动切换到 qwen2.5-coder:latest (本地)\n"
                estimated_time = doc_count * 20  # 每个文档约20秒
                status_msg += f"⏱️ 预计时间: {estimated_time}秒 (每个文档约10-30秒)\n"
                status_msg += f"📝 正在进行NER处理，请耐心等待...\n"
                
                # 先尝试 qwen-plus
                try:
                    search_engine.set_ner_api_config(
                        api_type="openai",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        default_model="qwen-plus"
                    )
                    result = search_engine.build_knowledge_graph("qwen-plus")
                    if "error" not in result:
                        success_msg = f"✅ {result['message']} (使用 qwen-plus)\n"
                        success_msg += f"⏱️ 实际构建时间: {result['build_time']:.2f}秒\n"
                        success_msg += f"💾 图谱已自动保存到 models/knowledge_graph.pkl"
                        return success_msg
                except Exception as e:
                    print(f"qwen-plus failed: {e}")
                
                # qwen-plus 失败，使用本地 ollama
                search_engine.set_ner_api_config(
                    api_type="ollama",
                    default_model="qwen2.5-coder:latest"
                )
                result = search_engine.build_knowledge_graph("qwen2.5-coder:latest")
                if "error" in result:
                    return f"❌ {result['error']}"
                else:
                    success_msg = f"✅ {result['message']} (使用 qwen2.5-coder:latest)\n"
                    success_msg += f"⏱️ 实际构建时间: {result['build_time']:.2f}秒\n"
                    success_msg += f"💾 图谱已自动保存到 models/knowledge_graph.pkl"
                    return success_msg
            except Exception as e:
                return f"❌ 构建知识图谱失败: {str(e)}"
        
        def show_rebuild_confirm():
            """显示重建确认界面"""
            return gr.update(visible=True), gr.update(visible=True), "请在下方输入 'CONFIRM' 确认重建操作"
        
        def rebuild_knowledge_graph(confirm_text):
            """重建知识图谱（需要确认，硬编码模型选择）"""
            if confirm_text.strip().upper() != "CONFIRM":
                return "❌ 请输入 'CONFIRM' 确认重建操作", gr.update(visible=False), gr.update(visible=False)
            
            try:
                # 获取当前图谱状态
                current_stats = search_engine.get_knowledge_graph_stats()
                
                status_msg = f"🔄 开始重建知识图谱...\n"
                status_msg += f"⚠️ 当前图谱: {current_stats.get('entity_count', 0)} 个实体, {current_stats.get('relation_count', 0)} 条关系\n"
                status_msg += f"🔧 尝试使用 qwen-plus (云端)，失败时自动切换到 qwen2.5-coder:latest (本地)\n"
                status_msg += f"📝 正在进行NER处理，预计需要2-5分钟，请耐心等待...\n"
                
                # 先尝试 qwen-plus
                try:
                    search_engine.set_ner_api_config(
                        api_type="openai",
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        default_model="qwen-plus"
                    )
                    result = search_engine.rebuild_knowledge_graph("qwen-plus")
                    if "error" not in result:
                        success_msg = f"✅ {result['message']} (使用 qwen-plus)\n⏱️ 重建时间: {result['build_time']:.2f}秒\n"
                        success_msg += f"💾 图谱已自动保存到磁盘"
                        return success_msg, gr.update(visible=False), gr.update(visible=False)
                except Exception as e:
                    print(f"qwen-plus rebuild failed: {e}")
                
                # qwen-plus 失败，使用本地 ollama
                search_engine.set_ner_api_config(
                    api_type="ollama",
                    default_model="qwen2.5-coder:latest"
                )
                result = search_engine.rebuild_knowledge_graph("qwen2.5-coder:latest")
                if "error" in result:
                    return f"❌ {result['error']}", gr.update(visible=False), gr.update(visible=False)
                else:
                    success_msg = f"✅ {result['message']} (使用 qwen2.5-coder:latest)\n⏱️ 重建时间: {result['build_time']:.2f}秒\n"
                    success_msg += f"💾 图谱已自动保存到磁盘"
                    return success_msg, gr.update(visible=False), gr.update(visible=False)
            except Exception as e:
                return f"❌ 重建知识图谱失败: {str(e)}", gr.update(visible=False), gr.update(visible=False)
        
        def clear_knowledge_graph():
            try:
                result = search_engine.clear_knowledge_graph()
                return f"✅ {result}"
            except Exception as e:
                return f"❌ 清空知识图谱失败: {str(e)}"
        
        def refresh_kg_stats():
            try:
                stats = search_engine.get_knowledge_graph_stats()
                
                # 添加持久化状态信息
                import os
                graph_file = "models/knowledge_graph.pkl"
                if os.path.exists(graph_file):
                    file_stats = os.stat(graph_file)
                    stats["persistence"] = {
                        "file_exists": True,
                        "file_path": graph_file,
                        "file_size_mb": round(file_stats.st_size / (1024*1024), 2),
                        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    stats["persistence"] = {
                        "file_exists": False,
                        "file_path": graph_file,
                        "message": "知识图谱文件不存在，需要先构建"
                    }
                
                return stats
            except Exception as e:
                return {"error": str(e)}
        
        def search_entities(query):
            if not query:
                return []
            
            try:
                results = search_engine.search_entities(query, limit=10)
                table_data = []
                for entity in results:
                    table_data.append([
                        entity.get("entity", ""),
                        entity.get("type", ""),
                        entity.get("description", "")[:100] + "..." if len(entity.get("description", "")) > 100 else entity.get("description", ""),
                        entity.get("doc_count", 0),
                        f"{entity.get('score', 0):.4f}"
                    ])
                return table_data
            except Exception as e:
                return [["错误", "N/A", str(e), "0", "0"]]
        
        def query_entity_relations(entity_name):
            if not entity_name:
                return {}
            
            try:
                results = search_engine.query_entity_relations(entity_name)
                return results
            except Exception as e:
                return {"error": str(e)}
        
        def export_knowledge_graph():
            try:
                filepath, message = search_engine.export_knowledge_graph()
                if filepath:
                    return gr.File(value=filepath, interactive=False), message
                else:
                    return None, message
            except Exception as e:
                return None, f"❌ 导出知识图谱失败: {str(e)}"
        
        # 绑定知识图谱事件
        build_kg_btn.click(
            fn=build_knowledge_graph,
            outputs=kg_build_status
        )
        

        
        # 重建知识图谱 - 分两步：先显示确认，后执行重建
        rebuild_kg_btn.click(
            fn=show_rebuild_confirm,
            outputs=[rebuild_confirm_input, rebuild_confirm_btn, kg_build_status]
        )
        
        rebuild_confirm_btn.click(
            fn=rebuild_knowledge_graph,
            inputs=rebuild_confirm_input,
            outputs=[kg_build_status, rebuild_confirm_input, rebuild_confirm_btn]
        )
        
        clear_kg_btn.click(
            fn=clear_knowledge_graph,
            outputs=kg_build_status
        )
        
        refresh_kg_stats_btn.click(
            fn=refresh_kg_stats,
            outputs=kg_stats_display
        )
        
        entity_search_btn.click(
            fn=search_entities,
            inputs=entity_search_query,
            outputs=entity_search_results
        )

        entity_query_btn.click(
            fn=query_entity_relations,
            inputs=entity_query_input,
            outputs=entity_query_results
        )
        
        export_kg_btn.click(
            fn=export_knowledge_graph,
            outputs=[kg_export_download, kg_export_status]
        )
        
        # 页面加载时自动刷新文档列表和知识图谱统计
        index_tab.load(
            fn=lambda: get_all_documents(search_engine),
            outputs=docs_list
        )
        
        index_tab.load(
            fn=refresh_kg_stats,
            outputs=kg_stats_display
        )
    
    return index_tab 