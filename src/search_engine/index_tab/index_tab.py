import gradio as gr
from datetime import datetime
import json
import os
import tempfile
import html
import pandas as pd

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
    """查看倒排索引内容（返回 pandas DataFrame，与 Gradio 4 gr.Dataframe 稳定兼容）。"""
    _h = ["词项", "文档ID列表"]
    try:
        index_service = search_engine.index_service
        inv = getattr(index_service, "index", None)
        if inv is None or not hasattr(inv, "index"):
            return pd.DataFrame(
                [["—", "当前为 Elasticsearch 全文索引，无内存倒排表；请使用检索或直连 ES 查看。"]],
                columns=_h,
            )
        # 直接访问底层InvertedIndex对象
        inverted_index = inv.index
        # 取前20个词项
        items = list(inverted_index.items())[:20]
        data = [[term, ', '.join(list(doc_ids)[:10])] for term, doc_ids in items]
        return pd.DataFrame(data, columns=_h) if data else pd.DataFrame(columns=_h)
    except Exception as e:
        return pd.DataFrame([["错误", str(e)[:500]]], columns=_h)

def get_all_documents(search_engine):
    """获取所有文档列表（pandas，供 gr.Dataframe）。"""
    _h = ["文档ID", "内容预览"]
    try:
        documents = search_engine.get_all_documents()
        if not documents:
            return pd.DataFrame(
                [["暂无文档", "请先合并预置或检查 models/index_data.json。"]],
                columns=_h,
            )
        data = []
        for doc_id, content in documents.items():
            preview = (content or "")[:100] + "..." if len((content or "")) > 100 else (content or "")
            data.append([str(doc_id), str(preview)])
        return pd.DataFrame(data, columns=_h) if data else pd.DataFrame(columns=_h)
    except Exception as e:
        return pd.DataFrame([["错误", str(e)[:500]]], columns=_h)

# 文档导入导出功能已禁用

def build_index_tab(search_engine):
    _default_focus_entity = "星际穿越"
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

            # 文档信息标签页
            with gr.Tab("📚 文档信息"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📋 文档列表")
                        gr.HTML("<p style='color: #28a745;'>系统包含50条中文维基百科文档，仅供只读使用。</p>")
                        refresh_docs_btn = gr.Button("🔄 查看文档", variant="primary")
                        docs_list = gr.Dataframe(
                            headers=["文档ID", "内容预览"],
                            label="文档（只读）",
                            interactive=False,
                            wrap=True
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 文档信息")
                        gr.HTML("""
                        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                            <h4>📚 文档信息</h4>
                            <ul>
                                <li><strong>数量:</strong> 50条中文维基百科文档</li>
                                <li><strong>来源:</strong> Hugging Face fjcanyue/wikipedia-zh-cn 数据集</li>
                                <li><strong>状态:</strong> 只读</li>
                                <li><strong>功能:</strong> 支持搜索、RAG问答、知识图谱构建</li>
                            </ul>
                        </div>
                        """)


            # 知识图谱标签页
            with gr.Tab("🕸️ 知识图谱"):
                gr.Markdown("### 🧠 知识图谱（JanusGraph）")

                # 使用说明：与当前架构一致，主存为 Gremlin/Janus
                gr.HTML("""
                <div style="background-color: #e8f4fc; padding: 15px; border-radius: 8px; border-left: 4px solid #0b6a9b; margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #0c4a6e;">架构说明</h4>
                    <ul style="margin-bottom: 0; line-height: 1.6;">
                        <li><strong>主存</strong>：图数据在 <strong>JanusGraph</strong>（经 Gremlin Server 持久化），<em>不</em>以本地 pkl/进程内图为权威。</li>
                        <li><strong>预置</strong>：可放置 <code>data/openkg_triples.tsv</code> 与/或 <code>data/preloaded_knowledge_graph.json</code>，在空头图时自动补种，或点击「从预置重新加载」全量重灌（仅清业务 <code>entity</code> 标签顶点）。</li>
                        <li><strong>操作策略</strong>：本页默认只读分析，避免线上图谱被误修改；如需重置，使用「从预置重新加载」。</li>
                    </ul>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        kg_build_status = gr.Textbox(
                            label="连接与摘要",
                            value="加载后自动刷新。含 Gremlin 地址与统计摘要。",
                            lines=3,
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        refresh_kg_stats_btn = gr.Button("📊 刷新统计", variant="secondary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Graph Workspace")

                        gr.HTML("""
                        <div style="background:linear-gradient(180deg,#fff7ed 0%,#ffffff 100%);color:#0f172a;padding:18px 20px;border-radius:12px;margin-bottom:16px;border:1px solid #fed7aa;box-shadow:0 1px 2px rgba(15,23,42,.04);">
                            <div style="font-size:12px;letter-spacing:0;text-transform:uppercase;color:#c2410c;margin-bottom:8px;font-weight:700;">Knowledge Graph Console</div>
                            <div style="font-size:24px;font-weight:700;line-height:1.2;margin-bottom:8px;color:#111827;">面向分析与维护的图谱工作台</div>
                            <div style="font-size:14px;line-height:1.6;color:#334155;">聚焦查询、筛选、邻居展开和关系审阅。下面的概览区会基于当前筛选条件生成可读子图，而不是只堆原始 JSON。</div>
                        </div>
                        """)

                        with gr.Row():
                            kg_focus_query = gr.Textbox(
                                label="聚焦实体 / 关键词",
                                placeholder="例如：OpenAI、移动应用、Portfolio::default",
                                value=_default_focus_entity,
                            )
                            kg_type_filter = gr.Textbox(
                                label="实体类型过滤",
                                placeholder="例如：公司、未分类；留空表示全部"
                            )
                            kg_relation_filter = gr.Textbox(
                                label="关系谓词过滤",
                                placeholder="例如：属于、投资；留空表示全部"
                            )

                        with gr.Row():
                            kg_max_nodes = gr.Slider(
                                minimum=10,
                                maximum=120,
                                value=36,
                                step=2,
                                label="子图节点上限"
                            )
                            kg_workspace_btn = gr.Button("刷新工作台", variant="primary")

                            kg_workspace_summary = gr.HTML(
                            label="工作台摘要",
                            value="<div style='padding:16px;border:1px solid #e2e8f0;border-radius:12px;background:#fff;color:#64748b;'>加载后会显示图谱概览、类型分布、邻接密度和当前筛选命中情况。</div>"
                        )

                        with gr.Accordion("搜索候选实体", open=False):
                            with gr.Row():
                                entity_search_query = gr.Textbox(
                                    label="搜索实体",
                                    placeholder="输入实体名称或关键词"
                                )
                                entity_search_btn = gr.Button("🔍 搜索实体", variant="primary")

                            entity_search_results = gr.Dataframe(
                                headers=["实体名称", "类型", "描述", "文档数量", "分数"],
                                label="搜索结果",
                                interactive=False,
                                wrap=True,
                            )

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("#### 🌐 子图概览")
                        kg_graph_plot = gr.HTML(
                            label="Knowledge Graph Overview",
                            value=(
                                "<div style='padding:18px;border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;color:#64748b;'>"
                                "点击「刷新工作台」后渲染交互子图。支持缩放、拖拽、悬浮查看节点/关系详情。"
                                "</div>"
                            ),
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📋 子图清单")
                        kg_node_inventory = gr.Dataframe(
                            headers=["实体", "类型", "文档数", "连接数", "说明"],
                            label="节点清单",
                            interactive=False,
                            wrap=True,
                        )
                        kg_edge_inventory = gr.Dataframe(
                            headers=["主语", "谓词", "宾语", "说明"],
                            label="关系清单",
                            interactive=False,
                            wrap=True,
                        )

                # 实体关系查询
                with gr.Row():
                    with gr.Column(scale=2):
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

                    with gr.Column(scale=3):
                        gr.Markdown("#### 🧭 实体详情面板")
                        entity_graph_viz = gr.HTML(
                            label="关系图",
                            value=(
                                "<div style='padding:18px;border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;color:#64748b;'>"
                                "查询实体后，这里会展示实体摘要、入/出边面板和可读的分析提示。"
                                "</div>"
                            ),
                        )

                with gr.Accordion("NER 三元组审阅入库", open=False):
                    gr.Markdown(
                        "输入文本后先抽取候选，再按候选ID删除不需要的行，"
                        "最后将剩余候选批量写入 JanusGraph（仅可入库=true 生效）。"
                    )
                    with gr.Row():
                        ner_input_text = gr.Textbox(
                            label="待抽取文本",
                            lines=6,
                            placeholder="粘贴一段业务文本或新闻摘要...",
                        )
                    with gr.Row():
                        ner_model_text = gr.Textbox(
                            label="模型（可空）",
                            placeholder="留空使用默认模型",
                        )
                        ner_extract_btn = gr.Button("抽取候选三元组", variant="primary")
                        ner_insert_btn = gr.Button("写入当前候选三元组", variant="secondary")
                    with gr.Row():
                        ner_delete_ids_text = gr.Textbox(
                            label="删除候选ID（逗号分隔）",
                            placeholder="示例: 2,5,8",
                        )
                        ner_delete_btn = gr.Button("删除指定候选行", variant="secondary")
                    ner_status = gr.Textbox(
                        label="NER 处理结果",
                        value="先抽取，再删除不需要的候选，最后执行写入。",
                        lines=2,
                        interactive=False,
                    )
                    ner_candidate_state = gr.State([])
                    ner_triples_df = gr.Dataframe(
                        headers=["候选ID", "可入库", "主语", "谓词", "宾语", "主语类型", "宾语类型", "说明", "原因", "原始谓词"],
                        datatype=["str", "bool", "str", "str", "str", "str", "str", "str", "str", "str"],
                        label="候选三元组（预览）",
                        interactive=False,
                        wrap=True,
                        type="array",
                    )

                with gr.Accordion("运维操作", open=False):
                    gr.Markdown("#### 图数据重载（只读模式下保留）")
                    with gr.Row():
                        reload_kg_btn = gr.Button("从预置重新加载", variant="primary", size="sm")
                    kg_modify_status = gr.Textbox(
                        label="重载结果",
                        value="将删除 Janus 中 label=entity 的顶点，本体等其它标签不受影响；重载时按预置 TSV 优先，否则 JSON。",
                        lines=2,
                        interactive=False
                    )

                with gr.Accordion("原始统计与配置", open=False):
                    gr.Markdown("这里保留原始 JSON 和预置说明，方便排查问题；日常分析时可以一直折叠。")
                    gr.HTML("""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                        <h5 style="margin-top: 0;">预置与工具</h5>
                        <p>• 三元组：<code>data/openkg_triples.tsv</code></p>
                        <p>• 完整 JSON 预置（可选）：<code>data/preloaded_knowledge_graph.json</code></p>
                        <p>• OpenKG 生成/更新见仓库内 <code>tools/openkg_generator.py</code>（如适用）</p>
                    </div>
                    """)
                    kg_stats_display = gr.JSON(label="知识图谱统计")

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

        # 文档操作功能已禁用

        def format_kg_status_line() -> str:
            """与统计 JSON 同步的简短人读摘要，用于「连接与摘要」。"""
            try:
                s = search_engine.get_knowledge_graph_stats()
                ec = int(s.get("entity_count", 0) or 0)
                rc = int(s.get("relation_count", 0) or 0)
                gurl = s.get("gremlin_url", "(未知)")
                gf = s.get("graph_file", "data/knowledge_graph_janusgraph_export.json")
                built = bool(s.get("is_graph_built"))
                y = "是" if built else "否"
                return (
                    f"业务实体: {ec} 条 · 关系: {rc} 条 · 可检索: {y}\n"
                    f"Gremlin: {gurl}\n"
                    f"默认 JSON 备份: {gf}"
                )
            except Exception as e:
                return f"无法获取知识图谱状态: {e}"

        def do_reload_knowledge_graph():
            try:
                result = search_engine.reload_knowledge_graph()
                if result.get("success"):
                    msg = f"✅ {result.get('message', '')} 实体: {result.get('entity_count', 0)}, 关系: {result.get('relation_count', 0)}"
                else:
                    msg = f"❌ {result.get('error', '重新加载失败')}"
                return msg, refresh_kg_stats(), format_kg_status_line()
            except Exception as e:
                return f"❌ 重新加载失败: {str(e)}", refresh_kg_stats(), format_kg_status_line()

        def refresh_kg_stats():
            try:
                stats = search_engine.get_knowledge_graph_stats()

                # 持久化说明：当前生产路径为 JanusGraph，主存服务端；pkl 仅为历史/兼容
                if stats.get("graph_backend") == "janusgraph":
                    export_path = stats.get("graph_file") or "data/knowledge_graph_janusgraph_export.json"
                    openkg = os.path.join("data", "openkg_triples.tsv")
                    pjson = os.path.join("data", "preloaded_knowledge_graph.json")
                    stats["persistence"] = {
                        "mode": "janusgraph",
                        "message": "主存在 JanusGraph；本页默认只读分析；重载来自预置 TSV/JSON。",
                        "default_backup_json": export_path,
                        "default_backup_json_exists": os.path.isfile(export_path),
                        "openkg_preload_path": openkg,
                        "openkg_preload_exists": os.path.isfile(openkg),
                        "preloaded_kg_json_path": pjson,
                        "preloaded_kg_json_exists": os.path.isfile(pjson),
                    }
                else:
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

        _entity_df_cols = ["实体名称", "类型", "描述", "文档数量", "分数"]
        _kg_node_cols = ["实体", "类型", "文档数", "连接数", "说明"]
        _kg_edge_cols = ["主语", "谓词", "宾语", "说明"]

        def _safe_int(value, default=0):
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return default

        def _safe_text(value):
            return str(value or "").strip()

        def _truncate_text(value, limit=80):
            text = _safe_text(value)
            return text if len(text) <= limit else text[:limit] + "..."

        def _build_kg_workspace_summary(stats, nodes, edges, focus_query, type_filter, relation_filter):
            type_counts = {}
            degree_map = {node.get("id", ""): 0 for node in nodes}
            for edge in edges:
                degree_map[edge.get("source", "")] = degree_map.get(edge.get("source", ""), 0) + 1
                degree_map[edge.get("target", "")] = degree_map.get(edge.get("target", ""), 0) + 1
            for node in nodes:
                t = _safe_text(node.get("type")) or "未分类"
                type_counts[t] = type_counts.get(t, 0) + 1
            top_types = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
            top_nodes = sorted(
                [(node.get("label", node.get("id", "")), degree_map.get(node.get("id", ""), 0)) for node in nodes],
                key=lambda item: (-item[1], item[0]),
            )[:5]
            density = 0.0
            n = len(nodes)
            if n > 1:
                density = len(edges) / float(n * (n - 1))

            type_chips = "".join(
                f"<span style='display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;background:#eff6ff;color:#1d4ed8;font-size:12px;font-weight:600;'>{html.escape(k)} {v}</span>"
                for k, v in top_types
            ) or "<span style='color:#94a3b8;'>暂无类型分布</span>"
            top_node_rows = "".join(
                f"<tr><td style='padding:6px 0;color:#0f172a;'>{html.escape(name)}</td><td style='padding:6px 0;text-align:right;color:#475569;'>{degree}</td></tr>"
                for name, degree in top_nodes
            ) or "<tr><td colspan='2' style='padding:8px 0;color:#94a3b8;'>暂无节点</td></tr>"

            return f"""
            <div style="border:1px solid #e2e8f0;border-radius:14px;background:#fff;padding:18px;">
              <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:12px;color:#64748b;">实体数</div>
                  <div style="font-size:28px;font-weight:700;color:#0f172a;">{len(nodes)}</div>
                  <div style="font-size:12px;color:#94a3b8;">全图 {int(stats.get('entity_count', 0) or 0)} 个业务实体</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:12px;color:#64748b;">关系数</div>
                  <div style="font-size:28px;font-weight:700;color:#0f172a;">{len(edges)}</div>
                  <div style="font-size:12px;color:#94a3b8;">全图 {int(stats.get('relation_count', 0) or 0)} 条关系</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:12px;color:#64748b;">子图密度</div>
                  <div style="font-size:28px;font-weight:700;color:#0f172a;">{density:.3f}</div>
                  <div style="font-size:12px;color:#94a3b8;">越高表示连接越集中</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:12px;color:#64748b;">当前聚焦</div>
                  <div style="font-size:16px;font-weight:700;color:#0f172a;word-break:break-word;">{html.escape(focus_query or '全图概览')}</div>
                  <div style="font-size:12px;color:#94a3b8;">类型：{html.escape(type_filter or '全部')} / 谓词：{html.escape(relation_filter or '全部')}</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:16px;">
                <div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:13px;font-weight:700;color:#334155;margin-bottom:10px;">类型分布</div>
                  <div style="display:flex;gap:8px;flex-wrap:wrap;">{type_chips}</div>
                </div>
                <div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px;">
                  <div style="font-size:13px;font-weight:700;color:#334155;margin-bottom:10px;">连接度最高的节点</div>
                  <table style="width:100%;font-size:13px;border-collapse:collapse;">
                    <thead><tr><th style='text-align:left;padding-bottom:6px;color:#94a3b8;font-weight:600;'>实体</th><th style='text-align:right;padding-bottom:6px;color:#94a3b8;font-weight:600;'>连接数</th></tr></thead>
                    <tbody>{top_node_rows}</tbody>
                  </table>
                </div>
              </div>
            </div>
            """

        def _build_kg_plot(nodes, edges, focus_query):
            if not nodes:
                return (
                    "<div style='padding:18px;border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;color:#64748b;'>"
                    "当前筛选条件下没有可渲染节点。请放宽筛选条件或清空聚焦关键词。"
                    "</div>"
                )

            palette = [
                "#2563eb", "#7c3aed", "#db2777", "#0891b2", "#059669",
                "#d97706", "#dc2626", "#4f46e5", "#0f766e", "#9333ea",
            ]
            type_colors = {}
            focus_tokens = [token.lower() for token in _safe_text(focus_query).split() if token.strip()]

            vis_nodes = []
            for node in nodes:
                node_id = _safe_text(node.get("id"))
                label = _safe_text(node.get("label") or node_id)
                node_type = _safe_text(node.get("type")) or "未分类"
                if node_type not in type_colors:
                    type_colors[node_type] = palette[len(type_colors) % len(palette)]
                is_focus = bool(focus_tokens) and any(token in label.lower() for token in focus_tokens)
                node_color = "#ef4444" if is_focus else type_colors[node_type]
                degree_hint = _safe_int(node.get("degree"), 0)
                doc_count = _safe_int(node.get("doc_count"), 0)
                size = max(16, min(46, 18 + degree_hint * 1.4 + doc_count * 0.8))
                title = (
                    f"节点: {html.escape(label)}<br/>"
                    f"类型: {html.escape(node_type)}<br/>"
                    f"文档数: {doc_count}<br/>"
                    f"说明: {html.escape(_truncate_text(node.get('description'), 220))}"
                )
                vis_nodes.append(
                    {
                        "id": node_id,
                        "label": _truncate_text(label, 22),
                        "title": title,
                        "shape": "dot",
                        "size": size,
                        "color": {"background": node_color, "border": "#0f172a"},
                        "font": {"color": "#111827", "size": 13},
                    }
                )

            vis_edges = []
            for edge in edges:
                source = _safe_text(edge.get("source"))
                target = _safe_text(edge.get("target"))
                predicate = _safe_text(edge.get("predicate"))
                if not source or not target:
                    continue
                vis_edges.append(
                    {
                        "from": source,
                        "to": target,
                        "label": _truncate_text(predicate, 16),
                        "title": html.escape(
                            f"{source} --[{predicate}]--> {target}\n{_safe_text(edge.get('description'))}"
                        ),
                        "arrows": "to",
                        "color": {"color": "#94a3b8", "highlight": "#ef4444"},
                        "font": {"align": "middle", "size": 11, "strokeWidth": 0},
                        "smooth": {"enabled": True, "type": "dynamic"},
                    }
                )

            nodes_json = json.dumps(vis_nodes, ensure_ascii=False).replace("</", "<\\/")
            edges_json = json.dumps(vis_edges, ensure_ascii=False).replace("</", "<\\/")
            graph_id = f"kg_vis_{int(datetime.now().timestamp() * 1000)}"
            title_text = html.escape(focus_query or "全图概览")
            legend_html = "".join(
                (
                    f"<span class='chip'><span class='dot' style='background:{color}'></span>"
                    f"{html.escape(label)}</span>"
                )
                for label, color in list(type_colors.items())[:8]
            )

            srcdoc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      background: #ffffff;
      color: #0f172a;
    }}
    .wrap {{
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid #e2e8f0;
      background: #f8fafc;
      flex-wrap: wrap;
    }}
    .legend {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border: 1px solid #e2e8f0;
      border-radius: 999px;
      background: #fff;
      font-size: 12px;
      color: #334155;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    #{graph_id} {{
      height: 560px;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <div>
        <div style="font-size:13px;font-weight:700;color:#0f172a;">知识图谱交互视图</div>
        <div style="font-size:12px;color:#64748b;">当前聚焦：{title_text} · 拖拽、缩放、悬停查看详情</div>
      </div>
      <div class="legend">{legend_html}</div>
    </div>
    <div id="{graph_id}"></div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script>
    const start = () => {{
      if (!window.vis) {{
        setTimeout(start, 120);
        return;
      }}
      const container = document.getElementById("{graph_id}");
      if (!container) return;
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const data = {{ nodes, edges }};
      const options = {{
        autoResize: true,
        interaction: {{
          hover: true,
          tooltipDelay: 120,
          navigationButtons: true,
          keyboard: true
        }},
        layout: {{
          improvedLayout: true
        }},
        physics: {{
          enabled: true,
          stabilization: {{ enabled: true, iterations: 180 }},
          barnesHut: {{
            gravitationalConstant: -5200,
            centralGravity: 0.18,
            springLength: 145,
            springConstant: 0.035,
            damping: 0.12
          }}
        }},
        edges: {{
          selectionWidth: 2.5
        }},
        nodes: {{
          borderWidth: 2,
          borderWidthSelected: 3
        }}
      }};
      const network = new vis.Network(container, data, options);
      network.once("stabilizationIterationsDone", function() {{
        network.setOptions({{ physics: false }});
      }});
    }};
    start();
  </script>
</body>
</html>"""
            escaped_srcdoc = html.escape(srcdoc, quote=True)
            return (
                "<iframe style='width:100%;height:620px;border:0;border-radius:14px;background:#fff;' "
                "sandbox='allow-scripts allow-same-origin' "
                f'srcdoc="{escaped_srcdoc}"></iframe>'
            )

        def build_kg_workspace(focus_query="", type_filter="", relation_filter="", max_nodes=36):
            focus_query = _safe_text(focus_query)
            type_filter = _safe_text(type_filter)
            relation_filter = _safe_text(relation_filter)
            max_nodes = max(10, min(120, _safe_int(max_nodes, 36)))
            fallback_notice = ""

            stats = refresh_kg_stats()
            try:
                raw_viz = search_engine.get_graph_visualization_data()
            except Exception as e:
                raw_viz = {"nodes": [], "edges": [], "error": str(e)}

            raw_nodes = raw_viz.get("nodes", []) or []
            raw_edges = raw_viz.get("edges", []) or []
            dedup_edges = []
            seen_edges = set()
            for edge in raw_edges:
                s = _safe_text(edge.get("source"))
                t = _safe_text(edge.get("target"))
                p = _safe_text(edge.get("predicate"))
                if not s or not t:
                    continue
                key = (s, t, p)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                dedup_edges.append(edge)
            raw_edges = dedup_edges
            node_map = {node.get("id"): node for node in raw_nodes if node.get("id")}

            filtered_nodes = []
            for node in raw_nodes:
                node_type = _safe_text(node.get("type"))
                label = _safe_text(node.get("label") or node.get("id"))
                if type_filter and type_filter.lower() not in node_type.lower():
                    continue
                if focus_query:
                    haystack = " ".join([label, node_type, _safe_text(node.get("description"))]).lower()
                    if focus_query.lower() not in haystack:
                        continue
                filtered_nodes.append(node)

            filtered_node_ids = {node.get("id") for node in filtered_nodes}
            filtered_edges = []
            for edge in raw_edges:
                predicate = _safe_text(edge.get("predicate"))
                if relation_filter and relation_filter.lower() not in predicate.lower():
                    continue
                if filtered_node_ids:
                    if edge.get("source") in filtered_node_ids or edge.get("target") in filtered_node_ids:
                        filtered_edges.append(edge)
                elif not focus_query and not type_filter:
                    filtered_edges.append(edge)

            if not filtered_nodes and not focus_query and not type_filter:
                filtered_nodes = raw_nodes[:max_nodes]
                filtered_node_ids = {node.get("id") for node in filtered_nodes}
                filtered_edges = [
                    edge for edge in raw_edges
                    if edge.get("source") in filtered_node_ids and edge.get("target") in filtered_node_ids
                ]
            elif not filtered_nodes and focus_query:
                # 默认聚焦实体未命中时，自动回退到全图子集，避免首屏空白。
                filtered_nodes = raw_nodes[:max_nodes]
                filtered_node_ids = {node.get("id") for node in filtered_nodes}
                filtered_edges = [
                    edge for edge in raw_edges
                    if edge.get("source") in filtered_node_ids and edge.get("target") in filtered_node_ids
                ]
                fallback_notice = (
                    f"默认聚焦“{html.escape(focus_query)}”未命中，已自动回退到全图概览子集。"
                )
            elif filtered_edges:
                expanded_ids = set()
                for edge in filtered_edges:
                    expanded_ids.add(edge.get("source"))
                    expanded_ids.add(edge.get("target"))
                filtered_nodes = [node_map[nid] for nid in expanded_ids if nid in node_map]

            degree_map = {node.get("id", ""): 0 for node in filtered_nodes}
            for edge in filtered_edges:
                if edge.get("source") in degree_map:
                    degree_map[edge["source"]] += 1
                if edge.get("target") in degree_map:
                    degree_map[edge["target"]] += 1

            filtered_nodes = sorted(
                filtered_nodes,
                key=lambda node: (-degree_map.get(node.get("id", ""), 0), node.get("label", "")),
            )[:max_nodes]
            limited_ids = {node.get("id") for node in filtered_nodes}
            filtered_edges = [
                edge for edge in filtered_edges
                if edge.get("source") in limited_ids and edge.get("target") in limited_ids
            ]

            summary_html = _build_kg_workspace_summary(stats, filtered_nodes, filtered_edges, focus_query, type_filter, relation_filter)
            if fallback_notice:
                summary_html = (
                    "<div style='margin-bottom:10px;padding:10px 12px;border:1px solid #fde68a;"
                    "border-radius:10px;background:#fffbeb;color:#92400e;font-size:13px;'>"
                    f"{fallback_notice}</div>"
                ) + summary_html
            plot = _build_kg_plot(filtered_nodes, filtered_edges, focus_query)

            node_rows = []
            for node in filtered_nodes:
                node_rows.append([
                    _safe_text(node.get("label") or node.get("id")),
                    _safe_text(node.get("type")) or "未分类",
                    _safe_int(node.get("doc_count"), 0),
                    degree_map.get(node.get("id", ""), 0),
                    _truncate_text(node.get("description"), 120),
                ])
            edge_rows = []
            for edge in filtered_edges[: max_nodes * 2]:
                edge_rows.append([
                    _safe_text(edge.get("source")),
                    _safe_text(edge.get("predicate")),
                    _safe_text(edge.get("target")),
                    _truncate_text(edge.get("description"), 120),
                ])

            node_df = pd.DataFrame(node_rows, columns=_kg_node_cols) if node_rows else pd.DataFrame(columns=_kg_node_cols)
            edge_df = pd.DataFrame(edge_rows, columns=_kg_edge_cols) if edge_rows else pd.DataFrame(columns=_kg_edge_cols)
            return summary_html, plot, node_df, edge_df

        def search_entities(query):
            """与搜索 Tab 一致，使用 pandas DataFrame 更新 gr.Dataframe，Gradio 4 下列表常无法稳定渲染。"""
            q = (query or "").strip()
            if not q:
                return pd.DataFrame(columns=_entity_df_cols)
            try:
                results = search_engine.search_entities(q, limit=10)
                if not results:
                    return pd.DataFrame(columns=_entity_df_cols)
                table_data = []
                for entity in results:
                    desc = (entity.get("description") or "") or entity.get("desc", "") or ""
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    try:
                        dc = int(entity.get("doc_count", 0) or 0)
                    except (TypeError, ValueError):
                        dc = 0
                    try:
                        sc = float(entity.get("score", 0) or 0.0)
                    except (TypeError, ValueError):
                        sc = 0.0
                    table_data.append(
                        [
                            str(entity.get("entity", "")),
                            str(entity.get("type") or entity.get("entity_type", "未分类")),
                            desc,
                            dc,
                            f"{sc:.4f}",
                        ]
                    )
                return pd.DataFrame(table_data, columns=_entity_df_cols)
            except Exception as e:
                return pd.DataFrame(
                    [["错误", "N/A", str(e)[:200], 0, "0.0000"]],
                    columns=_entity_df_cols,
                )

        def query_entity_relations(entity_name):
            name = (entity_name or "").strip()
            if not name:
                return (
                    {"hint": "请输入要查询的实体（与图中「name」一致，可全名或子串在搜索里找）"},
                    "<p style='text-align: center; color: #666;'>🔍 输入实体名称并查询</p>",
                )

            try:
                results = search_engine.query_entity_relations(name)
                viz_html = generate_relation_graph(name, results)
                return results, viz_html
            except Exception as e:
                error_html = f"<p style='color: red; text-align: center;'>❌ 查询失败: {str(e)}</p>"
                return {"error": str(e)}, error_html

        def generate_relation_graph(center_entity, relation_data):
            """
            关系图可视化。不使用 SVG：Gradio 4 对 gr.HTML 内联 SVG/defs 常做安全清洗，页面会整段空白。
            用 div + 列表边明细，经 html.escape 防注入。
            """
            def esc(s: str) -> str:
                return html.escape(str(s or ""), quote=True)

            c = esc(center_entity)

            if relation_data.get("error") and not relation_data.get("exists", True):
                return (
                    f"<div style='text-align:center; color: #b45309; padding: 1rem; border:1px solid #fbbf24; border-radius:8px; background:#fffbeb;'>"
                    f"<strong>{esc(relation_data.get('error', '未知错误'))}</strong></div>"
                )
            # 解析关系数据
            relations_dict = relation_data.get("relations", {}) or {}
            outgoing = relations_dict.get("outgoing", [])
            incoming = relations_dict.get("incoming", [])

            rows_out: list[str] = []
            for relation in outgoing:
                target = relation.get("target", "")
                if not target:
                    continue
                p = esc(relation.get("predicate", ""))
                rows_out.append(
                    f"<li style='margin:6px 0;line-height:1.5;'>"
                    f"<code style='background:#fee2e2;padding:2px 8px;border-radius:6px;color:#b91c1c;'>{c}</code> "
                    f"— <span style='color:#64748b;font-size:12px;'>「{p}」</span> → "
                    f"<code style='background:#e0f2fe;padding:2px 8px;border-radius:6px;color:#0369a1;'>{esc(target)}</code></li>"
                )
            rows_in: list[str] = []
            for relation in incoming:
                source = relation.get("source", "")
                if not source:
                    continue
                p = esc(relation.get("predicate", ""))
                rows_in.append(
                    f"<li style='margin:6px 0;line-height:1.5;'>"
                    f"<code style='background:#e0f2fe;padding:2px 8px;border-radius:6px;color:#0369a1;'>{esc(source)}</code> "
                    f"— <span style='color:#64748b;font-size:12px;'>「{p}」</span> → "
                    f"<code style='background:#fee2e2;padding:2px 8px;border-radius:6px;color:#b91c1c;'>{c}</code></li>"
                )

            out_count = len(rows_out)
            in_count = len(rows_in)
            type_label = esc(relation_data.get("entity_type", "未分类"))
            desc = esc(relation_data.get("description", "") or "暂无实体描述")

            if not rows_out and not rows_in:
                return f"""
<div style="padding:18px; border:1px solid #e2e8f0; border-radius:12px; background:#fff; min-height:100px;">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div style="display:inline-block; padding:10px 18px; background:#ef4444; color:#fff; border-radius:999px; font-weight:600; max-width:95%; word-break:break-all;">{c}</div>
      <div style="margin-top:10px;color:#475569;font-size:13px;">类型：{type_label}</div>
      <div style="margin-top:6px;color:#64748b;font-size:13px;max-width:720px;">{desc}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,110px));gap:8px;">
      <div style="border:1px solid #e2e8f0;border-radius:10px;padding:10px 12px;background:#f8fafc;"><div style="font-size:12px;color:#64748b;">出边</div><div style="font-size:22px;font-weight:700;color:#0f172a;">0</div></div>
      <div style="border:1px solid #e2e8f0;border-radius:10px;padding:10px 12px;background:#f8fafc;"><div style="font-size:12px;color:#64748b;">入边</div><div style="font-size:22px;font-weight:700;color:#0f172a;">0</div></div>
    </div>
  </div>
  <p style="color: #64748b; margin: 16px 0 0 0; font-size: 14px;">暂无有向边；该实体在图中为孤立点或仅存在未返回的元数据边。</p>
</div>"""

            parts = [f"""
<div class="kg-relation-viz" style="border:1px solid #e2e8f0;border-radius:12px;padding:16px;background:#ffffff;min-height:120px;max-width:100%; overflow-x:auto;">
  <div style="display:grid;grid-template-columns:minmax(0,1.4fr) minmax(260px,0.8fr);gap:16px;margin-bottom:14px;">
    <div>
      <div style="display:inline-block;padding:10px 18px;background:#ef4444;color:#fff;border-radius:999px;font-weight:600;word-break:break-all;max-width:100%;" title="中心实体">中心：{c}</div>
      <div style="margin-top:10px;font-size:13px;color:#475569;">类型：{type_label}</div>
      <div style="margin-top:6px;font-size:13px;color:#64748b;line-height:1.6;">{desc}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;">
      <div style="border:1px solid #e2e8f0;border-radius:10px;padding:12px;background:#f8fafc;"><div style="font-size:12px;color:#64748b;">出边</div><div style="font-size:24px;font-weight:700;color:#0f172a;">{out_count}</div></div>
      <div style="border:1px solid #e2e8f0;border-radius:10px;padding:12px;background:#f8fafc;"><div style="font-size:12px;color:#64748b;">入边</div><div style="font-size:24px;font-weight:700;color:#0f172a;">{in_count}</div></div>
      <div style="border:1px solid #e2e8f0;border-radius:10px;padding:12px;background:#f8fafc;grid-column:1 / span 2;"><div style="font-size:12px;color:#64748b;">分析提示</div><div style="font-size:13px;color:#0f172a;line-height:1.5;">优先看高频谓词和双向关联节点，再决定是否继续做路径分析或子图导出。</div></div>
    </div>
  </div>
  <h4 style="margin:0 0 8px 0;font-size:14px;color:#334155;">出边（自中心指出）</h4>
  <ul style="list-style:none;padding:0;margin:0 0 12px 0;">{"".join(rows_out) or "<li style='color:#94a3b8;'>无</li>"}</ul>
  <h4 style="margin:0 0 8px 0;font-size:14px;color:#334155;">入边（指向中心）</h4>
  <ul style="list-style:none;padding:0;margin:0;">{"".join(rows_in) or "<li style='color:#94a3b8;'>无</li>"}</ul>
  <p style="color:#94a3b8;font-size:12px;margin-top:12px;margin-bottom:0;">有向边列表与 Janus 中 <code>relation</code> 边方向一致；圆角块为实体名，中间为关系谓词。</p>
</div>"""]
            return "".join(parts)

        def refresh_kg_stats_with_line():
            return refresh_kg_stats(), format_kg_status_line()

        def extract_ner_triples_for_review(input_text, model_name):
            text = _safe_text(input_text)
            model = _safe_text(model_name)
            if not text:
                return "请输入待抽取文本。", [], []
            try:
                result = search_engine.extract_kg_triples_from_text(
                    text=text,
                    model=model or None,
                    max_items=300,
                )
            except Exception as e:
                return f"NER 调用失败: {e}", [], []
            if not result.get("success"):
                return f"抽取失败: {result.get('error', '未知错误')}", [], []

            rows = []
            candidates = []
            for idx, item in enumerate(result.get("triples", []) or []):
                cid = str(idx + 1)
                can_insert = bool(item.get("can_insert", False))
                candidates.append(
                    {
                        "id": cid,
                        "can_insert": can_insert,
                        "subject": _safe_text(item.get("subject")),
                        "predicate": _safe_text(item.get("predicate")),
                        "predicate_raw": _safe_text(item.get("predicate_raw")),
                        "object": _safe_text(item.get("object")),
                        "subject_type": _safe_text(item.get("subject_type")) or "未分类",
                        "object_type": _safe_text(item.get("object_type")) or "未分类",
                        "description": _safe_text(item.get("description")),
                        "reject_reason": _safe_text(item.get("reject_reason")),
                    }
                )
                rows.append(
                    [
                        cid,
                        can_insert,
                        _safe_text(item.get("subject")),
                        _safe_text(item.get("predicate")),
                        _safe_text(item.get("object")),
                        _safe_text(item.get("subject_type")) or "未分类",
                        _safe_text(item.get("object_type")) or "未分类",
                        _safe_text(item.get("description")),
                        _safe_text(item.get("reject_reason")),
                        _safe_text(item.get("predicate_raw")),
                    ]
                )
            msg = (
                f"抽取完成：候选三元组 {int(result.get('relations_count', 0) or 0)} 条，"
                f"识别实体 {int(result.get('entities_count', 0) or 0)} 个。"
                f"本次发现不合规谓词 {int(result.get('filtered_out_count', 0) or 0)} 条（仍显示供审阅）。"
                f"当前可用谓词 {int(result.get('openkg_predicate_count', 0) or 0)} 项。"
                "如需剔除，请填写候选ID并删除，再执行写入。"
            )
            return msg, rows, candidates

        def _rows_from_candidates(candidates):
            rows = []
            for item in candidates:
                rows.append(
                    [
                        _safe_text(item.get("id")),
                        bool(item.get("can_insert", False)),
                        _safe_text(item.get("subject")),
                        _safe_text(item.get("predicate")),
                        _safe_text(item.get("object")),
                        _safe_text(item.get("subject_type")) or "未分类",
                        _safe_text(item.get("object_type")) or "未分类",
                        _safe_text(item.get("description")),
                        _safe_text(item.get("reject_reason")),
                        _safe_text(item.get("predicate_raw")),
                    ]
                )
            return rows

        def delete_ner_candidates(delete_ids_text, candidates):
            ids_text = _safe_text(delete_ids_text)
            rows = candidates if isinstance(candidates, list) else []
            if not rows:
                return "当前没有候选可删除。", [], rows
            delete_ids = {
                part.strip()
                for part in ids_text.replace("，", ",").split(",")
                if part.strip()
            }
            if not delete_ids:
                return "未提供删除ID，未执行删除。", _rows_from_candidates(rows), rows
            remained = [item for item in rows if str(item.get("id", "")) not in delete_ids]
            deleted = len(rows) - len(remained)
            msg = f"删除完成：删除 {deleted} 行，剩余 {len(remained)} 行。"
            return msg, _rows_from_candidates(remained), remained

        def insert_selected_ner_triples(candidates):
            rows = candidates if isinstance(candidates, list) else []
            if not rows:
                return "没有可写入数据。", refresh_kg_stats(), format_kg_status_line()
            triples = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                can_insert = bool(item.get("can_insert", False))
                subject = _safe_text(item.get("subject"))
                predicate = _safe_text(item.get("predicate"))
                obj = _safe_text(item.get("object"))
                stype = _safe_text(item.get("subject_type")) or "未分类"
                otype = _safe_text(item.get("object_type")) or "未分类"
                desc = _safe_text(item.get("description"))
                raw_predicate = _safe_text(item.get("predicate_raw"))
                triples.append(
                    {
                        "selected": can_insert,
                        "subject": subject,
                        "predicate": predicate,
                        "predicate_raw": raw_predicate,
                        "object": obj,
                        "subject_type": stype,
                        "object_type": otype,
                        "description": desc,
                    }
                )
            selected_count = sum(1 for t in triples if t.get("selected"))
            if selected_count <= 0:
                return "未勾选任何三元组，未执行写入。", refresh_kg_stats(), format_kg_status_line()
            try:
                res = search_engine.insert_selected_kg_triples(triples)
            except Exception as e:
                return f"写入失败: {e}", refresh_kg_stats(), format_kg_status_line()
            if not res.get("success"):
                err = _safe_text(res.get("error")) or "未知错误"
                return f"写入失败: {err}", refresh_kg_stats(), format_kg_status_line()
            inserted = int(res.get("inserted", 0) or 0)
            skipped = int(res.get("skipped", 0) or 0)
            return (
                f"写入完成：成功 {inserted} 条，跳过 {skipped} 条。",
                refresh_kg_stats(),
                format_kg_status_line(),
            )

        # 知识图谱事件绑定
        reload_kg_btn.click(
            fn=do_reload_knowledge_graph,
            outputs=[kg_modify_status, kg_stats_display, kg_build_status]
        )

        refresh_kg_stats_btn.click(
            fn=refresh_kg_stats_with_line,
            outputs=[kg_stats_display, kg_build_status]
        )
        kg_workspace_btn.click(
            fn=build_kg_workspace,
            inputs=[kg_focus_query, kg_type_filter, kg_relation_filter, kg_max_nodes],
            outputs=[kg_workspace_summary, kg_graph_plot, kg_node_inventory, kg_edge_inventory]
        )
        ner_extract_btn.click(
            fn=extract_ner_triples_for_review,
            inputs=[ner_input_text, ner_model_text],
            outputs=[ner_status, ner_triples_df, ner_candidate_state],
        )
        ner_delete_btn.click(
            fn=delete_ner_candidates,
            inputs=[ner_delete_ids_text, ner_candidate_state],
            outputs=[ner_status, ner_triples_df, ner_candidate_state],
        )
        ner_insert_btn.click(
            fn=insert_selected_ner_triples,
            inputs=[ner_candidate_state],
            outputs=[ner_status, kg_stats_display, kg_build_status],
        )

        entity_search_btn.click(
            fn=search_entities,
            inputs=entity_search_query,
            outputs=entity_search_results
        )
        entity_search_query.submit(
            fn=search_entities,
            inputs=entity_search_query,
            outputs=entity_search_results
        )

        entity_query_btn.click(
            fn=query_entity_relations,
            inputs=entity_query_input,
            outputs=[entity_query_results, entity_graph_viz]
        )
        entity_query_btn.click(
            fn=lambda entity_name, max_nodes: build_kg_workspace(entity_name, "", "", max_nodes),
            inputs=[entity_query_input, kg_max_nodes],
            outputs=[kg_workspace_summary, kg_graph_plot, kg_node_inventory, kg_edge_inventory]
        )
        entity_query_input.submit(
            fn=query_entity_relations,
            inputs=entity_query_input,
            outputs=[entity_query_results, entity_graph_viz]
        )
        entity_query_input.submit(
            fn=lambda entity_name, max_nodes: build_kg_workspace(entity_name, "", "", max_nodes),
            inputs=[entity_query_input, kg_max_nodes],
            outputs=[kg_workspace_summary, kg_graph_plot, kg_node_inventory, kg_edge_inventory]
        )

        # 首屏一次加载：文档列表、图谱统计、倒排索引片段（与按钮逻辑相同，不重复发请求到后端）
        def on_index_tab_load():
            workspace_summary, workspace_plot, node_df, edge_df = build_kg_workspace(
                _default_focus_entity, "", "", 36
            )
            return (
                get_all_documents(search_engine),
                refresh_kg_stats(),
                view_inverted_index(search_engine),
                format_kg_status_line(),
                workspace_summary,
                workspace_plot,
                node_df,
                edge_df,
            )

        index_tab.load(
            fn=on_index_tab_load,
            outputs=[
                docs_list,
                kg_stats_display,
                view_index_output,
                kg_build_status,
                kg_workspace_summary,
                kg_graph_plot,
                kg_node_inventory,
                kg_edge_inventory,
            ],
        )

    return index_tab
