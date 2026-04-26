#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG标签页UI实现
"""

import gradio as gr
import json
import html as html_lib
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from .rag_service import RAGService


def _rv_safe(v: Any) -> str:
    return str(v or "").strip()


def _rv_trunc(s: str, n: int = 36) -> str:
    t = _rv_safe(s)
    return t if len(t) <= n else t[: max(0, n - 1)] + "…"


def build_retrieval_graph_html(
    *,
    query: str,
    is_oag: bool,
    graph_walk: Optional[List[Dict[str, Any]]] = None,
    retrieved_docs: Optional[List[Tuple[str, float, str]]] = None,
) -> str:
    """
    与离线索引「知识图谱」页相同的 vis-network 思路：将 OAG 游走三元组或 RAG 检索文档画成子图。
    """
    focus = _rv_trunc(query, 48)
    graph_id = f"rag_retrieval_vis_{int(datetime.now().timestamp() * 1000)}"
    height_px = 460

    vis_nodes: List[Dict[str, Any]] = []
    vis_edges: List[Dict[str, Any]] = []

    if is_oag:
        triples = graph_walk or []
        if not triples:
            return (
                "<div style='padding:16px;border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;color:#64748b;'>"
                "暂无可视化子图（尚无游走关系）。完成 ReAct 或出现图谱事实后将自动渲染。"
                "</div>"
            )
        node_ids = set()
        for item in triples:
            s = _rv_safe(item.get("source"))
            t = _rv_safe(item.get("target"))
            if s:
                node_ids.add(s)
            if t:
                node_ids.add(t)
        palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2"]
        for i, nid in enumerate(sorted(node_ids)):
            c = palette[i % len(palette)]
            is_q = bool(focus) and (focus.lower() in nid.lower())
            bg = "#ef4444" if is_q else c
            vis_nodes.append(
                {
                    "id": nid,
                    "label": _rv_trunc(nid, 20),
                    "title": html_lib.escape(f"实体: {nid}"),
                    "shape": "dot",
                    "size": 24 if is_q else 20,
                    "color": {"background": bg, "border": "#0f172a"},
                    "font": {"color": "#111827", "size": 12},
                }
            )
        for item in triples:
            s = _rv_safe(item.get("source"))
            p = _rv_safe(item.get("predicate"))
            t = _rv_safe(item.get("target"))
            if not s or not t:
                continue
            depth = int(item.get("depth", 0) or 0)
            vis_edges.append(
                {
                    "from": s,
                    "to": t,
                    "label": _rv_trunc(p, 14),
                    "title": html_lib.escape(f"[{depth}跳] {s} --[{p}]--> {t}"),
                    "arrows": "to",
                    "color": {"color": "#94a3b8", "highlight": "#ef4444"},
                    "font": {"align": "middle", "size": 10, "strokeWidth": 0},
                    "smooth": {"enabled": True, "type": "dynamic"},
                }
            )
        title = "OAG 检索子图（游走关系）"
        sub = f"问题：{focus or '—'} · 悬停边可查看谓词与跳数"
    else:
        docs = retrieved_docs or []
        if not docs:
            return (
                "<div style='padding:16px;border:1px dashed #cbd5e1;border-radius:12px;background:#f8fafc;color:#64748b;'>"
                "暂无可视化子图（未检索到文档或未开启检索）。"
                "</div>"
            )
        qid = "__user_query__"
        vis_nodes.append(
            {
                "id": qid,
                "label": _rv_trunc(query, 18) or "查询",
                "title": html_lib.escape(f"查询：{query}"),
                "shape": "box",
                "size": 22,
                "color": {"background": "#ef4444", "border": "#0f172a"},
                "font": {"color": "#fff", "size": 12},
            }
        )
        for i, (doc_id, score, content) in enumerate(docs):
            nid = f"doc_{i}"
            preview = _rv_trunc(content.replace("\n", " "), 120)
            vis_nodes.append(
                {
                    "id": nid,
                    "label": _rv_trunc(doc_id, 16),
                    "title": html_lib.escape(f"文档: {doc_id}\n相关度: {score:.4f}\n预览: {preview}"),
                    "shape": "ellipse",
                    "size": 18,
                    "color": {"background": "#3b82f6", "border": "#1e3a8a"},
                    "font": {"color": "#111827", "size": 11},
                }
            )
            vis_edges.append(
                {
                    "from": qid,
                    "to": nid,
                    "label": f"{score:.3f}",
                    "title": html_lib.escape(f"相关度 {score:.4f} · {doc_id}"),
                    "arrows": "to",
                    "color": {"color": "#94a3b8", "highlight": "#ef4444"},
                    "font": {"align": "middle", "size": 10, "strokeWidth": 0},
                    "smooth": {"enabled": True, "type": "dynamic"},
                }
            )
        title = "RAG 检索子图（查询 → 文档）"
        sub = f"边标签为 TF-IDF 相关度分数"

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(vis_edges, ensure_ascii=False).replace("</", "<\\/")

    title_html = html_lib.escape(title)
    sub_html = html_lib.escape(sub)

    srcdoc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; background:#fff; color:#0f172a; }}
    .wrap {{ border:1px solid #e2e8f0; border-radius:14px; overflow:hidden; background:#fff; }}
    .toolbar {{ padding:10px 14px; border-bottom:1px solid #e2e8f0; background:#f8fafc; }}
    #{graph_id} {{ height: {height_px}px; background:#fff; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <div style="font-size:13px;font-weight:700;color:#0f172a;">{title_html}</div>
      <div style="font-size:12px;color:#64748b;">{sub_html}</div>
    </div>
    <div id="{graph_id}"></div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <script>
    const start = () => {{
      if (!window.vis) {{ setTimeout(start, 120); return; }}
      const el = document.getElementById("{graph_id}");
      if (!el) return;
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const network = new vis.Network(el, {{ nodes, edges }}, {{
        autoResize: true,
        interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true }},
        layout: {{ improvedLayout: true }},
        physics: {{
          enabled: true,
          stabilization: {{ enabled: true, iterations: 160 }},
          barnesHut: {{ gravitationalConstant: -4800, centralGravity: 0.2, springLength: 140, springConstant: 0.04, damping: 0.12 }}
        }},
        edges: {{ selectionWidth: 2.2 }},
        nodes: {{ borderWidth: 2, borderWidthSelected: 3 }}
      }});
      network.once("stabilizationIterationsDone", function() {{ network.setOptions({{ physics: false }}); }});
    }};
    start();
  </script>
</body>
</html>"""
    esc = html_lib.escape(srcdoc, quote=True)
    return (
        "<iframe style='width:100%;height:"
        + str(height_px + 56)
        + "px;border:0;border-radius:14px;background:#fff;' "
        "sandbox='allow-scripts allow-same-origin' "
        f'srcdoc="{esc}"></iframe>'
    )

def build_rag_tab(index_service, inference_model=None, page_mode: str = "rag"):
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
    
    normalized_page_mode = str(page_mode).strip().lower()
    is_oag_page = normalized_page_mode == "oag"
    page_title = "🤖 RAG" if not is_oag_page else "🧠 OAG"
    page_desc = (
        "面向单步检索增强问答。"
        if not is_oag_page
        else "面向基于知识图谱游走的多步推理问答。"
    )

    with gr.Column():
        gr.Markdown("""
        # {page_title}
        
        支持两种模式：
        - **DashScope API**: 使用阿里云通义千问API（在线）
        - **本地模型**: 使用训练好的SFT/DPO模型（需先加载）
        
        当前页面定位：**{page_desc}**
        """.replace("{page_title}", page_title).replace("{page_desc}", page_desc))
        
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
                        maximum=10 if not is_oag_page else 4,
                        value=5 if not is_oag_page else 2,
                        step=1,
                        label="检索文档数量" if not is_oag_page else "图谱游走深度"
                    )
                    graph_expand_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        visible=is_oag_page,
                        label="每跳关系扩展上限"
                    )

                with gr.Row():
                    retrieval_enabled = gr.Checkbox(
                        label="开启检索增强 (RAG)",
                        value=True,
                        visible=not is_oag_page,
                    )
                    mode_hint = "固定模式：词法检索增强" if not is_oag_page else "固定模式：图谱游走多步推理（OAG）"
                    gr.Textbox(
                        label="推理流程",
                        value=mode_hint,
                        interactive=False
                    )
                
                rag_query_btn = gr.Button(
                    "🚀 执行查询" if not is_oag_page else "🚀 执行 OAG 查询",
                    variant="primary"
                )
                
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

                agent_steps_output = gr.JSON(
                    label="智能体工作步骤（OAG ReAct）",
                    visible=is_oag_page
                )
        
        # 4. 提示词展示
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 提示词/推理轨迹")
                prompt_display = gr.Textbox(
                    label="ReAct 推理轨迹" if is_oag_page else "检索材料（仅文档拼接，不含问题与指令）",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    placeholder="OAG：ReAct 各步与观察。RAG：仅检索到的文档原文拼接。",
                    show_copy_button=True,
                    autoscroll=False
                )
        
        # 5. 检索详情
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 检索结果详情")
                retrieved_docs = gr.DataFrame(
                    headers=["文档ID", "相关度分数", "文档内容"] if not is_oag_page else ["跳数", "起点实体", "关系", "目标实体"],
                    label="检索到的文档" if not is_oag_page else "图谱游走关系",
                    interactive=False
                )
                
                context_output = gr.Textbox(
                    label="完整 Prompt（实际发送给 LLM）",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )

                retrieval_graph_html = gr.HTML(
                    label="检索子图可视化（与索引知识图谱页相同技术：vis-network）",
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
    
    def oag_result_to_outputs(result: Dict[str, Any], mode_label: str) -> Tuple:
        walk_rows = [
            [int(item.get("depth", 0) or 0), item.get("source", ""), item.get("predicate", ""), item.get("target", "")]
            for item in result.get("graph_walk", [])
        ]
        phase = str(result.get("stream_phase", "") or "")
        processing_info = f"""处理时间: {result.get('processing_time', 0):.2f}秒
推理模式: {mode_label}
ReAct步数: {len(result.get('agent_steps', []))}
种子实体数: {len(result.get('seed_entities', []))}
游走事实数: {len(result.get('graph_walk', []))}"""
        if phase:
            processing_info = processing_info + f"\n阶段: {phase}"
        q_for_viz = _rv_safe(result.get("query", ""))
        viz = build_retrieval_graph_html(
            query=q_for_viz,
            is_oag=True,
            graph_walk=result.get("graph_walk", []),
            retrieved_docs=None,
        )
        return (
            result.get("answer", "生成回答失败"),
            processing_info,
            walk_rows,
            result.get("prompt_sent", ""),
            result.get("react_trace", "") or result.get("prompt_sent", ""),
            result.get("agent_steps", []),
            viz,
        )

    def process_rag_query(query: str, top_k: int, mode: str, retrieval_enabled_flag: bool, graph_expand: int):
        """处理RAG查询（支持DashScope API和本地模型）。OAG 在 ReAct 每步流式更新界面。"""
        if not query.strip():
            yield (
                "请输入您的问题",
                "未处理",
                [],
                "",
                "",
                [],
                "<div style='padding:12px;color:#94a3b8;font-size:13px;'>提交问题后将在此显示检索子图。</div>",
            )
            return
        
        # OAG 页面：固定走知识图谱游走链路。
        if is_oag_page:
            walk_depth = int(top_k or 2)
            walk_expand = int(graph_expand or 4)
            if mode == "DashScope API":
                for result in rag_service.iter_oag_query(
                    query=query,
                    max_depth=walk_depth,
                    expand_per_hop=walk_expand,
                    model="qwen-plus",
                    generate_answer=True,
                    use_react_agent=True,
                    max_steps=6,
                ):
                    yield oag_result_to_outputs(result, mode)
                return

            # 本地模型 OAG
            # 使用本地模型
            if not inference_model.loaded:
                yield (
                    "❌ 请先加载本地模型\n\n点击上方的「▶️ 加载模型」按钮",
                    "未处理",
                    [],
                    "",
                    "",
                    [],
                    "<div style='padding:12px;color:#94a3b8;'>加载本地模型后可使用 OAG。</div>",
                )
                return

            import time
            last_oag: Optional[Dict[str, Any]] = None
            for oag_part in rag_service.iter_oag_query(
                query=query,
                max_depth=walk_depth,
                expand_per_hop=walk_expand,
                generate_answer=False,
                use_react_agent=True,
                max_steps=6,
            ):
                last_oag = oag_part
                if oag_part.get("stream_phase") == "react":
                    yield oag_result_to_outputs(oag_part, "本地模型")
            if not last_oag or last_oag.get("stream_phase") != "final":
                return
            prompt = last_oag.get("prompt_sent", query)
            t_start = time.time()
            answer = inference_model.generate_once(
                prompt=prompt,
                temperature=0.7,
                max_new_tokens=512
            )
            gen_time = time.time() - t_start
            last_oag = dict(last_oag)
            last_oag["answer"] = answer
            last_oag["processing_time"] = float(last_oag.get("processing_time", 0) or 0) + gen_time
            yield oag_result_to_outputs(last_oag, "本地模型")
            return

        # RAG 页面：保持单步检索增强问答。
        if mode == "DashScope API":
            result = rag_service.rag_query(
                query=query,
                top_k=top_k,
                model="qwen-plus",
                retrieval_enabled=retrieval_enabled_flag,
                multi_step=False,
            )
        else:
            # 使用本地模型
            if not inference_model.loaded:
                yield (
                    "❌ 请先加载本地模型\n\n点击上方的「▶️ 加载模型」按钮",
                    "未处理",
                    [],
                    "",
                    "",
                    [],
                    "<div style='padding:12px;color:#94a3b8;'>加载模型后可显示检索子图。</div>",
                )
                return

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

            rag_viz = build_retrieval_graph_html(
                query=query,
                is_oag=False,
                graph_walk=None,
                retrieved_docs=retrieved_docs,
            )
            yield (
                answer,
                processing_info,
                retrieved_table,
                prompt,
                context,
                [],
                rag_viz,
            )
            return

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

        ds_viz = build_retrieval_graph_html(
            query=query,
            is_oag=False,
            graph_walk=None,
            retrieved_docs=result.get("retrieved_docs", []) or [],
        )

        yield (
            result.get("answer", "生成回答失败"),
            processing_info,
            retrieved_table,
            result.get("prompt_sent", ""),
            result.get("context", ""),
            [],
            ds_viz,
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
        inputs=[query_input, top_k_slider, inference_mode, retrieval_enabled, graph_expand_slider],
        outputs=[
            answer_output,
            processing_info,
            retrieved_docs,
            context_output,
            prompt_display,
            agent_steps_output,
            retrieval_graph_html,
        ],
    )
    
    # 页面加载时获取统计信息
    stats_display.value = get_rag_stats()
    
    return gr.Column() 