import gradio as gr
import pandas as pd
import uuid
from datetime import datetime
from ..training_tab.ctr_config import CTRSampleConfig, ctr_sample_config
from ..data_utils import (
    record_search_impression, 
    record_document_click, 
    get_ctr_dataframe,
    validate_search_params,
    validate_click_params
)
import re

# 全局变量用于存储当前request_id
current_request_id = None

def perform_search(index_service, data_service, query: str, sort_mode: str = "ctr"):
    if not query or not query.strip():
        return [], pd.DataFrame(), ""
    try:
        query_clean = query.strip()
        doc_ids = index_service.retrieve(query_clean, top_k=20)
        
        # 调用rank方法时传递sort_mode参数
        ranked = index_service.rank(query_clean, doc_ids, top_k=10, sort_mode=sort_mode)
        
        # 现在ranked已经是正确排序的结果，不需要再次排序
        final = ranked
        
        if not final:
            return [], pd.DataFrame(), ""
        
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
        docs_info = []
        for position, result in enumerate(final, 1):
            doc_id, tfidf_score, summary = parse_result_tuple(result)
            
            # 使用新的工具函数并添加参数验证
            validation_errors = validate_search_params(query_clean, doc_id, position, tfidf_score)
            if validation_errors:
                print(f"⚠️ 搜索参数验证失败: {validation_errors}")
                continue
            
            # 记录展示事件
            record_search_impression(query_clean, doc_id, position, tfidf_score, summary, request_id)
            
            # 添加CTR分数到文档信息中（如果有的话）
            doc_info = {
                'doc_id': doc_id,
                'tfidf_score': tfidf_score,
                'summary': summary,
                'position': position
            }
            
            # 如果结果包含CTR分数（4元组），则添加到信息中
            if len(result) == 4:
                doc_info['ctr_score'] = result[2]
            
            docs_info.append(doc_info)
        
        # 获取CTR数据框
        return docs_info, get_ctr_dataframe(request_id), request_id
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return [], pd.DataFrame(), ""

def apply_sorting_mode(results: list, sort_mode: str) -> list:
    """应用排序模式"""
    if not results:
        return results
    if sort_mode == "tfidf":
        return sorted(results, key=lambda x: x[1], reverse=True)
    elif sort_mode == "ctr":
        has_ctr_scores = any(len(x) == 4 and x[2] is not None for x in results)
        if has_ctr_scores:
            return sorted(results, key=lambda x: (x[2] if len(x) == 4 and x[2] is not None else 0), reverse=True)
        else:
            return sorted(results, key=lambda x: x[1], reverse=True)
    else:
        return sorted(results, key=lambda x: x[1], reverse=True)

def process_search_results(results: list):
    """处理搜索结果"""
    formatted_results = []
    for position, result in enumerate(results):
        doc_id, tfidf_score, summary = parse_result_tuple(result)
        doc_length = len(result[0]) if result else 0  # 简化处理
        formatted_results.append([
            doc_id,
            f"TF-IDF: {tfidf_score:.4f}",
            doc_length,
            summary[:100]
        ])
    first_doc_html = f"<p>文档ID: {results[0][0]}</p><p>摘要: {results[0][-1]}</p>" if results else "<p>无结果</p>"
    return formatted_results, first_doc_html

def parse_result_tuple(result):
    """解析结果元组"""
    if len(result) == 4:
        return result[0], result[1], result[3]
    else:
        return result[0], result[1], result[2]

def on_view_fulltext(search_engine, ctr_collector, current_query, request_id, doc_id):
    try:
        # 记录点击
        for d in ctr_collector.ctr_data:
            if d.get('request_id') == request_id:
                if d.get('doc_id') == doc_id:
                    d['clicked'] = 1
                else:
                    d['clicked'] = 0
        samples = [d for d in ctr_collector.ctr_data if d.get('request_id') == request_id]
        df = pd.DataFrame(samples) if samples else pd.DataFrame()
        content = search_engine.get_document(doc_id)
        html = f"<h3>文档ID: {doc_id}</h3><hr><pre style='white-space:pre-wrap'>{content}</pre>"
        return html, df
    except Exception as e:
        return f"获取文档详情失败: {str(e)}", pd.DataFrame()

def on_document_click(index_service, data_service, doc_id, request_id):
    """处理文档点击事件"""
    try:
        # 记录文档点击事件
        
        # 使用新的验证工具
        validation_errors = validate_click_params(doc_id, request_id)
        if validation_errors:
            error_msg = f"参数验证失败: {', '.join(validation_errors)}"
            print(f"⚠️ {error_msg}")
            return error_msg, pd.DataFrame()
        
        # 使用新的工具函数记录点击
        click_success = record_document_click(doc_id, request_id)
        if not click_success:
            print(f"⚠️ 点击记录失败: doc_id={doc_id}, request_id={request_id}")
        
        # 获取文档内容
        result = index_service.get_document_page(doc_id, request_id, data_service)
        html_content = result['html']
        
        # 获取更新后的CTR样本
        return html_content, get_ctr_dataframe(request_id)
        
    except Exception as e:
        error_msg = f"获取文档详情失败: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, pd.DataFrame()

def show_search_stats():
    """显示搜索统计"""
    return "<p>搜索统计展示</p>"

def show_fulltext(search_engine, doc_id):
    content = search_engine.get_document(doc_id)
    html = f"<h3>文档ID: {doc_id}</h3><hr><pre style='white-space:pre-wrap'>{content}</pre>"
    return html

def strip_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def build_search_tab(index_service, data_service):
    with gr.Blocks() as search_tab:
        gr.Markdown("""### 🔍 第二部分：在线召回排序""")
        sort_mode = gr.Dropdown(
            choices=["tfidf", "ctr"],
            value="ctr",
            label="排序算法",
            info="选择排序算法进行对比实验：TF-IDF/CTR"
        )
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="实验查询", placeholder="输入测试查询进行检索实验...", lines=1)
                with gr.Row():
                    search_btn = gr.Button("🔬 执行检索", variant="primary")
                    search_stats_btn = gr.Button("📊 搜索统计")
                # 检索结果 DataFrame 区
                results_df = gr.Dataframe(
                    headers=["文档ID", "分数", "附加信息", "摘要"],
                    label="检索结果",
                    interactive=False,
                    row_count=10,
                    col_count=4
                )
        doc_content = gr.HTML(value="<p>点击下方'查看全文'按钮查看文档内容...</p>", label="文档内容")
        back_btn = gr.Button("⬅️ 返回搜索结果", visible=False)
        sample_output = gr.Dataframe(
            headers=None,
            label="本次产生的CTR样本",
            interactive=False
        )
        request_id_state = gr.State("")
        with gr.Accordion("🧪 测试用例", open=False):
            gr.Markdown("""推荐测试查询：人工智能、机器学习、深度学习等""")
        # 检索按钮事件
        def update_results(query, sort_mode):
            docs_info, df, request_id = perform_search(index_service, data_service, query, sort_mode)
            
            # 转换为 DataFrame 展示格式，根据排序模式显示不同的列
            formatted_results = []
            for doc in docs_info:
                summary_plain = strip_html_tags(doc['summary'])
                
                if sort_mode == "ctr" and 'ctr_score' in doc:
                    # CTR排序模式：显示CTR分数
                    formatted_results.append([
                        doc['doc_id'],
                        f"{doc['tfidf_score']:.4f}",
                        f"{doc['ctr_score']:.4f}",
                        summary_plain[:100] + ("..." if len(summary_plain) > 100 else "")
                    ])
                else:
                    # TF-IDF排序模式：显示文档长度
                    formatted_results.append([
                        doc['doc_id'],
                        f"{doc['tfidf_score']:.4f}",
                        f"{len(doc['summary'])}",
                        summary_plain[:100] + ("..." if len(summary_plain) > 100 else "")
                    ])
            
            # 根据排序模式创建DataFrame
            if sort_mode == "ctr":
                # 创建CTR模式的DataFrame
                df_display = pd.DataFrame(formatted_results, columns=["文档ID", "TF-IDF分数", "CTR分数", "摘要"])
                mode_text = "CTR智能排序"
            else:
                # 创建TF-IDF模式的DataFrame
                df_display = pd.DataFrame(formatted_results, columns=["文档ID", "TF-IDF分数", "文档长度", "摘要"])
                mode_text = "TF-IDF传统排序"
            
            # 显示当前排序模式
            print(f"🔍 当前排序模式: {mode_text}")
            
            return df_display, df, request_id
        search_btn.click(
            fn=update_results,
            inputs=[query_input, sort_mode],
            outputs=[results_df, sample_output, request_id_state]
        )
        search_stats_btn.click(
            fn=show_search_stats,
            outputs=doc_content
        )
        query_input.submit(
            fn=update_results,
            inputs=[query_input, sort_mode],
            outputs=[results_df, sample_output, request_id_state]
        )
        def refresh_samples(rid):
            if rid:
                # 使用新的工具函数
                return get_ctr_dataframe(rid)
            return pd.DataFrame()
        query_input.change(
            fn=refresh_samples,
            inputs=request_id_state,
            outputs=sample_output
        )
        # 绑定 DataFrame 行点击事件
        def on_row_select(evt: gr.SelectData, df, request_id):
            if evt is None or evt.index is None:
                return "未选中行", pd.DataFrame(), gr.update(visible=False), gr.update(visible=True)
            # 只处理单行
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            row = df.iloc[idx]
            doc_id = row["文档ID"]
            html, samples = on_document_click(index_service, data_service, doc_id, request_id)
            return html, samples, gr.update(visible=True), gr.update(visible=False)
        results_df.select(
            fn=on_row_select,
            inputs=[results_df, request_id_state],
            outputs=[doc_content, sample_output, back_btn, results_df]
        )
        def on_back_click():
            # 恢复主检索结果区，隐藏返回按钮
            return gr.update(visible=False), gr.update(visible=True)
        back_btn.click(
            fn=on_back_click,
            outputs=[back_btn, results_df]
        )
    return search_tab 