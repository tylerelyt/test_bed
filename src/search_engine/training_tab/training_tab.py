import gradio as gr
from datetime import datetime
from ..data_utils import (
    get_data_statistics,
    get_ctr_dataframe,
    clear_all_data,
    export_ctr_data,
    import_ctr_data,
    analyze_click_patterns
)

def get_history_html(ctr_collector):
    """获取历史记录HTML"""
    try:
        history = ctr_collector.get_history()
        if not history:
            return "<p>暂无历史记录</p>"
        
        html_content = "<div style='max-height: 400px; overflow-y: auto;'>"
        html_content += "<h4>📊 点击行为历史记录</h4>"
        
        for record in history[:20]:  # 只显示前20条
            clicked_icon = "✅" if record.get('clicked', 0) else "❌"
            html_content += f"""
            <div style="border: 1px solid #ddd; margin: 5px 0; padding: 10px; border-radius: 5px;">
                <div><strong>查询:</strong> {record.get('query', 'N/A')}</div>
                <div><strong>文档ID:</strong> {record.get('doc_id', 'N/A')}</div>
                <div><strong>位置:</strong> {record.get('position', 'N/A')}</div>
                <div><strong>分数:</strong> {record.get('score', 'N/A'):.4f}</div>
                <div><strong>点击:</strong> {clicked_icon}</div>
                <div><strong>时间:</strong> {record.get('timestamp', 'N/A')}</div>
            </div>
            """
        
        html_content += "</div>"
        return html_content
    except Exception as e:
        return f"<p style='color: red;'>获取历史记录失败: {str(e)}</p>"

def train_ctr_model(ctr_model, ctr_collector):
    """训练CTR模型"""
    try:
        # 获取训练数据
        training_data = ctr_collector.export_data()
        records = training_data.get('records', [])
        
        if len(records) < 10:
            return (
                "<p style='color: orange;'>⚠️ 训练数据不足，至少需要10条记录</p>",
                "<p>请先进行一些搜索和点击操作收集数据</p>",
                "<p>暂无特征权重数据</p>"
            )
        
        # 训练模型
        result = ctr_model.train(records)
        
        if 'error' in result:
            return (
                f"<p style='color: red;'>❌ 训练失败: {result['error']}</p>",
                "<p>请检查数据质量</p>",
                "<p>暂无特征权重数据</p>"
            )
        
        # 生成训练结果HTML
        model_status = f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
            <h4>✅ CTR模型训练成功</h4>
            <p><strong>AUC:</strong> {result.get('auc', 0):.4f}</p>
            <p><strong>训练样本:</strong> {result.get('train_samples', 0)}</p>
            <p><strong>测试样本:</strong> {result.get('test_samples', 0)}</p>
        </div>
        """
        
        train_result = f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            <h4>📈 训练结果详情</h4>
            <ul>
                <li><strong>精确率:</strong> {result.get('precision', 0):.4f}</li>
                <li><strong>召回率:</strong> {result.get('recall', 0):.4f}</li>
                <li><strong>F1分数:</strong> {result.get('f1', 0):.4f}</li>
                <li><strong>训练准确率:</strong> {result.get('train_score', 0):.4f}</li>
                <li><strong>测试准确率:</strong> {result.get('test_score', 0):.4f}</li>
            </ul>
        </div>
        """
        
        # 特征权重可视化
        feature_weights = result.get('feature_weights', {})
        if feature_weights:
            weights_html = "<h4>🎯 特征重要性分析</h4><ul>"
            sorted_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
            for feature, weight in sorted_weights[:10]:  # 显示前10个特征
                weights_html += f"<li><strong>{feature}:</strong> {weight:.4f}</li>"
            weights_html += "</ul>"
        else:
            weights_html = "<p>暂无特征权重数据</p>"
        
        return model_status, train_result, weights_html
        
    except Exception as e:
        return (
            f"<p style='color: red;'>❌ 训练过程出错: {str(e)}</p>",
            "<p>请检查系统状态</p>",
            "<p>暂无特征权重数据</p>"
        )

def build_training_tab(model_service, data_service):
    with gr.Blocks() as training_tab:
        gr.Markdown("""### 📊 第三部分：数据回收训练""")
        
        with gr.Row():
            with gr.Column(scale=2):
                train_btn = gr.Button("🚀 开始训练", variant="primary")
                clear_data_btn = gr.Button("🗑️ 清空数据", variant="secondary")
                export_data_btn = gr.Button("📤 导出数据", variant="secondary")
                import_data_btn = gr.Button("📥 导入数据", variant="secondary")
                
            with gr.Column(scale=3):
                data_stats_output = gr.HTML(value="<p>点击按钮查看数据统计...</p>", label="数据统计")
        
        training_output = gr.HTML(value="<p>点击开始训练按钮进行模型训练...</p>", label="训练结果")
        
        sample_output = gr.Dataframe(
            headers=None,
            label="CTR样本数据",
            interactive=False
        )
        
        # 绑定事件
        def show_data_stats():
            # 使用新的工具函数获取统计信息
            stats = get_data_statistics()
            
            # 获取点击模式分析
            patterns = analyze_click_patterns()
            
            html = f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0; color: #333;">📊 CTR数据统计</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>总样本数:</strong> {stats['total_samples']}</li>
                    <li><strong>总点击数:</strong> {stats['total_clicks']}</li>
                    <li><strong>点击率:</strong> {stats['click_rate']:.2%}</li>
                    <li><strong>唯一查询数:</strong> {stats['unique_queries']}</li>
                    <li><strong>唯一文档数:</strong> {stats['unique_docs']}</li>
                    <li><strong>缓存状态:</strong> {'命中' if stats.get('cache_hit', False) else '未命中'}</li>
                </ul>
            </div>
            """
            
            # 如果有点击模式分析结果，添加到显示中
            if 'error' not in patterns:
                html += f"""
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 10px;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">🔍 点击模式分析</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>整体CTR:</strong> {patterns['overall_ctr']:.2%}</li>
                        <li><strong>总展示数:</strong> {patterns['total_impressions']}</li>
                        <li><strong>总点击数:</strong> {patterns['total_clicks']}</li>
                    </ul>
                </div>
                """
            
            return html
        
        def train_model():
            result = model_service.train_model(data_service)
            
            if result.get('success', False):
                html = f"""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb;">
                    <h4 style="margin: 0 0 10px 0;">✅ 训练成功</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>准确率:</strong> {result.get('accuracy', 0):.4f}</li>
                        <li><strong>AUC:</strong> {result.get('auc', 0):.4f}</li>
                        <li><strong>精确率:</strong> {result.get('precision', 0):.4f}</li>
                        <li><strong>召回率:</strong> {result.get('recall', 0):.4f}</li>
                        <li><strong>F1分数:</strong> {result.get('f1', 0):.4f}</li>
                    </ul>
                </div>
                """
            else:
                html = f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <h4 style="margin: 0 0 10px 0;">❌ 训练失败</h4>
                    <p style="margin: 0;">{result.get('error', '未知错误')}</p>
                </div>
                """
            
            return html
        
        def clear_data():
            # 使用新的工具函数
            clear_all_data()
            return "<p style='color: green;'>✅ 数据已清空</p>"
        
        def export_data():
            import os
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ctr_data_export_{timestamp}.json"
            filepath = os.path.join("data", filename)
            
            os.makedirs("data", exist_ok=True)
            # 使用新的工具函数
            if export_ctr_data(filepath):
                return f"<p style='color: green;'>✅ 数据导出成功: {filename}</p>"
            else:
                return "<p style='color: red;'>❌ 数据导出失败</p>"
        
        def import_data(file):
            if file is None:
                return "<p style='color: red;'>❌ 请选择要导入的文件</p>"
            
            # 使用新的工具函数
            if import_ctr_data(file.name):
                return "<p style='color: green;'>✅ 数据导入成功</p>"
            else:
                return "<p style='color: red;'>❌ 数据导入失败</p>"
        
        def refresh_samples():
            # 使用新的工具函数
            return get_ctr_dataframe()
        
        # 绑定事件
        train_btn.click(fn=train_model, outputs=training_output)
        clear_data_btn.click(fn=clear_data, outputs=training_output)
        export_data_btn.click(fn=export_data, outputs=training_output)
        
        import_file = gr.File(label="选择要导入的JSON文件")
        import_data_btn.click(fn=import_data, inputs=[import_file], outputs=training_output)
        
        # 显示数据统计
        show_data_stats_btn = gr.Button("📊 显示数据统计", variant="secondary")
        show_data_stats_btn.click(fn=show_data_stats, outputs=data_stats_output)
        
        # 刷新样本数据
        refresh_btn = gr.Button("🔄 刷新样本数据", variant="secondary")
        refresh_btn.click(fn=refresh_samples, outputs=sample_output)
        
        # 初始化样本数据
        sample_output.value = get_ctr_dataframe()
        # 兼容性方案：Tab构建后自动触发一次刷新按钮（如果有refresh_btn）
        # 或者在Blocks外部用gradio的on()事件（如支持）
        # 这里保留初始化赋值，用户切换Tab后如需刷新可手动点击刷新按钮
        
    return training_tab 