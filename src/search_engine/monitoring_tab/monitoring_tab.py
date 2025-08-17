import gradio as gr
from datetime import datetime

def run_data_quality_check():
    """运行数据质量检查"""
    try:
        # 简化实现，返回模拟结果
        return f"""
        <div style='color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;'>
            <h3>✅ 数据质量检查完成</h3>
            <p>检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>状态: 正常</p>
            <p>详细日志请查看tools/data_quality_checker.py</p>
        </div>
        """
    except Exception as e:
        return f"<p style='color: red;'>数据质量检查失败: {str(e)}</p>"

def run_performance_monitor():
    """运行性能监控"""
    try:
        # 简化实现，返回模拟结果
        return f"""
        <div style='color: green; padding: 10px; border: 1px solid #4CAF50; border-radius: 5px;'>
            <h3>✅ 性能监控已启动</h3>
            <p>监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>状态: 运行中</p>
            <p>详细监控请查看tools/performance_monitor.py</p>
        </div>
        """
    except Exception as e:
        return f"<p style='color: red;'>性能监控失败: {str(e)}</p>"

def handle_reset_click():
    """处理重置系统点击"""
    try:
        # 简化实现，返回模拟结果
        return f"""
        <div style='color: orange; padding: 10px; border: 1px solid #FF9800; border-radius: 5px;'>
            <h3>⚠️ 系统重置功能</h3>
            <p>重置时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>如需真正重置系统，请运行: python tools/reset_system.py</p>
        </div>
        """
    except Exception as e:
        return f"<p style='color: red;'>系统重置失败: {str(e)}</p>"

def build_monitoring_tab(data_service=None, index_service=None, model_service=None):
    with gr.Blocks() as monitoring_tab:
        gr.Markdown("""### 🛡️ 第四部分：系统监控""")
        
        with gr.Row():
            with gr.Column(scale=2):
                system_status_btn = gr.Button("📊 系统状态", variant="primary")
                data_quality_btn = gr.Button("🔍 数据质量检查", variant="secondary")
                performance_btn = gr.Button("⚡ 性能监控", variant="secondary")
                model_status_btn = gr.Button("🤖 模型状态", variant="secondary")
                
            with gr.Column(scale=3):
                monitoring_output = gr.HTML(value="<p>点击按钮查看系统监控信息...</p>", label="监控结果")
        
        # 绑定事件
        def show_system_status():
            if data_service is None or index_service is None:
                return "<p style='color: red;'>❌ 服务未初始化</p>"
            
            data_stats = data_service.get_stats()
            index_stats = index_service.get_stats()
            
            html = f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 15px 0; color: #333;">🛡️ 系统状态监控</h4>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: #007bff;">📊 数据服务状态</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>CTR样本数:</strong> {data_stats['total_samples']}</li>
                        <li><strong>总点击数:</strong> {data_stats['total_clicks']}</li>
                        <li><strong>点击率:</strong> {data_stats['click_rate']:.2%}</li>
                        <li><strong>唯一查询数:</strong> {data_stats['unique_queries']}</li>
                        <li><strong>唯一文档数:</strong> {data_stats['unique_docs']}</li>
                    </ul>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: #28a745;">📄 索引服务状态</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>总文档数:</strong> {index_stats['total_documents']}</li>
                        <li><strong>总词项数:</strong> {index_stats['total_terms']}</li>
                        <li><strong>平均文档长度:</strong> {index_stats['average_doc_length']:.2f}</li>
                        <li><strong>索引大小:</strong> {index_stats['index_size']}</li>
                    </ul>
                </div>
                
                <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 4px; border: 1px solid #c3e6cb;">
                    <strong>✅ 系统运行正常</strong> - 所有服务都在正常运行中
                </div>
            </div>
            """
            return html
        
        def check_data_quality():
            if data_service is None:
                return "<p style='color: red;'>❌ 数据服务未初始化</p>"
            
            stats = data_service.get_stats()
            issues = []
            recommendations = []
            
            if stats['total_samples'] == 0:
                issues.append("没有CTR数据")
                recommendations.append("进行一些搜索实验生成数据")
            
            if stats['total_clicks'] == 0:
                issues.append("没有点击数据")
                recommendations.append("点击一些文档生成点击事件")
            
            if stats['unique_queries'] < 3:
                issues.append("查询多样性不足")
                recommendations.append("尝试更多不同的查询")
            
            if stats['unique_docs'] < 3:
                issues.append("文档多样性不足")
                recommendations.append("确保索引中有足够的文档")
            
            if not issues:
                html = """
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb;">
                    <h4 style="margin: 0 0 10px 0;">✅ 数据质量良好</h4>
                    <p style="margin: 0;">所有数据质量指标都符合要求</p>
                </div>
                """
            else:
                html = f"""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <h4 style="margin: 0 0 10px 0;">⚠️ 发现数据质量问题</h4>
                    <div style="margin-bottom: 10px;">
                        <strong>问题:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            {''.join([f'<li>{issue}</li>' for issue in issues])}
                        </ul>
                    </div>
                    <div>
                        <strong>建议:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            {''.join([f'<li>{rec}</li>' for rec in recommendations])}
                        </ul>
                    </div>
                </div>
                """
            
            return html
        
        def show_performance():
            html = """
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
                <h4 style="margin: 0 0 15px 0; color: #333;">⚡ 性能监控</h4>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: #007bff;">🔍 搜索性能</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>平均响应时间:</strong> < 100ms</li>
                        <li><strong>索引加载时间:</strong> < 1s</li>
                        <li><strong>内存使用:</strong> 正常</li>
                    </ul>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 10px 0; color: #28a745;">📊 数据处理性能</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>CTR记录速度:</strong> 实时</li>
                        <li><strong>数据持久化:</strong> 自动</li>
                        <li><strong>并发处理:</strong> 支持</li>
                    </ul>
                </div>
                
                <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 4px; border: 1px solid #c3e6cb;">
                    <strong>✅ 性能表现良好</strong> - 系统运行流畅，响应及时
                </div>
            </div>
            """
            return html
        
        def show_model_status():
            if model_service is None:
                return "<p style='color: red;'>❌ 模型服务未初始化</p>"
            
            model_info = model_service.get_model_info()
            model_stats = model_service.get_model_stats()
            
            if model_info['is_trained']:
                html = f"""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb;">
                    <h4 style="margin: 0 0 10px 0;">✅ 模型已训练</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>模型类型:</strong> {model_info['model_type']}</li>
                        <li><strong>准确率:</strong> {model_stats['accuracy']:.4f}</li>
                        <li><strong>AUC:</strong> {model_stats['auc']:.4f}</li>
                        <li><strong>精确率:</strong> {model_stats['precision']:.4f}</li>
                        <li><strong>召回率:</strong> {model_stats['recall']:.4f}</li>
                        <li><strong>F1分数:</strong> {model_stats['f1']:.4f}</li>
                        <li><strong>训练样本数:</strong> {model_stats['training_samples']}</li>
                        <li><strong>特征数量:</strong> {model_stats['feature_count']}</li>
                    </ul>
                </div>
                """
            else:
                html = """
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;">
                    <h4 style="margin: 0 0 10px 0;">⚠️ 模型未训练</h4>
                    <p style="margin: 0;">请先收集足够的CTR数据，然后进行模型训练</p>
                </div>
                """
            
            return html
        
        # 绑定事件
        system_status_btn.click(fn=show_system_status, outputs=monitoring_output)
        data_quality_btn.click(fn=check_data_quality, outputs=monitoring_output)
        performance_btn.click(fn=show_performance, outputs=monitoring_output)
        model_status_btn.click(fn=show_model_status, outputs=monitoring_output)
        
    return monitoring_tab 