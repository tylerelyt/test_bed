import gradio as gr
from datetime import datetime
import os
import json
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

import jieba

# 延迟加载 gensim，并在缺失时自动安装（避免需要重启）
import importlib
import subprocess
import sys
Word2Vec = None

def ensure_gensim(auto_install: bool = True):
    """确保 gensim.Word2Vec 可用；必要时自动安装。
    Returns: (ok: bool, detail: str)
    """
    global Word2Vec
    if Word2Vec is not None:
        return True, "loaded"
    try:
        Word2Vec = importlib.import_module("gensim.models").Word2Vec
        return True, "imported"
    except Exception:
        if not auto_install:
            return False, "missing"
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gensim>=4.3.0"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            Word2Vec = importlib.import_module("gensim.models").Word2Vec
            return True, "installed"
        except Exception as e:
            return False, f"install failed: {e}"

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        CLIPProcessor,
        CLIPModel,
    )
    from PIL import Image
except Exception:
    torch = None
    AutoTokenizer = AutoModel = AutoModelForCausalLM = None
    CLIPProcessor = CLIPModel = None
    Image = None

from ..data_utils import (
    get_data_statistics,
    get_ctr_dataframe,
    clear_all_data,
    export_ctr_data,
    import_ctr_data,
    analyze_click_patterns
)
from .ctr_config import CTRModelConfig

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

def create_model_instance(model_type: str):
    """根据模型类型创建模型实例"""
    try:
        model_config = CTRModelConfig.get_model_config(model_type)
        if not model_config:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        module_name = model_config['module']
        class_name = model_config['class']
        
        if model_type == 'logistic_regression':
            from .ctr_model import CTRModel
            return CTRModel()
        elif model_type == 'wide_and_deep':
            from .ctr_wide_deep_model import WideAndDeepCTRModel
            return WideAndDeepCTRModel()
        else:
            raise ValueError(f"未实现的模型类型: {model_type}")
    except Exception as e:
        print(f"创建模型实例失败: {e}")
        # 回退到默认LR模型
        from .ctr_model import CTRModel
        return CTRModel()

def train_ctr_model_direct(ctr_model, data_service, model_type: str = "logistic_regression"):
    """直接使用data_service训练CTR模型"""
    try:
        # 获取训练数据
        records = data_service.get_all_samples()
        
        if len(records) < 10:
            return (
                "<p style='color: orange;'>⚠️ 训练数据不足，至少需要10条记录</p>",
                "<p>请先进行一些搜索和点击操作收集数据</p>",
                "<p>暂无特征权重数据</p>"
            )
        
        # 根据选择的模型类型创建模型实例
        if model_type != "logistic_regression":
            model_instance = create_model_instance(model_type)
        else:
            model_instance = ctr_model
        
        # 训练模型
        result = model_instance.train(records)
        
        if 'error' in result:
            return (
                f"<p style='color: red;'>❌ 训练失败: {result['error']}</p>",
                "<p>请检查数据质量</p>",
                "<p>暂无特征权重数据</p>"
            )
        
        # 获取模型配置信息
        model_config = CTRModelConfig.get_model_config(model_type)
        model_name = model_config.get('name', model_type)
        
        # 生成训练结果HTML
        model_status = f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
            <h4>✅ CTR模型训练成功</h4>
            <p><strong>模型类型:</strong> {model_name}</p>
            <p><strong>AUC:</strong> {result.get('auc', 0):.4f}</p>
            <p><strong>准确率:</strong> {result.get('accuracy', 0):.4f}</p>
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
        feature_importance = result.get('feature_importance', {})
        
        # 合并特征权重和重要性
        all_features = {**feature_weights, **feature_importance}
        
        if all_features:
            weights_html = "<h4>🎯 特征重要性分析</h4><ul>"
            sorted_weights = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
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

def train_ctr_model(ctr_model, ctr_collector, model_type: str = "logistic_regression"):
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
        
        # 根据选择的模型类型创建模型实例
        if model_type != "logistic_regression":
            model_instance = create_model_instance(model_type)
        else:
            model_instance = ctr_model
        
        # 训练模型
        result = model_instance.train(records)
        
        if 'error' in result:
            return (
                f"<p style='color: red;'>❌ 训练失败: {result['error']}</p>",
                "<p>请检查数据质量</p>",
                "<p>暂无特征权重数据</p>"
            )
        
        # 获取模型配置信息
        model_config = CTRModelConfig.get_model_config(model_type)
        model_name = model_config.get('name', model_type)
        
        # 生成训练结果HTML
        model_status = f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
            <h4>✅ CTR模型训练成功</h4>
            <p><strong>模型类型:</strong> {model_name}</p>
            <p><strong>AUC:</strong> {result.get('auc', 0):.4f}</p>
            <p><strong>准确率:</strong> {result.get('accuracy', 0):.4f}</p>
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
        feature_importance = result.get('feature_importance', {})
        
        # 合并特征权重和重要性
        all_features = {**feature_weights, **feature_importance}
        
        if all_features:
            weights_html = "<h4>🎯 特征重要性分析</h4><ul>"
            sorted_weights = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
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
        gr.Markdown("""### 📊 第三部分：模型训练与实验""")
        
        with gr.Tabs():
            # LLMOps 闭环系统标签页
            try:
                with gr.Tab("🔄 LLMOps 闭环"):
                    from .llmops_tab import build_llmops_content
                    train_engine = build_llmops_content()
            except Exception as e:
                print(f"❌ LLMOps 标签页加载失败: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                # 创建一个错误提示标签页
                with gr.Tab("🔄 LLMOps 闭环"):
                    gr.Markdown(f"""
                    ## ❌ LLMOps 标签页加载失败
                    
                    **错误类型**: {type(e).__name__}
                    
                    **错误信息**: {str(e)}
                    
                    请检查日志获取详细信息。
                    """)
                train_engine = None
            
            # CTR模型训练标签页
            with gr.Tab("🎯 CTR模型训练"):
                gr.Markdown("#### 点击率预测模型训练")
                
                # 在线学习配置区域
                with gr.Accordion("🔄 在线学习配置", open=True):
                    gr.Markdown("""
                    **在线学习模式**: 自动检测新增点击数据并触发模型训练，实时优化模型性能
                    - ✅ **离线训练**: 手动触发，全量训练，保存到 `models/offline/`
                    - 🔄 **在线训练**: 自动触发，增量训练，保存到 `models/online/`（保留最近5个checkpoint）
                    """)
                    with gr.Row():
                        online_learning_enabled = gr.Checkbox(
                            label="启用在线学习",
                            value=False,
                            info="开启后，每当新增一定数量的点击数据时自动触发训练"
                        )
                        online_training_threshold = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="训练触发阈值",
                            info="每新增N条点击数据触发一次在线训练"
                        )
                    
                    online_status_output = gr.HTML(
                        value="<p style='color: gray;'>⚪ 在线学习未启用</p>",
                        label="在线学习状态"
                    )
                
                # 模型选择区域
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎯 模型选择")
                        
                        # 获取支持的模型
                        model_choices = CTRModelConfig.get_model_names()
                        model_labels = [f"{config['name']} - {config['description']}" 
                                       for config in CTRModelConfig.get_supported_models().values()]
                        model_keys = list(CTRModelConfig.get_supported_models().keys())
                        
                        model_dropdown = gr.Dropdown(
                            choices=[(label, key) for label, key in zip(model_labels, model_keys)],
                            value="logistic_regression",
                            label="选择CTR模型",
                            info="选择要训练的CTR模型类型"
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        train_btn = gr.Button("🚀 开始离线训练", variant="primary")
                        clear_data_btn = gr.Button("🗑️ 清空数据", variant="secondary")
                        export_data_btn = gr.Button("📤 导出数据", variant="secondary")
                        
                    with gr.Column(scale=3):
                        data_stats_output = gr.HTML(value="<p>点击按钮查看数据统计...</p>", label="数据统计")
                
                # 数据管理按钮
                with gr.Row():
                    show_data_stats_btn = gr.Button("📊 显示数据统计", variant="secondary")
                    refresh_btn = gr.Button("🔄 刷新样本数据", variant="secondary")
                
                training_output = gr.HTML(value="<p>点击开始训练按钮进行模型训练...</p>", label="训练结果")
                train_details = gr.HTML(value="<p>训练详情将在这里显示...</p>", label="训练详情")
                feature_weights = gr.HTML(value="<p>特征重要性将在这里显示...</p>", label="特征重要性")
                
                # 模型评估与分析
                gr.Markdown("---")
                gr.Markdown("#### 📚 模型评估与分析")
                with gr.Tabs():
                    # 交叉验证标签页
                    with gr.Tab("📊 交叉验证"):
                        gr.Markdown("**功能**: 使用K折交叉验证评估模型在不同数据子集上的性能，了解模型的泛化能力")
                        with gr.Row():
                            cv_folds = gr.Slider(3, 10, value=5, step=1, label="交叉验证折数")
                            cv_btn = gr.Button("🚀 执行交叉验证", variant="primary")
                        cv_output = gr.HTML(value="<p>点击按钮执行交叉验证...</p>", label="交叉验证结果")
                    
                    # 可解释性分析标签页
                    with gr.Tab("🔍 可解释性分析"):
                        gr.Markdown("**功能**: 使用LIME和SHAP分析模型预测的原因和特征重要性")
                        with gr.Row():
                            with gr.Column():
                                interpret_method = gr.Radio(
                                    choices=["LIME", "SHAP", "特征重要性"],
                                    value="LIME",
                                    label="解释方法"
                                )
                                num_features = gr.Slider(5, 20, value=10, step=1, label="显示特征数")
                            with gr.Column():
                                interpret_btn = gr.Button("🔍 分析模型可解释性", variant="primary")
                        interpret_output = gr.HTML(value="<p>点击按钮进行可解释性分析...</p>", label="解释结果")
                    
                    # 公平性分析标签页
                    with gr.Tab("⚖️ 公平性分析"):
                        gr.Markdown("**功能**: 评估模型在不同群体上的性能差异，初步了解模型公平性")
                        with gr.Row():
                            fairness_group_by = gr.Dropdown(
                                choices=["position_range", "query", "doc_id", "score_range"],
                                value="position_range",
                                label="分组依据"
                            )
                            fairness_btn = gr.Button("⚖️ 分析模型公平性", variant="primary")
                        fairness_output = gr.HTML(value="<p>点击按钮进行公平性分析...</p>", label="公平性分析结果")
                    
                    # AutoML标签页
                    with gr.Tab("🤖 AutoML"):
                        gr.Markdown("**功能**: 使用AutoML工具进行模型搜索和超参数优化")
                        with gr.Row():
                            with gr.Column():
                                automl_method = gr.Radio(
                                    choices=["网格搜索", "Optuna优化"],
                                    value="网格搜索",
                                    label="优化方法"
                                )
                                automl_cv = gr.Slider(3, 10, value=3, step=1, label="交叉验证折数")
                            with gr.Column():
                                automl_btn = gr.Button("🤖 执行AutoML优化", variant="primary")
                        automl_output = gr.HTML(value="<p>点击按钮执行AutoML优化...</p>", label="AutoML结果")
                
                sample_output = gr.Dataframe(
                    headers=None,
                    label="CTR样本数据",
                    interactive=False
                )
        
            # 词表示：Word2Vec
            with gr.Tab("🧩 词表示 · Word2Vec"):
                gr.Markdown("从预置文档训练一个 Word2Vec 词向量模型，并查询近义词。")

                with gr.Row():
                    with gr.Column(scale=1):
                        w2v_vector_size = gr.Slider(50, 300, value=128, step=8, label="向量维度")
                        w2v_window = gr.Slider(2, 10, value=5, step=1, label="窗口大小")
                        w2v_min_count = gr.Slider(1, 5, value=2, step=1, label="最小词频")
                        w2v_epochs = gr.Slider(1, 10, value=3, step=1, label="训练轮次")
                        train_w2v_btn = gr.Button("🚀 训练 Word2Vec", variant="primary")
                    with gr.Column(scale=1):
                        query_word = gr.Textbox(label="查询词", placeholder="输入词语，查看近义词")
                        w2v_topk = gr.Slider(3, 20, value=10, step=1, label="TopK")
                        w2v_query_btn = gr.Button("🔎 查询近义词")
                w2v_status = gr.HTML(value="<p>尚未训练</p>")
                w2v_result = gr.Dataframe(headers=["词", "相似度"], interactive=False)

                # Word2Vec自监督学习数据格式可视化
                gr.Markdown("#### 📊 Word2Vec自监督学习数据格式")
                gr.Markdown("**CBOW任务**: 使用上下文预测中心词（Word2Vec自监督学习之一）")
                gr.Markdown("**Skip-gram任务**: 给定中心词，预测周围上下文词（Word2Vec自监督学习之一）")
                with gr.Row():
                    bow_top = gr.Dataframe(headers=["输入", "目标"], label="CBOW自监督任务样本", interactive=False)
                    skipgram_pairs = gr.Dataframe(headers=["输入", "目标"], label="Skip-gram自监督任务样本", interactive=False)
                run_w2v_viz_btn = gr.Button("🔎 查看Word2Vec自监督数据格式", variant="secondary")

            # 句子表示：BERT
            with gr.Tab("🧠 句子表示 · BERT"):
                gr.Markdown("使用 BERT 预训练模型抽取句子向量。默认使用 `bert-base-chinese`，可输入两句对比余弦相似度。")
                with gr.Row():
                    with gr.Column(scale=1):
                        bert_model_name = gr.Textbox(value="bert-base-chinese", label="模型名")
                        load_bert_btn = gr.Button("📦 加载模型", variant="secondary")
                    with gr.Column(scale=2):
                        sent_a = gr.Textbox(label="句子A", value="我喜欢人工智能")
                        sent_b = gr.Textbox(label="句子B", value="我热爱机器学习")
                        run_bert_btn = gr.Button("🔎 计算相似度", variant="primary")
                bert_status = gr.HTML(value="<p>模型未加载</p>")
                bert_similarity = gr.HTML(value="<p>相似度将在这里显示</p>")

            # 生成模型：OPT（生成 + 预置文档上的CLM微调演示）
            with gr.Tab("✍️ 生成模型 · OPT"):
                gr.Markdown("使用 `facebook/opt-125m` 进行文本生成，并在预置文档上做少量 CLM 训练演示（仅展示方法，CPU 少步数）。")
                with gr.Row():
                    with gr.Column(scale=1):
                        opt_model_name = gr.Textbox(value="facebook/opt-125m", label="模型名")
                        load_opt_btn = gr.Button("📦 加载模型", variant="secondary")
                        # CLM 训练参数
                        train_steps = gr.Slider(1, 50, value=5, step=1, label="训练步数")
                        lr_opt = gr.Slider(1e-6, 5e-5, value=1e-5, step=1e-6, label="学习率")
                        block_size = gr.Slider(64, 512, value=256, step=32, label="序列长度")
                        batch_size = gr.Slider(1, 4, value=1, step=1, label="批大小")
                        train_opt_btn = gr.Button("🎓 用预置文档做CLM训练(演示)", variant="primary")
                    with gr.Column(scale=2):
                        opt_prompt = gr.Textbox(label="Prompt", value="今天我学习了信息检索，它是……", lines=4)
                        max_new_tokens = gr.Slider(16, 128, value=64, step=8, label="最大生成长度")
                        gen_btn = gr.Button("📝 生成文本", variant="secondary")
                opt_status = gr.HTML(value="<p>模型未加载</p>")
                opt_output = gr.HTML(value="<p>生成结果将在这里显示</p>")

                # OPT Causal Language Modeling 自监督学习数据格式可视化
                gr.Markdown("#### 📊 OPT自监督学习：Causal Language Modeling (CLM)")
                gr.Markdown("**CLM任务**: 给定前文序列，预测下一个token（因果语言建模，自监督学习核心）")
                clm_pairs_df = gr.Dataframe(headers=["输入序列", "预测目标", "token_id", "位置"], label="CLM自监督任务样本 (Next Token Prediction)", interactive=False)
                run_clm_viz_btn = gr.Button("🔎 查看OPT自监督数据格式", variant="secondary")

            # 多模态：CLIP 对比学习微调（演示）
            with gr.Tab("🖼️🔤 多模态 · CLIP 微调"):
                gr.Markdown("#### CLIP多模态对比学习")
                gr.Markdown("**对比学习**: 同时训练文本编码器和图像编码器，使匹配的文本-图像对在嵌入空间中更相似（正样本），不匹配的更远离（负样本）")
                gr.Markdown("**自监督任务**: 大量图文对无需人工标注，通过对比损失自动学习跨模态表示")
                clip_info = gr.HTML(value="<p>基于内置图文对演示CLIP对比学习，CPU环境训练较慢。</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_model_name = gr.Textbox(value="openai/clip-vit-base-patch32", label="模型名")
                        load_clip_btn = gr.Button("📦 加载模型", variant="secondary")
                    with gr.Column(scale=2):
                        clip_train_btn = gr.Button("🎓 演示对比学习微调", variant="primary")
                clip_status = gr.HTML(value="<p>模型未加载</p>")
                clip_log = gr.HTML(value="<p>对比学习训练日志将在这里显示</p>")
                
                # CLIP对比学习数据格式说明与可视化
                gr.Markdown("#### 📊 CLIP对比学习数据格式")
                gr.Markdown("- **正样本对**: (图像, 匹配描述文本) → 拉近嵌入距离")
                gr.Markdown("- **负样本对**: (图像, 不匹配文本) → 推远嵌入距离")
                gr.Markdown("- **批内对比**: 一个batch内，每个图像与所有文本计算相似度矩阵，对角线为正样本")
                
                # CLIP训练数据可视化
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_data_viz = gr.Dataframe(headers=["图片路径", "匹配文本", "数据类型"], label="CLIP图文对训练数据", interactive=False)
                        viz_clip_data_btn = gr.Button("🔎 查看CLIP训练数据格式", variant="secondary")
                    with gr.Column(scale=1):
                        clip_image_gallery = gr.Gallery(label="训练图片预览", show_label=True, elem_id="clip_gallery", columns=2, rows=2, object_fit="contain", height="400px")
                        clip_text_display = gr.HTML(value="<p>图片描述将在这里显示</p>", label="对应文本描述")

        # 绑定事件（CTR）
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
        
        def train_model_with_selection(selected_model):
            # 使用新的训练函数，支持模型选择
            try:
                ctr_model = model_service.ctr_model if hasattr(model_service, 'ctr_model') else None
                
                # data_service本身就是数据收集器，不需要ctr_collector属性
                if not ctr_model:
                    return (
                        "<p style='color: red;'>❌ CTR模型不可用</p>",
                        "<p>请检查系统状态</p>",
                        "<p>暂无特征重要性数据</p>"
                    )
                
                return train_ctr_model_direct(ctr_model, data_service, selected_model)
            except Exception as e:
                return (
                    f"<p style='color: red;'>❌ 训练函数调用失败: {str(e)}</p>",
                    "<p>请检查系统状态</p>",
                    "<p>暂无特征重要性数据</p>"
                )
        
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
        
        # 删除导入功能：仅保留导出/清空
        
        def refresh_samples():
            # 使用新的工具函数
            return get_ctr_dataframe()
        
        def toggle_online_learning(enabled, threshold):
            """切换在线学习开关"""
            try:
                # 启用/禁用在线学习
                model_service.enable_online_learning(enabled)
                
                # 设置训练触发阈值
                data_service.set_online_training_threshold(int(threshold))
                
                # 关联模型服务到数据服务（如果还没有关联）
                if not data_service.model_service:
                    data_service.set_model_service(model_service)
                
                if enabled:
                    # 获取当前在线模型状态
                    latest_checkpoint = model_service._get_latest_online_checkpoint()
                    checkpoint_info = f"当前checkpoint: #{latest_checkpoint}" if latest_checkpoint else "尚无在线checkpoint"
                    
                    return f"""
                    <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                        <p style='color: #28a745; margin: 0;'><strong>🟢 在线学习已启用</strong></p>
                        <p style='margin: 5px 0 0 0;'><small>触发阈值: 每{int(threshold)}条新点击数据</small></p>
                        <p style='margin: 5px 0 0 0;'><small>{checkpoint_info}</small></p>
                        <p style='margin: 5px 0 0 0;'><small>模型保存: models/online/ (保留最近5个)</small></p>
                    </div>
                    """
                else:
                    return "<p style='color: gray;'>⚪ 在线学习未启用</p>"
            except Exception as e:
                return f"<p style='color: red;'>❌ 切换失败: {str(e)}</p>"
        
        # 绑定事件
        # 在线学习开关事件
        online_learning_enabled.change(
            fn=toggle_online_learning,
            inputs=[online_learning_enabled, online_training_threshold],
            outputs=[online_status_output]
        )
        online_training_threshold.change(
            fn=toggle_online_learning,
            inputs=[online_learning_enabled, online_training_threshold],
            outputs=[online_status_output]
        )
        
        train_btn.click(
            fn=train_model_with_selection, 
            inputs=[model_dropdown], 
            outputs=[training_output, train_details, feature_weights]
        )
        clear_data_btn.click(fn=clear_data, outputs=training_output)
        export_data_btn.click(fn=export_data, outputs=training_output)
        
        # 已移除导入控件与事件绑定
        
        # 绑定数据管理按钮事件
        show_data_stats_btn.click(fn=show_data_stats, outputs=data_stats_output)
        refresh_btn.click(fn=refresh_samples, outputs=sample_output)
        
        # ============ 模型评估与分析 ============
        from .model_evaluation import ModelEvaluator
        from .model_interpretability import ModelInterpretability
        from .model_fairness import ModelFairnessAnalyzer
        from .model_automl import AutoMLOptimizer
        from sklearn.linear_model import LogisticRegression
        
        evaluator = ModelEvaluator()
        interpretability = ModelInterpretability()
        fairness_analyzer = ModelFairnessAnalyzer()
        automl_optimizer = AutoMLOptimizer()
        
        def run_cross_validation(folds):
            """执行交叉验证"""
            try:
                ctr_model = model_service.ctr_model if hasattr(model_service, 'ctr_model') else None
                if not ctr_model or not ctr_model.is_trained:
                    return "<p style='color: orange;'>⚠️ 请先训练模型</p>"
                
                # 获取训练数据
                records = data_service.get_all_samples()
                
                if len(records) < folds * 2:
                    return f"<p style='color: red;'>❌ 数据量不足，至少需要{folds * 2}条记录</p>"
                
                # 执行交叉验证
                result = evaluator.cross_validate_model(
                    ctr_model.model,
                    records,
                    cv_folds=int(folds)
                )
                
                if 'error' in result:
                    return f"<p style='color: red;'>❌ {result['error']}</p>"
                
                # 格式化结果
                html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                html += f"<h4>📊 交叉验证结果（{folds}折）</h4>"
                html += f"<p><strong>样本数:</strong> {result.get('n_samples', 0)}</p>"
                html += f"<p><strong>特征数:</strong> {result.get('n_features', 0)}</p>"
                
                if result.get('cv_mean'):
                    html += "<h5>各指标的平均值和标准差:</h5><table border='1' style='border-collapse: collapse;'>"
                    html += "<tr><th>指标</th><th>平均值</th><th>标准差</th></tr>"
                    for metric, mean_val in result['cv_mean'].items():
                        std_val = result['cv_std'].get(metric, 0)
                        html += f"<tr><td>{metric}</td><td>{mean_val:.4f}</td><td>{std_val:.4f}</td></tr>"
                    html += "</table>"
                
                html += "</div>"
                return html
                
            except Exception as e:
                return f"<p style='color: red;'>❌ 交叉验证失败: {str(e)}</p>"
        
        def run_interpretability_analysis(method, num_feat):
            """执行可解释性分析"""
            try:
                ctr_model = model_service.ctr_model if hasattr(model_service, 'ctr_model') else None
                if not ctr_model or not ctr_model.is_trained:
                    return "<p style='color: orange;'>⚠️ 请先训练模型</p>"
                
                # 获取训练数据
                records = data_service.get_all_samples()
                
                if len(records) < 10:
                    return "<p style='color: red;'>❌ 数据量不足</p>"
                
                # 提取特征
                features, labels = ctr_model.extract_features(records)
                if len(features) == 0:
                    return "<p style='color: red;'>❌ 特征提取失败</p>"
                
                # 标准化
                if ctr_model.scaler:
                    features_scaled = ctr_model.scaler.transform(features)
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                
                # 获取特征名称
                from .ctr_config import CTRFeatureConfig
                feature_names = CTRFeatureConfig.get_feature_names()
                
                if method == "LIME":
                    # 准备LIME解释器
                    success, msg = interpretability.prepare_lime_explainer(
                        features_scaled,
                        feature_names[:len(features_scaled[0])] if len(feature_names) >= len(features_scaled[0]) else [f"特征{i}" for i in range(len(features_scaled[0]))]
                    )
                    if not success:
                        return f"<p style='color: red;'>❌ {msg}</p>"
                    
                    # 解释一个样本
                    sample_idx = 0
                    explanation = interpretability.explain_with_lime(
                        ctr_model.model,
                        features_scaled[sample_idx:sample_idx+1],
                        num_features=int(num_feat)
                    )
                    
                    if 'error' in explanation:
                        return f"<p style='color: red;'>❌ {explanation['error']}</p>"
                    
                    html = f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                    html += f"<h4>🔍 LIME解释结果</h4>"
                    html += f"<p><strong>预测概率:</strong> {explanation.get('prediction', 0):.4f}</p>"
                    html += "<h5>特征贡献:</h5><ul>"
                    for feat in explanation.get('features', [])[:int(num_feat)]:
                        color = "green" if feat['weight'] > 0 else "red"
                        html += f"<li><span style='color: {color};'>{feat['feature']}: {feat['weight']:.4f}</span></li>"
                    html += "</ul></div>"
                    return html
                
                elif method == "SHAP":
                    explanation = interpretability.explain_with_shap(
                        ctr_model.model,
                        features_scaled,
                        max_samples=50
                    )
                    
                    if 'error' in explanation:
                        return f"<p style='color: red;'>❌ {explanation['error']}</p>"
                    
                    html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                    html += "<h4>🔍 SHAP解释结果</h4>"
                    if 'feature_importance_dict' in explanation:
                        html += "<h5>特征重要性（平均绝对SHAP值）:</h5><ul>"
                        sorted_features = sorted(explanation['feature_importance_dict'].items(), key=lambda x: x[1], reverse=True)
                        for feat_name, importance in sorted_features[:int(num_feat)]:
                            html += f"<li><strong>{feat_name}:</strong> {importance:.4f}</li>"
                        html += "</ul>"
                    html += "</div>"
                    return html
                
                else:  # 特征重要性
                    importance = interpretability.get_feature_importance_from_model(ctr_model.model)
                    html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                    html += "<h4>🔍 模型特征重要性</h4>"
                    if 'by_feature' in importance and 'coefficients' in importance['by_feature']:
                        html += "<h5>特征系数（绝对值）:</h5><ul>"
                        sorted_features = sorted(importance['by_feature']['coefficients'].items(), key=lambda x: abs(x[1]), reverse=True)
                        for feat_name, coef in sorted_features[:int(num_feat)]:
                            html += f"<li><strong>{feat_name}:</strong> {coef:.4f}</li>"
                        html += "</ul>"
                    html += "</div>"
                    return html
                    
            except Exception as e:
                import traceback
                return f"<p style='color: red;'>❌ 可解释性分析失败: {str(e)}</p><pre>{traceback.format_exc()[:500]}</pre>"
        
        def run_fairness_analysis(group_by):
            """执行公平性分析"""
            try:
                ctr_model = model_service.ctr_model if hasattr(model_service, 'ctr_model') else None
                if not ctr_model or not ctr_model.is_trained:
                    return "<p style='color: orange;'>⚠️ 请先训练模型</p>"
                
                # 获取训练数据
                records = data_service.get_all_samples()
                
                if len(records) < 20:
                    return "<p style='color: red;'>❌ 数据量不足，至少需要20条记录</p>"
                
                # 执行公平性分析
                result = fairness_analyzer.analyze_fairness(
                    ctr_model.model,
                    records,
                    group_by=group_by,
                    model_instance_extract_features=ctr_model.extract_features
                )
                
                if 'error' in result:
                    return f"<p style='color: red;'>❌ {result['error']}</p>"
                
                # 生成报告
                report = fairness_analyzer.generate_fairness_report(result)
                return f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>{report}</div>"
                
            except Exception as e:
                import traceback
                return f"<p style='color: red;'>❌ 公平性分析失败: {str(e)}</p><pre>{traceback.format_exc()[:500]}</pre>"
        
        def run_automl_optimization(method, cv_folds):
            """执行AutoML优化"""
            try:
                # 获取训练数据
                records = data_service.get_all_samples()
                
                if len(records) < 30:
                    return "<p style='color: red;'>❌ 数据量不足，至少需要30条记录</p>"
                
                # 提取特征
                ctr_model = model_service.ctr_model if hasattr(model_service, 'ctr_model') else None
                if not ctr_model:
                    return "<p style='color: red;'>❌ CTR模型不可用</p>"
                
                features, labels = ctr_model.extract_features(records)
                if len(features) == 0:
                    return "<p style='color: red;'>❌ 特征提取失败</p>"
                
                # 标准化
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
                
                if method == "网格搜索":
                    # 定义参数网格
                    param_grid = {
                        'C': [0.1, 1.0, 10.0],
                        'max_iter': [500, 1000],
                        'solver': ['liblinear', 'lbfgs']
                    }
                    
                    result = automl_optimizer.simple_grid_search(
                        LogisticRegression,
                        X_scaled,
                        labels,
                        param_grid,
                        cv=int(cv_folds)
                    )
                    
                    if 'error' in result:
                        return f"<p style='color: red;'>❌ {result['error']}</p>"
                    
                    html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                    html += "<h4>🤖 网格搜索优化结果</h4>"
                    html += f"<p><strong>最佳参数:</strong> {result.get('best_params', {})}</p>"
                    html += f"<p><strong>最佳得分:</strong> {result.get('best_score', 0):.4f}</p>"
                    html += "</div>"
                    return html
                
                else:  # Optuna优化
                    param_space = {
                        'C': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
                        'max_iter': {'type': 'int', 'low': 500, 'high': 2000, 'log': False}
                    }
                    
                    result = automl_optimizer.optimize_hyperparameters_with_optuna(
                        LogisticRegression,
                        X_scaled,
                        labels,
                        param_space,
                        n_trials=10,
                        cv=int(cv_folds)
                    )
                    
                    if 'error' in result:
                        return f"<p style='color: red;'>❌ {result['error']}</p>"
                    
                    html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px;'>"
                    html += "<h4>🤖 Optuna优化结果</h4>"
                    html += f"<p><strong>最佳参数:</strong> {result.get('best_params', {})}</p>"
                    html += f"<p><strong>最佳得分:</strong> {result.get('best_score', 0):.4f}</p>"
                    html += f"<p><strong>试验次数:</strong> {result.get('n_trials', 0)}</p>"
                    html += "</div>"
                    return html
                    
            except Exception as e:
                import traceback
                return f"<p style='color: red;'>❌ AutoML优化失败: {str(e)}</p><pre>{traceback.format_exc()[:500]}</pre>"
        
        # 绑定新功能的事件
        cv_btn.click(fn=run_cross_validation, inputs=[cv_folds], outputs=[cv_output])
        interpret_btn.click(fn=run_interpretability_analysis, inputs=[interpret_method, num_features], outputs=[interpret_output])
        fairness_btn.click(fn=run_fairness_analysis, inputs=[fairness_group_by], outputs=[fairness_output])
        automl_btn.click(fn=run_automl_optimization, inputs=[automl_method, automl_cv], outputs=[automl_output])
        
        # 初始化样本数据
        sample_output.value = get_ctr_dataframe()
        # 兼容性方案：Tab构建后自动触发一次刷新按钮（如果有refresh_btn）
        # 或者在Blocks外部用gradio的on()事件（如支持）
        # 这里保留初始化赋值，用户切换Tab后如需刷新可手动点击刷新按钮
        
        # ============ Word2Vec 逻辑 ============
        def _load_preloaded_texts(limit: int = 5000) -> List[List[str]]:
            try:
                preloaded_path = os.path.join("data", "preloaded_documents.json")
                if not os.path.exists(preloaded_path):
                    return []
                with open(preloaded_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = data["documents"] if isinstance(data, dict) and "documents" in data else data
                sentences = []
                # 轻量级中文停用词与清洗
                stop = {
                    "的","了","在","是","和","与","及","并","也","对","中","上","下","为","以",
                    "一个","一种","一些","我们","你们","他们","以及","或者","而且","如果","因为",
                    "可以","通过","进行","使用","没有","包括","这种","这些","那些","由于","由于",
                }
                import re
                zh_re = re.compile(r"[\u4e00-\u9fff]{2,}")
                for i, (_id, content) in enumerate(docs.items() if isinstance(docs, dict) else docs):
                    if i >= limit:
                        break
                    raw_tokens = jieba.lcut(str(content).strip())
                    tokens = []
                    for w in raw_tokens:
                        if w in stop:
                            continue
                        if not zh_re.fullmatch(w):
                            continue
                        tokens.append(w)
                    if tokens:
                        sentences.append(tokens)
                return sentences
            except Exception:
                return []

        w2v_model_holder = {"model": None}

        def train_w2v(vector_size: int, window: int, min_count: int, epochs: int):
            ok, msg = ensure_gensim(auto_install=True)
            if not ok:
                return f"<p style='color:red'>gensim 不可用：{msg}</p>", []
            corpus = _load_preloaded_texts()
            if not corpus:
                return "<p style='color:red'>未找到预置文档或内容为空</p>", []
            model = Word2Vec(
                sentences=corpus,
                vector_size=int(vector_size),
                window=max(5, int(window)),
                min_count=max(2, int(min_count)),
                epochs=max(5, int(epochs)),
                sg=1,
                workers=1,
                seed=42,
            )
            w2v_model_holder["model"] = model
            return f"<p style='color:green'>✅ 训练完成，词表大小: {len(model.wv)}</p>", []

        def query_w2v(word: str, topk: int):
            model = w2v_model_holder.get("model")
            if model is None:
                return "<p style='color:red'>请先训练模型</p>", []
            if not word:
                return "<p style='color:red'>请输入查询词</p>", []
            try:
                sims = model.wv.most_similar(word, topn=int(topk))
                rows = [[w, float(s)] for w, s in sims]
                return "<p>如下为近义词</p>", rows
            except KeyError:
                return f"<p style='color:red'>词 '{word}' 不在词表中</p>", []

        train_w2v_btn.click(train_w2v, inputs=[w2v_vector_size, w2v_window, w2v_min_count, w2v_epochs], outputs=[w2v_status, w2v_result])
        w2v_query_btn.click(query_w2v, inputs=[query_word, w2v_topk], outputs=[w2v_status, w2v_result])
        
        # W2V 预处理可视化：CBOW与Skip-gram的自监督任务样本
        def _bow_and_skipgram(window: int, min_count: int):
            corpus = _load_preloaded_texts(limit=2000)
            if not corpus:
                return [], []
            from collections import Counter
            token_counter = Counter()
            for sent in corpus:
                token_counter.update(sent)

            # CBOW：上下文 → 中心词
            cbow_pairs = []
            win = int(window)
            for sent in corpus[:50]:  # 仅取前若干句做展示
                for i, center in enumerate(sent):
                    # 仅保留出现次数>=min_count的中心词，避免罕见词
                    if token_counter[center] < int(min_count):
                        continue
                    ctx = []
                    for j in range(max(0, i - win), min(len(sent), i + win + 1)):
                        if j == i:
                            continue
                        ctx.append(sent[j])
                    if not ctx:
                        continue
                    cbow_pairs.append([f"上下文:{' '.join(ctx)}", f"中心词:{center}"])
                    if len(cbow_pairs) >= 100:
                        break
                if len(cbow_pairs) >= 100:
                    break

            # Skip-gram：中心词 → 上下文词
            skip_pairs = []
            for sent in corpus[:50]:
                for i, center in enumerate(sent):
                    if token_counter[center] < int(min_count):
                        continue
                    for j in range(max(0, i - win), min(len(sent), i + win + 1)):
                        if j == i:
                            continue
                        skip_pairs.append([f"中心词:{center}", f"上下文:{sent[j]}"])
                        if len(skip_pairs) >= 100:
                            break
                    if len(skip_pairs) >= 100:
                        break
                if len(skip_pairs) >= 100:
                    break

            return cbow_pairs, skip_pairs

        def run_w2v_viz():
            top, pairs = _bow_and_skipgram(window=int(w2v_window.value), min_count=int(w2v_min_count.value))
            return top, pairs

        run_w2v_viz_btn.click(run_w2v_viz, outputs=[bow_top, skipgram_pairs])

        # ============ BERT 句向量 ============
        bert_holder = {"tok": None, "mdl": None}

        def load_bert(model_name: str):
            if AutoTokenizer is None:
                return "<p style='color:red'>transformers 未安装</p>"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModel.from_pretrained(model_name)
            mdl.eval()
            bert_holder.update({"tok": tok, "mdl": mdl})
            return f"<p style='color:green'>✅ 模型已加载: {model_name}</p>"

        def cosine(a, b):
            import numpy as np
            na = a / (np.linalg.norm(a) + 1e-8)
            nb = b / (np.linalg.norm(b) + 1e-8)
            return float((na * nb).sum())

        def run_bert(model_name: str, a: str, b: str):
            if bert_holder["tok"] is None:
                load_bert(model_name)
            tok, mdl = bert_holder["tok"], bert_holder["mdl"]
            with torch.no_grad():
                inputs = tok([a, b], return_tensors="pt", padding=True, truncation=True, max_length=128)
                outputs = mdl(**inputs)
                # 使用 [CLS] 向量或平均池化
                cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                sim = cosine(cls[0], cls[1])
            return gr.update(value=f"<p>相似度: {sim:.4f}</p>")

        load_bert_btn.click(load_bert, inputs=[bert_model_name], outputs=[bert_status])
        run_bert_btn.click(run_bert, inputs=[bert_model_name, sent_a, sent_b], outputs=[bert_similarity])

        # ============ OPT 生成 ============
        opt_holder = {"tok": None, "mdl": None}

        def load_opt(model_name: str):
            if AutoTokenizer is None:
                return "<p style='color:red'>transformers 未安装</p>"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForCausalLM.from_pretrained(model_name)
            opt_holder.update({"tok": tok, "mdl": mdl})
            return f"<p style='color:green'>✅ 模型已加载: {model_name}</p>"

        def generate_opt(model_name: str, prompt: str, max_new: int):
            if opt_holder["tok"] is None:
                load_opt(model_name)
            tok, mdl = opt_holder["tok"], opt_holder["mdl"]
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = mdl.generate(**inputs, max_new_tokens=int(max_new), do_sample=True, top_p=0.9)
            text = tok.decode(out[0], skip_special_tokens=True)
            return gr.update(value=f"<pre>{text}</pre>")

        def _load_preloaded_text_corpus(max_docs: int = 200) -> str:
            """将预置文档拼接为CLM训练文本（仅示例）。"""
            try:
                preloaded_path = os.path.join("data", "preloaded_documents.json")
                if not os.path.exists(preloaded_path):
                    return ""
                with open(preloaded_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = data["documents"] if isinstance(data, dict) and "documents" in data else data
                texts = []
                count = 0
                if isinstance(docs, dict):
                    for _, content in docs.items():
                        texts.append(str(content).strip())
                        count += 1
                        if count >= max_docs:
                            break
                else:
                    for content in docs:
                        texts.append(str(content).strip())
                        count += 1
                        if count >= max_docs:
                            break
                return "\n\n".join(texts)
            except Exception:
                return ""

        def train_opt_on_preloaded(model_name: str, steps: int, lr: float, block: int, bsize: int):
            if opt_holder["tok"] is None:
                load_opt(model_name)
            tok, mdl = opt_holder["tok"], opt_holder["mdl"]
            corpus = _load_preloaded_text_corpus()
            if not corpus:
                return "<p style='color:red'>未找到预置文本</p>"
            # 构造简易数据张量
            inputs = tok(corpus, return_tensors="pt", truncation=True, max_length=int(block))
            input_ids = inputs["input_ids"]
            optim = torch.optim.AdamW(mdl.parameters(), lr=float(lr))
            mdl.train()
            total_loss = 0.0
            for i in range(int(steps)):
                optim.zero_grad()
                out = mdl(input_ids=input_ids, labels=input_ids)
                loss = out.loss
                loss.backward()
                optim.step()
                total_loss += loss.item()
            mdl.eval()
            avg_loss = total_loss / max(1, int(steps))
            return f"<p style='color:green'>✅ CLM训练完成(演示) steps={int(steps)}, avg_loss={avg_loss:.4f}</p>"

        load_opt_btn.click(load_opt, inputs=[opt_model_name], outputs=[opt_status])
        gen_btn.click(generate_opt, inputs=[opt_model_name, opt_prompt, max_new_tokens], outputs=[opt_output])
        train_opt_btn.click(train_opt_on_preloaded, inputs=[opt_model_name, train_steps, lr_opt, block_size, batch_size], outputs=[opt_status])

        # CLM(Causal Language Modeling)自监督学习数据格式可视化
        def _clm_pairs_only(model_name: str):
            text = _load_preloaded_text_corpus(max_docs=200)
            if not text:
                return []
            tok = AutoTokenizer.from_pretrained(model_name)
            ids = tok(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).tolist()

            # CLM自监督任务：给定前文token id序列，预测下一个token id
            pairs = []
            context_len = 32  # 仅展示尾部若干id，避免过长
            for i in range(min(50, len(ids) - 1)):
                # 取到当前位置（包含当前token）的上下文id序列
                ctx_ids = ids[max(0, i - context_len):i + 1]
                next_id = int(ids[i + 1])
                # 解码上下文为字面文本，仅用于展示；可能包含不可见字符，用箭头替换换行
                ctx_text = tok.decode(ctx_ids).replace("\n", "↵")
                input_seq = "..." + ctx_text[-64:]
                # 仅对下一个token显示为[id]
                pairs.append([input_seq, f"[{next_id}]", next_id, int(i + 1)])
                if len(pairs) >= 30:
                    break
            return pairs

        def run_clm_viz():
            return _clm_pairs_only(opt_model_name.value)

        run_clm_viz_btn.click(run_clm_viz, outputs=[clm_pairs_df])

        # ============ CLIP 微调（演示） ============
        clip_holder = {"proc": None, "mdl": None}

        def load_clip(model_name: str):
            if CLIPProcessor is None:
                return "<p style='color:red'>transformers/Pillow 未安装</p>"
            proc = CLIPProcessor.from_pretrained(model_name)
            mdl = CLIPModel.from_pretrained(model_name)
            clip_holder.update({"proc": proc, "mdl": mdl})
            return f"<p style='color:green'>✅ 模型已加载: {model_name}</p>"

        def _load_builtin_pairs() -> List[Tuple[str, str]]:
            # CLIP自监督学习：文本-图像对比学习数据格式
            # 严格使用真实的图文对数据
            candidates = []
            
            # 首先尝试从本地索引加载
            try:
                idx_paths = [
                    os.path.join("test_images", "image_index.json"),  # 优先使用有描述的test_images
                    os.path.join("models", "images", "image_index.json"),
                ]
                for idx in idx_paths:
                    if os.path.exists(idx):
                        with open(idx, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        images = data.get("images", {})
                        for info in images.values():
                            img_path = info.get("stored_path") or info.get("path")
                            text = info.get("description") or ""
                            
                            # 严格验证：只使用有真实描述且图片存在的数据
                            if (img_path and os.path.exists(img_path) and 
                                text and len(text.strip()) > 0 and 
                                text != "A photo"):  # 排除通用占位符
                                candidates.append((img_path, text.strip()))
            except Exception:
                pass
            
            # 如果没有找到有效的图文对，返回空列表
            # 确保数据的真实性，不使用任何模拟或占位符数据
            
            return candidates[:6]  # 最多取6对，便于展示对比学习概念

        def finetune_clip(model_name: str):
            if clip_holder["proc"] is None:
                load_clip(model_name)
            try:
                proc, mdl = clip_holder["proc"], clip_holder["mdl"]
                pairs = _load_builtin_pairs()
                if not pairs:
                    return "<p style='color:red'>❌ 未找到真实图文对数据<br/>请确认 models/images 或 test_images 目录中有图片和索引文件<br/>CLIP演示需要真实的图文对数据才能进行</p>"
                
                # 验证所有图片文件都存在
                valid_pairs = [(p, t) for p, t in pairs if os.path.exists(p)]
                if not valid_pairs:
                    return "<p style='color:red'>❌ 未找到有效的图片文件<br/>请检查图片路径是否正确</p>"
                
                # 使用真实图片进行CLIP对比学习演示
                mdl.train()
                optim = torch.optim.AdamW(mdl.parameters(), lr=5e-6)
                
                # 准备图片和文本数据
                images = [Image.open(p).convert("RGB") for p, _ in valid_pairs]
                texts = [t for _, t in valid_pairs]
                
                # CLIP预处理：图片和文本编码
                inputs = proc(text=texts, images=images, return_tensors="pt", padding=True)
                
                # 前向传播
                outputs = mdl(**inputs)
                
                # CLIP对比学习核心：
                # 1. 获取标准化的图像和文本嵌入
                image_embeds = outputs.image_embeds  # [batch_size, embed_dim]
                text_embeds = outputs.text_embeds    # [batch_size, embed_dim]
                
                # 2. 计算相似度矩阵 (这是CLIP的核心机制)
                # logits_per_image: 每个图像与所有文本的相似度 [batch_size, batch_size]
                # logits_per_text: 每个文本与所有图像的相似度 [batch_size, batch_size]
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # 3. CLIP对比学习损失
                # 对角线元素是正样本对(image_i, text_i)，其他是负样本
                batch_size = len(images)
                labels = torch.arange(batch_size, dtype=torch.long)
                
                # 图像到文本的对比损失：每个图像应该与对应文本最相似
                loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
                # 文本到图像的对比损失：每个文本应该与对应图像最相似  
                loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
                
                # 总对比损失（CLIP标准做法）
                contrastive_loss = (loss_i2t + loss_t2i) / 2
                
                # 反向传播和优化
                optim.zero_grad()
                contrastive_loss.backward()
                optim.step()
                mdl.eval()
                
                # 计算训练后的相似度用于展示
                with torch.no_grad():
                    final_outputs = mdl(**inputs)
                    final_similarities = final_outputs.logits_per_image
                
                # 展示使用的图文对信息和训练结果
                data_info = "<h4>📊 使用的图文对数据</h4><ul>"
                for i, (img_path, text) in enumerate(valid_pairs):
                    img_name = os.path.basename(img_path)
                    data_info += f"<li><strong>样本{i+1}</strong>: {img_name} - {text}</li>"
                data_info += "</ul>"
                
                # 展示相似度矩阵（对比学习的核心概念）
                similarity_info = "<h4>🎯 对比学习相似度矩阵</h4>"
                similarity_info += "<p>每行表示一张图像与所有文本的相似度，对角线应该最高（正样本对）</p>"
                similarity_info += "<table style='border-collapse: collapse; margin: 10px 0;'>"
                similarity_info += "<tr><th style='border: 1px solid #ddd; padding: 4px;'>图像\\文本</th>"
                for j, (_, text) in enumerate(valid_pairs):
                    similarity_info += f"<th style='border: 1px solid #ddd; padding: 4px;'>文本{j+1}</th>"
                similarity_info += "</tr>"
                
                # 显示相似度数值（简化版本，仅显示前3x3）
                sim_matrix = final_similarities.cpu().numpy()
                display_size = min(3, batch_size)  # 最多显示3x3矩阵
                for i in range(display_size):
                    img_name = os.path.basename(valid_pairs[i][0])
                    similarity_info += f"<tr><td style='border: 1px solid #ddd; padding: 4px;'>{img_name}</td>"
                    for j in range(display_size):
                        sim_val = sim_matrix[i, j]
                        # 对角线元素（正样本）用绿色高亮
                        color = "background-color: #d4edda;" if i == j else ""
                        similarity_info += f"<td style='border: 1px solid #ddd; padding: 4px; {color}'>{sim_val:.3f}</td>"
                    similarity_info += "</tr>"
                similarity_info += "</table>"
                
                return f"<p style='color:green'>✅ CLIP对比学习演示完成</p>{data_info}<p><strong>训练结果:</strong><br/>对比损失: {contrastive_loss.item():.4f}<br/>图像→文本损失: {loss_i2t.item():.4f} | 文本→图像损失: {loss_t2i.item():.4f}<br/>训练样本: {batch_size}个真实图文正样本对</p>{similarity_info}"
                
            except Exception as e:
                return f"<p style='color:red'>CLIP对比学习演示失败: {str(e)}</p>"

        # CLIP训练数据可视化函数
        def visualize_clip_data():
            pairs = _load_builtin_pairs()
            if not pairs:
                return [["无数据", "未找到真实图文对数据", "错误"]], [], "<p style='color:red'>未找到真实图文对数据</p>"
            
            # 构建CLIP对比学习训练数据表格
            clip_data_rows = []
            valid_images = []
            text_descriptions = []
            
            for i, (img_path, text) in enumerate(pairs):
                if os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    # 正样本对：匹配的图文对
                    clip_data_rows.append([img_name, text, f"正样本{i+1}"])
                    
                    # 确保图片路径格式正确，用于Gradio Gallery显示
                    # 转换为绝对路径确保Gradio能正确加载
                    abs_img_path = os.path.abspath(img_path)
                    valid_images.append(abs_img_path)
                    
                    text_descriptions.append(f"<strong>图片{i+1}</strong>: {img_name}<br/><strong>描述</strong>: {text}")
                    
                    # 负样本对：与其他文本的不匹配组合（演示概念）
                    for j, (_, other_text) in enumerate(pairs):
                        if j != i and j < 2:  # 限制负样本展示数量
                            clip_data_rows.append([img_name, other_text, f"负样本{i+1}-{j+1}"])
            
            # 构建图文对应的HTML显示
            if text_descriptions:
                text_html = "<div style='max-height: 300px; overflow-y: auto;'>"
                text_html += "<h4>🖼️ 图文正样本对</h4>"
                for desc in text_descriptions:
                    text_html += f"<div style='margin: 10px 0; padding: 8px; border: 1px solid #ddd; border-radius: 4px;'>{desc}</div>"
                text_html += "</div>"
                text_html += "<p style='color: green; margin-top: 10px;'>✅ 对比学习时，匹配的图文对作为正样本，不匹配的作为负样本</p>"
                text_html += f"<p style='color: blue;'>📂 图片路径验证: 找到 {len(valid_images)} 张有效图片</p>"
            else:
                text_html = "<p style='color:red'>未找到有效的图文对数据</p>"
            
            # 调试信息
            print(f"CLIP可视化: 找到 {len(valid_images)} 张图片")
            for img_path in valid_images:
                print(f"图片路径: {img_path}, 存在: {os.path.exists(img_path)}")
                            
            return clip_data_rows, valid_images, text_html
        
        load_clip_btn.click(load_clip, inputs=[clip_model_name], outputs=[clip_status])
        clip_train_btn.click(finetune_clip, inputs=[clip_model_name], outputs=[clip_log])
        viz_clip_data_btn.click(visualize_clip_data, outputs=[clip_data_viz, clip_image_gallery, clip_text_display])
        
        # 添加 LLMOps Engine 的 resume 支持（参考 LLaMA-Factory 的设计）
        # 注意：暂时禁用 resume 功能，避免 Gradio 类型推断错误
        # 如果需要恢复功能，需要确保所有组件都已正确注册且类型注解正确
        # if train_engine is not None:
        #     try:
        #         output_elems = [elem for elem in train_engine.manager.get_elem_list() if elem is not None]
        #         if output_elems:
        #             training_tab.load(
        #                 train_engine.resume,
        #                 outputs=output_elems,
        #                 concurrency_limit=None
        #             )
        #     except Exception as e:
        #         print(f"警告: LLMOps resume 功能初始化失败: {e}")

        return training_tab 