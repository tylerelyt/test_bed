"""
LLMOps 闭环系统主界面
按照业务流程组织：CPT → SFT → DPO，每个Tab内聚数据+训练
"""
# 必须在导入任何其他模块之前设置环境变量，避免 transformers 导入 TensorFlow
import os
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import gradio as gr
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from .self_instruct_generator import SelfInstructGenerator
from .domain_corpus_processor import DomainCorpusProcessor
from .preference_collector import PreferenceCollector
from .llama_factory_config import LLaMAFactoryConfig
from .llamafactory_trainer import get_trainer
from .llmops_engine import LLMOpsEngine
from .llmops_models import LLaMAFactoryModels
from .inference_model import InferenceModel


def get_trained_models(stage: str = "cpt") -> List[str]:
    """扫描已训练的模型目录（基于训练阶段的输出目录）
    
    Args:
        stage: 训练阶段 "cpt", "sft", "dpo"
        - "cpt": 扫描 checkpoints/cpt/ 目录
        - "sft": 扫描 checkpoints/sft/ 目录
        - "dpo": 扫描 checkpoints/dpo/ 目录
    
    Returns:
        已训练模型路径列表（按修改时间排序，最新的在前）
    """
    models = []
    
    # 只扫描对应阶段的输出目录
    checkpoint_dir = os.path.join("checkpoints", stage)
    
    if not os.path.exists(checkpoint_dir):
        return models
    
    # 扫描目录下的所有子目录
    try:
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含 adapter_config.json（LoRA模型标志）
                adapter_config = os.path.join(item_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    models.append(item_path)
    except Exception as e:
        print(f"扫描 {checkpoint_dir} 目录失败: {e}")
        return models
    
    # 按修改时间排序，最新的在前
    if models:
        models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return models


def get_available_datasets(stage: str = None) -> List[str]:
    """获取已注册的数据集列表
    
    Args:
        stage: 训练阶段 "cpt", "sft", "dpo"。如果为None，返回所有数据集
    
    Returns:
        符合条件的数据集名称列表
    """
    dataset_info_path = "data/llmops/dataset_info.json"
    if not os.path.exists(dataset_info_path):
        return []
    
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        if stage is None:
            return list(dataset_info.keys())
        
        # 根据阶段过滤数据集
        filtered_datasets = []
        for name, config in dataset_info.items():
            if stage == "cpt":
                # CPT: 只要 prompt->text 的数据集（没有 formatting 或 formatting 不是 sharegpt）
                if config.get("columns", {}).get("prompt") == "text" and not config.get("formatting"):
                    filtered_datasets.append(name)
            elif stage == "sft":
                # SFT: ShareGPT 格式且有 messages 字段，没有 ranking
                if config.get("formatting") == "sharegpt" and "messages" in config.get("columns", {}) and not config.get("ranking"):
                    filtered_datasets.append(name)
            elif stage == "dpo":
                # DPO: ShareGPT 格式且有 ranking=True
                if config.get("formatting") == "sharegpt" and config.get("ranking"):
                    filtered_datasets.append(name)
        
        return filtered_datasets
    except Exception as e:
        print(f"读取数据集列表失败: {e}")
        return []


class LLMOpsSystem:
    """LLMOps 系统管理器"""
    
    def __init__(self):
        self.self_instruct = SelfInstructGenerator()
        self.corpus_processor = DomainCorpusProcessor()
        self.pref_collector = PreferenceCollector()
        self.config_manager = LLaMAFactoryConfig()
        self.inference_model = InferenceModel()  # 推理模型（借鉴 LLaMA-Factory）
        
        # 模型路径状态（确保流程依赖）
        self.cpt_output_path = ""  # CPT 输出的 Completion Model
        self.sft_output_path = ""  # SFT 输出的 Chat Model
        
        # 当前对比测试的查询和响应
        self.current_query = ""
        self.current_model = ""
        self.current_responses = {}


def build_llmops_content():
    """构建 LLMOps 内容（不创建 Blocks，直接渲染组件）
    
    Returns:
        train_engine: 训练引擎实例，可用于调用 resume() 等方法
    """
    
    system = LLMOpsSystem()
    
    gr.Markdown("""
    # 🔄 LLMOps 持续进化闭环系统
    
    **完整流程**: Base Model → CPT → Completion Model → SFT → Chat Model → DPO → Optimized Chat Model
    
    每个阶段内聚：数据准备 + 训练配置 + 模型输出
    """)
    
    # 用于存储训练引擎
    train_engines = {}
    
    # 获取模型列表
    model_choices = LLaMAFactoryModels.get_flat_choices()
    print(f"✅ 加载了 {len(model_choices)} 个支持的模型")
    
    with gr.Tabs():
        # ==================== Tab 1: CPT (Continued Pre-Training) ====================
        with gr.Tab("📚 阶段1: CPT - 继续预训练"):
            gr.Markdown("""
            ### 领域适配 - 注入行业知识
            **输入**: Base Model（如 Llama-3-8B）
            **数据**: 领域专业语料（无监督文本）
            **输出**: Completion Model（可用于文本补全、代码生成）
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📊 第1步：准备领域语料")
                    load_corpus_btn = gr.Button("📥 加载预置文档", variant="secondary")
                    corpus_limit = gr.Slider(100, 2000, value=500, step=100, label="加载文档数量")
                    process_corpus_btn = gr.Button("🔧 处理语料", variant="primary")
                    save_corpus_btn = gr.Button("💾 保存为CPT数据集", variant="secondary")
                    corpus_output = gr.HTML(value="<p>点击加载预置文档开始...</p>")
                    corpus_stats = gr.HTML(value="<p>处理后显示统计...</p>")
                
                with gr.Column():
                    gr.Markdown("#### ⚙️ 第2步：配置CPT训练")
                    
                    # 基础配置
                    with gr.Row():
                        cpt_model = gr.Dropdown(
                            choices=model_choices,
                            value="Qwen/Qwen2-0.5B",
                            label="Base Model",
                            info="支持100+主流开源模型",
                            allow_custom_value=True,
                            filterable=True,
                            interactive=True
                        )
                    
                    with gr.Row():
                        cpt_dataset = gr.Dropdown(
                            choices=get_available_datasets("cpt"),  # 只显示CPT数据集
                            value="test_corpus_large",  # 使用小数据集快速测试
                            label="数据集名称",
                            info="选择已保存的CPT数据集（纯文本格式）",
                            allow_custom_value=True,
                            interactive=True
                        )
                        cpt_output = gr.Textbox(value="checkpoints/cpt/qwen-0.5b-cpt", label="输出路径")
                    
                    # 训练参数
                    with gr.Accordion("🔧 训练参数", open=False):
                        with gr.Row():
                            cpt_epochs = gr.Slider(1, 10, value=1, step=1, label="训练轮数")
                            cpt_lr = gr.Slider(1e-5, 1e-3, value=5e-5, step=1e-5, label="学习率")
                        with gr.Row():
                            cpt_batch_size = gr.Slider(1, 16, value=1, step=1, label="批次大小")
                            cpt_grad_acc = gr.Slider(1, 16, value=2, step=1, label="梯度累积")
                        with gr.Row():
                            cpt_max_len = gr.Slider(512, 4096, value=512, step=128, label="最大序列长度")
                            cpt_save_steps = gr.Slider(10, 2000, value=50, step=10, label="保存步数")
                    
                    # LoRA 配置
                    with gr.Accordion("🎯 LoRA 配置", open=False):
                        with gr.Row():
                            cpt_lora_rank = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank")
                            cpt_lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA Alpha")
                        cpt_lora_dropout = gr.Slider(0, 0.5, value=0.05, step=0.05, label="LoRA Dropout")
                    
                    # 操作按钮
                    with gr.Row():
                        cpt_start_btn = gr.Button("🚀 开始CPT训练", variant="primary")
                        cpt_stop_btn = gr.Button("⏹️ 停止训练", variant="secondary")
                    
                    # 状态显示
                    cpt_progress = gr.Slider(0, 100, value=0, label="训练进度", visible=False, interactive=False)
                    cpt_status = gr.HTML(value="<p>未开始训练</p>")
                    
                    # 创建 CPT 训练引擎
                    cpt_engine = LLMOpsEngine()
                    train_engines['cpt'] = cpt_engine
        
        # ==================== Tab 2: SFT (Supervised Fine-Tuning) ====================
        with gr.Tab("📝 阶段2: SFT - 指令微调"):
            gr.Markdown("""
            ### 指令对齐 - 教会模型对话
            **输入**: CPT 输出的 Completion Model
            **数据**: 指令-回答对（instruction-response pairs）
            **输出**: Chat Model（可用于对话、问答）
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📊 第1步：生成指令数据")
                    instruct_count = gr.Slider(10, 200, value=50, step=10, label="生成指令数量")
                    generate_instruct_btn = gr.Button("🚀 生成指令数据", variant="primary")
                    save_instruct_btn = gr.Button("💾 保存为SFT数据集", variant="secondary")
                    instruct_output = gr.HTML(value="<p>点击生成指令数据开始...</p>")
                    instruct_stats = gr.HTML(value="<p>生成后显示统计...</p>")
                
                with gr.Column():
                    gr.Markdown("#### ⚙️ 第2步：配置SFT训练")
                    gr.Markdown("⚠️ **注意**: 模型路径必须使用 CPT 的输出")
                    
                    # 基础配置
                    with gr.Row():
                        sft_base_model = gr.Dropdown(
                            choices=model_choices,
                            value="Qwen/Qwen2-0.5B",
                            label="Base Model",
                            info="基础模型（与CPT相同）",
                            allow_custom_value=True,
                            filterable=True,
                            interactive=True
                        )
                    
                    with gr.Row():
                        # 初始化时加载可用的CPT模型
                        initial_cpt_models = get_trained_models("cpt")
                        
                        sft_cpt_model = gr.Dropdown(
                            choices=initial_cpt_models,
                            value=initial_cpt_models[0] if initial_cpt_models else None,
                            label="CPT Checkpoint",
                            info="选择CPT阶段的输出模型（必填，请先完成CPT训练）",
                            allow_custom_value=True,
                            interactive=True
                        )
                        sft_refresh_models = gr.Button("🔄", scale=0, min_width=50)
                    
                    with gr.Row():
                        sft_dataset = gr.Dropdown(
                            choices=get_available_datasets("sft"),  # 只显示SFT数据集
                            value="test_sft_data",  # 使用小数据集快速测试
                            label="数据集名称",
                            info="选择已保存的SFT数据集（ShareGPT对话格式）",
                            allow_custom_value=True,
                            interactive=True
                        )
                        sft_output = gr.Textbox(value="checkpoints/sft/qwen-0.5b-sft", label="输出路径")
                    
                    with gr.Row():
                        sft_template = gr.Dropdown(
                            choices=["llama3", "qwen", "chatglm3", "mistral"],
                            value="llama3",
                            label="对话模板"
                        )
                    
                    # 训练参数
                    with gr.Accordion("🔧 训练参数", open=False):
                        with gr.Row():
                            sft_epochs = gr.Slider(1, 10, value=1, step=1, label="训练轮数")
                            sft_lr = gr.Slider(1e-5, 1e-3, value=5e-5, step=1e-5, label="学习率")
                        with gr.Row():
                            sft_batch_size = gr.Slider(1, 16, value=1, step=1, label="批次大小")
                            sft_grad_acc = gr.Slider(1, 16, value=2, step=1, label="梯度累积")
                        with gr.Row():
                            sft_max_len = gr.Slider(512, 4096, value=512, step=128, label="最大序列长度")
                            sft_save_steps = gr.Slider(10, 2000, value=50, step=10, label="保存步数")
                    
                    # LoRA 配置
                    with gr.Accordion("🎯 LoRA 配置", open=False):
                        with gr.Row():
                            sft_lora_rank = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank")
                            sft_lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA Alpha")
                        sft_lora_dropout = gr.Slider(0, 0.5, value=0.05, step=0.05, label="LoRA Dropout")
                    
                    # 操作按钮
                    with gr.Row():
                        sft_start_btn = gr.Button("🚀 开始SFT训练", variant="primary")
                        sft_stop_btn = gr.Button("⏹️ 停止训练", variant="secondary")
                    
                    # 状态显示
                    sft_progress = gr.Slider(0, 100, value=0, label="训练进度", visible=False, interactive=False)
                    sft_status = gr.HTML(value="<p>未开始训练</p>")
                    
                    # 创建 SFT 训练引擎
                    sft_engine = LLMOpsEngine()
                    train_engines['sft'] = sft_engine
        
        # ==================== Tab 3: DPO/RLHF - 在线优化闭环 ====================
        with gr.Tab("🔬 阶段3: DPO/RLHF - 偏好对齐"):
            gr.Markdown("""
            ### 偏好对齐 - 持续优化
            **输入**: SFT 输出的 Chat Model
            **数据**: 用户偏好数据（通过AB测试收集）
            **输出**: Optimized Chat Model（v1.0 → v1.1 → v1.2...）
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🚀 第1步：加载推理模型")
                    gr.Markdown("*使用 LLaMA-Factory 内置推理引擎，直接加载模型到内存*")
                    
                    # 加载可用的SFT和DPO模型
                    inference_sft_models = get_trained_models("sft")
                    inference_dpo_models = get_trained_models("dpo")
                    inference_models = inference_sft_models + inference_dpo_models
                    
                    infer_model = gr.Dropdown(
                        choices=inference_models,
                        value=inference_models[0] if inference_models else None,
                        label="Chat Model (SFT/DPO)",
                        info="选择SFT或DPO模型",
                        allow_custom_value=True,
                        interactive=True
                    )
                    infer_refresh = gr.Button("🔄 刷新模型列表", variant="secondary")
                    
                    with gr.Row():
                        load_model_btn = gr.Button("▶️ 加载模型", variant="primary")
                        unload_model_btn = gr.Button("⏹️ 卸载模型", variant="secondary")
                    
                    infer_status = gr.Textbox(
                        label="模型状态",
                        value="未加载模型",
                        interactive=False,
                        lines=3
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔬 第2步：AB测试收集偏好")
                    gr.Markdown("同一模型生成两个不同回答（通过调整采样参数），用户投票选择更好的回答")
                    
                    ab_query = gr.Textbox(
                        label="输入测试问题",
                        placeholder="例如：介绍人工智能的发展历史",
                        lines=2
                    )
                    
                    with gr.Row():
                        ab_model = gr.Dropdown(
                            choices=inference_models,
                            value=inference_models[0] if inference_models else None,
                            label="选择模型",
                            info="用于生成AB对比的模型",
                            allow_custom_value=True,
                            interactive=True
                        )
                        ab_refresh_model = gr.Button("🔄", scale=0, min_width=50)
                    
                    with gr.Row():
                        ab_temperature_a = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature A（更保守）")
                        ab_temperature_b = gr.Slider(0.1, 2.0, value=1.2, step=0.1, label="Temperature B（更创造性）")
                    
                    ab_generate_btn = gr.Button("🔄 生成AB对比", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("##### 🅰️ 回答 A")
                            response_a_label = gr.Textbox(value="", label="模型版本", interactive=False)
                            response_a = gr.Textbox(label="回答内容", lines=6, interactive=False)
                            vote_a_btn = gr.Button("👍 选择 A 更好", variant="secondary", size="lg")
                        
                        with gr.Column():
                            gr.Markdown("##### 🅱️ 回答 B")
                            response_b_label = gr.Textbox(value="", label="模型版本", interactive=False)
                            response_b = gr.Textbox(label="回答内容", lines=6, interactive=False)
                            vote_b_btn = gr.Button("👍 选择 B 更好", variant="secondary", size="lg")
                    
                    ab_result = gr.HTML(value="<p>生成对比后投票，偏好数据自动保存到 prefs.jsonl</p>")
                    
                    with gr.Row():
                        view_prefs_btn = gr.Button("📊 查看偏好统计", variant="secondary")
                        export_prefs_btn = gr.Button("📤 导出DPO数据集", variant="primary")
                    
                    prefs_stats = gr.HTML(value="<p>点击查看偏好统计...</p>")
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ 第3步：配置DPO训练")
                    gr.Markdown("⚠️ **注意**: 模型路径必须使用 SFT 的输出")
                    
                    # 基础配置
                    with gr.Row():
                        dpo_base_model = gr.Dropdown(
                            choices=model_choices,
                            value="Qwen/Qwen2-0.5B",
                            label="Base Model",
                            info="基础模型（与SFT相同）",
                            allow_custom_value=True,
                            filterable=True,
                            interactive=True
                        )
                    
                    with gr.Row():
                        # 初始化时加载可用的SFT和DPO模型
                        initial_sft_models = get_trained_models("sft")
                        initial_dpo_models = get_trained_models("dpo")
                        # 合并SFT和DPO模型（DPO可以在之前的DPO基础上继续训练）
                        initial_models = initial_sft_models + initial_dpo_models
                        
                        dpo_sft_model = gr.Dropdown(
                            choices=initial_models,
                            value=initial_models[0] if initial_models else None,
                            label="SFT/DPO Checkpoint",
                            info="选择SFT或DPO模型（必填，请先完成SFT训练）",
                            allow_custom_value=True,
                            interactive=True
                        )
                        dpo_refresh_models = gr.Button("🔄", scale=0, min_width=50)
                    
                    with gr.Row():
                        dpo_dataset = gr.Dropdown(
                            choices=get_available_datasets("dpo"),  # 只显示DPO数据集
                            value="test_dpo_data",  # 使用小数据集快速测试
                            label="偏好数据集",
                            info="选择已保存的DPO数据集（ShareGPT Ranking格式）",
                            allow_custom_value=True,
                            interactive=True
                        )
                        dpo_output = gr.Textbox(value="checkpoints/dpo/qwen-0.5b-dpo", label="输出路径")
                    
                    # DPO 特有参数
                    with gr.Row():
                        dpo_beta = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="DPO Beta", info="偏好强度")
                        dpo_ftx = gr.Slider(0, 1, value=0, step=0.1, label="FTX权重", info="SFT损失权重")
                    
                    # 训练参数
                    with gr.Accordion("🔧 训练参数", open=False):
                        with gr.Row():
                            dpo_epochs = gr.Slider(1, 10, value=1, step=1, label="训练轮数")
                            dpo_lr = gr.Slider(1e-6, 1e-4, value=5e-6, step=1e-6, label="学习率（DPO通常更小）")
                        with gr.Row():
                            dpo_batch_size = gr.Slider(1, 16, value=1, step=1, label="批次大小")
                            dpo_grad_acc = gr.Slider(1, 16, value=2, step=1, label="梯度累积")
                        with gr.Row():
                            dpo_max_len = gr.Slider(512, 4096, value=512, step=128, label="最大序列长度")
                            dpo_save_steps = gr.Slider(10, 2000, value=50, step=10, label="保存步数")
                    
                    # LoRA 配置
                    with gr.Accordion("🎯 LoRA 配置", open=False):
                        with gr.Row():
                            dpo_lora_rank = gr.Slider(4, 64, value=8, step=4, label="LoRA Rank")
                            dpo_lora_alpha = gr.Slider(8, 128, value=16, step=8, label="LoRA Alpha")
                        dpo_lora_dropout = gr.Slider(0, 0.5, value=0.05, step=0.05, label="LoRA Dropout")
                    
                    # 操作按钮
                    with gr.Row():
                        dpo_start_btn = gr.Button("🚀 开始DPO训练", variant="primary")
                        dpo_stop_btn = gr.Button("⏹️ 停止训练", variant="secondary")
                    
                    # 状态显示
                    dpo_progress = gr.Slider(0, 100, value=0, label="训练进度", visible=False, interactive=False)
                    dpo_status = gr.HTML(value="<p>未开始训练</p>")
                    
                    # 创建 DPO 训练引擎
                    dpo_engine = LLMOpsEngine()
                    train_engines['dpo'] = dpo_engine
    
    # ==================== 事件绑定 ====================
    
    # === CPT Tab 事件 ===
    def load_corpus(limit):
        count = system.corpus_processor.load_from_preloaded(int(limit))
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ 已加载 {count} 篇文档</p>
        </div>
        """
    
    def process_corpus():
        count = system.corpus_processor.process()
        stats = system.corpus_processor.get_statistics()
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <h4>✅ 语料处理完成</h4>
            <ul>
                <li><strong>文本数:</strong> {stats['processed_count']}</li>
                <li><strong>总字符:</strong> {stats['total_chars']:,}</li>
                <li><strong>估计tokens:</strong> {stats['estimated_tokens']:,}</li>
            </ul>
        </div>
        """, f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            <p><strong>平均长度:</strong> {stats['avg_length']}</p>
            <p><strong>数据质量:</strong> 已去重、清洗</p>
        </div>
        """
    
    def save_corpus():
        filepath = system.corpus_processor.save_corpus()
        # 获取更新后的CPT数据集列表
        datasets = get_available_datasets("cpt")
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ CPT数据集已保存: <code>{filepath}</code></p>
            <p>💡 数据集已自动注册，可在训练配置中选择</p>
        </div>
        """, gr.update(choices=datasets, value="domain_corpus")
    
    # === SFT Tab 事件 ===
    def generate_instructions(count):
        instructions = system.self_instruct.generate_instructions(int(count), use_mock=True)
        stats = system.self_instruct.get_statistics()
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <h4>✅ 指令数据生成完成</h4>
            <ul>
                <li><strong>本次生成:</strong> {len(instructions)}</li>
                <li><strong>累计总数:</strong> {stats['total']}</li>
            </ul>
        </div>
        """, f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
            <p><strong>任务类型分布:</strong></p>
            {''.join(f"<li>{k}: {v}</li>" for k, v in stats.get('task_types', {}).items())}
        </div>
        """
    
    def save_instructions():
        filepath = system.self_instruct.save_dataset()
        # 获取更新后的SFT数据集列表
        datasets = get_available_datasets("sft")
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ SFT数据集已保存: <code>{filepath}</code></p>
            <p>💡 数据集已自动注册，可在训练配置中选择</p>
        </div>
        """, gr.update(choices=datasets, value="sft_data")
    
    # === 训练相关函数 ===
    def start_cpt_training(model, dataset, output, epochs, lr, batch_size, grad_acc, max_len, save_steps,
                          lora_rank, lora_alpha, lora_dropout):
        """启动CPT训练（使用 generator 持续监控进度）"""
        import time
        trainer = get_trainer()
        
        if trainer.is_training():
            yield gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⚠️ 已有训练任务在运行中</p>
            </div>
            """
            return
        
        # 构建训练配置
        config = {
            'stage': 'pt',
            'model_name_or_path': model,
            'dataset': dataset,
            'dataset_dir': 'data/llmops',
            'output_dir': output,
            'num_train_epochs': int(epochs),
            'learning_rate': float(lr),
            'per_device_train_batch_size': int(batch_size),
            'gradient_accumulation_steps': int(grad_acc),
            'cutoff_len': int(max_len),
            'save_steps': int(save_steps),
            'finetuning_type': 'lora',
            'lora_rank': int(lora_rank),
            'lora_alpha': int(lora_alpha),
            'lora_dropout': float(lora_dropout),
            'logging_steps': 10,
        }
        
        print(f"🚀 准备启动CPT训练: {config}")
        success = trainer.start_training(config)
        print(f"训练启动结果: {success}")
        
        if not success:
            yield gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练启动失败</h4>
                <p>请检查配置和环境</p>
            </div>
            """
            return
        
        # 显示启动成功
        yield gr.update(value=0, visible=True), f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <h4>✅ CPT训练已启动</h4>
            <p><strong>模型:</strong> {model}</p>
            <p><strong>数据集:</strong> {dataset}</p>
            <p><strong>输出:</strong> {output}</p>
            <p>训练进行中...</p>
        </div>
        """
        
        # 等待进程启动
        time.sleep(2)
        
        # 持续监控训练进度（参考 LLaMA-Factory 的 monitor）
        while trainer.is_training():
            # 检查进程状态
            return_code = trainer.check_process_status()
            if return_code is not None:
                # 进程已结束
                break
            
            # 获取训练进度
            progress, status_msg = trainer.get_training_progress()
            log_text = trainer.get_training_logs(max_lines=10)
            
            # 如果日志文件还未生成，显示友好提示
            if "训练尚未开始或日志文件不存在" in log_text or "暂无训练日志" in log_text:
                yield gr.update(value=0, visible=True), f"""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                    <h4>⏳ 正在加载模型和数据...</h4>
                    <p><strong>模型:</strong> {model}</p>
                    <p><strong>数据集:</strong> {dataset}</p>
                    <p><strong>状态:</strong> 训练进程已启动，正在初始化</p>
                    <p>💡 <strong>提示:</strong> 详细日志正在终端窗口实时输出</p>
                    <p>📊 首次训练需要下载模型，请耐心等待...</p>
                </div>
                """
            else:
                # 显示实际训练进度
                yield gr.update(value=progress, visible=True), f"""
                <div style="background-color: #d1ecf1; padding: 10px; border-radius: 5px;">
                    <h4>⏳ {status_msg}</h4>
                    {log_text}
                </div>
                """
            
            time.sleep(2)  # 每2秒更新一次
        
        # 训练完成
        return_code = trainer.check_process_status()
        if return_code == 0:
            final_log = trainer.get_training_logs(max_lines=20)
            yield gr.update(value=100, visible=True), f"""
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
                <h4>✅ CPT训练完成</h4>
                {final_log}
            </div>
            """
        else:
            yield gr.update(visible=False), f"""
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练失败</h4>
                <p>退出码: {return_code}</p>
            </div>
            """
    
    def stop_cpt_training():
        """停止CPT训练"""
        trainer = get_trainer()
        success = trainer.stop_training()
        
        if success:
            return gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⏹️ 训练已停止</p>
            </div>
            """
        else:
            return gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <p>❌ 没有运行中的训练任务</p>
            </div>
            """
    
    def start_sft_training(base_model, cpt_model, dataset, output, template, epochs, lr, batch_size, grad_acc, max_len, save_steps,
                          lora_rank, lora_alpha, lora_dropout):
        """启动SFT训练（使用 generator 持续监控进度）"""
        import time
        trainer = get_trainer()
        
        if trainer.is_training():
            yield gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⚠️ 已有训练任务在运行中</p>
            </div>
            """
            return
        
        # 验证必填项
        if not cpt_model:
            yield gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 配置错误</h4>
                <p>必须选择CPT Checkpoint</p>
            </div>
            """
            return
        
        config = {
            'stage': 'sft',
            'model_name_or_path': base_model,  # 基础模型
            'adapter_name_or_path': cpt_model,  # CPT checkpoint
            'dataset': dataset,
            'dataset_dir': 'data/llmops',
            'template': template,
            'output_dir': output,
            'num_train_epochs': int(epochs),
            'learning_rate': float(lr),
            'per_device_train_batch_size': int(batch_size),
            'gradient_accumulation_steps': int(grad_acc),
            'cutoff_len': int(max_len),
            'save_steps': int(save_steps),
            'finetuning_type': 'lora',
            'lora_rank': int(lora_rank),
            'lora_alpha': int(lora_alpha),
            'lora_dropout': float(lora_dropout),
            'logging_steps': 10,
        }
        
        print(f"🚀 准备启动SFT训练: {config}")
        success = trainer.start_training(config)
        print(f"训练启动结果: {success}")
        
        if not success:
            yield gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练启动失败</h4>
                <p>请检查配置和环境</p>
            </div>
            """
            return
        
        # 显示启动成功
        yield gr.update(value=0, visible=True), f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <h4>✅ SFT训练已启动</h4>
            <p><strong>基础模型:</strong> {base_model}</p>
            <p><strong>CPT Checkpoint:</strong> {cpt_model}</p>
            <p><strong>模板:</strong> {template}</p>
            <p><strong>数据集:</strong> {dataset}</p>
            <p><strong>输出:</strong> {output}</p>
            <p>训练进行中...</p>
        </div>
        """
        
        time.sleep(2)  # 等待进程启动
        
        # 持续监控训练进度
        while trainer.is_training():
            return_code = trainer.check_process_status()
            if return_code is not None:
                break
            
            progress, status_msg = trainer.get_training_progress()
            log_text = trainer.get_training_logs(max_lines=10)
            
            # 如果日志文件还未生成，显示友好提示
            if "训练尚未开始或日志文件不存在" in log_text or "暂无训练日志" in log_text:
                yield gr.update(value=0, visible=True), f"""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                    <h4>⏳ 正在加载模型和数据...</h4>
                    <p><strong>基础模型:</strong> {base_model}</p>
                    <p><strong>CPT Checkpoint:</strong> {cpt_model}</p>
                    <p><strong>数据集:</strong> {dataset}</p>
                    <p><strong>状态:</strong> 训练进程已启动，正在初始化</p>
                    <p>💡 <strong>提示:</strong> 详细日志正在终端窗口实时输出</p>
                </div>
                """
            else:
                yield gr.update(value=progress, visible=True), f"""
                <div style="background-color: #d1ecf1; padding: 10px; border-radius: 5px;">
                    <h4>⏳ {status_msg}</h4>
                    {log_text}
                </div>
                """
            
            time.sleep(2)
        
        # 训练完成
        return_code = trainer.check_process_status()
        if return_code == 0:
            final_log = trainer.get_training_logs(max_lines=20)
            yield gr.update(value=100, visible=True), f"""
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
                <h4>✅ SFT训练完成</h4>
                {final_log}
            </div>
            """
        else:
            yield gr.update(visible=False), f"""
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练失败</h4>
                <p>退出码: {return_code}</p>
            </div>
            """
    
    def stop_sft_training():
        """停止SFT训练"""
        trainer = get_trainer()
        success = trainer.stop_training()
        
        if success:
            return gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⏹️ 训练已停止</p>
            </div>
            """
        else:
            return gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <p>❌ 没有运行中的训练任务</p>
            </div>
            """
    
    def start_dpo_training(base_model, sft_model, dataset, output, beta, ftx, epochs, lr, batch_size, grad_acc, max_len, save_steps,
                          lora_rank, lora_alpha, lora_dropout):
        """启动DPO训练（使用 generator 持续监控进度）"""
        import time
        trainer = get_trainer()
        
        if trainer.is_training():
            yield gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⚠️ 已有训练任务在运行中</p>
            </div>
            """
            return
        
        # 验证必填项
        if not sft_model:
            yield gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 配置错误</h4>
                <p>必须选择SFT Checkpoint</p>
            </div>
            """
            return
        
        config = {
            'stage': 'dpo',
            'model_name_or_path': base_model,  # 基础模型
            'adapter_name_or_path': sft_model,  # SFT checkpoint
            'dataset': dataset,
            'dataset_dir': 'data/llmops',
            'template': 'qwen',  # 与SFT保持一致
            'output_dir': output,
            'num_train_epochs': int(epochs),
            'learning_rate': float(lr),
            'per_device_train_batch_size': int(batch_size),
            'gradient_accumulation_steps': int(grad_acc),
            'cutoff_len': int(max_len),
            'save_steps': int(save_steps),
            'finetuning_type': 'lora',
            'lora_rank': int(lora_rank),
            'lora_alpha': int(lora_alpha),
            'lora_dropout': float(lora_dropout),
            'pref_beta': float(beta),
            'pref_ftx': float(ftx),
            'logging_steps': 10,
        }
        
        print(f"🚀 准备启动DPO训练: {config}")
        success = trainer.start_training(config)
        print(f"训练启动结果: {success}")
        
        if not success:
            yield gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练启动失败</h4>
                <p>请检查配置和环境</p>
            </div>
            """
            return
        
        # 显示启动成功
        yield gr.update(value=0, visible=True), f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <h4>✅ DPO训练已启动</h4>
            <p><strong>基础模型:</strong> {base_model}</p>
            <p><strong>SFT Checkpoint:</strong> {sft_model}</p>
            <p><strong>Beta:</strong> {beta}</p>
            <p><strong>数据集:</strong> {dataset}</p>
            <p><strong>输出:</strong> {output}</p>
            <p>训练进行中...</p>
        </div>
        """
        
        time.sleep(1)  # 等待进程启动
        
        # 持续监控训练进度
        while trainer.is_training():
            return_code = trainer.check_process_status()
            if return_code is not None:
                break
            
            progress, status_msg = trainer.get_training_progress()
            log_text = trainer.get_training_logs(max_lines=10)
            
            yield gr.update(value=progress, visible=True), f"""
            <div style="background-color: #d1ecf1; padding: 10px; border-radius: 5px;">
                <h4>⏳ {status_msg}</h4>
                {log_text}
            </div>
            """
            
            time.sleep(2)
        
        # 训练完成
        return_code = trainer.check_process_status()
        if return_code == 0:
            final_log = trainer.get_training_logs(max_lines=20)
            yield gr.update(value=100, visible=True), f"""
            <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
                <h4>✅ DPO训练完成</h4>
                {final_log}
            </div>
            """
        else:
            yield gr.update(visible=False), f"""
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <h4>❌ 训练失败</h4>
                <p>退出码: {return_code}</p>
            </div>
            """
    
    def stop_dpo_training():
        """停止DPO训练"""
        trainer = get_trainer()
        success = trainer.stop_training()
        
        if success:
            return gr.update(visible=False), """
            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                <p>⏹️ 训练已停止</p>
            </div>
            """
        else:
            return gr.update(visible=False), """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px;">
                <p>❌ 没有运行中的训练任务</p>
            </div>
            """
    
    # === DPO Tab 事件 ===
    def load_infer_model(model_path):
        """加载推理模型（借鉴 LLaMA-Factory WebUI）"""
        if not model_path:
            yield "❌ 请选择模型"
            return
        
        # 从adapter路径中提取base model（假设都是用Qwen2-0.5B训练的）
        base_model = "Qwen/Qwen2-0.5B"
        
        # 调用InferenceModel的load_model（generator）
        for msg in system.inference_model.load_model(
            base_model=base_model,
            adapter_path=model_path,
            template="qwen"
        ):
            yield msg
    
    def unload_infer_model():
        """卸载推理模型"""
        for msg in system.inference_model.unload_model():
            yield msg
    
    def refresh_inference_models():
        """刷新可用的推理模型列表"""
        sft_models = get_trained_models("sft")
        dpo_models = get_trained_models("dpo")
        all_models = sft_models + dpo_models
        return gr.update(choices=all_models)
    
    def generate_ab_responses(query, model, temp_a, temp_b):
        """生成AB对比回答（使用实际模型推理）"""
        if not query:
            return "", "", "", "", "<p style='color: red;'>请输入问题</p>"
        
        if not system.inference_model.loaded:
            return "", "", "", "", "<p style='color: red;'>请先加载模型</p>"
        
        try:
            # 使用同一个模型，但不同的temperature生成两个回答
            print(f"生成A (temp={temp_a})...")
            response_a = system.inference_model.generate_once(
                prompt=query,
                temperature=temp_a,
                max_new_tokens=150
            )
            
            print(f"生成B (temp={temp_b})...")
            response_b = system.inference_model.generate_once(
                prompt=query,
                temperature=temp_b,
                max_new_tokens=150
            )
            
            # 随机打乱A/B位置，避免位置偏见
            import random
            responses = [
                (f"Temperature {temp_a:.1f}", response_a),
                (f"Temperature {temp_b:.1f}", response_b)
            ]
            random.shuffle(responses)
            
            # 保存当前问题和回答，用于投票时记录
            system.current_query = query
            system.current_model = model
            system.current_responses = {
                "A": {"label": responses[0][0], "response": responses[0][1]},
                "B": {"label": responses[1][0], "response": responses[1][1]}
            }
            
            return (
                responses[0][0], responses[0][1],
                responses[1][0], responses[1][1],
                f"<p>✅ 已生成对比（模型: {model}）。请选择你认为更好的回答</p>"
            )
        except Exception as e:
            return "", "", "", "", f"<p style='color: red;'>❌ 生成失败: {str(e)}</p>"
    
    def vote_for_a():
        if not system.current_query:
            return "<p style='color: red;'>请先生成对比</p>"
        
        system.pref_collector.add_preference(
            prompt=system.current_query,
            chosen=system.current_responses["A"]["response"],
            rejected=system.current_responses["B"]["response"],
            metadata={
                "chosen_model": system.current_responses["A"]["label"],
                "rejected_model": system.current_responses["B"]["label"],
                "vote_time": datetime.now().isoformat()
            }
        )
        
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ 偏好已记录：选择了 <strong>{system.current_responses["A"]["label"]}</strong></p>
            <p>💾 数据已写入 prefs.jsonl</p>
        </div>
        """
    
    def vote_for_b():
        if not system.current_query:
            return "<p style='color: red;'>请先生成对比</p>"
        
        system.pref_collector.add_preference(
            prompt=system.current_query,
            chosen=system.current_responses["B"]["response"],
            rejected=system.current_responses["A"]["response"],
            metadata={
                "chosen_model": system.current_responses["B"]["label"],
                "rejected_model": system.current_responses["A"]["label"],
                "vote_time": datetime.now().isoformat()
            }
        )
        
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ 偏好已记录：选择了 <strong>{system.current_responses["B"]["label"]}</strong></p>
            <p>💾 数据已写入 prefs.jsonl</p>
        </div>
        """
    
    def view_preferences():
        stats = system.pref_collector.get_statistics()
        prefs = system.pref_collector.get_all_preferences()
        
        html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h4>📊 偏好数据统计</h4>
            <ul>
                <li><strong>总数量:</strong> {stats['total_preferences']}</li>
                <li><strong>数据文件:</strong> <code>{stats['data_file']}</code></li>
            </ul>
            
            <h5>最近 3 条:</h5>
        """
        
        for pref in prefs[-3:][::-1]:
            html += f"""
            <div style="border: 1px solid #ddd; margin: 5px 0; padding: 8px; border-radius: 5px;">
                <p><strong>问题:</strong> {pref['prompt'][:50]}...</p>
                <p style="color: green;"><strong>✓ 偏好:</strong> {pref['chosen'][:60]}...</p>
            </div>
            """
        
        html += "</div>"
        return html
    
    def export_preferences():
        filepath = system.pref_collector.export_for_dpo()
        stats = system.pref_collector.get_statistics()
        # 获取更新后的DPO数据集列表
        datasets = get_available_datasets("dpo")
        return f"""
        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
            <p>✅ DPO数据集已导出: <code>{filepath}</code></p>
            <p><strong>总数量:</strong> {stats['total_preferences']}</p>
            <p>💡 数据集已自动注册，可在训练配置中选择</p>
        </div>
        """, gr.update(choices=datasets, value="prefs_data")
    
    # 绑定所有事件
    load_corpus_btn.click(load_corpus, inputs=[corpus_limit], outputs=[corpus_output])
    process_corpus_btn.click(process_corpus, outputs=[corpus_output, corpus_stats])
    save_corpus_btn.click(save_corpus, outputs=[corpus_output, cpt_dataset])
    
    generate_instruct_btn.click(generate_instructions, inputs=[instruct_count], outputs=[instruct_output, instruct_stats])
    save_instruct_btn.click(save_instructions, outputs=[instruct_output, sft_dataset])
    
    # CPT 训练事件绑定
    cpt_start_btn.click(
        start_cpt_training,
        inputs=[cpt_model, cpt_dataset, cpt_output, cpt_epochs, cpt_lr, cpt_batch_size, cpt_grad_acc, 
                cpt_max_len, cpt_save_steps, cpt_lora_rank, cpt_lora_alpha, cpt_lora_dropout],
        outputs=[cpt_progress, cpt_status]
    )
    cpt_stop_btn.click(stop_cpt_training, outputs=[cpt_progress, cpt_status])
    
    # SFT 训练事件绑定
    def refresh_cpt_models():
        """刷新CPT模型列表（供SFT阶段使用）"""
        models = get_trained_models("cpt")
        return gr.update(choices=models, value=models[0] if models else None)
    
    def refresh_dpo_models():
        """刷新SFT和DPO模型列表（供DPO阶段使用）"""
        sft_models = get_trained_models("sft")
        dpo_models = get_trained_models("dpo")
        # 合并SFT和DPO模型
        models = sft_models + dpo_models
        return gr.update(choices=models, value=models[0] if models else None)
    
    sft_refresh_models.click(refresh_cpt_models, outputs=[sft_cpt_model])
    
    sft_start_btn.click(
        start_sft_training,
        inputs=[sft_base_model, sft_cpt_model, sft_dataset, sft_output, sft_template, sft_epochs, sft_lr, sft_batch_size, sft_grad_acc,
                sft_max_len, sft_save_steps, sft_lora_rank, sft_lora_alpha, sft_lora_dropout],
        outputs=[sft_progress, sft_status]
    )
    sft_stop_btn.click(stop_sft_training, outputs=[sft_progress, sft_status])
    
    # DPO 训练事件绑定
    dpo_refresh_models.click(refresh_dpo_models, outputs=[dpo_sft_model])
    
    dpo_start_btn.click(
        start_dpo_training,
        inputs=[dpo_base_model, dpo_sft_model, dpo_dataset, dpo_output, dpo_beta, dpo_ftx, dpo_epochs, dpo_lr, dpo_batch_size, 
                dpo_grad_acc, dpo_max_len, dpo_save_steps, dpo_lora_rank, dpo_lora_alpha, dpo_lora_dropout],
        outputs=[dpo_progress, dpo_status]
    )
    dpo_stop_btn.click(stop_dpo_training, outputs=[dpo_progress, dpo_status])
    
    # 推理服务事件绑定
    def refresh_inference_models():
        """刷新可用于推理的模型列表（SFT和DPO）"""
        sft_models = get_trained_models("sft")
        dpo_models = get_trained_models("dpo")
        models = sft_models + dpo_models
        return gr.update(choices=models, value=models[0] if models else None)
    
    # 推理模型加载/卸载事件（借鉴 LLaMA-Factory WebUI）
    infer_refresh.click(refresh_inference_models, outputs=[infer_model, ab_model])
    load_model_btn.click(load_infer_model, inputs=[infer_model], outputs=[infer_status])
    unload_model_btn.click(unload_infer_model, outputs=[infer_status])
    
    # AB测试事件
    ab_refresh_model.click(refresh_inference_models, outputs=[ab_model])
    ab_generate_btn.click(
        generate_ab_responses,
        inputs=[ab_query, ab_model, ab_temperature_a, ab_temperature_b],
        outputs=[response_a_label, response_a, response_b_label, response_b, ab_result]
    )
    
    vote_a_btn.click(vote_for_a, outputs=[ab_result])
    vote_b_btn.click(vote_for_b, outputs=[ab_result])
    
    view_prefs_btn.click(view_preferences, outputs=[prefs_stats])
    export_prefs_btn.click(export_preferences, outputs=[prefs_stats, dpo_dataset])
    
    # 返回训练引擎（供外部调用 resume）
    # 优先返回 CPT 引擎，因为它是第一个阶段
    return train_engines.get('cpt')
