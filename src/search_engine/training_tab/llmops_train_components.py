"""
LLMOps 训练组件
参考 LLaMA-Factory 的 create_train_tab 设计，创建训练配置界面
"""
import gradio as gr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gradio.components import Component
    from .llmops_engine import LLMOpsEngine


def create_train_tab_components(engine: "LLMOpsEngine"):
    """创建训练配置组件（参考 LLaMA-Factory 的 create_train_tab 设计）
    
    Args:
        engine: LLMOps 引擎实例
        
    Returns:
        组件字典，键是组件名称，值是 Gradio 组件
    """
    # 参考 LLaMA-Factory 的设计：使用 input_elems 集合跟踪所有输入组件
    # Gradio 会自动将组件集合的值构建成字典传递给函数
    input_elems = set()
    elem_dict = {}
    
    # 顶部：训练阶段和数据集选择（参考 LLaMA-Factory 设计）
    with gr.Row():
        training_stage = gr.Dropdown(
            choices=["pt", "sft", "dpo"],
            value="sft",
            label="训练阶段",
            info="pt: 继续预训练, sft: 指令微调, dpo: 偏好对齐",
            scale=1
        )
        dataset_dir = gr.Textbox(
            value="data/llmops",
            label="数据目录",
            scale=1
        )
        dataset = gr.Dropdown(
            choices=["domain_corpus", "sft_data", "prefs_data"],
            value="sft_data",
            label="数据集",
            multiselect=True,  # 支持多选（参考 LLaMA-Factory）
            allow_custom_value=True,
            scale=4
        )
    
    input_elems.update([training_stage, dataset_dir, dataset])
    elem_dict.update({
        "training_stage": training_stage,
        "dataset_dir": dataset_dir,
        "dataset": dataset
    })
    
    # 训练参数（参考 LLaMA-Factory 的布局）
    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5", label="学习率")
        num_train_epochs = gr.Textbox(value="3.0", label="训练轮数")
        max_grad_norm = gr.Textbox(value="1.0", label="最大梯度范数")
        max_samples = gr.Textbox(value="100000", label="最大样本数")
        compute_type = gr.Dropdown(
            choices=["bf16", "fp16", "fp32"],
            value="bf16",
            label="计算类型"
        )
    
    input_elems.update([learning_rate, num_train_epochs, max_grad_norm, max_samples, compute_type])
    elem_dict.update({
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "max_grad_norm": max_grad_norm,
        "max_samples": max_samples,
        "compute_type": compute_type
    })
    
    with gr.Row():
        cutoff_len = gr.Slider(
            minimum=4, maximum=131072, value=2048, step=1,
            label="最大序列长度"
        )
        batch_size = gr.Slider(
            minimum=1, maximum=1024, value=2, step=1,
            label="批次大小"
        )
        gradient_accumulation_steps = gr.Slider(
            minimum=1, maximum=1024, value=8, step=1,
            label="梯度累积步数"
        )
        val_size = gr.Slider(
            minimum=0, maximum=1, value=0, step=0.001,
            label="验证集比例"
        )
        lr_scheduler_type = gr.Dropdown(
            choices=["cosine", "linear", "constant"],
            value="cosine",
            label="学习率调度器"
        )
    
    input_elems.update([cutoff_len, batch_size, gradient_accumulation_steps, val_size, lr_scheduler_type])
    elem_dict.update({
        "cutoff_len": cutoff_len,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "val_size": val_size,
        "lr_scheduler_type": lr_scheduler_type
    })
    
    # LoRA 配置（可折叠）
    with gr.Accordion("LoRA 配置", open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(minimum=1, maximum=1024, value=8, step=1, label="LoRA Rank")
            lora_alpha = gr.Slider(minimum=1, maximum=2048, value=16, step=1, label="LoRA Alpha")
            lora_dropout = gr.Slider(minimum=0, maximum=1, value=0.05, step=0.01, label="LoRA Dropout")
            lora_target = gr.Textbox(value="all", label="LoRA 目标模块")
    
    input_elems.update([lora_rank, lora_alpha, lora_dropout, lora_target])
    elem_dict.update({
        "lora_tab": lora_tab,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target": lora_target
    })
    
    # DPO 配置（可折叠）
    with gr.Accordion("DPO 配置（仅 DPO 阶段）", open=False) as rlhf_tab:
        with gr.Row():
            pref_beta = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.01, label="DPO Beta")
            pref_ftx = gr.Slider(minimum=0, maximum=10, value=0, step=0.01, label="参考模型权重")
            pref_loss = gr.Dropdown(
                choices=["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"],
                value="sigmoid",
                label="损失函数"
            )
    
    input_elems.update([pref_beta, pref_ftx, pref_loss])
    elem_dict.update({
        "rlhf_tab": rlhf_tab,
        "pref_beta": pref_beta,
        "pref_ftx": pref_ftx,
        "pref_loss": pref_loss
    })
    
    # 其他配置（可折叠）
    with gr.Accordion("其他配置", open=False) as extra_tab:
        with gr.Row():
            logging_steps = gr.Slider(minimum=1, maximum=1000, value=5, step=5, label="日志步数")
            save_steps = gr.Slider(minimum=10, maximum=5000, value=100, step=10, label="保存步数")
            warmup_steps = gr.Slider(minimum=0, maximum=5000, value=0, step=1, label="预热步数")
    
    input_elems.update([logging_steps, save_steps, warmup_steps])
    elem_dict.update({
        "extra_tab": extra_tab,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "warmup_steps": warmup_steps
    })
    
    # 输出目录
    with gr.Row():
        output_dir = gr.Textbox(
            value="",
            label="输出目录",
            placeholder="例如: checkpoints/sft-lora",
            info="训练结果保存路径"
        )
    
    input_elems.add(output_dir)
    elem_dict.update({
        "output_dir": output_dir
    })
    
    # 操作按钮
    with gr.Row():
        cmd_preview_btn = gr.Button("📋 预览配置", variant="secondary")
        start_btn = gr.Button("🚀 开始训练", variant="primary")
        stop_btn = gr.Button("⏹️ 停止训练", variant="stop")
    
    elem_dict.update({
        "cmd_preview_btn": cmd_preview_btn,
        "start_btn": start_btn,
        "stop_btn": stop_btn
    })
    
    # 输出区域
    with gr.Row():
        with gr.Column(scale=3):
            output_box = gr.Markdown(label="训练输出")
            progress_bar = gr.Slider(
                minimum=0, maximum=100, value=0, step=1,
                label="训练进度",
                visible=False,
                interactive=False
            )
        with gr.Column(scale=1):
            training_status = gr.HTML(value="<p>未开始训练</p>")
    
    elem_dict.update({
        "output_box": output_box,
        "progress_bar": progress_bar,
        "training_status": training_status
    })
    
    # 事件绑定（参考 LLaMA-Factory 的设计，在函数内部完成）
    # Gradio 会自动将组件集合的值构建成字典传递给函数
    output_elems = [output_box, progress_bar]
    
    cmd_preview_btn.click(
        engine.runner.preview_train,
        input_elems,  # 使用集合，Gradio 会自动构建字典
        output_elems,
        concurrency_limit=None
    )
    start_btn.click(
        engine.runner.run_train,
        input_elems,  # 使用集合，Gradio 会自动构建字典
        output_elems
    )
    stop_btn.click(engine.runner.set_abort)
    
    return elem_dict

