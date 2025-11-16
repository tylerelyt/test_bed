"""
LLMOps Runner
参考 LLaMA-Factory 的 Runner 设计，但直接调用后端而不是命令行
"""
import os
import gradio as gr
from typing import TYPE_CHECKING, Any, Dict
from collections.abc import Generator

from .llamafactory_trainer import get_trainer
from .llmops_control import get_trainer_info

if TYPE_CHECKING:
    from gradio.components import Component
    from .llmops_manager import LLMOpsManager


class LLMOpsRunner:
    """LLMOps 运行器（参考 LLaMA-Factory 的 Runner 设计，但直接调用后端）"""
    
    def __init__(self, manager: "LLMOpsManager"):
        self.manager = manager
        self.trainer = get_trainer()
        self.running = False
        self.aborted = False
        self.running_data = None
        self.do_train = True
    
    def set_abort(self) -> None:
        """停止训练（参考 LLaMA-Factory 的 set_abort）"""
        self.aborted = True
        # trainer 是单例，总是存在
        self.trainer.stop_training()
    
    def _initialize(self, data):
        """验证配置（参考 LLaMA-Factory 的 _initialize）"""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        
        if self.running:
            return "训练任务正在运行中，请先停止当前训练"
        
        training_stage = get("train.training_stage")
        dataset = get("train.dataset")
        output_dir = get("train.output_dir")
        
        if not training_stage:
            return "请选择训练阶段"
        
        # 处理数据集可能是列表的情况
        if isinstance(dataset, list):
            dataset = dataset[0] if dataset else None
        
        if not dataset:
            return "请选择数据集"
        
        if not output_dir:
            return "请指定输出目录"
        
        return ""
    
    def _finalize(self, finish_info: str) -> None:
        """清理资源（参考 LLaMA-Factory 的 _finalize）"""
        # 注意：trainer 是单例，不应该设置为 None
        # 只需要清理运行状态
        self.aborted = False
        self.running = False
        self.running_data = None
    
    def _parse_train_args(self, data):
        """解析训练参数（参考 LLaMA-Factory 的 _parse_train_args）"""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        
        # 基础配置
        dataset = get("train.dataset")
        # 处理数据集可能是列表的情况（参考 LLaMA-Factory）
        if isinstance(dataset, list):
            dataset_str = ",".join(dataset)
        else:
            dataset_str = dataset
        
        config = {
            'stage': get("train.training_stage"),
            'dataset_dir': get("train.dataset_dir"),
            'dataset': dataset_str,
            'model_name_or_path': "Qwen/Qwen2-1.5B",  # 默认值，可以从 top 组件获取
            'template': "qwen",  # 默认值
            'finetuning_type': "lora",  # 默认值
            'learning_rate': float(get("train.learning_rate")),
            'num_train_epochs': float(get("train.num_train_epochs")),
            'max_grad_norm': float(get("train.max_grad_norm")),
            'max_samples': int(get("train.max_samples")),
            'compute_type': get("train.compute_type"),
            'cutoff_len': int(get("train.cutoff_len")),
            'per_device_train_batch_size': int(get("train.batch_size")),
            'gradient_accumulation_steps': int(get("train.gradient_accumulation_steps")),
            'val_size': float(get("train.val_size")),
            'lr_scheduler_type': get("train.lr_scheduler_type"),
            'lora_rank': int(get("train.lora_rank")),
            'lora_alpha': int(get("train.lora_alpha")),
            'lora_dropout': float(get("train.lora_dropout")),
            'lora_target': get("train.lora_target"),
            'pref_beta': float(get("train.pref_beta")),
            'pref_ftx': float(get("train.pref_ftx")),
            'pref_loss': get("train.pref_loss"),
            'logging_steps': int(get("train.logging_steps")),
            'save_steps': int(get("train.save_steps")),
            'warmup_steps': int(get("train.warmup_steps")),
            'output_dir': get("train.output_dir")
        }
        
        return config
    
    def preview_train(self, data):
        """预览训练配置（参考 LLaMA-Factory 的 preview_train）
        
        Args:
            data: 字典，键是 Gradio 组件对象，值是组件的值
                Gradio 会自动将组件集合的值构建成字典传递
        """
        output_box = self.manager.get_elem_by_id("train.output_box")
        
        # 验证配置（参考 LLaMA-Factory 的 _preview）
        error = self._initialize(data)
        if error:
            gr.Warning(error)
            yield {output_box: error}
            return
        
        # 解析训练参数并生成配置预览
        try:
            config = self._parse_train_args(data)
            import yaml
            config_yaml = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
            preview_text = f"**训练配置预览**:\n\n```yaml\n{config_yaml}\n```"
        except Exception as e:
            preview_text = f"配置预览失败: {str(e)}"
        
        yield {output_box: preview_text}
    
    def _launch(self, data):
        """启动训练过程（参考 LLaMA-Factory 的 _launch，但直接调用后端）
        
        Args:
            data: 字典，键是 Gradio 组件对象，值是组件的值
        """
        output_box = self.manager.get_elem_by_id("train.output_box")
        error = self._initialize(data)
        if error:
            gr.Warning(error)
            yield {output_box: error}
            return
        
        # 保存运行数据
        self.running_data = data
        self.do_train = True
        
        # 解析训练参数
        try:
            config = self._parse_train_args(data)
        except Exception as e:
            yield {output_box: f"参数解析失败: {str(e)}"}
            return
        
        # 确保输出目录存在（使用 get_save_dir 处理路径）
        from .llmops_common import get_save_dir
        output_dir = config.get('output_dir', '')
        if os.path.sep not in output_dir or not os.path.isabs(output_dir):
            output_path = get_save_dir(output_dir)
        else:
            output_path = output_dir
        os.makedirs(output_path, exist_ok=True)
        # 更新 config 中的 output_dir 为实际路径
        config['output_dir'] = output_path
        
        # 启动训练任务（在后台线程中运行）
        success = self.trainer.start_training(config)
        
        if success:
            # 开始监控训练进度
            yield from self.monitor()
        else:
            yield {output_box: "❌ 训练任务启动失败"}
            self._finalize("训练启动失败")
    
    def monitor(self):
        """监控训练进度（参考 LLaMA-Factory 的 monitor）"""
        self.aborted = False
        self.running = True
        
        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        output_dir = get("train.output_dir")
        
        # 获取输出路径（参考 LLaMA-Factory 的 get_save_dir）
        from .llmops_common import get_save_dir
        # 如果 output_dir 是相对路径，使用 get_save_dir 处理
        if os.path.sep not in output_dir or not os.path.isabs(output_dir):
            # 使用默认保存目录结构
            output_path = get_save_dir(output_dir)
        else:
            output_path = output_dir
        
        output_box = self.manager.get_elem_by_id("train.output_box")
        progress_bar = self.manager.get_elem_by_id("train.progress_bar")
        
        import time
        
        # 持续监控直到训练完成或中断
        while self.trainer.is_training() and not self.aborted:
            if self.aborted:
                yield {
                    output_box: "正在中断训练...",
                    progress_bar: gr.update(visible=False),
                }
            else:
                # 使用 get_trainer_info 获取训练信息（参考 LLaMA-Factory）
                running_log, running_progress, running_info = get_trainer_info(output_path, self.do_train)
                
                return_dict = {
                    output_box: running_log if running_log else "训练进行中...",
                    progress_bar: running_progress,
                }
                
                # 如果有其他信息（如损失曲线），添加到返回字典
                if running_info:
                    return_dict.update(running_info)
                
                yield return_dict
            
            time.sleep(2)  # 每2秒更新一次
        
        # 训练完成或中断
        if self.aborted:
            finish_info = "训练已中断"
            finish_log = "⏹️ 训练已中断"
        elif self.trainer.is_training():
            finish_info = "训练异常结束"
            finish_log = "⚠️ 训练异常结束"
        else:
            finish_info = "训练完成"
            # 获取最终日志
            running_log, _, _ = get_trainer_info(output_path, self.do_train)
            finish_log = "✅ 训练完成！\n\n" + (running_log if running_log else "")
        
        self._finalize(finish_info)
        yield {
            output_box: finish_log,
            progress_bar: gr.update(visible=False)
        }
    
    def run_train(self, data):
        """运行训练（参考 LLaMA-Factory 的 run_train）"""
        yield from self._launch(data)
    
    def _get_output_elems(self):
        """获取所有输出组件"""
        return [
            self.manager.get_elem_by_id("train.output_box"),
            self.manager.get_elem_by_id("train.progress_bar"),
        ]

