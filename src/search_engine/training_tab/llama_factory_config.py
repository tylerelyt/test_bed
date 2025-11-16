"""
LLaMA-Factory 训练配置管理器
支持 CPT、SFT、DPO 三种训练模式
"""
import yaml
import os
from typing import Dict, Any, Optional
from datetime import datetime


class LLaMAFactoryConfig:
    """LLaMA-Factory 训练配置管理"""
    
    def __init__(self, config_dir: str = "configs/llmops"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    @staticmethod
    def create_cpt_config(
        model_name: str = "Qwen/Qwen2-1.5B",
        dataset: str = "domain_corpus",
        dataset_dir: str = "data/llmops",
        output_dir: str = "checkpoints/domain-cpt",
        num_train_epochs: int = 1,
        learning_rate: float = 1e-5,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        max_seq_length: int = 2048,
        save_steps: int = 500,
        logging_steps: int = 50,
        fp16: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32
    ) -> Dict[str, Any]:
        """
        创建 CPT（继续预训练）配置
        
        Args:
            model_name: 基础模型名称
            dataset: 数据集名称
            dataset_dir: 数据集目录
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            learning_rate: 学习率
            per_device_train_batch_size: 单设备批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_seq_length: 最大序列长度
            save_steps: 保存检查点步数
            logging_steps: 日志记录步数
            fp16: 是否使用混合精度
            use_lora: 是否使用 LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        config = {
            "model_name_or_path": model_name,
            "stage": "pt",  # pretrain
            "do_train": True,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "fp16": fp16,
            "save_strategy": "steps",
            "logging_strategy": "steps",
        }
        
        if use_lora:
            config.update({
                "finetuning_type": "lora",
                "lora_target": "all",
                "lora_rank": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": 0.05,
            })
        
        return config
    
    @staticmethod
    def create_sft_config(
        model_name: str = "Qwen/Qwen2-1.5B",
        dataset: str = "sft_data",
        dataset_dir: str = "data/llmops",
        output_dir: str = "checkpoints/sft-lora",
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
        save_steps: int = 100,
        logging_steps: int = 10,
        fp16: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        template: str = "qwen"
    ) -> Dict[str, Any]:
        """
        创建 SFT（指令微调）配置
        
        Args:
            model_name: 基础模型或 CPT 后的模型路径
            dataset: 数据集名称
            dataset_dir: 数据集目录
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            learning_rate: 学习率
            per_device_train_batch_size: 单设备批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_seq_length: 最大序列长度
            save_steps: 保存步数
            logging_steps: 日志步数
            fp16: 是否使用混合精度
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            template: 模板类型
        """
        config = {
            "model_name_or_path": model_name,
            "stage": "sft",
            "do_train": True,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "template": template,
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "fp16": fp16,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "evaluation_strategy": "no",
        }
        
        return config
    
    @staticmethod
    def create_dpo_config(
        model_name: str = "checkpoints/sft-lora",
        dataset: str = "prefs_data",
        dataset_dir: str = "data/llmops",
        output_dir: str = "checkpoints/dpo-lora",
        num_train_epochs: int = 1,
        learning_rate: float = 5e-6,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        max_seq_length: int = 2048,
        save_steps: int = 100,
        logging_steps: int = 10,
        fp16: bool = True,
        beta: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        template: str = "qwen"
    ) -> Dict[str, Any]:
        """
        创建 DPO（偏好对齐）配置
        
        Args:
            model_name: SFT 后的模型路径
            dataset: 偏好数据集名称
            dataset_dir: 数据集目录
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            learning_rate: 学习率（DPO 通常用更小的学习率）
            per_device_train_batch_size: 单设备批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_seq_length: 最大序列长度
            save_steps: 保存步数
            logging_steps: 日志步数
            fp16: 是否使用混合精度
            beta: DPO 温度参数
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            template: 模板类型
        """
        config = {
            "model_name_or_path": model_name,
            "stage": "dpo",
            "do_train": True,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "template": template,
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "fp16": fp16,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.05,
            "dpo_beta": beta,
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "evaluation_strategy": "no",
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str = None, stage: str = "sft") -> str:
        """
        保存配置到 YAML 文件
        
        Args:
            config: 配置字典
            filename: 文件名（可选）
            stage: 训练阶段
        
        Returns:
            配置文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stage}_config_{timestamp}.yaml"
        
        filepath = os.path.join(self.config_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return filepath
    
    def load_config(self, filepath: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def get_training_command(config_file: str, use_llamafactory_cli: bool = True) -> str:
        """
        生成训练命令
        
        Args:
            config_file: 配置文件路径
            use_llamafactory_cli: 是否使用 llamafactory-cli（否则使用 python -m）
        
        Returns:
            训练命令字符串
        """
        if use_llamafactory_cli:
            return f"llamafactory-cli train {config_file}"
        else:
            return f"python -m llmtuner.train {config_file}"
    
    @staticmethod
    def create_dataset_info(
        dataset_name: str,
        file_name: str,
        formatting: str = "sharegpt",
        columns: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        创建数据集信息配置（用于 dataset_info.json）
        
        Args:
            dataset_name: 数据集名称
            file_name: 文件名
            formatting: 格式类型（alpaca, sharegpt 等）
            columns: 列名映射
        
        Returns:
            数据集信息字典
        """
        dataset_info = {
            dataset_name: {
                "file_name": file_name,
                "formatting": formatting,
            }
        }
        
        if columns:
            dataset_info[dataset_name]["columns"] = columns
        
        return dataset_info

