"""
LLMOps 控制逻辑
参考 LLaMA-Factory 的 control.py，提供训练信息读取等功能
"""
import json
import os
from typing import Any, Tuple

import gradio as gr

# LLaMA-Factory 的常量
RUNNING_LOG = "running_log.txt"
TRAINER_LOG = "trainer_log.jsonl"


def get_trainer_info(output_path: os.PathLike, do_train: bool = True):
    """获取训练信息用于监控（参考 LLaMA-Factory 的 get_trainer_info）
    
    Args:
        output_path: 训练输出目录
        do_train: 是否为训练模式
        
    Returns:
        (running_log, running_progress, running_info)
        - running_log: 运行日志文本
        - running_progress: gr.update() 对象用于更新进度条
        - running_info: 其他信息字典（如损失曲线等）
    """
    running_log = ""
    running_progress = gr.update(visible=False)
    running_info = {}
    
    # 读取运行日志
    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        try:
            with open(running_log_path, encoding="utf-8") as f:
                content = f.read()
                # 避免日志过长，只取最后 20000 字符
                running_log = "```\n" + content[-20000:] + "\n```\n"
        except Exception:
            pass
    
    # 读取训练日志
    trainer_log_path = os.path.join(output_path, TRAINER_LOG)
    if os.path.isfile(trainer_log_path):
        trainer_log: list[dict[str, Any]] = []
        try:
            with open(trainer_log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        trainer_log.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            if len(trainer_log) != 0:
                latest_log = trainer_log[-1]
                percentage = latest_log.get("percentage", 0)
                
                # 构建进度条标签
                current_steps = latest_log.get("current_steps", 0)
                total_steps = latest_log.get("total_steps", 0)
                elapsed_time = latest_log.get("elapsed_time", "N/A")
                remaining_time = latest_log.get("remaining_time", "N/A")
                
                label = "运行 {:d}/{:d}: {} < {}".format(
                    current_steps,
                    total_steps,
                    elapsed_time,
                    remaining_time,
                )
                running_progress = gr.update(label=label, value=percentage, visible=True)
                
                # 如果有损失数据，可以添加损失曲线（需要 matplotlib）
                if do_train and "loss" in latest_log:
                    # 可以在这里添加损失曲线绘制
                    # if is_matplotlib_available():
                    #     running_info["loss_viewer"] = gr.Plot(gen_loss_plot(trainer_log))
                    pass
        except Exception:
            pass
    
    return running_log, running_progress, running_info

