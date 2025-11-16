"""
LLMOps Engine
参考 LLaMA-Factory 的 Engine 设计，协调 Manager 和 Runner
"""
import gradio as gr
from typing import TYPE_CHECKING, Any
from collections.abc import Generator

from .llmops_manager import LLMOpsManager
from .llmops_runner import LLMOpsRunner
from .llmops_common import load_config, get_time

if TYPE_CHECKING:
    from gradio.components import Component


class LLMOpsEngine:
    """LLMOps 引擎（参考 LLaMA-Factory 的 Engine 设计）"""
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.manager = LLMOpsManager()
        self.runner = LLMOpsRunner(self.manager)
    
    def _update_component(self, input_dict):
        """更新组件属性（参考 LLaMA-Factory 的 _update_component）"""
        output_dict = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            if elem is not None:
                # 使用 gr.update() 更新组件值，而不是创建新组件
                output_dict[elem] = gr.update(**elem_attr)
        return output_dict
    
    def resume(self):
        """恢复组件初始状态（参考 LLaMA-Factory 的 resume）"""
        user_config = load_config() if not self.demo_mode else {}
        lang = user_config.get("lang") or "zh"
        
        init_dict = {
            "train.output_dir": {"value": f"train_{get_time()}"}
        }
        
        yield self._update_component(init_dict)
        
        # 如果训练正在运行，恢复训练状态
        if self.runner.running and not self.demo_mode and self.runner.running_data:
            output_dict = {}
            for elem, value in self.runner.running_data.items():
                output_dict[elem] = gr.update(value=value)
            yield output_dict
    
    def change_lang(self, lang):
        """切换语言（参考 LLaMA-Factory 的 change_lang）
        
        注意：当前版本暂不支持国际化，此方法保留接口以便后续扩展
        """
        # TODO: 实现国际化支持
        return {}

