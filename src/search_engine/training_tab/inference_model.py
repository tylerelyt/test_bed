"""
推理模型管理器
借鉴 LLaMA-Factory WebUI 的 WebChatModel 架构
直接使用 transformers 加载模型，无需独立API服务
"""
import os
from typing import Any, Dict, Generator, List, Optional
from llamafactory.chat import ChatModel


class InferenceModel(ChatModel):
    """推理模型管理器（借鉴 LLaMA-Factory 的 WebChatModel）"""
    
    def __init__(self):
        self.engine: Optional[Any] = None
        self._loaded = False
    
    @property
    def loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded and self.engine is not None
    
    def load_model(self, 
                   base_model: str,
                   adapter_path: Optional[str] = None,
                   template: str = "qwen",
                   **kwargs) -> Generator[str, None, None]:
        """加载模型（支持base模型 + LoRA adapter）
        
        Args:
            base_model: 基础模型路径（如 "Qwen/Qwen2-0.5B"）
            adapter_path: LoRA adapter路径（可选，如 "checkpoints/sft/test_qwen_sft"）
            template: 对话模板名称
            **kwargs: 其他transformers参数
            
        Yields:
            加载状态消息
        """
        if self.loaded:
            yield "⚠️ 模型已加载，请先卸载"
            return
        
        if not base_model:
            yield "❌ 请指定基础模型路径"
            return
        
        try:
            yield "⏳ 正在加载模型..."
            
            # 构建参数（参考 LLaMA-Factory 的 load_model）
            args = {
                "model_name_or_path": base_model,
                "template": template,
                "trust_remote_code": True,
                "infer_backend": "huggingface",  # 使用 transformers
            }
            
            # 添加 LoRA adapter
            if adapter_path and os.path.exists(adapter_path):
                args["adapter_name_or_path"] = adapter_path
                args["finetuning_type"] = "lora"
            
            # 添加其他参数
            args.update(kwargs)
            
            # 调用父类初始化（这会加载模型到内存）
            super().__init__(args)
            self._loaded = True
            
            model_info = f"基础模型: {base_model}"
            if adapter_path:
                model_info += f"\nLoRA Adapter: {adapter_path}"
            
            yield f"✅ 模型加载成功\n{model_info}"
            
        except Exception as e:
            self._loaded = False
            yield f"❌ 模型加载失败: {str(e)}"
    
    def unload_model(self) -> Generator[str, None, None]:
        """卸载模型"""
        if not self.loaded:
            yield "⚠️ 模型未加载"
            return
        
        try:
            yield "⏳ 正在卸载模型..."
            self.engine = None
            self._loaded = False
            
            # 清理GPU内存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield "✅ 模型已卸载"
            
        except Exception as e:
            yield f"❌ 卸载失败: {str(e)}"
    
    def generate(self,
                 prompt: str,
                 system: Optional[str] = None,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **kwargs) -> Generator[str, None, None]:
        """生成文本（流式输出）
        
        Args:
            prompt: 用户输入
            system: 系统提示（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样
            **kwargs: 其他生成参数
            
        Yields:
            生成的文本片段
        """
        if not self.loaded:
            yield "❌ 请先加载模型"
            return
        
        try:
            # 构建消息列表
            messages = [{"role": "user", "content": prompt}]
            
            # 调用父类的 stream_chat 方法（这是 LLaMA-Factory 的核心）
            for new_text in self.stream_chat(
                messages,
                system=system or "",
                tools="",
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                **kwargs
            ):
                yield new_text
                
        except Exception as e:
            yield f"\n\n❌ 生成失败: {str(e)}"
    
    def generate_once(self,
                      prompt: str,
                      system: Optional[str] = None,
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      **kwargs) -> str:
        """生成文本（一次性返回完整结果）
        
        用于AB测试等需要完整结果的场景
        """
        result = ""
        for text in self.generate(
            prompt=prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        ):
            result += text
        return result

