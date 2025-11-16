"""
LLaMA-Factory 支持的模型列表配置
参考: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/model_cards.py
"""
from typing import Dict, List, Tuple


class LLaMAFactoryModels:
    """LLaMA-Factory 支持的模型配置"""
    
    # 主流开源模型列表（按系列分组）
    SUPPORTED_MODELS = {
        # Llama 系列
        "llama3": [
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        "llama2": [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ],
        
        # Qwen 系列
        "qwen2": [
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-1.5B",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-72B",
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2.5-32B",
            "Qwen/Qwen2.5-72B",
        ],
        
        # Mistral 系列
        "mistral": [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-v0.1",
        ],
        
        # Yi 系列
        "yi": [
            "01-ai/Yi-6B",
            "01-ai/Yi-6B-Chat",
            "01-ai/Yi-9B",
            "01-ai/Yi-34B",
            "01-ai/Yi-34B-Chat",
            "01-ai/Yi-1.5-6B",
            "01-ai/Yi-1.5-9B",
            "01-ai/Yi-1.5-34B",
        ],
        
        # ChatGLM 系列
        "chatglm": [
            "THUDM/chatglm3-6b",
            "THUDM/chatglm3-6b-base",
            "THUDM/glm-4-9b",
            "THUDM/glm-4-9b-chat",
        ],
        
        # Baichuan 系列
        "baichuan": [
            "baichuan-inc/Baichuan2-7B-Base",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc/Baichuan2-13B-Base",
            "baichuan-inc/Baichuan2-13B-Chat",
        ],
        
        # DeepSeek 系列
        "deepseek": [
            "deepseek-ai/deepseek-llm-7b-base",
            "deepseek-ai/deepseek-llm-7b-chat",
            "deepseek-ai/deepseek-llm-67b-base",
            "deepseek-ai/deepseek-llm-67b-chat",
            "deepseek-ai/DeepSeek-V2",
            "deepseek-ai/DeepSeek-V2-Chat",
        ],
        
        # InternLM 系列
        "internlm": [
            "internlm/internlm2-7b",
            "internlm/internlm2-7b-chat",
            "internlm/internlm2-20b",
            "internlm/internlm2-20b-chat",
        ],
        
        # Phi 系列（微软）
        "phi": [
            "microsoft/phi-2",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ],
        
        # Gemma 系列（Google）
        "gemma": [
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/gemma-7b",
            "google/gemma-7b-it",
            "google/gemma-2-9b",
            "google/gemma-2-27b",
        ],
    }
    
    # 常用模型（用于快速选择）
    POPULAR_MODELS = [
        "meta-llama/Meta-Llama-3-8B",
        "Qwen/Qwen2.5-7B",
        "mistralai/Mistral-7B-v0.3",
        "01-ai/Yi-6B",
        "THUDM/chatglm3-6b",
        "deepseek-ai/deepseek-llm-7b-base",
    ]
    
    # 模型模板映射
    MODEL_TEMPLATES = {
        "llama3": "llama3",
        "llama2": "llama2",
        "qwen2": "qwen",
        "mistral": "mistral",
        "yi": "yi",
        "chatglm": "chatglm3",
        "baichuan": "baichuan2",
        "deepseek": "deepseek",
        "internlm": "intern2",
        "phi": "phi",
        "gemma": "gemma",
    }
    
    @classmethod
    def get_all_models(cls) -> List[str]:
        """获取所有支持的模型"""
        all_models = []
        for models in cls.SUPPORTED_MODELS.values():
            all_models.extend(models)
        return sorted(all_models)
    
    @classmethod
    def get_models_by_series(cls, series: str) -> List[str]:
        """按系列获取模型列表"""
        return cls.SUPPORTED_MODELS.get(series, [])
    
    @classmethod
    def get_popular_models(cls) -> List[str]:
        """获取常用模型列表"""
        return cls.POPULAR_MODELS
    
    @classmethod
    def get_model_series(cls) -> List[str]:
        """获取所有模型系列"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_grouped_choices(cls) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """获取分组的模型选择列表（用于 Gradio Dropdown）
        
        Returns:
            List of (group_name, [(display_name, value), ...])
        """
        grouped = []
        
        # 常用模型组
        grouped.append((
            "⭐ 常用模型",
            [(model, model) for model in cls.POPULAR_MODELS]
        ))
        
        # 按系列分组
        series_display = {
            "llama3": "🦙 Llama 3",
            "llama2": "🦙 Llama 2",
            "qwen2": "🔷 Qwen",
            "mistral": "🌀 Mistral",
            "yi": "🎯 Yi",
            "chatglm": "💬 ChatGLM",
            "baichuan": "🐘 Baichuan",
            "deepseek": "🔍 DeepSeek",
            "internlm": "🧠 InternLM",
            "phi": "Φ Phi",
            "gemma": "💎 Gemma",
        }
        
        for series, models in cls.SUPPORTED_MODELS.items():
            display_name = series_display.get(series, series.upper())
            grouped.append((
                display_name,
                [(model, model) for model in models]
            ))
        
        return grouped
    
    @classmethod
    def get_flat_choices(cls) -> List[str]:
        """获取扁平的模型选择列表（用于简单的 Dropdown）"""
        return cls.get_all_models()
    
    @classmethod
    def get_template_for_model(cls, model_path: str) -> str:
        """根据模型路径推断对应的模板"""
        model_lower = model_path.lower()
        
        if "llama-3" in model_lower or "llama3" in model_lower:
            return "llama3"
        elif "llama-2" in model_lower or "llama2" in model_lower:
            return "llama2"
        elif "qwen" in model_lower:
            return "qwen"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "yi-" in model_lower or "/yi" in model_lower:
            return "yi"
        elif "chatglm" in model_lower or "glm-4" in model_lower:
            return "chatglm3"
        elif "baichuan" in model_lower:
            return "baichuan2"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "internlm" in model_lower:
            return "intern2"
        elif "phi" in model_lower:
            return "phi"
        elif "gemma" in model_lower:
            return "gemma"
        else:
            return "default"

