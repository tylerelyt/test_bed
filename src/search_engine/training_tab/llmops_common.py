"""
LLMOps 通用工具函数
参考 LLaMA-Factory 的 common.py，提供配置管理、路径处理等功能
"""
import os
from typing import Optional
from yaml import safe_dump, safe_load

# 默认目录常量（参考 LLaMA-Factory）
DEFAULT_CACHE_DIR = "llamaboard_cache"
DEFAULT_CONFIG_DIR = "llamaboard_config"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user_config.yaml"


def get_save_dir(*paths: str):
    """获取保存目录路径（参考 LLaMA-Factory 的 get_save_dir）
    
    Args:
        *paths: 路径组件
        
    Returns:
        完整的保存路径
    """
    if os.path.sep in paths[-1]:
        # 如果是绝对路径或包含路径分隔符，直接返回
        return paths[-1]
    
    # 清理路径并组合
    paths = (path.replace(" ", "").strip() for path in paths)
    return os.path.join(DEFAULT_SAVE_DIR, *paths)


def _get_config_path() -> os.PathLike:
    """获取用户配置文件路径"""
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


def load_config():
    """加载用户配置（参考 LLaMA-Factory 的 load_config）"""
    try:
        with open(_get_config_path(), encoding="utf-8") as f:
            return safe_load(f) or {}
    except Exception:
        return {"lang": None, "hub_name": None, "last_model": None, "path_dict": {}, "cache_dir": None}


def save_config(
    lang: str, 
    hub_name: Optional[str] = None, 
    model_name: Optional[str] = None, 
    model_path: Optional[str] = None
) -> None:
    """保存用户配置（参考 LLaMA-Factory 的 save_config）"""
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config.get("lang")
    if hub_name:
        user_config["hub_name"] = hub_name
    if model_name:
        user_config["last_model"] = model_name
    if model_name and model_path:
        if "path_dict" not in user_config:
            user_config["path_dict"] = {}
        user_config["path_dict"][model_name] = model_path
    
    with open(_get_config_path(), "w", encoding="utf-8") as f:
        safe_dump(user_config, f)


def get_time():
    """获取当前时间戳字符串（用于生成唯一目录名）"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

