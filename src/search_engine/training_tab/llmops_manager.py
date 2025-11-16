"""
LLMOps 组件管理器
参考 LLaMA-Factory 的 Manager 设计，管理所有 Gradio 组件
"""
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    import gradio as gr
    from gradio.components import Component


class LLMOpsManager:
    """管理所有 Gradio 组件的管理器（参考 LLaMA-Factory 的 Manager 设计）"""
    
    def __init__(self):
        self._id_to_elem: dict[str, "Component"] = {}
        self._elem_to_id: dict["Component", str] = {}
    
    def add_elems(self, tab_name: str, elem_dict: dict[str, "Component"]) -> None:
        """添加组件到管理器
        
        Args:
            tab_name: 标签页名称（如 "train", "top"）
            elem_dict: 组件字典，键是组件名称，值是 Gradio 组件
        """
        for elem_name, elem in elem_dict.items():
            elem_id = f"{tab_name}.{elem_name}"
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id
    
    def get_elem_list(self) -> list["Component"]:
        """返回所有组件的列表"""
        return list(self._id_to_elem.values())
    
    def get_elem_by_id(self, elem_id: str) -> "Component":
        """通过 ID 获取组件
        
        Example: "train.dataset", "top.model_name"
        """
        return self._id_to_elem.get(elem_id)
    
    def get_id_by_elem(self, elem: "Component") -> str:
        """通过组件获取 ID"""
        return self._elem_to_id.get(elem)
    
    def get_elem_iter(self) -> Generator[tuple[str, "Component"], None, None]:
        """返回所有组件的迭代器（名称, 组件）"""
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

