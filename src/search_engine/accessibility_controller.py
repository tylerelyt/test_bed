"""
Accessibility Controller 模块
提供通过 Accessibility API 直接控制 UI 元素的功能
相比 PyAutoGUI，这种方式更可靠、更精确

优势：
1. 不依赖坐标，直接操作元素引用
2. 窗口移动后仍然有效
3. 支持更多操作类型（点击、输入、选择等）
4. 更接近真实用户操作
"""

import platform
import logging
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

# macOS 依赖
if platform.system() == "Darwin":
    try:
        import ApplicationServices
        import Quartz
        MACOS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"macOS accessibility 依赖未安装: {e}")
        MACOS_AVAILABLE = False
else:
    MACOS_AVAILABLE = False

# Linux 依赖
if platform.system() == "Linux":
    try:
        import pyatspi
        LINUX_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Linux accessibility 依赖未安装: {e}")
        LINUX_AVAILABLE = False
else:
    LINUX_AVAILABLE = False


class AccessibilityController:
    """
    通过 Accessibility API 控制 UI 元素
    
    示例：
        controller = AccessibilityController()
        # 查找按钮
        button = controller.find_element_by_name("以后", role="AXButton")
        if button:
            # 直接点击按钮（不需要坐标）
            controller.click(button)
    """
    
    def __init__(self):
        self.platform = platform.system()
        
    def find_element_by_name(self, name: str, role: Optional[str] = None, app_name: Optional[str] = None) -> Optional[Any]:
        """
        根据名称查找 UI 元素
        
        Args:
            name: 元素名称（如按钮文本）
            role: 元素角色（如 "AXButton", "AXTextField"）
            app_name: 应用名称（可选，用于缩小搜索范围）
            
        Returns:
            元素引用，如果未找到则返回 None
        """
        if self.platform == "Darwin" and MACOS_AVAILABLE:
            return self._find_element_macos(name, role, app_name)
        elif self.platform == "Linux" and LINUX_AVAILABLE:
            return self._find_element_linux(name, role, app_name)
        else:
            logger.warning(f"平台 {self.platform} 不支持 Accessibility Controller")
            return None
    
    def _find_element_macos(self, name: str, role: Optional[str], app_name: Optional[str]) -> Optional[Any]:
        """macOS: 查找元素"""
        try:
            # 获取所有窗口
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID
            )
            
            if not window_list:
                return None
            
            # 如果指定了应用名称，过滤窗口
            if app_name:
                windows = [w for w in window_list if w.get("kCGWindowOwnerName") == app_name]
            else:
                windows = [w for w in window_list if w.get("kCGWindowLayer") == 0]
            
            # 遍历窗口查找元素
            for window in windows:
                pid = window.get("kCGWindowOwnerPID")
                if pid:
                    app_ref = ApplicationServices.AXUIElementCreateApplication(pid)
                    element = self._search_element_macos(app_ref, name, role)
                    if element:
                        return element
            
            return None
            
        except Exception as e:
            logger.error(f"查找 macOS 元素失败: {e}")
            return None
    
    def _search_element_macos(self, element: Any, name: str, role: Optional[str], depth: int = 0, max_depth: int = 20) -> Optional[Any]:
        """递归搜索 macOS 元素树"""
        if depth > max_depth:
            return None
        
        try:
            # 检查当前元素是否匹配
            error_code, title = ApplicationServices.AXUIElementCopyAttributeValue(element, "AXTitle", None)
            if error_code == 0 and title and name in str(title):
                # 如果指定了角色，检查角色是否匹配
                if role:
                    error_code, elem_role = ApplicationServices.AXUIElementCopyAttributeValue(element, "AXRole", None)
                    if error_code == 0 and elem_role == role:
                        return element
                else:
                    return element
            
            # 搜索子元素
            error_code, children = ApplicationServices.AXUIElementCopyAttributeValue(element, "AXChildren", None)
            if error_code == 0 and children:
                for child in children:
                    result = self._search_element_macos(child, name, role, depth + 1, max_depth)
                    if result:
                        return result
            
            return None
            
        except Exception as e:
            return None
    
    def _find_element_linux(self, name: str, role: Optional[str], app_name: Optional[str]) -> Optional[Any]:
        """Linux: 查找元素（待实现）"""
        logger.warning("Linux Accessibility Controller 尚未实现")
        return None
    
    def click(self, element: Any) -> bool:
        """
        点击元素
        
        Args:
            element: 元素引用（从 find_element_by_name 获取）
            
        Returns:
            是否成功
        """
        if self.platform == "Darwin" and MACOS_AVAILABLE:
            return self._click_macos(element)
        elif self.platform == "Linux" and LINUX_AVAILABLE:
            return self._click_linux(element)
        else:
            return False
    
    def _click_macos(self, element: Any) -> bool:
        """macOS: 点击元素"""
        try:
            # 执行按压动作（等同于点击）
            error_code = ApplicationServices.AXUIElementPerformAction(element, "AXPress")
            if error_code == 0:
                logger.info("✅ 成功通过 Accessibility API 点击元素")
                return True
            else:
                logger.warning(f"⚠️  点击失败，错误码: {error_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 点击元素失败: {e}")
            return False
    
    def _click_linux(self, element: Any) -> bool:
        """Linux: 点击元素"""
        try:
            # AT-SPI 的点击方法
            element.doAction(0)  # 默认动作通常是点击
            logger.info("✅ 成功通过 AT-SPI 点击元素")
            return True
        except Exception as e:
            logger.error(f"❌ 点击元素失败: {e}")
            return False
    
    def set_value(self, element: Any, value: str) -> bool:
        """
        设置元素值（用于文本框等）
        
        Args:
            element: 元素引用
            value: 要设置的值
            
        Returns:
            是否成功
        """
        if self.platform == "Darwin" and MACOS_AVAILABLE:
            return self._set_value_macos(element, value)
        elif self.platform == "Linux" and LINUX_AVAILABLE:
            return self._set_value_linux(element, value)
        else:
            return False
    
    def _set_value_macos(self, element: Any, value: str) -> bool:
        """macOS: 设置元素值"""
        try:
            error_code = ApplicationServices.AXUIElementSetAttributeValue(
                element, 
                "AXValue",
                value
            )
            if error_code == 0:
                logger.info(f"✅ 成功设置元素值: {value}")
                return True
            else:
                logger.warning(f"⚠️  设置值失败，错误码: {error_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 设置元素值失败: {e}")
            return False
    
    def _set_value_linux(self, element: Any, value: str) -> bool:
        """Linux: 设置元素值"""
        try:
            element.set_text(value)
            logger.info(f"✅ 成功设置元素值: {value}")
            return True
        except Exception as e:
            logger.error(f"❌ 设置元素值失败: {e}")
            return False
    
    def get_available_actions(self, element: Any) -> List[str]:
        """
        获取元素支持的所有动作
        
        Args:
            element: 元素引用
            
        Returns:
            动作列表
        """
        if self.platform == "Darwin" and MACOS_AVAILABLE:
            return self._get_actions_macos(element)
        elif self.platform == "Linux" and LINUX_AVAILABLE:
            return self._get_actions_linux(element)
        else:
            return []
    
    def _get_actions_macos(self, element: Any) -> List[str]:
        """macOS: 获取元素支持的动作"""
        try:
            error_code, actions = ApplicationServices.AXUIElementCopyActionNames(element, None)
            if error_code == 0 and actions:
                return list(actions)
            return []
        except Exception as e:
            logger.error(f"获取动作列表失败: {e}")
            return []
    
    def _get_actions_linux(self, element: Any) -> List[str]:
        """Linux: 获取元素支持的动作"""
        try:
            return [element.get_action_name(i) for i in range(element.get_n_actions())]
        except Exception as e:
            logger.error(f"获取动作列表失败: {e}")
            return []


def is_accessibility_controller_available() -> bool:
    """检查当前平台是否支持 Accessibility Controller"""
    os_name = platform.system()
    if os_name == "Darwin":
        return MACOS_AVAILABLE
    elif os_name == "Linux":
        return LINUX_AVAILABLE
    return False


# 示例用法
if __name__ == "__main__":
    controller = AccessibilityController()
    
    # 示例1: 查找并点击按钮
    button = controller.find_element_by_name("以后", role="AXButton")
    if button:
        print("找到按钮，准备点击...")
        success = controller.click(button)
        print(f"点击{'成功' if success else '失败'}")
    else:
        print("未找到按钮")
    
    # 示例2: 查找文本框并输入
    textfield = controller.find_element_by_name("搜索", role="AXTextField")
    if textfield:
        print("找到文本框，准备输入...")
        success = controller.set_value(textfield, "Hello World")
        print(f"输入{'成功' if success else '失败'}")

