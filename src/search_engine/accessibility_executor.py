"""
Accessibility Executor 模块
智能执行器，支持混合使用 PyAutoGUI 和 Accessibility API

核心思路：
1. 模型可以选择两种输出格式
2. 执行器自动识别和转换
3. 优先使用 Accessibility API（更精确），失败时回退到 PyAutoGUI

支持的命令格式：
- PyAutoGUI: pyautogui.click(1408, 752)
- Accessibility: click_element("以后", role="AXButton")
- Accessibility: set_element_value("搜索框", "keyword", role="AXTextField")
"""

import re
import logging
from typing import Optional, List, Tuple, Dict, Any
import pyautogui

from .accessibility_controller import AccessibilityController, is_accessibility_controller_available

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False


class AccessibilityExecutor:
    """
    混合执行器：智能选择 Accessibility API 或 PyAutoGUI
    
    示例：
        executor = AccessibilityExecutor()
        
        # 方式1：直接执行 PyAutoGUI 命令
        executor.execute('pyautogui.click(1408, 752)')
        
        # 方式2：执行 Accessibility 命令（新格式）
        executor.execute('click_element("以后", role="AXButton")')
        
        # 方式3：智能转换（推荐）
        # 如果 accessibility tree 中有元素名称，自动使用 API；否则用坐标
        executor.execute_smart('pyautogui.click(1408, 752)', 
                              accessibility_tree=tree,
                              element_hint="以后")
    """
    
    def __init__(self, enable_accessibility: bool = True, fallback_to_pyautogui: bool = True):
        """
        Args:
            enable_accessibility: 是否启用 Accessibility API
            fallback_to_pyautogui: Accessibility 失败时是否回退到 PyAutoGUI
        """
        self.enable_accessibility = enable_accessibility and is_accessibility_controller_available()
        self.fallback_to_pyautogui = fallback_to_pyautogui
        
        if self.enable_accessibility:
            self.controller = AccessibilityController()
            logger.info("✅ AccessibilityExecutor 初始化完成（支持 Accessibility API）")
        else:
            self.controller = None
            logger.info("ℹ️  AccessibilityExecutor 初始化完成（仅支持 PyAutoGUI）")
    
    def execute(self, command: str) -> bool:
        """
        执行单条命令（自动识别格式）
        
        Args:
            command: 命令字符串
            
        Returns:
            是否执行成功
        """
        command = command.strip()
        
        # 检查是否是 Accessibility 命令格式
        if self.enable_accessibility and self._is_accessibility_command(command):
            return self._execute_accessibility_command(command)
        
        # 否则按 PyAutoGUI 命令执行
        return self._execute_pyautogui_command(command)
    
    def _is_accessibility_command(self, command: str) -> bool:
        """判断是否是 Accessibility 命令"""
        accessibility_patterns = [
            r'click_element\(',
            r'set_element_value\(',
            r'find_element\(',
        ]
        return any(re.search(pattern, command) for pattern in accessibility_patterns)
    
    def _execute_accessibility_command(self, command: str) -> bool:
        """执行 Accessibility 命令"""
        try:
            # 解析 click_element("name", role="role") 格式
            click_match = re.search(r'click_element\(["\']([^"\']+)["\'](?:,\s*role=["\']([^"\']+)["\'])?\)', command)
            if click_match:
                name = click_match.group(1)
                role = click_match.group(2)
                logger.info(f"🔍 查找元素: {name} (role={role})")
                
                element = self.controller.find_element_by_name(name, role=role)
                if element:
                    success = self.controller.click(element)
                    if success:
                        return True
                    elif self.fallback_to_pyautogui:
                        logger.warning("⚠️  Accessibility API 点击失败，将回退到 PyAutoGUI")
                        return False
                else:
                    logger.warning(f"⚠️  未找到元素: {name}")
                    return False
            
            # 解析 set_element_value("name", "value", role="role") 格式
            setvalue_match = re.search(r'set_element_value\(["\']([^"\']+)["\'],\s*["\']([^"\']+)["\'](?:,\s*role=["\']([^"\']+)["\'])?\)', command)
            if setvalue_match:
                name = setvalue_match.group(1)
                value = setvalue_match.group(2)
                role = setvalue_match.group(3)
                logger.info(f"🔍 查找元素: {name} (role={role})")
                
                element = self.controller.find_element_by_name(name, role=role)
                if element:
                    success = self.controller.set_value(element, value)
                    if success:
                        return True
                else:
                    logger.warning(f"⚠️  未找到元素: {name}")
                    return False
            
            logger.warning(f"⚠️  无法解析 Accessibility 命令: {command}")
            return False
            
        except Exception as e:
            logger.error(f"❌ 执行 Accessibility 命令失败: {e}")
            return False
    
    def _execute_pyautogui_command(self, command: str) -> bool:
        """执行 PyAutoGUI 命令"""
        try:
            # 安全检查：只允许 pyautogui 命令
            if not command.startswith('pyautogui.'):
                logger.warning(f"⚠️  非 pyautogui 命令: {command}")
                return False
            
            # 执行命令
            logger.info(f"🖱️  执行 PyAutoGUI: {command}")
            exec(f"import pyautogui; import time; pyautogui.FAILSAFE = False; {command}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 执行 PyAutoGUI 命令失败: {e}")
            return False
    
    def execute_smart(self, 
                     command: str, 
                     accessibility_tree: Optional[str] = None,
                     element_hint: Optional[str] = None) -> bool:
        """
        智能执行：尝试从坐标命令转换为 Accessibility API 调用
        
        Args:
            command: PyAutoGUI 命令（如 pyautogui.click(1408, 752)）
            accessibility_tree: Accessibility Tree XML 字符串
            element_hint: 元素名称提示（模型可以在注释中提供）
            
        Returns:
            是否执行成功
        """
        # 如果不支持 Accessibility 或没有 tree，直接执行 PyAutoGUI
        if not self.enable_accessibility or not accessibility_tree:
            return self._execute_pyautogui_command(command)
        
        # 尝试从命令中提取坐标和 element_hint
        coord_match = re.search(r'pyautogui\.click\((\d+),\s*(\d+)\)', command)
        if not coord_match:
            return self._execute_pyautogui_command(command)
        
        x, y = int(coord_match.group(1)), int(coord_match.group(2))
        
        # 如果没有提供 element_hint，尝试从注释中提取
        if not element_hint:
            comment_match = re.search(r'#.*["\']([^"\']+)["\']', command)
            if comment_match:
                element_hint = comment_match.group(1)
        
        # 如果有 element_hint，尝试在 tree 中查找并使用 API
        if element_hint:
            logger.info(f"🔍 尝试使用 Accessibility API 查找: {element_hint}")
            
            # 从 accessibility tree 中查找元素
            if self._find_element_in_tree(accessibility_tree, element_hint, x, y):
                logger.info(f"✅ 在 accessibility tree 中找到元素，尝试使用 API")
                element = self.controller.find_element_by_name(element_hint)
                if element:
                    if self.controller.click(element):
                        logger.info("✅ 使用 Accessibility API 成功")
                        return True
                    else:
                        logger.warning("⚠️  Accessibility API 失败，回退到 PyAutoGUI")
        
        # 回退到 PyAutoGUI
        logger.info(f"🖱️  使用 PyAutoGUI 坐标点击: ({x}, {y})")
        return self._execute_pyautogui_command(command)
    
    def _find_element_in_tree(self, tree: str, name: str, x: int, y: int) -> bool:
        """
        在 accessibility tree 中查找元素，验证坐标是否匹配
        
        Args:
            tree: XML 格式的 accessibility tree
            name: 元素名称
            x, y: 坐标
            
        Returns:
            是否找到匹配的元素
        """
        try:
            # 简单检查：名称是否在 tree 中，坐标是否接近
            if name not in tree:
                return False
            
            # 查找包含该名称和坐标的元素
            # 例如：<button name="Later" cp:screencoord="(1408, 752)">
            pattern = rf'{re.escape(name)}.*cp:screencoord="\((\d+),\s*(\d+)\)"'
            match = re.search(pattern, tree, re.DOTALL)
            
            if match:
                tree_x, tree_y = int(match.group(1)), int(match.group(2))
                # 允许 5 像素的误差
                if abs(tree_x - x) <= 5 and abs(tree_y - y) <= 5:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 在 tree 中查找元素失败: {e}")
            return False


def create_enhanced_system_prompt() -> str:
    """
    生成支持 Accessibility API 的增强版 system prompt
    
    返回的 prompt 会告诉模型可以使用两种方式：
    1. PyAutoGUI (通用)
    2. Accessibility API (更精确)
    """
    base_prompt = """You are an agent which follow my instruction and perform desktop computer tasks as instructed.

You can use TWO methods to control the desktop:

**Method 1: PyAutoGUI (Coordinate-based) - Default**
```python
pyautogui.click(1408, 752)  # Click at specific coordinates
pyautogui.typewrite("text")
```

**Method 2: Accessibility API (Element-based) - Recommended when element name is clear**
```python
click_element("Later", role="AXButton")  # Click by element name (no coordinates needed!)
set_element_value("Search", "keyword", role="AXTextField")
```

**When to use which method:**
- Use Method 2 (Accessibility API) when:
  ✅ You can clearly identify the element name from accessibility tree
  ✅ You want more robust execution (works even if window moves)
  
- Use Method 1 (PyAutoGUI) when:
  ✅ Element has no clear name in accessibility tree
  ✅ Need to interact with specific screen regions
  ✅ Fallback option

**Example with both methods:**
```python
# Good: Use Accessibility API when element name is clear
click_element("Close", role="AXButton")  

# Acceptable: Use PyAutoGUI with comment indicating element
pyautogui.click(1408, 752)  # Click "Later" button

# Also good: Mix both methods
click_element("Username", role="AXTextField")
set_element_value("Username", "admin")
pyautogui.press("enter")
```

Return your code inside a code block. Return DONE when finished, FAIL when impossible, WAIT when need to wait."""
    
    return base_prompt


# 示例用法
if __name__ == "__main__":
    executor = AccessibilityExecutor()
    
    # 测试1: PyAutoGUI 命令
    print("\n测试1: PyAutoGUI 命令")
    executor.execute('pyautogui.click(100, 100)')
    
    # 测试2: Accessibility 命令
    print("\n测试2: Accessibility 命令")
    executor.execute('click_element("以后", role="AXButton")')
    
    # 测试3: 智能执行
    print("\n测试3: 智能执行（带提示）")
    tree = '<button name="Later" cp:screencoord="(1408, 752)"></button>'
    executor.execute_smart(
        'pyautogui.click(1408, 752)  # Click "Later" button',
        accessibility_tree=tree,
        element_hint="Later"
    )

