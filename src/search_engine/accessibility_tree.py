"""
Accessibility Tree 模块
基于 OSWorld 的实现，用于获取系统 UI 元素的结构化信息

支持平台：
- macOS: 使用 ApplicationServices AX API
- Linux: 使用 AT-SPI (pyatspi) - 待实现
- Windows: 使用 UI Automation - 待实现
"""

import platform
import logging
import re
from typing import Optional, Any, Dict, List, Set
import concurrent.futures

logger = logging.getLogger(__name__)

# macOS 依赖
if platform.system() == "Darwin":
    try:
        import ApplicationServices
        import Quartz
        import AppKit
        import lxml.etree
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
        from pyatspi import StateType, Accessible, Component
        import lxml.etree
        LINUX_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Linux accessibility 依赖未安装: {e}")
        LINUX_AVAILABLE = False
else:
    LINUX_AVAILABLE = False


# Accessibility Tree XML Namespaces (与 OSWorld 完全一致)
_accessibility_ns_map = {
    "macos": {
        "st": "https://accessibility.macos.example.org/ns/state",
        "attr": "https://accessibility.macos.example.org/ns/attributes",
        "cp": "https://accessibility.macos.example.org/ns/component",
        "doc": "https://accessibility.macos.example.org/ns/document",
        "txt": "https://accessibility.macos.example.org/ns/text",
        "val": "https://accessibility.macos.example.org/ns/value",
        "act": "https://accessibility.macos.example.org/ns/action",
        "role": "https://accessibility.macos.example.org/ns/role",
    },
    "ubuntu": {
        "st": "https://accessibility.ubuntu.example.org/ns/state",
        "attr": "https://accessibility.ubuntu.example.org/ns/attributes",
        "cp": "https://accessibility.ubuntu.example.org/ns/component",
        "doc": "https://accessibility.ubuntu.example.org/ns/document",
        "docattr": "https://accessibility.ubuntu.example.org/ns/document/attributes",
        "txt": "https://accessibility.ubuntu.example.org/ns/text",
        "val": "https://accessibility.ubuntu.example.org/ns/value",
        "act": "https://accessibility.ubuntu.example.org/ns/action",
    }
}

# OSWorld 常量
MAX_DEPTH = 50
MAX_WIDTH = 1024
MAX_CALLS = 5000


def _create_axui_node_macos(
    node: Any,
    nodes: Optional[Set] = None,
    depth: int = 0,
    bbox: Optional[tuple] = None
) -> Optional[Any]:
    """
    为 macOS AX UI 元素创建 XML 节点
    完全基于 OSWorld 的实现：https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/server/main.py
    """
    if not MACOS_AVAILABLE:
        return None
        
    nodes = nodes or set()
    if node in nodes:
        return None
    nodes.add(node)

    # OSWorld 的 reserved_keys 映射（完全一致）
    reserved_keys = {
        "AXEnabled": "st",
        "AXFocused": "st",
        "AXFullScreen": "st",
        "AXTitle": "attr",
        "AXChildrenInNavigationOrder": "attr",
        "AXChildren": "attr",
        "AXFrame": "attr",
        "AXRole": "role",
        "AXHelp": "attr",
        "AXRoleDescription": "role",
        "AXSubrole": "role",
        "AXURL": "attr",
        "AXValue": "val",
        "AXDescription": "attr",
        "AXDOMIdentifier": "attr",
        "AXSelected": "st",
        "AXInvalid": "st",
        "AXRows": "attr",
        "AXColumns": "attr",
    }
    attribute_dict = {}

    if depth == 0:
        bbox = (
            node["kCGWindowBounds"]["X"],
            node["kCGWindowBounds"]["Y"],
            node["kCGWindowBounds"]["X"] + node["kCGWindowBounds"]["Width"],
            node["kCGWindowBounds"]["Y"] + node["kCGWindowBounds"]["Height"]
        )
        app_ref = ApplicationServices.AXUIElementCreateApplication(node["kCGWindowOwnerPID"])

        attribute_dict["name"] = node["kCGWindowOwnerName"]
        if attribute_dict["name"] != "Dock":
            error_code, app_wins_ref = ApplicationServices.AXUIElementCopyAttributeValue(
                app_ref, "AXWindows", None)
            if error_code:
                logger.error("MacOS parsing %s encountered Error code: %d", app_ref, error_code)
                return None
        else:
            app_wins_ref = [app_ref]
        node = app_wins_ref[0]

    error_code, attr_names = ApplicationServices.AXUIElementCopyAttributeNames(node, None)

    if error_code:
        # -25202: AXError.invalidUIElement
        return None

    value = None

    if "AXFrame" in attr_names:
        error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, "AXFrame", None)
        rep = repr(attr_val)
        x_value = re.search(r"x:(-?[\d.]+)", rep)
        y_value = re.search(r"y:(-?[\d.]+)", rep)
        w_value = re.search(r"w:(-?[\d.]+)", rep)
        h_value = re.search(r"h:(-?[\d.]+)", rep)
        type_value = re.search(r"type\s?=\s?(\w+)", rep)
        value = {
            "x": float(x_value.group(1)) if x_value else None,
            "y": float(y_value.group(1)) if y_value else None,
            "w": float(w_value.group(1)) if w_value else None,
            "h": float(h_value.group(1)) if h_value else None,
            "type": type_value.group(1) if type_value else None,
        }

        if not any(v is None for v in value.values()):
            x_min = max(bbox[0], value["x"])
            x_max = min(bbox[2], value["x"] + value["w"])
            y_min = max(bbox[1], value["y"])
            y_max = min(bbox[3], value["y"] + value["h"])

            if x_min > x_max or y_min > y_max:
                # No intersection
                return None

    role = None
    text = None

    for attr_name, ns_key in reserved_keys.items():
        if attr_name not in attr_names:
            continue

        if value and attr_name == "AXFrame":
            bb = value
            if not any(v is None for v in bb.values()):
                attribute_dict["{{{:}}}screencoord".format(_accessibility_ns_map["macos"]["cp"])] = \
                    "({:d}, {:d})".format(int(bb["x"]), int(bb["y"]))
                attribute_dict["{{{:}}}size".format(_accessibility_ns_map["macos"]["cp"])] = \
                    "({:d}, {:d})".format(int(bb["w"]), int(bb["h"]))
            continue

        error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, attr_name, None)

        full_attr_name = f"{{{_accessibility_ns_map['macos'][ns_key]}}}{attr_name}"

        if attr_name == "AXValue" and not text:
            text = str(attr_val)
            continue

        if attr_name == "AXRoleDescription":
            role = attr_val
            continue

        # Set the attribute_dict
        if not (isinstance(attr_val, ApplicationServices.AXUIElementRef)
                or isinstance(attr_val, (AppKit.NSArray, list))):
            if attr_val is not None:
                attribute_dict[full_attr_name] = str(attr_val)

    node_role_name = role.lower().replace(" ", "_") if role else "unknown_role"

    xml_node = lxml.etree.Element(
        node_role_name,
        attrib=attribute_dict,
        nsmap=_accessibility_ns_map["macos"]
    )

    if text is not None and len(text) > 0:
        xml_node.text = text

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached")
        return xml_node

    future_to_child = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for attr_name, ns_key in reserved_keys.items():
            if attr_name not in attr_names:
                continue

            error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, attr_name, None)
            if isinstance(attr_val, ApplicationServices.AXUIElementRef):
                future_to_child.append(executor.submit(_create_axui_node_macos, attr_val, nodes, depth + 1, bbox))

            elif isinstance(attr_val, (AppKit.NSArray, list)):
                for child in attr_val:
                    future_to_child.append(executor.submit(_create_axui_node_macos, child, nodes, depth + 1, bbox))

        try:
            for future in concurrent.futures.as_completed(future_to_child):
                result = future.result()
                if result is not None:
                    xml_node.append(result)
        except Exception as e:
            logger.error(f"Exception occurred: {e}")

    return xml_node


def get_accessibility_tree(max_depth: int = 20, include_dock: bool = False, focused_window_only: bool = False) -> Optional[str]:
    """
    获取系统的 Accessibility Tree
    完全基于 OSWorld 的实现
    
    平台支持：
    - macOS: 使用 ApplicationServices AX API
    - Linux: 待实现（当前返回 None）
    - Windows: 待实现（当前返回 None）
    
    Args:
        max_depth: 已弃用，使用全局 MAX_DEPTH (50) 以与 OSWorld 保持一致
        include_dock: 是否包含 Dock
        focused_window_only: 是否只获取有焦点的窗口（默认 False，获取所有前台窗口）
                            True: 只获取当前活跃窗口，避免被遮挡窗口的干扰
                            False: 获取所有前台窗口（OSWorld 默认行为）
        
    Returns:
        XML 格式的 accessibility tree 字符串，如果失败或平台不支持则返回 None
    """
    os_name = platform.system()
    
    if os_name == "Darwin" and MACOS_AVAILABLE:
        try:
            xml_node = lxml.etree.Element("desktop", nsmap=_accessibility_ns_map['macos'])
            
            # 获取前台窗口
            window_list = Quartz.CGWindowListCopyWindowInfo(
                (Quartz.kCGWindowListExcludeDesktopElements |
                 Quartz.kCGWindowListOptionOnScreenOnly),
                Quartz.kCGNullWindowID
            )
            
            if window_list is None:
                logger.warning("CGWindowListCopyWindowInfo 返回 None")
                return None
            
            foreground_windows = [
                win for win in window_list
                if win.get("kCGWindowLayer") == 0 and win.get("kCGWindowOwnerName") != "Window Server"
            ]
            
            logger.info(f"找到 {len(foreground_windows)} 个前台窗口")
            
            # 可选：包含 Dock
            dock_info = []
            if include_dock:
                all_windows = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionAll,
                    Quartz.kCGNullWindowID
                )
                if all_windows is not None:
                    dock_info = [
                        win for win in all_windows
                        if win.get("kCGWindowName", None) == "Dock"
                    ]
            
            # 如果只需要焦点窗口，过滤窗口列表
            windows_to_process = foreground_windows + dock_info
            if focused_window_only and foreground_windows:
                # 获取当前活跃应用
                from AppKit import NSWorkspace
                active_app = NSWorkspace.sharedWorkspace().activeApplication()
                active_app_name = active_app.get('NSApplicationName') if active_app else None
                
                if active_app_name:
                    # 只保留活跃应用的窗口
                    windows_to_process = [
                        win for win in foreground_windows 
                        if win.get("kCGWindowOwnerName") == active_app_name
                    ] + dock_info
                    logger.info(f"只处理活跃应用 {active_app_name} 的窗口")
            
            # 并行处理所有窗口
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_create_axui_node_macos, wnd, None, 0)
                    for wnd in windows_to_process
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        xml_tree = future.result(timeout=10)
                        if xml_tree is not None:
                            xml_node.append(xml_tree)
                    except Exception as e:
                        logger.warning(f"处理窗口时出错: {e}")
            
            return lxml.etree.tostring(xml_node, encoding="unicode", pretty_print=True)
            
        except Exception as e:
            logger.error(f"获取 macOS accessibility tree 失败: {e}")
            return None
            
    elif os_name == "Linux" and LINUX_AVAILABLE:
        # TODO: 实现 Linux AT-SPI 支持
        logger.warning("Linux accessibility tree 尚未实现")
        return None
        
    else:
        # TODO: 实现 Windows UI Automation 支持
        logger.warning(f"当前平台 {os_name} 的 accessibility tree 支持待实现")
        return None


def is_accessibility_available() -> bool:
    """检查当前平台是否支持 accessibility tree"""
    os_name = platform.system()
    if os_name == "Darwin":
        return MACOS_AVAILABLE
    elif os_name == "Linux":
        return LINUX_AVAILABLE
    return False

