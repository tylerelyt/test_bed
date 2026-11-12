#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI-Agent 服务 - 基于 OSWorld 架构的桌面自动化代理
参考：https://github.com/xlang-ai/OSWorld
"""

import os
import base64
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageGrab, ImageDraw, ImageFont
import subprocess
import logging

logger = logging.getLogger(__name__)

# 导入 accessibility tree 支持
try:
    from .accessibility_tree import get_accessibility_tree, is_accessibility_available
    ACCESSIBILITY_AVAILABLE = is_accessibility_available()
except ImportError:
    ACCESSIBILITY_AVAILABLE = False
    get_accessibility_tree = None


def annotate_screenshot_with_coordinates(screenshot_bytes: bytes, screen_width: int, screen_height: int, enable_grid: bool = True) -> bytes:
    """
    在截图上标注坐标信息，帮助 VLM 更好地理解坐标系统
    
    策略：先将截图缩放到 PyAutoGUI 的逻辑分辨率，再进行标注
    这样 VLM 看到的图像分辨率与坐标系统完全一致（1:1 对应）
    
    标注内容：
    - 四个角的坐标
    - 中心点坐标
    - 可选：网格辅助线和交点坐标（增强模式）
    - 醒目但不干扰的样式
    
    Args:
        screenshot_bytes: 原始截图的字节数据
        screen_width: 屏幕逻辑宽度（PyAutoGUI 可控区域宽度）
        screen_height: 屏幕逻辑高度（PyAutoGUI 可控区域高度）
        enable_grid: 是否启用网格辅助线（默认 True）
    
    Returns:
        标注后的截图字节数据（已缩放到逻辑分辨率）
    """
    try:
        # 1. 打开原始截图
        img = Image.open(BytesIO(screenshot_bytes))
        original_width, original_height = img.size
        
        # 2. 将截图缩放到 PyAutoGUI 的逻辑分辨率
        # 这样 VLM 看到的图像尺寸 = 坐标系统尺寸
        if (original_width, original_height) != (screen_width, screen_height):
            print(f"🔄 缩放截图: {original_width}x{original_height} -> {screen_width}x{screen_height}")
            img = img.resize((screen_width, screen_height), Image.LANCZOS)
        else:
            print(f"✓ 截图分辨率已匹配逻辑分辨率: {screen_width}x{screen_height}")
        
        # 3. 创建绘图对象
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # 4. 字体设置（固定大小，因为图像已经是逻辑分辨率）
        try:
            # macOS 系统字体
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            try:
                # Linux 字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                # 降级到默认字体
                font = ImageFont.load_default()
                font_small = font
        
        # 5. 标注点配置（直接使用逻辑坐标，无需转换）
        annotations = [
            # (x坐标, y坐标, 标签, 位置)
            (0, 0, "(0, 0)", "topleft"),
            (screen_width - 1, 0, f"({screen_width-1}, 0)", "topright"),
            (0, screen_height - 1, f"(0, {screen_height-1})", "bottomleft"),
            (screen_width - 1, screen_height - 1, f"({screen_width-1}, {screen_height-1})", "bottomright"),
            (screen_width // 2, screen_height // 2, f"({screen_width//2}, {screen_height//2})", "center"),
        ]
        
        # 6. 绘制标注
        for coord_x, coord_y, label, position in annotations:
            # 图像已缩放到逻辑分辨率，直接使用坐标
            x, y = coord_x, coord_y
            
            # 绘制圆点（醒目的红色）
            point_radius = 8
            draw.ellipse(
                [x - point_radius, y - point_radius,
                 x + point_radius, y + point_radius],
                fill=(255, 0, 0, 200),  # 半透明红色
                outline=(255, 255, 255, 255),  # 白色边框
                width=2
            )
            
            # 计算文本位置和背景
            # 使用 textbbox 获取文本边界（PIL 10.0.0+）
            try:
                bbox = draw.textbbox((0, 0), label, font=font_small)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # 降级方案
                text_width = len(label) * 10
                text_height = 16
            
            # 根据位置调整文本坐标
            padding = 5
            if position == "topleft":
                text_x = x + point_radius + padding
                text_y = y + point_radius + padding
            elif position == "topright":
                text_x = x - point_radius - padding - text_width
                text_y = y + point_radius + padding
            elif position == "bottomleft":
                text_x = x + point_radius + padding
                text_y = y - point_radius - padding - text_height
            elif position == "bottomright":
                text_x = x - point_radius - padding - text_width
                text_y = y - point_radius - padding - text_height
            elif position == "center":
                text_x = x + point_radius + padding
                text_y = y - text_height // 2
            
            # 绘制半透明背景
            bg_padding = 4
            draw.rectangle(
                [text_x - bg_padding, text_y - bg_padding,
                 text_x + text_width + bg_padding, text_y + text_height + bg_padding],
                fill=(0, 0, 0, 180)  # 半透明黑色背景
            )
            
            # 绘制文字（白色）
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font_small)
        
        # 7. 绘制网格辅助线（可选）
        if enable_grid:
            # 根据屏幕尺寸自适应网格间距
            if max(screen_width, screen_height) < 1000:
                grid_spacing = 100
            elif max(screen_width, screen_height) < 2000:
                grid_spacing = 200
            else:
                grid_spacing = 300
            
            print(f"📐 绘制网格辅助线: 间距 {grid_spacing}px")
            
            # 字体设置（小号字体用于网格坐标）
            try:
                font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                try:
                    font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    font_tiny = font_small
            
            # 绘制垂直网格线
            for x in range(0, screen_width + 1, grid_spacing):
                if x == 0 or x >= screen_width:  # 跳过边界，已有标注
                    continue
                # 半透明灰色虚线
                for y in range(0, screen_height, 10):
                    draw.line([(x, y), (x, min(y + 5, screen_height))], 
                             fill=(128, 128, 128, 80), width=1)
            
            # 绘制水平网格线
            for y in range(0, screen_height + 1, grid_spacing):
                if y == 0 or y >= screen_height:  # 跳过边界，已有标注
                    continue
                # 半透明灰色虚线
                for x in range(0, screen_width, 10):
                    draw.line([(x, y), (min(x + 5, screen_width), y)], 
                             fill=(128, 128, 128, 80), width=1)
            
            # 在网格交点标注坐标（只标注关键交点，避免过于密集）
            for x in range(grid_spacing, screen_width, grid_spacing):
                for y in range(grid_spacing, screen_height, grid_spacing):
                    # 跳过已经标注的中心点附近
                    if abs(x - screen_width // 2) < grid_spacing // 2 and abs(y - screen_height // 2) < grid_spacing // 2:
                        continue
                    
                    coord_label = f"({x},{y})"
                    
                    try:
                        bbox = draw.textbbox((0, 0), coord_label, font=font_tiny)
                        label_w = bbox[2] - bbox[0]
                        label_h = bbox[3] - bbox[1]
                    except:
                        label_w = len(coord_label) * 6
                        label_h = 12
                    
                    # 半透明背景
                    bg_pad = 2
                    draw.rectangle(
                        [x - label_w // 2 - bg_pad, y - label_h // 2 - bg_pad,
                         x + label_w // 2 + bg_pad, y + label_h // 2 + bg_pad],
                        fill=(0, 0, 0, 120)
                    )
                    
                    # 绘制坐标文字
                    draw.text((x - label_w // 2, y - label_h // 2), 
                             coord_label, fill=(200, 200, 200, 200), font=font_tiny)
        
        # 8. 在顶部中央添加分辨率信息
        resolution_label = f"Screen: {screen_width}×{screen_height}"
        if enable_grid:
            resolution_label += " [Grid Enabled]"
        
        try:
            bbox = draw.textbbox((0, 0), resolution_label, font=font)
            res_text_width = bbox[2] - bbox[0]
            res_text_height = bbox[3] - bbox[1]
        except:
            res_text_width = len(resolution_label) * 12
            res_text_height = 20
        
        res_x = (img.width - res_text_width) // 2
        res_y = 10
        
        # 背景
        bg_padding = 8
        draw.rectangle(
            [res_x - bg_padding, res_y - bg_padding,
             res_x + res_text_width + bg_padding, res_y + res_text_height + bg_padding],
            fill=(0, 0, 0, 200)
        )
        
        # 文字
        draw.text((res_x, res_y), resolution_label, fill=(0, 255, 0, 255), font=font)
        
        # 9. 转换回字节
        output_buffer = BytesIO()
        img.save(output_buffer, format='PNG')
        return output_buffer.getvalue()
        
    except Exception as e:
        print(f"⚠️  标注截图失败: {e}")
        # 如果标注失败，返回原始截图
        return screenshot_bytes


class SimpleDesktopEnv:
    """
    简化版桌面环境 - 基于 OSWorld DesktopEnv 的核心思想
    支持本地 macOS/Ubuntu 环境的屏幕捕获和动作执行
    """
    
    def __init__(
        self,
        provider_name: str = "local",
        os_type: str = "macOS",
        action_space: str = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = False,
        require_terminal: bool = False
    ):
        self.provider_name = provider_name
        self.os_type = os_type
        self.action_space = action_space
        self.screen_size = screen_size
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
        
        self.current_task = None
        self.step_count = 0
        self.max_steps = 50
        self.history = []
        
        # 初始化 PyAutoGUI（如果可用）
        self._init_controller()
    
    def _init_controller(self):
        """初始化控制器"""
        try:
            import pyautogui
            self.controller = pyautogui
            # 设置安全暂停时间
            self.controller.PAUSE = 0.5
            # 禁用 fail-safe（在实际应用中应该保留）
            # self.controller.FAILSAFE = True
            print(f"✅ PyAutoGUI 控制器初始化成功 - 屏幕尺寸: {self.controller.size()}")
        except ImportError:
            print("⚠️  PyAutoGUI 未安装，部分功能将不可用")
            self.controller = None
    
    def reset(self, task_config: Optional[Dict] = None) -> Dict:
        """
        重置环境并加载任务配置
        
        Args:
            task_config: 任务配置字典，包含 instruction, evaluator, config 等
        
        Returns:
            初始观察字典
        """
        self.current_task = task_config or {}
        self.step_count = 0
        self.history = []
        
        # 捕获初始屏幕截图
        obs = self._get_observation()
        obs['instruction'] = self.current_task.get('instruction', '未指定任务')
        
        print(f"🔄 环境已重置 - 任务: {obs['instruction']}")
        return obs
    
    def _get_observation(self) -> Dict:
        """
        获取当前观察
        完全基于 OSWorld 的 _get_obs() 实现
        
        Returns:
            包含截图、accessibility tree 和其他信息的观察字典
        """
        obs = {
            'screenshot': None,
            'screenshot_path': None,
            'accessibility_tree': None,
            'terminal': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # 捕获截图（与 OSWorld 一致）
        try:
            screenshot = ImageGrab.grab()
            
            # 保存为 bytes
            buffer = BytesIO()
            screenshot.save(buffer, format='PNG')
            obs['screenshot'] = buffer.getvalue()
            
            # 可选：保存到文件
            screenshot_dir = Path('data/gui_agent/screenshots')
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f'step_{self.step_count}_{int(time.time())}.png'
            screenshot.save(screenshot_path)
            obs['screenshot_path'] = str(screenshot_path)
            
        except Exception as e:
            print(f"⚠️  截图失败: {e}")
        
        # 获取 Accessibility Tree（如果启用，与 OSWorld 一致）
        # OSWorld: accessibility_tree = self.controller.get_accessibility_tree() if self.require_a11y_tree else None
        if self.require_a11y_tree and ACCESSIBILITY_AVAILABLE and get_accessibility_tree:
            try:
                # 使用 OSWorld 标准深度（MAX_DEPTH=50，在 accessibility_tree.py 中定义）
                obs['accessibility_tree'] = get_accessibility_tree(include_dock=False)
            except Exception as e:
                logger.debug(f"获取 accessibility tree 失败: {e}")
                obs['accessibility_tree'] = None
        else:
            obs['accessibility_tree'] = None
        
        # 终端输出（可选，当前未实现）
        # obs['terminal'] = self.controller.get_terminal_output() if self.require_terminal else None
        
        return obs
    
    def step(self, action: str, pause: float = 2.0) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个动作并返回新的观察
        完全基于 OSWorld 的 step() 实现
        
        Args:
            action: 要执行的动作字符串（如 "pyautogui.click(100, 100)"）
            pause: 执行后等待时间（默认 2.0 秒，与 OSWorld 一致）
        
        Returns:
            (observation, reward, done, info) 元组
        """
        self.step_count += 1
        done = False
        reward = 0.0
        info = {'action': action, 'step': self.step_count}
        
        # 处理控制符
        if action in ["DONE", "FAIL", "WAIT"]:
            if action == "DONE":
                done = True
                reward = 1.0
                print(f"✅ 任务完成 - 步骤: {self.step_count}")
            elif action == "FAIL":
                done = True
                reward = 0.0
                print(f"❌ 任务失败 - 步骤: {self.step_count}")
            elif action == "WAIT":
                print(f"⏸️  等待 - 步骤: {self.step_count}")
            
            obs = self._get_observation()
            return obs, reward, done, info
        
        # 执行 PyAutoGUI 动作
        try:
            execution_result = self._execute_action(action)
            info['execution_result'] = execution_result
            
            # 等待界面响应
            time.sleep(pause)
            
        except Exception as e:
            print(f"⚠️  动作执行失败: {e}")
            info['error'] = str(e)
        
        # 获取新观察
        obs = self._get_observation()
        
        # 检查是否达到最大步数
        if self.step_count >= self.max_steps:
            done = True
            print(f"⏱️  达到最大步数 {self.max_steps}")
        
        # 记录历史
        self.history.append({
            'step': self.step_count,
            'action': action,
            'screenshot_path': obs.get('screenshot_path'),
            'info': info
        })
        
        return obs, reward, done, info
    
    def _execute_action(self, action: str) -> Dict:
        """
        执行单个 PyAutoGUI 动作
        
        Args:
            action: PyAutoGUI 命令字符串
        
        Returns:
            执行结果字典
        """
        if not self.controller:
            return {'status': 'error', 'message': 'PyAutoGUI 不可用'}
        
        # 安全检查：只允许 pyautogui 命令
        if not action.strip().startswith('pyautogui.'):
            return {'status': 'error', 'message': f'不安全的命令: {action}'}
        
        try:
            # 在安全的命名空间中执行
            namespace = {'pyautogui': self.controller}
            exec(action, namespace)
            
            return {'status': 'success', 'action': action}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'action': action}
    
    def get_history(self) -> List[Dict]:
        """获取执行历史"""
        return self.history
    
    def close(self):
        """清理资源"""
        print(f"🔒 环境已关闭 - 总步数: {self.step_count}")


class SimplePromptAgent:
    """
    简化版 Prompt Agent - 基于 OSWorld PromptAgent 的核心思想
    使用视觉语言模型进行任务推理和动作规划
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 3,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,
        use_trajectory: bool = True
    ):
        self.model = model
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.use_trajectory = use_trajectory
        
        # API 配置
        # 如果是 Qwen 模型，优先使用 DASHSCOPE_API_KEY
        if model.startswith("qwen") or model.startswith("qvq"):
            self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        else:
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # 历史轨迹
        self.trajectory = []
        
        # 初始化 OpenAI 客户端
        self._init_client()
    
    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"✅ OpenAI 客户端初始化成功 - 模型: {self.model}")
        except ImportError:
            print("⚠️  OpenAI 库未安装")
            self.client = None
        except Exception as e:
            print(f"⚠️  OpenAI 客户端初始化失败: {e}")
            self.client = None
    
    def predict(self, instruction: str, observation: Dict) -> Tuple[str, List[str]]:
        """
        基于当前观察和任务指令，预测下一步动作
        
        Args:
            instruction: 任务指令
            observation: 当前观察（包含截图）
        
        Returns:
            (思考过程, 动作列表) 元组
        """
        if not self.client:
            # 模拟模式：返回示例动作
            return self._mock_prediction(instruction, observation)
        
        try:
            # 提取屏幕分辨率 - 使用 PyAutoGUI 逻辑分辨率而不是截图物理分辨率
            # 重要：在 Retina/HiDPI 显示器上，截图的物理像素是逻辑像素的 2 倍
            # 但 PyAutoGUI 的坐标系统使用的是逻辑像素，所以必须使用 pyautogui.size()
            screen_size = None
            try:
                import pyautogui
                logical_size = pyautogui.size()
                screen_size = (logical_size.width, logical_size.height)
                print(f"📐 使用 PyAutoGUI 逻辑屏幕尺寸: {screen_size[0]}x{screen_size[1]}")
            except Exception as e:
                print(f"⚠️  无法获取 PyAutoGUI 屏幕尺寸: {e}")
                # 降级方案：从截图中提取（可能在 HiDPI 屏幕上不准确）
                if 'screenshot' in observation:
                    try:
                        img = Image.open(BytesIO(observation['screenshot']))
                        screen_size = (img.width, img.height)
                        print(f"⚠️  降级使用截图分辨率: {screen_size[0]}x{screen_size[1]} (可能在 Retina 屏幕上不准确)")
                    except Exception as e2:
                        print(f"⚠️  也无法提取截图分辨率: {e2}")
            
            # 在截图上标注坐标信息（帮助 VLM 理解坐标系统）
            annotated_screenshot = observation['screenshot']
            if screen_size:
                print(f"🎯 在截图上标注坐标基准点...")
                annotated_screenshot = annotate_screenshot_with_coordinates(
                    observation['screenshot'],
                    screen_size[0],
                    screen_size[1]
                )
                
                # 保存标注后的截图到文件（覆盖原始截图）
                if observation.get('screenshot_path'):
                    try:
                        annotated_img = Image.open(BytesIO(annotated_screenshot))
                        annotated_img.save(observation['screenshot_path'])
                        print(f"💾 标注后的截图已保存: {observation['screenshot_path']}")
                    except Exception as e:
                        print(f"⚠️  保存标注截图失败: {e}")
            
            # 编码截图为 base64（使用标注后的截图）
            screenshot_b64 = self._encode_screenshot(annotated_screenshot)
            
            # 保存当前截图的 base64（用于后续作为历史）
            observation['screenshot_b64'] = screenshot_b64
            
            # 获取 Accessibility Tree（如果 observation 中没有，则获取新的）
            # 与 OSWorld 一致：如果 step() 已经获取了新的 observation，这里直接使用
            accessibility_tree = observation.get('accessibility_tree')
            
            # 如果 observation 中没有 accessibility_tree，且系统支持，则尝试获取
            if accessibility_tree is None:
                if ACCESSIBILITY_AVAILABLE and get_accessibility_tree:
                    try:
                        print(f"🌲 获取 Accessibility Tree (使用 OSWorld 标准深度 MAX_DEPTH=50)...")
                        accessibility_tree = get_accessibility_tree(include_dock=False)
                        if accessibility_tree:
                            print(f"✅ Accessibility Tree 已获取 ({len(accessibility_tree)} 字符)")
                        else:
                            print(f"⚠️  Accessibility Tree 为空")
                    except Exception as e:
                        print(f"⚠️  获取 Accessibility Tree 失败: {e}")
                        accessibility_tree = None
                else:
                    # 系统不支持或模块未导入
                    if not ACCESSIBILITY_AVAILABLE:
                        print(f"ℹ️  Accessibility Tree 不可用 (当前平台不支持，仅使用截图模式)")
                    else:
                        print(f"ℹ️  Accessibility Tree 不可用 (模块未导入，仅使用截图模式)")
            elif accessibility_tree:
                # observation 中已有 accessibility_tree
                print(f"ℹ️  使用 observation 中的 Accessibility Tree ({len(accessibility_tree)} 字符)")
            
            # 构造消息
            messages = self._build_messages(instruction, screenshot_b64, screen_size, accessibility_tree)
            
            # 判断是否使用流式输出（Qwen 思考模式推荐使用流式）
            use_stream = (self.model.startswith("qwen") or self.model.startswith("qvq")) and self.enable_thinking
            
            if use_stream:
                # 流式输出模式（用于获取思考过程）
                reasoning_content = ""
                answer_content = ""
                is_answering = False
                
                # 准备参数
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 2000,
                    "stream": True
                }
                
                # 如果是 Qwen 模型且启用思考，添加 enable_thinking 参数
                if self.model.startswith("qwen") and self.enable_thinking:
                    # 注意：OpenAI 兼容 API 可能不支持直接传递额外参数
                    # 这里我们通过 extra_headers 或其他方式传递
                    # 如果 API 不支持，则使用非流式模式
                    pass
                
                # 调用流式 API
                stream = self.client.chat.completions.create(**params)
                
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
                
                for chunk in stream:
                    from .workbench_telemetry import record_usage
                    record_usage(chunk, 'configured')
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # 处理思考过程（Qwen 特有）
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        # 开始回复
                        if delta.content and not is_answering:
                            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                        
                        # 打印回复过程
                        if delta.content:
                            print(delta.content, end='', flush=True)
                            answer_content += delta.content
                
                # 合并思考过程和回复
                if reasoning_content:
                    response_text = f"【思考过程】\n{reasoning_content}\n\n【最终回复】\n{answer_content}"
                else:
                    response_text = answer_content
                
                # 打印模型的完整响应（流式模式下）
                print("\n" + "=" * 60)
                print("📥 模型返回的完整 Response")
                print("=" * 60)
                print(response_text)
                print("=" * 60 + "\n")
                
            else:
                # 非流式输出模式
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 2000
                }
                
                # 调用 VL 模型
                response = self.client.chat.completions.create(**params)
                
                # 提取响应
                from .workbench_telemetry import record_usage
                record_usage(response, 'configured')
                response_text = response.choices[0].message.content
                
                # 打印模型的完整响应
                print("\n" + "=" * 60)
                print("📥 模型返回的 Response")
                print("=" * 60)
                print(response_text)
                print("=" * 60 + "\n")
            
            # 解析动作
            actions = self._parse_actions(response_text)
            
            # 更新轨迹
            self._update_trajectory(instruction, observation, response_text, actions)
            
            return response_text, actions
            
        except Exception as e:
            print(f"⚠️  模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return f"错误: {str(e)}", ["FAIL"]
    
    def _encode_screenshot(self, screenshot_bytes: bytes) -> str:
        """将截图编码为 base64"""
        if screenshot_bytes is None:
            return ""
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    
    def _build_messages(self, instruction: str, screenshot_b64: str, screen_size: tuple = None, accessibility_tree: Optional[str] = None) -> List[Dict]:
        """构造发送给模型的消息 - 基于 OSWorld 官方 Prompt"""
        # OSWorld 官方 System Prompt
        # 根据是否有 accessibility tree 选择不同的模式：
        # - 有 a11y tree: SYS_PROMPT_IN_BOTH_OUT_CODE (screenshot + a11y tree)
        # - 仅截图: SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
        # 参考: https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/prompts.py
        
        # 动态生成坐标系统增强说明
        coordinate_enhancement = ""
        if screen_size:
            width, height = screen_size
            coordinate_enhancement = f"""
IMPORTANT - Coordinate System Enhancement:
The screenshot has been annotated with 5 red coordinate reference points to help you locate elements accurately:
- Top-left corner: (0, 0)
- Top-right corner: ({width-1}, 0)
- Bottom-left corner: (0, {height-1})
- Bottom-right corner: ({width-1}, {height-1})
- Center point: ({width//2}, {height//2})
The screen resolution {width}x{height} is displayed at the top center of the screenshot. Please use these reference points to accurately estimate the coordinates of target elements.
"""
        
        # 根据是否有 accessibility tree 构造不同的观察描述
        if accessibility_tree:
            observation_description = """For each step, you will get an observation of the desktop by 1) a screenshot; and 2) accessibility tree, which is based on AT-SPI library (Linux) or AX API (macOS). And you will predict the action of the computer based on the screenshot and accessibility tree.

**CRITICAL: When accessibility tree is provided, you MUST:**
1. First locate the target UI element in the accessibility tree by its name, role, or description
2. Extract the EXACT coordinates from the element's cp:screencoord attribute (format: "(x, y)")
3. Use these EXACT coordinates in your pyautogui commands
4. DO NOT estimate coordinates from the screenshot - always use accessibility tree coordinates for precision

Example: If you see <button name="Later" cp:screencoord="(1408, 752)" cp:size="(60, 20)">, you MUST use pyautogui.click(1408, 752) - not an estimated position."""
        else:
            observation_description = "For each step, you will get an observation of an image, which is the screenshot of the computer screen and you will predict the action of the computer based on the image."
        
        system_prompt = f"""You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
{observation_description}

You are required to use `pyautogui` to perform the action grounded to the observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DONOT USE `pyautogui.screenshot()` to make screenshot.
Return one line or multiple lines of python code to perform the action each time, be time efficient. When predicting multiple lines of code, make some small sleep like `time.sleep(0.5);` interval so that the machine could take; Each time you need to predict a complete code, no variables or function can be shared from history
You need to to specify the coordinates of by yourself based on your observation of current observation, but you should be careful to ensure that the coordinates are correct.
You ONLY need to return the code inside a code block, like this:
```python
# your code here
```
Specially, it is also allowed to return the following special code:
When you think you have to wait for some time, return ```WAIT```;
When you think the task can not be done, return ```FAIL```, don't easily say ```FAIL```, try your best to do the task;
When you think the task is done, return ```DONE```.
{coordinate_enhancement}
First give the current screenshot and previous things we did a short reflection, then RETURN ME THE CODE OR SPECIAL CODE I ASKED FOR. NEVER EVER RETURN ME ANYTHING ELSE.""".strip()
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 构建用户消息内容 - OSWorld 风格
        user_content = []
        
        # OSWorld 风格：简洁的任务描述
        task_text = f"Task: {instruction}\n\n"
        
        # 屏幕分辨率信息（简洁格式）
        if screen_size:
            width, height = screen_size
            task_text += f"Screen resolution: {width}x{height}\n"
            task_text += f"Coordinate range: x=[0, {width-1}], y=[0, {height-1}]\n\n"
        
        # 添加历史轨迹（如果启用）
        if self.use_trajectory and self.trajectory:
            recent_history = self.trajectory[-self.max_trajectory_length:]
            task_text += f"Previous actions (last {len(recent_history)} steps):\n"
            
            for i, t in enumerate(recent_history):
                task_text += f"Step {i+1}: {t['actions']}\n"
                
                # 添加历史截图
                if t.get('screenshot_b64'):
                    user_content.append({
                        "type": "text",
                        "text": f"--- Screenshot of Step {i+1} ---"
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{t['screenshot_b64']}"}
                    })
            
            task_text += "\n--- Current Screenshot ---\n"
        
        # 添加任务文本
        user_content.insert(0, {"type": "text", "text": task_text})
        
        # 添加当前截图
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
        })
        
        # 添加 Accessibility Tree（如果可用）
        if accessibility_tree:
            a11y_text = "\n--- Accessibility Tree ---\n"
            a11y_text += "The following is the accessibility tree of current desktop, which contains UI elements with their properties (role, coordinates, states, etc.):\n\n"
            a11y_text += """⚠️ CRITICAL INSTRUCTION - How to Use Accessibility Tree Coordinates:
1. ALWAYS search for your target element in the accessibility tree by matching name, role, or description
2. When you find the element, extract its cp:screencoord="(x, y)" attribute
3. Use these EXACT numbers in pyautogui.click(x, y) - DO NOT modify or estimate them
4. The accessibility tree provides precise coordinates - screenshot coordinates are LESS accurate

⚠️ WINDOW OCCLUSION WARNING:
- The accessibility tree includes ALL foreground windows, even if they are visually occluded by other windows
- Check st:AXFocused attribute: elements with st:AXFocused="True" are in the top-most window
- If multiple windows overlap, prioritize elements from the focused window or compare with screenshot
- An element may be in the tree but not clickable if covered by another window

Example workflow:
- Task: "Click the Later button"
- Find in tree: <button name="Later" cp:screencoord="(1408, 752)" st:AXFocused="False">
- Check screenshot: Is this button visible or covered?
- Your code: pyautogui.click(1408, 752)  # Use exact coordinates from tree

If element not found in tree, explain why and use screenshot estimation as fallback.

"""
            # OSWorld 方式：传递完整的 accessibility tree，不做长度截断
            # 只通过深度限制（MAX_DEPTH=50）控制树的大小
            # 注意：这可能会消耗较多 tokens，但能保证完整性
            a11y_text += accessibility_tree
            a11y_text += f"\n\n(Total tree size: {len(accessibility_tree)} characters)"
            user_content.append({
                "type": "text",
                "text": a11y_text
            })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 打印发送给模型的 prompt（用于调试）
        print("\n" + "=" * 60)
        print("📤 发送给模型的 Prompt")
        print("=" * 60)
        print(f"System Prompt:\n{system_prompt}\n")
        
        # 检查是否包含 accessibility tree
        has_a11y_tree = accessibility_tree is not None and len(accessibility_tree) > 0
        print(f"🌲 Accessibility Tree 状态: {'✅ 已包含' if has_a11y_tree else '❌ 未包含'}")
        if has_a11y_tree:
            print(f"   - 长度: {len(accessibility_tree)} 字符")
            print(f"   - 是否截断: {'是' if len(accessibility_tree) > 8000 else '否'}")
        
        print(f"\nUser Content (文本部分):")
        for item in user_content:
            if item.get("type") == "text":
                text = item.get("text", "")
                # 检查是否包含 accessibility tree
                if "Accessibility Tree" in text:
                    print(f"  ✅ [Accessibility Tree 文本块] ({len(text)} 字符)")
                    # 打印前 500 字符和后 200 字符以便查看
                    if len(text) > 700:
                        print(f"    前 500 字符: {text[:500]}...")
                        print(f"    ... (省略中间部分) ...")
                        print(f"    后 200 字符: ...{text[-200:]}")
                    else:
                        print(f"    完整内容预览: {text[:1000]}...")
                else:
                    print(f"  {text[:200]}..." if len(text) > 200 else f"  {text}")
            elif item.get("type") == "image_url":
                print(f"  📷 [截图图像]")
        print("=" * 60 + "\n")
        
        return messages
    
    def _parse_actions(self, response_text: str) -> List[str]:
        """
        从模型响应中解析动作列表
        
        Args:
            response_text: 模型的文本响应
        
        Returns:
            动作字符串列表
        """
        actions = []
        
        # 提取代码块
        code_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
        
        if code_blocks:
            for block in code_blocks:
                lines = block.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue
                    # 只接受 pyautogui 命令或控制符
                    if line.startswith('pyautogui.') or line in ['DONE', 'FAIL', 'WAIT']:
                        actions.append(line)
        
        # 如果没有找到代码块，尝试直接提取
        if not actions:
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('pyautogui.') or line in ['DONE', 'FAIL', 'WAIT']:
                    actions.append(line)
        
        # 如果还是没有动作，返回 WAIT
        if not actions:
            actions = ['WAIT']
        
        # 限制每轮最多 5 个动作
        return actions[:5]
    
    def _update_trajectory(self, instruction: str, observation: Dict, response: str, actions: List[str]):
        """更新历史轨迹（仅在 use_trajectory=True 时）"""
        if not self.use_trajectory:
            return
        
        self.trajectory.append({
            'instruction': instruction,
            'screenshot_path': observation.get('screenshot_path'),
            'screenshot_b64': observation.get('screenshot_b64'),  # 保存截图的 base64 编码
            'response': response,
            'actions': actions,
            'timestamp': datetime.now().isoformat()
        })
        
        # 只保留最近的轨迹
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory = self.trajectory[-self.max_trajectory_length:]
    
    def _mock_prediction(self, instruction: str, observation: Dict) -> Tuple[str, List[str]]:
        """模拟模式：返回示例动作（用于演示）"""
        response = f"收到任务: {instruction}\n\n由于 OpenAI API 未配置，这是一个模拟响应。"
        actions = ["WAIT"]
        return response, actions
    
    def reset(self):
        """重置代理状态"""
        self.trajectory = []
        print("🔄 代理已重置")
    
    def get_trajectory(self) -> List[Dict]:
        """获取执行轨迹"""
        return self.trajectory


class GUIAgentService:
    """
    GUI-Agent 服务 - 整合环境和代理
    """
    
    def __init__(self):
        self.env = None
        self.agent = None
        self.is_running = False
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0
        }
    
    def initialize(
        self,
        provider_name: str = "local",
        os_type: str = "macOS",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        require_a11y_tree: bool = False
    ) -> Dict:
        """
        初始化环境和代理
        
        Args:
            provider_name: 提供者名称（"local" 或 "vm"）
            os_type: 操作系统类型
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            require_a11y_tree: 是否启用 accessibility tree（默认 False，与 OSWorld 的 True 不同）
                             注意：OSWorld 默认 True，但我们默认 False 以避免权限问题
        """
        try:
            # 初始化环境
            self.env = SimpleDesktopEnv(
                provider_name=provider_name,
                os_type=os_type,
                action_space="pyautogui",
                require_a11y_tree=require_a11y_tree
            )
            
            # 初始化代理
            self.agent = SimplePromptAgent(
                model=model,
                action_space="pyautogui",
                api_key=api_key,
                base_url=base_url
            )
            
            return {
                'status': 'success',
                'message': f'环境和代理初始化成功 - 平台: {provider_name}, 模型: {model}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'初始化失败: {str(e)}'
            }
    
    def run_task(
        self,
        instruction: str,
        max_steps: int = 15,
        sleep_after_execution: float = 2.0
    ) -> Dict:
        """
        运行一个任务
        
        Args:
            instruction: 任务指令
            max_steps: 最大步数
            sleep_after_execution: 每步后等待时间（默认 2.0 秒，与 OSWorld 一致）
        
        Returns:
            任务执行结果
        """
        if not self.env or not self.agent:
            return {
                'status': 'error',
                'message': '环境或代理未初始化'
            }
        
        try:
            self.is_running = True
            self.stats['total_tasks'] += 1
            
            # 重置环境和代理
            task_config = {'instruction': instruction}
            obs = self.env.reset(task_config)
            self.agent.reset()
            
            # 设置最大步数
            self.env.max_steps = max_steps
            
            results = {
                'instruction': instruction,
                'steps': [],
                'final_status': 'running',
                'total_steps': 0
            }
            
            # 主循环
            done = False
            while not done and self.env.step_count < max_steps:
                # 模型推理
                response, actions = self.agent.predict(instruction, obs)
                
                step_result = {
                    'step': self.env.step_count + 1,
                    'response': response,
                    'actions': actions,
                    'action_results': []
                }
                
                # 执行动作
                for action in actions:
                    obs, reward, done, info = self.env.step(action, pause=sleep_after_execution)
                    
                    step_result['action_results'].append({
                        'action': action,
                        'reward': reward,
                        'done': done,
                        'info': info,
                        'screenshot_path': obs.get('screenshot_path')
                    })
                    
                    if done:
                        if reward > 0:
                            results['final_status'] = 'completed'
                            self.stats['successful_tasks'] += 1
                        else:
                            results['final_status'] = 'failed'
                            self.stats['failed_tasks'] += 1
                        break
                
                results['steps'].append(step_result)
                results['total_steps'] = self.env.step_count
                
                if done:
                    break
            
            # 如果达到最大步数
            if not done and self.env.step_count >= max_steps:
                results['final_status'] = 'max_steps_reached'
                self.stats['failed_tasks'] += 1
            
            self.is_running = False
            
            return {
                'status': 'success',
                'results': results,
                'history': self.env.get_history(),
                'trajectory': self.agent.get_trajectory()
            }
            
        except Exception as e:
            self.is_running = False
            self.stats['failed_tasks'] += 1
            return {
                'status': 'error',
                'message': f'任务执行失败: {str(e)}'
            }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_tasks': self.stats['total_tasks'],
            'successful_tasks': self.stats['successful_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'is_running': self.is_running,
            'has_pyautogui': self.env.controller is not None if self.env else False,
            'has_openai': self.agent.client is not None if self.agent else False
        }
    
    def stop_task(self):
        """停止当前任务"""
        self.is_running = False
        if self.env:
            self.env.close()


# 全局服务实例
gui_agent_service = GUIAgentService()

