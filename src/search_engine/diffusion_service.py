#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像生成服务客户端 - 调用独立的图像生成服务
通过 HTTP API 与独立运行的 Stable Diffusion XL 服务通信

独立服务位于: image_generation_service.py
服务地址: http://localhost:5001
"""

import os
import time
import requests
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class DiffusionService:
    """扩散模型图像生成服务客户端"""
    
    def __init__(self, service_url: str = "http://localhost:5001"):
        """
        初始化扩散模型服务客户端
        
        Args:
            service_url: 独立图像生成服务的 URL
        """
        self.service_url = service_url
        self.model_name = "Stable Diffusion v1.5"
        self.generation_history: List[Dict[str, Any]] = []
        
        print(f"🎨 图像生成服务客户端初始化完成 (服务地址: {service_url})")
    
    def _check_service(self) -> Tuple[bool, str]:
        """检查独立服务是否运行"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data['model_loaded']:
                    return True, f"✅ 服务正常，模型已加载: {data['model_name']}"
                else:
                    return True, "⚠️ 服务正常，但模型未加载"
            else:
                return False, "❌ 服务响应异常"
        except requests.exceptions.ConnectionError:
            return False, f"❌ 无法连接到服务 ({self.service_url})，请先启动独立服务"
        except Exception as e:
            return False, f"❌ 服务检查失败: {str(e)}"
    
    def load_model(self) -> Tuple[bool, str]:
        """
        加载模型（调用独立服务）
        
        Returns:
            (成功标志, 消息)
        """
        try:
            # 检查服务是否运行
            service_ok, service_msg = self._check_service()
            
            if not service_ok:
                return False, (
                    f"{service_msg}\n\n"
                    "💡 请使用 ./quick_start.sh 启动系统（会自动启动图像服务）\n\n"
                    "或手动启动:\n"
                    "1. conda activate testbed-image\n"
                    "2. python image_generation_service.py"
                )
            
            # 调用加载模型 API
            print("📥 正在加载 Stable Diffusion v1.5 模型...")
            response = requests.post(f"{self.service_url}/load_model", timeout=300)
            if response.status_code == 200:
                data = response.json()
                return data['success'], data['message']
            else:
                return False, f"❌ 服务请求失败: {response.status_code}"
                
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: int = 1
    ) -> Dict[str, Any]:
        """
        生成图像（调用独立服务）
        
        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            width: 图像宽度
            height: 图像高度
            seed: 随机种子（-1表示随机）
            num_images: 生成图像数量
            
        Returns:
            生成结果字典
        """
        try:
            # 检查服务
            service_ok, service_msg = self._check_service()
            if not service_ok:
                return {
                    'success': False,
                    'message': service_msg,
                    'images': [],
                    'paths': []
                }
            
            print(f"🎨 开始生成图像...")
            print(f"  提示词: {prompt[:50]}...")
            
            # 调用生成 API
            response = requests.post(
                f"{self.service_url}/generate",
                json={
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'width': width,
                    'height': height,
                    'seed': seed,
                    'num_images': num_images
                },
                timeout=600  # 10分钟超时
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # 下载生成的图像
                images = []
                for path in data['paths']:
                    filename = Path(path).name
                    img_response = requests.get(f"{self.service_url}/image/{filename}")
                    if img_response.status_code == 200:
                        from io import BytesIO
                        images.append(Image.open(BytesIO(img_response.content)))
                
                # 记录历史
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "model": self.model_name,
                    "seed": data['metadata']['seed'],
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "size": f"{width}x{height}",
                    "num_images": num_images,
                    "generation_time": data['generation_time'],
                    "paths": data['paths']
                }
                self.generation_history.append(history_entry)
                
                print(f"✅ {data['message']}")
                
                return {
                    'success': True,
                    'message': data['message'],
                    'images': images,
                    'paths': data['paths'],
                    'metadata': data['metadata'],
                    'generation_time': data['generation_time']
                }
            else:
                error_data = response.json()
                return {
                    'success': False,
                    'message': error_data.get('message', '生成失败'),
                    'images': [],
                    'paths': []
                }
                
        except Exception as e:
            import traceback
            error_msg = f"❌ 图像生成失败: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            return {
                'success': False,
                'message': error_msg,
                'images': [],
                'paths': [],
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息（调用独立服务）"""
        try:
            service_ok, service_msg = self._check_service()
            if not service_ok:
                return {
                    'loaded': False,
                    'message': service_msg
                }
            
            response = requests.get(f"{self.service_url}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return {
                    'loaded': data['model_loaded'],
                    'model_name': data.get('model_name', 'Unknown'),
                    'message': '服务正常' if data['model_loaded'] else '服务运行中，但模型未加载'
                }
            else:
                return {'loaded': False, 'message': '无法获取模型信息'}
        except Exception as e:
            return {'loaded': False, 'message': f'获取模型信息失败: {str(e)}'}
    
    def get_generation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取生成历史"""
        return self.generation_history[-limit:]
    
    def clear_history(self):
        """清空生成历史"""
        self.generation_history = []
        print("✅ 生成历史已清空")

