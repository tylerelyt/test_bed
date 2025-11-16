#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的图像生成服务 - Stable Diffusion XL
可以在单独的 conda 环境中运行，通过 HTTP API 与 Testbed 通信

使用方法：
1. 创建独立环境：conda create -n testbed-image python=3.10 -y
2. 激活环境：conda activate testbed-image
3. 安装依赖：pip install diffusers transformers accelerate safetensors torch pillow flask flask-cors
4. 启动服务：python image_generation_service.py
5. 服务将在 http://localhost:5001 运行
"""

import os
import time
import torch
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import hashlib

app = Flask(__name__)
CORS(app)

# 全局变量
pipe = None
model_loaded = False
model_name = "Stable Diffusion v1.5"
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = Path("models/generated_images")
output_dir.mkdir(parents=True, exist_ok=True)
generation_history = []

def load_model():
    """加载 SD 1.5 模型"""
    global pipe, model_loaded
    
    if model_loaded:
        return True, "模型已加载"
    
    try:
        from diffusers import StableDiffusionPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 正在加载模型: {model_name} (设备: {device})")
        print(f"   首次使用需要下载约4GB，请耐心等待...")
        
        start_time = time.time()
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        pipe = pipe.to(device)
        
        # 启用内存优化
        if device == "cuda":
            try:
                pipe.enable_attention_slicing()
                print("  ✓ 启用 Attention Slicing")
            except:
                pass
            
            try:
                pipe.enable_vae_slicing()
                print("  ✓ 启用 VAE Slicing")
            except:
                pass
        
        load_time = time.time() - start_time
        model_loaded = True
        
        message = f"✅ 模型加载成功 (耗时: {load_time:.1f}秒)"
        print(message)
        return True, message
        
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}"
        print(error_msg)
        return False, error_msg

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_name': model_name if model_loaded else None
    })

@app.route('/load_model', methods=['POST'])
def api_load_model():
    """加载模型 API"""
    success, message = load_model()
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/generate', methods=['POST'])
def generate():
    """生成图像 API"""
    global pipe, model_loaded
    
    if not model_loaded:
        return jsonify({
            'success': False,
            'message': '模型未加载，请先调用 /load_model'
        }), 400
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        num_inference_steps = int(data.get('num_inference_steps', 50))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        seed = int(data.get('seed', -1))
        num_images = int(data.get('num_images', 1))
        
        if seed == -1:
            seed = int(time.time() * 1000) % (2**32)
        
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        print(f"🎨 生成图像: {prompt[:50]}...")
        start_time = time.time()
        
        # 生成图像
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images
        )
        
        images = output.images
        generation_time = time.time() - start_time
        
        # 保存图像
        saved_paths = []
        for i, image in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"gen_{timestamp}_{prompt_hash}_{seed}_{i}.png"
            filepath = output_dir / filename
            
            # 保存图像（带元数据）
            from PIL import PngImagePlugin
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("prompt", prompt)
            pnginfo.add_text("negative_prompt", negative_prompt)
            pnginfo.add_text("steps", str(num_inference_steps))
            pnginfo.add_text("guidance_scale", str(guidance_scale))
            pnginfo.add_text("seed", str(seed))
            pnginfo.add_text("size", f"{width}x{height}")
            
            image.save(filepath, pnginfo=pnginfo)
            saved_paths.append(str(filepath))
        
        # 记录历史
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "seed": seed,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "size": f"{width}x{height}",
            "num_images": num_images,
            "generation_time": generation_time,
            "paths": saved_paths
        }
        generation_history.append(history_entry)
        
        print(f"✅ 生成完成 (耗时: {generation_time:.2f}秒)")
        
        return jsonify({
            'success': True,
            'message': f'生成了 {num_images} 张图像',
            'paths': saved_paths,
            'generation_time': generation_time,
            'metadata': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'size': f"{width}x{height}"
            }
        })
        
    except Exception as e:
        import traceback
        error_msg = f"生成失败: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': error_msg
        }), 500

@app.route('/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    """提供图像文件"""
    filepath = output_dir / filename
    if filepath.exists():
        return send_file(filepath, mimetype='image/png')
    else:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/history', methods=['GET'])
def get_history():
    """获取生成历史"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify({
        'history': generation_history[-limit:]
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🎨 Stable Diffusion XL 图像生成服务")
    print("=" * 60)
    print(f"📦 模型: {model_name}")
    print(f"🌐 服务地址: http://localhost:5001")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    print("\n💡 提示: 首次使用前请调用 POST /load_model 加载模型")
    print("   然后使用 POST /generate 生成图像\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)

