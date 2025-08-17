#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片索引服务 - 基于CLIP的图片检索系统
支持图片存储、图搜图、文搜图功能
"""

import os
import json
import hashlib
import time
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from pathlib import Path

class ImageService:
    """图片索引服务 - 基于CLIP的图片检索"""
    
    def __init__(self, storage_dir: str = "models/images"):
        """
        初始化图片服务
        
        Args:
            storage_dir: 图片存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 索引文件路径
        self.index_file = self.storage_dir / "image_index.json"
        self.embeddings_file = self.storage_dir / "image_embeddings.npy"
        
        # 初始化CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._init_clip_model()
        
        # 图片索引和嵌入
        self.image_index: Dict[str, Dict] = {}
        self.image_embeddings: Optional[np.ndarray] = None
        self.image_ids: List[str] = []
        
        # 加载现有索引
        self._load_index()
    
    def _init_clip_model(self):
        """初始化CLIP模型"""
        try:
            print(f"🤖 初始化CLIP模型 (设备: {self.device})...")
            model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print("✅ CLIP模型加载成功")
        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            raise e
    
    def _load_index(self):
        """加载图片索引"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.image_index = data.get('images', {})
                    self.image_ids = list(self.image_index.keys())
                    print(f"📸 加载图片索引: {len(self.image_index)} 张图片")
            
            if self.embeddings_file.exists() and len(self.image_ids) > 0:
                self.image_embeddings = np.load(self.embeddings_file)
                print(f"🔢 加载图片嵌入: {self.image_embeddings.shape}")
            
        except Exception as e:
            print(f"⚠️ 加载索引失败: {e}")
            self.image_index = {}
            self.image_embeddings = None
            self.image_ids = []
    
    def _save_index(self):
        """保存图片索引"""
        try:
            # 保存图片元数据
            index_data = {
                'images': self.image_index,
                'last_updated': datetime.now().isoformat(),
                'total_images': len(self.image_index)
            }
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            # 保存嵌入向量
            if self.image_embeddings is not None:
                np.save(self.embeddings_file, self.image_embeddings)
            
            print(f"💾 图片索引已保存: {len(self.image_index)} 张图片")
            
        except Exception as e:
            print(f"❌ 保存索引失败: {e}")
    
    def _generate_image_id(self, image_path: str) -> str:
        """生成图片ID"""
        # 使用文件内容的哈希值作为ID
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def _encode_image(self, image_path: str) -> np.ndarray:
        """对图片进行CLIP编码"""
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # 获取图片嵌入
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ 图片编码失败 {image_path}: {e}")
            raise e
    
    def _encode_text(self, text: str) -> np.ndarray:
        """对文本进行CLIP编码"""
        try:
            # 预处理文本
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # 获取文本嵌入
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ 文本编码失败 '{text}': {e}")
            raise e
    
    def add_image(self, image_path: str, description: str = "", tags: List[str] = None) -> str:
        """
        添加图片到索引
        
        Args:
            image_path: 图片文件路径
            description: 图片描述
            tags: 图片标签
            
        Returns:
            图片ID
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 生成图片ID
            image_id = self._generate_image_id(image_path)
            
            if image_id in self.image_index:
                print(f"📸 图片已存在: {image_id}")
                return image_id
            
            # 复制图片到存储目录
            file_ext = Path(image_path).suffix
            stored_path = self.storage_dir / f"{image_id}{file_ext}"
            shutil.copy2(image_path, stored_path)
            
            # 编码图片
            print(f"🔄 正在编码图片: {Path(image_path).name}")
            embedding = self._encode_image(image_path)
            
            # 获取图片信息
            image = Image.open(image_path)
            width, height = image.size
            file_size = os.path.getsize(image_path)
            
            # 添加到索引
            self.image_index[image_id] = {
                'id': image_id,
                'original_name': Path(image_path).name,
                'stored_path': str(stored_path),
                'description': description,
                'tags': tags or [],
                'width': width,
                'height': height,
                'file_size': file_size,
                'format': image.format,
                'created_at': datetime.now().isoformat(),
                'embedding_index': len(self.image_ids)
            }
            
            # 更新嵌入矩阵
            if self.image_embeddings is None:
                self.image_embeddings = embedding.reshape(1, -1)
            else:
                self.image_embeddings = np.vstack([self.image_embeddings, embedding])
            
            self.image_ids.append(image_id)
            
            # 保存索引
            self._save_index()
            
            print(f"✅ 图片添加成功: {image_id}")
            return image_id
            
        except Exception as e:
            print(f"❌ 添加图片失败: {e}")
            raise e
    
    def search_by_image(self, query_image_path: str, top_k: int = 10) -> List[Dict]:
        """
        图搜图
        
        Args:
            query_image_path: 查询图片路径
            top_k: 返回最相似的K张图片
            
        Returns:
            相似图片列表
        """
        try:
            if len(self.image_ids) == 0:
                return []
            
            # 编码查询图片
            query_embedding = self._encode_image(query_image_path)
            
            # 计算相似度
            similarities = np.dot(self.image_embeddings, query_embedding)
            
            # 获取最相似的图片
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                image_id = self.image_ids[idx]
                image_info = self.image_index[image_id].copy()
                image_info['similarity'] = float(similarities[idx])
                results.append(image_info)
            
            return results
            
        except Exception as e:
            print(f"❌ 图搜图失败: {e}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        文搜图
        
        Args:
            query_text: 查询文本
            top_k: 返回最相似的K张图片
            
        Returns:
            相似图片列表
        """
        try:
            if len(self.image_ids) == 0:
                return []
            
            # 编码查询文本
            query_embedding = self._encode_text(query_text)
            
            # 计算相似度
            similarities = np.dot(self.image_embeddings, query_embedding)
            
            # 获取最相似的图片
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                image_id = self.image_ids[idx]
                image_info = self.image_index[image_id].copy()
                image_info['similarity'] = float(similarities[idx])
                results.append(image_info)
            
            return results
            
        except Exception as e:
            print(f"❌ 文搜图失败: {e}")
            return []
    
    def get_image_info(self, image_id: str) -> Optional[Dict]:
        """获取图片信息"""
        return self.image_index.get(image_id)
    
    def get_all_images(self) -> List[Dict]:
        """获取所有图片信息"""
        return list(self.image_index.values())
    
    def delete_image(self, image_id: str) -> bool:
        """删除图片"""
        try:
            if image_id not in self.image_index:
                return False
            
            # 获取图片信息
            image_info = self.image_index[image_id]
            embedding_index = image_info['embedding_index']
            
            # 删除存储的图片文件
            stored_path = Path(image_info['stored_path'])
            if stored_path.exists():
                stored_path.unlink()
            
            # 从索引中删除
            del self.image_index[image_id]
            
            # 从嵌入矩阵中删除
            if self.image_embeddings is not None:
                self.image_embeddings = np.delete(self.image_embeddings, embedding_index, axis=0)
            
            # 从ID列表中删除
            self.image_ids.remove(image_id)
            
            # 更新其他图片的embedding_index
            for img_id, img_info in self.image_index.items():
                if img_info['embedding_index'] > embedding_index:
                    img_info['embedding_index'] -= 1
            
            # 保存索引
            self._save_index()
            
            print(f"✅ 图片删除成功: {image_id}")
            return True
            
        except Exception as e:
            print(f"❌ 删除图片失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_size = 0
        formats = {}
        
        for image_info in self.image_index.values():
            total_size += image_info['file_size']
            format_name = image_info['format'] or 'Unknown'
            formats[format_name] = formats.get(format_name, 0) + 1
        
        return {
            'total_images': len(self.image_index),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'formats': formats,
            'storage_dir': str(self.storage_dir),
            'model_device': self.device,
            'embedding_dimension': self.image_embeddings.shape[1] if self.image_embeddings is not None else 0
        }
    
    def clear_index(self):
        """清空图片索引"""
        try:
            # 删除所有存储的图片
            for image_info in self.image_index.values():
                stored_path = Path(image_info['stored_path'])
                if stored_path.exists():
                    stored_path.unlink()
            
            # 清空索引
            self.image_index = {}
            self.image_embeddings = None
            self.image_ids = []
            
            # 删除索引文件
            if self.index_file.exists():
                self.index_file.unlink()
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            
            print("✅ 图片索引已清空")
            
        except Exception as e:
            print(f"❌ 清空索引失败: {e}")
