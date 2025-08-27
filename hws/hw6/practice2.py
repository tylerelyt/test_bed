#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP 图像检索系统 - 以图搜图
使用 OpenAI CLIP 模型构建图像检索系统

功能：
1. 图像特征提取（image embedding）
2. 图片索引库构建
3. 基于余弦相似度的图像检索
4. Top-K 检索结果返回
5. 检索效果可视化
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import pickle
from typing import List, Dict, Tuple, Optional
import time
from tqdm import tqdm
import argparse

# CLIP 相关导入
try:
    import clip
    from clip import load
    CLIP_AVAILABLE = True
except ImportError:
    try:
        import open_clip
        CLIP_AVAILABLE = True
        print("使用 open-clip 作为 CLIP 实现")
    except ImportError:
        CLIP_AVAILABLE = False
        print("CLIP 未安装，请运行: pip install clip-by-openai 或 pip install open-clip")

class CLIPImageRetriever:
    """CLIP 图像检索器"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        初始化 CLIP 检索器
        
        Args:
            model_name: CLIP 模型名称
            device: 计算设备
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP 未安装，请先安装 CLIP 库")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # 加载 CLIP 模型
        self.model, self.preprocess = self._load_clip_model()
        
        # 图像索引库
        self.image_index = {}  # {image_path: embedding}
        self.image_paths = []  # 图像路径列表
        self.embeddings = None  # 所有图像的特征矩阵
        
        print(f"CLIP 检索器初始化完成")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
    
    def _load_clip_model(self):
        """加载 CLIP 模型"""
        try:
            # 尝试使用官方 CLIP
            if 'clip' in globals():
                model, preprocess = load(self.model_name, device=self.device)
                print(f"使用官方 CLIP 模型: {self.model_name}")
                return model, preprocess
        except:
            pass
        
        try:
            # 尝试使用 open-clip
            if 'open_clip' in globals():
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained='openai'
                )
                model = model.to(self.device)
                print(f"使用 open-clip 模型: {self.model_name}")
                return model, preprocess
        except:
            pass
        
        raise RuntimeError("无法加载 CLIP 模型")
    
    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        """
        提取单张图像的特征向量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            np.ndarray: 图像特征向量
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 预处理图像
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                
                # 标准化特征向量
                image_features = F.normalize(image_features, p=2, dim=1)
                
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"图像特征提取失败 {image_path}: {e}")
            return None
    
    def build_image_index(self, image_dir: str, save_path: str = None) -> Dict[str, np.ndarray]:
        """
        构建图像索引库
        
        Args:
            image_dir: 图像目录路径
            save_path: 索引保存路径
            
        Returns:
            Dict[str, np.ndarray]: 图像索引字典
        """
        print(f"开始构建图像索引库...")
        print(f"   图像目录: {image_dir}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 收集图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).rglob(f"*{ext}"))
            image_files.extend(Path(image_dir).rglob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return {}
        
        print(f"   找到 {len(image_files)} 张图像")
        
        # 提取特征向量
        self.image_index = {}
        valid_embeddings = []
        valid_paths = []
        
        for image_path in tqdm(image_files, desc="提取图像特征"):
            embedding = self.extract_image_embedding(str(image_path))
            if embedding is not None:
                self.image_index[str(image_path)] = embedding
                valid_embeddings.append(embedding)
                valid_paths.append(str(image_path))
        
        # 构建特征矩阵
        if valid_embeddings:
            self.embeddings = np.vstack(valid_embeddings)
            self.image_paths = valid_paths
            
            print(f"索引库构建完成: {len(self.image_index)} 张图像")
            print(f"   特征维度: {self.embeddings.shape[1]}")
            
            # 保存索引
            if save_path:
                self.save_index(save_path)
            
            return self.image_index
        else:
            print("没有成功提取到任何图像特征")
            return {}
    
    def save_index(self, save_path: str):
        """保存图像索引"""
        try:
            index_data = {
                'image_index': self.image_index,
                'image_paths': self.image_paths,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            print(f"索引库已保存到: {save_path}")
            
        except Exception as e:
            print(f"索引保存失败: {e}")
    
    def load_index(self, load_path: str):
        """加载图像索引"""
        try:
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.image_index = index_data['image_index']
            self.image_paths = index_data['image_paths']
            self.embeddings = index_data['embeddings']
            
            print(f"索引库加载完成: {len(self.image_index)} 张图像")
            return True
            
        except Exception as e:
            print(f"索引加载失败: {e}")
            return False
    
    def search_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        搜索相似图像
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回前K个最相似的图像
            
        Returns:
            List[Tuple[str, float]]: (图像路径, 相似度分数) 的列表
        """
        if not self.image_index:
            print("图像索引库为空，请先构建索引")
            return []
        
        print(f"开始搜索相似图像...")
        print(f"   查询图像: {query_image_path}")
        print(f"   返回数量: Top-{top_k}")
        
        # 提取查询图像特征
        query_embedding = self.extract_image_embedding(query_image_path)
        if query_embedding is None:
            return []
        
        # 计算余弦相似度
        similarities = []
        for image_path, embedding in self.image_index.items():
            # 计算余弦相似度
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((image_path, similarity))
        
        # 按相似度排序，返回 Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"搜索完成，找到 {len(top_results)} 个结果")
        for i, (path, score) in enumerate(top_results):
            print(f"   {i+1}. {os.path.basename(path)} (相似度: {score:.4f})")
        
        return top_results
    
    def visualize_search_results(self, query_image_path: str, search_results: List[Tuple[str, float]], 
                               save_path: str = None):
        """
        可视化检索结果
        
        Args:
            query_image_path: 查询图像路径
            search_results: 检索结果列表
            save_path: 保存路径
        """
        if not search_results:
            print("没有检索结果可可视化")
            return
        
        # 计算子图布局
        n_results = len(search_results)
        cols = min(5, n_results + 1)  # 查询图 + 结果图
        rows = (n_results + cols) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 显示查询图像
        try:
            query_img = Image.open(query_image_path).convert('RGB')
            axes[0, 0].imshow(query_img)
            axes[0, 0].set_title(f"查询图像\n{os.path.basename(query_image_path)}", fontsize=10)
            axes[0, 0].axis('off')
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f"查询图像加载失败\n{str(e)}", 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title("查询图像")
            axes[0, 0].axis('off')
        
        # 显示检索结果
        for i, (result_path, similarity) in enumerate(search_results):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            try:
                result_img = Image.open(result_path).convert('RGB')
                axes[row, col].imshow(result_img)
                axes[row, col].set_title(f"结果 {i+1}\n{os.path.basename(result_path)}\n相似度: {similarity:.4f}", 
                                       fontsize=10)
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"图像加载失败\n{str(e)}", 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f"结果 {i+1}")
                axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_results + 1, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"CLIP 图像检索结果 (Top-{len(search_results)})", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def batch_search(self, query_images: List[str], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        批量搜索相似图像
        
        Args:
            query_images: 查询图像路径列表
            top_k: 返回前K个最相似的图像
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: 每个查询图像的检索结果
        """
        results = {}
        
        for query_path in tqdm(query_images, desc="批量搜索"):
            try:
                search_results = self.search_similar_images(query_path, top_k)
                results[query_path] = search_results
            except Exception as e:
                print(f"查询图像 {query_path} 搜索失败: {e}")
                results[query_path] = []
        
        return results
    
    def get_index_statistics(self) -> Dict:
        """获取索引库统计信息"""
        if not self.image_index:
            return {}
        
        # 计算特征向量的统计信息
        embeddings_array = np.array(list(self.image_index.values()))
        
        stats = {
            'total_images': len(self.image_index),
            'feature_dimension': embeddings_array.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            'image_paths': list(self.image_index.keys())
        }
        
        return stats

class ImageRetrievalDemo:
    """图像检索演示类"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.retriever = CLIPImageRetriever(model_name)
        self.index_built = False
    
    def demo_build_index(self, image_dir: str, save_path: str = "image_index.pkl"):
        """演示构建索引库"""
        print("=" * 60)
        print("演示：构建图像索引库")
        print("=" * 60)
        
        # 构建索引
        self.retriever.build_image_index(image_dir, save_path)
        self.index_built = True
        
        # 显示统计信息
        stats = self.retriever.get_index_statistics()
        if stats:
            print(f"\n索引库统计信息:")
            print(f"   总图像数: {stats['total_images']}")
            print(f"   特征维度: {stats['feature_dimension']}")
            print(f"   特征向量平均范数: {stats['mean_norm']:.4f}")
            print(f"   特征向量标准差: {stats['std_norm']:.4f}")
    
    def demo_search(self, query_image_path: str, top_k: int = 5, save_vis: str = None):
        """演示图像检索"""
        if not self.index_built:
            print("请先构建图像索引库")
            return
        
        print("=" * 60)
        print("演示：图像检索")
        print("=" * 60)
        
        # 执行检索
        start_time = time.time()
        search_results = self.retriever.search_similar_images(query_image_path, top_k)
        search_time = time.time() - start_time
        
        if search_results:
            print(f"\n检索耗时: {search_time:.3f} 秒")
            
            # 可视化结果
            self.retriever.visualize_search_results(query_image_path, search_results, save_vis)
        else:
            print("检索失败，没有找到结果")
    
    def demo_batch_search(self, query_images: List[str], top_k: int = 5):
        """演示批量检索"""
        if not self.index_built:
            print("请先构建图像索引库")
            return
        
        print("=" * 60)
        print("演示：批量图像检索")
        print("=" * 60)
        
        # 批量检索
        start_time = time.time()
        batch_results = self.retriever.batch_search(query_images, top_k)
        total_time = time.time() - start_time
        
        print(f"\n批量检索完成，总耗时: {total_time:.3f} 秒")
        print(f"   查询图像数: {len(query_images)}")
        print(f"   平均每张图像检索时间: {total_time/len(query_images):.3f} 秒")
        
        # 显示结果摘要
        for query_path, results in batch_results.items():
            if results:
                best_match = results[0]
                print(f"   {os.path.basename(query_path)} -> {os.path.basename(best_match[0])} (相似度: {best_match[1]:.4f})")
            else:
                print(f"   {os.path.basename(query_path)} -> 无结果")

def main():
    """主函数 - 演示完整流程"""
    print("CLIP 图像检索系统演示")
    print("=" * 60)
    
    # 检查 CLIP 可用性
    if not CLIP_AVAILABLE:
        print("CLIP 库未安装")
        print("请运行以下命令安装:")
        print("  pip install clip-by-openai")
        print("  或")
        print("  pip install open-clip")
        return
    
    # 创建演示实例
    demo = ImageRetrievalDemo(model_name="ViT-B/32")
    
    # 检查是否有预构建的索引
    index_path = "image_index.pkl"
    if os.path.exists(index_path):
        print(f"发现预构建索引: {index_path}")
        if demo.retriever.load_index(index_path):
            demo.index_built = True
            print("索引加载成功")
        else:
            print("索引加载失败，将重新构建")
    
    # 如果没有索引，构建新的索引库
    if not demo.index_built:
        # 检查图像目录
        image_dir = "images"  # 默认图像目录
        if not os.path.exists(image_dir):
            print(f"\n图像目录不存在: {image_dir}")
            print("请创建 images 目录并放入图像文件，或修改 image_dir 变量")
            print("目录结构示例:")
            print("  images/")
            print("  ├── cat1.jpg")
            print("  ├── cat2.jpg")
            print("  ├── dog1.jpg")
            print("  └── ...")
            return
        
        # 构建索引
        demo.demo_build_index(image_dir, index_path)
    
    # 演示图像检索
    if demo.index_built:
        # 查找一些测试图像
        test_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_images.extend(Path("images").rglob(f"*{ext}"))
            test_images.extend(Path("images").rglob(f"*{ext.upper()}"))
        
        if test_images:
            # 选择第一张图像作为查询图像
            query_image = str(test_images[0])
            print(f"\n使用图像作为查询: {os.path.basename(query_image)}")
            
            # 执行检索
            demo.demo_search(query_image, top_k=5, save_vis="search_results.png")
            
            # 批量检索演示
            if len(test_images) > 1:
                demo_images = [str(img) for img in test_images[:3]]  # 取前3张作为查询
                demo.demo_batch_search(demo_images, top_k=3)
        else:
            print("没有找到测试图像")
    
    print(f"\n演示完成！")
    print(f"索引库包含 {len(demo.retriever.image_index)} 张图像")
    print(f"索引文件: {index_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CLIP 图像检索系统")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP 模型名称")
    parser.add_argument("--build_index", help="构建索引的图像目录")
    parser.add_argument("--search", help="查询图像路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回前K个结果")
    parser.add_argument("--save_vis", help="保存可视化结果")
    
    args = parser.parse_args()
    
    if args.build_index:
        # 构建索引模式
        retriever = CLIPImageRetriever(args.model)
        retriever.build_image_index(args.build_index, "image_index.pkl")
        
    elif args.search:
        # 检索模式
        retriever = CLIPImageRetriever(args.model)
        if retriever.load_index("image_index.pkl"):
            results = retriever.search_similar_images(args.search, args.top_k)
            if results:
                retriever.visualize_search_results(args.search, results, args.save_vis)
        else:
            print("请先构建索引库")
            
    else:
        # 交互式演示模式
        main()
