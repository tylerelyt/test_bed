#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文维基百科数据集下载工具
从Hugging Face下载中文维基百科数据集并转换为系统预置文档格式
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

def download_wikipedia_dataset(dataset_name: str = "fjcanyue/wikipedia-zh-cn", 
                             split: str = "train", 
                             max_samples: int = 1000,
                             output_file: str = "data/preloaded_documents.json") -> Dict[str, str]:
    """
    从Hugging Face下载中文维基百科数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割 (train, validation, test)
        max_samples: 最大样本数
        output_file: 输出文件路径
        
    Returns:
        Dict[str, str]: 文档字典 {doc_id: content}
    """
    
    print(f"🔍 开始下载中文维基百科数据集: {dataset_name}")
    print(f"📊 分割: {split}, 最大样本数: {max_samples}")
    print(f"💡 只下载第一个文件: wikipedia-zh-cn-20240901.json")
    
    try:
        # 使用datasets库下载
        from datasets import load_dataset
        print("📦 使用datasets库下载...")
        
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10分钟超时
        
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        print(f"✅ 数据集加载成功")
        
        documents = {}
        sample_count = 0
        print(f"🔄 处理前{max_samples}个样本...")
        
        for i, sample in enumerate(tqdm(dataset, desc="处理样本")):
            if sample_count >= max_samples:
                break
                
            title = sample.get("title", "")
            text = sample.get("text", "")
            tags = sample.get("tags", "")
            
            # 只保留纯文本内容，不包含标题和标签
            content = text.strip()
            
            doc_id = f"wiki_{i:06d}"
            documents[doc_id] = content
            sample_count += 1
            
        print(f"✅ 成功处理 {sample_count} 个样本")
        return documents
        
    except ImportError:
        print("❌ 需要安装datasets库: pip install datasets")
        return {}
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return {}

def save_documents(documents: Dict[str, str], output_file: str):
    """保存文档到JSON文件"""
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 构建输出数据
    output_data = {
        "export_time": datetime.now().isoformat(),
        "dataset_info": {
            "name": "Wikipedia Chinese",
            "description": "中文维基百科数据集",
            "total_documents": len(documents),
            "source": "Hugging Face - fjcanyue/wikipedia-zh-cn"
        },
        "documents": documents
    }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 文档已保存到: {output_file}")
    print(f"📊 共保存 {len(documents)} 个文档")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文维基百科数据集下载工具")
    parser.add_argument("--dataset", default="fjcanyue/wikipedia-zh-cn",
                       help="数据集名称")
    parser.add_argument("--split", default="train", 
                       choices=["train", "validation", "test"],
                       help="数据集分割")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="最大样本数 (默认: 1000)")
    parser.add_argument("--output", default="data/preloaded_documents.json",
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📚 中文维基百科数据集下载工具")
    print("=" * 60)
    
    print(f"🔍 下载数据集: {args.dataset}")
    documents = download_wikipedia_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_file=args.output
    )
    
    if documents:
        save_documents(documents, args.output)
        print("\n✅ 完成！")
        print(f"💡 现在可以启动系统，预置文档将自动加载")
        print(f"📖 文档数量: {len(documents)}")
        print(f"📁 文件位置: {args.output}")
    else:
        print("\n❌ 失败！")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 安装datasets库: pip install datasets")
        print("   3. 确认数据集名称正确")

if __name__ == "__main__":
    main()
