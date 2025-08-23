#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenKG 中文知识图谱生成器

提供端到端的一键体验：
- 生成示例规模的中文知识图谱三元组数据
- 保存到 data/openkg_triples.tsv
- 启动时 KGRetrievalService 会优先加载该文件的前若干条构建只读图谱

注意：本脚本生成的是演示数据，实际应用中请使用真实的 OpenKG 数据集。
"""

import os
import sys
import argparse
import random


def download_openkg_sample(dest: str, max_lines: int = 1000) -> None:
    """下载 OpenKG OpenConcepts 示例数据"""
    
    import urllib.request
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"🔽 Downloading OpenKG OpenConcepts sample data...")
    
    # OpenConcepts 的概念层次数据
    source_url = "https://raw.githubusercontent.com/OpenKG-ORG/OpenConcepts/main/level1_level2.sample.txt"
    
    try:
        # 下载数据
        response = urllib.request.urlopen(source_url)
        content = response.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # 转换为三元组格式：概念 \t 属于 \t 类别
        triples = []
        for line in lines[:max_lines]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    concept = parts[0]
                    category = parts[1]
                    # 创建 "是" 关系的三元组
                    triple = f"{concept}\t属于\t{category}"
                    triples.append(triple)
        
        # 保存文件
        with open(dest, 'w', encoding='utf-8') as f:
            f.write('\n'.join(triples))
        
        print(f"✅ Downloaded and saved to: {dest}")
        print(f"📊 Total triples: {len(triples)}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="OpenKG 中文知识图谱下载器")
    parser.add_argument("--output", default="data/openkg_triples.tsv", help="输出文件路径")
    parser.add_argument("--max-lines", type=int, default=1000, help="下载的最大行数")
    args = parser.parse_args()

    try:
        download_openkg_sample(args.output, args.max_lines)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


