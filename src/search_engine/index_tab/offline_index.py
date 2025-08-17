#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线模块 - 索引构建+样本收集
负责倒排索引构建、文档管理、样本收集等离线任务
"""

import jieba
import re
import json
import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime
import os

class InvertedIndex:
    """倒排索引类"""
    
    def __init__(self):
        self.index = defaultdict(set)  # 词项 -> 文档ID集合
        self.doc_lengths = {}          # 文档ID -> 文档长度
        self.documents = {}            # 文档ID -> 文档内容
        self.term_freq = defaultdict(dict)  # 词项 -> {文档ID: 词频}
        self.doc_freq = defaultdict(int)    # 词项 -> 文档频率
        
        # 停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 分词
        words = jieba.lcut(text.lower())
        
        # 过滤停用词和短词
        words = [word for word in words if len(word) > 1 and word not in self.stop_words]
        
        return words
    
    def add_document(self, doc_id: str, content: str):
        """添加文档到索引"""
        # 保存原始文档
        self.documents[doc_id] = content
        
        # 预处理文本
        words = self.preprocess_text(content)
        
        # 计算文档长度
        self.doc_lengths[doc_id] = len(words)
        
        # 统计词频
        word_freq = Counter(words)
        
        # 更新倒排索引
        for word, freq in word_freq.items():
            self.index[word].add(doc_id)
            self.term_freq[word][doc_id] = freq
        
        # 更新文档频率
        for word in word_freq:
            self.doc_freq[word] = len(self.index[word])
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档从索引"""
        if doc_id not in self.documents:
            return False
        
        # 获取文档的词频信息
        content = self.documents[doc_id]
        words = self.preprocess_text(content)
        word_freq = Counter(words)
        
        # 从倒排索引中移除文档
        for word in word_freq:
            if word in self.index:
                self.index[word].discard(doc_id)
                # 如果词项没有文档了，删除该词项
                if not self.index[word]:
                    del self.index[word]
                    if word in self.term_freq:
                        del self.term_freq[word]
                    if word in self.doc_freq:
                        del self.doc_freq[word]
                else:
                    # 更新词频信息
                    if word in self.term_freq and doc_id in self.term_freq[word]:
                        del self.term_freq[word][doc_id]
                    # 更新文档频率
                    if word in self.doc_freq:
                        self.doc_freq[word] = len(self.index[word])
        
        # 删除文档相关数据
        del self.documents[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """搜索文档 - 优化版本使用真正的倒排索引"""
        # 预处理查询
        query_words = self.preprocess_text(query)
        
        if not query_words:
            return []
        
        # 使用倒排索引快速找到候选文档
        candidate_docs = set()
        for word in query_words:
            if word in self.index:
                candidate_docs.update(self.index[word])
        
        if not candidate_docs:
            return []
        
        # 只对候选文档计算TF-IDF分数
        scores = {}
        total_docs = len(self.documents)
        
        for doc_id in candidate_docs:
            score = 0
            for word in query_words:
                if word in self.index and doc_id in self.index[word]:
                    # TF
                    tf = self.term_freq[word][doc_id] / self.doc_lengths[doc_id]
                    # IDF
                    idf = math.log(total_docs / self.doc_freq[word])
                    # TF-IDF
                    score += tf * idf
            
            if score > 0:
                scores[doc_id] = score
        
        # 排序并返回结果
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成摘要
        results = []
        for doc_id, score in sorted_results[:top_k]:
            summary = self.generate_summary(doc_id, query_words)
            results.append((doc_id, score, summary))
        
        return results
    
    def generate_summary(self, doc_id: str, query_words: List[str], max_length: int = 200) -> str:
        """生成文档摘要 - 优化版本"""
        content = self.documents[doc_id]
        
        # 快速生成摘要：找到第一个查询词的位置，然后截取周围文本
        if not query_words:
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # 在原文中找到第一个查询词的位置
        best_pos = 0
        for word in query_words:
            pos = content.lower().find(word.lower())
            if pos != -1:
                best_pos = max(0, pos - max_length // 3)  # 从查询词前1/3位置开始
                break
        
        # 截取摘要
        summary = content[best_pos:best_pos + max_length]
        if best_pos > 0:
            summary = "..." + summary
        if len(content) > best_pos + max_length:
            summary = summary + "..."
        
        # 高亮查询词
        highlighted_summary = self.highlight_keywords(summary, query_words)
        
        return highlighted_summary
    
    def highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """高亮关键词"""
        highlighted_text = text
        for keyword in keywords:
            if keyword in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    keyword, 
                    f'<span style="background-color: yellow; font-weight: bold;">{keyword}</span>'
                )
        return highlighted_text
    
    def get_document(self, doc_id: str) -> str:
        """获取文档内容"""
        return self.documents.get(doc_id, "")
    
    def get_all_documents(self) -> Dict[str, str]:
        """获取所有文档"""
        return self.documents.copy()
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        total_documents = len(self.documents)
        total_terms = len(self.index)
        
        if total_documents > 0:
            average_doc_length = sum(self.doc_lengths.values()) / total_documents
        else:
            average_doc_length = 0
        
        return {
            'total_documents': total_documents,
            'total_terms': total_terms,
            'average_doc_length': average_doc_length
        }
    
    def save_to_file(self, filename: str):
        """保存索引到文件"""
        data = {
            'index': {k: list(v) for k, v in self.index.items()},
            'doc_lengths': self.doc_lengths,
            'documents': self.documents,
            'term_freq': {k: dict(v) for k, v in self.term_freq.items()},
            'doc_freq': dict(self.doc_freq)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 索引已保存到: {filename}")
    
    def load_from_file(self, filename: str):
        """从文件加载索引"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.index = defaultdict(set)
        for k, v in data['index'].items():
            self.index[k] = set(v)
        
        self.doc_lengths = data['doc_lengths']
        self.documents = data['documents']
        
        self.term_freq = defaultdict(dict)
        for k, v in data['term_freq'].items():
            self.term_freq[k] = v
        
        self.doc_freq = defaultdict(int)
        for k, v in data['doc_freq'].items():
            self.doc_freq[k] = v
        
        print(f"✅ 索引已从文件加载: {filename}")

class SampleCollector:
    """样本收集器"""
    
    def __init__(self):
        self.samples = []
    
    def add_sample(self, sample: Dict):
        """添加样本"""
        self.samples.append(sample)
    
    def get_samples(self) -> List[Dict]:
        """获取所有样本"""
        return self.samples
    
    def export_samples(self, filename: str):
        """导出样本到文件"""
        df = pd.DataFrame(self.samples)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✅ 样本已导出到: {filename}")
    
    def get_stats(self) -> Dict:
        """获取样本统计"""
        if not self.samples:
            return {
                'total_samples': 0,
                'total_clicks': 0,
                'click_rate': 0
            }
        
        total_samples = len(self.samples)
        total_clicks = sum(sample.get('clicked', 0) for sample in self.samples)
        click_rate = total_clicks / total_samples if total_samples > 0 else 0
        
        return {
            'total_samples': total_samples,
            'total_clicks': total_clicks,
            'click_rate': click_rate
        }

def create_sample_documents():
    """创建示例文档"""
    documents = {
        "doc1": """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """,
        "doc2": """
        机器学习是人工智能的一个子集，它使用统计学方法让计算机系统能够"学习"（即，逐步提高特定任务的性能），而无需明确编程。
        机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。
        """,
        "doc3": """
        深度学习是机器学习的一个分支，它基于人工神经网络，特别是深层神经网络。
        深度学习模型可以自动学习数据的层次表示，这使得它们在图像识别、语音识别和自然语言处理等任务中表现出色。
        """,
        "doc4": """
        自然语言处理是人工智能和语言学的一个交叉领域，它研究计算机与人类语言之间的交互。
        NLP技术被广泛应用于机器翻译、情感分析、问答系统和聊天机器人等应用。
        """,
        "doc5": """
        计算机视觉是人工智能的一个分支，它使计算机能够从数字图像或视频中获得高层次的理解。
        计算机视觉技术包括图像识别、目标检测、图像分割和视频分析等。
        """,
        "doc6": """
        神经网络是一种模仿生物神经系统的计算模型，由大量相互连接的神经元组成。
        神经网络能够学习复杂的非线性关系，在模式识别和预测任务中表现出色。
        """,
        "doc7": """
        强化学习是机器学习的一种方法，通过让智能体与环境交互来学习最优策略。
        强化学习在游戏、机器人控制和自动驾驶等领域有重要应用。
        """,
        "doc8": """
        知识图谱是一种结构化的知识表示方法，将实体和关系组织成图结构。
        知识图谱在搜索引擎、推荐系统和问答系统中发挥重要作用。
        """,
        "doc9": """
        大数据是指无法用传统数据处理软件在合理时间内处理的数据集。
        大数据技术包括数据存储、数据处理、数据分析和数据可视化等方面。
        """,
        "doc10": """
        云计算是一种通过互联网提供计算资源的服务模式。
        云计算包括基础设施即服务、平台即服务和软件即服务等不同层次。
        """
    }
    return documents

def build_index_from_documents(documents: Dict[str, str], save_path: str = ""):
    """从文档构建索引"""
    print("🔨 构建倒排索引...")
    
    index = InvertedIndex()
    
    for doc_id, content in documents.items():
        index.add_document(doc_id, content)
        print(f"   添加文档: {doc_id}")
    
    stats = index.get_index_stats()
    print(f"✅ 索引构建完成:")
    print(f"   总文档数: {stats['total_documents']}")
    print(f"   总词项数: {stats['total_terms']}")
    print(f"   平均文档长度: {stats['average_doc_length']:.2f}")
    
    if save_path:
        index.save_to_file(save_path)
    
    return index

def main():
    """主函数 - 构建索引"""
    print("🏗️  离线索引构建模块")
    print("=" * 50)
    
    # 优先使用预置文档
    import os
    import json
    
    preloaded_path = os.path.join("data", "preloaded_documents.json")
    if os.path.exists(preloaded_path):
        print("📄 使用预置文档构建索引")
        with open(preloaded_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 支持两种格式
        if isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            documents = data
        print(f"✅ 加载预置文档成功，共{len(documents)}个文档")
    else:
        print("⚠️ 未找到预置文档，使用示例文档")
        # 回退到示例文档
        documents = create_sample_documents()
        print(f"✅ 创建示例文档成功，共{len(documents)}个文档")
    
    # 构建索引
    index = build_index_from_documents(documents, 'models/index_data.json')
    
    # 测试搜索
    print("\n🔍 测试搜索功能:")
    test_queries = ["人工智能", "机器学习", "深度学习"]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = index.search(query, top_k=3)
        for doc_id, score, summary in results:
            print(f"  - {doc_id}: {score:.4f}")
    
    print("\n✅ 离线索引构建完成!")
    print("💡 现在可以启动在线服务: python online_service.py")

if __name__ == "__main__":
    main() 