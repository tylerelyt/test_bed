"""
领域语料处理器
用于 CPT（继续预训练）的数据准备
"""
import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime
import hashlib


class DomainCorpusProcessor:
    """领域语料处理器"""
    
    def __init__(self, output_dir: str = "data/llmops/cpt"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.corpus = []
        self.processed_corpus = []
    
    def load_from_preloaded(self, limit: int = 1000) -> int:
        """从预加载文档中加载语料"""
        try:
            preloaded_path = os.path.join("data", "preloaded_documents.json")
            if not os.path.exists(preloaded_path):
                return 0
            
            with open(preloaded_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = data["documents"] if isinstance(data, dict) and "documents" in data else data
            
            count = 0
            if isinstance(docs, dict):
                for doc_id, content in docs.items():
                    if count >= limit:
                        break
                    self.corpus.append(str(content).strip())
                    count += 1
            else:
                for content in docs:
                    if count >= limit:
                        break
                    self.corpus.append(str(content).strip())
                    count += 1
            
            return count
        except Exception as e:
            print(f"加载预置文档失败: {e}")
            return 0
    
    def add_text(self, text: str):
        """添加文本到语料库"""
        if text and len(text.strip()) > 0:
            self.corpus.append(text.strip())
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符（保留中英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、,.!?;:\s]', '', text)
        return text.strip()
    
    def deduplicate(self):
        """去重"""
        seen = set()
        unique_corpus = []
        
        for text in self.corpus:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique_corpus.append(text)
        
        self.corpus = unique_corpus
        return len(unique_corpus)
    
    def segment_texts(self, max_length: int = 2048) -> List[str]:
        """
        将文本分段
        
        Args:
            max_length: 最大字符数（约等于 tokens）
        """
        segments = []
        
        for text in self.corpus:
            # 清洗
            text = self.clean_text(text)
            
            if len(text) < 10:  # 过滤太短的文本（降低阈值以支持短句）
                continue
            
            # 如果文本太长，按句子分段
            if len(text) > max_length:
                sentences = re.split(r'[。！？\n]', text)
                current_segment = ""
                
                for sentence in sentences:
                    if len(current_segment) + len(sentence) < max_length:
                        current_segment += sentence + "。"
                    else:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = sentence + "。"
                
                if current_segment:
                    segments.append(current_segment)
            else:
                segments.append(text)
        
        return segments
    
    def process(self, max_length: int = 2048) -> int:
        """
        处理语料库：清洗 → 去重 → 分段
        
        Returns:
            处理后的文本数量
        """
        # 去重
        self.deduplicate()
        
        # 分段
        self.processed_corpus = self.segment_texts(max_length)
        
        return len(self.processed_corpus)
    
    def save_corpus(self, filename: str = None) -> str:
        """
        保存为 JSONL 格式（每行一个 JSON 对象）
        适用于 LLaMA-Factory CPT 训练
        同时自动注册到 dataset_info.json
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"domain_corpus_{timestamp}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for text in self.processed_corpus:
                json_obj = {"text": text}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        # 自动注册数据集到 dataset_info.json
        self._register_dataset(filename, "domain_corpus")
        
        return filepath
    
    def _register_dataset(self, filename: str, dataset_name: str = None):
        """将数据集注册到 dataset_info.json（LLaMA-Factory 数据集配置）"""
        # dataset_info.json 统一放在 data/llmops/ 根目录
        dataset_info_path = "data/llmops/dataset_info.json"
        
        # 读取现有配置
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # 使用文件名（不含扩展名）或自定义名称作为数据集名称
        if dataset_name is None:
            dataset_name = os.path.splitext(filename)[0]
        
        # 添加或更新数据集配置（使用相对于data/llmops的路径）
        dataset_info[dataset_name] = {
            "file_name": f"cpt/{filename}",
            "columns": {
                "prompt": "text"  # CPT 数据格式：每行一个 text 字段
            }
        }
        
        # 保存配置
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据集已注册: {dataset_name} -> {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.processed_corpus:
            return {
                "raw_count": len(self.corpus),
                "processed_count": 0,
                "total_chars": 0,
                "avg_length": 0,
                "estimated_tokens": 0
            }
        
        total_chars = sum(len(text) for text in self.processed_corpus)
        
        return {
            "raw_count": len(self.corpus),
            "processed_count": len(self.processed_corpus),
            "total_chars": total_chars,
            "avg_length": total_chars // len(self.processed_corpus) if self.processed_corpus else 0,
            "estimated_tokens": total_chars // 2  # 粗略估计：中文 1 字符 ≈ 0.5 token
        }

