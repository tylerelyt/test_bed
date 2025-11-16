"""
偏好数据采集器
用于 DPO（偏好对齐）的数据收集
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class PreferenceCollector:
    """偏好数据采集器"""
    
    def __init__(self, output_dir: str = "data/llmops/dpo"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.preferences = []
        self.prefs_file = os.path.join(output_dir, "prefs.jsonl")
        
        # 加载已有偏好数据
        self._load_existing_prefs()
    
    def _load_existing_prefs(self):
        """加载已有的偏好数据"""
        if os.path.exists(self.prefs_file):
            try:
                with open(self.prefs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.preferences.append(json.loads(line))
            except Exception as e:
                print(f"加载偏好数据失败: {e}")
    
    def add_preference(self, 
                      prompt: str, 
                      chosen: str, 
                      rejected: str,
                      metadata: Optional[Dict] = None) -> bool:
        """
        添加一条偏好数据
        
        Args:
            prompt: 用户输入/问题
            chosen: 用户偏好的回答
            rejected: 用户不偏好的回答
            metadata: 元数据（如模型版本、时间戳等）
        """
        if not prompt or not chosen or not rejected:
            return False
        
        pref_data = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            pref_data["metadata"] = metadata
        
        self.preferences.append(pref_data)
        
        # 实时写入文件
        self._append_to_file(pref_data)
        
        return True
    
    def _append_to_file(self, pref_data: Dict[str, Any]):
        """追加到文件"""
        try:
            with open(self.prefs_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pref_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"写入偏好数据失败: {e}")
    
    def add_batch_preferences(self, preferences: List[Dict[str, Any]]) -> int:
        """批量添加偏好数据"""
        count = 0
        for pref in preferences:
            if self.add_preference(
                pref.get("prompt", ""),
                pref.get("chosen", ""),
                pref.get("rejected", ""),
                pref.get("metadata")
            ):
                count += 1
        return count
    
    def get_all_preferences(self) -> List[Dict[str, Any]]:
        """获取所有偏好数据"""
        return self.preferences
    
    def clear_preferences(self):
        """清空偏好数据"""
        self.preferences = []
        if os.path.exists(self.prefs_file):
            os.remove(self.prefs_file)
    
    def export_for_dpo(self, output_file: str = None) -> str:
        """
        导出为 DPO 训练格式（ShareGPT格式）
        
        Returns:
            导出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"dpo_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, output_file)
        
        dpo_dataset = []
        for pref in self.preferences:
            # DPO ShareGPT 格式
            dpo_data = {
                "conversations": [
                    {"role": "user", "content": pref["prompt"]}
                ],
                "chosen": {"role": "assistant", "content": pref["chosen"]},
                "rejected": {"role": "assistant", "content": pref["rejected"]}
            }
            dpo_dataset.append(dpo_data)
        
        # 保存为 JSON 数组格式（不是 JSONL）
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)
        
        # 自动注册数据集
        self._register_dataset(output_file, "prefs_data")
        
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
        
        # 添加或更新数据集配置（DPO ShareGPT Ranking 格式，使用相对于data/llmops的路径）
        dataset_info[dataset_name] = {
            "file_name": f"dpo/{filename}",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        
        # 保存配置
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ DPO数据集已注册（ShareGPT Ranking格式）: {dataset_name} -> {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_preferences": len(self.preferences),
            "data_file": self.prefs_file,
            "file_exists": os.path.exists(self.prefs_file)
        }
    
    def create_mock_preferences(self, count: int = 10) -> int:
        """创建模拟偏好数据（用于演示）"""
        mock_data = [
            {
                "prompt": "介绍一下人工智能",
                "chosen": "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。它包括机器学习、自然语言处理、计算机视觉等多个子领域，广泛应用于医疗、金融、教育等行业。",
                "rejected": "人工智能就是一种很复杂的技术。"
            },
            {
                "prompt": "什么是机器学习？",
                "chosen": "机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中自动学习和改进，而无需显式编程。主要方法包括监督学习、无监督学习和强化学习。",
                "rejected": "机器学习是让机器学东西的方法。"
            },
            {
                "prompt": "解释深度学习的概念",
                "chosen": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。它在图像识别、语音识别、自然语言处理等领域取得了突破性进展，是现代AI技术的核心驱动力。",
                "rejected": "深度学习就是很深的学习方法。"
            },
            {
                "prompt": "自然语言处理是什么？",
                "chosen": "自然语言处理（NLP）是人工智能的一个重要分支，致力于让计算机理解、解释和生成人类语言。它涵盖文本分类、情感分析、机器翻译、问答系统等多个应用场景。",
                "rejected": "NLP就是处理语言的技术。"
            },
            {
                "prompt": "介绍一下强化学习",
                "chosen": "强化学习是机器学习的一种范式，智能体通过与环境交互，根据奖励信号来学习最优策略。它成功应用于游戏AI、机器人控制、推荐系统等领域。",
                "rejected": "强化学习就是不断强化的学习。"
            },
            {
                "prompt": "计算机视觉的应用有哪些？",
                "chosen": "计算机视觉使计算机能够理解和分析视觉信息。主要应用包括：人脸识别、物体检测、图像分割、自动驾驶、医疗影像分析等。它是AI技术在视觉感知方面的核心体现。",
                "rejected": "计算机视觉就是让计算机看东西。"
            },
            {
                "prompt": "什么是神经网络？",
                "chosen": "神经网络是受生物神经系统启发的计算模型，由大量相互连接的节点（神经元）组成。通过调整连接权重，神经网络可以学习复杂的模式和关系，是深度学习的基础。",
                "rejected": "神经网络就是模仿大脑的网络。"
            },
            {
                "prompt": "解释监督学习和无监督学习的区别",
                "chosen": "监督学习使用标注数据训练模型，学习输入到输出的映射关系，如分类和回归任务。无监督学习则从未标注数据中发现隐藏模式，如聚类和降维。两者的主要区别在于是否需要标注数据。",
                "rejected": "监督学习有老师，无监督学习没老师。"
            },
            {
                "prompt": "什么是迁移学习？",
                "chosen": "迁移学习是一种机器学习方法，将在一个任务上学到的知识应用到另一个相关任务上。它可以显著减少训练数据需求和计算成本，在实际应用中非常有效。",
                "rejected": "迁移学习就是把知识转移过去。"
            },
            {
                "prompt": "AI伦理为什么重要？",
                "chosen": "AI伦理关注人工智能技术的公平性、透明度、隐私保护和社会影响。随着AI在各领域的广泛应用，确保技术造福人类、避免偏见和歧视、保护用户权益变得至关重要。",
                "rejected": "AI伦理就是关于AI的道德问题。"
            }
        ]
        
        added = 0
        for i in range(min(count, len(mock_data))):
            if self.add_preference(
                mock_data[i]["prompt"],
                mock_data[i]["chosen"],
                mock_data[i]["rejected"],
                {"source": "mock", "index": i}
            ):
                added += 1
        
        return added

