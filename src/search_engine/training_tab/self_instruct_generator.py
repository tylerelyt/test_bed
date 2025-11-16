"""
Self-Instruct 数据生成器
实现自动化指令数据生成，用于 SFT 训练
"""
import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


class SelfInstructGenerator:
    """Self-Instruct 数据生成器"""
    
    def __init__(self, seed_file: str = None, output_dir: str = "data/llmops/sft"):
        self.seed_file = seed_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 种子数据集
        self.seed_data = []
        # 生成的数据集
        self.generated_data = []
        
        # 任务类型模板
        self.task_types = {
            "classification": "分类任务",
            "generation": "生成任务",
            "summarization": "总结任务",
            "qa": "问答任务",
            "rewrite": "改写任务",
            "reasoning": "推理任务"
        }
        
        # 加载种子数据
        if seed_file and os.path.exists(seed_file):
            self.load_seed_data(seed_file)
        else:
            self._create_default_seeds()
    
    def _create_default_seeds(self):
        """创建默认种子数据"""
        self.seed_data = [
            {
                "instruction": "判断这条评论是好评还是差评",
                "input": "物流速度很快，包装完好，质量不错",
                "output": "好评",
                "task_type": "classification"
            },
            {
                "instruction": "将下面的文本总结为一句话",
                "input": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解。",
                "output": "人工智能是创建能执行类人智能任务的计算机系统。",
                "task_type": "summarization"
            },
            {
                "instruction": "根据给定的主题写一段短文",
                "input": "可持续发展",
                "output": "可持续发展是指在满足当代人需求的同时，不损害后代人满足其需求的能力。它强调经济增长、环境保护和社会公平的平衡发展，是人类社会长期繁荣的关键。",
                "task_type": "generation"
            },
            {
                "instruction": "回答以下问题",
                "input": "什么是机器学习？",
                "output": "机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习和改进，而无需显式编程。通过分析数据模式，机器学习算法可以做出预测或决策。",
                "task_type": "qa"
            },
            {
                "instruction": "将下面的句子改写得更正式",
                "input": "这个东西真的很好用",
                "output": "该产品的实用性表现优异",
                "task_type": "rewrite"
            },
            {
                "instruction": "根据给定条件进行逻辑推理",
                "input": "如果今天下雨，我就带伞。今天下雨了。",
                "output": "因此，我应该带伞。",
                "task_type": "reasoning"
            }
        ]
    
    def load_seed_data(self, file_path: str):
        """加载种子数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.seed_data = json.load(f)
    
    def save_seed_data(self, file_path: str):
        """保存种子数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.seed_data, f, ensure_ascii=False, indent=2)
    
    def generate_instructions(self, 
                            num_instructions: int = 100,
                            use_mock: bool = True) -> List[Dict[str, Any]]:
        """
        生成指令数据
        
        Args:
            num_instructions: 要生成的指令数量
            use_mock: 是否使用模拟生成（实际应用中应调用 LLM API）
        """
        generated = []
        
        for i in range(num_instructions):
            # 从种子中随机选择示例
            seed_sample = random.choice(self.seed_data)
            task_type = seed_sample.get("task_type", "generation")
            
            if use_mock:
                # 模拟生成（实际应调用 GPT-4 或 Qwen2-72B）
                new_instruction = self._mock_generate(seed_sample, i)
            else:
                # 实际 API 调用（需要配置 API key）
                new_instruction = self._api_generate(seed_sample)
            
            # 质量过滤
            if self._quality_check(new_instruction, generated):
                generated.append(new_instruction)
        
        self.generated_data.extend(generated)
        return generated
    
    def _mock_generate(self, seed: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """模拟生成新指令（演示用）"""
        task_type = seed.get("task_type", "generation")
        
        # 根据任务类型生成变体
        mock_templates = {
            "classification": [
                ("判断这段文字的情感倾向", "这个产品性价比很高", "积极"),
                ("识别这条新闻的类别", "科技公司发布新款智能手机", "科技"),
                ("判断这个句子是陈述句还是疑问句", "今天天气怎么样", "疑问句")
            ],
            "generation": [
                ("写一段关于教育重要性的文字", "", "教育是个人成长和社会进步的基石，它不仅传授知识，更培养思维能力和价值观。"),
                ("创作一首关于春天的小诗", "", "春风拂面暖如酥，万物复苏绿满途。"),
                ("描述一下未来城市的样子", "", "未来城市将实现智能化管理，清洁能源驱动，人与自然和谐共生。")
            ],
            "summarization": [
                ("总结这段话的核心观点", "深度学习在图像识别、自然语言处理等领域取得了突破性进展。", "深度学习在多个AI领域取得突破。"),
                ("用一句话概括这篇文章", "气候变化是当今世界面临的重大挑战之一。", "气候变化是全球重大挑战。")
            ],
            "qa": [
                ("回答：深度学习和机器学习的区别是什么？", "", "深度学习是机器学习的子集，使用多层神经网络自动学习特征。"),
                ("什么是自然语言处理？", "", "自然语言处理是让计算机理解和处理人类语言的技术。")
            ],
            "rewrite": [
                ("把这句话改写得更简洁", "由于天气不好，所以我们决定取消外出计划", "天气不好，我们取消了外出。"),
                ("将这段文字改写为正式语体", "这个方法挺不错的", "该方法具有良好的效果。")
            ],
            "reasoning": [
                ("进行逻辑推理", "所有学生都要参加考试。小明是学生。", "因此，小明要参加考试。"),
                ("分析因果关系", "经常运动可以增强体质。", "所以坚持运动有益健康。")
            ]
        }
        
        templates = mock_templates.get(task_type, mock_templates["generation"])
        template = templates[idx % len(templates)]
        
        return {
            "instruction": template[0],
            "input": template[1],
            "output": template[2],
            "task_type": task_type,
            "generated": True,
            "generation_method": "mock"
        }
    
    def _api_generate(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """通过 API 生成新指令（需要 LLM API）"""
        # 这里应该调用 GPT-4 或 Qwen2-72B API
        # 示例 prompt 结构：
        prompt = f"""
        基于以下示例，生成一个类似的新任务：
        
        示例：
        指令：{seed['instruction']}
        输入：{seed['input']}
        输出：{seed['output']}
        任务类型：{seed['task_type']}
        
        要求：
        1. 生成的任务应该与示例类型相同
        2. 内容要有创新性，不要重复
        3. 保持相似的难度水平
        
        请生成：
        """
        
        # 实际实现需要调用 API
        # response = call_llm_api(prompt)
        # return parse_response(response)
        
        return self._mock_generate(seed, 0)
    
    def _quality_check(self, instruction: Dict[str, Any], existing: List[Dict[str, Any]]) -> bool:
        """质量检查：去重和基本验证"""
        # 检查必要字段
        if not all(k in instruction for k in ["instruction", "output"]):
            return False
        
        # 检查内容长度
        if len(instruction["instruction"]) < 5 or len(instruction["output"]) < 2:
            return False
        
        # 简单去重：检查指令相似度
        inst_hash = hashlib.md5(instruction["instruction"].encode()).hexdigest()
        for exist in existing:
            exist_hash = hashlib.md5(exist["instruction"].encode()).hexdigest()
            if inst_hash == exist_hash:
                return False
        
        return True
    
    def convert_to_sharegpt(self, instructions: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """转换为 ShareGPT 格式（LLaMA-Factory SFT 标准格式）"""
        if instructions is None:
            instructions = self.generated_data
        
        sharegpt_data = []
        for inst in instructions:
            # 构建用户消息内容
            user_content = inst["instruction"]
            if inst.get("input"):
                user_content += "\n" + inst["input"]
            
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": inst["output"]}
            ]
            sharegpt_data.append({"messages": messages})
        
        return sharegpt_data
    
    def save_dataset(self, filename: str = None, format: str = "sharegpt"):
        """保存数据集并自动注册到 dataset_info.json"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sft_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 默认使用 ShareGPT 格式（LLaMA-Factory 标准）
        if format == "sharegpt":
            data = self.convert_to_sharegpt()
        else:
            data = self.generated_data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 自动注册数据集到 dataset_info.json
        self._register_dataset(filename, "sft_data")
        
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
        
        # 添加或更新数据集配置（ShareGPT 格式，使用相对于data/llmops的路径）
        dataset_info[dataset_name] = {
            "file_name": f"sft/{filename}",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
        
        # 保存配置
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ SFT数据集已注册（ShareGPT格式）: {dataset_name} -> {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.generated_data:
            return {"total": 0}
        
        stats = {
            "total": len(self.generated_data),
            "seed_count": len(self.seed_data),
            "task_types": {}
        }
        
        for item in self.generated_data:
            task_type = item.get("task_type", "unknown")
            stats["task_types"][task_type] = stats["task_types"].get(task_type, 0) + 1
        
        return stats

